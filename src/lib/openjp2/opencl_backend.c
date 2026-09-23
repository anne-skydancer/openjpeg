/* SPDX-License-Identifier: BSD-2-Clause
 * Opt-in experimental synchronous OpenCL backend. No OpenCL link dependency:
 * the installed ICD is loaded only with an explicit decoder or encoder device
 * selector. Encoding is independently enabled by OPJ_OPENCL_ENCODE_DEVICE.
 * Unsupported input, allocation refusal and errors use the original CPU stage.
 * Eligible concurrent calls share bounded worker slots and wait for admission.
 */
#define OPJ_SKIP_POISON
#include "opj_includes.h"
#include "opencl_backend.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "opencl_kernels.h"
#ifdef _WIN32
#include <windows.h>
static SRWLOCK cl_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE cl_available = CONDITION_VARIABLE_INIT;
#define LOCK() AcquireSRWLockExclusive(&cl_lock)
#define WAIT() SleepConditionVariableSRW(&cl_available,&cl_lock,INFINITE,0)
#define WAKE() WakeAllConditionVariable(&cl_available)
#define UNLOCK() ReleaseSRWLockExclusive(&cl_lock)
#define LIB_OPEN() LoadLibraryExW(L"OpenCL.dll",NULL,LOAD_LIBRARY_SEARCH_SYSTEM32)
#define SYMBOL(h,n) GetProcAddress((HMODULE)(h),(n))
#else
#include <dlfcn.h>
#include <pthread.h>
static pthread_mutex_t cl_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cl_available = PTHREAD_COND_INITIALIZER;
#define LOCK() pthread_mutex_lock(&cl_lock)
#define WAIT() (pthread_cond_wait(&cl_available,&cl_lock) == 0)
#define WAKE() pthread_cond_broadcast(&cl_available)
#define UNLOCK() pthread_mutex_unlock(&cl_lock)
#define LIB_OPEN() dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL)
#define SYMBOL(h,n) dlsym((h),(n))
#endif

#define CL_FUNCTIONS(X) \
X(GetPlatformIDs) X(GetDeviceIDs) X(GetDeviceInfo) X(CreateContext) \
X(CreateCommandQueue) X(CreateProgramWithSource) X(BuildProgram) \
X(GetProgramBuildInfo) X(CreateKernel) X(CreateBuffer) X(SetKernelArg) \
X(EnqueueNDRangeKernel) X(EnqueueWriteBuffer) X(EnqueueReadBuffer) X(Finish) X(ReleaseMemObject) \
X(ReleaseKernel) X(ReleaseProgram) X(ReleaseCommandQueue) X(ReleaseContext) \
X(GetEventProfilingInfo) X(ReleaseEvent)
/* Function pointer signatures match the Khronos OpenCL 1.2 declarations. */
#define DECL(ret,n,args) static ret (CL_API_CALL *fn##n) args
DECL(cl_int,GetPlatformIDs,(cl_uint,cl_platform_id*,cl_uint*));
DECL(cl_int,GetDeviceIDs,(cl_platform_id,cl_device_type,cl_uint,cl_device_id*,cl_uint*));
DECL(cl_int,GetDeviceInfo,(cl_device_id,cl_device_info,size_t,void*,size_t*));
DECL(cl_context,CreateContext,(const cl_context_properties*,cl_uint,const cl_device_id*,void(CL_CALLBACK*)(const char*,const void*,size_t,void*),void*,cl_int*));
DECL(cl_command_queue,CreateCommandQueue,(cl_context,cl_device_id,cl_command_queue_properties,cl_int*));
DECL(cl_program,CreateProgramWithSource,(cl_context,cl_uint,const char**,const size_t*,cl_int*));
DECL(cl_int,BuildProgram,(cl_program,cl_uint,const cl_device_id*,const char*,void(CL_CALLBACK*)(cl_program,void*),void*));
DECL(cl_int,GetProgramBuildInfo,(cl_program,cl_device_id,cl_program_build_info,size_t,void*,size_t*));
DECL(cl_kernel,CreateKernel,(cl_program,const char*,cl_int*));
DECL(cl_mem,CreateBuffer,(cl_context,cl_mem_flags,size_t,void*,cl_int*));
DECL(cl_int,SetKernelArg,(cl_kernel,cl_uint,size_t,const void*));
DECL(cl_int,EnqueueNDRangeKernel,(cl_command_queue,cl_kernel,cl_uint,const size_t*,const size_t*,const size_t*,cl_uint,const cl_event*,cl_event*));
DECL(cl_int,EnqueueReadBuffer,(cl_command_queue,cl_mem,cl_bool,size_t,size_t,void*,cl_uint,const cl_event*,cl_event*));
DECL(cl_int,EnqueueWriteBuffer,(cl_command_queue,cl_mem,cl_bool,size_t,size_t,const void*,cl_uint,const cl_event*,cl_event*));
DECL(cl_int,Finish,(cl_command_queue));
DECL(cl_int,ReleaseMemObject,(cl_mem)); DECL(cl_int,ReleaseKernel,(cl_kernel));
DECL(cl_int,ReleaseProgram,(cl_program)); DECL(cl_int,ReleaseCommandQueue,(cl_command_queue));
DECL(cl_int,ReleaseContext,(cl_context));
DECL(cl_int,GetEventProfilingInfo,(cl_event,cl_profiling_info,size_t,void*,size_t*));
DECL(cl_int,ReleaseEvent,(cl_event));

#define BUDGET (64u*1024u*1024u)
#define MAX_WORKERS 8
/* A leased slot has exclusive ownership of mutable arguments, queue and buffers.
 * Context/program/device are immutable once initialization completes. */
typedef struct {
    int busy;
    unsigned admitted_active;
    OPJ_UINT64 admitted_bytes;
    cl_command_queue queue;
    cl_kernel kernels[7];
    cl_mem buffers[8];
    size_t capacities[8];
    cl_event events[260];
    unsigned stages[260], event_count;
} decode_worker;

static struct {
    void *library;
    int attempted, profiling;
    cl_context context;
    cl_device_id device;
    cl_program program;
    char selector[256], driver[256];
    unsigned worker_count;
    decode_worker workers[MAX_WORKERS];
} runtime;

/* Called only under the admission lock, or during process teardown. */
static void release_buffers(decode_worker *worker)
{
    unsigned i;
    for(i=0;i<8;i++) {
        if(worker->buffers[i]) fnReleaseMemObject(worker->buffers[i]);
        worker->buffers[i]=NULL; worker->capacities[i]=0;
    }
}

static void destroy_worker(decode_worker *worker)
{
    unsigned i;
    if(worker->queue) fnFinish(worker->queue);
    release_buffers(worker);
    for(i=0;i<7;i++) if(worker->kernels[i]) {
        fnReleaseKernel(worker->kernels[i]); worker->kernels[i]=NULL;
    }
    if(worker->queue) { fnReleaseCommandQueue(worker->queue); worker->queue=NULL; }
}

static void destroy_runtime(void)
{
    unsigned i;
    for(i=0;i<MAX_WORKERS;i++) destroy_worker(&runtime.workers[i]);
    if(runtime.program) { fnReleaseProgram(runtime.program); runtime.program=NULL; }
    if(runtime.context) { fnReleaseContext(runtime.context); runtime.context=NULL; }
    /* Keep the ICD module loaded until process teardown. */
}

static OPJ_BOOL initialize(const char *selector, const char *driver, char *warning, size_t warning_size)
{
    cl_platform_id platforms[32];
    cl_device_id devices[32], selected=NULL;
    cl_uint np=0, nd=0, p, d, matches=0;
    cl_int error;
    const char *workers=getenv("OPJ_OPENCL_WORKERS");
    char *end=NULL;
    long count=workers?strtol(workers,&end,10):4;
    if (runtime.attempted) return runtime.context && !strcmp(selector,runtime.selector) && !strcmp(driver,runtime.driver);
    runtime.attempted=1;
    if(count<1 || count>MAX_WORKERS || (workers && (!*workers || *end))) return OPJ_FALSE;
    runtime.worker_count=(unsigned)count;
    if (strlen(selector)>=sizeof(runtime.selector) || strlen(driver)>=sizeof(runtime.driver)) return OPJ_FALSE;
    strcpy(runtime.selector,selector); strcpy(runtime.driver,driver);
    runtime.library=(void*)LIB_OPEN();
    if (!runtime.library) return OPJ_FALSE;
#define LOAD(n) do { void *address=(void*)SYMBOL(runtime.library,"cl" #n); if(!address) return OPJ_FALSE; memcpy(&fn##n,&address,sizeof(address)); } while(0);
    CL_FUNCTIONS(LOAD)
#undef LOAD
    if (fnGetPlatformIDs(32,platforms,&np) || np>32) return OPJ_FALSE;
    for (p=0;p<np;p++) {
        error=fnGetDeviceIDs(platforms[p],CL_DEVICE_TYPE_GPU,32,devices,&nd);
        if (error==CL_DEVICE_NOT_FOUND) continue;
        if (error || nd>32) return OPJ_FALSE;
        for (d=0;d<nd;d++) {
            char name[256]={0}, version[256]={0};
            if (fnGetDeviceInfo(devices[d],CL_DEVICE_NAME,sizeof(name),name,NULL) ||
                    fnGetDeviceInfo(devices[d],CL_DRIVER_VERSION,sizeof(version),version,NULL)) continue;
            name[255]=0; version[255]=0;
            if (strstr(name,selector) && (!*driver || strstr(version,driver))) { selected=devices[d]; ++matches; }
        }
    }
    if (matches!=1) {
        snprintf(warning,warning_size,"OpenCL selector matched %u GPUs; using CPU\n",matches);
        return OPJ_FALSE;
    }
    runtime.context=fnCreateContext(NULL,1,&selected,NULL,NULL,&error);
    if (!runtime.context || error) goto fail;
    runtime.profiling=getenv("OPJ_OPENCL_PROFILE")!=NULL;
    runtime.device=selected;
    runtime.program=fnCreateProgramWithSource(runtime.context,(cl_uint)(sizeof(opj_cl_sources)/sizeof(opj_cl_sources[0])),(const char**)opj_cl_sources,NULL,&error);
    if (!runtime.program || error) goto fail;
    error=fnBuildProgram(runtime.program,1,&selected,"-cl-std=CL1.2",NULL,NULL);
    if (error) {
        char log[8192]={0};
        fnGetProgramBuildInfo(runtime.program,selected,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,NULL);
        snprintf(warning,warning_size,"OpenCL kernel compilation failed: %s\n",log);
        goto fail;
    }
#if defined(_WIN32) && defined(OPJ_EXPORTS)
    {
        HMODULE module;
        /* DLL atexit handlers run during CRT detach under the loader lock.
         * Calling a GPU driver there can deadlock. This optional runtime is
         * process-lived: pin its owning module and let process teardown reclaim
         * the context. Pinning also prevents unload/reload from duplicating it.
         * Static builds can safely release it before EXE CRT termination. */
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_PIN,
                              (LPCWSTR)(const void*)&runtime,&module)) goto fail;
    }
#else
    atexit(destroy_runtime);
#endif
    return OPJ_TRUE;
fail:
    destroy_runtime();
    return OPJ_FALSE;
}

/* Admission is the only serialized part. Waiting releases the lock; a job
 * never holds it while uploading, decoding, reading back or invoking callbacks.
 * Idle capacity may be reclaimed, but active buffers are never touched. */
static decode_worker *acquire_worker(const char *selector,const char *driver,
                                     const size_t sizes[8],opj_event_mgr_t *manager)
{
    decode_worker *worker=NULL;
    char warning[8448]={0};
    unsigned i,j;
    cl_int error;
    const char *names[7]={"decode_blocks","place_blocks","inverse_line","finish_pixels","encode_blocks","prepare_encode","forward_line"};
    LOCK();
    if(!initialize(selector,driver,warning,sizeof(warning))) goto fail;
    for(;;) {
        OPJ_UINT64 capacity=0;
        worker=NULL;
        for(i=0;i<runtime.worker_count;i++) if(!runtime.workers[i].busy) {
            worker=&runtime.workers[i]; break;
        }
        if(worker) {
            for(i=0;i<runtime.worker_count;i++) for(j=0;j<8;j++) {
                size_t n=runtime.workers[i].capacities[j];
                if(&runtime.workers[i]==worker && sizes[j]>n) n=sizes[j];
                capacity+=n;
            }
            if(capacity>BUDGET) {
                capacity=0;
                for(i=0;i<runtime.worker_count;i++) {
                    decode_worker *other=&runtime.workers[i];
                    if(!other->busy) release_buffers(other);
                    for(j=0;j<8;j++) capacity+=other->capacities[j];
                }
                for(j=0;j<8;j++) capacity+=sizes[j];
            }
            if(capacity<=BUDGET) break;
        }
        if(!WAIT()) { worker=NULL; goto fail; }
    }
    if(!worker->queue) {
        worker->queue=fnCreateCommandQueue(runtime.context,runtime.device,
                            runtime.profiling?CL_QUEUE_PROFILING_ENABLE:0,&error);
        if(!worker->queue || error) goto fail;
        for(i=0;i<7;i++) {
            worker->kernels[i]=fnCreateKernel(runtime.program,names[i],&error);
            if(!worker->kernels[i] || error) goto fail;
        }
    }
    for(i=0;i<8;i++) if(worker->capacities[i]<sizes[i]) {
        if(worker->buffers[i]) fnReleaseMemObject(worker->buffers[i]);
        worker->capacities[i]=0;
        worker->buffers[i]=fnCreateBuffer(runtime.context,CL_MEM_READ_WRITE,sizes[i],NULL,&error);
        if(!worker->buffers[i] || error) goto fail;
        worker->capacities[i]=sizes[i];
    }
    worker->busy=1;
    worker->admitted_active=0; worker->admitted_bytes=0;
    for(i=0;i<runtime.worker_count;i++) {
        worker->admitted_active+=runtime.workers[i].busy?1:0;
        for(j=0;j<8;j++) worker->admitted_bytes+=runtime.workers[i].capacities[j];
    }
    UNLOCK();
    return worker;
fail:
    if(worker) destroy_worker(worker);
    WAKE();
    UNLOCK();
    if(*warning) opj_event_msg(manager,EVT_WARNING,"%s",warning);
    return NULL;
}

static void release_worker(decode_worker *worker,OPJ_BOOL success)
{
    LOCK();
    if(!success) release_buffers(worker);
    worker->busy=0;
    WAKE();
    UNLOCK();
}

typedef struct {
    OPJ_UINT32 blocks, segments, bytes, coefficients, max_flag_bytes;
    OPJ_UINT32 *desc, *segs, *place, *status;
    OPJ_BYTE *input;
    float *scale;
} decode_plan;

static void free_plan(decode_plan *p)
{
    opj_free(p->desc); opj_free(p->segs); opj_free(p->place);
    opj_free(p->status); opj_free(p->input); opj_free(p->scale);
}

/* Two passes: count and validate before any GPU allocation; then copy an owned
 * compact plan. All totals are bounded before 32-bit device offsets are formed. */
static OPJ_BOOL plan_blocks(opj_tcd_t *tcd, decode_plan *p, OPJ_UINT32 stride,
                           OPJ_UINT32 samples, OPJ_UINT32 components, int copy)
{
    OPJ_UINT32 c,r,b,pr,k,s,ch;
    OPJ_UINT32 nb=0, ns=0, nbytes=0, nc=0, max_flags=0;
    for(c=0;c<components;c++) {
        opj_tcd_tilecomp_t *tc=&tcd->tcd_image->tiles->comps[c];
        opj_tccp_t *coding=&tcd->tcp->tccps[c];
        for(r=0;r<tc->minimum_num_resolutions;r++) {
            opj_tcd_resolution_t *res=&tc->resolutions[r];
            if(res->numbands>3 || (OPJ_UINT64)res->pw*res->ph>65536) return OPJ_FALSE;
            for(b=0;b<res->numbands;b++) {
                opj_tcd_band_t *band=&res->bands[b];
                if(band->bandno>3 || (!r && band->bandno) || !(band->stepsize>0.0f)) return OPJ_FALSE;
                for(pr=0;pr<res->pw*res->ph;pr++) {
                    opj_tcd_precinct_t *prec=&band->precincts[pr];
                    if((OPJ_UINT64)prec->cw*prec->ch>65536) return OPJ_FALSE;
                    for(k=0;k<prec->cw*prec->ch;k++) {
                        opj_tcd_cblk_dec_t *block=&prec->cblks.dec[k];
                        OPJ_INT32 w=block->x1-block->x0, h=block->y1-block->y0;
                        OPJ_INT32 x=block->x0-band->x0, y=block->y0-band->y0;
                        OPJ_UINT64 len=0, seglen=0, passes=0;
                        OPJ_UINT32 offset=nbytes;
                        OPJ_INT32 bitplane;
                        OPJ_UINT32 coding_pass=2;
                        if(w==0 || h==0) continue;
                        if(w<0 || h<0 || w>1024 || h>1024 || w*h>4096 || block->corrupted ||
                                block->numbps>30 || coding->roishift<0 || coding->roishift>30 ||
                                block->numbps+(OPJ_UINT32)coding->roishift>30 || coding->cblksty & ~63u ||
                                block->real_num_segs>90 || block->numchunks>65536) return OPJ_FALSE;
                        for(ch=0;ch<block->numchunks;ch++) len+=block->chunks[ch].len;
                        bitplane=(OPJ_INT32)block->numbps+coding->roishift;
                        for(s=0;s<block->real_num_segs;s++) {
                            OPJ_UINT32 pass;
                            OPJ_BOOL raw=(coding->cblksty&1) && bitplane<=(OPJ_INT32)block->numbps-4 && coding_pass<2;
                            if(!block->segs[s].real_num_passes || block->segs[s].real_num_passes>90) return OPJ_FALSE;
                            seglen+=block->segs[s].len; passes+=block->segs[s].real_num_passes;
                            for(pass=0;pass<block->segs[s].real_num_passes;pass++) {
                                OPJ_BOOL mode=(coding->cblksty&1) && bitplane<=(OPJ_INT32)block->numbps-4 && coding_pass<2;
                                if(bitplane<1 || mode!=raw) return OPJ_FALSE;
                                if(++coding_pass==3) { coding_pass=0;--bitplane; }
                            }
                        }
                        if(len!=seglen || len>65536 || passes>3u*(block->numbps+coding->roishift) ||
                                nb>=65536 || ns+block->real_num_segs>65536*90u ||
                                (OPJ_UINT64)nbytes+len>BUDGET || (OPJ_UINT64)nc+w*h>BUDGET/4) return OPJ_FALSE;
                        if(band->bandno&1) x+=tc->resolutions[r-1].x1-tc->resolutions[r-1].x0;
                        if(band->bandno&2) y+=tc->resolutions[r-1].y1-tc->resolutions[r-1].y0;
                        if(x<0 || y<0 || (OPJ_UINT32)(x+w)>stride || (OPJ_UINT64)(y+h)*stride>samples) return OPJ_FALSE;
                        if(copy) {
                            OPJ_UINT32 *d=p->desc+nb*12, *place=p->place+nb*4;
                            d[0]=w;d[1]=h;d[2]=band->bandno;d[3]=block->numbps;d[4]=coding->cblksty;
                            d[5]=nbytes;d[6]=(OPJ_UINT32)len;d[7]=ns;d[8]=block->real_num_segs;d[9]=nc;
                            d[10]=(OPJ_UINT32)coding->roishift;
                            d[11]=(tcd->tcp->num_layers_to_decode==tcd->tcp->numlayers && (tcd->tcp->tccps[0].cblksty&16)) ? 1:0;
                            place[0]=c*samples+y*stride+x;place[1]=stride;place[2]=w;place[3]=h;
                            p->scale[nb]=0.5f*band->stepsize;
                            for(ch=0;ch<block->numchunks;ch++) {
                                memcpy(p->input+offset,block->chunks[ch].data,block->chunks[ch].len);
                                offset+=block->chunks[ch].len;
                            }
                            for(s=0;s<block->real_num_segs;s++) {
                                p->segs[(ns+s)*2]=block->segs[s].len;
                                p->segs[(ns+s)*2+1]=block->segs[s].real_num_passes;
                            }
                        }
                        /* Four rows share a word; partial stripes still need a full word. */
                        max_flags=opj_uint_max(max_flags,(OPJ_UINT32)(w*((h+3)/4)*4));
                        ++nb; ns+=block->real_num_segs; nbytes+=(OPJ_UINT32)len; nc+=w*h;
                    }
                }
            }
        }
    }
    if(!copy) { p->blocks=nb;p->segments=ns;p->bytes=nbytes;p->coefficients=nc;p->max_flag_bytes=max_flags; }
    return nb>0;
}

static int arg(cl_kernel k,cl_uint index,size_t size,const void *value)
{ return fnSetKernelArg(k,index,size,value)==CL_SUCCESS; }
static int run(decode_worker *worker,cl_kernel k,size_t count)
{
    cl_event event=NULL;
    size_t local=(k==worker->kernels[1] || k==worker->kernels[2] || k==worker->kernels[6])?64:1;
    if(local==64) count*=64;
    unsigned i;
    if(!count) return 1;
    if(fnEnqueueNDRangeKernel(worker->queue,k,1,NULL,&count,(k!=worker->kernels[3] && k!=worker->kernels[5])?&local:NULL,0,NULL,runtime.profiling?&event:NULL)) return 0;
    if(event) {
        if(worker->event_count>=260) { fnReleaseEvent(event); return 0; }
        for(i=0;i<7;i++) if(k==worker->kernels[i]) break;
        worker->stages[worker->event_count]=i;
        worker->events[worker->event_count++]=event;
    }
    return 1;
}
#define ARG(k,i,v) if(!arg((k),(i),sizeof(v),&(v))) goto cleanup

OPJ_BOOL opj_opencl_decode_tile(opj_tcd_t *tcd,opj_event_mgr_t *manager)
{
    const char *selector=getenv("OPJ_OPENCL_DEVICE"), *driver=getenv("OPJ_OPENCL_DRIVER");
    opj_tcd_tile_t *tile=tcd->tcd_image->tiles;
    opj_tcd_tilecomp_t *first=&tile->comps[0];
    opj_tccp_t *coding=&tcd->tcp->tccps[0];
    OPJ_UINT32 components=tile->numcomps, levels=first->minimum_num_resolutions;
    OPJ_UINT32 width,height,samples,c,r,i,rev=coding->qmfbid,mct=tcd->tcp->mct;
    OPJ_UINT32 precision=tcd->image->comps[0].prec,sgnd=tcd->image->comps[0].sgnd;
    OPJ_INT32 shift=coding->m_dc_level_shift;
    OPJ_UINT64 total,budget;
    OPJ_BOOL success=OPJ_FALSE;
    decode_worker *worker=NULL;
    unsigned completed_slot=0, admitted_active=0;
    OPJ_UINT64 admitted_bytes=0;
    double profile_ms[4]={0};
    OPJ_BOOL have_profile=OPJ_FALSE;
    decode_plan p={0};
    /* desc, segments, input, coefficients, status, placement, scale, planes */
    cl_mem mem[8]={0};
    size_t sizes[8];
    void *host[8]={0};
    OPJ_INT32 *result=NULL;
    cl_kernel kernel;
    if(!selector || !*selector || !tcd->whole_tile_decoding || tcd->used_component ||
            components<1 || components>4 || !levels || levels>33 || mct>1 ||
            (mct && components<3) || precision<1 || precision>16 || rev>1) return OPJ_FALSE;
    if(!driver) driver="";
    width=first->resolutions[levels-1].x1-first->resolutions[levels-1].x0;
    height=first->resolutions[levels-1].y1-first->resolutions[levels-1].y0;
    if(!width || !height || width>4096 || height>4096) return OPJ_FALSE;
    samples=width*height; total=(OPJ_UINT64)samples*components;
    if(total>BUDGET/12) return OPJ_FALSE;
    for(c=0;c<components;c++) {
        opj_tcd_tilecomp_t *tc=&tile->comps[c];
        opj_tccp_t *cc=&tcd->tcp->tccps[c];
        opj_image_comp_t *ic=&tcd->image->comps[c];
        if(tc->minimum_num_resolutions!=levels || ic->resno_decoded+1!=levels ||
                cc->qmfbid!=rev || cc->m_dc_level_shift!=shift || ic->prec!=precision || ic->sgnd!=sgnd ||
                tc->data_size<(OPJ_SIZE_T)samples*4 || !tc->data) return OPJ_FALSE;
        for(r=0;r<levels;r++) {
            opj_tcd_resolution_t *a=&tc->resolutions[r], *b=&first->resolutions[r];
            if(a->x0!=b->x0 || a->x1!=b->x1 || a->y0!=b->y0 || a->y1!=b->y1) return OPJ_FALSE;
        }
    }
    if(!plan_blocks(tcd,&p,width,samples,components,0)) goto cleanup;
    sizes[0]=(size_t)p.blocks*12*4; sizes[1]=opj_uint_max(1,p.segments)*2*4;
    sizes[2]=opj_uint_max(1,p.bytes); sizes[3]=(size_t)p.coefficients*4;
    sizes[4]=(size_t)p.blocks*4;
    sizes[5]=(size_t)p.blocks*4*4; sizes[6]=(size_t)p.blocks*4;
    sizes[7]=(size_t)total*4;
    budget=0;for(i=0;i<8;i++) budget+=sizes[i];
    if(budget>BUDGET) goto cleanup;
    worker=acquire_worker(selector,driver,sizes,manager);
    if(!worker) goto cleanup;
    p.desc=(OPJ_UINT32*)opj_malloc(sizes[0]);p.segs=(OPJ_UINT32*)opj_calloc(1,sizes[1]);
    p.input=(OPJ_BYTE*)opj_calloc(1,sizes[2]);p.status=(OPJ_UINT32*)opj_malloc(sizes[4]);
    p.place=(OPJ_UINT32*)opj_malloc(sizes[5]);p.scale=(float*)opj_malloc(sizes[6]);
    result=(OPJ_INT32*)opj_calloc(1,sizes[7]);
    if(!p.desc||!p.segs||!p.input||!p.status||!p.place||!p.scale||!result) goto cleanup;
    if(!plan_blocks(tcd,&p,width,samples,components,1)) goto cleanup;
    host[0]=p.desc;host[1]=p.segs;host[2]=p.input;host[5]=p.place;host[6]=p.scale;host[7]=result;
    /* Host staging outlives every asynchronous write, including errors. */
    for(i=0;i<8;i++) {
        mem[i]=worker->buffers[i];
        if(host[i] && fnEnqueueWriteBuffer(worker->queue,mem[i],CL_FALSE,0,sizes[i],host[i],0,NULL,NULL)) goto cleanup;
    }
    kernel=worker->kernels[0];
    for(i=0;i<4;i++) { ARG(kernel,i,mem[i]); }
    ARG(kernel,5,mem[4]);
    if(!arg(kernel,4,(size_t)p.max_flag_bytes,NULL)) goto cleanup;
    ARG(kernel,6,p.blocks);
    if(!run(worker,kernel,p.blocks)) goto cleanup;
    kernel=worker->kernels[1];
    ARG(kernel,0,mem[0]);ARG(kernel,1,mem[5]);ARG(kernel,2,mem[6]);ARG(kernel,3,mem[3]);
    ARG(kernel,4,mem[7]);ARG(kernel,5,p.blocks);ARG(kernel,6,rev);
    if(!run(worker,kernel,p.blocks)) goto cleanup;
    kernel=worker->kernels[2];
    ARG(kernel,0,mem[7]);ARG(kernel,3,width);ARG(kernel,9,rev);
    if(!arg(kernel,1,(size_t)opj_uint_max(width,height)*4,NULL)) goto cleanup;
    for(c=0;c<components;c++) {
        OPJ_UINT32 base=c*samples;
        ARG(kernel,2,base);
        for(r=1;r<levels;r++) {
            opj_tcd_resolution_t *res=&first->resolutions[r], *prev=&first->resolutions[r-1];
            OPJ_UINT32 rw=res->x1-res->x0,rh=res->y1-res->y0;
            OPJ_UINT32 low=prev->x1-prev->x0,parity=res->x0&1,vertical=0;
            ARG(kernel,4,rw);ARG(kernel,5,rh);ARG(kernel,6,low);ARG(kernel,7,parity);ARG(kernel,8,vertical);
            if(!run(worker,kernel,rh)) goto cleanup;
            low=prev->y1-prev->y0;parity=res->y0&1;vertical=1;
            ARG(kernel,4,rh);ARG(kernel,5,rw);ARG(kernel,6,low);ARG(kernel,7,parity);ARG(kernel,8,vertical);
            if(!run(worker,kernel,rw)) goto cleanup;
        }
    }
    kernel=worker->kernels[3];
    ARG(kernel,0,mem[7]);ARG(kernel,1,samples);ARG(kernel,2,components);ARG(kernel,3,mct);
    ARG(kernel,4,rev);ARG(kernel,5,precision);ARG(kernel,6,sgnd);ARG(kernel,7,shift);
    if(!run(worker,kernel,samples)) goto cleanup;
    if(fnEnqueueReadBuffer(worker->queue,mem[4],CL_FALSE,0,sizes[4],p.status,0,NULL,NULL)) goto cleanup;
    if(fnEnqueueReadBuffer(worker->queue,mem[7],CL_TRUE,0,sizes[7],result,0,NULL,NULL)) goto cleanup;
    for(i=0;i<p.blocks;i++) {
        if(p.status[i]&255) goto cleanup;
    }

    for(c=0;c<components;c++) memcpy(tile->comps[c].data,result+c*samples,(size_t)samples*4);
    success=OPJ_TRUE;
cleanup:
    if(worker && worker->queue) fnFinish(worker->queue);
    if(worker && worker->event_count) {
        have_profile=OPJ_TRUE;
        for(i=0;i<worker->event_count;i++) {
            cl_ulong begin=0,end=0;
            if(fnGetEventProfilingInfo(worker->events[i],CL_PROFILING_COMMAND_START,sizeof(begin),&begin,NULL)==CL_SUCCESS &&
               fnGetEventProfilingInfo(worker->events[i],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)==CL_SUCCESS &&
               worker->stages[i]<4) profile_ms[worker->stages[i]]+=(end-begin)/1000000.0;
            fnReleaseEvent(worker->events[i]);
        }
        worker->event_count=0;
    }
    if(worker) {
        completed_slot=(unsigned)(worker-runtime.workers);
        admitted_active=worker->admitted_active; admitted_bytes=worker->admitted_bytes;
        release_worker(worker,success);
    }
    /* User callbacks may invoke another decoder. Return the slot first so a
     * nested decode cannot wait on a lease held by its own callback. */
    if(success) {
        for(i=0;i<p.blocks;i++) if(p.status[i]&1536)
            opj_event_msg(manager,EVT_WARNING,"OpenCL PTERM diagnostic %u\n",p.status[i]&1536);
        opj_event_msg(manager,EVT_INFO,"OpenCL decoded tile %u (%u blocks) worker %u active %u pooled %llu\n",
                      tcd->tcd_tileno,p.blocks,completed_slot,admitted_active,(unsigned long long)admitted_bytes);
    }
    if(have_profile)
        opj_event_msg(manager,EVT_INFO,"OpenCL timings: T1 %.3f place %.3f DWT %.3f finish %.3f ms\n",
                      profile_ms[0],profile_ms[1],profile_ms[2],profile_ms[3]);
    free_plan(&p);opj_free(result);
    return success;
}

/* Encoder staging is independent of tile storage until every block succeeds.
 * Rate allocation remains on the CPU with upstream floating-point weights. */
typedef struct {
    opj_tcd_cblk_enc_t *block;
    OPJ_UINT32 comp,level,orient,reversible;
    OPJ_FLOAT64 stepsize;
} encode_ref;

static OPJ_FLOAT64 encode_distortion(OPJ_INT32 nmse,const encode_ref *ref,
                                     OPJ_INT32 bitplane,const OPJ_FLOAT64 *norms,
                                     OPJ_UINT32 norm_count)
{
    OPJ_FLOAT64 w1=1,w2,step=ref->stepsize,value;
    if(norms && ref->comp<norm_count) w1=norms[ref->comp];
    if(ref->reversible) w2=opj_dwt_getnorm(ref->level,ref->orient);
    else {
        OPJ_INT32 gain=ref->orient==0?0:ref->orient==3?2:1;
        w2=opj_dwt_getnorm_real(ref->level,ref->orient);step/=(1<<gain);
    }
    value=w1*w2*step*(1<<bitplane);
    value*=value*nmse/8192.0;
    return value;
}

OPJ_BOOL opj_opencl_encode_tier1(opj_tcd_t *tcd,opj_event_mgr_t *manager)
{
    const char *selector=getenv("OPJ_OPENCL_ENCODE_DEVICE"),*driver=getenv("OPJ_OPENCL_DRIVER");
    opj_tcd_tile_t *tile=tcd->tcd_image->tiles;
    OPJ_UINT32 n=0,samples=0,bytes=0,max_flags=0,c,r,b,pr,k,i,pass;
    OPJ_UINT32 *desc=NULL,*results=NULL;
    OPJ_INT32 *coeff=NULL;
    OPJ_BYTE *output=NULL;
    encode_ref *refs=NULL;
    const OPJ_FLOAT64 *norms=NULL;
    OPJ_UINT32 norm_count=0;
    size_t sizes[8]={0};
    decode_worker *worker=NULL;
    OPJ_BOOL success=OPJ_FALSE;
    unsigned slot=0,active=0;OPJ_UINT64 pooled=0;
    double device_ms=0;
    cl_kernel kernel;
    if(!selector||!*selector||tile->numcomps<1||tile->numcomps>4)return OPJ_FALSE;
    if(!driver)driver="";
    for(c=0;c<tile->numcomps;c++)
        if(tcd->tcp->tccps[c].cblksty || tcd->tcp->tccps[c].roishift ||
           tcd->image->comps[c].prec>16 || !tcd->image->comps[c].prec)return OPJ_FALSE;
    if(tcd->tcp->mct==1) {
        norm_count=3;
        norms=tcd->tcp->tccps[0].qmfbid?opj_mct_get_mct_norms():opj_mct_get_mct_norms_real();
    } else {norm_count=tcd->image->numcomps;norms=(const OPJ_FLOAT64*)tcd->tcp->mct_norms;}
    for(pass=0;pass<2;pass++) {
        n=0;samples=0;bytes=0;max_flags=0;
        for(c=0;c<tile->numcomps;c++) {
            opj_tcd_tilecomp_t *tc=&tile->comps[c];
            opj_tccp_t *cc=&tcd->tcp->tccps[c];
            OPJ_UINT32 stride=(OPJ_UINT32)(tc->x1-tc->x0),height=(OPJ_UINT32)(tc->y1-tc->y0);
            if(!tc->data||!stride||!height||stride>4096||height>4096||
               tc->numresolutions<1||tc->numresolutions>33||cc->qmfbid>1)goto cleanup;
            for(r=0;r<tc->numresolutions;r++) {
                opj_tcd_resolution_t *res=&tc->resolutions[r];
                for(b=0;b<res->numbands;b++) {
                    opj_tcd_band_t *band=&res->bands[b];
                    if(band->x1==band->x0||band->y1==band->y0)continue;
                    if(band->bandno>3||(!r&&band->bandno)||!(band->stepsize>0))goto cleanup;
                    for(pr=0;pr<res->pw*res->ph;pr++) {
                        opj_tcd_precinct_t *prec=&band->precincts[pr];
                        for(k=0;k<prec->cw*prec->ch;k++) {
                            opj_tcd_cblk_enc_t *block=&prec->cblks.enc[k];
                            OPJ_INT32 w=block->x1-block->x0,h=block->y1-block->y0;
                            OPJ_INT32 x=block->x0-band->x0,y=block->y0-band->y0;
                            OPJ_UINT32 capacity,j;
                            if(w==0||h==0)continue;
                            if(w<0||h<0||w>1024||h>1024||w*h>4096||!block->data||!block->passes||
                               block->data_size>65536||block->data_size<2)goto cleanup;
                            capacity=block->data_size+1;
                            if(n>=65536||(OPJ_UINT64)samples+w*h>BUDGET/4||(OPJ_UINT64)bytes+capacity>BUDGET)goto cleanup;
                            if(band->bandno&1)x+=tc->resolutions[r-1].x1-tc->resolutions[r-1].x0;
                            if(band->bandno&2)y+=tc->resolutions[r-1].y1-tc->resolutions[r-1].y0;
                            if(x<0||y<0||(OPJ_UINT32)(x+w)>stride||(OPJ_UINT32)(y+h)>height)goto cleanup;
                            if(pass) {
                                OPJ_UINT32 *d=desc+6*n;
                                d[0]=w;d[1]=h;d[2]=band->bandno;d[3]=samples;d[4]=bytes;d[5]=capacity;
                                refs[n].block=block;refs[n].comp=c;refs[n].level=tc->numresolutions-1-r;
                                refs[n].orient=band->bandno;refs[n].reversible=cc->qmfbid;refs[n].stepsize=band->stepsize;
                                for(j=0;j<(OPJ_UINT32)(w*h);j++) {
                                    OPJ_UINT32 at=(y+j/w)*stride+x+j%w;
                                    if(cc->qmfbid) {
                                        OPJ_INT32 v=tc->data[at];
                                        if(v<=-33554432||v>=33554432)goto cleanup;
                                        coeff[samples+j]=(OPJ_INT32)((OPJ_UINT32)v<<6);
                                    } else {
                                        OPJ_FLOAT32 v=(((OPJ_FLOAT32*)tc->data)[at]/band->stepsize)*64;
                                        if(!(v>-2147483000.0f&&v<2147483000.0f))goto cleanup;
                                        coeff[samples+j]=(OPJ_INT32)opj_lrintf(v);
                                    }
                                }
                            }
                            max_flags=opj_uint_max(max_flags,(OPJ_UINT32)(w*((h+3)/4)*4));
                            ++n;samples+=w*h;bytes+=capacity;
                        }
                    }
                }
            }
        }
        if(!pass) {
            OPJ_UINT64 budget=(OPJ_UINT64)n*(6+184)*4+(OPJ_UINT64)samples*4+bytes;
            if(!n||budget>BUDGET)goto cleanup;
            sizes[0]=(size_t)n*6*4;sizes[1]=(size_t)samples*4;sizes[2]=bytes;sizes[3]=(size_t)n*184*4;
            desc=(OPJ_UINT32*)opj_malloc(sizes[0]);coeff=(OPJ_INT32*)opj_malloc(sizes[1]);
            output=(OPJ_BYTE*)opj_malloc(sizes[2]);results=(OPJ_UINT32*)opj_malloc(sizes[3]);
            refs=(encode_ref*)opj_calloc(n,sizeof(encode_ref));
            if(!desc||!coeff||!output||!results||!refs)goto cleanup;
        }
    }
    worker=acquire_worker(selector,driver,sizes,manager);if(!worker)goto cleanup;
    if(fnEnqueueWriteBuffer(worker->queue,worker->buffers[0],CL_FALSE,0,sizes[0],desc,0,NULL,NULL)||
       fnEnqueueWriteBuffer(worker->queue,worker->buffers[1],CL_FALSE,0,sizes[1],coeff,0,NULL,NULL))goto cleanup;
    kernel=worker->kernels[4];
    for(i=0;i<4;i++){ARG(kernel,i,worker->buffers[i]);}
    if(!arg(kernel,4,max_flags,NULL))goto cleanup;
    ARG(kernel,5,n);if(!run(worker,kernel,n))goto cleanup;
    if(fnEnqueueReadBuffer(worker->queue,worker->buffers[2],CL_FALSE,0,sizes[2],output,0,NULL,NULL)||
       fnEnqueueReadBuffer(worker->queue,worker->buffers[3],CL_TRUE,0,sizes[3],results,0,NULL,NULL))goto cleanup;
    /* Validate all metadata before changing any CPU block or distortion total. */
    for(i=0;i<n;i++) {
        OPJ_UINT32 *v=results+184*i,previous=0;
        if(v[3]||v[0]>25||v[1]!=(v[0]?3*v[0]-2:0)||v[2]>=desc[6*i+5])goto cleanup;
        for(k=0;k<v[1];k++) {
            OPJ_UINT32 rate=v[4+2*k];
            if(rate<previous||rate>v[2])goto cleanup;
            previous=rate;
        }
    }
    tile->distotile=0;
    for(i=0;i<n;i++) {
        OPJ_UINT32 *v=results+184*i,previous=0,type=2;
        OPJ_INT32 bp=(OPJ_INT32)v[0]-1;
        OPJ_FLOAT64 cumulative=0;
        opj_tcd_cblk_enc_t *block=refs[i].block;
        block->numbps=v[0];block->totalpasses=v[1];
        memcpy(block->data,output+desc[6*i+4]+1,v[2]);
        for(k=0;k<v[1];k++) {
            opj_tcd_pass_t *p=&block->passes[k];
            p->rate=v[4+2*k];p->len=p->rate-previous;previous=p->rate;p->term=(k+1==v[1]);
            cumulative+=encode_distortion((OPJ_INT32)v[5+2*k],&refs[i],bp,norms,norm_count);
            p->distortiondec=cumulative;if(++type==3){type=0;--bp;}
        }
        tile->distotile+=cumulative;
    }
    success=OPJ_TRUE;
cleanup:
    if(worker) {
        fnFinish(worker->queue);
        for(i=0;i<worker->event_count;i++) {
            cl_ulong start=0,end=0;
            if(fnGetEventProfilingInfo(worker->events[i],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)==CL_SUCCESS &&
               fnGetEventProfilingInfo(worker->events[i],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)==CL_SUCCESS)
                device_ms+=(end-start)/1000000.0;
            fnReleaseEvent(worker->events[i]);
        }
        worker->event_count=0;slot=(unsigned)(worker-runtime.workers);
        active=worker->admitted_active;pooled=worker->admitted_bytes;release_worker(worker,success);
    }
    if(success)opj_event_msg(manager,EVT_INFO,"OpenCL encoded Tier-1 tile %u (%u blocks) worker %u active %u pooled %llu\n",tcd->tcd_tileno,n,slot,active,(unsigned long long)pooled);
    if(success && runtime.profiling)opj_event_msg(manager,EVT_INFO,"OpenCL encode Tier-1 device %.3f ms\n",device_ms);
    opj_free(desc);opj_free(coeff);opj_free(output);opj_free(results);opj_free(refs);
    return success;
}


OPJ_BOOL opj_opencl_encode_transform(opj_tcd_t *tcd,opj_event_mgr_t *manager)
{
    const char *selector=getenv("OPJ_OPENCL_ENCODE_DEVICE"),*driver=getenv("OPJ_OPENCL_DRIVER");
    opj_tcd_tile_t *tile=tcd->tcd_image->tiles;
    opj_tcd_tilecomp_t *first=&tile->comps[0];
    opj_tccp_t *coding=&tcd->tcp->tccps[0];
    OPJ_UINT32 width=first->x1-first->x0,height=first->y1-first->y0;
    OPJ_UINT32 components=tile->numcomps,levels=first->numresolutions,rev=coding->qmfbid,mct=tcd->tcp->mct;
    OPJ_INT32 shift=coding->m_dc_level_shift;
    OPJ_UINT32 samples,c,r,i;
    OPJ_INT32 *planes=NULL;
    size_t sizes[8]={0};
    decode_worker *worker=NULL;
    cl_kernel kernel;
    OPJ_BOOL success=OPJ_FALSE;
    unsigned slot=0,active=0;OPJ_UINT64 pooled=0;
    double device_ms=0;
    if(!selector||!*selector||!width||!height||width>4096||height>4096||
       components<1||components>4||levels<1||levels>33||rev>1||mct>1||(mct&&components<3))return OPJ_FALSE;
    if(!driver)driver="";
    samples=width*height;
    if((OPJ_UINT64)samples*components*4>BUDGET)return OPJ_FALSE;
    for(c=0;c<components;c++) {
        opj_tcd_tilecomp_t *tc=&tile->comps[c];opj_tccp_t *cc=&tcd->tcp->tccps[c];
        if(!tc->data||(tc->ownsData && tc->data_size<(OPJ_SIZE_T)samples*4)||
           (!tc->ownsData && (tc->data!=tcd->image->comps[c].data ||
             (OPJ_UINT64)tcd->image->comps[c].w*tcd->image->comps[c].h<samples))||tc->numresolutions!=levels||
           cc->qmfbid!=rev||cc->m_dc_level_shift!=shift||
           tcd->image->comps[c].prec<1||tcd->image->comps[c].prec>16)return OPJ_FALSE;
        for(r=0;r<levels;r++) {
            opj_tcd_resolution_t *a=&tc->resolutions[r],*b=&first->resolutions[r];
            if(a->x0!=b->x0||a->y0!=b->y0||a->x1!=b->x1||a->y1!=b->y1)return OPJ_FALSE;
        }
    }
    sizes[7]=(size_t)samples*components*4;
    planes=(OPJ_INT32*)opj_malloc(sizes[7]);if(!planes)goto cleanup;
    for(c=0;c<components;c++) {
        OPJ_UINT32 precision=tcd->image->comps[c].prec;
        OPJ_INT32 lo=tcd->image->comps[c].sgnd?-(1<<(precision-1)):0;
        OPJ_INT32 hi=tcd->image->comps[c].sgnd?(1<<(precision-1))-1:(1<<precision)-1;
        for(i=0;i<samples;i++) {
            OPJ_INT32 v=tile->comps[c].data[i];if(v<lo||v>hi)goto cleanup;
            planes[c*samples+i]=v;
        }
    }
    worker=acquire_worker(selector,driver,sizes,manager);if(!worker)goto cleanup;
    if(fnEnqueueWriteBuffer(worker->queue,worker->buffers[7],CL_FALSE,0,sizes[7],planes,0,NULL,NULL))goto cleanup;
    kernel=worker->kernels[5];
    ARG(kernel,0,worker->buffers[7]);ARG(kernel,1,samples);ARG(kernel,2,components);
    ARG(kernel,3,mct);ARG(kernel,4,rev);ARG(kernel,5,shift);
    if(!run(worker,kernel,samples))goto cleanup;
    kernel=worker->kernels[6];ARG(kernel,0,worker->buffers[7]);ARG(kernel,3,width);ARG(kernel,9,rev);
    if(!arg(kernel,1,(size_t)opj_uint_max(width,height)*4,NULL))goto cleanup;
    for(c=0;c<components;c++) {
        OPJ_UINT32 base=c*samples;ARG(kernel,2,base);
        for(r=levels-1;r>0;r--) {
            opj_tcd_resolution_t *res=&first->resolutions[r],*prev=&first->resolutions[r-1];
            OPJ_UINT32 rw=res->x1-res->x0,rh=res->y1-res->y0;
            OPJ_UINT32 low=prev->y1-prev->y0,parity=res->y0&1,vertical=1;
            ARG(kernel,4,rh);ARG(kernel,5,rw);ARG(kernel,6,low);ARG(kernel,7,parity);ARG(kernel,8,vertical);
            if(!run(worker,kernel,rw))goto cleanup;
            low=prev->x1-prev->x0;parity=res->x0&1;vertical=0;
            ARG(kernel,4,rw);ARG(kernel,5,rh);ARG(kernel,6,low);ARG(kernel,7,parity);ARG(kernel,8,vertical);
            if(!run(worker,kernel,rh))goto cleanup;
        }
    }
    if(fnEnqueueReadBuffer(worker->queue,worker->buffers[7],CL_TRUE,0,sizes[7],planes,0,NULL,NULL))goto cleanup;
    for(c=0;c<components;c++)memcpy(tile->comps[c].data,planes+c*samples,(size_t)samples*4);
    success=OPJ_TRUE;
cleanup:
    if(worker) {
        fnFinish(worker->queue);
        for(i=0;i<worker->event_count;i++) {
            cl_ulong start=0,end=0;
            if(fnGetEventProfilingInfo(worker->events[i],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL)==CL_SUCCESS &&
               fnGetEventProfilingInfo(worker->events[i],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)==CL_SUCCESS)
                device_ms+=(end-start)/1000000.0;
            fnReleaseEvent(worker->events[i]);
        }
        worker->event_count=0;slot=(unsigned)(worker-runtime.workers);
        active=worker->admitted_active;pooled=worker->admitted_bytes;release_worker(worker,success);
    }
    if(success)opj_event_msg(manager,EVT_INFO,"OpenCL forward transformed tile %u worker %u active %u pooled %llu\n",tcd->tcd_tileno,slot,active,(unsigned long long)pooled);
    if(success && runtime.profiling)opj_event_msg(manager,EVT_INFO,"OpenCL encode transforms device %.3f ms\n",device_ms);
    opj_free(planes);return success;
}
