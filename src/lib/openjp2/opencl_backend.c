/* SPDX-License-Identifier: BSD-2-Clause
 * Opt-in experimental synchronous OpenCL backend. No OpenCL link dependency:
 * the installed ICD is loaded only when OPJ_OPENCL_DEVICE is explicitly set.
 * Unsupported input, contention, allocation refusal and errors use CPU decoding.
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
#define LOCK() TryAcquireSRWLockExclusive(&cl_lock)
#define UNLOCK() ReleaseSRWLockExclusive(&cl_lock)
#define LIB_OPEN() LoadLibraryExW(L"OpenCL.dll",NULL,LOAD_LIBRARY_SEARCH_SYSTEM32)
#define SYMBOL(h,n) GetProcAddress((HMODULE)(h),(n))
#else
#include <dlfcn.h>
#include <pthread.h>
static pthread_mutex_t cl_lock = PTHREAD_MUTEX_INITIALIZER;
#define LOCK() (pthread_mutex_trylock(&cl_lock) == 0)
#define UNLOCK() pthread_mutex_unlock(&cl_lock)
#define LIB_OPEN() dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL)
#define SYMBOL(h,n) dlsym((h),(n))
#endif

#define CL_FUNCTIONS(X) \
X(GetPlatformIDs) X(GetDeviceIDs) X(GetDeviceInfo) X(CreateContext) \
X(CreateCommandQueue) X(CreateProgramWithSource) X(BuildProgram) \
X(GetProgramBuildInfo) X(CreateKernel) X(CreateBuffer) X(SetKernelArg) \
X(EnqueueNDRangeKernel) X(EnqueueReadBuffer) X(Finish) X(ReleaseMemObject) \
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
DECL(cl_int,Finish,(cl_command_queue));
DECL(cl_int,ReleaseMemObject,(cl_mem)); DECL(cl_int,ReleaseKernel,(cl_kernel));
DECL(cl_int,ReleaseProgram,(cl_program)); DECL(cl_int,ReleaseCommandQueue,(cl_command_queue));
DECL(cl_int,ReleaseContext,(cl_context));
DECL(cl_int,GetEventProfilingInfo,(cl_event,cl_profiling_info,size_t,void*,size_t*));
DECL(cl_int,ReleaseEvent,(cl_event));

static struct {
    void *library;
    int attempted;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernels[4];
    char selector[256], driver[256];
    int profiling;
    cl_event events[260];
    unsigned stages[260], event_count;
} runtime;

static void destroy_runtime(void)
{
    int i;
    if (runtime.queue) fnFinish(runtime.queue);
    for (i=0;i<4;i++) if (runtime.kernels[i]) {
        fnReleaseKernel(runtime.kernels[i]); runtime.kernels[i]=NULL;
    }
    if (runtime.program) { fnReleaseProgram(runtime.program); runtime.program=NULL; }
    if (runtime.queue) { fnReleaseCommandQueue(runtime.queue); runtime.queue=NULL; }
    if (runtime.context) { fnReleaseContext(runtime.context); runtime.context=NULL; }
    /* Keep the ICD module loaded until process teardown. */
}

static OPJ_BOOL initialize(const char *selector, const char *driver, opj_event_mgr_t *manager)
{
    cl_platform_id platforms[32];
    cl_device_id devices[32], selected=NULL;
    cl_uint np=0, nd=0, p, d, matches=0;
    cl_int error;
    const char *names[4]={"decode_blocks","place_blocks","inverse_line","finish_pixels"};
    unsigned i;
    if (runtime.attempted) return runtime.context && !strcmp(selector,runtime.selector) && !strcmp(driver,runtime.driver);
    runtime.attempted=1;
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
        opj_event_msg(manager,EVT_WARNING,"OpenCL selector matched %u GPUs; using CPU\n",matches);
        return OPJ_FALSE;
    }
    runtime.context=fnCreateContext(NULL,1,&selected,NULL,NULL,&error);
    if (!runtime.context || error) goto fail;
    runtime.profiling=getenv("OPJ_OPENCL_PROFILE")!=NULL;
    runtime.queue=fnCreateCommandQueue(runtime.context,selected,runtime.profiling?CL_QUEUE_PROFILING_ENABLE:0,&error);
    if (!runtime.queue || error) goto fail;
    runtime.program=fnCreateProgramWithSource(runtime.context,3,(const char**)opj_cl_sources,NULL,&error);
    if (!runtime.program || error) goto fail;
    error=fnBuildProgram(runtime.program,1,&selected,"-cl-std=CL1.2",NULL,NULL);
    if (error) {
        char log[8192]={0};
        fnGetProgramBuildInfo(runtime.program,selected,CL_PROGRAM_BUILD_LOG,sizeof(log)-1,log,NULL);
        opj_event_msg(manager,EVT_WARNING,"OpenCL kernel compilation failed: %s\n",log);
        goto fail;
    }
    for (i=0;i<4;i++) {
        runtime.kernels[i]=fnCreateKernel(runtime.program,names[i],&error);
        if (!runtime.kernels[i] || error) goto fail;
    }
    atexit(destroy_runtime);
    return OPJ_TRUE;
fail:
    destroy_runtime();
    return OPJ_FALSE;
}

#define BUDGET (64u*1024u*1024u)
typedef struct {
    OPJ_UINT32 blocks, segments, bytes, coefficients, flag_cells;
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
    OPJ_UINT32 nb=0, ns=0, nbytes=0, nc=0, group_start=0, group_max=0;
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
                            OPJ_UINT32 *d=p->desc+nb*13, *place=p->place+nb*4;
                            d[0]=w;d[1]=h;d[2]=band->bandno;d[3]=block->numbps;d[4]=coding->cblksty;
                            d[5]=nbytes;d[6]=(OPJ_UINT32)len;d[7]=ns;d[8]=block->real_num_segs;d[9]=nc;
                            d[10]=(OPJ_UINT32)coding->roishift;
                            d[11]=(tcd->tcp->num_layers_to_decode==tcd->tcp->numlayers && (tcd->tcp->tccps[0].cblksty&16)) ? 1:0;
                            d[12]=group_start+(nb%32);
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
                        group_max=opj_uint_max(group_max,(OPJ_UINT32)(w*h));
                        if(nb%32==31) { group_start+=group_max*32; group_max=0; }
                        ++nb; ns+=block->real_num_segs; nbytes+=(OPJ_UINT32)len; nc+=w*h;
                    }
                }
            }
        }
    }
    if(!copy) { p->blocks=nb;p->segments=ns;p->bytes=nbytes;p->coefficients=nc;p->flag_cells=group_start+group_max*32; }
    return nb>0;
}

static int arg(cl_kernel k,cl_uint index,size_t size,const void *value)
{ return fnSetKernelArg(k,index,size,value)==CL_SUCCESS; }
static int run(cl_kernel k,size_t count)
{
    cl_event event=NULL;
    unsigned i;
    if(!count) return 1;
    if(fnEnqueueNDRangeKernel(runtime.queue,k,1,NULL,&count,NULL,0,NULL,runtime.profiling?&event:NULL)) return 0;
    if(event) {
        if(runtime.event_count>=260) { fnReleaseEvent(event); return 0; }
        for(i=0;i<4;i++) if(k==runtime.kernels[i]) break;
        runtime.stages[runtime.event_count]=i;
        runtime.events[runtime.event_count++]=event;
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
    OPJ_BOOL success=OPJ_FALSE,locked=OPJ_FALSE;
    decode_plan p={0};
    cl_mem mem[10]={0};
    size_t sizes[10];
    void *host[10]={0};
    OPJ_INT32 *result=NULL;
    cl_kernel kernel;
    cl_int error=CL_SUCCESS;
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
    if(!LOCK()) return OPJ_FALSE;
    locked=OPJ_TRUE;
    if(!plan_blocks(tcd,&p,width,samples,components,0)) goto cleanup;
    sizes[0]=(size_t)p.blocks*13*4; sizes[1]=opj_uint_max(1,p.segments)*2*4;
    sizes[2]=opj_uint_max(1,p.bytes); sizes[3]=(size_t)p.coefficients*4;
    sizes[4]=(size_t)p.flag_cells*4; sizes[5]=(size_t)p.blocks*4;
    sizes[6]=(size_t)p.blocks*4*4; sizes[7]=(size_t)p.blocks*4;
    sizes[8]=(size_t)total*4; sizes[9]=(size_t)samples*4;
    budget=0;for(i=0;i<10;i++) budget+=sizes[i];
    if(budget>BUDGET || !initialize(selector,driver,manager)) goto cleanup;
    p.desc=(OPJ_UINT32*)opj_malloc(sizes[0]);p.segs=(OPJ_UINT32*)opj_calloc(1,sizes[1]);
    p.input=(OPJ_BYTE*)opj_calloc(1,sizes[2]);p.status=(OPJ_UINT32*)opj_malloc(sizes[5]);
    p.place=(OPJ_UINT32*)opj_malloc(sizes[6]);p.scale=(float*)opj_malloc(sizes[7]);
    result=(OPJ_INT32*)opj_calloc(1,sizes[8]);
    if(!p.desc||!p.segs||!p.input||!p.status||!p.place||!p.scale||!result) goto cleanup;
    if(!plan_blocks(tcd,&p,width,samples,components,1)) goto cleanup;
    host[0]=p.desc;host[1]=p.segs;host[2]=p.input;host[6]=p.place;host[7]=p.scale;host[8]=result;
    for(i=0;i<10;i++) {
        mem[i]=fnCreateBuffer(runtime.context,CL_MEM_READ_WRITE|(host[i]?CL_MEM_COPY_HOST_PTR:0),sizes[i],host[i],&error);
        if(!mem[i] || error) goto cleanup;
    }
    kernel=runtime.kernels[0];
    for(i=0;i<6;i++) ARG(kernel,i,mem[i]);
    ARG(kernel,6,p.blocks);
    if(!run(kernel,p.blocks)) goto cleanup;
    if(fnEnqueueReadBuffer(runtime.queue,mem[5],CL_TRUE,0,sizes[5],p.status,0,NULL,NULL)) goto cleanup;
    for(i=0;i<p.blocks;i++) {
        if(p.status[i]&255) goto cleanup;
        if(p.status[i]&1536) opj_event_msg(manager,EVT_WARNING,"OpenCL PTERM diagnostic %u\n",p.status[i]&1536);
    }
    kernel=runtime.kernels[1];
    ARG(kernel,0,mem[0]);ARG(kernel,1,mem[6]);ARG(kernel,2,mem[7]);ARG(kernel,3,mem[3]);
    ARG(kernel,4,mem[8]);ARG(kernel,5,p.blocks);ARG(kernel,6,rev);
    if(!run(kernel,p.blocks)) goto cleanup;
    kernel=runtime.kernels[2];
    ARG(kernel,0,mem[8]);ARG(kernel,1,mem[9]);ARG(kernel,3,width);ARG(kernel,9,rev);
    for(c=0;c<components;c++) {
        OPJ_UINT32 base=c*samples;
        ARG(kernel,2,base);
        for(r=1;r<levels;r++) {
            opj_tcd_resolution_t *res=&first->resolutions[r], *prev=&first->resolutions[r-1];
            OPJ_UINT32 rw=res->x1-res->x0,rh=res->y1-res->y0;
            OPJ_UINT32 low=prev->x1-prev->x0,parity=res->x0&1,vertical=0;
            ARG(kernel,4,rw);ARG(kernel,5,rh);ARG(kernel,6,low);ARG(kernel,7,parity);ARG(kernel,8,vertical);
            if(!run(kernel,rh)) goto cleanup;
            low=prev->y1-prev->y0;parity=res->y0&1;vertical=1;
            ARG(kernel,4,rh);ARG(kernel,5,rw);ARG(kernel,6,low);ARG(kernel,7,parity);ARG(kernel,8,vertical);
            if(!run(kernel,rw)) goto cleanup;
        }
    }
    kernel=runtime.kernels[3];
    ARG(kernel,0,mem[8]);ARG(kernel,1,samples);ARG(kernel,2,components);ARG(kernel,3,mct);
    ARG(kernel,4,rev);ARG(kernel,5,precision);ARG(kernel,6,sgnd);ARG(kernel,7,shift);
    if(!run(kernel,samples)) goto cleanup;
    if(fnEnqueueReadBuffer(runtime.queue,mem[8],CL_TRUE,0,sizes[8],result,0,NULL,NULL)) goto cleanup;
    for(c=0;c<components;c++) memcpy(tile->comps[c].data,result+c*samples,(size_t)samples*4);
    success=OPJ_TRUE;
    opj_event_msg(manager,EVT_INFO,"OpenCL decoded tile %u (%u blocks)\n",tcd->tcd_tileno,p.blocks);
cleanup:
    if(runtime.queue && locked) fnFinish(runtime.queue);
    if(locked && runtime.event_count) {
        double ms[4]={0};
        for(i=0;i<runtime.event_count;i++) {
            cl_ulong begin=0,end=0;
            if(fnGetEventProfilingInfo(runtime.events[i],CL_PROFILING_COMMAND_START,sizeof(begin),&begin,NULL)==CL_SUCCESS &&
               fnGetEventProfilingInfo(runtime.events[i],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL)==CL_SUCCESS &&
               runtime.stages[i]<4) ms[runtime.stages[i]]+=(end-begin)/1000000.0;
            fnReleaseEvent(runtime.events[i]);
        }
        runtime.event_count=0;
        opj_event_msg(manager,EVT_INFO,"OpenCL timings: T1 %.3f place %.3f DWT %.3f finish %.3f ms\n",ms[0],ms[1],ms[2],ms[3]);
    }
    for(i=0;i<10;i++) if(mem[i]) fnReleaseMemObject(mem[i]);
    free_plan(&p);opj_free(result);
    if(locked) UNLOCK();
    return success;
}
