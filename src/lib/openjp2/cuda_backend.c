/* SPDX-License-Identifier: BSD-2-Clause
 * NVIDIA decode backend. Public OpenJPEG calls stay synchronous; independent
 * callers use bounded CUDA streams. CPU tile samples are committed atomically.
 */
#define OPJ_SKIP_POISON
#include "opj_includes.h"
#include "accel_decode.h"
#include "accel_budget.h"
#include <cuda.h>
#include "cuda_kernels.h"
#ifdef _WIN32
#include <windows.h>
static SRWLOCK lock=SRWLOCK_INIT;
#define LOCK() AcquireSRWLockExclusive(&lock)
#define UNLOCK() ReleaseSRWLockExclusive(&lock)
#define LIB_OPEN() LoadLibraryExW(L"nvcuda.dll",NULL,LOAD_LIBRARY_SEARCH_SYSTEM32)
#define SYMBOL(h,n) GetProcAddress((HMODULE)(h),(n))
#else
#include <dlfcn.h>
#include <pthread.h>
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
#define LOCK() pthread_mutex_lock(&lock)
#define UNLOCK() pthread_mutex_unlock(&lock)
#define LIB_OPEN() dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL)
#define SYMBOL(h,n) dlsym((h),(n))
#endif
#define DECL(n,args) static CUresult (CUDAAPI *fn##n) args
DECL(Init,(unsigned)); DECL(DeviceGetCount,(int*)); DECL(DeviceGet,(CUdevice*,int));
DECL(DeviceGetAttribute,(int*,CUdevice_attribute,CUdevice));
DECL(DeviceGetName,(char*,int,CUdevice)); DECL(DeviceGetPCIBusId,(char*,int,CUdevice));
DECL(DeviceGetUuid,(CUuuid*,CUdevice)); DECL(DevicePrimaryCtxRetain,(CUcontext*,CUdevice));
DECL(DevicePrimaryCtxRelease,(CUdevice)); DECL(CtxPushCurrent,(CUcontext)); DECL(CtxPopCurrent,(CUcontext*));
DECL(ModuleLoadData,(CUmodule*,const void*)); DECL(ModuleGetFunction,(CUfunction*,CUmodule,const char*));
DECL(ModuleUnload,(CUmodule)); DECL(StreamCreate,(CUstream*,unsigned)); DECL(StreamDestroy,(CUstream));
DECL(StreamSynchronize,(CUstream)); DECL(MemAlloc,(CUdeviceptr*,size_t)); DECL(MemFree,(CUdeviceptr));
DECL(MemHostAlloc,(void**,size_t,unsigned)); DECL(MemFreeHost,(void*));
DECL(MemcpyHtoDAsync,(CUdeviceptr,const void*,size_t,CUstream));
DECL(MemcpyDtoHAsync,(void*,CUdeviceptr,size_t,CUstream));
DECL(LaunchKernel,(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**));
#define BUDGET (64u*1024u*1024u)
#define MAX_WORKERS 8
#define T1_CHUNK 4096u
#include "accel_decode_plan.h"
typedef struct {
    int busy;
    CUstream stream;
    CUdeviceptr buffers[8];
    size_t capacities[8];
    unsigned char *staging;
    size_t host_capacity;
} cuda_worker;
static struct {
    void *library;
    int attempted,ready,poisoned;
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernels[4];
    unsigned workers;
    char name[256];
    cuda_worker slots[MAX_WORKERS];
} runtime;

/* Process-lived runtime: never reset another user's primary context or invoke
 * driver cleanup from DLL detach. The OS reclaims the retained context at exit.
 * Selection/settings are immutable after the first decode attempt. */
static int initialize(void)
{
    int count=0,i,matches=0,selected=-1,sm=0;
    CUcontext previous;
    const char *selector=getenv("OPJ_CUDA_DEVICE"),*workers=getenv("OPJ_CUDA_WORKERS");
    const char *names[]={"decode_blocks","place_blocks","inverse_line","finish_pixels"};
    char *end=NULL;
    long requested=workers?strtol(workers,&end,10):0;
    if(runtime.attempted) return runtime.ready && !runtime.poisoned;
    runtime.attempted=1;
    if(workers && (!*workers || *end || requested<1 || requested>MAX_WORKERS)) return 0;
    if(selector && (!strcmp(selector,"off") || !strcmp(selector,"0"))) return 0;
    runtime.library=(void*)LIB_OPEN(); if(!runtime.library) return 0;
#define LOAD(n,s) do { void *address=(void*)SYMBOL(runtime.library,s); if(!address)return 0; memcpy(&fn##n,&address,sizeof(address)); } while(0)
    LOAD(Init,"cuInit"); LOAD(DeviceGetCount,"cuDeviceGetCount"); LOAD(DeviceGet,"cuDeviceGet");
    LOAD(DeviceGetAttribute,"cuDeviceGetAttribute"); LOAD(DeviceGetName,"cuDeviceGetName");
    LOAD(DeviceGetPCIBusId,"cuDeviceGetPCIBusId"); LOAD(DeviceGetUuid,"cuDeviceGetUuid");
    LOAD(DevicePrimaryCtxRetain,"cuDevicePrimaryCtxRetain"); LOAD(DevicePrimaryCtxRelease,"cuDevicePrimaryCtxRelease_v2");
    LOAD(CtxPushCurrent,"cuCtxPushCurrent_v2"); LOAD(CtxPopCurrent,"cuCtxPopCurrent_v2");
    LOAD(ModuleLoadData,"cuModuleLoadData"); LOAD(ModuleGetFunction,"cuModuleGetFunction"); LOAD(ModuleUnload,"cuModuleUnload");
    LOAD(StreamCreate,"cuStreamCreate"); LOAD(StreamDestroy,"cuStreamDestroy_v2"); LOAD(StreamSynchronize,"cuStreamSynchronize");
    LOAD(MemAlloc,"cuMemAlloc_v2"); LOAD(MemFree,"cuMemFree_v2"); LOAD(MemHostAlloc,"cuMemHostAlloc"); LOAD(MemFreeHost,"cuMemFreeHost");
    LOAD(MemcpyHtoDAsync,"cuMemcpyHtoDAsync_v2"); LOAD(MemcpyDtoHAsync,"cuMemcpyDtoHAsync_v2"); LOAD(LaunchKernel,"cuLaunchKernel");
#undef LOAD
    if(fnInit(0)!=CUDA_SUCCESS || fnDeviceGetCount(&count)!=CUDA_SUCCESS || count<1) return 0;
    for(i=0;i<count;i++) {
        CUdevice device; CUuuid uuid; char pci[32]={0},id[64]; int major=0,shared=0;
        unsigned char *u=(unsigned char*)uuid.bytes;
        if(fnDeviceGet(&device,i)!=CUDA_SUCCESS ||
           fnDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device)!=CUDA_SUCCESS || major<5 ||
           fnDeviceGetAttribute(&shared,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,device)!=CUDA_SUCCESS || shared<16384) continue;
        if(selector && *selector && strcmp(selector,"auto")) {
            if(fnDeviceGetPCIBusId(pci,sizeof(pci),device)!=CUDA_SUCCESS || fnDeviceGetUuid(&uuid,device)!=CUDA_SUCCESS) continue;
            snprintf(id,sizeof(id),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
            if(strcmp(selector,pci) && strcmp(selector,id)) continue;
        }
        matches++; selected=i;
    }
    /* No guessing on machines with multiple NVIDIA devices. */
    if(matches!=1 || fnDeviceGet(&runtime.device,selected)!=CUDA_SUCCESS) return 0;
    if(fnDevicePrimaryCtxRetain(&runtime.context,runtime.device)!=CUDA_SUCCESS) return 0;
    if(fnCtxPushCurrent(runtime.context)!=CUDA_SUCCESS) goto fail_context;
    if(fnModuleLoadData(&runtime.module,opj_cuda_fatbin)!=CUDA_SUCCESS) goto fail_current;
    for(i=0;i<4;i++) if(fnModuleGetFunction(&runtime.kernels[i],runtime.module,names[i])!=CUDA_SUCCESS) goto fail_current;
    if(fnDeviceGetAttribute(&sm,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,runtime.device)!=CUDA_SUCCESS) goto fail_current;
    runtime.workers=requested?(unsigned)requested:(unsigned)opj_int_min(4,opj_int_max(1,sm/20));
    fnDeviceGetName(runtime.name,sizeof(runtime.name),runtime.device);
#if defined(_WIN32) && defined(OPJ_EXPORTS)
    { HMODULE module;
      if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
                            (LPCWSTR)(const void*)&runtime,&module)) goto fail_current; }
#endif
    if(fnCtxPopCurrent(&previous)!=CUDA_SUCCESS) { runtime.poisoned=1; return 0; }
    runtime.ready=1; return 1;
fail_current:
    if(runtime.module) fnModuleUnload(runtime.module);
    runtime.module=NULL; fnCtxPopCurrent(&previous);
fail_context:
    fnDevicePrimaryCtxRelease(runtime.device); runtime.context=NULL; return 0;
}
OPJ_BOOL opj_cuda_available(void)
{
    int ready; LOCK(); ready=initialize(); UNLOCK(); return ready?OPJ_TRUE:OPJ_FALSE;
}
static void release_host(cuda_worker *w)
{
    if(w->staging) {
        if(fnMemFreeHost(w->staging)==CUDA_SUCCESS) {
            opj_accel_release(w->host_capacity,1);
            w->staging=NULL;w->host_capacity=0;
        } else runtime.poisoned=1;
    }
}
static unsigned char *acquire_host(cuda_worker *w,size_t bytes)
{
    unsigned i;
    unsigned char *result=NULL;
    LOCK();
    if(runtime.poisoned) goto done;
    if(w->host_capacity<bytes) {
        release_host(w);
        if(runtime.poisoned) goto done;
        if(!opj_accel_reserve(bytes,1)) {
            for(i=0;i<runtime.workers;i++)
                if(!runtime.slots[i].busy) release_host(&runtime.slots[i]);
            if(runtime.poisoned || !opj_accel_reserve(bytes,1)) goto done;
        }
        if(fnMemHostAlloc((void**)&w->staging,bytes,0)!=CUDA_SUCCESS) {
            opj_accel_release(bytes,1);goto done;
        }
        w->host_capacity=bytes;
    }
    result=w->staging;
done:
    UNLOCK();return result;
}
static void release_buffers(cuda_worker *w)
{
    unsigned i;
    for(i=0;i<8;i++) if(w->buffers[i]) {
        if(fnMemFree(w->buffers[i])==CUDA_SUCCESS) {
            opj_accel_release(w->capacities[i],0);w->buffers[i]=0;w->capacities[i]=0;
        } else runtime.poisoned=1;
    }
}
/* Never wait holding a decoder worker: bounded admission declines to CPU when
 * slots or the shared CUDA/OpenCL memory allowance are exhausted. */
static cuda_worker *acquire(const size_t sizes[8])
{
    unsigned i,j; cuda_worker *w=NULL;
    LOCK();
    if(runtime.poisoned) goto done;
    for(i=0;i<runtime.workers;i++) if(!runtime.slots[i].busy) { w=&runtime.slots[i];break; }
    if(!w) goto done;
    if(!w->stream && fnStreamCreate(&w->stream,CU_STREAM_NON_BLOCKING)!=CUDA_SUCCESS) goto fail;
    for(j=0;j<8;j++) if(w->capacities[j]<sizes[j]) {
        if(w->buffers[j]) {
            if(fnMemFree(w->buffers[j])!=CUDA_SUCCESS) { runtime.poisoned=1;goto fail; }
            opj_accel_release(w->capacities[j],0); w->buffers[j]=0;w->capacities[j]=0;
        }
        if(!opj_accel_reserve(sizes[j],0)) {
            for(i=0;i<runtime.workers;i++) if(&runtime.slots[i]!=w && !runtime.slots[i].busy) release_buffers(&runtime.slots[i]);
            if(runtime.poisoned || !opj_accel_reserve(sizes[j],0)) goto fail;
        }
        if(fnMemAlloc(&w->buffers[j],sizes[j])!=CUDA_SUCCESS) { opj_accel_release(sizes[j],0);goto fail; }
        w->capacities[j]=sizes[j];
    }
    w->busy=1;goto done;
fail:
    if(!runtime.poisoned) release_buffers(w);
    w=NULL;
done:
    UNLOCK();return w;
}
static int launch(cuda_worker *w,unsigned k,unsigned grid,unsigned block,unsigned shared,void **args)
{
    return fnLaunchKernel(runtime.kernels[k],grid,1,1,block,1,1,shared,w->stream,args,NULL)==CUDA_SUCCESS;
}
OPJ_BOOL opj_cuda_decode_tile(opj_tcd_t *tcd,opj_event_mgr_t *manager)
{
    opj_tcd_tile_t *tile=tcd->tcd_image->tiles;
    opj_tcd_tilecomp_t *first=&tile->comps[0];
    opj_tccp_t *coding=&tcd->tcp->tccps[0];
    unsigned components=tile->numcomps,levels=first->minimum_num_resolutions;
    unsigned width,height,samples,c,r,i,rev=coding->qmfbid,mct=tcd->tcp->mct;
    unsigned precision=tcd->image->comps[0].prec,sgnd=tcd->image->comps[0].sgnd;
    int shift=coding->m_dc_level_shift;
    OPJ_UINT64 total,budget=0;
    decode_plan p={0}; cuda_worker *w=NULL; CUcontext previous;
    size_t sizes[8],offsets[8],host_bytes=0;
    void *input[8]={0}; unsigned char *staging=NULL;
    int pushed=0,success=0,drained=1;
    unsigned diagnostics=0,slot=0,active=0;
    size_t device_peak=0,host_peak=0;
    const char *profile_setting=getenv("OPJ_CUDA_PROFILE");
    int profile=profile_setting && !strcmp(profile_setting,"1");
    double started=opj_clock(),prepared=0,entropy=0,finished=0;
    if(!tcd->whole_tile_decoding || tcd->used_component || components<1 || components>4 || !levels || levels>33 || mct>1 ||
       (mct && components<3) || precision<1 || precision>16 || rev>1) return OPJ_FALSE;
    width=first->resolutions[levels-1].x1-first->resolutions[levels-1].x0;
    height=first->resolutions[levels-1].y1-first->resolutions[levels-1].y0;
    if(!width || !height || width>4096 || height>4096) return OPJ_FALSE;
    samples=width*height;total=(OPJ_UINT64)samples*components;
    if(total>BUDGET/12) return OPJ_FALSE;
    for(c=0;c<components;c++) {
        opj_tcd_tilecomp_t *tc=&tile->comps[c];opj_tccp_t *cc=&tcd->tcp->tccps[c];opj_image_comp_t *ic=&tcd->image->comps[c];
        if(tc->minimum_num_resolutions!=levels || ic->resno_decoded+1!=levels || cc->qmfbid!=rev ||
           cc->m_dc_level_shift!=shift || ic->prec!=precision || ic->sgnd!=sgnd || tc->data_size<(OPJ_SIZE_T)samples*4 || !tc->data) return OPJ_FALSE;
        for(r=0;r<levels;r++) { opj_tcd_resolution_t *a=&tc->resolutions[r],*b=&first->resolutions[r];
            if(a->x0!=b->x0 || a->x1!=b->x1 || a->y0!=b->y0 || a->y1!=b->y1) return OPJ_FALSE; }
    }
    if(!plan_blocks(tcd,&p,width,samples,components,0)) goto cleanup;
    sizes[0]=(size_t)p.blocks*12*4;sizes[1]=(size_t)opj_uint_max(1,p.segments)*2*4;
    sizes[2]=opj_uint_max(1,p.bytes);sizes[3]=(size_t)p.coefficients*4;
    sizes[4]=(size_t)p.blocks*4;sizes[5]=(size_t)p.blocks*4*4;sizes[6]=(size_t)p.blocks*4;sizes[7]=(size_t)total*4;
    for(i=0;i<8;i++) budget+=sizes[i];
    if(budget>BUDGET || !opj_cuda_available()) goto cleanup;
    if(fnCtxPushCurrent(runtime.context)!=CUDA_SUCCESS) goto cleanup;
    pushed=1;w=acquire(sizes);if(!w) goto cleanup;
    for(i=0;i<8;i++) { host_bytes=(host_bytes+15)&~(size_t)15;offsets[i]=host_bytes;host_bytes+=(i==3?0:sizes[i]); }
    staging=acquire_host(w,host_bytes);if(!staging) goto cleanup;
    LOCK();
    slot=(unsigned)(w-runtime.slots);
    for(i=0;i<runtime.workers;i++) active+=runtime.slots[i].busy?1:0;
    device_peak=opj_accel_usage(0);host_peak=opj_accel_usage(1);
    UNLOCK();
    memset(staging,0,host_bytes);
    p.desc=(OPJ_UINT32*)(staging+offsets[0]);p.segs=(OPJ_UINT32*)(staging+offsets[1]);p.input=staging+offsets[2];
    p.status=(OPJ_UINT32*)(staging+offsets[4]);p.place=(OPJ_UINT32*)(staging+offsets[5]);p.scale=(float*)(staging+offsets[6]);
    if(!plan_blocks(tcd,&p,width,samples,components,1)) goto cleanup;
    input[0]=p.desc;input[1]=p.segs;input[2]=p.input;input[5]=p.place;input[6]=p.scale;input[7]=staging+offsets[7];
    prepared=opj_clock();
    drained=0;
    for(i=0;i<8;i++) if(input[i] && fnMemcpyHtoDAsync(w->buffers[i],input[i],sizes[i],w->stream)!=CUDA_SUCCESS) goto cleanup;
    for(i=0;i<p.blocks;i+=T1_CHUNK) {
        unsigned end=opj_uint_min(p.blocks,i+T1_CHUNK),first_block=i;
        void *args[]={&w->buffers[0],&w->buffers[1],&w->buffers[2],&w->buffers[3],&w->buffers[4],&end,&first_block};
        if(!launch(w,0,end-i,1,p.max_flag_bytes,args)) goto cleanup;
    }
    if(profile) { if(fnStreamSynchronize(w->stream)!=CUDA_SUCCESS) goto cleanup; entropy=opj_clock(); }
    { void *args[]={&w->buffers[0],&w->buffers[5],&w->buffers[6],&w->buffers[3],&w->buffers[7],&p.blocks,&rev};
      if(!launch(w,1,p.blocks,64,0,args)) goto cleanup; }
    for(c=0;c<components;c++) for(r=1;r<levels;r++) {
        opj_tcd_resolution_t *res=&first->resolutions[r],*prev=&first->resolutions[r-1];
        unsigned base=c*samples,rw=res->x1-res->x0,rh=res->y1-res->y0,low=prev->x1-prev->x0,parity=res->x0&1,vertical=0;
        void *args[]={&w->buffers[7],&base,&width,&rw,&rh,&low,&parity,&vertical,&rev};
        if(!launch(w,2,rh,64,rw*4,args)) goto cleanup;
        low=prev->y1-prev->y0;parity=res->y0&1;vertical=1;args[3]=&rh;args[4]=&rw;
        if(!launch(w,2,rw,64,rh*4,args)) goto cleanup;
    }
    { void *args[]={&w->buffers[7],&samples,&components,&mct,&rev,&precision,&sgnd,&shift};
      if(!launch(w,3,(samples+127)/128,128,0,args)) goto cleanup; }
    if(fnMemcpyDtoHAsync(p.status,w->buffers[4],sizes[4],w->stream)!=CUDA_SUCCESS ||
       fnMemcpyDtoHAsync(staging+offsets[7],w->buffers[7],sizes[7],w->stream)!=CUDA_SUCCESS) goto cleanup;
    if(fnStreamSynchronize(w->stream)!=CUDA_SUCCESS) goto cleanup;
    drained=1;
    for(i=0;i<p.blocks;i++) {
        if(p.status[i]&255) goto cleanup;
        diagnostics |= p.status[i]&1536;
    }
    for(c=0;c<components;c++) memcpy(tile->comps[c].data,staging+offsets[7]+(size_t)c*samples*4,(size_t)samples*4);
    finished=opj_clock();
    success=1;
cleanup:
    if(w && !drained) drained=fnStreamSynchronize(w->stream)==CUDA_SUCCESS;
    /* An uncompleted DMA must retain its pinned staging. Disable all future
     * CUDA work; no reset, no unsafe free, bounded quarantine until process exit. */
    if(w) {
        LOCK();
        if(!drained) runtime.poisoned=1;
        else if(!success) { release_buffers(w);release_host(w); }
        w->busy=0;UNLOCK();
    }
    if(pushed && fnCtxPopCurrent(&previous)!=CUDA_SUCCESS) { LOCK();runtime.poisoned=1;UNLOCK(); }
    /* Return the worker before callbacks: recursive decoding must not deadlock. */
    if(success && profile) opj_event_msg(manager,EVT_INFO,"CUDA timings: prepare %.3f entropy+upload %.3f reconstruct+download %.3f cleanup %.3f ms\n",(prepared-started)*1000,(entropy-prepared)*1000,(finished-entropy)*1000,(opj_clock()-finished)*1000);
    if(success && diagnostics) opj_event_msg(manager,EVT_WARNING,"CUDA PTERM diagnostic %u\n",diagnostics);
    if(success) opj_event_msg(manager,EVT_INFO,"CUDA decoded tile %u (%u blocks) worker %u active %u pooled %llu pinned %llu on %s\n",tcd->tcd_tileno,p.blocks,slot,active,(unsigned long long)device_peak,(unsigned long long)host_peak,runtime.name);
    return success?OPJ_TRUE:OPJ_FALSE;
}
