/* SPDX-License-Identifier: BSD-2-Clause
 * Concurrent public-API decoding benchmark and exact-pixel reference checker.
 * Worker startup/join are included in each corpus time; kernel warmup is not.
 */
#include "openjpeg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
typedef HANDLE thread_handle;
static SRWLOCK task_lock = SRWLOCK_INIT;
#define LOCK() AcquireSRWLockExclusive(&task_lock)
#define UNLOCK() ReleaseSRWLockExclusive(&task_lock)
static double now_ms(void) { LARGE_INTEGER t,f; QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f); return 1000.0*(double)t.QuadPart/(double)f.QuadPart; }
#else
#include <pthread.h>
#include <time.h>
typedef pthread_t thread_handle;
static pthread_mutex_t task_lock = PTHREAD_MUTEX_INITIALIZER;
#define LOCK() pthread_mutex_lock(&task_lock)
#define UNLOCK() pthread_mutex_unlock(&task_lock)
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0+t.tv_nsec/1000000.0; }
#endif

typedef struct {
    const char *path;
    unsigned gpu_tiles, cpu_tiles, worker_mask, peak_active;
    unsigned long long peak_pool_bytes;
    unsigned long long checksum;
    int ok;
} job_result;

static job_result *jobs;
static int next_job, job_count, cpu_threads;
static const char *reference_dir;
static int write_reference;

static void info(const char *message,void *data)
{
    job_result *job=(job_result*)data;
    if(strstr(message,"OpenCL decoded tile")) {
        unsigned slot,active;
        unsigned long long bytes;
        ++job->gpu_tiles;
        if(sscanf(message,"OpenCL decoded tile %*u (%*u blocks) worker %u active %u pooled %llu",&slot,&active,&bytes)==3) {
            if(slot<8) job->worker_mask|=1u<<slot;
            if(active>job->peak_active) job->peak_active=active;
            if(bytes>job->peak_pool_bytes) job->peak_pool_bytes=bytes;
        }
    }
    if(strstr(message,"Header of tile ")) ++job->cpu_tiles;
}
static void error(const char *message,void *data)
{ (void)data; fputs(message,stderr); }

static int reference_bytes(FILE *file,const void *data,size_t bytes)
{
    unsigned char buffer[16384];
    const unsigned char *expected=(const unsigned char*)data;
    if(write_reference) return fwrite(data,1,bytes,file)==bytes;
    while(bytes) {
        size_t n=bytes<sizeof(buffer)?bytes:sizeof(buffer);
        if(fread(buffer,1,n,file)!=n || memcmp(buffer,expected,n)) return 0;
        bytes-=n; expected+=n;
    }
    return 1;
}

static int check_reference(opj_image_t *image,int index)
{
    char path[4096];
    FILE *file;
    unsigned c;
    int ok=1;
    OPJ_UINT32 header[5]={image->x0,image->y0,image->x1,image->y1,image->numcomps};
    int length=snprintf(path,sizeof(path),"%s/%d.pixels",reference_dir,index);
    if(length<0 || (size_t)length>=sizeof(path)) return 0;
    file=fopen(path,write_reference?"wb":"rb");
    if(!file) return 0;
    ok=reference_bytes(file,header,sizeof(header));
    for(c=0;ok && c<image->numcomps;c++) {
        opj_image_comp_t *comp=&image->comps[c];
        OPJ_UINT32 metadata[9]={comp->dx,comp->dy,comp->w,comp->h,comp->x0,comp->y0,comp->prec,comp->sgnd,comp->alpha};
        ok=reference_bytes(file,metadata,sizeof(metadata)) &&
           reference_bytes(file,comp->data,(size_t)comp->w*comp->h*sizeof(OPJ_INT32));
    }
    if(!write_reference && fgetc(file)!=EOF) ok=0;
    if(fclose(file)) ok=0;
    return ok;
}

static void decode_job(int index)
{
    job_result *job=&jobs[index];
    opj_dparameters_t parameters;
    opj_codec_t *codec=NULL;
    opj_stream_t *stream=NULL;
    opj_image_t *image=NULL;
    unsigned c;
    job->gpu_tiles=job->cpu_tiles=job->worker_mask=job->peak_active=0;
    job->checksum=job->peak_pool_bytes=0; job->ok=0;
    opj_set_default_decoder_parameters(&parameters);
    codec=opj_create_decompress(OPJ_CODEC_J2K);
    stream=opj_stream_create_default_file_stream(job->path,OPJ_TRUE);
    if(!codec || !stream) goto done;
    opj_set_info_handler(codec,info,job); opj_set_error_handler(codec,error,NULL);
    if(!opj_setup_decoder(codec,&parameters) || !opj_codec_set_threads(codec,cpu_threads) ||
       !opj_read_header(stream,codec,&image) || !opj_decode(codec,stream,image) ||
       !opj_end_decompress(codec,stream)) goto done;
    for(c=0;c<image->numcomps;c++) {
        size_t n=(size_t)image->comps[c].w*image->comps[c].h,j;
        for(j=0;j<n;j++) job->checksum=job->checksum*33+(unsigned)image->comps[c].data[j];
    }
    if(reference_dir && !check_reference(image,index)) goto done;
    job->ok=1;
done:
    if(image) opj_image_destroy(image);
    if(stream) opj_stream_destroy(stream);
    if(codec) opj_destroy_codec(codec);
}

#ifdef _WIN32
static DWORD WINAPI worker(void *unused)
#else
static void *worker(void *unused)
#endif
{
    (void)unused;
    for(;;) {
        int index;
        LOCK(); index=next_job++; UNLOCK();
        if(index>=job_count) break;
        decode_job(index);
    }
    return 0;
}

int main(int argc,char **argv)
{
    thread_handle handles[32];
    int workers,repeat,round,i,created;
    unsigned gpu_tiles=0,tiles=0,worker_mask=0,peak_active=0;
    unsigned long long peak_pool_bytes=0;
    unsigned long long *expected=NULL;
    double cold=0,elapsed=0;
    if(argc>2 && (!strcmp(argv[1],"--write-reference") || !strcmp(argv[1],"--verify-reference"))) {
        write_reference=!strcmp(argv[1],"--write-reference");
        reference_dir=argv[2]; argc-=2; argv+=2;
    }
    if(argc<5) {
        fprintf(stderr,"Usage: bench_workers [--write-reference DIR | --verify-reference DIR] workers cpu_threads repeats files...\n");
        return 2;
    }
    workers=atoi(argv[1]); cpu_threads=atoi(argv[2]); repeat=atoi(argv[3]);
    if(workers<1 || workers>32 || cpu_threads<1 || cpu_threads>256 || repeat<1 || repeat>100) return 2;
    job_count=argc-4;
    jobs=(job_result*)calloc((size_t)job_count,sizeof(*jobs));
    expected=(unsigned long long*)calloc((size_t)job_count,sizeof(*expected));
    if(!jobs || !expected) { free(jobs); free(expected); return 3; }
    for(i=0;i<job_count;i++) jobs[i].path=argv[i+4];
    for(round=0;round<=repeat;round++) {
        double started=now_ms();
        next_job=0;
        for(created=0;created<workers;created++) {
#ifdef _WIN32
            handles[created]=CreateThread(NULL,0,worker,NULL,0,NULL);
            if(!handles[created]) break;
#else
            if(pthread_create(&handles[created],NULL,worker,NULL)) break;
#endif
        }
        for(i=0;i<created;i++) {
#ifdef _WIN32
            WaitForSingleObject(handles[i],INFINITE); CloseHandle(handles[i]);
#else
            pthread_join(handles[i],NULL);
#endif
        }
        if(!round) cold=now_ms()-started; else elapsed+=now_ms()-started;
        if(created!=workers) { free(jobs); free(expected); return 4; }
        for(i=0;i<job_count;i++) {
            if(!jobs[i].ok || (round && expected[i]!=jobs[i].checksum)) {
                fprintf(stderr,"Decode or output verification failed for input %d\n",i);
                free(jobs); free(expected); return 5;
            }
            expected[i]=jobs[i].checksum;
            gpu_tiles+=jobs[i].gpu_tiles; tiles+=jobs[i].cpu_tiles;
            worker_mask|=jobs[i].worker_mask;
            if(jobs[i].peak_active>peak_active) peak_active=jobs[i].peak_active;
            if(jobs[i].peak_pool_bytes>peak_pool_bytes) peak_pool_bytes=jobs[i].peak_pool_bytes;
        }
    }
    printf("{\"workers\":%d,\"cpu_threads\":%d,\"warm_decodes\":%d,\"cold_corpus_ms\":%.3f,\"warm_total_ms\":%.3f,\"warm_ms_per_image\":%.3f,\"images_per_second\":%.3f,\"gpu_tiles\":%u,\"tiles\":%u,\"checksums\":[",
           workers,cpu_threads,repeat*job_count,cold,elapsed,elapsed/(repeat*job_count),1000.0*repeat*job_count/elapsed,gpu_tiles,tiles);
    for(i=0;i<job_count;i++) printf("%s\"%llu\"",i?",":"",expected[i]);
    printf("],\"worker_mask\":%u,\"peak_active\":%u,\"peak_pool_bytes\":%llu}\n",worker_mask,peak_active,peak_pool_bytes);
    fflush(stdout);
    free(jobs); free(expected); return 0;
}
