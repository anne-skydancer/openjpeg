/* SPDX-License-Identifier: BSD-2-Clause
 * Development benchmark: warm file cache, same process/context, one CPU thread.
 */
#include "openjpeg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) { LARGE_INTEGER t,f; QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f); return 1000.0*(double)t.QuadPart/(double)f.QuadPart; }
#else
#include <time.h>
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0+t.tv_nsec/1000000.0; }
#endif
static unsigned gpu_tiles;
static void info(const char *message,void *data) { (void)data; if(strstr(message,"OpenCL decoded tile")) ++gpu_tiles; if(strstr(message,"OpenCL timings:")) fputs(message,stderr); }
static void error(const char *message,void *data) { (void)data; fputs(message,stderr); }

int main(int argc,char **argv)
{
    int repeat,round,i;
    unsigned decoded=0;
    double started,elapsed=0,cold=0;
    unsigned long long checksum=0;
    if(argc<3) { fprintf(stderr,"Usage: bench_decode repeats file.j2k ...\n"); return 2; }
    repeat=atoi(argv[1]); if(repeat<1 || repeat>100) return 2;
    for(round=0;round<=repeat;round++) {
        started=now_ms();
        for(i=2;i<argc;i++) {
            opj_dparameters_t parameters;
            opj_codec_t *codec;
            opj_stream_t *stream;
            opj_image_t *image=NULL;
            unsigned c;
            opj_set_default_decoder_parameters(&parameters);
            codec=opj_create_decompress(OPJ_CODEC_J2K);
            stream=opj_stream_create_default_file_stream(argv[i],OPJ_TRUE);
            if(!codec || !stream) return 3;
            opj_set_info_handler(codec,info,NULL); opj_set_error_handler(codec,error,NULL);
            if(!opj_setup_decoder(codec,&parameters) || !opj_codec_set_threads(codec,1) ||
               !opj_read_header(stream,codec,&image) || !opj_decode(codec,stream,image) ||
               !opj_end_decompress(codec,stream)) return 4;
            for(c=0;c<image->numcomps;c++) {
                size_t n=(size_t)image->comps[c].w*image->comps[c].h,j;
                for(j=0;j<n;j++) checksum=checksum*33+(unsigned)image->comps[c].data[j];
            }
            opj_image_destroy(image);opj_stream_destroy(stream);opj_destroy_codec(codec);++decoded;
        }
        if(!round) cold=now_ms()-started; else elapsed+=now_ms()-started;
    }
    printf("{\"images\":%u,\"warm_decodes\":%d,\"cold_corpus_ms\":%.3f,\"warm_total_ms\":%.3f,\"warm_mean_ms\":%.3f,\"gpu_tiles\":%u,\"checksum\":\"%llu\"}\n",
           decoded,repeat*(argc-2),cold,elapsed,elapsed/(repeat*(argc-2)),gpu_tiles,checksum);
    return 0;
}
