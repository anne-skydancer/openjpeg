/* SPDX-License-Identifier: BSD-2-Clause */
#include "opj_includes.h"
#include "accel_decode.h"
#ifdef OPJ_HAVE_OPENCL
#include "opencl_backend.h"
#endif
OPJ_BOOL opj_accel_decode_tile(opj_tcd_t *tcd, opj_event_mgr_t *manager)
{
    const char *mode=getenv("OPJ_DECODE_BACKEND");
    const char *legacy=getenv("OPJ_OPENCL_DEVICE");
    if(!mode || !*mode) {
        /* Preserve the meaning of all explicit legacy device overrides. */
        if(legacy && *legacy && strcmp(legacy,"auto")) mode="opencl";
        else mode="auto";
    }
    if(!strcmp(mode,"cpu")) return OPJ_FALSE;
#ifdef OPJ_HAVE_CUDA
    if(!strcmp(mode,"cuda")) return opj_cuda_decode_tile(tcd,manager);
    /* A selected CUDA job fails directly to CPU, never repeatedly tries GPUs. */
    if(!strcmp(mode,"auto") && opj_cuda_available()) return opj_cuda_decode_tile(tcd,manager);
#endif
#ifdef OPJ_HAVE_OPENCL
    if(!strcmp(mode,"opencl") || !strcmp(mode,"auto")) return opj_opencl_decode_tile(tcd,manager);
#endif
    return OPJ_FALSE;
}
