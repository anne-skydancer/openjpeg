/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef OPJ_ACCEL_DECODE_H
#define OPJ_ACCEL_DECODE_H
OPJ_BOOL opj_accel_decode_tile(opj_tcd_t *tcd, opj_event_mgr_t *manager);
#ifdef OPJ_HAVE_CUDA
OPJ_BOOL opj_cuda_decode_tile(opj_tcd_t *tcd, opj_event_mgr_t *manager);
OPJ_BOOL opj_cuda_available(void);
#endif
#endif
