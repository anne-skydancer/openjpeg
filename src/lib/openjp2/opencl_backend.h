/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef OPJ_OPENCL_BACKEND_H
#define OPJ_OPENCL_BACKEND_H
/* Return true only after atomically committing a complete decoded tile.
 * False leaves CPU tile storage untouched and requests the original CPU path. */
OPJ_BOOL opj_opencl_decode_tile(opj_tcd_t *tcd, opj_event_mgr_t *manager);
/* Encoder stages commit only after complete success; false requests the CPU
 * implementation of that stage. GPU selection is automatic unless overridden. */
/* Fused path leaves all CPU input samples and blocks untouched on failure. */
OPJ_BOOL opj_opencl_encode_tile(opj_tcd_t *tcd, opj_event_mgr_t *manager);
OPJ_BOOL opj_opencl_encode_tier1(opj_tcd_t *tcd, opj_event_mgr_t *manager);
OPJ_BOOL opj_opencl_encode_transform(opj_tcd_t *tcd, opj_event_mgr_t *manager);
#endif
