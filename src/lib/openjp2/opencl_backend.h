/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef OPJ_OPENCL_BACKEND_H
#define OPJ_OPENCL_BACKEND_H
/* Return true only after atomically committing a complete decoded tile.
 * False leaves CPU tile storage untouched and requests the original CPU path. */
OPJ_BOOL opj_opencl_decode_tile(opj_tcd_t *tcd, opj_event_mgr_t *manager);
#endif
