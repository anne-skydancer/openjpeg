/* SPDX-License-Identifier: BSD-2-Clause
 * Copyright (c) 2026, OpenJPEG OpenCL contributors
 * Experimental inverse reversible 5/3 lifting, OpenCL C 1.2.
 *
 * Input is interleaved low/high coefficients, not packed subband planes.
 * parity is the tile-component line origin modulo two. Host validation must
 * ensure disjoint lines, valid bounds and the qualified coefficient range.
 * Launch undo_update, then undo_predict with an explicit queue dependency.
 * No workgroup barrier can replace that dependency.
 */
__kernel void undo_update(__global int *data, uint length, uint lines,
                          uint sample_stride, uint line_stride, uint parity)
{
    size_t i = get_global_id(0);
    size_t line = get_global_id(1);
    if (i >= length || line >= lines) return;
    size_t base = line * line_stride;
    size_t at = base + i * sample_stride;
    if (length == 1) {
        if (parity) data[at] /= 2;
        return;
    }
    if (((i + parity) & 1) == 0) {
        size_t left = i == 0 ? 1 : i - 1;
        size_t right = i + 1 == length ? i - 1 : i + 1;
        data[at] -= (data[base + left * sample_stride] +
                     data[base + right * sample_stride] + 2) >> 2;
    }
}

__kernel void undo_predict(__global int *data, uint length, uint lines,
                           uint sample_stride, uint line_stride, uint parity)
{
    size_t i = get_global_id(0);
    size_t line = get_global_id(1);
    if (length <= 1 || i >= length || line >= lines) return;
    if (((i + parity) & 1) != 0) {
        size_t base = line * line_stride;
        size_t left = i == 0 ? 1 : i - 1;
        size_t right = i + 1 == length ? i - 1 : i + 1;
        data[base + i * sample_stride] +=
            (data[base + left * sample_stride] +
             data[base + right * sample_stride]) >> 1;
    }
}
