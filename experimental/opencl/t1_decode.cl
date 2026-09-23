/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2007, Callum Lerwick <seg@haxxed.com>
 * Copyright (c) 2012, Carl Hetherington
 * Copyright (c) 2017, IntoPIX SA <support@intopix.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/* SPDX-License-Identifier: BSD-2-Clause
 * Experimental scalar-per-block JPEG2000 Tier-1 decoder, OpenCL C 1.2.
 * MQ arithmetic and context semantics adapted from OpenJPEG v2.5.4.
 * Upstream copyright and license are retained in mq_states.clh.
 * Experimental Part 1 MQ/RAW path; ROI, PTERM checks and HT not yet supported.
 * This is a correctness implementation, not an optimized production backend.
 */
typedef struct { uint a, c, ct, pos; uint context[19]; } MQ;

uint input_byte(__global const uchar *src, uint length, uint pos)
{
    return pos < length ? src[pos] : 255u;
}
void bytein(MQ *m, __global const uchar *src, uint length)
{
    uint next = input_byte(src, length, m->pos + 1);
    if (input_byte(src, length, m->pos) == 255) {
        if (next > 143) { m->c += 0xff00; m->ct = 8; }
        else { ++m->pos; m->c += next << 9; m->ct = 7; }
    } else { ++m->pos; m->c += next << 8; m->ct = 8; }
}
void reset_contexts(MQ *m)
{
    for (uint i = 0; i < 19; ++i) m->context[i] = 0;
    m->context[0] = 8; m->context[17] = 6; m->context[18] = 92;
}
uint raw_decision(MQ *m, __global const uchar *src, uint length)
{
    if (!m->ct) {
        uint next = input_byte(src,length,m->pos);
        if (m->c == 255) {
            if (next > 143) { m->c = 255; m->ct = 8; }
            else { m->c = next; ++m->pos; m->ct = 7; }
        } else { m->c = next; ++m->pos; m->ct = 8; }
    }
    --m->ct;
    return (m->c >> m->ct) & 1;
}
void init_segment(MQ *m, __global const uchar *src, uint length)
{
    m->pos = 0; m->c = input_byte(src, length, 0) << 16;
    bytein(m, src, length); m->c <<= 7; m->ct -= 7; m->a = 0x8000;
}
uint decision(MQ *m, __global const uchar *src, uint length, uint context)
{
    uint4 s = mq_states[m->context[context]]; /* Qe, MPS, NMPS, NLPS */
    uint bit;
    m->a -= s.x;
    if ((m->c >> 16) < s.x) {
        if (m->a < s.x) { bit = s.y; m->context[context] = s.z; }
        else { bit = 1 - s.y; m->context[context] = s.w; }
        m->a = s.x;
    } else {
        m->c -= s.x << 16;
        if (m->a & 0x8000) return s.y;
        if (m->a < s.x) { bit = 1 - s.y; m->context[context] = s.w; }
        else { bit = s.y; m->context[context] = s.z; }
    }
    do {
        if (!m->ct) bytein(m, src, length);
        m->a <<= 1; m->c <<= 1; --m->ct;
    } while (m->a < 0x8000);
    return bit;
}
int significant(__global const uchar *flags, int x, int y, int w, int h)
{
    return x >= 0 && y >= 0 && x < w && y < h ? (flags[y*w+x] & 1) : 0;
}
int neighbors(__global const uchar *flags, int x, int y, int w, int h, uint style)
{
    if ((style & 8) && (y & 3) == 3) h = min(h,y+1);
    int n = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (dx || dy) n += significant(flags, x+dx, y+dy, w, h);
    return n;
}
uint zero_context(__global const uchar *flags, int x, int y, int w, int h, uint orient, uint style)
{
    if ((style & 8) && (y & 3) == 3) h = min(h,y+1);
    int hc = significant(flags,x-1,y,w,h)+significant(flags,x+1,y,w,h);
    int vc = significant(flags,x,y-1,w,h)+significant(flags,x,y+1,w,h);
    int dc = significant(flags,x-1,y-1,w,h)+significant(flags,x+1,y-1,w,h)
           + significant(flags,x-1,y+1,w,h)+significant(flags,x+1,y+1,w,h);
    /* OpenJPEG's stored orientation swaps the generator's HL/LH indices. */
    if (orient == 1) { int t = hc; hc = vc; vc = t; }
    if (orient == 3) {
        int hv = hc + vc;
        return dc == 0 ? (hv == 0 ? 0 : hv == 1 ? 1 : 2) :
               dc == 1 ? (hv == 0 ? 3 : hv == 1 ? 4 : 5) :
               dc == 2 ? (hv == 0 ? 6 : 7) : 8;
    }
    return hc == 0 ? (vc == 0 ? (dc == 0 ? 0 : dc == 1 ? 1 : 2) : vc == 1 ? 3 : 4) :
           hc == 1 ? (vc == 0 ? (dc == 0 ? 5 : 6) : 7) : 8;
}
int signed_neighbor(__global const int *data, __global const uchar *flags,
                    int x, int y, int w, int h)
{
    return significant(flags,x,y,w,h) ? (data[y*w+x] < 0 ? -1 : 1) : 0;
}
void decode_sign(MQ *m, __global const uchar *src, uint length,
                 __global int *data, __global uchar *flags,
                 int x, int y, int w, int h, int value, uint style, uint raw)
{
    if ((style & 8) && (y & 3) == 3) h = min(h,y+1);
    int hc = clamp(signed_neighbor(data,flags,x-1,y,w,h)+signed_neighbor(data,flags,x+1,y,w,h),-1,1);
    int vc = clamp(signed_neighbor(data,flags,x,y-1,w,h)+signed_neighbor(data,flags,x,y+1,w,h),-1,1);
    uint prediction = hc < 0 || (hc == 0 && vc < 0);
    if (hc < 0) { hc = -hc; vc = -vc; }
    uint context = 9 + (hc == 0 ? (vc == 0 ? 0 : 1) : vc == -1 ? 2 : vc == 0 ? 3 : 4);
    uint sign = raw ? raw_decision(m,src,length) : decision(m,src,length,context) ^ prediction;
    data[y*w+x] = sign ? -value : value;
    flags[y*w+x] |= 1;
}

/* Descriptor: w,h,orientation,numbps,style,inputOffset,inputLength,
 * segmentOffset,segmentCount,coefficientOffset. Segments: length, passCount.
 * Host verifies every range before upload; each block owns disjoint scratch.
 */
__kernel void decode_blocks(__global const uint *desc, __global const uint2 *segments,
                            __global const uchar *input, __global int *output,
                            __global uchar *scratch, __global uint *status, uint count)
{
    size_t block = get_global_id(0);
    if (block >= count) return;
    __global const uint *d = desc + block*10;
    int w = d[0], h = d[1];
    uint orient = d[2], style = d[4], offset = 0;
    __global int *data = output + d[9];
    __global uchar *flags = scratch + d[9];
    status[block] = 0;
    if (style & ~47u || w <= 0 || h <= 0 || w*h > 4096 || orient > 3 || d[3] > 30) {
        status[block] = 1; return;
    }
    for (int i = 0; i < w*h; ++i) { data[i] = 0; flags[i] = 0; }
    MQ m; reset_contexts(&m);
    int bp = d[3]; uint pass = 2;
    for (uint seg = 0; seg < d[8]; ++seg) {
        uint2 s = segments[d[7]+seg];
        if (offset > d[6] || s.x > d[6]-offset) { status[block] = 2; return; }
        __global const uchar *src = input + d[5] + offset;
        offset += s.x;
        uint raw = (style & 1) && bp <= (int)d[3]-4 && pass < 2;
        if (raw) { m.c = 0; m.ct = 0; m.pos = 0; }
        else init_segment(&m,src,s.x);
        for (uint p = 0; p < s.y && bp >= 1; ++p) {
            int midpoint = (1 << bp) >> 1, value = (1 << bp) | midpoint;
            for (int stripe = 0; stripe < h; stripe += 4) {
                int rows = min(4,h-stripe);
                for (int x = 0; x < w; ++x) {
                    int start = 0, forced = 0;
                    if (pass == 2 && rows == 4) {
                        int aggregate = 1;
                        for (int j = 0; j < 4; ++j)
                            if ((flags[(stripe+j)*w+x] & 3) || neighbors(flags,x,stripe+j,w,h,style)) aggregate = 0;
                        if (aggregate) {
                            if (!decision(&m,src,s.x,17)) continue;
                            start = (int)decision(&m,src,s.x,18) << 1;
                            start |= decision(&m,src,s.x,18); forced = 1;
                        }
                    }
                    for (int j = start; j < rows; ++j) {
                        int y = stripe+j, at = y*w+x;
                        if (pass == 1) {
                            if ((flags[at] & 3) == 1) {
                                uint context = 14 + ((flags[at] & 4) ? 2 : neighbors(flags,x,y,w,h,style) ? 1 : 0);
                                uint bit = raw ? raw_decision(&m,src,s.x) : decision(&m,src,s.x,context);
                                data[at] += (bit ^ (data[at] < 0)) ? midpoint : -midpoint;
                                flags[at] |= 4;
                            }
                        } else if (!(flags[at] & 3)) {
                            if (pass == 0 && !neighbors(flags,x,y,w,h,style)) continue;
                            uint bit = forced ? 1 : raw ? raw_decision(&m,src,s.x) : decision(&m,src,s.x,zero_context(flags,x,y,w,h,orient,style));
                            forced = 0;
                            if (bit) decode_sign(&m,src,s.x,data,flags,x,y,w,h,value,style,raw);
                            if (pass == 0) flags[at] |= 2;
                        }
                    }
                }
            }
            if (pass == 2) {
                for (int i = 0; i < w*h; ++i) flags[i] &= ~2;
                if (style & 32) {
                    uint symbol = 0;
                    for (int i = 0; i < 4; ++i) symbol = (symbol << 1) | decision(&m,src,s.x,18);
                    /* Upstream consumes but does not reject a bad SEGSYM.
                     * Preserve partial-quality output; retain a diagnostic bit. */
                    if (symbol != 10) status[block] |= 256;
                }
            }
            if ((style & 2) && !raw) reset_contexts(&m);
            if (++pass == 3) { pass = 0; --bp; }
        }
    }
}
