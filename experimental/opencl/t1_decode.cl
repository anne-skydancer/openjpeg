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
 * Experimental Part 1 MQ/RAW path, ROI undo and PTERM diagnostics; no HT.
 * Each one-item work-group owns bounded packed stripe flags in local memory.
 * Context tables and constant-pass/common-geometry specialization reduce the
 * work per ordered MQ decision without changing the codestream.
 */
typedef struct { uint a, c, ct, pos, synthesized; uint ctx0, ctx1, ctx2, ctx3, ctx4; } MQ;

uint input_byte(__global const uchar *src, uint length, uint pos)
{
    return pos < length ? src[pos] : 255u;
}
void bytein(MQ *m, __global const uchar *src, uint length)
{
    uint next = input_byte(src, length, m->pos + 1);
    if (input_byte(src, length, m->pos) == 255) {
        if (next > 143) { m->c += 0xff00; m->ct = 8; ++m->synthesized; }
        else { ++m->pos; m->c += next << 9; m->ct = 7; }
    } else { ++m->pos; m->c += next << 8; m->ct = 8; }
}
uint context_state(MQ *m,uint context)
{
    uint word=context<4 ? m->ctx0 : context<8 ? m->ctx1 : context<12 ? m->ctx2 : context<16 ? m->ctx3 : m->ctx4;
    return (word >> ((context&3)*8)) & 255;
}
void set_context(MQ *m,uint context,uint state)
{
    uint shift=(context&3)*8,mask=~(255u<<shift),value=state<<shift;
    if(context<4) m->ctx0=(m->ctx0&mask)|value;
    else if(context<8) m->ctx1=(m->ctx1&mask)|value;
    else if(context<12) m->ctx2=(m->ctx2&mask)|value;
    else if(context<16) m->ctx3=(m->ctx3&mask)|value;
    else m->ctx4=(m->ctx4&mask)|value;
}
void reset_contexts(MQ *m)
{
    m->ctx0=8; m->ctx1=0; m->ctx2=0; m->ctx3=0; m->ctx4=(6u<<8)|(92u<<16);
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
    m->pos = 0; m->synthesized = 0; m->c = input_byte(src, length, 0) << 16;
    bytein(m, src, length); m->c <<= 7; m->ct -= 7; m->a = 0x8000;
}
uint decision(MQ *m, __global const uchar *src, uint length, uint context)
{
    uint4 s = mq_states[context_state(m,context)]; /* Qe, MPS, NMPS, NLPS */
    uint bit;
    m->a -= s.x;
    if ((m->c >> 16) < s.x) {
        if (m->a < s.x) { bit = s.y; set_context(m,context,s.z); }
        else { bit = 1 - s.y; set_context(m,context,s.w); }
        m->a = s.x;
    } else {
        m->c -= s.x << 16;
        if (m->a & 0x8000) return s.y;
        if (m->a < s.x) { bit = 1 - s.y; set_context(m,context,s.w); }
        else { bit = s.y; set_context(m,context,s.z); }
    }
    do {
        if (!m->ct) bytein(m, src, length);
        m->a <<= 1; m->c <<= 1; --m->ct;
    } while (m->a < 0x8000);
    return bit;
}
/* Four-row stripe flags, following OpenJPEG's SIGMA/CHI/MU/PI layout.
 * Bits 0..17: significance for three columns and six rows (with vertical halo).
 * Bits 18,19,22,25,28,31: vertical sign samples, including the halo.
 * Bits 20,23,26,29: refinement; 21,24,27,30: significance-pass visits.
 * No padding words: boundary propagation is explicitly bounded. */
#define SIG 16u
#define VISITED 512u
#define REFINED 1024u
#define NEIGHBORS 495u
#define PI_ALL ((1u<<21)|(1u<<24)|(1u<<27)|(1u<<30))
/* Present the current row as significance (0..8), visited (9), refined (10). */
uint sample_flags(__local const uint *flags,int x,int y,int w)
{
    uint f=flags[(y>>2)*w+x] >> (3*(y&3));
    return (f&511u) | ((f>>12)&512u) | ((f>>10)&1024u);
}
void mark_flag(__local uint *flags,int x,int y,int w,uint bit)
{
    flags[(y>>2)*w+x] |= bit << (3*(y&3));
}
int neighbors(__local const uint *flags, int x, int y, int w, int h, uint style)
{
    return (sample_flags(flags,x,y,w) & NEIGHBORS) != 0;
}
uint zero_context(__local const uint *flags, int x, int y, int w, int h, uint orient, uint style)
{
    return lut_ctxno_zc[orient*512+(sample_flags(flags,x,y,w)&NEIGHBORS)];
}
void decode_sign(MQ *m, __global const uchar *src, uint length,
                 __global int *data, __local uint *flags,
                 int x, int y, int w, int h, int value, uint style, uint raw)
{
    uint f=sample_flags(flags,x,y,w), stripe=(y>>2)*w+x, row=y&3;
    uint packed=flags[stripe], north=row ? 19+3*(row-1) : 18;
    uint south=row<3 ? 19+3*(row+1) : 31;
    uint signs=((packed>>north)&1u)<<11 | ((packed>>south)&1u)<<14;
    if(x) signs |= ((flags[stripe-1]>>(19+3*row))&1u)<<12;
    if(x+1<w) signs |= ((flags[stripe+1]>>(19+3*row))&1u)<<13;
    f |= signs;
    uint lu=(f&170u) | ((f>>12)&1u) | ((f>>11)&4u) | ((f>>7)&16u) | ((f>>8)&64u);
    uint prediction=lut_spb[lu], context=lut_ctxno_sc[lu];
    uint sign = raw ? raw_decision(m,src,length) : decision(m,src,length,context) ^ prediction;
    data[y*w+x] = sign ? -value : value;
    flags[stripe] |= sign << (19+3*row);
    /* Propagate a single significance bit per neighbouring stripe column. */
    for(int dx=-1;dx<=1;++dx) {
        int tx=x+dx;
        if(tx<0 || tx>=w) continue;
        flags[(y>>2)*w+tx] |= 1u << (3*(row+1)+1-dx);
        if(row==0 && y>0 && !(style&8))
            flags[((y>>2)-1)*w+tx] |= 1u << (16-dx);
        if(row==3 && y+1<h)
            flags[((y>>2)+1)*w+tx] |= 1u << (1-dx);
    }
    if(row==0 && y>0 && !(style&8)) flags[stripe-w] |= sign<<31;
    if(row==3 && y+1<h) flags[stripe+w] |= sign<<18;

}

/* Constant pass arguments remove pass dispatch from coefficient loops. */
__attribute__((always_inline)) inline void decode_pass(MQ *m,
    __global const uchar *src,uint length,__global int *data,__local uint *flags,
    int w,int h,uint orient,uint style,uint pass,uint raw,int midpoint,int value)
{
    for (int stripe = 0; stripe < h; stripe += 4) {
        int rows = min(4,h-stripe);
        for (int x = 0; x < w; ++x) {
            int start = 0, forced = 0;
            if (pass == 2 && rows == 4) {
                int aggregate = !(flags[(stripe>>2)*w+x] & (0x3ffffu|PI_ALL));
                if (aggregate) {
                    if (!decision(m,src,length,17)) continue;
                    start = (int)decision(m,src,length,18) << 1;
                    start |= decision(m,src,length,18); forced = 1;
                }
            }
            for (int j = start; j < rows; ++j) {
                int y = stripe+j, at = y*w+x;
                if (pass == 1) {
                    if ((sample_flags(flags,x,y,w) & (SIG|VISITED)) == SIG) {
                        uint context = 14 + ((sample_flags(flags,x,y,w) & REFINED) ? 2 : neighbors(flags,x,y,w,h,style) ? 1 : 0);
                        uint bit = raw ? raw_decision(m,src,length) : decision(m,src,length,context);
                        data[at] += (bit ^ (data[at] < 0)) ? midpoint : -midpoint;
                        mark_flag(flags,x,y,w,1u<<20);
                    }
                } else if (!(sample_flags(flags,x,y,w) & (SIG|VISITED))) {
                    if (pass == 0 && !neighbors(flags,x,y,w,h,style)) continue;
                    uint bit = forced ? 1 : raw ? raw_decision(m,src,length) : decision(m,src,length,zero_context(flags,x,y,w,h,orient,style));
                    forced = 0;
                    if (bit) decode_sign(m,src,length,data,flags,x,y,w,h,value,style,raw);
                    if (pass == 0) mark_flag(flags,x,y,w,1u<<21);
                }
            }
        }
    }
}

/* Descriptor: w,h,orientation,numbps,style,inputOffset,inputLength,
 * segmentOffset,segmentCount,coefficientOffset,roiShift,checkPterm.
 * Segments: length, passCount.
 * Host verifies every range before upload; each block owns disjoint scratch.
 */
__attribute__((reqd_work_group_size(1,1,1)))
__kernel void decode_blocks(__global const uint *desc, __global const uint2 *segments,
                            __global const uchar *input, __global int *output,
                            __local uint *flags, __global uint *status, uint count)
{
    size_t block = get_global_id(0);
    if (block >= count) return;
    __global const uint *d = desc + block*12;
    int w = d[0], h = d[1];
    uint orient = d[2], style = d[4], offset = 0;
    __global int *data = output + d[9];
    status[block] = 0;
    if (style & ~63u || w <= 0 || h <= 0 || w*h > 4096 || orient > 3 || d[3] > 30 || d[10] > 30 || d[3]+d[10] > 30) {
        status[block] = 1; return;
    }
    for (int i = 0; i < w*h; ++i) data[i] = 0;
    for (int i = 0; i < w*((h+3)/4); ++i) flags[i] = 0;
    MQ m; reset_contexts(&m); m.synthesized = 0;
    int bp = d[3]+d[10]; uint pass = 2, last_length = 0;
    for (uint seg = 0; seg < d[8]; ++seg) {
        uint2 s = segments[d[7]+seg];
        last_length = s.x;
        if (offset > d[6] || s.x > d[6]-offset) { status[block] = 2; return; }
        __global const uchar *src = input + d[5] + offset;
        offset += s.x;
        uint raw = (style & 1) && bp <= (int)d[3]-4 && pass < 2;
        if (raw) { m.c = 0; m.ct = 0; m.pos = 0; }
        else init_segment(&m,src,s.x);
        for (uint p = 0; p < s.y && bp >= 1; ++p) {
            int midpoint = (1 << bp) >> 1, value = (1 << bp) | midpoint;
            /* The common 64x64/MQ path gives the compiler constant geometry,
             * coding style and pass. Edge blocks and all other styles stay GPU. */
            if (w==64 && h==64 && style==0) {
                if (pass==0) decode_pass(&m,src,s.x,data,flags,64,64,orient,0,0,0,midpoint,value);
                else if (pass==1) decode_pass(&m,src,s.x,data,flags,64,64,orient,0,1,0,midpoint,value);
                else decode_pass(&m,src,s.x,data,flags,64,64,orient,0,2,0,midpoint,value);
            }
            else if (pass == 0) decode_pass(&m,src,s.x,data,flags,w,h,orient,style,0,raw,midpoint,value);
            else if (pass == 1) decode_pass(&m,src,s.x,data,flags,w,h,orient,style,1,raw,midpoint,value);
            else decode_pass(&m,src,s.x,data,flags,w,h,orient,style,2,0,midpoint,value);
            if (pass == 2) {
                for (int i = 0; i < w*((h+3)/4); ++i) flags[i] &= ~PI_ALL;
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
    if (d[11] && d[8]) {
        if (m.pos + 2 < last_length) status[block] |= 512;
        else if (m.synthesized > 2) status[block] |= 1024;
    }
    if (d[10]) {
        int threshold = 1 << d[10];
        for (int i = 0; i < w*h; ++i) {
            int magnitude = abs(data[i]);
            if (magnitude >= threshold) {
                magnitude >>= d[10];
                data[i] = data[i] < 0 ? -magnitude : magnitude;
            }
        }
    }
}
