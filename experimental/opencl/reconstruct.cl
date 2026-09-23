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
 * Copyright (c) 2007, Jonathan Ballard <dzonatas@dzonux.net>
 * Copyright (c) 2007, Callum Lerwick <seg@haxxed.com>
 * Copyright (c) 2017, IntoPIX SA <support@intopix.com>
 * Copyright (c) 2006-2007, Parvatha Elangovan
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
 * Copyright (c) 2012, CS Systemes d'Information, France
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
 * Reconstruction semantics follow OpenJPEG v2.5.4 dwt.c/mct.c/tcd.c.
 * See upstream notices in the source tree and embedded kernel bundle.
 */
#pragma OPENCL FP_CONTRACT OFF

__kernel void place_blocks(__global const uint *desc, __global const uint4 *place,
                           __global const float *scale, __global const int *coeff,
                           __global int *planes, uint count, uint reversible)
{
    uint b = get_global_id(0);
    if (b >= count) return;
    uint4 p = place[b]; /* destination, stride, width, height */
    uint offset = desc[b*13+9];
    for (uint y=0; y<p.w; ++y) for (uint x=0; x<p.z; ++x) {
        int v = coeff[offset+y*p.z+x];
        uint target = p.x+y*p.y+x;
        if (reversible) planes[target] = v/2;
        else ((__global float*)planes)[target] = convert_float(v)*scale[b];
    }
}

/* One work-item owns an entire line; scratch lines are disjoint. This first
 * complete implementation favors transparent arithmetic over throughput.
 * Horizontal and vertical dispatches are ordered by the host queue. */
__kernel void inverse_line(__global int *planes, __global int *scratch,
                           uint base, uint stride, uint length, uint lines,
                           uint low_count, uint parity, uint vertical, uint reversible)
{
    uint line = get_global_id(0);
    if (line >= lines) return;
    uint origin = base + (vertical ? line : line*stride);
    uint step = vertical ? stride : 1;
    __global int *tmp = scratch + line*length;
    __global float *f = (__global float*)tmp;
    for (uint i=0; i<length; ++i) {
        uint source = ((i+parity)&1) ? low_count+i/2 : i/2;
        tmp[i] = planes[origin+source*step];
    }
    if (reversible) {
        if (length == 1) { if (parity) tmp[0] /= 2; }
        else {
            for (uint i=parity; i<length; i+=2) {
                uint l=i ? i-1 : 1, r=i+1<length ? i+1 : i-1;
                tmp[i] -= (tmp[l]+tmp[r]+2)>>2;
            }
            for (uint i=1-parity; i<length; i+=2) {
                uint l=i ? i-1 : 1, r=i+1<length ? i+1 : i-1;
                tmp[i] += (tmp[l]+tmp[r])>>1;
            }
        }
    } else if (length > 1) {
        for (uint i=0; i<length; ++i)
            f[i] *= ((i+parity)&1) ? 1.625732422f : 1.230174105f;
        const float lift[4] = {-0.443506852f,-0.882911075f,0.052980118f,1.586134342f};
        for (uint phase=0; phase<4; ++phase) {
            for (uint i=(phase&1) ? 1-parity : parity; i<length; i+=2) {
                uint l=i ? i-1 : 1, r=i+1<length ? i+1 : i-1;
                f[i] = f[i] + (f[l]+f[r])*lift[phase];
            }
        }
    }
    for (uint i=0; i<length; ++i) planes[origin+i*step] = tmp[i];
}

__kernel void finish_pixels(__global int *planes, uint samples, uint components,
                            uint mct, uint reversible, uint precision, uint is_signed,
                            int dc_shift)
{
    uint i=get_global_id(0);
    if (i>=samples) return;
    int lo=is_signed ? -(1<<(precision-1)) : 0;
    int hi=is_signed ? (1<<(precision-1))-1 : (1<<precision)-1;
    if (mct && components>=3) {
        if (reversible) {
            int y=planes[i], u=planes[samples+i], v=planes[2*samples+i];
            int g=y-((u+v)>>2);
            planes[i]=v+g; planes[samples+i]=g; planes[2*samples+i]=u+g;
        } else {
            __global float *p=(__global float*)planes;
            float y=p[i], u=p[samples+i], v=p[2*samples+i];
            p[i]=y+v*1.402f;
            p[samples+i]=(y-u*0.34413f)-v*0.71414f;
            p[2*samples+i]=y+u*1.772f;
        }
    }
    for (uint c=0;c<components;++c) {
        uint at=c*samples+i;
        long v;
        if (reversible) v=planes[at];
        else v=convert_long_sat_rte(((__global float*)planes)[at]);
        planes[at]=(int)clamp(v+(long)dc_shift,(long)lo,(long)hi);
    }
}
