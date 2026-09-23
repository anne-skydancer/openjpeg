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
 * Forward transforms, preserving the operation ordering of OpenJPEG v2.5.4.
 */
#pragma OPENCL FP_CONTRACT OFF
__kernel void prepare_encode(__global int *planes,uint samples,uint components,
    uint mct,uint reversible,int shift)
{
    uint i=get_global_id(0);if(i>=samples)return;
    for(uint c=0;c<components;++c) {
        uint at=c*samples+i;int v=planes[at]-shift;
        if(reversible)planes[at]=v;else ((__global float*)planes)[at]=convert_float(v);
    }
    if(mct&&components>=3) {
        if(reversible) {
            int r=planes[i],g=planes[samples+i],b=planes[2*samples+i];
            planes[i]=(r+2*g+b)>>2;planes[samples+i]=b-g;planes[2*samples+i]=r-g;
        } else {
            __global float *p=(__global float*)planes;
            float r=p[i],g=p[samples+i],b=p[2*samples+i];
            p[i]=(0.299f*r+0.587f*g)+0.114f*b;
            p[samples+i]=(-0.16875f*r-0.331260f*g)+0.5f*b;
            p[2*samples+i]=(0.5f*r-0.41869f*g)-0.08131f*b;
        }
    }
}
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void forward_line(__global int *planes,__local int *tmp,
    uint base,uint stride,uint length,uint lines,uint low_count,
    uint parity,uint vertical,uint reversible)
{
    uint line=get_group_id(0),lane=get_local_id(0);if(line>=lines)return;
    uint origin=base+(vertical?line:line*stride),step=vertical?stride:1;
    __local float *f=(__local float*)tmp;
    for(uint i=lane;i<length;i+=64)tmp[i]=planes[origin+i*step];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(reversible) {
        if(length==1) {if(parity&&lane==0)tmp[0]*=2;}
        else {
            for(uint i=1-parity+2*lane;i<length;i+=128) {
                uint l=i?i-1:1,r=i+1<length?i+1:i-1;
                tmp[i]-=(tmp[l]+tmp[r])>>1;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for(uint i=parity+2*lane;i<length;i+=128) {
                uint l=i?i-1:1,r=i+1<length?i+1:i-1;
                tmp[i]+=(tmp[l]+tmp[r]+2)>>2;
            }
        }
    } else if(length>1) {
        const float lift[4]={-1.586134342f,-0.052980118f,0.882911075f,0.443506852f};
        for(uint phase=0;phase<4;++phase) {
            for(uint i=((phase&1)?parity:1-parity)+2*lane;i<length;i+=128) {
                uint l=i?i-1:1,r=i+1<length?i+1:i-1;
                f[i]=f[i]+(f[l]+f[r])*lift[phase];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for(uint i=lane;i<length;i+=64)f[i]*=((i+parity)&1)?1.230174105f:0.812893093f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint i=lane;i<length;i+=64) {
        uint target=((i+parity)&1)?low_count+i/2:i/2;
        planes[origin+target*step]=tmp[i];
    }
}
