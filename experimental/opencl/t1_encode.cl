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
 * Part 1 MQ encoder, common style-zero path. OpenJPEG v2.5.4 semantics.
 * One work-item per block. Output includes a bounded prefix carry byte.
 */
typedef struct { MQ m; __global uchar *out; uint capacity, error; } ENCODER;
uint enc_byte(ENCODER *e,uint at) { return at<e->capacity?e->out[at]:0; }
void enc_put(ENCODER *e,uint at,uint value)
{
    if(at<e->capacity) e->out[at]=(uchar)value; else e->error=1;
}
void enc_byteout(ENCODER *e)
{
    MQ *m=&e->m;
    if(enc_byte(e,m->pos)==255) {
        enc_put(e,++m->pos,m->c>>20);m->c&=0xfffff;m->ct=7;
    } else if(!(m->c&0x8000000)) {
        enc_put(e,++m->pos,m->c>>19);m->c&=0x7ffff;m->ct=8;
    } else {
        enc_put(e,m->pos,enc_byte(e,m->pos)+1);
        if(enc_byte(e,m->pos)==255) {
            m->c&=0x7ffffff;enc_put(e,++m->pos,m->c>>20);m->c&=0xfffff;m->ct=7;
        } else { enc_put(e,++m->pos,m->c>>19);m->c&=0x7ffff;m->ct=8; }
    }
}
void enc_bit(ENCODER *e,uint context,uint bit)
{
    MQ *m=&e->m;
    uint4 s=mq_states[context_state(m,context)];
    m->a-=s.x;
    if(bit==s.y) {
        if(m->a&0x8000) { m->c+=s.x;return; }
        if(m->a<s.x) m->a=s.x; else m->c+=s.x;
        set_context(m,context,s.z);
    } else {
        if(m->a<s.x) m->c+=s.x; else m->a=s.x;
        set_context(m,context,s.w);
    }
    /* Shift directly to the next normalization or byte-output boundary. */
    uint shifts=clz(m->a)-16;
    do {
        uint n=min(shifts,m->ct);
        m->a<<=n;m->c<<=n;m->ct-=n;shifts-=n;
        if(!m->ct)enc_byteout(e);
    } while(shifts);
}
void enc_flush(ENCODER *e)
{
    MQ *m=&e->m;uint temp=m->c+m->a;m->c|=0xffff;
    if(m->c>=temp) m->c-=0x8000;
    m->c<<=m->ct;enc_byteout(e);m->c<<=m->ct;enc_byteout(e);
    if(enc_byte(e,m->pos)!=255) ++m->pos;
}
void enc_sign(ENCODER *e,__local uint *flags,int x,int y,int w,int h,uint sign)
{
    uint f=sample_flags(flags,x,y,w), stripe=(y>>2)*w+x, row=y&3;
    uint packed=flags[stripe], north=row?19+3*(row-1):18;
    uint south=row<3?19+3*(row+1):31;
    uint signs=((packed>>north)&1u)<<11 | ((packed>>south)&1u)<<14;
    if(x) signs|=((flags[stripe-1]>>(19+3*row))&1u)<<12;
    if(x+1<w) signs|=((flags[stripe+1]>>(19+3*row))&1u)<<13;
    f|=signs;
    uint lu=(f&170u)|((f>>12)&1u)|((f>>11)&4u)|((f>>7)&16u)|((f>>8)&64u);
    enc_bit(e,lut_ctxno_sc[lu],sign^lut_spb[lu]);
    flags[stripe]|=sign<<(19+3*row);
    for(int dx=-1;dx<=1;++dx) {
        int tx=x+dx;if(tx<0||tx>=w)continue;
        flags[(y>>2)*w+tx]|=1u<<(3*(row+1)+1-dx);
        if(row==0&&y>0) flags[((y>>2)-1)*w+tx]|=1u<<(16-dx);
        if(row==3&&y+1<h) flags[((y>>2)+1)*w+tx]|=1u<<(1-dx);
    }
    if(row==0&&y>0)flags[stripe-w]|=sign<<31;
    if(row==3&&y+1<h)flags[stripe+w]|=sign<<18;
}
__attribute__((always_inline)) inline int encode_pass(ENCODER *e,
    __global const int *data,__local uint *flags,int w,int h,int stride,uint orient,
    uint pass,int bp)
{
    uint one=1u<<(bp+6);int nmse=0;
    for(int stripe=0;stripe<h;stripe+=4) {
        int rows=min(4,h-stripe);
        for(int x=0;x<w;++x) {
            int start=0,forced=0;
            if(pass==2&&rows==4&&!(flags[(stripe>>2)*w+x]&(0x3ffffu|PI_ALL))) {
                while(start<4 && !(abs(data[(stripe+start)*stride+x])&one))++start;
                enc_bit(e,17,start<4);
                if(start==4)continue;
                enc_bit(e,18,(uint)start>>1);enc_bit(e,18,(uint)start&1);forced=1;
            }
            for(int j=start;j<rows;++j) {
                int y=stripe+j,at=y*stride+x;uint f=sample_flags(flags,x,y,w),mag=abs(data[at]);
                uint bit=(mag&one)!=0,index=(mag>>bp)&127;
                if(pass==1) {
                    if((f&(SIG|VISITED))==SIG) {
                        enc_bit(e,14+((f&REFINED)?2:(f&NEIGHBORS)?1:0),bit);
                        nmse+=bp?lut_nmsedec_ref[index]:lut_nmsedec_ref0[index];
                        mark_flag(flags,x,y,w,1u<<20);
                    }
                } else if(!(f&(SIG|VISITED))) {
                    if(pass==0&&!(f&NEIGHBORS))continue;
                    if(!forced)enc_bit(e,lut_ctxno_zc[orient*512+(f&NEIGHBORS)],bit);
                    forced=0;
                    if(bit) {
                        enc_sign(e,flags,x,y,w,h,data[at]<0);
                        nmse+=bp?lut_nmsedec_sig[index]:lut_nmsedec_sig0[index];
                    }
                    if(pass==0)mark_flag(flags,x,y,w,1u<<21);
                }
            }
        }
    }
    return nmse;
}

/* Descriptor: width,height,orientation,coefficient offset,output offset,capacity,row stride.
 * Result: numbps,passes,bytes,error, followed by 90 (rate,nmsedec) pairs.
 * Coefficients carry the upstream six fractional distortion bits. */
__attribute__((reqd_work_group_size(1,1,1)))
__kernel void encode_blocks(__global const uint *desc,__global const int *coeff,
    __global uchar *output,__global uint *results,__local uint *flags,uint count)
{
    uint block=get_global_id(0);if(block>=count)return;
    __global const uint *d=desc+7*block;
    int w=d[0],h=d[1],stride=d[6];uint orient=d[2];
    __global const int *data=coeff+d[3];__global uint *result=results+184*block;
    for(uint i=0;i<184;++i)result[i]=0;
    if(w<1||h<1||w>1024||h>1024||w*h>4096||orient>3||d[5]<2){result[3]=1;return;}
    for(int i=0;i<w*((h+3)/4);++i)flags[i]=0;
    uint maximum=0;
    for(int i=0;i<w*h;++i)maximum=max(maximum,abs(data[(i/w)*stride+i%w]));
    if(maximum>=0x80000000u){result[3]=2;return;}
    uint numbps=maximum>=64?26-clz(maximum):0;
    result[0]=numbps;if(!numbps)return;
    ENCODER e;e.out=output+d[4];e.capacity=d[5];e.error=0;
    e.m.a=0x8000;e.m.c=0;e.m.ct=12;e.m.pos=0;reset_contexts(&e.m);enc_put(&e,0,0);
    uint pass=2,passno=0;
    for(int bp=numbps-1;bp>=0;) {
        int nmse=0;
        if(w==64 && h==64) {
            if(pass==0)nmse=encode_pass(&e,data,flags,64,64,stride,orient,0,bp);
            else if(pass==1)nmse=encode_pass(&e,data,flags,64,64,stride,orient,1,bp);
            else nmse=encode_pass(&e,data,flags,64,64,stride,orient,2,bp);
        } else {
            if(pass==0)nmse=encode_pass(&e,data,flags,w,h,stride,orient,0,bp);
            else if(pass==1)nmse=encode_pass(&e,data,flags,w,h,stride,orient,1,bp);
            else nmse=encode_pass(&e,data,flags,w,h,stride,orient,2,bp);
        }
        if(pass==2)for(int i=0;i<w*((h+3)/4);++i)flags[i]&=~PI_ALL;
        if(pass==2&&bp==0)enc_flush(&e);
        result[4+2*passno]=e.m.pos-1+((pass==2&&bp==0)?0:3);
        result[5+2*passno]=nmse;++passno;
        if(++pass==3){pass=0;--bp;}
    }
    result[1]=passno;result[2]=e.m.pos-1;result[3]=e.error;
    uint last=result[2];
    for(int p=(int)passno-1;p>=0;--p){last=min(last,result[4+2*p]);result[4+2*p]=last;}
    for(uint p=0;p<passno;++p){uint rate=result[4+2*p];if(rate&&enc_byte(&e,rate)==255)--result[4+2*p];}
}
