/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef OPJ_ACCEL_DECODE_PLAN_H
#define OPJ_ACCEL_DECODE_PLAN_H
/* Private host-side plan shared by CUDA and OpenCL; included after opj_includes.h. */
typedef struct {
    OPJ_UINT32 blocks, segments, bytes, coefficients, max_flag_bytes;
    OPJ_UINT32 *desc, *segs, *place, *status;
    OPJ_BYTE *input;
    float *scale;
} decode_plan;

/* Two passes: count and validate before any GPU allocation; then copy an owned
 * compact plan. All totals are bounded before 32-bit device offsets are formed. */
static OPJ_BOOL plan_blocks(opj_tcd_t *tcd, decode_plan *p, OPJ_UINT32 stride,
                           OPJ_UINT32 samples, OPJ_UINT32 components, int copy)
{
    OPJ_UINT32 c,r,b,pr,k,s,ch;
    OPJ_UINT32 nb=0, ns=0, nbytes=0, nc=0, max_flags=0;
    for(c=0;c<components;c++) {
        opj_tcd_tilecomp_t *tc=&tcd->tcd_image->tiles->comps[c];
        opj_tccp_t *coding=&tcd->tcp->tccps[c];
        for(r=0;r<tc->minimum_num_resolutions;r++) {
            opj_tcd_resolution_t *res=&tc->resolutions[r];
            if(res->numbands>3 || (OPJ_UINT64)res->pw*res->ph>65536) return OPJ_FALSE;
            for(b=0;b<res->numbands;b++) {
                opj_tcd_band_t *band=&res->bands[b];
                if(band->bandno>3 || (!r && band->bandno) || !(band->stepsize>0.0f)) return OPJ_FALSE;
                for(pr=0;pr<res->pw*res->ph;pr++) {
                    opj_tcd_precinct_t *prec=&band->precincts[pr];
                    if((OPJ_UINT64)prec->cw*prec->ch>65536) return OPJ_FALSE;
                    for(k=0;k<prec->cw*prec->ch;k++) {
                        opj_tcd_cblk_dec_t *block=&prec->cblks.dec[k];
                        OPJ_INT32 w=block->x1-block->x0, h=block->y1-block->y0;
                        OPJ_INT32 x=block->x0-band->x0, y=block->y0-band->y0;
                        OPJ_UINT64 len=0, seglen=0, passes=0;
                        OPJ_UINT32 offset=nbytes;
                        OPJ_INT32 bitplane;
                        OPJ_UINT32 coding_pass=2;
                        if(w==0 || h==0) continue;
                        if(w<0 || h<0 || w>1024 || h>1024 || w*h>4096 || block->corrupted ||
                                block->numbps>30 || coding->roishift<0 || coding->roishift>30 ||
                                block->numbps+(OPJ_UINT32)coding->roishift>30 || coding->cblksty & ~63u ||
                                block->real_num_segs>90 || block->numchunks>65536) return OPJ_FALSE;
                        for(ch=0;ch<block->numchunks;ch++) len+=block->chunks[ch].len;
                        bitplane=(OPJ_INT32)block->numbps+coding->roishift;
                        for(s=0;s<block->real_num_segs;s++) {
                            OPJ_UINT32 pass;
                            OPJ_BOOL raw=(coding->cblksty&1) && bitplane<=(OPJ_INT32)block->numbps-4 && coding_pass<2;
                            if(!block->segs[s].real_num_passes || block->segs[s].real_num_passes>90) return OPJ_FALSE;
                            seglen+=block->segs[s].len; passes+=block->segs[s].real_num_passes;
                            for(pass=0;pass<block->segs[s].real_num_passes;pass++) {
                                OPJ_BOOL mode=(coding->cblksty&1) && bitplane<=(OPJ_INT32)block->numbps-4 && coding_pass<2;
                                if(bitplane<1 || mode!=raw) return OPJ_FALSE;
                                if(++coding_pass==3) { coding_pass=0;--bitplane; }
                            }
                        }
                        if(len!=seglen || len>65536 || passes>3u*(block->numbps+coding->roishift) ||
                                nb>=65536 || ns+block->real_num_segs>65536*90u ||
                                (OPJ_UINT64)nbytes+len>BUDGET || (OPJ_UINT64)nc+w*h>BUDGET/4) return OPJ_FALSE;
                        if(band->bandno&1) x+=tc->resolutions[r-1].x1-tc->resolutions[r-1].x0;
                        if(band->bandno&2) y+=tc->resolutions[r-1].y1-tc->resolutions[r-1].y0;
                        if(x<0 || y<0 || (OPJ_UINT32)(x+w)>stride || (OPJ_UINT64)(y+h)*stride>samples) return OPJ_FALSE;
                        if(copy) {
                            OPJ_UINT32 *d=p->desc+nb*12, *place=p->place+nb*4;
                            d[0]=w;d[1]=h;d[2]=band->bandno;d[3]=block->numbps;d[4]=coding->cblksty;
                            d[5]=nbytes;d[6]=(OPJ_UINT32)len;d[7]=ns;d[8]=block->real_num_segs;d[9]=nc;
                            d[10]=(OPJ_UINT32)coding->roishift;
                            d[11]=(tcd->tcp->num_layers_to_decode==tcd->tcp->numlayers && (tcd->tcp->tccps[0].cblksty&16)) ? 1:0;
                            place[0]=c*samples+y*stride+x;place[1]=stride;place[2]=w;place[3]=h;
                            p->scale[nb]=0.5f*band->stepsize;
                            for(ch=0;ch<block->numchunks;ch++) {
                                memcpy(p->input+offset,block->chunks[ch].data,block->chunks[ch].len);
                                offset+=block->chunks[ch].len;
                            }
                            for(s=0;s<block->real_num_segs;s++) {
                                p->segs[(ns+s)*2]=block->segs[s].len;
                                p->segs[(ns+s)*2+1]=block->segs[s].real_num_passes;
                            }
                        }
                        /* Four rows share a word; partial stripes still need a full word. */
                        max_flags=opj_uint_max(max_flags,(OPJ_UINT32)(w*((h+3)/4)*4));
                        ++nb; ns+=block->real_num_segs; nbytes+=(OPJ_UINT32)len; nc+=w*h;
                    }
                }
            }
        }
    }
    if(!copy) { p->blocks=nb;p->segments=ns;p->bytes=nbytes;p->coefficients=nc;p->max_flag_bytes=max_flags; }
    return nb>0;
}


#endif
