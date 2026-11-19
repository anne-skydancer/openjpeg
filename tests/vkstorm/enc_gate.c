/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Encoder gate of the Vulkanstorm hotfix 2.5.4-cpu-avx2.5 (not built by the project's CMake; build it
 * against a library of build_lib.cmd, see README.md).
 *
 *   enc_gate upload              the viewer's upload encodes (llimagej2coj.cpp JPEG2KEncode::encode:
 *                                RLCP, lossless one layer or lossy with estimate_num_layers layers and
 *                                set_tcp_rates, MCT for 3 or more components, the resolutions limited
 *                                for images under 64 px): 8-bit unsigned synthetic images of 32 to
 *                                1024 px, one line "name size fnv1a64" each
 *   enc_gate cfg                 nine encoder configurations of opj_compress (LRCP/RLCP/RPCL/PCRL/CPRL,
 *                                TERMALL, BYPASS, ERTERM+SEGSYM, 9/7, tiles, precincts, SOP/EPH)
 *   enc_gate rand8 <n> <seed>    n random encodes of 8-bit unsigned images and random parameters,
 *                                one line "i ok|fail size fnv1a64" each; the output of two library
 *                                builds must be identical line by line
 *   enc_gate term16 <n> <seed>   n random TERMALL encodes of 9 to 16 bit samples, signed or not, small
 *                                code-blocks (run under AddressSanitizer: the MQ flush of a terminated
 *                                pass wrote one byte past the code-block buffer)
 *   enc_gate guard <n> <seed>    n random lossless round trips of low precision images at deep
 *                                transforms; prints how many encoded and came back equal, how many
 *                                the encoder refused (not enough guard bits), and how many encoded
 *                                and came back different (the defect, must be 0)
 *   enc_gate guard1              the 33 x 31, two signed 1-bit components, five resolutions image
 *                                of neo-j2k OD-6; exit 0 when the encoder refuses it or it round-trips
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "openjpeg.h"

typedef struct {
    unsigned char *data;
    size_t size, capacity, pos;
} buf_t;

static OPJ_SIZE_T bwrite(void *p, OPJ_SIZE_T n, void *u)
{
    buf_t *b = (buf_t *)u;
    if (b->pos + n > b->capacity) {
        size_t c = b->capacity ? b->capacity : 4096;
        while (c < b->pos + n) {
            c *= 2;
        }
        b->data = (unsigned char *)realloc(b->data, c);
        b->capacity = c;
    }
    memcpy(b->data + b->pos, p, n);
    b->pos += n;
    if (b->pos > b->size) {
        b->size = b->pos;
    }
    return n;
}
static OPJ_SIZE_T bread(void *p, OPJ_SIZE_T n, void *u)
{
    buf_t *b = (buf_t *)u;
    size_t t;
    if (b->pos >= b->size) {
        return (OPJ_SIZE_T) - 1;
    }
    t = b->size - b->pos < n ? b->size - b->pos : n;
    memcpy(p, b->data + b->pos, t);
    b->pos += t;
    return t;
}
static OPJ_OFF_T bskip(OPJ_OFF_T n, void *u)
{
    buf_t *b = (buf_t *)u;
    OPJ_OFF_T np = (OPJ_OFF_T)b->pos + n;
    if (np < 0) {
        return -1;
    }
    if ((size_t)np > b->size) {
        np = (OPJ_OFF_T)b->size;
    }
    n = np - (OPJ_OFF_T)b->pos;
    b->pos = (size_t)np;
    return n;
}
static OPJ_BOOL bseek(OPJ_OFF_T n, void *u)
{
    buf_t *b = (buf_t *)u;
    if (n < 0 || (size_t)n > b->size) {
        return OPJ_FALSE;
    }
    b->pos = (size_t)n;
    return OPJ_TRUE;
}
static opj_stream_t *mkstream(buf_t *b, OPJ_BOOL in)
{
    opj_stream_t *s = opj_stream_create(4096, in);
    b->pos = 0;
    opj_stream_set_user_data(s, b, NULL);
    opj_stream_set_user_data_length(s, in ? b->size : 0);
    opj_stream_set_read_function(s, bread);
    opj_stream_set_write_function(s, bwrite);
    opj_stream_set_skip_function(s, bskip);
    opj_stream_set_seek_function(s, bseek);
    return s;
}
static void quiet(const char *m, void *u)
{
    (void)m;
    (void)u;
}
static int g_verbose;
static void say(const char *m, void *u)
{
    (void)u;
    if (g_verbose) {
        fputs(m, stderr);
    }
}

static uint64_t fnv(const unsigned char *p, size_t n)
{
    uint64_t h = 1469598103934665603ULL;
    size_t i;
    for (i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 1099511628211ULL;
    }
    return h;
}

static uint64_t g_rng = 88172645463325252ULL;
static uint32_t rnd(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}
static int rndr(int lo, int hi)
{
    return lo + (int)(rnd() % (uint32_t)(hi - lo + 1));
}

/* Pattern kinds: 0 noise, 1 smooth gradient, 2 sines and noise, 3 alternating extremes,
   4 constant, 5 checker of extremes by blocks, 6 sparse impulses */
static void fill(opj_image_t *img, int kind, int prec, int sgnd)
{
    OPJ_UINT32 i, x, y;
    const int lo = sgnd ? -(1 << (prec - 1)) : 0;
    const int hi = sgnd ? (1 << (prec - 1)) - 1 : (1 << prec) - 1;
    const int span = hi - lo;
    for (i = 0; i < img->numcomps; ++i) {
        const OPJ_UINT32 w = img->comps[i].w, h = img->comps[i].h;
        for (y = 0; y < h; ++y) {
            for (x = 0; x < w; ++x) {
                double v;
                switch (kind) {
                case 0:
                    v = lo + (double)(rnd() % (uint32_t)(span + 1));
                    break;
                case 1:
                    v = lo + span * ((x * 0.5 + y * 0.5 + i * 3) / (w + h + 1.0));
                    break;
                case 2:
                    v = lo + span * (0.5 + 0.35 * sin(x * 0.31 + i) * cos(y * 0.23) +
                                     0.1 * ((rnd() & 255) / 255.0 - 0.5));
                    break;
                case 3:
                    v = ((x + y + i) & 1) ? hi : lo;
                    break;
                case 4:
                    v = lo + span / 3 + (int)i;
                    break;
                case 5:
                    v = (((x / 3 + y / 3 + i) & 1)) ? hi : lo;
                    break;
                default:
                    v = (rnd() % 29 == 0) ? hi : (lo + span / 2);
                    break;
                }
                if (v < lo) {
                    v = lo;
                }
                if (v > hi) {
                    v = hi;
                }
                img->comps[i].data[y * w + x] = (OPJ_INT32)floor(v + 0.5);
            }
        }
    }
}

static opj_image_t *mkimage(int w, int h, int comps, int prec, int sgnd, int kind)
{
    opj_image_cmptparm_t cp[4];
    opj_image_t *img;
    int i;
    memset(cp, 0, sizeof(cp));
    for (i = 0; i < comps; ++i) {
        cp[i].dx = cp[i].dy = 1;
        cp[i].w = (OPJ_UINT32)w;
        cp[i].h = (OPJ_UINT32)h;
        cp[i].prec = (OPJ_UINT32)prec;
        cp[i].sgnd = (OPJ_UINT32)sgnd;
    }
    img = opj_image_create((OPJ_UINT32)comps, cp,
                           comps >= 3 ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY);
    img->x0 = img->y0 = 0;
    img->x1 = (OPJ_UINT32)w;
    img->y1 = (OPJ_UINT32)h;
    fill(img, kind, prec, sgnd);
    return img;
}

/* 1: encoded, 0: refused. Takes the image (opj_start_compress moves nothing out of it we use again). */
static int encode(opj_cparameters_t *p, opj_image_t *img, buf_t *out, const char *guard_opt)
{
    opj_codec_t *c = opj_create_compress(OPJ_CODEC_J2K);
    opj_stream_t *s;
    int ok;
    memset(out, 0, sizeof(*out));
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, quiet, NULL);
    opj_set_error_handler(c, say, NULL);
    s = mkstream(out, OPJ_FALSE);
    ok = opj_setup_encoder(c, p, img);
    if (ok && guard_opt) {
        const char *o[2];
        o[0] = guard_opt;
        o[1] = NULL;
        ok = opj_encoder_set_extra_options(c, o);
    }
    ok = ok && opj_start_compress(c, img, s) && opj_encode(c, s) && opj_end_compress(c, s);
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    if (!ok) {
        free(out->data);
        memset(out, 0, sizeof(*out));
    }
    return ok;
}

static void set_rates(opj_cparameters_t *p, int nl)
{
    int i;
    p->tcp_numlayers = nl;
    for (i = nl - 1; i >= 0; i--) {
        p->tcp_rates[nl - 1 - i] = 8.0f * (float)(1 << (i << 1));
    }
}

static int est_layers(unsigned s)
{
    return s <= 1024 ? 2 : s <= 16384 ? 3 : s <= 262144 ? 4 : s <= 1048576 ? 5 : 6;
}

static int run_upload(void)
{
    static const int dims[][2] = { {32, 32}, {64, 64}, {96, 48}, {128, 128}, {200, 120}, {256, 256}, {512, 512}, {1024, 1024}, {1024, 256} };
    int d, comps, kind, lossy, n = 0;
    for (d = 0; d < (int)(sizeof(dims) / sizeof(dims[0])); ++d) {
        for (comps = 1; comps <= 4; ++comps) {
            if (comps == 2) {
                continue;
            }
            for (kind = 0; kind < 7; ++kind) {
                if (dims[d][0] * dims[d][1] > 262144 && kind != 0 && kind != 2 && kind != 3) {
                    continue;
                }
                for (lossy = 0; lossy < 2; ++lossy) {
                    opj_cparameters_t p;
                    opj_image_t *img;
                    buf_t out;
                    int wt, ht, ok;
                    const int w = dims[d][0], h = dims[d][1];
                    g_rng = 88172645463325252ULL + (uint64_t)(d * 1000 + comps * 100 + kind * 10 + lossy);
                    img = mkimage(w, h, comps, 8, 0, kind);
                    opj_set_default_encoder_parameters(&p);
                    p.cod_format = OPJ_CODEC_J2K;
                    p.prog_order = OPJ_RLCP;
                    p.cp_disto_alloc = 1;
                    p.max_cs_size = 0;
                    if (!lossy) {
                        p.irreversible = 0;
                        p.tcp_numlayers = 1;
                        p.tcp_rates[0] = 0.0f;
                    } else {
                        p.irreversible = 1;
                    }
                    p.tcp_mct = comps >= 3 ? 1 : 0;
                    if (comps == 4) {
                        p.tcp_mct = 0;
                    }
                    if (lossy) {
                        set_rates(&p, est_layers((unsigned)(w * h)));
                    }
                    wt = w >> 6;
                    ht = h >> 6;
                    if (wt == 0 && w >= 32) {
                        wt = 1;
                    }
                    if (ht == 0 && h >= 32) {
                        ht = 1;
                    }
                    if (wt == 1 || ht == 1) {
                        int md = w < h ? w : h;
                        p.numresolution = 1 + (int)floor(log((double)md) / log(2.0));
                    }
                    ok = encode(&p, img, &out, NULL);
                    printf("upload %dx%d c%d k%d %s %s %zu %016llx\n", w, h, comps, kind,
                           lossy ? "lossy" : "lossless", ok ? "ok" : "fail", out.size,
                           (unsigned long long)fnv(out.data, out.size));
                    free(out.data);
                    opj_image_destroy(img);
                    ++n;
                }
            }
        }
    }
    fprintf(stderr, "upload: %d encodes\n", n);
    return 0;
}

static int run_cfg(void)
{
    static const struct {
        const char *name;
        int prog, mode, irr, tile, prec_sz, sop, eph, layers, cbw, cbh, res;
    } c[9] = {
        { "lrcp-default",   OPJ_LRCP, 0,   0, 0,  0, 0, 0, 1, 64, 64, 6 },
        { "rlcp-lossy",     OPJ_RLCP, 0,   1, 0,  0, 0, 0, 4, 64, 64, 6 },
        { "rpcl-precinct",  OPJ_RPCL, 0,   0, 0,  64, 0, 0, 2, 32, 32, 5 },
        { "pcrl-tiles",     OPJ_PCRL, 0,   0, 64, 0, 0, 0, 3, 64, 64, 4 },
        { "cprl-sop-eph",   OPJ_CPRL, 0,   1, 64, 0, 1, 1, 2, 64, 64, 5 },
        { "termall",        OPJ_LRCP, 4,   0, 0,  0, 0, 0, 2, 32, 32, 6 },
        { "bypass",         OPJ_LRCP, 1,   0, 0,  0, 0, 0, 2, 64, 64, 6 },
        { "erterm-segsym",  OPJ_RLCP, 32 | 64 | 8, 1, 0, 0, 0, 0, 3, 16, 16, 6 },
        { "reset-vsc-tiny", OPJ_LRCP, 2 | 8 | 16, 0, 0, 0, 0, 0, 1, 4, 4, 3 }
    };
    int i, kind;
    for (i = 0; i < 9; ++i) {
        for (kind = 0; kind < 3; ++kind) {
            opj_cparameters_t p;
            opj_image_t *img;
            buf_t out;
            int ok, comps = (kind == 1) ? 3 : 1;
            g_rng = 12345 + (uint64_t)i * 77 + (uint64_t)kind;
            img = mkimage(150, 130, comps, 8, 0, kind == 0 ? 2 : kind == 1 ? 0 : 3);
            opj_set_default_encoder_parameters(&p);
            p.prog_order = (OPJ_PROG_ORDER)c[i].prog;
            p.mode = c[i].mode;
            p.irreversible = c[i].irr;
            p.cp_disto_alloc = 1;
            p.tcp_numlayers = c[i].layers;
            if (c[i].layers > 1 || c[i].irr) {
                set_rates(&p, c[i].layers);
            } else {
                p.tcp_rates[0] = 0.0f;
            }
            p.tcp_mct = comps >= 3 ? 1 : 0;
            p.cblockw_init = c[i].cbw;
            p.cblockh_init = c[i].cbh;
            p.numresolution = c[i].res;
            if (c[i].tile) {
                p.tile_size_on = OPJ_TRUE;
                p.cp_tdx = p.cp_tdy = c[i].tile;
            }
            if (c[i].prec_sz) {
                int r;
                p.csty |= 1;
                p.res_spec = c[i].res;
                for (r = 0; r < c[i].res; ++r) {
                    p.prcw_init[r] = p.prch_init[r] = c[i].prec_sz;
                }
            }
            if (c[i].sop) {
                p.csty |= 2;
            }
            if (c[i].eph) {
                p.csty |= 4;
            }
            ok = encode(&p, img, &out, NULL);
            printf("cfg %s k%d %s %zu %016llx\n", c[i].name, kind, ok ? "ok" : "fail", out.size,
                   (unsigned long long)fnv(out.data, out.size));
            free(out.data);
            opj_image_destroy(img);
        }
    }
    return 0;
}

static int roundtrip(const buf_t *enc, const opj_image_t *src);

static int run_rand8(int n, uint64_t seed)
{
    int i;
    g_rng = seed * 2654435761ULL + 1;
    for (i = 0; i < n; ++i) {
        opj_cparameters_t p;
        opj_image_t *img, *ref;
        buf_t out;
        int w = rndr(1, 90), h = rndr(1, 90), comps = rndr(1, 4), ok, ci, rt;
        int kind = rndr(0, 6), layers = rndr(1, 4), irr = rndr(0, 1);
        int mode = 0, md = w < h ? w : h, maxres = 1;
        while ((1 << maxres) <= md) {
            ++maxres;
        }
        if (rndr(0, 1)) {
            mode = rndr(0, 63);
        }
        img = mkimage(w, h, comps, 8, 0, kind);
        /* opj_start_compress takes the samples out of img: keep a copy to compare with */
        ref = mkimage(w, h, comps, 8, 0, 4);
        for (ci = 0; ci < comps; ++ci) {
            memcpy(ref->comps[ci].data, img->comps[ci].data, (size_t)w * (size_t)h * sizeof(OPJ_INT32));
        }
        opj_set_default_encoder_parameters(&p);
        p.prog_order = (OPJ_PROG_ORDER)rndr(0, 4);
        p.mode = mode;
        p.irreversible = irr;
        p.cp_disto_alloc = 1;
        if (layers > 1 || irr) {
            set_rates(&p, layers);
        } else {
            p.tcp_rates[0] = 0.0f;
            p.tcp_numlayers = 1;
        }
        p.tcp_mct = (comps >= 3 && rndr(0, 3)) ? 1 : 0;
        p.cblockw_init = 1 << rndr(2, 6);
        p.cblockh_init = 1 << rndr(2, 6);
        if (p.cblockw_init * p.cblockh_init > 4096) {
            p.cblockh_init = 4096 / p.cblockw_init;
        }
        p.numresolution = rndr(1, maxres);
        if (rndr(0, 3) == 0) {
            p.csty |= 2 | 4;
        }
        if (rndr(0, 4) == 0) {
            p.tile_size_on = OPJ_TRUE;
            p.cp_tdx = rndr(16, 64);
            p.cp_tdy = rndr(16, 64);
        }
        ok = encode(&p, img, &out, NULL);
        /* a lossless encode must decode to the samples; lossy ones are not compared */
        rt = (ok && !irr && layers == 1) ? roundtrip(&out, ref) : -2;
        printf("%d %s %zu %016llx rt%d\n", i, ok ? "ok" : "fail", out.size,
               (unsigned long long)fnv(out.data, out.size), rt);
        if (g_verbose && (!ok || rt == 0)) {
            fprintf(stderr, "%d %s: %dx%d c%d kind %d irr %d layers %d mode %d mct %d res %d cb %dx%d prog %d csty %d tile %d\n",
                    i, ok ? "DIFFERENT" : "refused", w, h, comps, kind, irr, layers, mode, p.tcp_mct,
                    p.numresolution, p.cblockw_init, p.cblockh_init, (int)p.prog_order, p.csty,
                    p.tile_size_on ? p.cp_tdx : 0);
        }
        free(out.data);
        opj_image_destroy(img);
        opj_image_destroy(ref);
    }
    return 0;
}

static int run_term16(int n, uint64_t seed)
{
    int i, okc = 0, refused = 0;
    g_rng = seed * 2654435761ULL + 7;
    for (i = 0; i < n; ++i) {
        opj_cparameters_t p;
        opj_image_t *img;
        buf_t out;
        int w = rndr(1, 40), h = rndr(1, 40), comps = rndr(1, 3);
        int prec = rndr(9, 16), sgnd = rndr(0, 1), kind = rndr(0, 6);
        int mode = 4 | (rndr(0, 1) ? 1 : 0) | (rndr(0, 1) ? 32 : 0) | (rndr(0, 1) ? 8 : 0) |
                   (rndr(0, 1) ? 16 : 0) | (rndr(0, 1) ? 2 : 0);
        int md = w < h ? w : h, maxres = 1;
        while ((1 << maxres) <= md) {
            ++maxres;
        }
        img = mkimage(w, h, comps, prec, sgnd, kind);
        opj_set_default_encoder_parameters(&p);
        p.mode = mode;
        p.irreversible = 0;
        p.cp_disto_alloc = 1;
        p.tcp_numlayers = rndr(1, 4);
        set_rates(&p, p.tcp_numlayers);
        p.tcp_rates[p.tcp_numlayers - 1] = 0.0f;
        p.tcp_mct = (comps >= 3 && rndr(0, 1)) ? 1 : 0;
        p.cblockw_init = 1 << rndr(2, 5);
        p.cblockh_init = 1 << rndr(2, 5);
        p.numresolution = rndr(1, maxres > 3 ? 3 : maxres);
        if (rndr(0, 1)) {
            p.csty |= 4;
        }
        if (g_verbose) {
            fprintf(stderr, "term16 %d: %dx%d c%d prec %d sgnd %d kind %d mode %d layers %d mct %d cb %dx%d res %d csty %d\n",
                    i, w, h, comps, prec, sgnd, kind, mode, p.tcp_numlayers, p.tcp_mct, p.cblockw_init,
                    p.cblockh_init, p.numresolution, p.csty);
        }
        if (encode(&p, img, &out, "GUARD_BITS=7")) {
            ++okc;
        } else {
            ++refused;
        }
        free(out.data);
        opj_image_destroy(img);
    }
    printf("term16: %d encodes, %d ok, %d failed\n", n, okc, refused);
    return 0;
}

/* Lossless round trip; 1 equal, 0 different, -1 decode failed */
static int roundtrip(const buf_t *enc, const opj_image_t *src)
{
    opj_dparameters_t dp;
    opj_codec_t *c = opj_create_decompress(OPJ_CODEC_J2K);
    opj_stream_t *s;
    opj_image_t *dec = NULL;
    buf_t in = *enc;
    int r = -1;
    OPJ_UINT32 i;
    opj_set_default_decoder_parameters(&dp);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, quiet, NULL);
    opj_set_error_handler(c, quiet, NULL);
    opj_setup_decoder(c, &dp);
    s = mkstream(&in, OPJ_TRUE);
    if (opj_read_header(s, c, &dec) && opj_decode(c, s, dec) && dec->numcomps == src->numcomps) {
        r = 1;
        for (i = 0; i < src->numcomps && r == 1; ++i) {
            size_t k, nn = (size_t)src->comps[i].w * src->comps[i].h;
            for (k = 0; k < nn; ++k) {
                if (src->comps[i].data[k] != dec->comps[i].data[k]) {
                    r = 0;
                    break;
                }
            }
        }
    }
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    opj_image_destroy(dec);
    return r;
}

static int one_guard(int w, int h, int comps, int prec, int sgnd, int kind, int res, int mct,
                     int *encoded)
{
    opj_cparameters_t p;
    opj_image_t *img = mkimage(w, h, comps, prec, sgnd, kind);
    /* opj_start_compress takes the samples out of img: keep a copy to compare with */
    opj_image_t *ref = mkimage(w, h, comps, prec, sgnd, 4);
    buf_t out;
    int rt = -2;
    int ci;
    for (ci = 0; ci < comps; ++ci) {
        memcpy(ref->comps[ci].data, img->comps[ci].data,
               (size_t)w * (size_t)h * sizeof(OPJ_INT32));
    }
    opj_set_default_encoder_parameters(&p);
    p.cp_disto_alloc = 1;
    p.tcp_numlayers = 1;
    p.tcp_rates[0] = 0.0f;
    p.irreversible = 0;
    p.numresolution = res;
    p.tcp_mct = mct;
    p.cblockw_init = 32;
    p.cblockh_init = 32;
    *encoded = encode(&p, img, &out, NULL);
    if (*encoded) {
        rt = roundtrip(&out, ref);
    }
    free(out.data);
    opj_image_destroy(img);
    opj_image_destroy(ref);
    return rt;
}

static int run_guard(int n, uint64_t seed)
{
    int i, equal = 0, refused = 0, bad = 0, nodec = 0;
    g_rng = seed * 2654435761ULL + 3;
    for (i = 0; i < n; ++i) {
        int w = rndr(4, 70), h = rndr(4, 70), comps = rndr(1, 3), prec = rndr(1, 8);
        int sgnd = rndr(0, 1), kind = rndr(0, 6), md = w < h ? w : h, maxres = 1, enc, rt;
        while ((1 << maxres) <= md) {
            ++maxres;
        }
        rt = one_guard(w, h, comps, prec, sgnd, kind, rndr(1, maxres), (comps >= 3 && rndr(0, 1)) ? 1 : 0,
                       &enc);
        if (g_verbose && (!enc || rt == 0)) {
            fprintf(stderr, "%s: %dx%d c%d prec %d sgnd %d kind %d\n", enc ? "DIFFERENT" : "refused",
                    w, h, comps, prec, sgnd, kind);
        }
        if (!enc) {
            ++refused;
        } else if (rt == 1) {
            ++equal;
        } else if (rt == 0) {
            ++bad;
        } else {
            ++nodec;
        }
    }
    printf("guard: %d images: %d encoded and equal, %d refused, %d encoded and DIFFERENT, %d did not decode\n",
           n, equal, refused, bad, nodec);
    return bad ? 1 : 0;
}

static int run_guard1(void)
{
    int enc, rt = one_guard(33, 31, 2, 1, 1, 0, 5, 0, &enc);
    printf("guard1: %s\n", !enc ? "refused" : rt == 1 ? "encoded, round trip equal" : "encoded, DIFFERENT");
    return (!enc || rt == 1) ? 0 : 1;
}

int main(int argc, char **argv)
{
    g_verbose = getenv("ENC_GATE_VERBOSE") != NULL;
    if (argc >= 2 && !strcmp(argv[1], "upload")) {
        return run_upload();
    }
    if (argc >= 2 && !strcmp(argv[1], "cfg")) {
        return run_cfg();
    }
    if (argc >= 4 && !strcmp(argv[1], "rand8")) {
        return run_rand8(atoi(argv[2]), (uint64_t)atoll(argv[3]));
    }
    if (argc >= 4 && !strcmp(argv[1], "term16")) {
        return run_term16(atoi(argv[2]), (uint64_t)atoll(argv[3]));
    }
    if (argc >= 4 && !strcmp(argv[1], "guard")) {
        return run_guard(atoi(argv[2]), (uint64_t)atoll(argv[3]));
    }
    if (argc >= 2 && !strcmp(argv[1], "guard1")) {
        return run_guard1();
    }
    fprintf(stderr, "usage: enc_gate upload|cfg|rand8 n seed|term16 n seed|guard n seed|guard1\n");
    return 2;
}
