/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * neo-j2k stage 1a probe for the OpenJPEG fork (derived from the fork-audit probe of 2026-11-18).
 * Drives libopenjp2 the way the viewer does (memory stream, strict mode off, reduce factor set
 * before the header), then:
 *   conf  <data dir>                         ISO/IEC 15444-4 class 0 (reduced) and class 1 checks
 *   cut   <file> <reduce list> <step> <csv>  decode every prefix length, PSNR against the whole file
 *   cuthash <file> <reduce list> <step> <out> decode every prefix length, one output hash per prefix
 *   one   <file> <reduce> <len>              decode one prefix, print result and first error
 *   hash  <reduce list> <files...>           whole-file output hashes
 *   mem   <file> <reduce> [len]              decode once: result, peak commit and working set, time
 *   hdr   <W> <H> <tdx> <tdy> <comps> <layers>  synthetic header: time and peak memory of read_header
 *   loops <W> <layers> <Scod> [pp]           synthetic one-packet stream: time of the packet loop
 *   setfactor <file> <factor>                read_header, then set_decoded_resolution_factor, then decode
 *   census <file> <reduce> [repeat]          allocation census (only with the census build)
 *   layhash <file> <reduce list> <layers>    whole-file hashes limited to 1..layers layers (cp_layer),
 *                                            in the cuthash format: the oracle of b5_check.py
 *   cache <texture.entries> <texture.cache> <reduce>  the viewer cache's headers under the limits profile
 * PROBE_PROFILE=sl in the environment selects the decoder limits profile OPJ_DECODER_LIMITS_SL_TEXTURE
 * (only in builds with PROBE_LIMITS, against a library that has opj_decoder_set_limits).
 * PROBE_LAYERS=<n> limits every decode to n layers (cp_layer); PROBE_JP2=1 decodes JP2 files;
 * PROBE_STRICT=1 decodes in strict mode (OpenJPEG's default) instead of the viewer's non-strict mode.
 * Not a viewer component; nothing here ships.
 */
#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "openjpeg.h"

/* ---------------------------------------------------------------- memory stream (as llimagej2coj.cpp) */
typedef struct {
    const unsigned char *buf;
    size_t size;
    OPJ_OFF_T off;
} mem_t;

static OPJ_SIZE_T mem_read(void *b, OPJ_SIZE_T n, void *u)
{
    mem_t *m = (mem_t *)u;
    size_t rem, t;
    if (m->off < 0 || (size_t)m->off >= m->size) {
        m->off = (OPJ_OFF_T)m->size;
        return (OPJ_SIZE_T) - 1;
    }
    rem = m->size - (size_t)m->off;
    t = n < rem ? n : rem;
    memcpy(b, m->buf + m->off, t);
    m->off += (OPJ_OFF_T)t;
    return t;
}
static OPJ_OFF_T mem_skip(OPJ_OFF_T n, void *u)
{
    mem_t *m = (mem_t *)u;
    OPJ_OFF_T no = m->off + n;
    if (no < 0 || no > (OPJ_OFF_T)m->size) {
        m->off = no < 0 ? 0 : (OPJ_OFF_T)m->size;
        return (OPJ_OFF_T) - 1;
    }
    m->off = no;
    return n;
}
static OPJ_BOOL mem_seek(OPJ_OFF_T o, void *u)
{
    mem_t *m = (mem_t *)u;
    if (o < 0 || o > (OPJ_OFF_T)m->size) {
        return OPJ_FALSE;
    }
    m->off = o;
    return OPJ_TRUE;
}

static int g_profile = 0; /* 1: OPJ_DECODER_LIMITS_SL_TEXTURE */
static unsigned g_layers =
    0; /* cp_layer of every decode (PROBE_LAYERS), 0: all */
static int g_jp2 = 0; /* PROBE_JP2=1: decode as JP2 (OPJ_CODEC_JP2) */
static int g_strict = 0; /* PROBE_STRICT=1: every decode in strict mode */

static void apply_profile(opj_codec_t *c)
{
#ifdef PROBE_LIMITS
    if (g_profile) {
        opj_decoder_limits_t l;
        if (!opj_decoder_limits_init(&l, OPJ_DECODER_LIMITS_SL_TEXTURE) ||
                !opj_decoder_set_limits(c, &l)) {
            fprintf(stderr, "cannot set the limits profile\n");
            exit(3);
        }
    }
#else
    (void)c;
#endif
}

typedef struct {
    char err[256];
    char warn[256];
    int nerr, nwarn;
} msgs_t;
static void cb_err(const char *msg, void *u)
{
    msgs_t *s = (msgs_t *)u;
    if (s) {
        if (!s->nerr) {
            strncpy(s->err, msg, sizeof(s->err) - 1);
        }
        s->nerr++;
    }
}
static void cb_warn(const char *msg, void *u)
{
    msgs_t *s = (msgs_t *)u;
    if (s) {
        if (!s->nwarn) {
            strncpy(s->warn, msg, sizeof(s->warn) - 1);
        }
        s->nwarn++;
    }
}
static void cb_info(const char *msg, void *u)
{
    (void)msg;
    (void)u;
}

/* returns the decoded image or NULL; *stage: 0 header failed, 1 decode failed, 2 ok */
static opj_image_t *decode_mem(const unsigned char *buf, size_t len, int reduce,
                               int strict,
                               msgs_t *ms, int *stage)
{
    opj_dparameters_t p;
    opj_codec_t *c;
    opj_stream_t *s;
    opj_image_t *img = NULL;
    mem_t m;
    memset(ms, 0, sizeof(*ms));
    opj_set_default_decoder_parameters(&p);
    p.cp_reduce = (OPJ_UINT32)reduce;
    p.cp_layer = g_layers;
    c = opj_create_decompress(g_jp2 ? OPJ_CODEC_JP2 : OPJ_CODEC_J2K);
    opj_set_error_handler(c, cb_err, ms);
    opj_set_warning_handler(c, cb_warn, ms);
    opj_set_info_handler(c, cb_info, ms);
    opj_setup_decoder(c, &p);
    s = opj_stream_create(len ? len : 1, OPJ_TRUE);
    m.buf = buf;
    m.size = len;
    m.off = 0;
    opj_stream_set_user_data(s, &m, NULL);
    opj_stream_set_user_data_length(s, len);
    opj_stream_set_read_function(s, mem_read);
    opj_stream_set_skip_function(s, mem_skip);
    opj_stream_set_seek_function(s, mem_seek);
    opj_set_decoded_resolution_factor(c, (OPJ_UINT32)reduce);
    opj_decoder_set_strict_mode(c, (strict || g_strict) ? OPJ_TRUE : OPJ_FALSE);
    apply_profile(c);
    *stage = 0;
    if (!opj_read_header(s, c, &img)) {
        opj_image_destroy(img);
        img = NULL;
    } else {
        *stage = 1;
        if (!opj_decode(c, s, img) || !img->numcomps) {
            opj_image_destroy(img);
            img = NULL;
        } else {
            opj_end_decompress(c, s);
            *stage = 2;
        }
    }
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    return img;
}

static unsigned char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    unsigned char *b;
    long n;
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = (unsigned char *)malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(b);
        return NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return b;
}

/* compare two decoded images: returns PSNR in dB over all compared samples (99 = identical),
   -1 if geometry differs. maxdiff receives the largest absolute difference. */
static double compare_images(const opj_image_t *a, const opj_image_t *b,
                             long long *maxdiff,
                             long long *ndiff)
{
    unsigned int c;
    double se = 0.0, n = 0.0, peak = 255.0;
    *maxdiff = 0;
    *ndiff = 0;
    if (a->numcomps != b->numcomps) {
        return -1.0;
    }
    for (c = 0; c < a->numcomps; ++c) {
        size_t i, cnt;
        if (a->comps[c].w != b->comps[c].w || a->comps[c].h != b->comps[c].h) {
            return -1.0;
        }
        if (!a->comps[c].data || !b->comps[c].data) {
            return -1.0;
        }
        peak = (double)((1u << a->comps[c].prec) - 1u);
        cnt = (size_t)a->comps[c].w * a->comps[c].h;
        for (i = 0; i < cnt; ++i) {
            long long d = (long long)a->comps[c].data[i] - (long long)b->comps[c].data[i];
            if (d < 0) {
                d = -d;
            }
            if (d) {
                (*ndiff)++;
            }
            if (d > *maxdiff) {
                *maxdiff = d;
            }
            se += (double)d * (double)d;
        }
        n += (double)cnt;
    }
    if (se == 0.0) {
        return 99.0;
    }
    return 10.0 * log10(peak * peak / (se / n));
}

/* ---------------------------------------------------------------- conformance (15444-4) */
typedef struct {
    const char *in;     /* input codestream */
    int reduce;
    const char
    *ref;    /* reference PGX (component 0) for class 0; prefix for class 1 */
    int ncomp;          /* class 1: number of components compared */
    const char *peak;   /* ':'-separated per component */
    const char *mse;
} conf_t;

/* Class 0 tolerances: tests/conformance/CMakeLists.txt (commented out upstream), 15444-4 Table C.1 / C.4 */
static const conf_t conf_c0[] = {
    {"p0_01.j2k", 0, "c0p0_01.pgx", 1, "0", "0"},
    {"p0_02.j2k", 0, "c0p0_02.pgx", 1, "0", "0"},
    {"p0_03.j2k", 0, "c0p0_03r0.pgx", 1, "0", "0"},
    {"p0_03.j2k", 1, "c0p0_03r1.pgx", 1, "0", "0"},
    {"p0_04.j2k", 3, "c0p0_04.pgx", 1, "33", "55.8"},
    {"p0_05.j2k", 3, "c0p0_05.pgx", 1, "54", "68"},
    {"p0_06.j2k", 3, "c0p0_06.pgx", 1, "109", "743"},
    {"p0_07.j2k", 0, "c0p0_07.pgx", 1, "10", "0.34"},
    {"p0_08.j2k", 5, "c0p0_08.pgx", 1, "7", "6.72"},
    {"p0_09.j2k", 2, "c0p0_09.pgx", 1, "4", "1.47"},
    {"p0_10.j2k", 0, "c0p0_10.pgx", 1, "10", "2.84"},
    {"p0_11.j2k", 0, "c0p0_11.pgx", 1, "0", "0"},
    {"p0_12.j2k", 0, "c0p0_12.pgx", 1, "0", "0"},
    {"p0_13.j2k", 0, "c0p0_13.pgx", 1, "0", "0"},
    {"p0_14.j2k", 2, "c0p0_14.pgx", 1, "0", "0"},
    {"p0_15.j2k", 0, "c0p0_15r0.pgx", 1, "0", "0"},
    {"p0_15.j2k", 1, "c0p0_15r1.pgx", 1, "0", "0"},
    {"p0_16.j2k", 0, "c0p0_16.pgx", 1, "0", "0"},
    {"p1_01.j2k", 0, "c0p1_01.pgx", 1, "0", "0"},
    {"p1_02.j2k", 3, "c0p1_02.pgx", 1, "35", "74"},
    {"p1_03.j2k", 3, "c0p1_03.pgx", 1, "28", "18.8"},
    {"p1_04.j2k", 0, "c0p1_04r0.pgx", 1, "2", "0.550"},
    {"p1_04.j2k", 3, "c0p1_04r3.pgx", 1, "2", "0.550"},
    {"p1_05.j2k", 4, "c0p1_05.pgx", 1, "128", "16384"},
    {"p1_06.j2k", 1, "c0p1_06.pgx", 1, "128", "16384"},
    {"p1_07.j2k", 0, "c0p1_07.pgx", 1, "0", "0"},
};
/* Class 1: active upstream tests (C1P0, C1P1) */
static const conf_t conf_c1[] = {
    {"p0_01.j2k", 0, "c1p0_01", 1, "0", "0"},
    {"p0_02.j2k", 0, "c1p0_02", 1, "0", "0"},
    {"p0_03.j2k", 0, "c1p0_03", 1, "0", "0"},
    {"p0_04.j2k", 0, "c1p0_04", 3, "5:4:6", "0.776:0.626:1.070"},
    {"p0_05.j2k", 0, "c1p0_05", 4, "2:2:2:0", "0.302:0.307:0.269:0"},
    {"p0_06.j2k", 0, "c1p0_06", 4, "635:403:378:0", "11287:6124:3968:0"},
    {"p0_07.j2k", 0, "c1p0_07", 3, "0:0:0", "0:0:0"},
    {"p0_08.j2k", 1, "c1p0_08", 3, "0:0:0", "0:0:0"},
    {"p0_09.j2k", 0, "c1p0_09", 1, "0", "0"},
    {"p0_10.j2k", 0, "c1p0_10", 3, "0:0:0", "0:0:0"},
    {"p0_11.j2k", 0, "c1p0_11", 1, "0", "0"},
    {"p0_12.j2k", 0, "c1p0_12", 1, "0", "0"},
    {"p0_13.j2k", 0, "c1p0_13", 4, "0:0:0:0", "0:0:0:0"},
    {"p0_14.j2k", 0, "c1p0_14", 3, "0:0:0", "0:0:0"},
    {"p0_15.j2k", 0, "c1p0_15", 1, "0", "0"},
    {"p0_16.j2k", 0, "c1p0_16", 1, "0", "0"},
    {"p1_01.j2k", 0, "c1p1_01", 1, "0", "0"},
    {"p1_02.j2k", 0, "c1p1_02", 3, "5:4:6", "0.765:0.616:1.051"},
    {"p1_03.j2k", 0, "c1p1_03", 4, "2:2:1:0", "0.3:0.210:0.200:0"},
    {"p1_04.j2k", 0, "c1p1_04", 1, "624", "3080"},
    {"p1_05.j2k", 0, "c1p1_05", 3, "40:40:40", "8.458:9.816:10.154"},
    {"p1_06.j2k", 0, "c1p1_06", 3, "2:2:2", "0.6:0.6:0.6"},
    {"p1_07.j2k", 0, "c1p1_07", 2, "0:0", "0:0"},
};

typedef struct {
    int w, h, sgnd, prec;
    int *d;
} pgx_t;

static int load_pgx(const char *path, pgx_t *p)
{
    size_t len, i, hdr;
    unsigned char *b = read_file(path, &len);
    char e1, e2, sg = '+';
    int bytes;
    const char *q;
    char *end;
    if (!b || len < 8 || b[0] != 'P' || b[1] != 'G') {
        free(b);
        return 0;
    }
    /* "PG ML [+|-] depth w h\n": the sign is optional and may touch the depth */
    q = (const char *)b + 2;
    while (*q == ' ') {
        ++q;
    }
    e1 = q[0];
    e2 = q[1];
    (void)e2;
    q += 2;
    while (*q == ' ') {
        ++q;
    }
    if (*q == '+' || *q == '-') {
        sg = *q++;
    }
    p->prec = (int)strtol(q, &end, 10);
    p->w = (int)strtol(end, &end, 10);
    p->h = (int)strtol(end, &end, 10);
    if (p->prec <= 0 || p->prec > 32 || p->w <= 0 || p->h <= 0) {
        free(b);
        return 0;
    }
    hdr = (size_t)((const unsigned char *)end - b) +
          1; /* one whitespace after the height */
    p->sgnd = (sg == '-');
    bytes = p->prec <= 8 ? 1 : (p->prec <= 16 ? 2 : 4);
    p->d = (int *)malloc(sizeof(int) * (size_t)p->w * p->h);
    for (i = 0; i < (size_t)p->w * p->h; ++i) {
        const unsigned char *q = b + hdr + i * (size_t)bytes;
        unsigned int v = 0;
        int k;
        if (hdr + (i + 1) * (size_t)bytes > len) {
            free(b);
            return 0;
        }
        for (k = 0; k < bytes; ++k) {
            v = (e1 == 'M') ? ((v << 8) | q[k]) : (v | ((unsigned int)q[k] << (8 * k)));
        }
        if (p->sgnd) {
            unsigned int sbit = 1u << (bytes * 8 - 1);
            p->d[i] = (v & sbit) ? (int)(v - (sbit << 1)) : (int)v;
        } else {
            p->d[i] = (int)v;
        }
    }
    free(b);
    return 1;
}

static double nth_tol(const char *list, int k)
{
    const char *q = list;
    while (k-- > 0 && q) {
        q = strchr(q, ':');
        if (q) {
            ++q;
        }
    }
    return q ? atof(q) : 0.0;
}

static int conf_one(const char *dir, const conf_t *t, int cls)
{
    char path[1024];
    size_t len;
    unsigned char *buf;
    msgs_t ms;
    int stage, c, ok = 1;
    opj_image_t *img;
    snprintf(path, sizeof(path), "%s/input/conformance/%s", dir, t->in);
    buf = read_file(path, &len);
    if (!buf) {
        printf("class%d %s r%d: missing input\n", cls, t->in, t->reduce);
        return 0;
    }
    int r = t->reduce, shift;
    img = decode_mem(buf, len, r, 0, &ms, &stage);
    if (!img) {
        printf("class%d %s r%d: DECODE FAILED (stage %d): %s", cls, t->in, r, stage,
               ms.err);
        free(buf);
        return 0;
    }
    for (c = 0; c < t->ncomp; ++c) {
        pgx_t p;
        size_t i, n;
        long long peak = 0;
        double se = 0.0, mse;
        if (cls == 0) {
            snprintf(path, sizeof(path), "%s/baseline/conformance/%s", dir, t->ref);
        } else {
            snprintf(path, sizeof(path), "%s/baseline/conformance/%s_%d.pgx", dir, t->ref,
                     c);
        }
        if (!load_pgx(path, &p)) {
            printf("class%d %s r%d c%d: missing or bad reference %s\n", cls, t->in, r, c,
                   path);
            ok = 0;
            continue;
        }
        /* class 0: the reference may be at another reduction than the CMake list says;
           take the reduction whose size matches the reference (reported) */
        while (cls == 0 && r < 8 && img && ((int)img->comps[0].w != p.w ||
                                            (int)img->comps[0].h != p.h)) {
            opj_image_destroy(img);
            ++r;
            img = decode_mem(buf, len, r, 0, &ms, &stage);
        }
        if (!img) {
            printf("class%d %s: no reduction gives the reference size %dx%d\n", cls, t->in,
                   p.w, p.h);
            free(p.d);
            free(buf);
            return 0;
        }
        /* class 0 references can have fewer bits than the codestream: compare at the reference precision */
        shift = (cls == 0 &&
                 (int)img->comps[c].prec > p.prec) ? (int)img->comps[c].prec - p.prec : 0;
        if ((unsigned)c >= img->numcomps || (int)img->comps[c].w != p.w ||
                (int)img->comps[c].h != p.h) {
            printf("class%d %s r%d c%d: SIZE MISMATCH decoded %ux%u reference %dx%d\n", cls,
                   t->in, t->reduce,
                   c, (unsigned)c < img->numcomps ? img->comps[c].w : 0,
                   (unsigned)c < img->numcomps ? img->comps[c].h : 0,
                   p.w, p.h);
            free(p.d);
            ok = 0;
            continue;
        }
        n = (size_t)p.w * p.h;
        for (i = 0; i < n; ++i) {
            long long d = (long long)(img->comps[c].data[i] >> shift) - p.d[i];
            if (d < 0) {
                d = -d;
            }
            if (d > peak) {
                peak = d;
            }
            se += (double)d * d;
        }
        mse = se / (double)n;
        {
            double pl = nth_tol(t->peak, c), ml = nth_tol(t->mse, c);
            int pass = (double)peak <= pl && mse <= ml + 1e-9;
            printf("class%d %s r%d%s c%d: %ux%u%s peak %lld (limit %g) mse %.4f (limit %g) %s\n",
                   cls, t->in,
                   r, r != t->reduce ? " (size-matched)" : "", c, img->comps[c].w, img->comps[c].h,
                   shift ? " (scaled to the reference precision)" : "", peak, pl, mse, ml,
                   pass ? "PASS" : "FAIL");
            if (!pass) {
                ok = 0;
            }
        }
        free(p.d);
    }
    opj_image_destroy(img);
    free(buf);
    return ok;
}

static int cmd_conf(const char *dir)
{
    size_t i;
    int pass0 = 0, pass1 = 0;
    for (i = 0; i < sizeof(conf_c0) / sizeof(conf_c0[0]); ++i) {
        pass0 += conf_one(dir, &conf_c0[i], 0);
    }
    for (i = 0; i < sizeof(conf_c1) / sizeof(conf_c1[0]); ++i) {
        pass1 += conf_one(dir, &conf_c1[i], 1);
    }
    printf("SUMMARY class0 %d/%d pass, class1 %d/%d pass\n", pass0,
           (int)(sizeof(conf_c0) / sizeof(conf_c0[0])), pass1,
           (int)(sizeof(conf_c1) / sizeof(conf_c1[0])));
    return 0;
}

/* ---------------------------------------------------------------- truncation sweep */
static const char *classify(const msgs_t *ms)
{
    if (strstr(ms->err, "EPH")) {
        return "EPH";
    }
    if (strstr(ms->err, "Stream too short")) {
        return "short";
    }
    if (strstr(ms->err, "Failed to decode")) {
        return "tile";
    }
    if (ms->nerr) {
        return "other";
    }
    return "-";
}

static int cmd_cut(const char *file, const char *reduces, int step,
                   const char *csv)
{
    size_t len, L;
    unsigned char *buf = read_file(file, &len);
    FILE *out;
    const char *q;
    if (!buf) {
        fprintf(stderr, "cannot read %s\n", file);
        return 1;
    }
    out = fopen(csv, "w");
    fprintf(out, "file,len,reduce,cut,stage,psnr,maxdiff,ndiff,nwarn,errclass\n");
    for (q = reduces; *q;) {
        int r = atoi(q), stage;
        msgs_t ms;
        opj_image_t *ref = decode_mem(buf, len, r, 1, &ms, &stage);
        double best = -1.0, worst_drop = 0.0;
        long long fails = 0, drops = 0, total = 0;
        size_t worst_at = 0;
        if (!ref) {
            printf("%s r%d: whole file does not decode (stage %d): %s", file, r, stage,
                   ms.err);
        } else {
            for (L = 16; L <= len; L += (size_t)step) {
                long long md = 0, nd = 0;
                double ps = -2.0;
                opj_image_t *img = decode_mem(buf, L, r, 0, &ms, &stage);
                total++;
                if (img) {
                    ps = compare_images(ref, img, &md, &nd);
                    opj_image_destroy(img);
                    if (best >= 0.0 && ps < best - 1.0) {
                        drops++;
                        if (best - ps > worst_drop) {
                            worst_drop = best - ps;
                            worst_at = L;
                        }
                    }
                    if (ps > best) {
                        best = ps;
                    }
                } else if (stage == 1) {
                    fails++;
                }
                fprintf(out, "%s,%zu,%d,%zu,%d,%.3f,%lld,%lld,%d,%s\n", file, len, r, L, stage,
                        ps, md, nd,
                        ms.nwarn, classify(&ms));
            }
            printf("%s r%d: %lld prefixes, %lld decode failures after a good header, %lld prefixes >1 dB worse "
                   "than a shorter one (worst %.1f dB at %zu of %zu)\n",
                   file, r, total, fails, drops, worst_drop, worst_at, len);
            opj_image_destroy(ref);
        }
        while (*q && *q != ',') {
            ++q;
        }
        if (*q == ',') {
            ++q;
        }
    }
    fclose(out);
    free(buf);
    return 0;
}

static int cmd_one(const char *file, int r, size_t L)
{
    size_t len;
    unsigned char *buf = read_file(file, &len);
    msgs_t ms;
    int stage;
    opj_image_t *img, *ref;
    long long md = 0, nd = 0;
    if (!buf) {
        return 1;
    }
    if (L == 0 || L > len) {
        L = len;
    }
    ref = decode_mem(buf, len, r, 1, &ms, &stage);
    img = decode_mem(buf, L, r, 0, &ms, &stage);
    printf("%s r%d cut %zu/%zu: stage %d warnings %d errors %d first error: %s",
           file, r, L, len, stage,
           ms.nwarn, ms.nerr, ms.nerr ? ms.err : "-\n");
    if (ms.nwarn) {
        printf("  first warning: %s", ms.warn);
    }
    if (img && ref) {
        double ps = compare_images(ref, img, &md, &nd);
        printf("  %ux%u x%u, PSNR against the whole file %.2f dB, max diff %lld, %lld samples differ\n",
               img->comps[0].w, img->comps[0].h, img->numcomps, ps, md, nd);
    }
    opj_image_destroy(img);
    opj_image_destroy(ref);
    free(buf);
    return 0;
}

/* ---------------------------------------------------------------- synthetic header cost */
static void put16(unsigned char **p, unsigned v)
{
    (*p)[0] = (unsigned char)(v >> 8);
    (*p)[1] = (unsigned char)v;
    *p += 2;
}
static void put32(unsigned char **p, unsigned v)
{
    put16(p, v >> 16);
    put16(p, v & 0xffff);
}

static int cmd_hdr(unsigned W, unsigned H, unsigned tdx, unsigned tdy,
                   unsigned C, unsigned layers)
{
    unsigned char *b = (unsigned char *)calloc(1, 1 << 16), *p = b;
    unsigned i;
    PROCESS_MEMORY_COUNTERS_EX pm0, pm1;
    LARGE_INTEGER f, t0, t1, t2;
    msgs_t ms;
    int stage;
    opj_image_t *img;
    put16(&p, 0xFF4F);                 /* SOC */
    put16(&p, 0xFF51);                 /* SIZ */
    put16(&p, 38 + 3 * C);
    put16(&p, 0);                      /* Rsiz */
    put32(&p, W);
    put32(&p, H);
    put32(&p, 0);
    put32(&p, 0);
    put32(&p, tdx);
    put32(&p, tdy);
    put32(&p, 0);
    put32(&p, 0);
    put16(&p, C);
    for (i = 0; i < C; ++i) {
        *p++ = 7;                      /* 8-bit unsigned */
        *p++ = 1;
        *p++ = 1;
    }
    put16(&p, 0xFF52);                 /* COD: LRCP, layers, no MCT, 5 levels, 64x64, 5/3 */
    put16(&p, 12);
    *p++ = 0;
    *p++ = 0;
    put16(&p, layers);
    *p++ = 0;
    *p++ = 5;
    *p++ = 4;
    *p++ = 4;
    *p++ = 0;
    *p++ = 1;
    put16(&p, 0xFF5C);                 /* QCD: no quantisation, 2 guard bits, 16 bands */
    put16(&p, 3 + 16);
    *p++ = 0x40;
    for (i = 0; i < 16; ++i) {
        *p++ = (unsigned char)(9 << 3);
    }
    put16(&p, 0xFF90);                 /* SOT tile 0, Psot 0 (last), TPsot 0, TNsot 1 */
    put16(&p, 10);
    put16(&p, 0);
    put32(&p, 0);
    *p++ = 0;
    *p++ = 1;
    put16(&p, 0xFF93);                 /* SOD, no data */
    put16(&p, 0xFFD9);                 /* EOC */
    QueryPerformanceFrequency(&f);
    pm0.cb = sizeof(pm0);
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pm0,
                         sizeof(pm0));
    QueryPerformanceCounter(&t0);
    {
        /* header only, as the viewer's metadata pass */
        opj_dparameters_t prm;
        opj_codec_t *c = opj_create_decompress(OPJ_CODEC_J2K);
        opj_stream_t *s = opj_stream_create((size_t)(p - b), OPJ_TRUE);
        mem_t m;
        memset(&ms, 0, sizeof(ms));
        opj_set_default_decoder_parameters(&prm);
        opj_set_error_handler(c, cb_err, &ms);
        opj_set_warning_handler(c, cb_warn, &ms);
        opj_setup_decoder(c, &prm);
        m.buf = b;
        m.size = (size_t)(p - b);
        m.off = 0;
        opj_stream_set_user_data(s, &m, NULL);
        opj_stream_set_user_data_length(s, m.size);
        opj_stream_set_read_function(s, mem_read);
        opj_stream_set_skip_function(s, mem_skip);
        opj_stream_set_seek_function(s, mem_seek);
        opj_decoder_set_strict_mode(c, OPJ_FALSE);
        apply_profile(c);
        img = NULL;
        stage = opj_read_header(s, c, &img);
        QueryPerformanceCounter(&t1);
        pm1.cb = sizeof(pm1);
        GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pm1,
                             sizeof(pm1));
        printf("codestream %d bytes: %ux%u, tiles %ux%u = %u, %u comps, %u layers\n",
               (int)(p - b), W, H,
               (W + tdx - 1) / tdx, (H + tdy - 1) / tdy,
               ((W + tdx - 1) / tdx) * ((H + tdy - 1) / tdy), C, layers);
        printf("read_header: %s in %.1f ms, peak commit +%.1f MB (%s)\n",
               stage ? "ok" : "FAILED",
               1000.0 * (double)(t1.QuadPart - t0.QuadPart) / (double)f.QuadPart,
               (double)(pm1.PeakPagefileUsage - pm0.PeakPagefileUsage) / 1048576.0,
               ms.nerr ? ms.err : "no error\n");
        opj_image_destroy(img);
        opj_stream_destroy(s);
        opj_destroy_codec(c);
        QueryPerformanceCounter(&t2);
        printf("destroy: %.1f ms\n",
               1000.0 * (double)(t2.QuadPart - t1.QuadPart) / (double)f.QuadPart);
    }
    free(b);
    return 0;
}

/* ---------------------------------------------------------------- allocation census */
#ifdef OPJ_ALLOC_CENSUS
/* the library's own allocators (opj_malloc.c is compiled without the census macros) */
void *opj_malloc(size_t size);
void *opj_calloc(size_t n, size_t size);
void *opj_realloc(void *p, size_t size);
void *opj_aligned_malloc(size_t size);
void *opj_aligned_32_malloc(size_t size);
void *opj_aligned_realloc(void *p, size_t size);
#define opj_malloc_real opj_malloc
#define opj_calloc_real opj_calloc
#define opj_realloc_real opj_realloc
#define opj_aligned_malloc_real opj_aligned_malloc
#define opj_aligned_32_malloc_real opj_aligned_32_malloc
#define opj_aligned_realloc_real opj_aligned_realloc
typedef struct {
    const char *file;
    int line;
    long long n;
    long long bytes;
} site_t;
static site_t g_sites[4096];
static int g_nsites, g_on;
static void census_add(const char *file, int line, size_t bytes)
{
    int i;
    const char *base = strrchr(file, '\\');
    if (!g_on) {
        return;
    }
    base = base ? base + 1 : file;
    if (strrchr(base, '/')) {
        base = strrchr(base, '/') + 1;
    }
    for (i = 0; i < g_nsites; ++i) {
        if (g_sites[i].line == line && strcmp(g_sites[i].file, base) == 0) {
            g_sites[i].n++;
            g_sites[i].bytes += (long long)bytes;
            return;
        }
    }
    if (g_nsites < 4096) {
        g_sites[g_nsites].file = base;
        g_sites[g_nsites].line = line;
        g_sites[g_nsites].n = 1;
        g_sites[g_nsites].bytes = (long long)bytes;
        g_nsites++;
    }
}
void *opj_census_malloc(size_t s, const char *f, int l)
{
    census_add(f, l, s);
    return opj_malloc_real(s);
}
void *opj_census_calloc(size_t n, size_t s, const char *f, int l)
{
    census_add(f, l, n * s);
    return opj_calloc_real(n, s);
}
void *opj_census_realloc(void *p, size_t s, const char *f, int l)
{
    census_add(f, l, s);
    return opj_realloc_real(p, s);
}
void *opj_census_aligned_malloc(size_t s, const char *f, int l)
{
    census_add(f, l, s);
    return opj_aligned_malloc_real(s);
}
void *opj_census_aligned_32_malloc(size_t s, const char *f, int l)
{
    census_add(f, l, s);
    return opj_aligned_32_malloc_real(s);
}
void *opj_census_aligned_realloc(void *p, size_t s, const char *f, int l)
{
    census_add(f, l, s);
    return opj_aligned_realloc_real(p, s);
}
static int cmp_site(const void *a, const void *b)
{
    const site_t *x = (const site_t *)a, *y = (const site_t *)b;
    return (y->n > x->n) - (y->n < x->n);
}
static int cmd_census(const char *file, int r, int repeat)
{
    size_t len;
    unsigned char *buf = read_file(file, &len);
    msgs_t ms;
    int stage, i;
    long long total = 0, bytes = 0;
    opj_image_t *img;
    if (!buf) {
        return 1;
    }
    g_on = 1;
    for (i = 0; i < repeat; ++i) {
        img = decode_mem(buf, len, r, 0, &ms, &stage);
        opj_image_destroy(img);
    }
    g_on = 0;
    qsort(g_sites, (size_t)g_nsites, sizeof(site_t), cmp_site);
    for (i = 0; i < g_nsites; ++i) {
        total += g_sites[i].n;
        bytes += g_sites[i].bytes;
    }
    printf("%s r%d stage %d: %lld allocation calls, %lld bytes per decode, %d call sites\n",
           file, r, stage,
           total / repeat, bytes / repeat, g_nsites);
    for (i = 0; i < g_nsites; ++i) {
        printf("  %6.1f  %8lld B  %s:%d\n", (double)g_sites[i].n / repeat,
               g_sites[i].bytes / repeat,
               g_sites[i].file, g_sites[i].line);
    }
    free(buf);
    return 0;
}
#endif


/* ---------------------------------------------------------------- output hashes (bit-identity checks) */
static int cmd_hash(int argc, char **argv)
{
    int a;
    for (a = 3; a < argc; ++a) {
        size_t len;
        unsigned char *buf = read_file(argv[a], &len);
        const char *q;
        if (!buf) {
            printf("%s: unreadable\n", argv[a]);
            continue;
        }
        for (q = argv[2]; *q;) {
            int r = atoi(q), stage;
            msgs_t ms;
            opj_image_t *img = decode_mem(buf, len, r, 0, &ms, &stage);
            unsigned long long h = 1469598103934665603ULL;
            if (img) {
                unsigned int c;
                for (c = 0; c < img->numcomps; ++c) {
                    size_t i, n = (size_t)img->comps[c].w * img->comps[c].h;
                    h ^= img->comps[c].w;
                    h *= 1099511628211ULL;
                    h ^= img->comps[c].h;
                    h *= 1099511628211ULL;
                    for (i = 0; i < n; ++i) {
                        h ^= (unsigned int)img->comps[c].data[i];
                        h *= 1099511628211ULL;
                    }
                }
                printf("%s r%d %ux%ux%u %016llx\n", argv[a], r, img->comps[0].w,
                       img->comps[0].h, img->numcomps, h);
                opj_image_destroy(img);
            } else {
                printf("%s r%d FAILED stage %d\n", argv[a], r, stage);
            }
            while (*q && *q != ',') {
                ++q;
            }
            if (*q == ',') {
                ++q;
            }
        }
        free(buf);
    }
    return 0;
}

/* ---------------------------------------------------------------- packet-loop cost and the reduce setter */
/* A small image whose COD declares many layers and whose data is one empty packet: after the data ends,
   the decoder still visits every packet. scod 0 = no SOP/EPH, 2 = SOP allowed (warning per packet). */
static int cmd_loops(unsigned W, unsigned layers, unsigned scod, unsigned pp)
{
    unsigned char *b = (unsigned char *)calloc(1, 4096), *p = b;
    unsigned i;
    LARGE_INTEGER f, t0, t1;
    msgs_t ms;
    int stage;
    opj_image_t *img;
    put16(&p, 0xFF4F);
    put16(&p, 0xFF51);
    put16(&p, 38 + 3 * 3);
    put16(&p, 0);
    put32(&p, W);
    put32(&p, W);
    put32(&p, 0);
    put32(&p, 0);
    put32(&p, W);
    put32(&p, W);
    put32(&p, 0);
    put32(&p, 0);
    put16(&p, 3);
    for (i = 0; i < 3; ++i) {
        *p++ = 7;
        *p++ = 1;
        *p++ = 1;
    }
    put16(&p, 0xFF52);
    put16(&p, 12 + (pp ? 6 : 0));
    *p++ = (unsigned char)(scod | (pp ? 1 : 0));
    *p++ = 0;                          /* LRCP */
    put16(&p, layers);
    *p++ = 0;
    *p++ = 5;
    *p++ = 4;
    *p++ = 4;
    *p++ = 0;
    *p++ = 1;
    if (pp) {                          /* user precincts: 2^pp x 2^pp at every resolution */
        for (i = 0; i < 6; ++i) {
            *p++ = (unsigned char)(pp | (pp << 4));
        }
    }
    put16(&p, 0xFF5C);
    put16(&p, 3 + 16);
    *p++ = 0x40;
    for (i = 0; i < 16; ++i) {
        *p++ = (unsigned char)(9 << 3);
    }
    put16(&p, 0xFF90);
    put16(&p, 10);
    put16(&p, 0);
    put32(&p, 12 + 2 + 1);             /* Psot: SOT, SOD and one data byte */
    *p++ = 0;
    *p++ = 1;
    put16(&p, 0xFF93);
    *p++ = 0x00;                       /* one empty packet header, then the tile data ends */
    put16(&p, 0xFFD9);
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    img = decode_mem(b, (size_t)(p - b), 0, 0, &ms, &stage);
    QueryPerformanceCounter(&t1);
    printf("codestream %d bytes, %ux%u x3, %u layers, Scod 0x%02x: stage %d, %d warnings, %.1f ms\n",
           (int)(p - b), W, W, layers, scod, stage, ms.nwarn,
           1000.0 * (double)(t1.QuadPart - t0.QuadPart) / (double)f.QuadPart);
    if (ms.nerr) {
        printf("  first error: %s", ms.err);
    }
    if (ms.nwarn) {
        printf("  first warning: %s", ms.warn);
    }
    opj_image_destroy(img);
    free(b);
    return 0;
}

/* opj_read_header, then opj_set_decoded_resolution_factor(factor) without checking its result
   (the order a header-once caller would use), then opj_decode. */
static int cmd_setfactor(const char *file, unsigned factor)
{
    size_t len;
    unsigned char *buf = read_file(file, &len);
    opj_dparameters_t prm;
    opj_codec_t *c;
    opj_stream_t *s;
    opj_image_t *img = NULL;
    mem_t m;
    msgs_t ms;
    OPJ_BOOL set_ok, dec_ok = OPJ_FALSE;
    if (!buf) {
        return 1;
    }
    memset(&ms, 0, sizeof(ms));
    opj_set_default_decoder_parameters(&prm);
    c = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(c, cb_err, &ms);
    opj_set_warning_handler(c, cb_warn, &ms);
    opj_setup_decoder(c, &prm);
    s = opj_stream_create(len, OPJ_TRUE);
    m.buf = buf;
    m.size = len;
    m.off = 0;
    opj_stream_set_user_data(s, &m, NULL);
    opj_stream_set_user_data_length(s, len);
    opj_stream_set_read_function(s, mem_read);
    opj_stream_set_skip_function(s, mem_skip);
    opj_stream_set_seek_function(s, mem_seek);
    opj_decoder_set_strict_mode(c, OPJ_FALSE);
    apply_profile(c);
    if (opj_read_header(s, c, &img)) {
        set_ok = opj_set_decoded_resolution_factor(c, factor);
        printf("set_decoded_resolution_factor(%u) returned %d; decoding anyway\n",
               factor, set_ok);
        fflush(stdout);
        dec_ok = opj_decode(c, s, img);
        printf("opj_decode returned %d (%s)\n", dec_ok,
               ms.nerr ? ms.err : "no error\n");
    }
    opj_image_destroy(img);
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    free(buf);
    return 0;
}

/* ---------------------------------------------------------------- per-prefix output hashes */
static unsigned long long image_hash(const opj_image_t *img)
{
    unsigned long long h = 1469598103934665603ULL;
    unsigned int c;
    h ^= img->numcomps;
    h *= 1099511628211ULL;
    for (c = 0; c < img->numcomps; ++c) {
        size_t i, n = (size_t)img->comps[c].w * img->comps[c].h;
        h ^= img->comps[c].w;
        h *= 1099511628211ULL;
        h ^= img->comps[c].h;
        h *= 1099511628211ULL;
        h ^= (unsigned long long)img->comps[c].prec | ((unsigned long long)
                img->comps[c].sgnd << 8) |
             ((unsigned long long)img->comps[c].factor << 16) |
             ((unsigned long long)img->comps[c].resno_decoded << 24);
        h *= 1099511628211ULL;
        if (!img->comps[c].data) {
            continue;
        }
        for (i = 0; i < n; ++i) {
            h ^= (unsigned int)img->comps[c].data[i];
            h *= 1099511628211ULL;
        }
    }
    return h;
}

/* every prefix length from 2 up to the whole file (step), at each reduce: "r len stage hash" per line.
   stage 0 = header failed, 1 = decode failed, 2 = decoded (hash of the image, its geometry included). */
static int cmd_cuthash(const char *file, const char *reduces, int step,
                       const char *outp)
{
    size_t len, L;
    unsigned char *buf = read_file(file, &len);
    FILE *out;
    const char *q;
    long long n = 0, ok = 0;
    if (!buf) {
        fprintf(stderr, "cannot read %s\n", file);
        return 1;
    }
    out = fopen(outp, "w");
    if (!out) {
        free(buf);
        return 1;
    }
    for (q = reduces; *q;) {
        int r = atoi(q);
        for (L = 2;; L += (size_t)step) {
            msgs_t ms;
            int stage;
            opj_image_t *img;
            if (L > len) {
                L = len;
            }
            img = decode_mem(buf, L, r, 0, &ms, &stage);
            ++n;
            if (img) {
                ++ok;
                fprintf(out, "%d %zu 2 %016llx\n", r, L, image_hash(img));
                opj_image_destroy(img);
            } else {
                fprintf(out, "%d %zu %d -\n", r, L, stage);
            }
            if (L == len) {
                break;
            }
        }
        while (*q && *q != ',') {
            ++q;
        }
        if (*q == ',') {
            ++q;
        }
    }
    fclose(out);
    free(buf);
    printf("%s: %lld decodes, %lld images\n", file, n, ok);
    return 0;
}

/* whole-file decodes limited to 1..n layers (cp_layer), one image_hash per (reduce, layers): the B5 oracle */
static int cmd_layhash(const char *file, const char *reduces, unsigned nlayers)
{
    size_t len;
    unsigned char *buf = read_file(file, &len);
    const char *q;
    if (!buf) {
        fprintf(stderr, "cannot read %s\n", file);
        return 1;
    }
    for (q = reduces; *q;) {
        int r = atoi(q);
        unsigned k;
        for (k = 1; k <= nlayers; ++k) {
            msgs_t ms;
            int stage;
            opj_image_t *img;
            g_layers = k;
            img = decode_mem(buf, len, r, 0, &ms, &stage);
            if (img) {
                printf("%d L%u 2 %016llx\n", r, k, image_hash(img));
                opj_image_destroy(img);
            } else {
                printf("%d L%u %d -\n", r, k, stage);
            }
        }
        while (*q && *q != ',') {
            ++q;
        }
        if (*q == ',') {
            ++q;
        }
    }
    g_layers = 0;
    free(buf);
    return 0;
}

/* ---------------------------------------------------------------- one decode: memory and time */
static int cmd_mem(const char *file, int r, size_t L)
{
    size_t len;
    unsigned char *buf = read_file(file, &len);
    msgs_t ms;
    int stage;
    opj_image_t *img;
    PROCESS_MEMORY_COUNTERS_EX pm0, pm1;
    LARGE_INTEGER f, t0, t1;
    if (!buf) {
        return 1;
    }
    if (L == 0 || L > len) {
        L = len;
    }
    pm0.cb = sizeof(pm0);
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pm0,
                         sizeof(pm0));
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    img = decode_mem(buf, L, r, 0, &ms, &stage);
    QueryPerformanceCounter(&t1);
    pm1.cb = sizeof(pm1);
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pm1,
                         sizeof(pm1));
    printf("%s r%d len %zu/%zu: stage %d, %d warnings, %d errors, %.1f ms, peak commit +%.1f MB, peak working set +%.1f MB",
           file, r, L, len, stage, ms.nwarn, ms.nerr,
           1000.0 * (double)(t1.QuadPart - t0.QuadPart) / (double)f.QuadPart,
           (double)(pm1.PeakPagefileUsage - pm0.PeakPagefileUsage) / 1048576.0,
           (double)(pm1.PeakWorkingSetSize - pm0.PeakWorkingSetSize) / 1048576.0);
    if (img) {
        printf(", image %ux%u x%u\n", img->comps[0].w, img->comps[0].h, img->numcomps);
    } else {
        printf("\n");
    }
    if (ms.nerr) {
        printf("  first error: %s", ms.err);
    }
    if (ms.nwarn) {
        printf("  first warning: %s", ms.warn);
    }
    opj_image_destroy(img);
    free(buf);
    return 0;
}

/* ---------------------------------------------------------------- texture cache check (read only, aggregates only) */
/* The viewer's header cache (lltexturecache.cpp): texture.entries holds a 44-byte header (version 1.71,
   address 64, encoder name, entry count) and 28-byte entries (16-byte id, image size, body size, time);
   texture.cache holds the first 600 bytes of entry i at i * 600. Every J2C prefix is decoded at reduce r
   with the current profile; only counts are printed. */
static int cmd_cache(const char *entries_path, const char *cache_path, int r)
{
    size_t elen, clen, i;
    unsigned char *e = read_file(entries_path, &elen);
    unsigned char *c = read_file(cache_path, &clen);
    unsigned int count;
    long long parsed = 0, hdr_ok = 0, dec_ok = 0, hdr_limit = 0, dec_limit = 0,
              hdr_other = 0, dec_other = 0;
    unsigned int max_w = 0, max_h = 0, max_c = 0, max_layers = 0, max_tiles = 0;
    char first_other[256] = "";
    if (!e || !c || elen < 44) {
        fprintf(stderr, "cannot read the cache\n");
        return 1;
    }
    memcpy(&count, e + 40, 4);
    if (elen != 44 + (size_t)28 * count) {
        fprintf(stderr, "unrecognised entries file\n");
        return 1;
    }
    for (i = 0; i < count; ++i) {
        int image_size;
        size_t len;
        const unsigned char *p;
        msgs_t ms;
        int stage;
        opj_image_t *img;
        memcpy(&image_size, e + 44 + 28 * i + 16, 4);
        if (image_size <= 0 || (i + 1) * 600 > clen) {
            continue;
        }
        p = c + i * 600;
        len = image_size < 600 ? (size_t)image_size : 600;
        if (len < 4 || p[0] != 0xFF || p[1] != 0x4F || p[2] != 0xFF || p[3] != 0x51) {
            continue;
        }
        ++parsed;
        /* the header facts, without limits */
        {
            opj_dparameters_t prm;
            opj_codec_t *cd = opj_create_decompress(OPJ_CODEC_J2K);
            opj_stream_t *s = opj_stream_create(len, OPJ_TRUE);
            opj_image_t *hi = NULL;
            mem_t m;
            opj_set_default_decoder_parameters(&prm);
            opj_set_error_handler(cd, cb_info, NULL);
            opj_set_warning_handler(cd, cb_info, NULL);
            opj_set_info_handler(cd, cb_info, NULL);
            opj_setup_decoder(cd, &prm);
            m.buf = p;
            m.size = len;
            m.off = 0;
            opj_stream_set_user_data(s, &m, NULL);
            opj_stream_set_user_data_length(s, len);
            opj_stream_set_read_function(s, mem_read);
            opj_stream_set_skip_function(s, mem_skip);
            opj_stream_set_seek_function(s, mem_seek);
            opj_decoder_set_strict_mode(cd, OPJ_FALSE);
            if (opj_read_header(s, cd, &hi)) {
                opj_codestream_info_v2_t *ci = opj_get_cstr_info(cd);
                if (hi->x1 - hi->x0 > max_w) {
                    max_w = hi->x1 - hi->x0;
                }
                if (hi->y1 - hi->y0 > max_h) {
                    max_h = hi->y1 - hi->y0;
                }
                if (hi->numcomps > max_c) {
                    max_c = hi->numcomps;
                }
                if (ci) {
                    if (ci->m_default_tile_info.numlayers > max_layers) {
                        max_layers = ci->m_default_tile_info.numlayers;
                    }
                    if (ci->tw * ci->th > max_tiles) {
                        max_tiles = ci->tw * ci->th;
                    }
                    opj_destroy_cstr_info(&ci);
                }
            }
            opj_image_destroy(hi);
            opj_stream_destroy(s);
            opj_destroy_codec(cd);
        }
        /* the decode the viewer would run, with the current profile */
        img = decode_mem(p, len, r, 0, &ms, &stage);
        if (stage == 0) {
            if (strstr(ms.err, "limit")) {
                ++hdr_limit;
            } else {
                ++hdr_other;
                if (!first_other[0]) {
                    strncpy(first_other, ms.err, sizeof(first_other) - 1);
                }
            }
        } else {
            ++hdr_ok;
            if (img) {
                ++dec_ok;
            } else if (strstr(ms.err, "limit")) {
                ++dec_limit;
            } else {
                ++dec_other;
            }
        }
        opj_image_destroy(img);
    }
    printf("entries %u, J2C headers %lld (largest %u x %u, at most %u components, %u layers, %u tiles)\n",
           count, parsed, max_w, max_h, max_c, max_layers, max_tiles);
    printf("reduce %d, profile %s: header accepted %lld, refused by a limit %lld, other header failures %lld;"
           " 600-byte prefix decoded %lld, decode refused by a limit %lld, other decode failures %lld\n",
           r, g_profile ? "SL" : "none", hdr_ok, hdr_limit, hdr_other, dec_ok, dec_limit,
           dec_other);
    if (first_other[0]) {
        printf("  first other header failure: %s", first_other);
    }
    free(e);
    free(c);
    return 0;
}

int main(int argc, char **argv)
{
    {
        const char *pf = getenv("PROBE_PROFILE");
        g_profile = (pf && strcmp(pf, "sl") == 0) ? 1 : 0;
    }
    {
        const char *pl = getenv("PROBE_LAYERS");
        g_layers = pl ? (unsigned)atoi(pl) : 0U;
    }
    {
        const char *pj = getenv("PROBE_JP2");
        g_jp2 = (pj && pj[0] == '1') ? 1 : 0;
    }
    {
        const char *ps = getenv("PROBE_STRICT");
        g_strict = (ps && ps[0] == '1') ? 1 : 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "cache")) {
        return cmd_cache(argv[2], argv[3], atoi(argv[4]));
    }
#ifdef PROBE_LIMITS
    if (argc >= 2 && !strcmp(argv[1], "apitest")) {
        /* the limits API's contract: profiles, unknown profile, NULL arguments, a compressor */
        opj_decoder_limits_t l;
        opj_codec_t *dec = opj_create_decompress(OPJ_CODEC_J2K);
        opj_codec_t *jp2 = opj_create_decompress(OPJ_CODEC_JP2);
        opj_codec_t *enc = opj_create_compress(OPJ_CODEC_J2K);
        int ok = 1;
        memset(&l, 0xAB, sizeof(l));
        ok &= opj_decoder_limits_init(&l, OPJ_DECODER_LIMITS_NONE) == OPJ_TRUE &&
              l.max_tiles == 0 &&
              l.max_components == 0 && l.max_layers == 0 && l.max_decoded_side == 0 &&
              l.max_memory == 0;
        ok &= opj_decoder_limits_init(&l, OPJ_DECODER_LIMITS_SL_TEXTURE) == OPJ_TRUE &&
              l.max_tiles == 1024 &&
              l.max_components == 8 && l.max_layers == 64 && l.max_decoded_side == 2048 &&
              l.max_memory == (OPJ_UINT64)256 * 1024 * 1024;
        memset(&l, 0xAB, sizeof(l));
        ok &= opj_decoder_limits_init(&l, 7) == OPJ_FALSE && l.max_tiles == 0 &&
              l.max_memory == 0;
        ok &= opj_decoder_limits_init(NULL, OPJ_DECODER_LIMITS_SL_TEXTURE) == OPJ_FALSE;
        opj_decoder_limits_init(&l, OPJ_DECODER_LIMITS_SL_TEXTURE);
        ok &= opj_decoder_set_limits(dec, &l) == OPJ_TRUE;
        ok &= opj_decoder_set_limits(jp2, &l) == OPJ_TRUE;
        ok &= opj_decoder_set_limits(dec, NULL) == OPJ_FALSE;
        ok &= opj_decoder_set_limits(NULL, &l) == OPJ_FALSE;
        opj_set_error_handler(enc, cb_info, NULL);
        ok &= opj_decoder_set_limits(enc, &l) == OPJ_FALSE;
        opj_destroy_codec(dec);
        opj_destroy_codec(jp2);
        opj_destroy_codec(enc);
        printf("limits API contract: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
#endif
    if (argc >= 5 && !strcmp(argv[1], "layhash")) {
        return cmd_layhash(argv[2], argv[3], (unsigned)atoi(argv[4]));
    }
    if (argc >= 6 && !strcmp(argv[1], "cuthash")) {
        return cmd_cuthash(argv[2], argv[3], atoi(argv[4]), argv[5]);
    }
    if (argc >= 4 && !strcmp(argv[1], "mem")) {
        return cmd_mem(argv[2], atoi(argv[3]), argc >= 5 ? (size_t)_strtoui64(argv[4],
                       NULL, 10) : 0);
    }
    if (argc >= 5 && !strcmp(argv[1], "loops")) {
        return cmd_loops((unsigned)atoi(argv[2]), (unsigned)atoi(argv[3]),
                         (unsigned)strtoul(argv[4], NULL, 0), argc >= 6 ? (unsigned)atoi(argv[5]) : 0);
    }
    if (argc >= 4 && !strcmp(argv[1], "setfactor")) {
        return cmd_setfactor(argv[2], (unsigned)atoi(argv[3]));
    }
    if (argc >= 4 && !strcmp(argv[1], "hash")) {
        return cmd_hash(argc, argv);
    }
    if (argc >= 3 && !strcmp(argv[1], "conf")) {
        return cmd_conf(argv[2]);
    }
    if (argc >= 6 && !strcmp(argv[1], "cut")) {
        return cmd_cut(argv[2], argv[3], atoi(argv[4]), argv[5]);
    }
    if (argc >= 5 && !strcmp(argv[1], "one")) {
        return cmd_one(argv[2], atoi(argv[3]), (size_t)_strtoui64(argv[4], NULL, 10));
    }
    if (argc >= 8 && !strcmp(argv[1], "hdr")) {
        return cmd_hdr((unsigned)atoi(argv[2]), (unsigned)atoi(argv[3]),
                       (unsigned)atoi(argv[4]),
                       (unsigned)atoi(argv[5]), (unsigned)atoi(argv[6]), (unsigned)atoi(argv[7]));
    }
#ifdef OPJ_ALLOC_CENSUS
    if (argc >= 4 && !strcmp(argv[1], "census")) {
        return cmd_census(argv[2], atoi(argv[3]), argc >= 5 ? atoi(argv[4]) : 1);
    }
#endif
    fprintf(stderr, "usage: probe conf|cut|one|hdr|census ...\n");
    return 2;
}
