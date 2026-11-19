/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Non-regression tests for decoding cut codestreams in non-strict mode and for
 * the decoder's resource limits (opj_decoder_set_limits). Self-contained: every
 * codestream is encoded or written in memory, except for ppt_layers, which reads
 * a stream of the conformance data.
 *
 * usage: test_partial_decode_limits [test]   (no test: all of them but ppt_layers
 *        and decode_area_iso)
 *        test_partial_decode_limits ppt_layers <p1_02.j2k>
 *   ppt_layers  every cut at a layer boundary of the conformance stream p1_02.j2k
 *               (packet headers in PPT) decodes as the whole file limited to
 *               those layers
 *   setfactor   a reduce factor set after the header is checked before it is
 *               applied (upstream issue #1631)
 *   limits_api  the contract of opj_decoder_limits_init and opj_decoder_set_limits
 *   limits      what OPJ_DECODER_LIMITS_SL_TEXTURE refuses, and what it decodes
 *   psot0       every cut of a codestream whose tile-part has Psot = 0 decodes
 *               as the same cut with Psot set
 *   packed      packet headers packed in PPM or PPT that end inside a header:
 *               the decode stays short and equals the in-band case
 *   cuts        every cut of small layer-progressive codestreams decodes, and
 *               every layer-limited decode of the whole file is one of them
 *   decode_area_iso <p0_12.j2k>  the same for every window of the conformance
 *               stream p0_12.j2k
 *   decode_area every window set with opj_set_decode_area decodes to the same
 *               samples as that window cut out of the whole image (5/3 exactly,
 *               9/7 within one level); run under AddressSanitizer and
 *               UndefinedBehaviorSanitizer it catches the windowed inverse DWT
 *               reading samples it never loaded (ISO p0_12.j2k, 3 x 5, 4
 *               resolutions: signed overflow at dwt.c in the lifting step)
   encode_termall16 TERMALL encodes of 9 to 16 bit samples stay inside their code-block buffers
               (run under AddressSanitizer) and decode to their samples
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "openjpeg.h"

#ifndef OPJ_HAVE_DECODER_LIMITS
/* a library without the limits API: the limits tests are skipped */
typedef struct {
    int unused;
} opj_decoder_limits_t;
#endif

static int g_failures = 0;

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAILED %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            ++g_failures; \
        } \
    } while (0)

static void quiet(const char *msg, void *client_data)
{
    (void)msg;
    (void)client_data;
}

/* ------------------------------------------------------------- byte buffers */

typedef struct {
    unsigned char *data;
    size_t size;      /* bytes written, or readable */
    size_t capacity;
    size_t pos;
} buffer_t;

static void buf_reserve(buffer_t *b, size_t n)
{
    if (n > b->capacity) {
        size_t c = b->capacity ? b->capacity : 4096;
        while (c < n) {
            c *= 2;
        }
        b->data = (unsigned char *)realloc(b->data, c);
        if (!b->data) {
            fprintf(stderr, "out of memory\n");
            exit(2);
        }
        b->capacity = c;
    }
}

static void put(buffer_t *b, const void *p, size_t n)
{
    buf_reserve(b, b->size + n);
    memcpy(b->data + b->size, p, n);
    b->size += n;
}

static void put8(buffer_t *b, unsigned v)
{
    unsigned char c = (unsigned char)v;
    put(b, &c, 1);
}

static void put16(buffer_t *b, unsigned v)
{
    put8(b, v >> 8);
    put8(b, v);
}

static void put32(buffer_t *b, unsigned long v)
{
    put16(b, (unsigned)(v >> 16) & 0xffffU);
    put16(b, (unsigned)v & 0xffffU);
}

/* ------------------------------------------------------------- memory streams */

static OPJ_SIZE_T mem_read(void *p, OPJ_SIZE_T n, void *user)
{
    buffer_t *b = (buffer_t *)user;
    size_t left;
    if (b->pos >= b->size) {
        return (OPJ_SIZE_T) - 1;
    }
    left = b->size - b->pos;
    if (n > left) {
        n = left;
    }
    memcpy(p, b->data + b->pos, n);
    b->pos += n;
    return n;
}

static OPJ_SIZE_T mem_write(void *p, OPJ_SIZE_T n, void *user)
{
    buffer_t *b = (buffer_t *)user;
    buf_reserve(b, b->pos + n);
    memcpy(b->data + b->pos, p, n);
    b->pos += n;
    if (b->pos > b->size) {
        b->size = b->pos;
    }
    return n;
}

static OPJ_OFF_T mem_skip(OPJ_OFF_T n, void *user)
{
    buffer_t *b = (buffer_t *)user;
    OPJ_OFF_T to = (OPJ_OFF_T)b->pos + n;
    if (to < 0) {
        return (OPJ_OFF_T) - 1;
    }
    b->pos = (size_t)to;
    return n;
}

static OPJ_BOOL mem_seek(OPJ_OFF_T to, void *user)
{
    buffer_t *b = (buffer_t *)user;
    if (to < 0) {
        return OPJ_FALSE;
    }
    b->pos = (size_t)to;
    return OPJ_TRUE;
}

static opj_stream_t *mem_stream(buffer_t *b, OPJ_BOOL input)
{
    opj_stream_t *s = opj_stream_create(4096, input);
    b->pos = 0;
    opj_stream_set_user_data(s, b, NULL);
    if (input) {
        opj_stream_set_user_data_length(s, b->size);
        opj_stream_set_read_function(s, mem_read);
    } else {
        opj_stream_set_write_function(s, mem_write);
    }
    opj_stream_set_skip_function(s, mem_skip);
    opj_stream_set_seek_function(s, mem_seek);
    return s;
}

/* ------------------------------------------------------------- encode, decode */

/* A w x h image of comps 8-bit components with a deterministic texture,
 * encoded in LRCP with the given layers (rates halving down to lossless or
 * near lossless), resolutions, code-block style and wavelet. */
static buffer_t encode(int w, int h, int comps, int numres, int layers,
                       int irreversible, int mode)
{
    opj_cparameters_t p;
    opj_image_cmptparm_t cp[4];
    opj_image_t *img;
    opj_codec_t *c;
    opj_stream_t *s;
    buffer_t out;
    int i, x, y;

    memset(&out, 0, sizeof(out));
    memset(cp, 0, sizeof(cp));
    for (i = 0; i < comps; ++i) {
        cp[i].dx = cp[i].dy = 1;
        cp[i].w = (OPJ_UINT32)w;
        cp[i].h = (OPJ_UINT32)h;
        cp[i].prec = 8;
        cp[i].sgnd = 0;
    }
    img = opj_image_create((OPJ_UINT32)comps, cp,
                           comps >= 3 ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY);
    img->x0 = img->y0 = 0;
    img->x1 = (OPJ_UINT32)w;
    img->y1 = (OPJ_UINT32)h;
    for (i = 0; i < comps; ++i) {
        for (y = 0; y < h; ++y) {
            for (x = 0; x < w; ++x) {
                img->comps[i].data[y * w + x] = ((x * 7 + y * 13 + i * 29) ^ (x * y)) & 255;
            }
        }
    }

    opj_set_default_encoder_parameters(&p);
    p.tcp_numlayers = layers;
    for (i = 0; i < layers; ++i) {
        p.tcp_rates[i] = (float)(1 << (layers - 1 - i) * 2);
    }
    p.tcp_rates[layers - 1] = 1.0f;
    p.cp_disto_alloc = 1;
    p.numresolution = numres;
    p.cblockw_init = 16;
    p.cblockh_init = 16;
    p.irreversible = irreversible;
    p.mode = mode;
    p.prog_order = OPJ_LRCP;
    p.tcp_mct = comps >= 3 ? 1 : 0;

    c = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, quiet, NULL);
    opj_set_error_handler(c, quiet, NULL);
    s = mem_stream(&out, OPJ_FALSE);
    if (!opj_setup_encoder(c, &p, img) || !opj_start_compress(c, img, s) ||
            !opj_encode(c, s) || !opj_end_compress(c, s)) {
        fprintf(stderr, "encoding failed\n");
        exit(2);
    }
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    opj_image_destroy(img);
    return out;
}

/* Decodes the first len bytes as the viewer does (memory stream, non-strict
 * unless asked, reduce factor before the header). *stage: 0 the header failed,
 * 1 the decode failed, 2 decoded. */
static opj_image_t *decode(const buffer_t *in, size_t len, unsigned reduce,
                           unsigned layers, int strict,
                           const opj_decoder_limits_t *limits,
                           OPJ_CODEC_FORMAT format, int *stage)
{
    opj_dparameters_t p;
    opj_codec_t *c;
    opj_stream_t *s;
    opj_image_t *img = NULL;
    buffer_t view = *in;

    view.size = len;
    opj_set_default_decoder_parameters(&p);
    p.cp_reduce = reduce;
    p.cp_layer = layers;
    c = opj_create_decompress(format);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, quiet, NULL);
    opj_set_error_handler(c, quiet, NULL);
    opj_setup_decoder(c, &p);
    opj_decoder_set_strict_mode(c, strict ? OPJ_TRUE : OPJ_FALSE);
#ifdef OPJ_HAVE_DECODER_LIMITS
    if (limits && !opj_decoder_set_limits(c, limits)) {
        fprintf(stderr, "opj_decoder_set_limits failed\n");
        exit(2);
    }
#else
    (void)limits;
#endif
    s = mem_stream(&view, OPJ_TRUE);
    *stage = 0;
    if (opj_read_header(s, c, &img)) {
        *stage = 1;
        if (opj_decode(c, s, img) && img->numcomps) {
            opj_end_decompress(c, s);
            *stage = 2;
        }
    }
    if (*stage != 2) {
        opj_image_destroy(img);
        img = NULL;
    }
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    return img;
}

static int same_image(const opj_image_t *a, const opj_image_t *b)
{
    OPJ_UINT32 i;
    if (!a || !b) {
        return a == b;
    }
    if (a->numcomps != b->numcomps) {
        return 0;
    }
    for (i = 0; i < a->numcomps; ++i) {
        const opj_image_comp_t *x = &a->comps[i], *y = &b->comps[i];
        if (x->w != y->w || x->h != y->h || x->prec != y->prec ||
                memcmp(x->data, y->data, (size_t)x->w * x->h * sizeof(OPJ_INT32))) {
            return 0;
        }
    }
    return 1;
}

/* PSNR of an image against a reference of the same geometry, over all samples
 * (8-bit peak); 999 when identical, -1 when the geometry differs */
static double psnr(const opj_image_t *img, const opj_image_t *ref)
{
    double se = 0.0, n = 0.0;
    OPJ_UINT32 i;
    size_t k, m;
    if (!img || !ref || img->numcomps != ref->numcomps) {
        return -1.0;
    }
    for (i = 0; i < img->numcomps; ++i) {
        const opj_image_comp_t *a = &img->comps[i], *b = &ref->comps[i];
        if (a->w != b->w || a->h != b->h) {
            return -1.0;
        }
        m = (size_t)a->w * a->h;
        for (k = 0; k < m; ++k) {
            const double d = (double)a->data[k] - (double)b->data[k];
            se += d * d;
        }
        n += (double)m;
    }
    if (se == 0.0) {
        return 999.0;
    }
    return 10.0 * log10(255.0 * 255.0 * n / se);
}

static unsigned long long image_hash(const opj_image_t *img)
{
    unsigned long long h = 1469598103934665603ULL;
    OPJ_UINT32 i;
    size_t k, n;
    h = (h ^ img->numcomps) * 1099511628211ULL;
    for (i = 0; i < img->numcomps; ++i) {
        const opj_image_comp_t *c = &img->comps[i];
        h = (h ^ c->w) * 1099511628211ULL;
        h = (h ^ c->h) * 1099511628211ULL;
        n = (size_t)c->w * c->h;
        for (k = 0; k < n; ++k) {
            h = (h ^ (unsigned)c->data[k]) * 1099511628211ULL;
        }
    }
    return h;
}

/* Length of the shortest cut with tile data: the first data byte after the SOD
 * marker of the first tile-part (a cut with no tile data at all is reported as
 * a failure, as it always was) */
static size_t first_cut(const buffer_t *b)
{
    size_t i;
    for (i = 0; i + 1 < b->size; ++i) {
        if (b->data[i] == 0xff && b->data[i + 1] == 0x93) {
            return i + 3;
        }
    }
    return b->size;
}

/* ------------------------------------------------------------- written codestreams */

typedef struct {
    unsigned w, h, tile_w, tile_h, comps, layers, levels;
    unsigned xcb, ycb;       /* code-block exponents - 2 */
    int precinct;            /* precinct exponent of every resolution, -1: none */
    int packed;              /* 0: in-band packet headers, 1: PPM, 2: PPT */
    unsigned char header;    /* the packet header byte (packed, or in-band) */
} craft_t;

/* SOC, SIZ, COD, QCD, PPM, then one tile-part per tile carrying one byte of
 * packet data (the in-band packet header, or the body when headers are packed),
 * then EOC; 5/3 wavelet, 8-bit unsigned components. */
static buffer_t craft(const craft_t *k)
{
    buffer_t b;
    unsigned i, n, tiles;
    const unsigned body = k->packed ? 0x00 : k->header;

    memset(&b, 0, sizeof(b));
    put16(&b, 0xff4f);
    put16(&b, 0xff51);
    put16(&b, 38 + 3 * k->comps);
    put16(&b, 0);
    put32(&b, k->w);
    put32(&b, k->h);
    put32(&b, 0);
    put32(&b, 0);
    put32(&b, k->tile_w);
    put32(&b, k->tile_h);
    put32(&b, 0);
    put32(&b, 0);
    put16(&b, k->comps);
    for (i = 0; i < k->comps; ++i) {
        put8(&b, 7);
        put8(&b, 1);
        put8(&b, 1);
    }
    put16(&b, 0xff52);
    put16(&b, 12 + (k->precinct >= 0 ? k->levels + 1 : 0));
    put8(&b, k->precinct >= 0 ? 1 : 0);
    put8(&b, 0);                       /* LRCP */
    put16(&b, k->layers);
    put8(&b, 0);
    put8(&b, k->levels);
    put8(&b, k->xcb);
    put8(&b, k->ycb);
    put8(&b, 0);
    put8(&b, 1);                       /* 5/3 */
    if (k->precinct >= 0) {
        for (i = 0; i <= k->levels; ++i) {
            put8(&b, (unsigned)k->precinct | ((unsigned)k->precinct << 4));
        }
    }
    put16(&b, 0xff5c);
    put16(&b, 3 + 3 * k->levels + 1);
    put8(&b, 0x40);                    /* no quantization, 2 guard bits */
    for (i = 0; i < 3 * k->levels + 1; ++i) {
        put8(&b, 9 << 3);
    }
    tiles = ((k->w + k->tile_w - 1) / k->tile_w) * ((k->h + k->tile_h - 1) /
            k->tile_h);
    if (k->packed == 1) {
        put16(&b, 0xff60);
        put16(&b, 2 + 1 + tiles * 5);
        put8(&b, 0);
        for (n = 0; n < tiles; ++n) {
            put32(&b, 1);
            put8(&b, k->header);
        }
    }
    for (n = 0; n < tiles; ++n) {
        const unsigned ppt = k->packed == 2 ? 2 + 2 + 1 + 1 : 0;
        put16(&b, 0xff90);
        put16(&b, 10);
        put16(&b, n);
        put32(&b, 12 + ppt + 2 + 1);
        put8(&b, 0);
        put8(&b, 1);
        if (ppt) {
            put16(&b, 0xff61);
            put16(&b, 2 + 1 + 1);
            put8(&b, 0);
            put8(&b, k->header);
        }
        put16(&b, 0xff93);
        put8(&b, body);
    }
    put16(&b, 0xffd9);
    return b;
}

/* A JP2 file whose palette maps one w x w component (the crafted codestream)
 * to npc output components. */
static buffer_t palette_jp2(unsigned w, unsigned npc)
{
    craft_t k = { 0 };
    buffer_t cs, b, box;
    unsigned i;

    k.w = k.h = k.tile_w = k.tile_h = w;
    k.comps = 1;
    k.layers = 1;
    k.levels = 5;
    k.xcb = k.ycb = 4;
    k.precinct = -1;
    cs = craft(&k);
    memset(&b, 0, sizeof(b));
    memset(&box, 0, sizeof(box));
    put32(&b, 12);
    put(&b, "jP  \r\n\x87\n", 8);
    put32(&b, 20);
    put(&b, "ftypjp2 ", 8);
    put32(&b, 0);
    put(&b, "jp2 ", 4);
    /* jp2h: ihdr, colr, pclr, cmap */
    put32(&box, 22);
    put(&box, "ihdr", 4);
    put32(&box, w);
    put32(&box, w);
    put16(&box, 1);
    put8(&box, 7);
    put8(&box, 7);
    put8(&box, 0);
    put8(&box, 0);
    put32(&box, 15);
    put(&box, "colr", 4);
    put8(&box, 1);
    put8(&box, 0);
    put8(&box, 0);
    put32(&box, 16);
    put32(&box, 8 + 3 + 2 * npc);
    put(&box, "pclr", 4);
    put16(&box, 1);
    put8(&box, npc);
    for (i = 0; i < npc; ++i) {
        put8(&box, 7);
    }
    for (i = 0; i < npc; ++i) {
        put8(&box, i);
    }
    put32(&box, 8 + 4 * npc);
    put(&box, "cmap", 4);
    for (i = 0; i < npc; ++i) {
        put16(&box, 0);
        put8(&box, 1);
        put8(&box, i);
    }
    put32(&b, (unsigned long)(8 + box.size));
    put(&b, "jp2h", 4);
    put(&b, box.data, box.size);
    put32(&b, (unsigned long)(8 + cs.size));
    put(&b, "jp2c", 4);
    put(&b, cs.data, cs.size);
    free(box.data);
    free(cs.data);
    return b;
}

/* ------------------------------------------------------------- the tests */

static void test_setfactor(void)
{
    buffer_t in = encode(16, 16, 1, 5, 1, 0, 0);
    unsigned factor;

    for (factor = 4; factor <= 6; ++factor) {
        opj_dparameters_t p;
        opj_codec_t *c = opj_create_decompress(OPJ_CODEC_J2K);
        opj_stream_t *s;
        opj_image_t *img = NULL, *ref;
        OPJ_BOOL set;
        int stage;

        opj_set_info_handler(c, quiet, NULL);
        opj_set_warning_handler(c, quiet, NULL);
        opj_set_error_handler(c, quiet, NULL);
        opj_set_default_decoder_parameters(&p);
        opj_setup_decoder(c, &p);
        opj_decoder_set_strict_mode(c, OPJ_FALSE);
        s = mem_stream(&in, OPJ_TRUE);
        CHECK(opj_read_header(s, c, &img), "setfactor: header");
        /* 5 resolutions: 4 is the last valid factor */
        set = opj_set_decoded_resolution_factor(c, factor);
        CHECK(set == (factor < 5), "setfactor: factor %u accepted %d", factor, set);
        CHECK(opj_decode(c, s, img), "setfactor: decode after factor %u", factor);
        ref = decode(&in, in.size, set ? factor : 0, 0, 0, NULL, OPJ_CODEC_J2K, &stage);
        CHECK(stage == 2 && same_image(img, ref),
              "setfactor: decode after factor %u differs from the decode at %u", factor,
              set ? factor : 0);
        opj_image_destroy(ref);
        opj_image_destroy(img);
        opj_stream_destroy(s);
        opj_destroy_codec(c);
    }
    free(in.data);
}

#ifdef OPJ_HAVE_DECODER_LIMITS
static void test_limits_api(void)
{
    opj_decoder_limits_t l, zero;
    opj_codec_t *c;
    opj_stream_t *s;
    opj_image_t *img = NULL;
    buffer_t in = encode(16, 16, 1, 3, 1, 0, 0);
    int f;
    const OPJ_CODEC_FORMAT formats[2] = { OPJ_CODEC_J2K, OPJ_CODEC_JP2 };

    memset(&zero, 0, sizeof(zero));
    CHECK(!opj_decoder_limits_init(NULL, OPJ_DECODER_LIMITS_NONE),
          "limits_init(NULL)");
    memset(&l, 0xff, sizeof(l));
    CHECK(opj_decoder_limits_init(&l, OPJ_DECODER_LIMITS_NONE) &&
          !memcmp(&l, &zero, sizeof(l)), "limits_init(NONE)");
    CHECK(opj_decoder_limits_init(&l, OPJ_DECODER_LIMITS_SL_TEXTURE) &&
          l.max_tiles == 1024 && l.max_components == 8 && l.max_layers == 64 &&
          l.max_decoded_side == 2048 && l.max_memory == (OPJ_UINT64)256 * 1024 * 1024,
          "limits_init(SL_TEXTURE)");
    memset(&l, 0xff, sizeof(l));
    CHECK(!opj_decoder_limits_init(&l, 12345) && !memcmp(&l, &zero, sizeof(l)),
          "limits_init(unknown) must give no limits and fail");
    opj_decoder_limits_init(&l, OPJ_DECODER_LIMITS_SL_TEXTURE);

    CHECK(!opj_decoder_set_limits(NULL, &l), "set_limits(NULL codec)");
    c = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_error_handler(c, quiet, NULL);
    CHECK(!opj_decoder_set_limits(c, &l), "set_limits(compressor)");
    opj_destroy_codec(c);

    for (f = 0; f < 2; ++f) {
        c = opj_create_decompress(formats[f]);
        opj_set_error_handler(c, quiet, NULL);
        opj_set_warning_handler(c, quiet, NULL);
        CHECK(!opj_decoder_set_limits(c, NULL), "set_limits(NULL limits)");
        CHECK(opj_decoder_set_limits(c, &l), "set_limits before the header");
        if (formats[f] == OPJ_CODEC_J2K) {
            s = mem_stream(&in, OPJ_TRUE);
            CHECK(opj_read_header(s, c, &img), "limits_api: header");
            CHECK(!opj_decoder_set_limits(c, &l),
                  "set_limits after the header must fail");
            opj_image_destroy(img);
            img = NULL;
            opj_stream_destroy(s);
        }
        opj_destroy_codec(c);
    }
    free(in.data);
}

static int decodes(const buffer_t *in, unsigned reduce,
                   const opj_decoder_limits_t *limits, OPJ_CODEC_FORMAT format,
                   int *stage)
{
    opj_image_t *img = decode(in, in->size, reduce, 0, 0, limits, format, stage);
    opj_image_destroy(img);
    return *stage == 2;
}

static void test_limits(void)
{
    opj_decoder_limits_t sl;
    craft_t k;
    buffer_t in;
    int stage;

    opj_decoder_limits_init(&sl, OPJ_DECODER_LIMITS_SL_TEXTURE);
    memset(&k, 0, sizeof(k));
    k.layers = 1;
    k.levels = 5;
    k.xcb = k.ycb = 4;
    k.precinct = -1;

    /* 1,025 tiles of 1 x 1 */
    k.w = 1025;
    k.h = 1;
    k.tile_w = k.tile_h = 1;
    k.comps = 1;
    k.levels = 0;
    in = craft(&k);
    CHECK(!decodes(&in, 0, &sl, OPJ_CODEC_J2K, &stage) && stage == 0,
          "1,025 tiles must be refused at the header (stage %d)", stage);
    CHECK(decodes(&in, 0, NULL, OPJ_CODEC_J2K, &stage),
          "1,025 tiles without limits");
    free(in.data);
    k.levels = 5;

    /* 9 components */
    k.w = k.h = k.tile_w = k.tile_h = 64;
    k.comps = 9;
    in = craft(&k);
    CHECK(!decodes(&in, 0, &sl, OPJ_CODEC_J2K, &stage) && stage == 0,
          "9 components must be refused at the header (stage %d)", stage);
    free(in.data);

    /* 65 layers */
    k.comps = 3;
    k.layers = 65;
    in = craft(&k);
    CHECK(!decodes(&in, 0, &sl, OPJ_CODEC_J2K, &stage) && stage == 0,
          "65 layers must be refused at the header (stage %d)", stage);
    free(in.data);
    k.layers = 8;

    /* the largest Second Life texture: 2048 x 2048, 5 components, 8 layers */
    k.w = k.h = k.tile_w = k.tile_h = 2048;
    k.comps = 5;
    in = craft(&k);
    CHECK(decodes(&in, 0, &sl, OPJ_CODEC_J2K, &stage),
          "2048 x 2048 x 5 with the SL profile");
    free(in.data);

    /* 4096 x 4096: too large at reduce 0, decodable at reduce 1 */
    k.w = k.h = k.tile_w = k.tile_h = 4096;
    k.comps = 1;
    in = craft(&k);
    CHECK(!decodes(&in, 0, &sl, OPJ_CODEC_J2K, &stage) && stage == 1,
          "4096 x 4096 at reduce 0 must be refused at the decode (stage %d)", stage);
    CHECK(decodes(&in, 1, &sl, OPJ_CODEC_J2K, &stage), "4096 x 4096 at reduce 1");
    free(in.data);

    /* 65536 x 65536 with 4 components at reduce 5 (a 2048 x 2048 output): the
     * structures of the full resolution exceed the memory budget */
    k.w = k.h = k.tile_w = k.tile_h = 65536;
    k.comps = 4;
    in = craft(&k);
    CHECK(!decodes(&in, 5, &sl, OPJ_CODEC_J2K, &stage) && stage == 1,
          "65536 x 65536 x 4 at reduce 5 must be refused at the decode (stage %d)",
          stage);
    free(in.data);

    /* 4 tiles of 2048 x 2048 with 4 x 4 code-blocks at reduce 5: each tile fits,
     * the decode does not */
    k.w = k.h = 4096;
    k.tile_w = k.tile_h = 2048;
    k.comps = 1;
    k.layers = 1;
    k.xcb = k.ycb = 0;
    in = craft(&k);
    CHECK(!decodes(&in, 5, &sl, OPJ_CODEC_J2K, &stage) && stage == 1,
          "4 tiles of 2048 x 2048 with 4 x 4 code-blocks must exceed the budget (stage %d)",
          stage);
    free(in.data);

    /* a JP2 palette of 255 columns: 255 components */
    in = palette_jp2(64, 255);
    CHECK(!decodes(&in, 0, &sl, OPJ_CODEC_JP2, &stage),
          "a 255-column palette must be refused with the SL profile");
    CHECK(decodes(&in, 0, NULL, OPJ_CODEC_JP2, &stage),
          "a 255-column palette without limits");
    free(in.data);
    in = palette_jp2(64, 3);
    CHECK(decodes(&in, 0, &sl, OPJ_CODEC_JP2, &stage),
          "a 3-column palette with the SL profile");
    free(in.data);
}
#else
static void test_limits_api(void)
{
    printf("limits_api: skipped, the library has no limits API\n");
}

static void test_limits(void)
{
    printf("limits: skipped, the library has no limits API\n");
}
#endif

static void test_psot0(void)
{
    buffer_t in = encode(32, 32, 3, 4, 3, 0, 0), zero;
    size_t sot, len, start;
    unsigned differ = 0, decoded = 0;

    for (sot = 0; sot + 1 < in.size && !(in.data[sot] == 0xff &&
                                         in.data[sot + 1] == 0x90); ++sot) {
    }
    CHECK(sot + 12 < in.size &&
          in.data[sot + 11] == 1, "psot0: one tile-part expected");
    zero = in;
    zero.data = (unsigned char *)malloc(in.size);
    memcpy(zero.data, in.data, in.size);
    memset(zero.data + sot + 6, 0, 4);
    start = first_cut(&in);
    for (len = start; len <= in.size; ++len) {
        int s1, s2;
        opj_image_t *a = decode(&in, len, 0, 0, 0, NULL, OPJ_CODEC_J2K, &s1);
        opj_image_t *b = decode(&zero, len, 0, 0, 0, NULL, OPJ_CODEC_J2K, &s2);
        if (s1 != s2 || !same_image(a, b)) {
            ++differ;
        }
        decoded += (s1 == 2);
        opj_image_destroy(a);
        opj_image_destroy(b);
    }
    CHECK(differ == 0, "psot0: %u of %u cuts decode differently with Psot = 0",
          differ,
          (unsigned)(in.size - start + 1));
    CHECK(decoded == in.size - start + 1, "psot0: %u of %u cuts decode", decoded,
          (unsigned)(in.size - start + 1));
    free(in.data);
    free(zero.data);
}

static void test_packed(void)
{
    /* 256 x 256, 3 components, 16 x 16 precincts, 2,000 layers, and a single
     * packet header byte 0xFF that ends inside the first header: in-band, in a
     * PPM marker or in a PPT marker. Every packet after it is absent, so all
     * three decode to the same image about as fast as the in-band form, which
     * stops at the end of the tile's data; parsing the packed byte again for
     * each of the 2 million packets took seconds to minutes. */
    static const char *const names[3] = { "in-band", "PPM", "PPT" };
    craft_t k;
    opj_image_t *ref = NULL;
    double inband = 0.0;
    int form;

    memset(&k, 0, sizeof(k));
    k.w = k.h = k.tile_w = k.tile_h = 256;
    k.comps = 3;
    k.layers = 2000;
    k.levels = 5;
    k.xcb = k.ycb = 4;
    k.precinct = 4;
    k.header = 0xff;
    for (form = 0; form < 3; ++form) {
        int stage;
        clock_t t0;
        double seconds;
        opj_image_t *img;
        buffer_t in;

        k.packed = form;
        in = craft(&k);
        t0 = clock();
        img = decode(&in, in.size, 0, 0, 0, NULL, OPJ_CODEC_J2K, &stage);
        seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("packed: %s headers, %.3f s\n", names[form], seconds);
        CHECK(stage == 2, "packed: %s headers: decode", names[form]);
        if (form == 0) {
            ref = img;
            inband = seconds;
        } else {
            CHECK(seconds <= 1.0 + 10.0 * inband,
                  "packed: %s headers took %.3f s, the in-band ones %.3f s", names[form],
                  seconds, inband);
            CHECK(same_image(ref, img),
                  "packed: %s headers: the image differs from in-band",
                  names[form]);
            opj_image_destroy(img);
        }
        free(in.data);
    }
    opj_image_destroy(ref);
}

static void test_cuts(void)
{
    /* 5/3 with no mode switch, 5/3 with all of them, 9/7: LRCP, 5 layers */
    static const int kinds[3][2] = { { 0, 0 }, { 0, 63 }, { 1, 0 } };
    static const unsigned reduces[2] = { 0, 2 };
    int kind, r;

    for (kind = 0; kind < 3; ++kind) {
        buffer_t in = encode(32, 32, 3, 4, 5, kinds[kind][0], kinds[kind][1]);
        const size_t start = first_cut(&in);
        for (r = 0; r < 2; ++r) {
            const size_t n = in.size - start + 1;
            unsigned long long *hashes = (unsigned long long *)malloc(n * sizeof(*hashes));
            unsigned failed = 0, drops = 0, layers;
            size_t len, i;
            double best = -1.0;
            int whole_stage;
            opj_image_t *whole = decode(&in, in.size, reduces[r], 0, 0, NULL, OPJ_CODEC_J2K,
                                        &whole_stage);

            /* every cut decodes, and none is clearly worse than a shorter one: a
             * longer cut keeps everything a shorter one had (refinement can move a
             * sample past its final value by less than 1.2 dB on tiny outputs) */
            for (len = start; len <= in.size; ++len) {
                int stage;
                opj_image_t *img = decode(&in, len, reduces[r], 0, 0, NULL, OPJ_CODEC_J2K,
                                          &stage);
                const double q = psnr(img, whole);
                hashes[len - start] = img ? image_hash(img) : 0;
                failed += (stage != 2);
                if (q >= 0.0 && q < best - 1.2) {
                    ++drops;
                }
                if (q > best) {
                    best = q;
                }
                opj_image_destroy(img);
            }
            opj_image_destroy(whole);
            CHECK(failed == 0, "cuts kind %d reduce %u: %u of %u cuts do not decode", kind,
                  reduces[r], failed, (unsigned)n);
            CHECK(drops == 0,
                  "cuts kind %d reduce %u: %u of %u cuts decode more than 1.2 dB "
                  "worse than a shorter one", kind, reduces[r], drops, (unsigned)n);
            for (layers = 1; layers <= 5; ++layers) {
                int stage, found = 0;
                opj_image_t *img = decode(&in, in.size, reduces[r], layers, 0, NULL,
                                          OPJ_CODEC_J2K, &stage);
                const unsigned long long h = img ? image_hash(img) : 1;
                for (i = 0; i < n && !found; ++i) {
                    found = hashes[i] == h;
                }
                CHECK(stage == 2 && found,
                      "cuts kind %d reduce %u: no cut decodes as the first %u layers", kind,
                      reduces[r], layers);
                opj_image_destroy(img);
            }
            free(hashes);
        }
        free(in.data);
    }
}

/* ISO/IEC 15444-4 p1_02.j2k (conformance data): one tile, LRCP, 19 layers, packet
 * headers in PPT markers, so the tile data carries the packet bodies only. A cut
 * where layer k's data ends must decode exactly as the whole file limited to k
 * layers (cp_layer), at reduce 0 and 2: the packed headers that follow the cut are
 * read in step, and a code-block that lost passes gets no more. The offsets are
 * where each layer's packet data ends in that file (layer 19 has none). */
static void test_ppt_layers(const char *path)
{
    static const size_t ends[19] = {
        6096, 6524, 7144, 9671, 9695, 12818, 12980, 16240, 16483, 19511, 21315, 23184,
        43497, 54849, 90586, 122332, 209066, 263088, 263088
    };
    static const unsigned reduces[2] = { 0, 2 };
    buffer_t in;
    FILE *f = fopen(path, "rb");
    long n;
    unsigned k, r;

    memset(&in, 0, sizeof(in));
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf_reserve(&in, (size_t)n);
    in.size = fread(in.data, 1, (size_t)n, f);
    fclose(f);
    CHECK(in.size == 263090, "ppt_layers: %s is not the expected p1_02.j2k", path);
    if (in.size != 263090) {
        free(in.data);
        return;
    }
    for (r = 0; r < 2; ++r) {
        for (k = 1; k <= 19; ++k) {
            int cut_stage, layer_stage;
            opj_image_t *cut = decode(&in, ends[k - 1], reduces[r], 0, 0, NULL,
                                      OPJ_CODEC_J2K, &cut_stage);
            opj_image_t *layers = decode(&in, in.size, reduces[r], k, 0, NULL,
                                         OPJ_CODEC_J2K, &layer_stage);
            CHECK(cut_stage == 2 && layer_stage == 2 && same_image(cut, layers),
                  "ppt_layers reduce %u: the cut at the end of layer %u does not decode "
                  "as the first %u layers", reduces[r], k, k);
            opj_image_destroy(cut);
            opj_image_destroy(layers);
        }
    }
    free(in.data);
}

/* Decodes the window [x0,x1) x [y0,y1) of the whole-resolution image */
static opj_image_t *decode_window(const buffer_t *in, unsigned x0, unsigned y0,
                                  unsigned x1, unsigned y1)
{
    opj_dparameters_t p;
    opj_codec_t *c;
    opj_stream_t *s;
    opj_image_t *img = NULL;
    buffer_t view = *in;
    int ok = 0;

    opj_set_default_decoder_parameters(&p);
    c = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, quiet, NULL);
    opj_set_error_handler(c, quiet, NULL);
    opj_setup_decoder(c, &p);
    s = mem_stream(&view, OPJ_TRUE);
    if (opj_read_header(s, c, &img) &&
            opj_set_decode_area(c, img, (OPJ_INT32)x0, (OPJ_INT32)y0,
                                (OPJ_INT32)x1, (OPJ_INT32)y1) &&
            opj_decode(c, s, img) && img->numcomps) {
        ok = 1;
    }
    if (!ok) {
        opj_image_destroy(img);
        img = NULL;
    }
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    return img;
}

static void check_windows(const char *name, const buffer_t *in, int irreversible,
                          unsigned step)
{
    int stage;
    opj_image_t *full = decode(in, in->size, 0, 0, 1, NULL, OPJ_CODEC_J2K, &stage);
    unsigned w, h, x0, y0, x1, y1, n = 0;

    CHECK(full != NULL, "%s: whole decode failed", name);
    if (!full) {
        return;
    }
    w = full->comps[0].w; /* one tile at the origin, components not subsampled */
    h = full->comps[0].h;
    for (y0 = 0; y0 < h; y0 += step) {
        for (x0 = 0; x0 < w; x0 += step) {
            for (y1 = y0 + 1; y1 <= h; y1 += step) {
                for (x1 = x0 + 1; x1 <= w; x1 += step) {
                    opj_image_t *win = decode_window(in, x0, y0, x1, y1);
                    OPJ_UINT32 i;
                    unsigned x, y;
                    int bad = 0;
                    ++n;
                    if (!win) {
                        CHECK(0, "%s: window %u,%u,%u,%u did not decode", name, x0, y0, x1, y1);
                        continue;
                    }
                    for (i = 0; i < win->numcomps && !bad; ++i) {
                        const opj_image_comp_t *c = &win->comps[i];
                        if (c->w != x1 - x0 || c->h != y1 - y0) {
                            bad = 1;
                            break;
                        }
                        for (y = 0; y < c->h && !bad; ++y) {
                            for (x = 0; x < c->w; ++x) {
                                const int d = c->data[(size_t)y * c->w + x] -
                                              full->comps[i].data[(size_t)(y0 + y) * w + x0 + x];
                                if (irreversible ? (d > 1 || d < -1) : d != 0) {
                                    bad = 1;
                                    break;
                                }
                            }
                        }
                    }
                    CHECK(!bad, "%s: window %u,%u,%u,%u differs from the cut-out of the whole image",
                          name, x0, y0, x1, y1);
                    opj_image_destroy(win);
                }
            }
        }
    }
    CHECK(n > 0, "%s: no window tested", name);
    opj_image_destroy(full);
}

static void test_decode_area(void)
{
    static const struct {
        const char *name;
        int w, h, comps, numres, layers, irreversible;
        unsigned step;
    } cfg[] = {
        /* the geometry of ISO p0_12.j2k, every window */
        /* (the encoder allows log2(min side) + 1 resolutions at most; the 4
         * resolutions of the 3 x 5 stream come with the conformance file) */
        { "3x5 5/3 2 res", 3, 5, 1, 2, 1, 0, 1 },
        { "5x3 5/3 2 res", 5, 3, 1, 2, 1, 0, 1 },
        { "7x9 5/3 3 res", 7, 9, 1, 3, 1, 0, 1 },
        { "37x29 5/3 5 res", 37, 29, 1, 5, 1, 0, 3 },
        { "40x33 5/3 RGB 4 res", 40, 33, 3, 4, 2, 0, 5 },
        { "3x5 9/7 2 res", 3, 5, 1, 2, 1, 1, 1 },
        { "37x29 9/7 5 res", 37, 29, 1, 5, 1, 1, 3 },
    };
    size_t i;
    for (i = 0; i < sizeof(cfg) / sizeof(cfg[0]); ++i) {
        buffer_t b = encode(cfg[i].w, cfg[i].h, cfg[i].comps, cfg[i].numres,
                            cfg[i].layers, cfg[i].irreversible, 0);
        check_windows(cfg[i].name, &b, cfg[i].irreversible, cfg[i].step);
        free(b.data);
    }
}

/* ISO/IEC 15444-4 p0_12.j2k (conformance data): 3 x 5, one component, 4
 * resolutions, 5/3. Every window. */
static void test_decode_area_iso(const char *path)
{
    buffer_t in;
    FILE *f = fopen(path, "rb");
    long n;

    memset(&in, 0, sizeof(in));
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf_reserve(&in, (size_t)n);
    in.size = fread(in.data, 1, (size_t)n, f);
    fclose(f);
    check_windows("p0_12.j2k", &in, 0, 1);
    free(in.data);
}

/* ------------------------------------------------------------ encoder fixes */

static unsigned g_seed = 12345;
static unsigned lcg(void)
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

typedef struct {
    int w, h, comps, prec, sgnd;
    int kind;     /* 0 noise over the whole range, 1 3 x 3 blocks of the two extremes, 2 a ramp, 3 sparse extremes */
    int numres, mode, cbw, cbh, layers, mct, eph;
    const char *guard; /* "GUARD_BITS=n" or NULL */
} enc_case_t;

/* Encodes without loss. Returns 1 and the stream in out, or 0 when the encoder refuses the
 * image; ref receives the samples the stream must decode to. */
static int enc_case(const enc_case_t *k, buffer_t *out, opj_image_t **ref)
{
    opj_cparameters_t p;
    opj_image_cmptparm_t cp[4];
    opj_image_t *img;
    opj_codec_t *c;
    opj_stream_t *s;
    int i, x, y, ok;
    const int lo = k->sgnd ? -(1 << (k->prec - 1)) : 0;
    const int hi = k->sgnd ? (1 << (k->prec - 1)) - 1 : (1 << k->prec) - 1;

    memset(out, 0, sizeof(*out));
    memset(cp, 0, sizeof(cp));
    for (i = 0; i < k->comps; ++i) {
        cp[i].dx = cp[i].dy = 1;
        cp[i].w = (OPJ_UINT32)k->w;
        cp[i].h = (OPJ_UINT32)k->h;
        cp[i].prec = (OPJ_UINT32)k->prec;
        cp[i].sgnd = (OPJ_UINT32)k->sgnd;
    }
    img = opj_image_create((OPJ_UINT32)k->comps, cp,
                           k->comps >= 3 ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY);
    *ref = opj_image_create((OPJ_UINT32)k->comps, cp,
                            k->comps >= 3 ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY);
    img->x0 = img->y0 = 0;
    img->x1 = (OPJ_UINT32)k->w;
    img->y1 = (OPJ_UINT32)k->h;
    for (i = 0; i < k->comps; ++i) {
        for (y = 0; y < k->h; ++y) {
            for (x = 0; x < k->w; ++x) {
                const int v = k->kind == 0 ? lo + (int)(lcg() % ((unsigned)(hi - lo) + 1u)) :
                              k->kind == 2 ? lo + (int)(((double)(hi - lo)) *
                                      (x + 2 * y + 3 * i + (int)(lcg() & 3)) / (k->w + 2 * k->h + 8)) :
                              k->kind == 3 ? ((lcg() % 29) == 0 ? hi : lo + (hi - lo) / 2) :
                              (((x / 3 + y / 3 + i) & 1) ? hi : lo);
                img->comps[i].data[y * k->w + x] = v;
                (*ref)->comps[i].data[y * k->w + x] = v;
            }
        }
    }

    opj_set_default_encoder_parameters(&p);
    p.cp_disto_alloc = 1;
    p.tcp_numlayers = k->layers;
    for (i = 0; i < k->layers; ++i) {
        p.tcp_rates[i] = 1.0f + 0.25f * (float)(k->layers - 1 - i);
    }
    p.tcp_rates[k->layers - 1] = 0.0f;
    p.numresolution = k->numres;
    p.cblockw_init = k->cbw;
    p.cblockh_init = k->cbh;
    p.irreversible = 0;
    p.mode = k->mode;
    p.tcp_mct = k->mct;
    if (k->eph) {
        p.csty |= 4;
    }

    c = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_info_handler(c, quiet, NULL);
    opj_set_warning_handler(c, quiet, NULL);
    opj_set_error_handler(c, quiet, NULL);
    s = mem_stream(out, OPJ_FALSE);
    ok = opj_setup_encoder(c, &p, img);
    if (ok && k->guard) {
        const char *opts[2];
        opts[0] = k->guard;
        opts[1] = NULL;
        ok = opj_encoder_set_extra_options(c, opts);
    }
    ok = ok && opj_start_compress(c, img, s) && opj_encode(c, s) && opj_end_compress(c, s);
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    opj_image_destroy(img);
    if (!ok) {
        free(out->data);
        memset(out, 0, sizeof(*out));
    }
    return ok;
}

/* 1 when the stream decodes to the samples of ref */
static int same_as(const buffer_t *enc, const opj_image_t *ref)
{
    int stage, same = 1;
    opj_image_t *dec = decode(enc, enc->size, 0, 0, 1, NULL, OPJ_CODEC_J2K, &stage);
    OPJ_UINT32 i;
    size_t k;

    if (!dec || dec->numcomps != ref->numcomps) {
        opj_image_destroy(dec);
        return 0;
    }
    for (i = 0; i < ref->numcomps && same; ++i) {
        const size_t n = (size_t)ref->comps[i].w * ref->comps[i].h;
        for (k = 0; k < n; ++k) {
            if (dec->comps[i].data[k] != ref->comps[i].data[k]) {
                same = 0;
                break;
            }
        }
    }
    opj_image_destroy(dec);
    return same;
}

/* Fork audit S5 / neo-j2k OD-4: with TERMALL every pass is terminated and the MQ flush of a
 * pass wrote up to three bytes past the code-block buffer, whose margin was measured on 8 bit
 * samples (a one byte heap overflow with 16 bit samples; AddressSanitizer reports it). The
 * encodes below are the case that found it (28 x 2, three 16 bit signed components, 4 x 32
 * code-blocks, RESET and TERMALL, four layers, EPH) and its neighbours; every one that is
 * encoded must decode to its samples. */
static void test_encode_termall16(void)
{
    int i, encoded = 0;

    g_seed = 7;
    for (i = 0; i < 120; ++i) {
        enc_case_t k;
        buffer_t out;
        opj_image_t *ref;

        memset(&k, 0, sizeof(k));
        if (i >= 70) {
            /* a code-block of one sample (an edge block of the image) with the 16 bit extreme
             * the most passes: 58 terminated passes of about two bytes, the 78 bytes of the
             * buffer hold 38 */
            k.w = 1 + (i - 70) % 3;
            k.h = 1;
            k.comps = 1;
            k.prec = 16;
            k.sgnd = (i - 70) / 3 & 1;
            k.kind = 1;
            k.numres = 1;
            k.mode = 4 | ((i & 1) ? 1 : 0);
            k.cbw = k.cbh = 4;
            k.layers = 1;
            k.guard = "GUARD_BITS=5";
            if (enc_case(&k, &out, &ref)) {
                ++encoded;
                CHECK(same_as(&out, ref), "case %d: the stream does not decode to the samples", i);
                free(out.data);
            }
            opj_image_destroy(ref);
            continue;
        }
        k.w = i < 35 ? 28 : 1 + (int)(lcg() % 40);
        k.h = i < 35 ? 2 : 1 + (int)(lcg() % 40);
        k.comps = 3;
        k.prec = i < 35 ? 16 : 9 + (int)(lcg() % 8);
        k.sgnd = i & 1;
        k.kind = i % 4;
        k.numres = 1 + (int)(lcg() % 2);
        k.mode = 1 | 4 | (i % 3 == 0 ? 32 : 0) | (i % 5 == 0 ? 8 : 0);
        k.cbw = 4;
        k.cbh = i < 35 ? 32 : 4 << (int)(lcg() % 4);
        k.layers = 1 + (int)(lcg() % 2);
        k.mct = 0;
        k.eph = 1;
        k.guard = "GUARD_BITS=5"; /* the guard bits are not what this test is about */
        if (enc_case(&k, &out, &ref)) {
            ++encoded;
            CHECK(same_as(&out, ref), "case %d (%dx%d, %d bit): the stream does not decode to the samples",
                  i, k.w, k.h, k.prec);
            free(out.data);
        }
        opj_image_destroy(ref);
    }
    CHECK(encoded >= 40, "only %d of 120 TERMALL encodes of 9 to 16 bit samples succeeded", encoded);
}

int main(int argc, char **argv)
{
    static const struct {
        const char *name;
        void (*run)(void);
    } tests[] = {
        { "setfactor", test_setfactor },
        { "limits_api", test_limits_api },
        { "limits", test_limits },
        { "psot0", test_psot0 },
        { "packed", test_packed },
        { "cuts", test_cuts },
        { "decode_area", test_decode_area },
        { "encode_termall16", test_encode_termall16 },
    };
    size_t i;
    int ran = 0;

    if (argc >= 3 && !strcmp(argv[1], "decode_area_iso")) {
        test_decode_area_iso(argv[2]);
        ++ran;
    }
    if (argc >= 3 && !strcmp(argv[1], "ppt_layers")) {
        test_ppt_layers(argv[2]);
        ++ran;
    }
    for (i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        if (argc < 2 || !strcmp(argv[1], tests[i].name)) {
            tests[i].run();
            ++ran;
        }
    }
    if (!ran) {
        fprintf(stderr, "unknown test %s\n", argv[1]);
        return 2;
    }
    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("ok\n");
    return 0;
}
