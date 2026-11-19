/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * neo-j2k stage 1a: libFuzzer target for the viewer's decode path of the OpenJPEG fork.
 * Input: 3 control bytes, then a J2K codestream.
 *   byte 0: bits 0-2 reduce factor (mod 6), bit 3 limits profile OPJ_DECODER_LIMITS_SL_TEXTURE,
 *           bit 4 strict mode, bit 5 set the reduce factor after the header (header-once order),
 *           bit 6 decode area (the top-left quarter of the image), only with FUZZ1A_DECODE_AREA=1
 *   bytes 1-2: cut, little endian; 0 = whole codestream, else length = cut % (length + 1)
 * The decode mirrors llimagej2coj.cpp (memory stream with read/skip/seek, no-op message handlers).
 * Not a viewer component; nothing here ships.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "openjpeg.h"

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
/* The decode area (control bit 6) is off: the windowed 5/3 inverse DWT reads uninitialised
 * samples on ISO p0_12.j2k already in the shipped 3b1a6a96 (UBSan, dwt.c:2627), a path the viewer
 * never takes. FUZZ1A_DECODE_AREA=1 in the environment turns it on. */
static int g_decode_area = -1;

static void quiet(const char *msg, void *u)
{
    (void)msg;
    (void)u;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    opj_dparameters_t p;
    opj_codec_t *c;
    opj_stream_t *s;
    opj_image_t *img = NULL;
    mem_t m;
    unsigned ctl, cut, reduce;
    size_t len;
    if (g_decode_area < 0) {
        const char *e = getenv("FUZZ1A_DECODE_AREA");
        g_decode_area = (e && e[0] == '1') ? 1 : 0;
    }
    if (size < 4 || size > (1u << 22)) {
        return 0;
    }
    ctl = data[0];
    cut = (unsigned)data[1] | ((unsigned)data[2] << 8);
    data += 3;
    size -= 3;
    len = cut == 0 ? size : (size_t)cut % (size + 1);
    if (len == 0) {
        len = 1;
    }
    reduce = (ctl & 7u) % 6u;

    opj_set_default_decoder_parameters(&p);
    if (!(ctl & 0x20)) {
        p.cp_reduce = reduce;
    }
    c = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(c, quiet, NULL);
    opj_set_warning_handler(c, quiet, NULL);
    opj_set_info_handler(c, quiet, NULL);
    opj_setup_decoder(c, &p);
#ifdef OPJ_HAVE_DECODER_LIMITS
    if (ctl & 0x08) {
        opj_decoder_limits_t l;
        opj_decoder_limits_init(&l, OPJ_DECODER_LIMITS_SL_TEXTURE);
        opj_decoder_set_limits(c, &l);
    }
#endif
    s = opj_stream_create(len, OPJ_TRUE);
    m.buf = data;
    m.size = len;
    m.off = 0;
    opj_stream_set_user_data(s, &m, NULL);
    opj_stream_set_user_data_length(s, len);
    opj_stream_set_read_function(s, mem_read);
    opj_stream_set_skip_function(s, mem_skip);
    opj_stream_set_seek_function(s, mem_seek);
    if (!(ctl & 0x20)) {
        opj_set_decoded_resolution_factor(c, reduce);
    }
    opj_decoder_set_strict_mode(c, (ctl & 0x10) ? OPJ_TRUE : OPJ_FALSE);
    if (opj_read_header(s, c, &img)) {
        OPJ_BOOL ok = OPJ_TRUE;
        if (ctl & 0x20) {
            /* the order a header-once caller uses; the result is not checked, on purpose */
            (void)opj_set_decoded_resolution_factor(c, reduce);
        }
        if ((ctl & 0x40) && g_decode_area && img->x1 > img->x0 && img->y1 > img->y0) {
            ok = opj_set_decode_area(c, img, (OPJ_INT32)img->x0, (OPJ_INT32)img->y0,
                                     (OPJ_INT32)(img->x0 + (img->x1 - img->x0 + 1) / 2),
                                     (OPJ_INT32)(img->y0 + (img->y1 - img->y0 + 1) / 2));
        }
        /* bound the work per input (resource exhaustion is the limits' business, not
         * the fuzzer's): at most 32 Mi samples over all components at full resolution */
        if (ok && img->numcomps > 0 &&
                (OPJ_UINT64)(img->x1 - img->x0) * (img->y1 - img->y0) * img->numcomps <=
                ((OPJ_UINT64)1 << 25)) {
            if (opj_decode(c, s, img)) {
                opj_end_decompress(c, s);
            }
        }
    }
    opj_image_destroy(img);
    opj_stream_destroy(s);
    opj_destroy_codec(c);
    return 0;
}
