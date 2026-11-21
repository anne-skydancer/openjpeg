/* A resolution count the tile cannot hold must be refused with an error,
 * not shifted out of range (signed 1 << 31 in opj_j2k_encoding_validation).
 * Runs the encoder set-up at 32 and 33 resolutions; meaningful under UBSan. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openjpeg.h"

static void quiet(const char *m, void *d) { (void)m; (void)d; }

static int try_encode(int numres)
{
    opj_cparameters_t p;
    opj_image_cmptparm_t c;
    opj_image_t *img;
    opj_codec_t *codec;
    opj_stream_t *s;
    OPJ_BOOL ok_setup, ok_start;
    int i, rc = 0;

    opj_set_default_encoder_parameters(&p);
    p.numresolution = numres;
    memset(&c, 0, sizeof(c));
    c.dx = c.dy = 1;
    c.w = c.h = 64;
    c.prec = 8;
    img = opj_image_create(1, &c, OPJ_CLRSPC_GRAY);
    if (!img) return 1;
    img->x0 = img->y0 = 0;
    img->x1 = img->y1 = 64;
    for (i = 0; i < 64 * 64; i++) img->comps[0].data[i] = i & 255;

    codec = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec, quiet, NULL);
    opj_set_warning_handler(codec, quiet, NULL);
    ok_setup = opj_setup_encoder(codec, &p, img);
    s = opj_stream_default_create(OPJ_FALSE);
    /* Either set-up or start must refuse; neither may succeed. */
    ok_start = OPJ_FALSE;
    if (ok_setup) {
        /* write to a file that is removed afterwards */
        opj_stream_destroy(s);
        s = opj_stream_create_default_file_stream("testresolutions.tmp", OPJ_FALSE);
        ok_start = s ? opj_start_compress(codec, img, s) : OPJ_FALSE;
    }
    if (ok_start) {
        fprintf(stderr, "numresolution %d was accepted\n", numres);
        rc = 1;
    }
    opj_stream_destroy(s);
    opj_destroy_codec(codec);
    opj_image_destroy(img);
    remove("testresolutions.tmp");
    return rc;
}

int main(void)
{
    return try_encode(32) | try_encode(33);
}
