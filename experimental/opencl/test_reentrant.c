/* SPDX-License-Identifier: BSD-2-Clause
 * A user info callback may decode a different image with another codec. The
 * completed job must release its GPU slot before entering that callback.
 */
#include "openjpeg.h"
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *path;
    opj_image_t *nested;
    unsigned entered, nested_gpu;
} test_state;

static opj_image_t *decode(const char *path,opj_msg_callback callback,void *data)
{
    opj_dparameters_t parameters;
    opj_codec_t *codec=opj_create_decompress(OPJ_CODEC_J2K);
    opj_stream_t *stream=opj_stream_create_default_file_stream(path,OPJ_TRUE);
    opj_image_t *image=NULL;
    opj_set_default_decoder_parameters(&parameters);
    if(codec) opj_set_info_handler(codec,callback,data);
    if(!codec || !stream || !opj_setup_decoder(codec,&parameters) ||
       !opj_codec_set_threads(codec,1) || !opj_read_header(stream,codec,&image) ||
       !opj_decode(codec,stream,image) || !opj_end_decompress(codec,stream)) {
        if(image) opj_image_destroy(image);
        image=NULL;
    }
    if(stream) opj_stream_destroy(stream);
    if(codec) opj_destroy_codec(codec);
    return image;
}

static void nested_info(const char *message,void *data)
{
    test_state *state=(test_state*)data;
    if(strstr(message,"OpenCL decoded tile")) ++state->nested_gpu;
}

static void outer_info(const char *message,void *data)
{
    test_state *state=(test_state*)data;
    if(!state->entered && strstr(message,"OpenCL decoded tile")) {
        state->entered=1;
        state->nested=decode(state->path,nested_info,state);
    }
}

int main(int argc,char **argv)
{
    test_state state={0};
    opj_image_t *outer;
    unsigned c;
    int ok;
    if(argc!=2) return 2;
    state.path=argv[1]; outer=decode(state.path,outer_info,&state);
    ok=outer && state.nested && state.entered && state.nested_gpu;
    if(ok) ok=outer->numcomps==state.nested->numcomps;
    for(c=0;ok && c<outer->numcomps;c++) {
        opj_image_comp_t *a=&outer->comps[c],*b=&state.nested->comps[c];
        ok=a->w==b->w && a->h==b->h && a->prec==b->prec && a->sgnd==b->sgnd &&
           !memcmp(a->data,b->data,(size_t)a->w*a->h*sizeof(OPJ_INT32));
    }
    if(outer) opj_image_destroy(outer);
    if(state.nested) opj_image_destroy(state.nested);
    if(!ok) return 1;
    puts("PASS: nested GPU decode from callback returns exact pixels with one slot");
    fflush(stdout);
    return 0;
}
