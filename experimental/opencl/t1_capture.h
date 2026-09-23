/* SPDX-License-Identifier: BSD-2-Clause
 * Development-only capture, included in t1.c after OpenJPEG internal headers.
 * Set OPJ_T1_CAPTURE_FILE to a dedicated per-process JSONL path. Each record
 * is serialized under the decoder's existing job mutex. No global counter or
 * library API is added. Capture errors fail this experimental decode.
 */
static OPJ_BOOL opj_capture_t1(const opj_tcd_cblk_dec_t *block,
                             const opj_tcd_band_t *band,
                             const opj_tcd_tilecomp_t *tile,
                             const opj_tccp_t *coding, OPJ_UINT32 resolution,
                             const OPJ_INT32 *coefficients,
                             OPJ_UINT32 width, OPJ_UINT32 height,
                             opj_mutex_t *mutex)
{
    const char *path = getenv("OPJ_T1_CAPTURE_FILE");
    FILE *out;
    OPJ_UINT32 i, j;
    int failed;
    if (!path || !*path) {
        return OPJ_TRUE;
    }
    if (mutex) {
        opj_mutex_lock(mutex);
    }
    out = fopen(path, "ab");
    if (!out) {
        if (mutex) { opj_mutex_unlock(mutex); }
        return OPJ_FALSE;
    }
    fprintf(out, "{\"version\":1,\"width\":%u,\"height\":%u,"
            "\"x0\":%d,\"y0\":%d,\"component\":%u,\"resolution\":%u,"
            "\"orientation\":%u,\"numbps\":%u,\"style\":%u,\"roi\":%d,"
            "\"qmfbid\":%u,\"stepsize\":%.9g,\"corrupted\":%d,\"segments\":[",
            width, height, block->x0, block->y0, tile->compno, resolution,
            band->bandno, block->numbps, coding->cblksty, coding->roishift,
            coding->qmfbid, (double)band->stepsize, block->corrupted);
    for (i = 0; i < block->real_num_segs; ++i) {
        fprintf(out, "%s[%u,%u]", i ? "," : "",
                block->segs[i].len, block->segs[i].real_num_passes);
    }
    fprintf(out, "],\"bytes\":\"");
    for (i = 0; i < block->numchunks; ++i) {
        for (j = 0; j < block->chunks[i].len; ++j) {
            fprintf(out, "%02x", (unsigned)block->chunks[i].data[j]);
        }
    }
    fprintf(out, "\",\"coefficients\":[");
    for (i = 0; i < width * height; ++i) {
        fprintf(out, "%s%d", i ? "," : "", coefficients[i]);
    }
    fprintf(out, "]}\n");
    failed = ferror(out);
    if (fclose(out)) { failed = 1; }
    if (mutex) { opj_mutex_unlock(mutex); }
    return failed ? OPJ_FALSE : OPJ_TRUE;
}
