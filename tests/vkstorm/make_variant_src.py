# SPDX-License-Identifier: BSD-2-Clause
# Patches a copy of a source tree (best a git archive of the commit under test) into a gate variant:
#   census  opj_malloc.h sends the library's allocation calls to the counters of probe.c (build the
#           library with build_lib.cmd census and the probe with /DOPJ_ALLOC_CENSUS). One opj_bio_t is
#           allocated per packet header read, so the census line of bio.c counts the packets visited.
#   noexit  the packet loop of opj_t2_decode_packets keeps visiting packets after a tile's data is
#           exhausted (the exit of CVE-2023-39327 is off): its cut decodes must equal those of the
#           tree it was made from, which shows that the exit changes no output.
#   pktlog  opj_t2_decode_packets prints "PKT <tile> <layer> <res> <comp> <prec> <end>" to stderr for
#           every packet visited, <end> being the bytes of the tile's data read so far: the input of
#           layer_cuts.py, which finds the layer boundaries of a stream whose headers are in PPM or PPT.
# usage: py -3 -I make_variant_src.py <src dir> census|noexit|pktlog
import sys
from pathlib import Path

src, kind = Path(sys.argv[1]), sys.argv[2]
lib = src / "src" / "lib" / "openjp2"


def patch(path, old, new):
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{path}: the anchor is not found exactly once")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="")


if kind == "census":
    patch(lib / "opj_malloc.h", "#if defined(__GNUC__) && !defined(OPJ_SKIP_POISON)",
          """#ifdef OPJ_ALLOC_CENSUS
/* gate variant only: count allocation calls per call site (the counters live in the probe) */
void *opj_census_malloc(size_t s, const char *f, int l);
void *opj_census_calloc(size_t n, size_t s, const char *f, int l);
void *opj_census_realloc(void *p, size_t s, const char *f, int l);
void *opj_census_aligned_malloc(size_t s, const char *f, int l);
void *opj_census_aligned_32_malloc(size_t s, const char *f, int l);
void *opj_census_aligned_realloc(void *p, size_t s, const char *f, int l);
#ifndef OPJ_CENSUS_SELF
#define opj_malloc(s) opj_census_malloc((s), __FILE__, __LINE__)
#define opj_calloc(n, s) opj_census_calloc((n), (s), __FILE__, __LINE__)
#define opj_realloc(p, s) opj_census_realloc((p), (s), __FILE__, __LINE__)
#define opj_aligned_malloc(s) opj_census_aligned_malloc((s), __FILE__, __LINE__)
#define opj_aligned_32_malloc(s) opj_census_aligned_32_malloc((s), __FILE__, __LINE__)
#define opj_aligned_realloc(p, s) opj_census_aligned_realloc((p), (s), __FILE__, __LINE__)
#endif
#endif

#if defined(__GNUC__) && !defined(OPJ_SKIP_POISON)""")
    patch(lib / "opj_malloc.c", "#define OPJ_SKIP_POISON\n",
          "#define OPJ_SKIP_POISON\n#define OPJ_CENSUS_SELF\n")
elif kind == "noexit":
    patch(lib / "t2.c", "            if (p_max_len == 0 && !l_cp->strict &&\n",
          "            if (0 && p_max_len == 0 && !l_cp->strict &&\n")
elif kind == "pktlog":
    patch(lib / "t2.c", "            p_max_len -= l_nb_bytes_read;\n",
          "            p_max_len -= l_nb_bytes_read;\n"
          "            fprintf(stderr, \"PKT %u %u %u %u %u %u\\n\", p_tile_no, l_current_pi->layno,\n"
          "                    l_current_pi->resno, l_current_pi->compno, l_current_pi->precno,\n"
          "                    (OPJ_UINT32)(l_current_data - p_src));\n")
else:
    raise SystemExit("usage: make_variant_src.py <src dir> census|noexit|pktlog")
print(f"{src}: {kind} variant")
