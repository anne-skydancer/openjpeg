file(READ "${INPUT}" binary HEX)
string(REGEX REPLACE "(..)" "0x\\1," binary "${binary}")
file(WRITE "${OUTPUT}" "/* Generated CUDA fatbinary; do not edit. */\n#if defined(_MSC_VER)\n__declspec(align(16))\n#else\n__attribute__((aligned(16)))\n#endif\nstatic const unsigned char opj_cuda_fatbin[] = {${binary}};\n")
