@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem neo-j2k 1a: builds the probe (probe.c) or the libFuzzer target (fuzz_partial.c) against a library
rem built with build_lib.cmd.
rem usage: build_tools.cmd probe rel|san <src dir> <build dir> <out exe> [extra defines, e.g. /DPROBE_LIMITS]
rem        build_tools.cmd fuzz <src dir> <build dir> <out exe>
rem   probe rel links the rel build; probe san links a san or sandbg build. PROBE_LIMITS: the library has
rem   opj_decoder_set_limits (the probe then honours PROBE_PROFILE=sl).
rem   fuzz builds its own library in <build dir> (clang-cl, ASan+UBSan, static CRT as libFuzzer needs).
rem   VCVARS and OPJ_LLVM_RT as for build_lib.cmd (OPJ_LLVM_RT also needs clang_rt.fuzzer_no_main-x86_64.lib
rem   and clang_rt.asan_static_runtime_thunk-x86_64.lib for the fuzzer).
setlocal
if "%VCVARS%"=="" set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul 2>nul
set WHAT=%~1
if "%WHAT%"=="fuzz" goto fuzz
set KIND=%~2
set SRC=%~3
set BLD=%~4
set OUT=%~5
set EXTRA=%~6
set INC=/I"%SRC%\src\lib\openjp2" /I"%BLD%\src\lib\openjp2"
if "%KIND%"=="san" (
  clang-cl /nologo /MD /Zi /O1 /Oy- -mavx2 -fsanitize=address,undefined -fno-sanitize=vptr,function -fno-sanitize-recover=undefined /DOPJ_STATIC %EXTRA% %INC% "%~dp0probe.c" /Fe"%OUT%" /Fo"%OUT%.obj" /link /DEBUG "%BLD%\bin\openjp2.lib" psapi.lib -libpath:"%OPJ_LLVM_RT%" clang_rt.asan_dynamic-x86_64.lib -wholearchive:clang_rt.asan_dynamic_runtime_thunk-x86_64.lib || exit /b 1
) else (
  cl /nologo /MD /O2 /arch:AVX2 /fp:precise /DOPJ_STATIC %EXTRA% %INC% "%~dp0probe.c" /Fe"%OUT%" /Fo"%OUT%.obj" /link "%BLD%\bin\openjp2.lib" psapi.lib || exit /b 1
)
exit /b 0
:fuzz
set SRC=%~2
set BLD=%~3
set OUT=%~4
cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DCMAKE_TRY_COMPILE_CONFIGURATION=RelWithDebInfo -DCMAKE_POLICY_DEFAULT_CMP0091=NEW ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_C_FLAGS="/Zi /Oy- -mavx2 -fsanitize=address,undefined,fuzzer-no-link -fno-sanitize=vptr,function -fno-sanitize-recover=undefined" ^
  -DCMAKE_C_FLAGS_RELWITHDEBINFO="/O1 /DNDEBUG" ^
  -DCMAKE_EXE_LINKER_FLAGS="/DEBUG -libpath:%OPJ_LLVM_RT% clang_rt.asan_dynamic-x86_64.lib -wholearchive:clang_rt.asan_static_runtime_thunk-x86_64.lib clang_rt.fuzzer_no_main-x86_64.lib" ^
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DBUILD_CODEC=OFF -DBUILD_TESTING=OFF ^
  -DOPJ_ENABLE_CUDA=OFF -DOPJ_ENABLE_OPENCL=OFF -DOPJ_PRECISE_FP=ON || exit /b 1
cmake --build "%BLD%" --target openjp2 -- -j 6 || exit /b 1
clang-cl /nologo /MT /Zi /O1 /Oy- -mavx2 -fsanitize=address,undefined,fuzzer -fno-sanitize=vptr,function -fno-sanitize-recover=undefined /DOPJ_STATIC /I"%SRC%\src\lib\openjp2" /I"%BLD%\src\lib\openjp2" "%~dp0fuzz_partial.c" /Fe"%OUT%" /Fo"%OUT%.obj" /link /DEBUG "%BLD%\bin\openjp2.lib" -libpath:"%OPJ_LLVM_RT%" || exit /b 1
