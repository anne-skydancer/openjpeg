@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem neo-j2k 1a: builds an OpenJPEG source tree (not the viewer) for the gates of this folder.
rem   rel     MSVC Release, static, the package's CPU flags (/O2 /Ob3 /Oi /Ot /Gy /Gw /fp:precise /arch:AVX2,
rem           no CUDA, no OpenCL, OPJ_PRECISE_FP), codec tools, no LTCG
rem   dll     the same as a DLL (for the bench comparison of real textures)
rem   pkgdll  the same DLL with link-time code generation, as the Windows package builds it
rem   san     clang-cl ASan+UBSan, static, codec tools, NDEBUG as shipped. Not for timing.
rem   sandbg  the same with assertions on (no NDEBUG): they check, among others, that
rem           opj_tcd_decode_need covers what opj_tcd_init_tile builds
rem   census  the rel flags with OPJ_ALLOC_CENSUS, library only, for a tree patched by
rem           make_variant_src.py census (the probe counts the allocation calls)
rem usage: build_lib.cmd rel|dll|pkgdll|san|sandbg|census <src dir> <build dir> [extra C flags]
rem   VCVARS: vcvars64.bat of Visual Studio (default: VS 2026 Community)
rem   OPJ_LLVM_RT (san, sandbg): a folder without spaces holding the clang runtime import libraries
rem     (clang_rt.asan_dynamic-x86_64.lib, clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
setlocal
if "%VCVARS%"=="" set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul 2>nul
set KIND=%~1
if not "%KIND%"=="rel" if not "%KIND%"=="dll" if not "%KIND%"=="pkgdll" if not "%KIND%"=="san" if not "%KIND%"=="sandbg" if not "%KIND%"=="census" (
  echo usage: build_lib.cmd rel^|dll^|pkgdll^|san^|sandbg^|census ^<src dir^> ^<build dir^> [extra C flags]
  exit /b 1
)
set SRC=%~2
set BLD=%~3
set EXTRA=%~4
set CPU=/MD /O2 /Ob3 /Oi /Ot /Gy /Gw /fp:precise /arch:AVX2
set OPTS=-DOPJ_ENABLE_CUDA=OFF -DOPJ_ENABLE_OPENCL=OFF -DOPJ_PRECISE_FP=ON -DBUILD_TESTING=OFF
if "%KIND%"=="rel" (
  cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_C_FLAGS="%CPU% %EXTRA%" -DCMAKE_C_FLAGS_RELEASE="/DNDEBUG" ^
    -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DBUILD_CODEC=ON %OPTS% || exit /b 1
) else if "%KIND%"=="dll" (
  cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_C_FLAGS="%CPU% %EXTRA%" -DCMAKE_C_FLAGS_RELEASE="/DNDEBUG" ^
    -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DBUILD_CODEC=OFF %OPTS% || exit /b 1
) else if "%KIND%"=="pkgdll" (
  cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_C_FLAGS="%CPU% %EXTRA%" -DCMAKE_C_FLAGS_RELEASE="/DNDEBUG" ^
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=ON ^
    -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DBUILD_CODEC=OFF %OPTS% || exit /b 1
) else if "%KIND%"=="census" (
  cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_C_FLAGS="%CPU% /DOPJ_ALLOC_CENSUS %EXTRA%" -DCMAKE_C_FLAGS_RELEASE="/DNDEBUG" ^
    -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DBUILD_CODEC=OFF %OPTS% || exit /b 1
  cmake --build "%BLD%" --target openjp2 -- -j 6 || exit /b 1
  exit /b 0
) else (
  if "%OPJ_LLVM_RT%"=="" (
    echo set OPJ_LLVM_RT to the folder of the clang runtime import libraries
    exit /b 1
  )
  set NDBG=/DNDEBUG
  if "%KIND%"=="sandbg" set NDBG=
)
if "%KIND%"=="san" goto sanitizer
if "%KIND%"=="sandbg" goto sanitizer
goto build
:sanitizer
cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DCMAKE_TRY_COMPILE_CONFIGURATION=RelWithDebInfo -DCMAKE_POLICY_DEFAULT_CMP0091=NEW ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_C_FLAGS="/Zi /Oy- -mavx2 -fsanitize=address,undefined -fno-sanitize=vptr,function -fno-sanitize-recover=undefined %EXTRA%" ^
  -DCMAKE_CXX_FLAGS="/Zi /Oy- -mavx2 -fsanitize=address,undefined -fno-sanitize=vptr" ^
  -DCMAKE_C_FLAGS_RELWITHDEBINFO="/O1 %NDBG%" -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="/O1 %NDBG%" ^
  -DCMAKE_EXE_LINKER_FLAGS="/DEBUG -libpath:%OPJ_LLVM_RT% clang_rt.asan_dynamic-x86_64.lib -wholearchive:clang_rt.asan_dynamic_runtime_thunk-x86_64.lib" ^
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DBUILD_CODEC=ON %OPTS% || exit /b 1
:build
cmake --build "%BLD%" -- -j 6 || exit /b 1
