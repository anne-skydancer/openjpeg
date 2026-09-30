# CUDA Driver API only at runtime; no cudart or nvJPEG dependency.
find_program(OPJ_NVCC nvcc HINTS "${OPJ_CUDA_ROOT}/bin" "$ENV{CUDA_PATH}/bin" REQUIRED)
find_path(OPJ_CUDA_INCLUDE_DIR cuda.h HINTS "${OPJ_CUDA_ROOT}/include" "$ENV{CUDA_PATH}/include" REQUIRED)
set(OPJ_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90;120" CACHE STRING "CUDA kernel architectures")
set(OPJ_CUDA_PTX_ARCHITECTURE "50" CACHE STRING "Portable CUDA PTX architecture")
set(cuda_arch_flags)
foreach(arch ${OPJ_CUDA_ARCHITECTURES})
  list(APPEND cuda_arch_flags "--generate-code=arch=compute_${arch},code=sm_${arch}")
endforeach()
list(APPEND cuda_arch_flags "--generate-code=arch=compute_${OPJ_CUDA_PTX_ARCHITECTURE},code=compute_${OPJ_CUDA_PTX_ARCHITECTURE}")
set(cuda_host_flags)
if(MSVC)
  get_filename_component(cuda_host_dir "${CMAKE_C_COMPILER}" DIRECTORY)
  list(APPEND cuda_host_flags --compiler-bindir "${cuda_host_dir}")
endif()
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/decode.fatbin"
  COMMAND "${OPJ_NVCC}" --fatbin -O3 --std=c++14 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true
    ${cuda_host_flags} ${cuda_arch_flags} -Xptxas=-v
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda/decode.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/decode.fatbin"
  DEPENDS cuda/decode.cu VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels.h"
  COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/decode.fatbin"
    "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels.h" -P "${CMAKE_CURRENT_SOURCE_DIR}/cuda/Embed.cmake"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/decode.fatbin" cuda/Embed.cmake VERBATIM)
