find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_executable(bench_decode bench_decode.c)
target_include_directories(bench_decode PRIVATE "${OPENJPEG_SOURCE_DIR}/src/lib/openjp2"
  "${OPENJPEG_BINARY_DIR}/src/lib/openjp2")
target_link_libraries(bench_decode openjp2)
find_package(Threads REQUIRED)
add_executable(bench_workers bench_workers.c)
target_include_directories(bench_workers PRIVATE "${OPENJPEG_SOURCE_DIR}/src/lib/openjp2"
  "${OPENJPEG_BINARY_DIR}/src/lib/openjp2")
target_link_libraries(bench_workers openjp2 Threads::Threads)
add_executable(bench_encode bench_workers.c)
target_compile_definitions(bench_encode PRIVATE OPJ_BENCH_ENCODING)
target_include_directories(bench_encode PRIVATE "${OPENJPEG_SOURCE_DIR}/src/lib/openjp2"
  "${OPENJPEG_BINARY_DIR}/src/lib/openjp2")
target_link_libraries(bench_encode openjp2 Threads::Threads)
add_executable(test_opencl_reentrant test_reentrant.c)
target_include_directories(test_opencl_reentrant PRIVATE "${OPENJPEG_SOURCE_DIR}/src/lib/openjp2"
  "${OPENJPEG_BINARY_DIR}/src/lib/openjp2")
target_link_libraries(test_opencl_reentrant openjp2)
option(OPJ_CAPTURE_T1_REFERENCES "Enable development-only Tier-1 reference capture" OFF)
if(OPJ_CAPTURE_T1_REFERENCES)
  target_compile_definitions(openjp2 PRIVATE OPJ_CAPTURE_T1_REFERENCES)
  if(TARGET openjp2_static)
    target_compile_definitions(openjp2_static PRIVATE OPJ_CAPTURE_T1_REFERENCES)
  endif()
endif()
set(OPJ_OPENCL_TEST_DEVICE "" CACHE STRING "Explicit OpenCL test device-name substring")
set(OPJ_OPENCL_TEST_DRIVER "" CACHE STRING "Optional OpenCL test driver-version substring")
if(NOT OPJ_OPENCL_TEST_DEVICE)
  message(FATAL_ERROR "Set OPJ_OPENCL_TEST_DEVICE explicitly; enumerate with test_idwt53.py --list")
endif()
set(driver_args)
if(OPJ_OPENCL_TEST_DRIVER)
  list(APPEND driver_args --driver "${OPJ_OPENCL_TEST_DRIVER}")
endif()
add_test(NAME opencl_idwt53
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_idwt53.py"
          --device "${OPJ_OPENCL_TEST_DEVICE}" ${driver_args})
set_tests_properties(opencl_idwt53 PROPERTIES LABELS "experimental;opencl" TIMEOUT 60 RUN_SERIAL TRUE)
if(OPJ_ENABLE_OPENCL AND BUILD_CODEC)
  add_test(NAME opencl_encode_workers
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_encode_workers.py"
      --bin "$<TARGET_FILE_DIR:bench_encode>" --out "${CMAKE_CURRENT_BINARY_DIR}/encode-workers"
      --device "${OPJ_OPENCL_TEST_DEVICE}" ${driver_args})
  set_tests_properties(opencl_encode_workers PROPERTIES LABELS "experimental;opencl" TIMEOUT 240 RUN_SERIAL TRUE)
  add_test(NAME opencl_encode
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_encode.py"
      --bin "$<TARGET_FILE_DIR:opj_compress>" --out "${CMAKE_CURRENT_BINARY_DIR}/encode"
      --device "${OPJ_OPENCL_TEST_DEVICE}" ${driver_args})
  set_tests_properties(opencl_encode PROPERTIES LABELS "experimental;opencl" TIMEOUT 240 RUN_SERIAL TRUE)
  add_test(NAME opencl_complete
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_complete.py"
      --bin "$<TARGET_FILE_DIR:opj_decompress>" --out "${CMAKE_CURRENT_BINARY_DIR}/complete"
      --device "${OPJ_OPENCL_TEST_DEVICE}" ${driver_args})
  set_tests_properties(opencl_complete PROPERTIES LABELS "experimental;opencl" TIMEOUT 180 RUN_SERIAL TRUE)
  add_test(NAME opencl_fallback
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_fallback.py"
      --bin "$<TARGET_FILE_DIR:opj_decompress>" --out "${CMAKE_CURRENT_BINARY_DIR}/fallback"
      --device "${OPJ_OPENCL_TEST_DEVICE}" ${driver_args})
  set_tests_properties(opencl_fallback PROPERTIES LABELS "experimental;opencl" TIMEOUT 120 RUN_SERIAL TRUE)
  add_test(NAME opencl_reuse
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_reuse.py"
      --bin "$<TARGET_FILE_DIR:bench_decode>" --out "${CMAKE_CURRENT_BINARY_DIR}/reuse"
      --device "${OPJ_OPENCL_TEST_DEVICE}" ${driver_args})
  set_tests_properties(opencl_reuse PROPERTIES LABELS "experimental;opencl" TIMEOUT 120 RUN_SERIAL TRUE)
  add_test(NAME opencl_workers
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_workers.py"
      --bin "$<TARGET_FILE_DIR:bench_workers>" --out "${CMAKE_CURRENT_BINARY_DIR}/workers"
      --device "${OPJ_OPENCL_TEST_DEVICE}" ${driver_args})
  set_tests_properties(opencl_workers PROPERTIES LABELS "experimental;opencl" TIMEOUT 180 RUN_SERIAL TRUE)
endif()
add_test(NAME opencl_t1_validation
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_t1_validation.py")
if(OPJ_CAPTURE_T1_REFERENCES AND BUILD_CODEC)
  add_test(NAME opencl_t1_capture
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/make_t1_corpus.py"
      --bin "$<TARGET_FILE_DIR:opj_decompress>" --out "${CMAKE_CURRENT_BINARY_DIR}/corpus")
  set_tests_properties(opencl_t1_capture PROPERTIES FIXTURES_SETUP t1_corpus TIMEOUT 120 RUN_SERIAL TRUE)
  add_test(NAME opencl_t1_decode
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_t1.py"
      --corpus "${CMAKE_CURRENT_BINARY_DIR}/corpus/blocks.jsonl"
      --device "${OPJ_OPENCL_TEST_DEVICE}" ${driver_args})
  set_tests_properties(opencl_t1_decode PROPERTIES FIXTURES_REQUIRED t1_corpus
    LABELS "experimental;opencl" TIMEOUT 60 RUN_SERIAL TRUE)
endif()
