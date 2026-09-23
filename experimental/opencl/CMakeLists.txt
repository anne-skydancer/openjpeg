find_package(Python3 COMPONENTS Interpreter REQUIRED)
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
