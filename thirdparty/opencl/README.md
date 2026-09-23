# Khronos OpenCL headers

Private build dependency from KhronosGroup/OpenCL-Headers, revision
`e6060189f4ebe8b52d885c37af71b9a50c272154`.

Only cl.h, cl_platform.h and cl_version.h are included, unmodified, with their
Apache-2.0 notices and LICENSE. These provide ABI declarations for dynamic loading;
no OpenCL runtime or import library is bundled. The default Windows/Linux build
therefore requires neither an installed OpenCL SDK nor network access.
