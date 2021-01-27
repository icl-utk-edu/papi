
cmake_minimum_required (VERSION 3.12)

project (FAKE C)

cmake_policy (SET CMP0079 NEW)

file (WRITE ${CMAKE_BINARY_DIR}/fake_util.c "
#include <stdio.h>
#include <stdlib.h>

#include \"perfmon/perf_event.h\"

int
main (int argc, char *argv[])
{
    int ret = perf_event_open(NULL, 0, 0, 0, 0x00);
    return 0;
}")

set (fake_sources ${CMAKE_BINARY_DIR}/fake_util.c)

add_executable (fake ${fake_sources})
target_link_libraries (fake PRIVATE pfm4_lib)

set (pfm_root "libpfm4")
set (pfm_incdir "libpfm4/include")
set (pfm_libdir "libpfm4/lib")

#find_library (pfm4_lib
#    NAMES libpfm.a # add any other relevant name for thet static version
#    PATHS
#    # TODO: Add system paths, and pfm_libdir / pfm_root/lib
#    ${CMAKE_SOURCE_DIR}/lib
#    ${CMAKE_BINARY_DIR}/lib
#    ${pfm_root}/lib
#    ${pfm_libdir})

if (NOT TARGET pfm4_lib)
    message (STATUS "SOURCE_DIR         ${CMAKE_CURRENT_SOURCE_DIR}")
    message (STATUS "BINARY_DIR         ${CMAKE_CURRENT_BINARY_DIR}")

    find_program(MAKE_EXE NAMES make nmake gmake)
    # pfm4_build -> pfm4_install_clean -> pfm4_lib -> papi

    set (bin_lib ${CMAKE_CURRENT_BINARY_DIR}/lib)
    set (src_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib)

    add_custom_command (
        OUTPUT
        ${src_lib}/libpfm.so
        ${src_lib}/libpfm.a
        COMMAND ${MAKE_EXE} clean
        COMMAND ${MAKE_EXE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_custom_target (
        pfm4_build
        DEPENDS
        ${src_lib}/libpfm.so
        ${src_lib}/libpfm.a)

    add_custom_command (
        OUTPUT ${bin_lib}/libpfm.so ${bin_lib}/libpfm.a
        # install the libs
        COMMAND ${CMAKE_COMMAND} -E make_directory ${bin_lib}
        COMMAND ${CMAKE_COMMAND} -E copy ${src_lib}/libpfm.so ${bin_lib}
        COMMAND ${CMAKE_COMMAND} -E copy ${src_lib}/libpfm.a  ${bin_lib}
        # copy the headers?
        COMMAND ${MAKE_EXE} clean
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_custom_target (
        pfm4_install_clean
        # that should trigger the build
        DEPENDS
        ${bin_lib}/libpfm.so
        ${bin_lib}/libpfm.a)

    add_dependencies(pfm4_install_clean pfm4_build)

    add_library(pfm4_lib SHARED IMPORTED)
    set_target_properties (pfm4_lib PROPERTIES
        IMPORTED_LOCATION ${bin_lib}/libpfm.so)

    target_include_directories(pfm4_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_dependencies (fake pfm4_install_clean)

endif (NOT TARGET pfm4_lib)

target_link_libraries(fake PRIVATE pfm4_lib)



