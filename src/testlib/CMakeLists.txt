
set (c_testlib_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/clockcore.c
    ${CMAKE_CURRENT_SOURCE_DIR}/do_loops.c
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils.c)

#TODO: Check Fortran compiler exists
set (f_testlib_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/ftests_util.F)

if (NOT TARGET testlib)
    add_library (testlib OBJECT)
endif(NOT TARGET testlib)

target_sources (testlib PRIVATE ${c_testlib_srcs} ${f_testlib_srcs})

target_include_directories (testlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (testlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

set_target_properties (testlib PROPERTIES LINKER_LANGUAGE Fortran)
