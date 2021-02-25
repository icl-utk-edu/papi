if (NOT WITH_TESTS)
      #    return ()
endif (NOT WITH_TESTS)

add_executable(rapl_basic
    ${CMAKE_CURRENT_SOURCE_DIR}/rapl_basic.c)
target_link_libraries (rapl_basic
    PRIVATE papi)
target_compile_definitions(rapl_basic
    PRIVATE BASIC_TEST=1)
target_include_directories(rapl_basic
    PRIVATE ${CMAKE_SOURCE_DIR}/testlib)
target_link_libraries (rapl_basic
        PRIVATE testlib papi)
add_test(NAME test_rapl_basic
    COMMAND rapl_basic
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(rapl_busy
    ${CMAKE_CURRENT_SOURCE_DIR}/rapl_basic.c)
target_link_libraries (rapl_busy
    PRIVATE papi)
target_compile_definitions(rapl_busy
    PRIVATE BUSY_TEST=1)
target_include_directories(rapl_busy
    PRIVATE ${CMAKE_SOURCE_DIR}/testlib)
target_link_libraries (rapl_busy
        PRIVATE testlib papi)
add_test(NAME test_rapl_busy
    COMMAND rapl_busy
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(rapl_wraparound
    ${CMAKE_CURRENT_SOURCE_DIR}/rapl_basic.c)
target_link_libraries (rapl_wraparound
    PRIVATE papi)
target_compile_definitions(rapl_wraparound
  PRIVATE WRAP_TEST=1)
target_include_directories(rapl_wraparound
    PRIVATE ${CMAKE_SOURCE_DIR}/testlib)
target_link_libraries (rapl_wraparound
        PRIVATE testlib papi)
add_test(NAME test_rapl_wraparound
    COMMAND rapl_wraparound
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_papi_component_test (rapl_overflow)
