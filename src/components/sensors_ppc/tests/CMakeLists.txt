if (NOT WITH_TESTS)
      #    return ()
endif (NOT WITH_TESTS)

add_executable(sensors_ppc_basic
    ${CMAKE_CURRENT_SOURCE_DIR}/sensors_ppc_basic.c)

target_link_libraries(sensors_ppc_basic
    PRIVATE papi)

add_test(NAME test_sensors_ppc_basic
    COMMAND sensors_ppc_basic
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

