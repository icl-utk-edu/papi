
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_APPIO "Build APPIO component" ON)

if (WITH_APPIO)
    message(STATUS "  - component ${component}")

    set (APPIO_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/appio.c)

    list (APPEND SUBCOMPONENTS_LIST_LCL appio)
    set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
    list (APPEND COMPONENTS_LIST_LCL appio)
    set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

    target_sources (papi PRIVATE ${APPIO_SRCS})

    set (APPIO_TESTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_list_events.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_test_blocking.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_test_fread_fwrite.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_test_pthreads.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_test_read_write.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_test_recv.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_test_seek.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_test_select.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_test_socket.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_values_by_code.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/appio_values_by_name.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/init_fini.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/fileop.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/gengnuplot.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/gnu3d.dem
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/gnuplot.dem
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/gnuplotps.dem
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/Gnuplot.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/iozone.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/iozone_visualizer.pl
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/libasync.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/libbif.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/makefile
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/pit_server.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/report.pl
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/iozone/spec.in
    )

    add_custom_target(install-appio-tests
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/appio
	COMMAND ${CMAKE_COMMAND} -E copy ${APPIO_TESTS_SOURCES} ${EXAMPLES_DIR}/appio/
    )

    add_dependencies (install-examples    install-appio-tests)
endif (WITH_APPIO)
