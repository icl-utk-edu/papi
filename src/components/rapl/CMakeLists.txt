
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_RAPL "Build RAPL component" ON)

if (WITH_RAPL)
    message(STATUS "  - component ${component}")

    set (RAPL_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-rapl.c)

    list (APPEND SUBCOMPONENTS_LIST_LCL rapl)
    set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
    list (APPEND COMPONENTS_LIST_LCL rapl)
    set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

    target_sources (papi PRIVATE ${RAPL_SRCS})

    add_subdirectory(tests)

    set (RAPL_TESTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/rapl_basic.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/rapl_overflow.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
    )

    add_custom_target(install-rapl-tests
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/rapl
        COMMAND ${CMAKE_COMMAND} -E copy ${RAPL_SOURCES} ${EXAMPLES_DIR}/rapl/
    )
    add_dependencies (install-examples    install-rapl-tests)
endif (WITH_RAPL)

