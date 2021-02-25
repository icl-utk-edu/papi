
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_EXAMPLE "Build the Example component" OFF)

if (WITH_EXAMPLE)
    message(STATUS "  - component ${component}")

    set (EXAMPLE_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/example.c)

    list (APPEND SUBCOMPONENTS_LIST_LCL example)
    set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
    list (APPEND COMPONENTS_LIST_LCL example)
    set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

    target_sources (papi PRIVATE ${EXAMPLE_SRCS})

    add_subdirectory(tests)

    set (EXAMPLE_TESTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/example_basic.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/example_multiple_components.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
    )

    add_custom_target(install-example-tests
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/example
        COMMAND ${CMAKE_COMMAND} -E copy ${EXAMPLE_SOURCES} ${EXAMPLES_DIR}/example/
    )
    add_dependencies (install-examples    install-example-tests)
endif (WITH_EXAMPLE)

