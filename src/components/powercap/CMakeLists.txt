
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_POWERCAP "Build PowerCap component" ON)

if (WITH_POWERCAP)
    message(STATUS "  - component ${component}")

    set (POWERCAP_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-powercap.c)

    list (APPEND SUBCOMPONENTS_LIST_LCL powercap)
    set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
    list (APPEND COMPONENTS_LIST_LCL powercap)
    set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

    target_sources (papi PRIVATE ${POWERCAP_SRCS})

    add_subdirectory(tests)

    set (POWERCAP_TESTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/powercap_basic.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/powercap_limit.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
    )

    add_custom_target(install-powercap-tests
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/powercap
        COMMAND ${CMAKE_COMMAND} -E copy ${POWERCAP_SOURCES} ${EXAMPLES_DIR}/powercap/
    )
    add_dependencies (install-examples    install-powercap-tests)
endif (WITH_POWERCAP)

