
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_POWERCAP_PPC "Build PowerCap PPC component" ON)

if (WITH_POWERCAP_PPC)
    message(STATUS "  - component ${component}")

    find_file(POWERCAP_CURRENT
      powercap_current
      PATHS
      /sys/firmware/opal/powercap)

    if (NOT POWERCAP_CURRENT)
        message (STATUS "/sys/firmware/opal/powercap/powercap_current doesn't exist, disabling POWERCAP_PPC.")
        set (WITH_POWERCAP_PPC OFF)
        return ()
    endif (NOT POWERCAP_CURRENT)

    set (POWERCAP_PPC_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-powercap-ppc.c)

    list (APPEND SUBCOMPONENTS_LIST_LCL powercap_ppc)
    set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
    list (APPEND COMPONENTS_LIST_LCL powercap_ppc)
    set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

    target_sources (papi PRIVATE ${POWERCAP_PPC_SRCS})

    add_subdirectory(tests)

    set (POWERCAP_PPC_TESTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/powercap_basic.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/powercap_limit.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
    )

    add_custom_target(install-powercap-ppc-tests
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/powercap_ppc
        COMMAND ${CMAKE_COMMAND} -E copy ${POWERCAP_PPC_SOURCES} ${EXAMPLES_DIR}/powercap_ppc/
    )
    add_dependencies (install-examples    install-powercap-ppc-tests)
endif (WITH_POWERCAP_PPC)

