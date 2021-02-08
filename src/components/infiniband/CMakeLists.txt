
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_INFINIBAND "Build Infiniband component" ON)

if (WITH_INFINIBAND)
    message(STATUS "  - component ${component}")

    find_file(SYS_CLASS_INFINIBAND
        infiniband
        PATHS
        /sys/class)

    if (NOT SYS_CLASS_INFINIBAND)
        message (STATUS "/sys/class/infiniband doesn't exist, disabling Infiniband.")
        set (WITH_INFINIBAND OFF)
        return ()
    endif (NOT SYS_CLASS_INFINIBAND)

    set (INFINIBAND_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-infiniband.c)

    list (APPEND SUBCOMPONENTS_LIST_LCL infiniband)
    set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
    list (APPEND COMPONENTS_LIST_LCL infiniband)
    set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

    target_sources (papi PRIVATE ${INFINIBAND_SRCS})

    add_subdirectory(tests)

    set (INFINIBAND_TESTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/infiniband_list_events.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/infiniband_values_by_code.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/MPI_test_infiniband_events.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
    )

    add_custom_target(install-infiniband-tests
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/infiniband
        COMMAND ${CMAKE_COMMAND} -E copy ${INFINIBAND_TESTS_SOURCES} ${EXAMPLES_DIR}/infiniband/
    )

    add_dependencies (install-examples    install-infiniband-tests)
endif (WITH_INFINIBAND)

