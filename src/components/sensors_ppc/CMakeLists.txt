
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_SENSORS_PPC "Build Sensors PPC component" ON)

if (WITH_SENSORS_PPC)
    message(STATUS "  - component ${component}")

    find_file(OCC_INBAND_SENSORS
      occ_inband_sensors
      PATHS
      /sys/firmware/opal/exports)

    if (NOT OCC_INBAND_SENSORS)
        message (STATUS "/sys/firmware/opal/exports/occ_inband_sensors doesn't exist, disabling SENSORS_PPC.")
        set (WITH_SENSORS_PPC OFF)
        return ()
    endif (NOT OCC_INBAND_SENSORS)

    set (SENSORS_PPC_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-sensors-ppc.c)

    list (APPEND SUBCOMPONENTS_LIST_LCL sensors_ppc)
    set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
    list (APPEND COMPONENTS_LIST_LCL sensors_ppc)
    set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

    target_sources (papi PRIVATE ${SENSORS_PPC_SRCS})

    add_subdirectory(tests)

    set (SENSORS_PPC_TESTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/sensors_ppc_basic.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
    )

    add_custom_target(install-sensors-ppc-tests
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/sensors_ppc
	COMMAND ${CMAKE_COMMAND} -E copy ${SENSORS_PPC_TESTS_SOURCES} ${EXAMPLES_DIR}/sensors_ppc/
    )

    add_dependencies (install-examples    install-sensors-ppc-tests)
endif (WITH_SENSORS_PPC)

