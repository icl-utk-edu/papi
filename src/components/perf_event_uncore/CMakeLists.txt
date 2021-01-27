
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_PERF_EVENTS_UNCORE "Build Perf Events Uncore component" ON)

set (force_perf_events no)
set (disable_uncore    no)
set (perfmon "0")
set (perfctr "0")
set (perf_events no)

if (WITH_PERF_EVENTS_UNCORE)
    message(STATUS "  - component ${component}")

    set (force_perf_events yes)
    set (user_specified_interface pe)
    set (perf_events yes)

    #TODO: Add missgin logic to change the perf interface: perfmon/perfctr, default enforces pe
    #TODO: If we force_perf_events, should we activate the component?

    # configure.in:1066
    if (user_specified_interface STREQUAL "perfmon" AND NOT perfmon STREQUAL "0")
        target_compile_definitions(papi PRIVATE PFMLIB_OLD_PFMV2)
    endif (user_specified_interface STREQUAL "perfmon" AND NOT perfmon STREQUAL "0")

    if (user_specified_interface STREQUAL "pe" AND WITH_PERFEVENT_RDPMC)
        target_compile_definitions(papi PRIVATE USE_PERFEVENT_RDPMC=1)
    endif (user_specified_interface STREQUAL "pe" AND WITH_PERFEVENT_RDPMC)

    if (perf_events)
        if (NOT pfm_built)
            include (install_libpfm)
            set (pfm_built yes PARENT_SCOPE)
        endif (NOT pfm_built)

        set (PERF_EVENTS_UNCORE_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/perf_event_uncore.c)

        target_sources (papi PRIVATE ${PERF_EVENTS_UNCORE_SRCS})

        # list are duplicated when entering a new scope
        list (APPEND SUBCOMPONENTS_LIST_LCL "perf_event_uncore")
        set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
        list (APPEND COMPONENTS_LIST_LCL perf_events_uncore)
        set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)
	#message(STATUS "pushing component perf_events_uncore into list")

	set (PERF_EVENT_UNCORE_TESTS_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_amd_northbridge.c
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_uncore_attach.c
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_uncore.c
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_uncore_cbox.c
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_uncore_lib.c
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_uncore_lib.h
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_uncore_multiple.c
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
        )

        add_custom_target(install-perf-event-uncore-tests
            COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/perf_event_uncore
            COMMAND ${CMAKE_COMMAND} -E copy ${PERF_EVENT_UNCORE_TESTS_SOURCES} ${EXAMPLES_DIR}/perf_event_uncore/
        )
        add_dependencies (install-examples    install-perf-event-uncore-tests)

    endif(perf_events)

    if (NOT user_specified_interface STREQUAL "perfmon" AND NOT user_specified_interface STREQUAL "pe" AND NOT user_specified_interface STREQUAL "perfctr")
        message (FATAL_ERROR "Unknown user_specified_interface=${user_specified_interface} perfctr=${perfctr} perfmon=${perfmon} perfmon2=${perfmon2} perf-events=${perf_events}")
    endif (NOT user_specified_interface STREQUAL "perfmon" AND NOT user_specified_interface STREQUAL "pe" AND NOT user_specified_interface STREQUAL "perfctr")
endif (WITH_PERF_EVENTS_UNCORE)

