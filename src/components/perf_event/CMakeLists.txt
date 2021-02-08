
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_PERF_EVENTS "Build Perf Events component" ON)

if (MIC)
    set (perf_events no PARENT_SCOPE)
    set (perf_events no)
endif (MIC)

if (WITH_PERF_EVENTS)
    message(STATUS "  - component ${component}")

    set (force_perf_events yes CACHE INTERNAL "dffd" FORCE)
    set (user_specified_interface pe CACHE INTERNAL "argadrg" FORCE)
    # configure.in:1020
    option (WITH_PERFEVENT_RDPMC "Enable userspace rdpmc instruction on perf_event" ON)
    if (WITH_PERFEVENT_RDPMC)
        set (enable_perfevent_rpdmc yes)
        #pecflags just append to papi cflags, do it directly
    else (WITH_PERFEVENT_RDPMC)
        set (enable_perfevent_rpdmc no)
    endif (WITH_PERFEVENT_RDPMC)

    if (force_perf_events)
        set (perf_events yes CACHE INTERNAL "asdfg" FORCE)
        message(STATUS "Setting perf_events=${perf_events}")
    endif (force_perf_events)

    if (NOT CROSS_COMPILING)
        find_file (perf_event_paranoid_file perf_event_paranoid
            PATHS /proc/sys/kernel)
        message (STATUS "perf_event_paranoid_file=${perf_event_paranoid_file}")
        if (perf_event_paranoid_file)
            if (NOT pfm_built)
                include (install_libpfm)
                set (pfm_built yes PARENT_SCOPE)
            endif (NOT pfm_built)

            set (PERF_EVENTS_SRCS
                ${CMAKE_CURRENT_SOURCE_DIR}/perf_event.c
                ${CMAKE_CURRENT_SOURCE_DIR}/pe_libpfm4_events.c
                #TODO: why is this source file sitting in the root dir?
                ${CMAKE_SOURCE_DIR}/papi_libpfm4_events.c)

            target_sources (papi PRIVATE ${PERF_EVENTS_SRCS})

            if (SUBCOMPONENTS_LIST_LCL)
	        #if list is non empty, there is a component in position 0 that has to move away
	        list (GET SUBCOMPONENTS_LIST_LCL 0 OLD_HEAD)
	        list (REMOVE_AT SUBCOMPONENTS_LIST_LCL 0)
		list (INSERT SUBCOMPONENTS_LIST_LCL 0 perf_event)
		list (APPEND SUBCOMPONENTS_LIST_LCL ${OLD_HEAD})
            else (SUBCOMPONENTS_LIST_LCL)
	 	list (APPEND SUBCOMPONENTS_LIST_LCL perf_event)
	    endif (SUBCOMPONENTS_LIST_LCL)
	    # list are duplicated when entering a new scope
            set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)

	    if (COMPONENTS_LIST_LCL)
	        list (GET COMPONENTS_LIST_LCL 0 OLD_HEAD)
	        list (REMOVE_AT COMPONENTS_LIST_LCL 0)
	        list (INSERT COMPONENTS_LIST_LCL 0 perf_event)
	        list (APPEND COMPONENTS_LIST_LCL ${OLD_HEAD})
            else (COMPONENTS_LIST_LCL)
	        list (APPEND COMPONENTS_LIST_LCL perf_event)
            endif (COMPONENTS_LIST_LCL)

	    set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)
	    #message(STATUS "pushing component perf_events into list")

            set (PERF_EVENT_TESTS_SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/event_name_lib.c
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/broken_events.c
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/nmi_watchdog.c
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_offcore_response.c
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/event_name_lib.h
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_system_wide.c
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/perf_event_user_kernel.c
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
            )

            add_custom_target(install-perf-event-tests
                COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/perf_event
		COMMAND ${CMAKE_COMMAND} -E copy ${PERF_EVENT_TESTS_SOURCES} ${EXAMPLES_DIR}/perf_event/
            )
            add_dependencies (install-examples    install-perf-event-tests)

        endif (perf_event_paranoid_file)
    endif (NOT CROSS_COMPILING)

    # configure.in:1066
    if (user_specified_interface STREQUAL "pe" AND WITH_PERFEVENT_RDPMC)
        target_compile_definitions(papi PRIVATE USE_PERFEVENT_RDPMC=1)
    endif (user_specified_interface STREQUAL "pe" AND WITH_PERFEVENT_RDPMC)

    if (NOT user_specified_interface STREQUAL "perfmon" AND NOT user_specified_interface STREQUAL "pe" AND NOT user_specified_interface STREQUAL "perfctr")
        message (FATAL_ERROR "Unknown user_specified_interface=${user_specified_interface} perfctr=${perfctr} perfmon=${perfmon} perfmon2=${perfmon2} perf_events=${perf_events}")
    endif (NOT user_specified_interface STREQUAL "perfmon" AND NOT user_specified_interface STREQUAL "pe" AND NOT user_specified_interface STREQUAL "perfctr")
endif (WITH_PERF_EVENTS)
