
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_PERFMON "Build Perf Monitor component" OFF)

#if (WITH_PERFMON)
if (0)
    message(STATUS "  - component ${component}")

    ##############################################################
    # perfmon
    # configure.in:924
    set (old_pfmv2 "n")
    set (perfmon "0")
    set (perfmon2 "no")
    set (force_perfmon2 "no")

    set (WITH_PERFMON "" CACHE STRING "<x.y> Specify perfmon as the performance interface and specify version")
    if (WITH_PERFMON)
        set (USER_SPECIFIED_INTERFACE "perfmon" CACHE INTERNAL "" FORCE)
        set (force_perfmon2 "yes" CACHE INTERNAL "" FORCE)
        set (perfmon_incdir "libpfm-3.y/include" CACHE PATH "" FORCE)
        if (WITH_PERFMON VERSION_GREATER_EQUAL 2.0)
            set (perfmon2 "yes" CACHE INTERNAL "" FORCE)
        endif (WITH_PERFMON VERSION_GREATER_EQUAL 2.0)
        if (WITH_PERFMON VERSION_LESS_EQUAL 2.5)
            set (old_pfmv2 "y" CACHE INTERNAL "" FORCE) #something, something about y instead of yes, otherwise that piece of software breaks
        endif (WITH_PERFMON VERSION_LESS_EQUAL 2.5)
    else (WITH_PERFMON)
        if (CROSS_COMPILING STREQUAL "no")
            find_file (perfmon_version_file version
                PATH /sys/kernel/perfmon)
            if (perfmon_version_file)
                execute_process (
                    COMMAND cat ${perfmon_version_file}
                    OUTPUT_VARIABLE perfmon)
            else (perfmon_version_file)
                find_file (perfmon_version_file2 perfmon
                    PATH /proc)
                if (perfmon_version_file2)
                    execute_process (
                        COMMAND cat ${perfmon_version_file2}
                        COMMAND grep version
                        COMMAND cur -d: -f2
                        OUTPUT_VARIABLE perfmon)
                else (perfmon_version_file2)
                    set (perfmon "0")
                endif (perfmon_version_file2)
            endif (perfmon_version_file)
            if (perfmon)
                set (perfmon_incdir "libpfm-3.y/include" CACHE PATH "" FORCE)
                if (perfmon VERSION_GREATER_EQUAL 2.0)
                    set (perfmon2 "yes" CACHE INTERNAL "" FORCE)
                endif (perfmon VERSION_GREATER_EQUAL 2.0)
                if (WITH_PERFMON VERSION_LESS_EQUAL 2.5)
                    set (old_pfmv2 "y" CACHE INTERNAL "" FORCE)
                endif (WITH_PERFMON VERSION_LESS_EQUAL 2.5)
            endif (perfmon)
        endif (CROSS_COMPILING STREQUAL "no")
    endif (WITH_PERFMON)


    ##############################################################
    # configure.in:1105
    if (NOT perfmon STREQUAL "0" OR perf_events)
        if (WITH_PFM_PREFIX)
            set (dotest 1 CACHE INTERNAL "" FORCE)
        elseif (NOT WITH_PFM_ROOT AND NOT WITH_PFM_LIBDIR)
            set (dotest 1 CACHE INTERNAL "" FORCE)
        else (NOT WITH_PFM_ROOT AND NOT WITH_PFM_LIBDIR)
            set (dotest 0 CACHE INTERNAL "" FORCE)
        endif (NOT WITH_PFM_ROOT AND NOT WITH_PFM_LIBDIR)

        if (dotest)
            find_library (pfm_lib pfm
                PATHS ${pfm_libdir} ${WITH_PFM_LIBDIR})
            if (pfm_lib)
                cmake_push_check_state()
                set (CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${pfm_lib}")
                set (CMAKE_REQUIRED_INCLUDE "${CMAKE_REQUIRED_INCLUDE};${pfm_incdir}")

                try_compile (PFM_COMPILE ${CMAKE_CURRENT_BINARY_DIR}/platform
                    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_pfm_initialize.c)
                if (PFM_COMPILE)
                    find_path (pfm_hdr pfmlib.h
                        PATHS ${pfm_incdir}
                        PATH_SUFFIXES perfmon/)
                    if (pfm_hdr)
                        set (HAVE_PERFMON_PFMLIB_H 1 CACHE INTERNAL "" FORCE)
                        if (arch STREQUAL "ia64")
                            find_path (pfm_montecito_hdr pfmlib_montecito.h
                                PATHS ${pfm_hdr} ${pfm_incdir}
                                PATH_SUFFIXES perfmon/)
                            if (pfm_montecito_hdr)
                                set (HAVE_PERFMON_PFMLIB_MONTECITO_H  1 CACHE INTERNAL "" FORCE)
                            endif (pfm_montecito_hdr)
                        endif (arch STREQUAL "ia64")

                        check_symbol_exists(pfm_get_event_description "" HAVE_PFM_GET_EVENT_DESCRIPTION)
                        check_struct_has_member(pfmlib_reg_t reg_evt_idx "" HAVE_PFM_REG_EVT_IDX)
                        check_struct_has_member(pfmlib_output_param_t pfp_pmd_count
                            "perfmon/pfmlib.h" HAVE_PFMLIB_OUTPUT_PFP_PMD_COUNT)
                        check_struct_has_member(pfm_msg_t type "perfmon/perfmon.h"
                            HAVE_PFM_MSG_TYPE)
                    else (pfm_hdr)
                        message (FATAL_ERROR "libpfm.a not found, rerun configure with different flags")
                    endif (pfm_hdr)
                endif (PFM_COMPILE)
                cmake_pop_check_state()
            endif (pfm_lib)
        else (dotest)
            set (HAVE_PERFMON_PFMLIB_MONTECITO_H  1 CACHE INTERNAL "Montecito headers" FORCE)
            set (HAVE_PFM_GET_EVENT_DESCRIPTION   1 CACHE INTERNAL "event description function" FORCE)
            set (HAVE_PFMLIB_OUTPUT_PFP_PMD_COUNT 1 CACHE INTERNAL "new pfmlib_output_param_t" FORCE)
        endif (dotest)
    endif (NOT perfmon STREQUAL "0" OR perf_events)

#endif (WITH_PERFMON)
endif (0)




