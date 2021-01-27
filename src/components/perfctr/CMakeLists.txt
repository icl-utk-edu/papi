
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_PERFCTR "Build Perf Counter component" OFF)

if (WITH_PERFCTR)
    # configure.in:848
    if (NOT perfctr STREQUAL "0")
        set (PFM_INCDIR CACHE PATH "libpfm-3.y/include/" FORCE)
        if (perfctr STREQUAL "7" AND NOT ARCH STREQUAL "ppc64")
            message (FATAL_ERROR "Perfctr 2.7.x only works on PPC64 machines. Patch kernel with 2.6.x.")
        endif (perfctr STREQUAL "7" AND NOT ARCH STREQUAL "ppc64")
    endif (NOT perfctr STREQUAL "0")

    # configure.in:857,869
    set (WITH_PERFCTR_ROOT "" CACHE PATH "Specify path to source tree (for use by developers only)." FORCE)
    set (WITH_PERFCTR_PREFIX "" CACHE PATH "Specify prefix to installed perfctr distribution." FORCE)
    set (WITH_PERFCTR_INCDIR "" CACHE PATH "Specify directory of perfctr header files in non-standard location." FORCE)
    set (WITH_PERFCTR_LIBDIR "" CACHE PATH "Specify directory of perfctr library in non-standard location." FORCE)

    if (WITH_PERFCTR_ROOT OR WITH_PERFCTR_PREFIX OR WITH_PERFCTR_INCDIR OR WITH_PERFCTR_LIBDIR)
        set (USER_SPECIFIED_INTERFACE "perfctr")
    endif (WITH_PERFCTR_ROOT OR WITH_PERFCTR_PREFIX OR WITH_PERFCTR_INCDIR OR WITH_PERFCTR_LIBDIR)

    message (STATUS "perfctr=${perfctr}, force_perfctr=${force_perfctr}, USER_SPECIFIED_INTERFACE=${USER_SPECIFIED_INTERFACE}")
    message (STATUS "WITH_PERFCTR_ROOT=${WITH_PERFCTR_ROOT}, WITH_PERFCTR_PREFIX=${WITH_PERFCTR_PREFIX}")
    message (STATUS "WITH_PERFCTR_INCDIR=${WITH_PERFCTR_INCDIR}, WITH_PERFCTR_LIBDIR=${WITH_PERFCTR_LIBDIR}")

    # configure.in:874
    if (NOT perfctr STREQUAL "0")
        set (do_test OFF)
        if (WITH_PERFCTR_INCDIR)
            set (do_test ON)
            find_path (PERFCTR_HDR libperfctr.h
                PATHS ${WITH_PERFCTR_INCDIR})
            if (PERFCTR_HDR)
                target_include_directories(papi PRIVATE ${PERFCTR_HDR})
            endif (PERFCTR_HDR)
        endif (WITH_PERFCTR_INCDIR)
        if (WITH_PERFCTR_ROOT OR WITH_PERFCTR_PREFIX OR WITH_PERFCTR_LIBDIR)
            set (do_test ON)
            find_library (PERFCTR_LIB perfctr
                PATHS
                ${WITH_PERFCTR_ROOT}
                ${WITH_PERFCTR_PREFIX}
                ${WITH_PERFCTR_LIBDIR}
                PATH_SUFFIXES usr.lib lib)
            if (PERFCTR_LIB)
                target_link_libraries(papi PRIVATE ${PERFCTR_LIB})
            endif (PERFCTR_LIB)

            find_path (PERFCTR_HDR libperfctr.h
                PATHS ${WITH_PERFCTR_ROOT} ${WITH_PERFCTR_PREFIX}
                PATH_SUFFIXES linux/include include)
            if (PERFCTR_HDR)
                target_include_directories(papi PRIVATE ${PERFCTR_HDR})
            endif (PERFCTR_HDR)
        else (WITH_PERFCTR_ROOT OR WITH_PERFCTR_PREFIX OR WITH_PERFCTR_LIBDIR)
            if (WITH_PERFCTR_INCDIR)
            endif (WITH_PERFCTR_INCDIR)
        endif (WITH_PERFCTR_ROOT OR WITH_PERFCTR_PREFIX OR WITH_PERFCTR_LIBDIR)

        if (do_test)
            try_compile(HAVE_VPERFCTR_OPEN
                ${CMAKE_CURRENT_BINARY_DIR}/platform
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_vperfctr_open.c)

            if (HAVE_VPERFCTR_OPEN)
                find_path (PERFCTR_HDR_PATH libperfctr.h
                    PATHS ${WITH_PERFCTR_INCDIR})
                if (PERFCTR_HDR_PATH)
                    try_run(HAVE_PERFCTR_COMPILE HAVE_PERFCTR_RUN
                        ${CMAKE_CURRENT_BINARY_DIR}/platform
                        ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_perfctr.c)
                    if (HAVE_PERFCTR_RUN STREQUAL "0")
                        set (perfctr "6" CACHE INTERNAL "" FORCE)
                    else (HAVE_PERFCTR_RUN STREQUAL "0")
                        set (perfctr "7" CACHE INTERNAL "" FORCE)
                    endif (HAVE_PERFCTR_RUN STREQUAL "0")
                endif (PERFCTR_HDR_PATH)
            else (HAVE_VPERFCTR_OPEN)
                message (FATAL_ERROR "libperfctr.a not found, rerun configure with different flags.")
            endif (HAVE_VPERFCTR_OPEN)
        endif (do_test)
    endif (NOT perfctr STREQUAL "0")

    message (STATUS "perfctr=${perfctr}, force_perfctr=${force_perfctr}")

endif (WITH_PERFCTR)
