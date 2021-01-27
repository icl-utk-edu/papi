
if (NOT TARGET pfm4_lib)
    set (force_pfm_incdir "no")

    set (WITH_PFM_ROOT "" CACHE PATH "Specify path to source tree (for use by developers only)")
    set (WITH_PFM_PREFIX "" CACHE PATH "Specify prefix to installed pfm distribution")
    set (WITH_PFM_INCDIR "" CACHE PATH "Specify directory of pfm header files in non-standard location")
    set (WITH_PFM_LIBDIR "" CACHE PATH "Specify directory of pfm library in non-standard location")

    if (WITH_PFM_ROOT OR WITH_PFM_PREFIX)
        find_path (pfm_incdir perf_event.h
            PATHS ${WITH_PFM_ROOT} ${WITH_PFM_PREFIX}
            PATH_SUFFIXES include/perfmon)
        if (pfm_incdir)
            set (pfm_libdir ${pfm_incdir}/lib)
        endif (pfm_incdir)
    endif (WITH_PFM_ROOT OR WITH_PFM_PREFIX)

    if (WITH_PFM_INCDIR)
        find_path (pfm_incdir perf_event.h
            PATHS ${WITH_PFM_ROOT} ${WITH_PFM_PREFIX}
            PATH_SUFFIXES perfmon)
        if (pfm_incdir)
            set (pfm_libdir ${pfm_incdir}/../lib)
        endif (pfm_incdir)
    endif (WITH_PFM_INCDIR)

    if (WITH_PFM_LIBDIR)
        set (pfm_libdir ${WITH_PFM_LIBDIR})
    endif (WITH_PFM_LIBDIR)

    if (NOT pfm_libdir AND NOT pfm_incdir)
        # rebuild the one we embed
        # beware, don't use CURRENT_ paths, we are called from the components
        set (pfm_src_root   ${CMAKE_SOURCE_DIR}/libpfm4)
        set (pfm_src_incdir ${CMAKE_SOURCE_DIR}/libpfm4/include)
        set (pfm_src_libdir ${CMAKE_SOURCE_DIR}/libpfm4/lib)

        set (pfm_bin_root   ${CMAKE_BINARY_DIR}/libpfm4)
        set (pfm_bin_incdir ${CMAKE_BINARY_DIR}/libpfm4/include)
        set (pfm_bin_libdir ${CMAKE_BINARY_DIR}/libpfm4/lib)

        find_program(MAKE_EXE NAMES make nmake gmake)

        add_custom_command (
            OUTPUT ${pfm_bin_libdir}/libpfm.so ${pfm_bin_libdir}/libpfm.a
            # build
            COMMAND ${MAKE_EXE}
            # install the libs
            COMMAND ${CMAKE_COMMAND} -E copy ${pfm_src_libdir}/libpfm.so ${pfm_bin_libdir}/libpfm.so
            COMMAND ${CMAKE_COMMAND} -E copy ${pfm_src_libdir}/libpfm.a  ${pfm_bin_libdir}/libpfm.a
            # copy the headers?

            COMMAND ${MAKE_EXE} clean
            WORKING_DIRECTORY ${pfm_src_root})

        add_custom_target (
            pfm4_install_clean
            # that should trigger the build
            DEPENDS ${pfm_bin_libdir}/libpfm.so ${pfm_bin_libdir}/libpfm.a)

        # .so and .a are installed in BINARY tree
        add_library(pfm4_lib SHARED IMPORTED)
        set_target_properties (pfm4_lib PROPERTIES
            IMPORTED_LOCATION ${pfm_bin_libdir}/libpfm.so)
        target_include_directories(pfm4_lib INTERFACE ${pfm_src_incdir})

        message (STATUS "include: ${pfm_src_incdir}")
        target_compile_definitions (papi PRIVATE PEINCLUDE=\"${pfm_src_incdir}/perfmon/perf_event.h\")
        message (STATUS "library: ${pfm_bin_libdir}")

        add_dependencies(pfm4_lib pfm4_install_clean)

    else (NOT pfm_libdir AND NOT pfm_incdir)
        # system/exotic installation?
        find_library (pfm4_lib_file
            NAMES libpfm.a # add any other relevant name for thet static version
            PATHS
            # TODO: Add system paths, and pfm_libdir / pfm_root/lib
            ${pfm_libdir})

        if (NOT pfm4_lib)
            message (WARNING "libpfm4 not found, disabling component!")
        endif (NOT pfm4_lib)

        find_path (pfm_incdir perf_event.h
            # TODO: check paths are correct
            PATHS ${pfm_incdir} 
            PATH_SUFFIXES perfmon)

        if (NOT pfm_incdir)
            message (WARNING "libpfm4 header missing, disabling component!")
        endif (NOT pfm_incdir)

        if (pfm_incdir AND pfm4_lib_file)
            add_library(pfm4_lib STATIC IMPORTED)

            set_target_properties(pfm4_lib PROPERTIES
                IMPORTED_LOCATION ${pfm4_lib_file})

            target_include_directories (pfm4_lib
                INTERFACE )
        endif (pfm_incdir AND pfm4_lib_file)

    endif (NOT pfm_libdir AND NOT pfm_incdir)

    if (TARGET pfm4_lib)
        target_link_libraries (papi PRIVATE pfm4_lib)
        target_link_libraries (papi_static PUBLIC pfm4_lib)
        target_link_libraries (papi_shared PUBLIC pfm4_lib)

        set (HAVE_PERFMON_PFMLIB_MONTECITO_H  1 CACHE INTERNAL "" FORCE)
        set (HAVE_PFM_GET_EVENT_DESCRIPTION  1 CACHE INTERNAL "" FORCE)
        set (HAVE_PFMLIB_OUTPUT_PFP_PMD_COUNT  1 CACHE INTERNAL "" FORCE)
    endif (TARGET pfm4_lib)

endif (NOT TARGET pfm4_lib)
