
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_ROCM "Build ROCM components" ON)

if (WITH_ROCM)
    message(STATUS "  - component ${component}")
    set (PAPI_ROCM_ROOT "$ENV{PAPI_ROCM_ROOT}" CACHE PATH "Path to the root of the ROCM environment")

    set (PAPI_ROCM_METRICS "${PAPI_ROCM_ROOT}/rocprofiler/lib/metrics.xml" CACHE FILEPATH "Absolute path to metrics.xml, usually $ROCM_ROOT/rocprofiler/lib/metrics.xml")
    set (PAPI_ROCM_HSA     "" CACHE PATH "Absolut path to hsa/lib/libhsa-runtime64.so.1")
    set (PAPI_ROCM_PROF    "" CACHE PATH "Absolute path to rocprofiler/lib/librocprofiler64.so.1")

    # those guys have to be set in env
    set (ROCMPROFILER_LOG 1)
    set (HSA_VEN_AMD_AQLPROFILE_LOG 1)
    set (AQLPROFILE_READ_API 1)

    find_path (HSA_H_PATH hsa.h
        PATHS
	${PAPI_ROCM_ROOT}/hsa/include/hsa
	$ENV{PAPI_ROCM_ROOT}/hsa/include/hsa)

    find_path (ROCPROFILER_H_PATH rocprofiler.h
        PATHS
	${PAPI_ROCM_ROOT}/rocprofiler/include
	$ENV{PAPI_ROCM_ROOT}/rocprofiler/include)

    if (NOT HSA_H_PATH)
        message (STATUS "Path to hsa/include/hsa.h was not found, please provide PAPI_ROCM_ROOT.")
    endif (NOT HSA_H_PATH)

    if (NOT ROCPROFILER_H_PATH)
        message (STATUS "Path to rocprofiler/include/rocprofiler.h was not found, please provide PAPI_ROCM_ROOT.")
    endif (NOT ROCPROFILER_H_PATH)

    if (NOT HSA_H_PATH OR NOT ROCPROFILER_H_PATH)
        message (STATUS "Disabling component rocm.")
	set (WITH_ROCM OFF)
        return ()
    endif (NOT HSA_H_PATH OR NOT ROCPROFILER_H_PATH)

    set (ROCM_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-rocm.c)
    set (ALL_LIBS 1)
    find_library(HSA_RUNTIME hsa-runtime64
        PATHS
	${PAPI_ROCM_ROOT}/hsa/lib
	${PAPI_ROCM_HSA}
	${PAPI_ROCM_HSA}/hsa/lib
	$ENV{PAPI_ROCM_HSA}
	$ENV{PAPI_ROCM_HSA}/hsa/lib)
    if (NOT HSA_RUNTIME)
        message (STATUS "Path to libhsa-runtime64.so was not found, please provide PAPI_ROCM_HSA.")
        set (ALL_LIBS 0)
    endif (NOT HSA_RUNTIME)

    find_library(ROCPROFILER_LIB rocprofiler64
        PATHS
	${PAPI_ROCM_ROOT}/rocprofiler/lib
	${PAPI_ROCM_PROF}
	${PAPI_ROCM_PROF}/rocprofiler/lib)
    if (NOT ROCPROFILER_LIB)
        message (STATUS "Path to librocprofiler64.so was not found, please provide PAPI_ROCM_PROF.")
        set (ALL_LIBS 0)
    endif (NOT ROCPROFILER_LIB)

    if (ALL_LIBS)
        message (STATUS "libs \t${HSA_RUNTIME}\n        \t${ROCPROFILER_LIB}")

        list (APPEND SUBCOMPONENTS_LIST_LCL rocm)
        set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
        list (APPEND COMPONENTS_LIST_LCL rocm)
        set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

	target_sources (papi PRIVATE ${ROCM_SRCS})
        target_include_directories (papi
            PRIVATE
	    ${PAPI_ROCM_ROOT}/include
	    ${HSA_H_PATH}
	    ${ROCPROFILER_H_PATH})
        target_compile_definitions (papi
            PRIVATE
	    PAPI_ROCM_HSA=\"${PAPI_ROCM_HSA}\"
	    PAPI_ROCM_PROF=\"${PAPI_ROCM_PROF}\")

        set (ROCM_TESTS_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rocm_all.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rocm_command_line.c
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/ROCM_Makefile
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_papi.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
        )

        add_custom_target(install-rocm-tests
            COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/rocm
            COMMAND ${CMAKE_COMMAND} -E copy ${PERF_EVENT_UNCORE_TESTS_SOURCES} ${EXAMPLES_DIR}/rocm/
        )
        add_dependencies (install-examples    install-rocm-tests)
      endif (ALL_LIBS)
endif (WITH_ROCM)
