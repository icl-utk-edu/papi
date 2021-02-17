
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_CUDA "Build CUDA components" ON)

if (WITH_CUDA)
    message(STATUS "  - component ${component}")

    set (PAPI_CUDA_ROOT "/opt/cuda" CACHE PATH "Path to the root of the CUDA environment")

    set (PAPI_CUDA_MAIN "" CACHE FILEPATH "Absolute path to libcuda.so")
    #    if (NOT PAPI_CUDA_MAIN STREQUAL "")
        set (CUDA_MAIN_LIB ${PAPI_CUDA_MAIN})
    #endif (NOT PAPI_CUDA_MAIN STREQUAL "")

    #if (NOT CUDA_MAIN_LIB)
    #    find_library (CUDA_MAIN_LIB cuda
    #        HINTS
    #        $ENV{PAPI_CUDA_ROOT}
    #        ${PAPI_CUDA_ROOT}
    #        PATH_SUFFIXES
    #        lib64)
    #endif (NOT CUDA_MAIN_LIB)

    set (PAPI_CUDA_RUNTIME "" CACHE FILEPATH "Absolute path to libcudart.so.so")
    #if (NOT PAPI_CUDA_RUNTIME STREQUAL "")
        set (CUDA_RUNTIME_LIB ${PAPI_CUDA_RUNTIME})
        #endif (NOT PAPI_CUDA_RUNTIME STREQUAL "")

        #if (NOT CUDA_RUNTIME_LIB)
        #    find_library (CUDA_RUNTIME_LIB cudart
        #    HINTS
        #    $ENV{PAPI_CUDA_ROOT}/lib64
        #    ${PAPI_CUDA_ROOT}/lib64)
        #endif (NOT CUDA_RUNTIME_LIB)

    set (PAPI_CUDA_CUPTI "" CACHE FILEPATH "Absolute path to libcupti.so")
    #if (NOT PAPI_CUDA_CUPTI STREQUAL "")
        set (CUDA_CUPTI_LIB ${PAPI_CUDA_CUPTI})
        #endif (NOT PAPI_CUDA_CUPTI STREQUAL "")

        #if (NOT CUDA_CUPTI_LIB)
        #find_library (CUDA_CUPTI_LIB cupti
        #    HINTS
        #    $ENV{PAPI_CUDA_ROOT}/extras/CUPTI/lib64
        #    ${PAPI_CUDA_ROOT}/extras/CUPTI/lib64)
        #endif (NOT CUDA_CUPTI_LIB)

    find_path (CUDA_H cuda.h
        PATHS
        ${PAPI_CUDA_ROOT}
        $ENV{PAPI_CUDA_ROOT}
        PATH_SUFFIXES
        include)

    find_path (CUPTI_H cupti.h
        PATHS
        ${PAPI_CUDA_CUPTI}
        $ENV{PAPI_CUDA_CUPTI}
        ${PAPI_CUDA_ROOT}
        $ENV{PAPI_CUDA_ROOT}
        PATH_SUFFIXES
        include)
      message(STATUS "include ${CUPTI_H}")

    if (NOT CUDA_H)
        message (STATUS "Path to include/cuda.h was not found, please provide PAPI_CUDA_ROOT.")
    endif (NOT CUDA_H)

    if (NOT CUPTI_H)
        message (STATUS "Path to include/cupti.h was not found, please provide either PAPI_CUDA_ROOT or PAPI_CUDA_CUPTI.")
    endif (NOT CUPTI_H)

    if (NOT CUDA_H OR NOT CUPTI_H)
        message (STATUS "Disabling component cuda.")
        set (WITH_CUDA OFF)
        return ()
    endif (NOT CUDA_H OR NOT CUPTI_H)

    set (CUDA_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-cuda.c)

    set (ALL_LIBS 1)

    set (CUDA_TESTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_ld_preload_example.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/simpleMultiGPU.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/timer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cudaTest_cupti_only.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/HelloWorld.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/likeComp_cupti_only.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/nvlink_all.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/nvlink_bandwidth.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/nvlink_bandwidth_cupti_only.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/simpleMultiGPU.cu
	${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
    )

    #if (NOT CUDA_MAIN_LIB)
    #    message (STATUS "Missing libcuda.so library, provide its path in PAPI_CUDA_MAIN (or the PAPI_CUDA_ROOT)")
    #    set (WITH_CUDA OFF)
    #    set (ALL_LIBS 0)
    #endif (NOT CUDA_MAIN_LIB)

    #if (NOT CUDA_RUNTIME_LIB)
    #    message (STATUS "Missing libcudart.so library, provide its path in PAPI_CUDA_RUNTIME (or the PAPI_CUDA_ROOT)")
    #    set (WITH_CUDA OFF)
    #    set (ALL_LIBS 0)
    #endif (NOT CUDA_RUNTIME_LIB)

    #if (NOT CUDA_CUPTI_LIB)
    #    message (STATUS "Missing libcupti.so library, provide its path in PAPI_CUDA_CUPTI (or the PAPI_CUDA_ROOT)")
    #    set (WITH_CUDA OFF)
    #    set (ALL_LIBS 0)
    #endif (NOT CUDA_CUPTI_LIB)

    if (ALL_LIBS)
        message (STATUS "libs \t${CUDA_MAIN_LIB}\n        \t${CUDA_RUNTIME_LIB}\n        \t${CUDA_CUPTI_LIB}")

        list (APPEND SUBCOMPONENTS_LIST_LCL cuda)
        set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
        list (APPEND COMPONENTS_LIST_LCL cuda)
        set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

        target_sources (papi PRIVATE ${CUDA_SRCS})
        target_include_directories (papi
            PRIVATE
            ${PAPI_CUDA_ROOT}/include
            ${PAPI_CUDA_ROOT}/extras/CUPTI/include
            ${CUPTI_H})
        target_compile_definitions (papi
            PRIVATE
            PAPI_CUDA_MAIN=\"${CUDA_MAIN_LIB}\"
            PAPI_CUDA_RUNTIME=\"${CUDA_RUNTIME_LIB}\"
            PAPI_CUDA_CUPTI=\"${CUDA_CUPTI_LIB}\")

        add_custom_target(install-cuda-tests
            COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/cuda
    	    COMMAND ${CMAKE_COMMAND} -E copy ${CUDA_TESTS_SOURCES} ${EXAMPLES_DIR}/cuda/
        )
        add_dependencies (install-examples    install-cuda-tests)

    endif (ALL_LIBS)
endif (WITH_CUDA)
