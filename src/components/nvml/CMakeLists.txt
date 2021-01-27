
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_NVML "Build NVML components" ON)

if (WITH_NVML)
    message(STATUS "  - component ${component}")

    if (NOT WITH_CUDA)
        message (STATUS "CUDA is disabled, disabling NVML.")
	set (WITH_NVML OFF)
	return ()
    endif (NOT WITH_CUDA)

    set (PAPI_NNVML_MAIN "" CACHE FILEPATH "Absolute path to libnvidia-ml.so")
    set (CUDA_NVML_LIB ${PAPI_NVML_MAIN})

    find_path (NVML_H nvml.h
        PATHS
        ${PAPI_CUDA_ROOT}
        $ENV{PAPI_CUDA_ROOT}
        PATH_SUFFIXES
        include)

    if (NOT NVML_H)
        message (STATUS "Path to include/nvml.h was not found, please provide PAPI_CUDA_ROOT.")
    endif (NOT NVML_H)

    if (NOT NVML_H)
        message (STATUS "Disabling component nvml.")
	set (WITH_NVML OFF)
        return ()
    endif (NOT NVML_H)

    set (NVML_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-nvml.c)

    set (ALL_LIBS 1)

    if (ALL_LIBS)
	    message (STATUS "libs \t${CUDA_NVML_LIB}\n")

        list (APPEND SUBCOMPONENTS_LIST_LCL nvml)
        set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
        list (APPEND COMPONENTS_LIST_LCL nvml)
        set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

	target_sources (papi PRIVATE ${NVML_SRCS})
        target_include_directories (papi
            PRIVATE
            ${PAPI_CUDA_ROOT}/include)
        target_compile_definitions (papi
            PRIVATE
	    PAPI_NVML_MAIN=\"${CUDA_NVML_LIB}\")

        set (NVML_TESTS_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchSANVML.c
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/HelloWorld.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/nvml_power_limiting_test.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/nvml_power_limit_read_test.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
        )

        add_custom_target(install-nvml-tests
            COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/nvml
	    COMMAND ${CMAKE_COMMAND} -E copy ${NVML_TESTS_SOURCES} ${EXAMPLES_DIR}/nvml/
        )
        add_dependencies (install-examples    install-nvml-tests)
    endif (ALL_LIBS)
endif (WITH_NVML)

