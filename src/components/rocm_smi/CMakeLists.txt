
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_ROCM_SMI "Build ROCM_SMI components" ON)

if (NOT WITH_ROCM)
    set (WITH_ROCM_SMI OFF)
endif (NOT WITH_ROCM)

if (WITH_ROCM_SMI)
    message(STATUS "  - component ${component}")
    set (PAPI_ROCM_ROOT "$ENV{PAPI_ROCM_ROOT}" CACHE PATH "Path to the root of the ROCM environment")

    set (PAPI_ROCM_SMI_MAIN     "" CACHE PATH "Absolut path to rocprofiler/lib/librocm_smi64.so")

    find_path (ROCM_SMI_H_PATH rocm_smi.h
        PATHS
	${PAPI_ROCM_ROOT}/rocm_smi/include/rocm_smi
	$ENV{PAPI_ROCM_ROOT}/rocm_smi/include/rocm_smi)

    if (NOT ROCM_SMI_H_PATH)
        message (STATUS "Path to rocm_smi/include/rocm_smi/rocm_smi.h was not found, please provide PAPI_ROCM_ROOT.")
        message (STATUS "Disabling component rocm.")
	set (WITH_ROCM_SMI OFF)
        return ()
    endif (NOT ROCM_SMI_H_PATH)

    set (ROCM_SMI_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-rocm-smi.c)

    set (ALL_LIBS 1)
    find_library(ROCM_SMI_LIB rocm_smi64
        PATHS
	${PAPI_ROCM_ROOT}/rocm_smi/lib
	${PAPI_ROCM_SMI_MAIN}
	$ENV{PAPI_ROCM_ROOT}/rocm_smi/lib
	$ENV{PAPI_ROCM_SMI_MAIN})
    if (NOT ROCM_SMI_LIB)
        message (STATUS "Path to librocm_smi64.so was not found, please provide PAPI_ROCM_SMI_MAIN.")
        set (ALL_LIBS 0)
    endif (NOT ROCM_SMI_LIB)

    if (ALL_LIBS)
        message (STATUS "libs \t${ROCM_SMI_LIB}")

        list (APPEND SUBCOMPONENTS_LIST_LCL rocm)
        set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
        list (APPEND COMPONENTS_LIST_LCL rocm)
        set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)

	target_sources (papi PRIVATE ${ROCM_SMI_SRCS})
        target_include_directories (papi
            PRIVATE
	    ${PAPI_ROCM_ROOT}/rocm_smi/include
	    ${PAPI_ROCM_ROOT}/rocm_smi/include/rocm_smi
	    ${ROCM_SMI_H_PATH})
        target_compile_definitions (papi
            PRIVATE
	    PAPI_ROCM_SMI_MAIN=\"${PAPI_ROCM_SMI_MAIN}\")

        set (ROCM_SMI_TESTS_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rocmcap_plot.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rocm_command_line.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rocm_smi_all.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/ROCM_SMI_Makefile
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rocm_smi_writeTests.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/Makefile
        )

        add_custom_target(install-rocm_smi-tests
            COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLES_DIR}/rocm_smi
	    COMMAND ${CMAKE_COMMAND} -E copy ${ROCM_SMI_TESTS_SOURCES} ${EXAMPLES_DIR}/rocm_smi/
        )
        add_dependencies (install-examples    install-rocm_smi-tests)

      endif (ALL_LIBS)
endif (WITH_ROCM_SMI)

