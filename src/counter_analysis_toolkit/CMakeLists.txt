cmake_minimum_required(VERSION 3.18)

#-------------------------------------------------------------------------------
# Compiler info.
message( "" )
message( "------------------ Compiler ------------------" )

project(
    CounterAnalysisToolkit
    VERSION 2025.07.14
    LANGUAGES C
)

include(CheckCCompilerFlag)

if(DEFINED CACHE{CAT_CC_STAT})
    message( STATUS "Using cached compiler info." )
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CAT_CC_STAT ON CACHE BOOL "Whether or not compiler was found")
    message( STATUS "Using GCC.")
else()
    message( FATAL_ERROR "GCC is required. Please set -DCMAKE_C_COMPILER=/path/to/gcc" )
endif()

#-------------------------------------------------------------------------------
# Detect CPU architecture.
message( "" )
message( "-------------------- Arch --------------------" )

if(DEFINED CACHE{ARCH})
    message( STATUS "Using cached arch info." )
else()
    message( STATUS "Detecting CPU architecture..." )
endif()

if(DEFINED ARCH AND NOT DEFINED CACHE{ARCH})
    message( WARNING "User has provided -DARCH. ARCH is determined automatically." )
endif()

execute_process(COMMAND lscpu OUTPUT_VARIABLE FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH   "Flags:[^\n]*"  FLAGS "${FLAGS}")
string(REGEX REPLACE "Flags:[ ]*" "" FLAGS "${FLAGS}")
string(REPLACE " " ";" FLAGS "${FLAGS}")

# Get microarchitectural details.
execute_process(COMMAND uname -m OUTPUT_VARIABLE UNAME_M OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "uname -m : ${UNAME_M}")
if(UNAME_M STREQUAL "x86_64")
    set(ARCH "X86" CACHE STRING "Architecture identifier in CAT")
    message(STATUS "Setting ARCH=${ARCH}")
    set(FLOP -D${ARCH})
    set(INSTR "")
    set(VEC_META "")
    set(VEC128 -O0 -D${ARCH})
    set(VEC256 -O0 -D${ARCH})
    set(VEC512 -O0 -D${ARCH})
    set(VEC128_FMA -O0 -D${ARCH})
    set(VEC256_FMA -O0 -D${ARCH})
    set(VEC512_FMA -O0 -D${ARCH})
    set(VEC     -O0 -D${ARCH})
    set(VEC_FMA -O0 -D${ARCH})
    set(VEC_ALL -O0 -D${ARCH})

    # Check to see what the hardware supports.
    set(AVX512VL_AVAIL OFF)
    set(AVX512_AVAIL   OFF)
    set(AVX256_AVAIL   OFF)
    set(AVX128_AVAIL   OFF)
    set(FMA_AVAIL      OFF)
    foreach(flag IN LISTS FLAGS)
        if(flag STREQUAL "avx512vl")
            set(AVX512VL_AVAIL ON)
        elseif(flag STREQUAL "avx512f")
            set(AVX512_AVAIL ON)
        elseif(flag STREQUAL "avx2")
            set(AVX256_AVAIL ON)
        elseif(flag STREQUAL "avx")
            set(AVX128_AVAIL ON)
        elseif(flag STREQUAL "fma")
            set(FMA_AVAIL ON)
        endif()
    endforeach()

    # Check to see what the compiler supports.
    check_c_compiler_flag("-mavx512vl" AVX512VL_CC_AVAIL)
    check_c_compiler_flag("-mavx512f"  AVX512_CC_AVAIL)
    check_c_compiler_flag("-mavx2"     AVX256_CC_AVAIL)
    check_c_compiler_flag("-mavx"      AVX128_CC_AVAIL)
    check_c_compiler_flag("-mfma"      FMA_CC_AVAIL)

    # Check AVX128 availability.
    if(AVX128_CC_AVAIL)
        message(STATUS "Compiler supports AVX128")
        if(AVX128_AVAIL)
            message(STATUS "Hardware supports AVX128")
            list(APPEND VEC_META   -DAVX128_AVAIL)
            list(APPEND VEC128     -mavx -DX86_VEC_WIDTH_128B)
            list(APPEND VEC128_FMA -mavx -DX86_VEC_WIDTH_128B)
            list(APPEND VEC        -mavx)
        else()
            message(STATUS "Hardware does not support AVX128!")
        endif()
    else()
        message(STATUS "Compiler does not support AVX128!")
    endif()

    # Check AVX256 availability.
    if(AVX256_CC_AVAIL)
        message(STATUS "Compiler supports AVX256")
        if(AVX256_AVAIL)
            message(STATUS "Hardware supports AVX256")
            list(APPEND VEC_META   -DAVX256_AVAIL)
            list(APPEND VEC256     -mavx2 -DX86_VEC_WIDTH_256B)
            list(APPEND VEC256_FMA -mavx2 -DX86_VEC_WIDTH_256B)
        else()
            message(STATUS "Hardware does not support AVX256!")
        endif()
    else()
        message(STATUS "Compiler does not support AVX256!")
    endif()

    # Check AVX512 availability.
    if(AVX512_CC_AVAIL)
        message(STATUS "Compiler supports AVX512")
        if(AVX512_AVAIL)
            message(STATUS "Hardware supports AVX512")
            list(APPEND VEC_META   -DAVX512_AVAIL)
            list(APPEND VEC512     -mavx512f -DX86_VEC_WIDTH_512B)
            list(APPEND VEC512_FMA -mavx512f -DX86_VEC_WIDTH_512B)
        else()
            message(STATUS "Hardware does not support AVX512!")
        endif()
        if(AVX512VL_CC_AVAIL)
            message(STATUS "Compiler supports AVX512VL")
            if(AVX512VL_AVAIL)
                message(STATUS "Hardware supports AVX512VL")
                list(APPEND INSTR  -mavx512vl)
            else()
                message(STATUS "Hardware does not support AVX512VL!")
            endif()
        else()
            message(STATUS "Compiler does not support AVX512VL!")
        endif()
    else()
        message(STATUS "Compiler does not support AVX512!")
    endif()

    # Check FMA availability. -- prob need a -DAVX128_FMA_AVAIL flag at some point.
    if(FMA_CC_AVAIL)
        message(STATUS "Compiler supports FMA")
        if(FMA_AVAIL)
            message(STATUS "Hardware supports FMA")
            list(APPEND FLOP -mfma)
            list(APPEND VEC128_FMA -mfma)
            list(APPEND VEC256_FMA -mfma)
            list(APPEND VEC512_FMA -mfma)
            list(APPEND VEC_FMA    -mfma)
        else()
            message(STATUS "Hardware does not support FMA!")
        endif()
    else()
        message(STATUS "Compiler does not support FMA!")
    endif()

    # Update remaining vector flags.
    list(APPEND VEC128     ${VEC_META})
    list(APPEND VEC256     ${VEC_META})
    list(APPEND VEC512     ${VEC_META})
    list(APPEND VEC128_FMA ${VEC_META})
    list(APPEND VEC256_FMA ${VEC_META})
    list(APPEND VEC512_FMA ${VEC_META})
    list(APPEND VEC_ALL    ${VEC} ${VEC_FMA})

    # Conditionally add AVX128 dependencies.
    if(AVX128_CC_AVAIL AND AVX128_AVAIL)
        list(APPEND VEC_OBJECTS $<TARGET_OBJECTS:vec_nonfma_hp-128B>
                                $<TARGET_OBJECTS:vec_nonfma_sp-128B>
                                $<TARGET_OBJECTS:vec_nonfma_dp-128B>)
        add_library(vec_nonfma_hp-128B OBJECT src/vec_nonfma_hp.c)
        target_compile_options(vec_nonfma_hp-128B PRIVATE ${VEC128})
        set_target_properties(vec_nonfma_hp-128B PROPERTIES OUTPUT_NAME "vec_nonfma_hp-128B")

        add_library(vec_nonfma_sp-128B OBJECT src/vec_nonfma_sp.c)
        target_compile_options(vec_nonfma_sp-128B PRIVATE ${VEC128})
        set_target_properties(vec_nonfma_sp-128B PROPERTIES OUTPUT_NAME "vec_nonfma_sp-128B")

        add_library(vec_nonfma_dp-128B OBJECT src/vec_nonfma_dp.c)
        target_compile_options(vec_nonfma_dp-128B PRIVATE ${VEC128})
        set_target_properties(vec_nonfma_dp-128B PROPERTIES OUTPUT_NAME "vec_nonfma_dp-128B")

        if(FMA_CC_AVAIL AND FMA_AVAIL)
            list(APPEND VEC_OBJECTS $<TARGET_OBJECTS:vec_fma_hp-128B>
                                    $<TARGET_OBJECTS:vec_fma_sp-128B>
                                    $<TARGET_OBJECTS:vec_fma_dp-128B>)
            add_library(vec_fma_hp-128B OBJECT src/vec_fma_hp.c)
            target_compile_options(vec_fma_hp-128B PRIVATE ${VEC128_FMA})
            set_target_properties(vec_fma_hp-128B PROPERTIES OUTPUT_NAME "vec_fma_hp-128B")

            add_library(vec_fma_sp-128B OBJECT src/vec_fma_sp.c)
            target_compile_options(vec_fma_sp-128B PRIVATE ${VEC128_FMA})
            set_target_properties(vec_fma_sp-128B PROPERTIES OUTPUT_NAME "vec_fma_sp-128B")

            add_library(vec_fma_dp-128B OBJECT src/vec_fma_dp.c)
            target_compile_options(vec_fma_dp-128B PRIVATE ${VEC128_FMA})
            set_target_properties(vec_fma_dp-128B PROPERTIES OUTPUT_NAME "vec_fma_dp-128B")
        endif()
    endif()

    # Conditionally add AVX256 dependencies.
    if(AVX256_CC_AVAIL AND AVX256_AVAIL)
        list(APPEND VEC_OBJECTS $<TARGET_OBJECTS:vec_nonfma_hp-256B>
                                $<TARGET_OBJECTS:vec_nonfma_sp-256B>
                                $<TARGET_OBJECTS:vec_nonfma_dp-256B>)
        add_library(vec_nonfma_hp-256B OBJECT src/vec_nonfma_hp.c)
        target_compile_options(vec_nonfma_hp-256B PRIVATE ${VEC256})
        set_target_properties(vec_nonfma_hp-256B PROPERTIES OUTPUT_NAME "vec_nonfma_hp-256B")

        add_library(vec_nonfma_sp-256B OBJECT src/vec_nonfma_sp.c)
        target_compile_options(vec_nonfma_sp-256B PRIVATE ${VEC256})
        set_target_properties(vec_nonfma_sp-256B PROPERTIES OUTPUT_NAME "vec_nonfma_sp-256B")

        add_library(vec_nonfma_dp-256B OBJECT src/vec_nonfma_dp.c)
        target_compile_options(vec_nonfma_dp-256B PRIVATE ${VEC256})
        set_target_properties(vec_nonfma_dp-256B PROPERTIES OUTPUT_NAME "vec_nonfma_dp-256B")

        if(FMA_CC_AVAIL AND FMA_AVAIL)
            list(APPEND VEC_OBJECTS $<TARGET_OBJECTS:vec_fma_hp-256B>
                                    $<TARGET_OBJECTS:vec_fma_sp-256B>
                                    $<TARGET_OBJECTS:vec_fma_dp-256B>)
            add_library(vec_fma_hp-256B OBJECT src/vec_fma_hp.c)
            target_compile_options(vec_fma_hp-256B PRIVATE ${VEC256_FMA})
            set_target_properties(vec_fma_hp-256B PROPERTIES OUTPUT_NAME "vec_fma_hp-256B")

            add_library(vec_fma_sp-256B OBJECT src/vec_fma_sp.c)
            target_compile_options(vec_fma_sp-256B PRIVATE ${VEC256_FMA})
            set_target_properties(vec_fma_sp-256B PROPERTIES OUTPUT_NAME "vec_fma_sp-256B")

            add_library(vec_fma_dp-256B OBJECT src/vec_fma_dp.c)
            target_compile_options(vec_fma_dp-256B PRIVATE ${VEC256_FMA})
            set_target_properties(vec_fma_dp-256B PROPERTIES OUTPUT_NAME "vec_fma_dp-256B")
        endif()
    endif()

    # Conditionally add AVX512 dependencies.
    if(AVX512_CC_AVAIL AND AVX512_AVAIL)
        list(APPEND VEC_OBJECTS $<TARGET_OBJECTS:vec_nonfma_hp-512B>
                                $<TARGET_OBJECTS:vec_nonfma_sp-512B>
                                $<TARGET_OBJECTS:vec_nonfma_dp-512B>)
        add_library(vec_nonfma_hp-512B OBJECT src/vec_nonfma_hp.c)
        target_compile_options(vec_nonfma_hp-512B PRIVATE ${VEC512})
        set_target_properties(vec_nonfma_hp-512B PROPERTIES OUTPUT_NAME "vec_nonfma_hp-512B")

        add_library(vec_nonfma_sp-512B OBJECT src/vec_nonfma_sp.c)
        target_compile_options(vec_nonfma_sp-512B PRIVATE ${VEC512})
        set_target_properties(vec_nonfma_sp-512B PROPERTIES OUTPUT_NAME "vec_nonfma_sp-512B")

        add_library(vec_nonfma_dp-512B OBJECT src/vec_nonfma_dp.c)
        target_compile_options(vec_nonfma_dp-512B PRIVATE ${VEC512})
        set_target_properties(vec_nonfma_dp-512B PROPERTIES OUTPUT_NAME "vec_nonfma_dp-512B")

        if(FMA_CC_AVAIL AND FMA_AVAIL)
            list(APPEND VEC_OBJECTS $<TARGET_OBJECTS:vec_fma_hp-512B>
                                    $<TARGET_OBJECTS:vec_fma_sp-512B>
                                    $<TARGET_OBJECTS:vec_fma_dp-512B>)
            add_library(vec_fma_hp-512B OBJECT src/vec_fma_hp.c)
            target_compile_options(vec_fma_hp-512B PRIVATE ${VEC512_FMA})
            set_target_properties(vec_fma_hp-512B PROPERTIES OUTPUT_NAME "vec_fma_hp-512B")

            add_library(vec_fma_sp-512B OBJECT src/vec_fma_sp.c)
            target_compile_options(vec_fma_sp-512B PRIVATE ${VEC512_FMA})
            set_target_properties(vec_fma_sp-512B PROPERTIES OUTPUT_NAME "vec_fma_sp-512B")

            add_library(vec_fma_dp-512B OBJECT src/vec_fma_dp.c)
            target_compile_options(vec_fma_dp-512B PRIVATE ${VEC512_FMA})
            set_target_properties(vec_fma_dp-512B PROPERTIES OUTPUT_NAME "vec_fma_dp-512B")
        endif()
    endif()
elseif(UNAME_M STREQUAL "aarch64")
    set(ARCH "ARM" CACHE STRING "Architecture identifier in CAT")
    message(STATUS "Setting ARCH=${ARCH}")
    set(FLOP -march=armv8.2-a+fp16 -DARM)
    set(VEC -march=armv8.2-a+fp16 -O0 -DARM)
    set(VEC_FMA -march=armv8.2-a+fp16 -O0 -DARM)
    set(VEC_ALL ${VEC} -O0 -DARM)
    list(APPEND VEC_OBJECTS $<TARGET_OBJECTS:vec_fma_hp>
                            $<TARGET_OBJECTS:vec_fma_sp>
                            $<TARGET_OBJECTS:vec_fma_dp>
                            $<TARGET_OBJECTS:vec_nonfma_hp>
                            $<TARGET_OBJECTS:vec_nonfma_sp>
                            $<TARGET_OBJECTS:vec_nonfma_dp>)
    add_library(vec_fma_hp OBJECT src/vec_fma_hp.c)
    target_compile_options(vec_fma_hp PRIVATE ${VEC_FMA})

    add_library(vec_fma_sp OBJECT src/vec_fma_sp.c)
    target_compile_options(vec_fma_sp PRIVATE ${VEC_FMA})

    add_library(vec_fma_dp OBJECT src/vec_fma_dp.c)
    target_compile_options(vec_fma_dp PRIVATE ${VEC_FMA})

    add_library(vec_nonfma_hp OBJECT src/vec_nonfma_hp.c)
    target_compile_options(vec_nonfma_hp PRIVATE ${VEC})

    add_library(vec_nonfma_sp OBJECT src/vec_nonfma_sp.c)
    target_compile_options(vec_nonfma_sp PRIVATE ${VEC})

    add_library(vec_nonfma_dp OBJECT src/vec_nonfma_dp.c)
    target_compile_options(vec_nonfma_dp PRIVATE ${VEC})
elseif(UNAME_M STREQUAL "ppc64le")
    set(ARCH "POWER" CACHE STRING "Architecture identifier in CAT")
    message(STATUS "Setting ARCH=${ARCH}")
    set(FLOP -maltivec -DPOWER)
    set(VEC -maltivec -DPOWER)
    set(VEC_FMA -maltivec -DPOWER)
    set(VEC_ALL ${VEC} -DPOWER)
    list(APPEND VEC_OBJECTS $<TARGET_OBJECTS:vec_fma_hp>
                            $<TARGET_OBJECTS:vec_fma_sp>
                            $<TARGET_OBJECTS:vec_fma_dp>
                            $<TARGET_OBJECTS:vec_nonfma_hp>
                            $<TARGET_OBJECTS:vec_nonfma_sp>
                            $<TARGET_OBJECTS:vec_nonfma_dp>)
    add_library(vec_fma_hp OBJECT src/vec_fma_hp.c)
    target_compile_options(vec_fma_hp PRIVATE ${VEC_FMA})

    add_library(vec_fma_sp OBJECT src/vec_fma_sp.c)
    target_compile_options(vec_fma_sp PRIVATE ${VEC_FMA})

    add_library(vec_fma_dp OBJECT src/vec_fma_dp.c)
    target_compile_options(vec_fma_dp PRIVATE ${VEC_FMA})

    add_library(vec_nonfma_hp OBJECT src/vec_nonfma_hp.c)
    target_compile_options(vec_nonfma_hp PRIVATE ${VEC})

    add_library(vec_nonfma_sp OBJECT src/vec_nonfma_sp.c)
    target_compile_options(vec_nonfma_sp PRIVATE ${VEC})

    add_library(vec_nonfma_dp OBJECT src/vec_nonfma_dp.c)
    target_compile_options(vec_nonfma_dp PRIVATE ${VEC})
else()
    message( FATAL_ERROR "Detected arch ${UNAME_M} is not supported!" )
endif()

#-------------------------------------------------------------------------------
# Main target.
add_executable(cat_collect src/main.c)

#-------------------------------------------------------------------------------
# Branch benchmarks target.
add_library(branch OBJECT src/branch.c)
target_compile_options(branch PRIVATE -O0)

#-------------------------------------------------------------------------------
# Data cache benchmarks targets.
add_library(timing_kernels OBJECT src/timing_kernels.c)
target_compile_options(timing_kernels PRIVATE -O2)

add_library(prepareArray OBJECT src/prepareArray.c)
target_compile_options(prepareArray PRIVATE -O2)

add_library(compar OBJECT src/compar.c)
target_compile_options(compar PRIVATE -O2)

add_library(dcache OBJECT src/dcache.c)
target_compile_options(dcache PRIVATE -O2)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.cat_cfg
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/src/.cat_cfg ${CMAKE_CURRENT_BINARY_DIR}/.cat_cfg
    VERBATIM
)

#-------------------------------------------------------------------------------
# Event stock target.
add_library(eventstock OBJECT src/eventstock.c)
target_compile_options(eventstock PRIVATE -O0)

#-------------------------------------------------------------------------------
# FLOPs benchmark target.
add_library(flops OBJECT src/flops.c)
target_compile_options(flops PRIVATE -O1 ${FLOP})

#-------------------------------------------------------------------------------
# Instructions benchmark target. Legacy Makefile used manual silence.
add_library(instructions OBJECT src/instructions.c)
target_compile_options(instructions PRIVATE -O2 -ftree-vectorize ${FLOP} ${INSTR})

#-------------------------------------------------------------------------------
# Weak symbols target.
add_library(weak_symbols OBJECT src/weak_symbols.c)

#-------------------------------------------------------------------------------
# Instruction cache targets.
add_library(icache OBJECT src/icache.c)
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/src/icache_seq.h
           ${CMAKE_SOURCE_DIR}/src/icache_seq.c
           ${CMAKE_SOURCE_DIR}/src/icache_seq_kernel.c
    COMMAND bash ${CMAKE_SOURCE_DIR}/src/gen_seq_dlopen.sh ${CMAKE_SOURCE_DIR}/src
    DEPENDS ${CMAKE_SOURCE_DIR}/src/gen_seq_dlopen.sh
    VERBATIM
)
target_compile_options(icache PRIVATE -O0)

add_library(icache_seq OBJECT ${CMAKE_SOURCE_DIR}/src/icache_seq.c)
target_compile_options(icache_seq PRIVATE -O0)

add_library(icache_seq_kernel OBJECT ${CMAKE_SOURCE_DIR}/src/icache_seq_kernel.c)
target_compile_options(icache_seq_kernel PRIVATE -O0)
set_target_properties(icache_seq_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(icache_seq_kernel_shared SHARED $<TARGET_OBJECTS:icache_seq_kernel>)
target_compile_options(icache_seq_kernel_shared PRIVATE -O0)
set_target_properties(icache_seq_kernel_shared PROPERTIES OUTPUT_NAME "icache_seq_kernel_0" PREFIX "")
add_custom_command(
    TARGET cat_collect
    COMMAND bash ${CMAKE_SOURCE_DIR}/src/replicate.sh ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_SOURCE_DIR}/src/replicate.sh
    VERBATIM
)

#-------------------------------------------------------------------------------
# Vector FLOPs targets.
add_library(vec OBJECT src/vec.c)
target_compile_options(vec PRIVATE -D${ARCH} ${VEC_META})

add_library(scalar OBJECT src/vec_scalar_verify.c)
target_compile_options(scalar PRIVATE ${VEC_ALL})

#-------------------------------------------------------------------------------
# Formatting.
option( color "Use ANSI color output" true )

#-------------------------------------------------------------------------------
# Default compiler flags.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -Wextra")

#-------------------------------------------------------------------------------
# PAPI -- required.
message( "" )
message( "-------------------- PAPI --------------------" )
if(NOT DEFINED PAPI_DIR AND NOT DEFINED ENV{PAPI_DIR})
    message( FATAL_ERROR "PAPI is required" )
else()
    set(PAPI_DIR $ENV{PAPI_DIR} CACHE STRING "Location of PAPI installation")
    message( STATUS "PAPI_DIR = ${PAPI_DIR}" )
    target_link_directories(cat_collect PRIVATE ${PAPI_DIR}/lib)
    target_link_libraries(cat_collect PRIVATE papi)
    include_directories(${PAPI_DIR}/include)
endif()

#-------------------------------------------------------------------------------
# OpenMP -- required.
message( "" )
message( "------------------- OpenMP -------------------" )
find_package( OpenMP REQUIRED )
if(DEFINED CACHE{CAT_OMP_STAT})
    message( STATUS "Using cached OpenMP info." )
elseif(OpenMP_C_FOUND)
    set(CAT_OMP_STAT ON CACHE BOOL "Whether or not OpenMP was found")
else()
    message( FATAL_ERROR "OpenMP is required" )
endif()
target_link_libraries( cat_collect PUBLIC OpenMP::OpenMP_C)
target_link_libraries( timing_kernels PUBLIC OpenMP::OpenMP_C )
target_link_libraries( dcache PUBLIC OpenMP::OpenMP_C )

#-------------------------------------------------------------------------------
# MPI support -- optional.
set(USE_MPI OFF CACHE BOOL "Enable MPI support")
message( "" )
message( "-------------------- MPI ---------------------" )
set(USE_MPI ${USE_MPI})
message( STATUS "USE_MPI=${USE_MPI}" )
if(NOT USE_MPI)
    message( STATUS "User has not requested to use MPI" )
else()
    set(MPI_DETERMINE_LIBRARY_VERSION true)
    find_package( MPI COMPONENTS C )
    if (MPI_FOUND)
        get_target_property( mpi_lib MPI::MPI_C INTERFACE_LINK_LIBRARIES )
        message( STATUS "Found MPI: ${mpi_lib}" )

        target_link_libraries( cat_collect PUBLIC MPI::MPI_C )
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_MPI")

        # Remove Open MPI flags that are incompatible with nvcc (-fexceptions).
        get_target_property( mpi_c_defines MPI::MPI_C INTERFACE_COMPILE_OPTIONS )
        message( DEBUG "mpi_c_defines = '${mpi_c_defines}'" )
        list( REMOVE_ITEM mpi_c_defines "-fexceptions" )
        message( DEBUG "mpi_c_defines = '${mpi_c_defines}'" )
        set_target_properties( MPI::MPI_C PROPERTIES INTERFACE_COMPILE_OPTIONS "${mpi_c_defines}" )
    endif()
endif()
message( "" )

#-------------------------------------------------------------------------------
# Ensure the sample events list file is copied by "using" the result.
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/event_list.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/src/event_list.txt ${CMAKE_CURRENT_BINARY_DIR}/event_list.txt
    VERBATIM
)
add_custom_target(sample_input_file ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/event_list.txt)
add_dependencies(cat_collect sample_input_file)

#-------------------------------------------------------------------------------
# Ensure the shared objects get generated by "using" the result.
# Otherwise, the target that generates them is inert.
file(GLOB shared_files "${CMAKE_CURRENT_BINARY_DIR}/icache_seq_kernel_*.so")
add_custom_target(shared_objects ALL DEPENDS ${shared_files})
add_dependencies(shared_objects icache_seq_kernel_shared)
add_dependencies(cat_collect shared_objects)

#-------------------------------------------------------------------------------
# Ensure the config file is copied by "using" the result.
add_custom_target(cat_config_file ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/.cat_cfg)
add_dependencies(cat_collect cat_config_file)

#-------------------------------------------------------------------------------
# Link build directory because icache benchmarks use .so's.
target_link_directories(cat_collect PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

#-------------------------------------------------------------------------------
# Link all remaining object code.
target_link_libraries(cat_collect PRIVATE m pthread dl rt)
target_sources(cat_collect PRIVATE
                $<TARGET_OBJECTS:branch>
                $<TARGET_OBJECTS:timing_kernels>
                $<TARGET_OBJECTS:prepareArray>
                $<TARGET_OBJECTS:compar>
                $<TARGET_OBJECTS:dcache>
                $<TARGET_OBJECTS:eventstock>
                $<TARGET_OBJECTS:flops>
                $<TARGET_OBJECTS:instructions>
                $<TARGET_OBJECTS:weak_symbols>
                $<TARGET_OBJECTS:icache>
                $<TARGET_OBJECTS:icache_seq>
                $<TARGET_OBJECTS:vec>
                $<TARGET_OBJECTS:scalar>
                ${VEC_OBJECTS})
