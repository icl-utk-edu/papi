cmake_minimum_required (VERSION 3.12)

project (PAPI C)

# Allow targets from different subdirectories to be linked together
cmake_policy (SET CMP0079 NEW)

set (PAPI_VERSION_MAJOR 6)
set (PAPI_VERSION_MINOR 0)
set (PAPI_VERSION_PATCH 0)

include(CMakePrintHelpers)

set (CMAKE_NO_SYSTEM_FROM_IMPORTED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/)

if (NOT DEFINED papi)
    add_library(papi OBJECT)
endif (NOT DEFINED papi)

# try_run doesn't work in cross-compilation
# it produces the test bin for the target but cannot run it (except with CMAKE_CROSSCOMPILING_EMULATOR see kitware doc)
# instead, itgenerated cache variable of what would have been the output. user should provide the values.

# should we pursue the try_run path, or go for toolchain per architecture
# there are not so many arch, it's only a handful of toolchain files to maintain

# user option, bitmode
set (BITMODE "64" CACHE STRING "Specifiy bit mode of library (32|64)")
set_property (CACHE BITMODE PROPERTY STRINGS 32 64)

# user option, architecture
# default value should be a uname -m
set (ARCH "" CACHE STRING "Specify architecture (uname -m)")
if (ARCH STREQUAL "")
    if (CMAKE_UNAME)
        execute_process (
            COMMAND ${CMAKE_UNAME} -p
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE OSVERSION)
    else (CMAKE_UNAME)
        message (FATAL_ERROR "Cannot decide for target architecture without an hint or uname -p")
    endif (CMAKE_UNAME)
endif (ARCH STREQUAL "")

# user option, OS for cross-compiling
# default value should be a uname | tr '[A-Z]' '[a-z]'
# if overwrote by user, activate cross-compiling
set (TARGET_OS "" CACHE STRING "Specify target operating system for cross-compiling. Leave blank otherwise.")
if (TARGET_OS STREQUAL "")
    if (CMAKE_UNAME)
        execute_process (
            COMMAND ${CMAKE_UNAME}
            COMMAND tr '[A-Z]' '[a-z]'
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE _OS)
    endif (CMAKE_UNAME)
else (TARGET_OS STREQUAL "")
    set (_OS ${TARGET_OS} CACHE STRING "" FORCE)
    set (CMAKE_SYSTEM_NAME ${TARGET_OS} CACHE STRING "Cross-compiling target system" FORCE)
endif (TARGET_OS STREQUAL "")
if (_OS STREQUAL "SunOS" OR _OS STREQUAL "sunos")
    set (_OS "solaris" CACHE INTERNAL "" FORCE)
endif (_OS STREQUAL "SunOS" OR _OS STREQUAL "sunos")

# user option, OS version
# default value should be a uname -r on non-BlueGene
set (TARGET_OS_VERSION "" CACHE STRING "Specify target operating system version for cross-compiling. You must provide it if you set a TARGET_OS. Leave blank otherwise")
if (TARGET_OS_VERSION STREQUAL "")
    if (CMAKE_UNAME)
        execute_process (
            COMMAND ${CMAKE_UNAME} -p
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE _OSVERSION)
    endif (CMAKE_UNAME)
else (TARGET_OS_VERSION STREQUAL "")
    set (_OSVERSION ${TARGET_OS_VERSION} CACHE STRING "" FORCE)
    set (CMAKE_SYSTEM_VERSION ${TARGET_OS_VERSION} CACHE STRING "Cross-compiling target system version" FORCE)
endif (TARGET_OS_VERSION STREQUAL "")

# user option, OS assumed version
set (ASSUMED_KERNEL "" CACHE STRING "Assume kernel version is <ver> for purposes of workarounds")
if (ASSUMED_KERNEL STREQUAL "")
    target_compile_definitions(papi PRIVATE ASSUME_KERNEL=\"${ASSUMED_KERNEL}\")
endif (ASSUMED_KERNEL STREQUAL "")

#message(STATUS "Found: _OS=${_OS}; _OSVERSION=${_OSVERSION}")

# user option, target > mic
option (TARGET_MIC "Compile PAPI for Intel MIC" OFF)
if (TARGET_MIC)
    set (MIC             yes               CACHE INTERNAL "Choosing MIC forces that entry" FORCE)
    set (TLS             __thread          CACHE INTERNAL "Choosing MIC forces that entry" FORCE)
    set (VIRTUALTIMER    cputime_id        CACHE INTERNAL "Choosing MIC forces that entry" FORCE)
    set (PERF_EVENTS     yes               CACHE INTERNAL "Choosing MIC forces that entry" FORCE)
    set (WALLTIMER       clock_realtime_hr CACHE INTERNAL "Choosing MIC forces that entry" FORCE)
    set (FFSLL           no                CACHE INTERNAL "Choosing MIC forces that entry" FORCE)
    set (CROSS_COMPILING yes               CACHE INTERNAL "Choosing MIC forces that entry" FORCE)
    set (ARCH            k1om              CACHE INTERNAL "Enabling MIC forces that entry" FORCE)
else (TARGET_MIC)
    set (MIC             no                CACHE INTERNAL "Disabling TARGET_MIC forces that entry" FORCE)
endif (TARGET_MIC)

set (CC_COMMON_NAME
    $<$<C_COMPILER_ID:AppleClang,Clang>:"clang">
    $<$<C_COMPILER_ID:GNU>:"gcc">
    $<$<C_COMPILER_ID:Intel>:"icc">
    $<$<C_COMPILER_ID:XL>:"xlc">)

target_compile_options(papi
    PRIVATE
    $<$<C_COMPILER_ID:Intel>:"-diag-disable 188,869,271">
    $<$<AND:$<C_COMPILER_ID:Intel>,$<BOOL:${MIC}>>:"-mmic">)

set_target_properties(papi PROPERTIES
    $<$<AND:$<C_COMPILER_ID:Intel>,$<BOOL:${MIC}>>:POSITION_INDEPENDENT_CODE ON>)

#message (STATUS "MIC=${MIC}")

include (CMakePushCheckState)
# All the checks we might need
include (CheckCCompilerFlag)
include (CheckCSourceCompiles)
include (CheckCSourceRuns)
include (CheckIncludeFiles)
include (CheckLibraryExists)
include (CheckStructHasMember)
include (CheckSymbolExists)
include (CheckTypeSize)


# sys/time.h c_asm.h intrinsics.h mach/mach_time.h sched.h
check_include_files ("sys/time.h" HAVE_SYS_TIME_H)
check_include_files ("sys/stat.h" HAVE_SYS_STAT_H)
check_include_files ("sys/types.h" HAVE_SYS_TYPES_H)
check_include_files ("stdint.h" HAVE_STDINT_H)
check_include_files ("stdlib.h" HAVE_STDLIB_H)
check_include_files ("strings.h" HAVE_STRINGS_H)
check_include_files ("string.h" HAVE_STRING_H)
check_include_files ("unistd.h" HAVE_UNISTD_H)
check_include_files (c_asm.h HAVE_C_ASM_H)
check_include_files ("intrinsics.h" HAVE_INTRINSICS_H)
check_include_files ("mach/mach_time.h" HAVE_MACH_MACH_TIME_H)
check_include_files (sched.h HAVE_SCHED_H)
check_include_files (inttypes.h HAVE_INTTYPES_H)
check_include_files (memory.h HAVE_MEMORY_H)

check_include_files ("sys/time.h;time.h" TIME_WITH_SYS_TIME)

#check_c_source_compiles ("#include <sys/time.h>
##include <time.h>
#int main (int argc, char *argv[]) { return 0; }" HAVE_TIME_WITH_SYS_TIME)

# gethrtime read_real_time time_base_to_time clock_gettime mach_absolute_time sched_getcpu
check_symbol_exists (gethrtime "sys/time.h" HAVE_GETHRTIME)
check_symbol_exists (read_real_time "sys/time.h" HAVE_READ_REAL_TIME)
check_symbol_exists (time_base_to_time "sys/time.h" HAVE_TIME_BASE_TO_TIME)
check_symbol_exists (clock_gettime "sys/time.h" HAVE_CLOCK_GETTIME)
check_symbol_exists (mach_absolute_time "inttypes.h;mach/mach_time.h" HAVE_MACH_ABSOLUTE_TIME)
#check_symbol_exists (sched_getcpu "sched.h" HAVE_SCHED_GETCPU)
check_c_source_compiles ("#include <sched.h>
int main (int argc, char *argv[]) {
    return sched_getcpu();
}" HAVE_SCHED_GETCPU)

# those flags that have a value without any proper detection
set (_ALL_SOURCE 1)
set (_GNU_SOURCE 1)
set (_POSIX_PTHREAD_SEMANTICS 1)
set (_TANDEM_SOURCE 1)
set (__EXTENSIONS__ 1)



#message (STATUS "HAVE_SYS_TIME_H=${HAVE_SYS_TIME_H}, HAVE_C_ASM_H=${HAVE_C_ASM_H}, HAVE_INTRINSICS_H=${HAVE_INTRINSICS_H}, HAVE_MACH_MACH_TIME_H=${HAVE_MACH_MACH_TIME_H}")
#message (STATUS "HAVE_GETHRTIME=${HAVE_GETHRTIME}, HAVE_READ_REAL_TIME=${HAVE_READ_REAL_TIME}, HAVE_TIME_BASE_TO_TIME=${HAVE_TIME_BASE_TO_TIME}, HAVE_CLOCK_GETTIME=${HAVE_CLOCK_GETTIME}")
#message (STATUS "HAVE_MEMORY_H=${HAVE_MEMORY_H} HAVE_INTTYPES_H=${HAVE_INTTYPES_H}, HAVE_MACH_ABSOLUTE_TIME=${HAVE_MACH_ABSOLUTE_TIME}, HAVE_SCHED_GETCPU=${HAVE_SCHED_GETCPU}")

# check how dl is provided
cmake_push_check_state ()
check_c_source_compiles ("#include <dlfcn.h>
    int main(int argc, char *argv) {
    void *p = dlopen(\"\", 0);

    return 0; }" HAVE_DLOPEN_WOLINK)
if (NOT HAVE_DLOPEN_WOLINK)
    set (CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS})
    check_c_source_compiles ("#include <dlfcn.h>
        int main(int argc, char *argv) {
        void *p = dlopen(\"\", 0);
        return 0; }" HAVE_DLOPEN_LINK)
    if (HAVE_DLOPEN_LINK)
        target_link_libraries(papi PUBLIC ${CMAKE_DL_LIBS})
        if (TARGET papi_shared)
            target_link_libraries(papi_shared PUBLIC ${CMAKE_DL_LIBS})
        endif (TARGET papi_shared)
        if (TARGET papi_static)
            target_link_libraries(papi_static PUBLIC ${CMAKE_DL_LIBS})
        endif (TARGET papi_static)
    else (HAVE_DLOPEN_LINK)
        message (FATAL_ERROR "Missing dl library")
    endif (HAVE_DLOPEN_LINK)
endif (NOT HAVE_DLOPEN_WOLINK)
cmake_pop_check_state()

#message (STATUS "HAVE_DLOPEN_WOLINK=${HAVE_DLOPEN_WOLINK}, HAVE_DLOPEN_LINK=${HAVE_DLOPEN_LINK}")

try_run (TARGET_RUN_X86 TARGET_COMPILE_X86
    ${CMAKE_CURRENT_BINARY_DIR}/platform
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_x86.c)
if (1 EQUAL TARGET_RUN_X86)
    message (STATUS "Found Architecture X86")
    target_sources(papi PRIVATE ${})
endif (1 EQUAL TARGET_RUN_X86)

#message (STATUS "TARGET_RUN_X86=${TARGET_RUN_X86}, TARGET_COMPILE_X86=${TARGET_COMPILE_X86}")

##############################################################
# configure.in:206 Linux
if (_OS STREQUAL "linux")
    if (ARCH STREQUAL "ppc64" OR ARCH STREQUAL "x86_64")
        if (BITMODE STREQUAL "64" AND ${CMAKE_INSTALL_PREFIX}/lib STREQUAL ${CMAKE_INSTALL_LIBDIR})
            set (CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib64 CACHE INTERNAL "" FORCE)
        endif (BITMODE STREQUAL "64" AND ${CMAKE_INSTALL_PREFIX}/lib STREQUAL ${CMAKE_INSTALL_LIBDIR})
    endif (ARCH STREQUAL "ppc64" OR ARCH STREQUAL "x86_64")
endif (_OS STREQUAL "linux")

#message (STATUS "CMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}")

##############################################################
# Enable testing
option (WITH_TESTS "Build PAPI tests" OFF)
if (WITH_TESTS)
    enable_testing()
endif (WITH_TESTS)

##############################################################
# Memory debugger mode
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
    option (WITH_MEMORY_DEBUG "Build PAPI in debug mode with the memory tracker" OFF)
endif (CMAKE_BUILD_TYPE MATCHES DEBUG)

target_compile_definitions(papi
    PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<NOT:$<BOOL:${WITH_MEMORY_DEBUG}>>:PAPI_NO_MEMORY_MANAGEMENT>)

##############################################################
# Check gcc version 3.4.0 or more, adjust the Wextra flag, set the override init for libpfm4
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_C_COMPILER_VERSION VERSION_GREATER "3.4.0")
        target_compile_options(papi PRIVATE -Wextra)
    else (CMAKE_C_COMPILER_VERSION VERSION_GREATER "3.4.0")
        target_compile_options(papi PRIVATE -W)
    endif (CMAKE_C_COMPILER_VERSION VERSION_GREATER "3.4.0")

    check_c_compiler_flag("-Wno-override-init" HAVE_NO_OVERRIDE_INIT)
    #alternative push_state + _flags -Werror + source code configure.in:278
endif (CMAKE_C_COMPILER_ID STREQUAL "GNU")

##############################################################
# CPU, user option
set (TARGET_CPU "" CACHE STRING "Specify target CPU for cross-compiling. Leave blank otherwise.")
if (TARGET_CPU STREQUAL "core" OR TARGET_CPU STREQUAL "core2" OR TARGET_CPU STREQUAL "i7"
        OR TARGET_CPU STREQUAL "atom" OR TARGET_CPU STREQUAL "p4" OR TARGET_CPU STREQUAL "p3"
        OR TARGET_CPU STREQUAL "opteron" OR TARGET_CPU STREQUAL "athlon")
    target_sources(papi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/x86_cpuid_info.c)
else () # CPU is not specified -OR- recognized
    # Are we on aix
    if (_OS STREQUAL "aix")
        execute_process (
            COMMAND /usr/sbin/lsattr -E -l proc0
            COMMAND grep type
            COMMAND cut -d '_' -f 2
            COMMAND cut -d ' ' -f 1
            COMMAND tr '[A-Z]' '[a-z]'
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE _CPU)
        if (NOT _CPU)
            execute_process (
                COMMAND /usr/sbin/lsattr -E -l proc1
                COMMAND grep type
                COMMAND cut -d '_' -f 2
                COMMAND cut -d ' ' -f 1
                COMMAND tr '[A-Z]' '[a-z]'
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE _CPU)
        endif (NOT _CPU)
    endif (_OS STREQUAL "aix")
    # Are we on freebsd
    if (_OS STREQUAL "freebsd")
        if (CMAKE_UNAME)
            execute_process (
                COMMAND ${CMAKE_UNAME} -m
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE _FAMILY)
            if (_FAMILY STREQUAL "amd64")
                target_sources(papi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/x86_cpuid_info.c)
            endif(_FAMILY STREQUAL "amd64")

            if (_FAMILY STREQUAL "i386")
                target_sources(papi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/x86_cpuid_info.c)
            endif (_FAMILY STREQUAL "i386")

        else (CMAKE_UNAME)
            message (FATAL_ERROR "Cannot decide on the target CPU without an hint or uname")
        endif (CMAKE_UNAME)
    endif (_OS STREQUAL "freebsd")

    # Are we on mac
    if (_OS STREQUAL "darwin")
        if (CMAKE_UNAME)
            execute_process (
                COMMAND ${CMAKE_UNAME} -m
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE _FAMILY)
            target_sources(papi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/x86_cpuid_info.c)
        else (CMAKE_UNAME)
            message (FATAL_ERROR "Cannot decide on the target CPU without an hint or uname")
        endif (CMAKE_UNAME)

    endif (_OS STREQUAL "darwin")

    # Are we on linux
    if (_OS STREQUAL "linux")
        if (CMAKE_UNAME)
            execute_process (
                COMMAND ${CMAKE_UNAME} -m
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE _FAMILY)
            if (_FAMILY STREQUAL "x86_64")
                target_sources(papi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/x86_cpuid_info.c)
                set (_CPU "x86" CACHE INTERNAL "" FORCE)
            endif(_FAMILY STREQUAL "x86_64")

            if (_FAMILY STREQUAL "i686")
                target_sources(papi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/x86_cpuid_info.c)
                set (_CPU "x86" CACHE INTERNAL "" FORCE)
            endif (_FAMILY STREQUAL "i686")

            if (_FAMILY STREQUAL "ppc64")
                execute_process (
                    COMMAND cat /proc/cpuinfo
                    COMMAND grep cpu
                    COMMAND cut -d: -f2
                    COMMAND cut -d' ' -f2
                    COMMAND sed '2,$d'
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _CPU_INFO)
                if (_CPU_INFO STREQUAL "PPC970*")
                    set (_CPU "PPC970" CACHE INTERNAL "" FORCE)
                else (  )
                    set (_CPU ${_CPU_INFO} CACHE INTERNAL "" FORCE)
                endif (_CPU_INFO STREQUAL "PPC970*")
            endif (_FAMILY STREQUAL "ppc64")
        else (CMAKE_UNAME)
            message (FATAL_ERROR "Cannot decide on the target CPU without an hint or uname")
        endif (CMAKE_UNAME)
    endif (_OS STREQUAL "linux")

    # configure.in:341 Are we on Solaris
    if (_OS STREQUAL "solaris")
        try_run (RUN_CPC HAVE_CPC_V2 ${CMAKE_CURRENT_BINARY_DIR}/platform
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_cpc_version2.c
            LINK_LIBRARIES cpc)
        if (NOT COMPILE_CRC)
            message (FATAL_ERROR "Running on Solaris requires libcpc")
        endif (NOT COMPILE_CRC)

        if (HAVE_CPC_V2)
            set (HAVE_LIBCPC_H 1 CACHE INTERNAL "" FORCE)
            if (HAVE_CPC_V2 EQUAL 0)
                set (CPC_VERSION 2 CACHE INTERNAL "" FORCE)
            else (HAVE_CPC_V2 EQUAL 0)
                set (CPC_VERSION 0 CACHE INTERNAL "" FORCE)
            endif (HAVE_CPC_V2 EQUAL 0)
        endif (HAVE_CPC_V2)

        if (CMAKE_UNAME)
            execute_process (
                COMMAND ${CMAKE_UNAME} -p
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE _PROCESSOR)
            execute_process (
                COMMAND ${CMAKE_UNAME} -m
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE _FAMILY)

            if (_PROCESSOR STREQUAL "sparc")
                if (_FAMILY STREQUAL "sun4u")
                    set (_CPU "ultra" CACHE INTERNAL "" FORCE)
                        check_symbol_exists(cpc_take_sample "stdlib.h;libcpc.h"
                            HAVE_LIBCPC)
                        #try_run (RUN_CPC_SYM HAVE_CPC ${CMAKE_CURRENT_BINARY_DIR}/platform
                        #${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_cpc_symb.c
                        #LINK_LIBRARIES cpc)
                        if (NOT HAVE_LIBCPC) #FIXME: beware, config was looking for a symbol in libcpc.a object
                        message (FATAL_ERROR "libcpc.a is required on Solaris, install SUNWcpc.")
                    endif (NOT HAVE_LIBCPC)
                elseif (_FAMILY STREQUAL "sun4v")
                    set (_CPU "niagara2" CACHE INTERNAL "" FORCE)
                    if (NOT HAVE_CPC_V2 EQUAL 2)
                        message (FATAL_ERROR "PAPI requires libcpc version 2 for Niagara 2 processor")
                    endif (NOT HAVE_CPC_V2 EQUAL 2)
                else ()
                    message (FATAL_ERROR "${_FAMILY} is not supported (we support: uname -m -> sun4u | sun4v)")
                endif (_FAMILY STREQUAL "sun4v")
            endif (_PROCESSOR STREQUAL "sparc")
        endif (CMAKE_UNAME)
    endif (_OS STREQUAL "solaris")

    # Are we on BG/P
    if (_OS STREQUAL "bgp")
        set (_CPU "bgp" CACHE INTERNAL "" FORCE)
    endif (_OS STREQUAL "bgp")

    # Are we on BG/Q
    if (_OS STREQUAL "bgq")
        set (_CPU "bgq" CACHE INTERNAL "" FORCE)
    endif (_OS STREQUAL "bgq")

endif () # that giant OR expression

#message (STATUS "_FAMILY=${_FAMILY}, _CPU=${_CPU}")

##############################################################
# Based on arch, we force pthread_mutexes
if (ARCH STREQUAL "aarch64" OR ARCH MATCHES "^arm" OR ARCH MATCHES "^parisc")
    set (PTHREAD_MUTEXES ON CACHE INTERNAL "Forcing use of pthread_mutexes" FORCE)
else ()
    # And then let's see what the user actually wanted :)
    option (PTHREAD_MUTEXES "Specify use of pthread mutexes rather than custom PAPI locks" OFF)
endif (ARCH STREQUAL "aarch64" OR ARCH MATCHES "^arm" OR ARCH MATCHES "^parisc")

if (PTHREAD_MUTEXES)
    target_compile_definitions(papi PRIVATE USE_PTHREAD_MUTEXES)
endif (PTHREAD_MUTEXES)

##############################################################
option (WITH_FFSLL "Specify use of the ffsll() function. Required for cross-compilation" ON)

if (WITH_FFSLL)
    #see configure.in:397
    set (FFSLL  yes)
    set (HAVE_FFSLL 1)
else (WITH_FFSLL)
    if (CROSS_COMPILING)
        message (STATUS "Cross-compilation requires ffsll.")
        check_symbol_exists(ffsll "" HAVE_FFSLL)
        if (HAVE_FFSLL)
            set (FFSLL  yes)
        endif (HAVE_FFSLL)
    endif (CROSS_COMPILING)
endif (WITH_FFSLL)

##############################################################
# configure.in:417
#check_symbol_exists(gettid "sys/types.h" HAVE_GETTID)
#if (NOT HAVE_GETTID)
check_c_source_compiles("#include <sys/types.h>
    int main() { pid_t a = gettid(); return 0; }" HAVE_GETTID)
if (NOT HAVE_GETTID)
    check_c_source_compiles("#include <sys/types.h>
        #include <sys/syscall.h>
        int main() { pid_t a = syscall(SYS_gettid); return 0; }" HAVE_SYSCALL_GETTID)
endif (NOT HAVE_GETTID)
#endif (NOT HAVE_GETTID)

#message (STATUS "HAVE_GETTID=${HAVE_GETTID}, HAVE_SYSCALL_GETTID=${HAVE_SYSCALL_GETTID}")

##############################################################
# configure.in:430 About that walltimer
set (WITH_WALLTIMER CACHE STRING "Specify realtime timer. Possible values are: gettimeofday,mmtimer,clock_realtime_hr,clock_realtime,cycle" )
set_property(CACHE WITH_WALLTIMER PROPERTY STRINGS gettimeofday mmtimer clock_realtime_hr clock_realtime cycle)

if (WITH_WALLTIMER STREQUAL "")
    if (CROSS_COMPILING)
        message (FATAL_ERROR "walltimer must be specified for cross compile")
    endif (CROSS_COMPILING)

    try_run(HAVE_MMTIMER_RUN HAVE_MMTIMER_COMPILE
        ${CMAKE_CURRENT_BINARY_DIR}/platform
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_mmtimer.c)

    if (HAVE_MMTIMER_RUN STREQUAL "0")
        set (WITH_WALLTIMER "mmtimer" CACHE STRING "" FORCE)
    else (HAVE_MMTIMER_RUN STREQUAL "0")

        try_run(HAVE_CLOCK_REALTIME_HR_RUN HAVE_CLOCK_REALTIME_HR_COMPILE
            ${CMAKE_CURRENT_BINARY_DIR}/platform
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_clock_realtime_hr.c)

        if (HAVE_CLOCK_REALTIME_HR_RUN STREQUAL "0")
            set (WITH_WALLTIMER "clock_realtime_hr" CACHE STRING "" FORCE)
        else (HAVE_CLOCK_REALTIME_HR_RUN STREQUAL "0")

            try_run(HAVE_CLOCK_REALTIME_RUN HAVE_CLOCK_REALTIME_COMPILE
                ${CMAKE_CURRENT_BINARY_DIR}/platform
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_clock_realtime.c)

            if (HAVE_CLOCK_REALTIME_RUN STREQUAL "0")
                set (WITH_WALLTIMER "clock_realtime" CACHE STRING "" FORCE)
            else (HAVE_CLOCK_REALTIME_RUN STREQUAL "0")
                set (WITH_WALLTIMER "cycle" CACHE STRING "" FORCE)
            endif (HAVE_CLOCK_REALTIME_RUN STREQUAL "0")
        endif (HAVE_CLOCK_REALTIME_HR_RUN STREQUAL "0")
    endif (HAVE_MMTIMER_RUN STREQUAL "0")
endif (WITH_WALLTIMER STREQUAL "")

if (WITH_WALLTIMER STREQUAL "gettimeofday")
    set (HAVE_GETTIMEOFDAY 1 CACHE INTERNAL "" FORCE)
endif (WITH_WALLTIMER STREQUAL "gettimeofday")

if (WITH_WALLTIMER STREQUAL "mmtimer")
    set (HAVE_MMTIMER 1 CACHE INTERNAL "" FORCE)
    #FIXME: be sure it goes with papi
    target_compile_definitions(papi PRIVATE ALTIX)
endif (WITH_WALLTIMER STREQUAL "mmtimer")

if (WITH_WALLTIMER STREQUAL "clock_realtime_hr")
    set (HAVE_CLOCK_GETTIME 1 CACHE INTERNAL "" FORCE)
    set (HAVE_CLOCK_GETTIME_REALTIME_HR 1 CACHE STRING "" FORCE)
endif (WITH_WALLTIMER STREQUAL "clock_realtime_hr")

if (WITH_WALLTIMER STREQUAL "clock_realtime")
    set (HAVE_CLOCK_GETTIME 1 CACHE INTERNAL "" FORCE)
    set (HAVE_CLOCK_GETTIME_REALTIME 1 CACHE STRING "" FORCE)
endif (WITH_WALLTIMER STREQUAL "clock_realtime")

if (WITH_WALLTIMER STREQUAL "cycle")
    set (HAVE_CYCLE 1 CACHE INTERNAL "" FORCE)
endif (WITH_WALLTIMER STREQUAL "cycle")

#message (STATUS "WITH_WALLTIMER=${WITH_WALLTIMER}, HAVE_GETTIMEOFDAY=${HAVE_GETTIMEOFDAY}, HAVE_MMTIMER=${HAVE_MMTIMER}, HAVE_CLOCK_GETTIME=${HAVE_CLOCK_GETTIME}, HAVE_CLOCK_GETTIME_REALTIME_HR=${HAVE_CLOCK_GETTIME_REALTIME_HR}, HAVE_CLOCK_GETTIME_REALTIME=${HAVE_CLOCK_GETTIME_REALTIME}, HAVE_CYCLE=${HAVE_CYCLE}")

##############################################################
# configure.in:528 Let's talk tls now
set (WITH_TLS "" CACHE STRING "This platform supports thread local storage with a keyword")

if (WITH_TLS STREQUAL "")
    if (CROSS_COMPILING)
        message (FATAL_ERROR "tls must be specified for cross compile")
    endif (CROSS_COMPILING)
    try_run (HAVE___THREAD_RUN HAVE___THREAD_COMPILE
        ${CMAKE_CURRENT_BINARY_DIR}/platform
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check___thread.c
        CMAKE_FLAGS -DLINK_LIBRARIES:STRING=-pthread)

    if (HAVE___THREAD_RUN STREQUAL "0")
        set (TLS __thread)
    else (HAVE___THREAD_RUN STREQUAL "0")
        set (TLS no)
    endif (HAVE___THREAD_RUN STREQUAL "0")

    if (_OS STREQUAL "linux")
        if (TLS STREQUAL "__thread")
            # On some linux distributions, TLS works in executables, but linking against
            # a shared library containing TLS fails with: undefined reference to __tls_get_addr
            execute_process (
                COMMAND ${CMAKE_C_COMPILER} -fPIC --shared -o check_tls_get_addr.so  ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_tls_get_addr.c)
            execute_process (
                COMMAND ${CMAKE_C_COMPILER} -o check_tls_get_addr check_tls_get_addr.so
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_VARIABLE TLS_LINK_FAILED)
            if (TLS_LINK_FAILED)
                set (TLS no)
            endif(TLS_LINK_FAILED)
        endif (TLS STREQUAL "__thread")
    endif (_OS STREQUAL "linux")
else (WITH_TLS STREQUAL "")
    set (TLS ${WITH_TLS})
endif (WITH_TLS STREQUAL "")

if (TLS STREQUAL "no")
    target_compile_definitions(papi PRIVATE NO_TLS)
else (TLS STREQUAL "no")
    if (TLS)
        if (TLS STREQUAL "yes")
            set (_TLS __thread)
        endif (TLS STREQUAL "yes")
        target_compile_definitions(papi PRIVATE USE_COMPILER_TLS)
        set (HAVE_THREAD_LOCAL_STORAGE ${TLS} CACHE STRING "Keyword for per-thread variables" FORCE)
    endif (TLS)
endif (TLS STREQUAL "no")
#configure.in:643
#message (STATUS "HAVE___THREAD_RUN=${HAVE___THREAD_RUN}, HAVE___THREAD_COMPILE=${HAVE___THREAD_COMPILE}")
#message (STATUS "WITH_TLS=${WITH_TLS}, TLS=${TLS}, HAVE_THREAD_LOCAL_STORAGE=${HAVE_THREAD_LOCAL_STORAGE}")


##############################################################
set (WITH_VIRTUALTIMER "auto-detect" CACHE STRING "Specify per-thread virtual timer. Possible values are: clock_thread_cputime_id, times, proc, getrusage, perfctr" )
set_property(CACHE WITH_VIRTUALTIMER PROPERTY STRINGS auto-detect clock_thread_cputime_id times proc getrusage perfctr)

#NOTE: configure.in:588
if (WITH_VIRTUALTIMER STREQUAL "auto-detect")
    if (CROSS_COMPILING)
        message (FATAL_ERROR "virtualtimer must be specified for cross compile")
    endif (CROSS_COMPILING)

    try_run(HAVE_CLOCK_THREAD_CPUTIME_RUN HAVE_CLOCK_THREAD_CPUTIME_COMPILE
        ${CMAKE_CURRENT_BINARY_DIR}/platform
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_clock_thread_cputime_id.c
        CMAKE_FLAGS -DLINK_LIBRARIES:STRING=-pthread)

    if (HAVE_CLOCK_THREAD_CPUTIME_RUN STREQUAL "0")
        set (VIRTUALTIMER "clock_thread_cputime_id")
    else (HAVE_CLOCK_THREAD_CPUTIME_RUN STREQUAL "0")
        try_run (HAVE_TIMES_RUN HAVE_TIMES_COMPILE
            ${CMAKE_CURRENT_BINARY_DIR}/platform
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_times.c
            CMAKE_FLAGS -DLINK_LIBRARIES:STRING=-pthread)

        if (HAVE_TIMES_RUN STREQUAL "0")
            set (VIRTUALTIMER "times")
        else (HAVE_TIMES_RUN STREQUAL "0")
            set (VIRTUALTIMER "default")
        endif (HAVE_TIMES_RUN STREQUAL "0")

    endif (HAVE_CLOCK_THREAD_CPUTIME_RUN STREQUAL "0")
endif (WITH_VIRTUALTIMER STREQUAL "auto-detect")

if (DEFINED VIRTUALTIMER)
    if (VIRTUALTIMER STREQUAL "times")
        set (HAVE_PER_THREAD_TIMES 1)
    endif (VIRTUALTIMER STREQUAL "times")

    if (VIRTUALTIMER STREQUAL "getrusage")
        set (HAVE_PER_THREAD_GETRUSAGE 1)
    endif (VIRTUALTIMER STREQUAL "getrusage")

    if (VIRTUALTIMER STREQUAL "clock_thread_cputime_id")
        set (HAVE_CLOCK_GETTIME_THREAD CLOCK_THREAD_CPUTIME_ID)
    endif (VIRTUALTIMER STREQUAL "clock_thread_cputime_id")

    if (VIRTUALTIMER STREQUAL "proc" OR VIRTUALTIMER STREQUAL "default" OR VIRTUALTIMER STREQUAL "perfctr")
        set (USE_PROC_PTTIMER 1)
        set (USE_PERFCTR_PTTIMER 1)
    endif (VIRTUALTIMER STREQUAL "proc" OR VIRTUALTIMER STREQUAL "default" OR VIRTUALTIMER STREQUAL "perfctr")
endif (DEFINED VIRTUALTIMER)

#message (STATUS "VIRTUALTIMER=${VIRTUALTIMER}, HAVE_PER_THREAD_TIMES=${HAVE_PER_THREAD_TIMES}, HAVE_PER_THREAD_GETRUSAGE=${HAVE_PER_THREAD_GETRUSAGE}, HAVE_CLOCK_GETTIME_THREAD=${HAVE_CLOCK_GETTIME_THREAD}, USE_PROC_PTTIMER=${USE_PROC_PTTIMER}, USE_PERFCTR_PTTIMER=${USE_PERFCTR_PTTIMER}")

##############################################################
# configure.in:748
option (WITH_STATIC_USER_EVENTS "Build with a static user events file." OFF)
if (WITH_STATIC_USER_EVENTS)
    target_compile_definitions (papi PRIVATE STATIC_USER_EVENTS)
endif (WITH_STATIC_USER_EVENTS)

##############################################################
# configure.in:758
option (WITH_STATIC_PAPI_EVENTS "Build with a static papi events file." ON)
if (WITH_STATIC_PAPI_EVENTS)
    target_compile_definitions (papi PRIVATE STATIC_PAPI_EVENTS_TABLE)
endif (WITH_STATIC_PAPI_EVENTS)

##############################################################
# configure.in:768
option (BUILD_STATIC_LIB "Build a static library." ON)
if (BUILD_STATIC_LIB)
    add_library (papi_static STATIC $<TARGET_OBJECTS:papi>)
    set_target_properties (papi
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    set_target_properties (papi_static
        PROPERTIES
        OUTPUT_NAME papi)
    target_link_libraries(papi_static PUBLIC ${CMAKE_DL_LIBS} m)
endif (BUILD_STATIC_LIB)

##############################################################
# configure.in:776
option (BUILD_SHARED_LIB "Build a shared library" ON)
if (BUILD_SHARED_LIB)
    add_library (papi_shared SHARED $<TARGET_OBJECTS:papi>)
    set_target_properties (papi
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    set_target_properties (papi_shared
        PROPERTIES
        OUTPUT_NAME papi)
    target_link_libraries(papi_shared PUBLIC ${CMAKE_DL_LIBS} m)
endif (BUILD_SHARED_LIB)

##############################################################
# configure.in:782
if (NOT BUILD_STATIC_LIB AND NOT BUILD_SHARED_LIB)
    message (FATAL_ERROR "Both shared and static libs are disabled.")
endif (NOT BUILD_STATIC_LIB AND NOT BUILD_SHARED_LIB)

##############################################################
#configure.in:786
# nothing really

##############################################################
# configure.in:793
option (WITH_STATIC_TOOLS "Specify static compile of tests and utilities." OFF)
# this option should trigger set_target_properties(tool PROPERTIES LINK_SEARCH_START_STATIC 1) to add -static or equivalent
if (WITH_STATIC_TOOLS AND NOT BUILD_STATIC_LIB)
    message (FATAL_ERROR "Building tests and utilities static but no static papi library to be built.")
endif (WITH_STATIC_TOOLS AND NOT BUILD_STATIC_LIB)

##############################################################
# configure.in:804
option (WITH_SHLIB_TOOLS "Specify linking with papi library of tests and utilities." OFF)

if (WITH_SHLIB_TOOLS)
    if (NOT BUILD_SHARED_LIB)
        message (FATAL_ERROR "Building shared but specified static linking.")
    endif (NOT BUILD_SHARED_LIB)
    if (WITH_STATIC_TOOLS)
        message (FATAL_ERROR "Building shared but specified static linking.")
    endif (WITH_STATIC_TOOLS)
else (WITH_SHLIB_TOOLS)
    if (NOT BUILD_STATIC_LIB)
        message (FATAL_ERROR "Building shared but specified static linking for tests and utilities.")
    endif (NOT BUILD_STATIC_LIB)
endif (WITH_SHLIB_TOOLS)

##############################################################
# perfctr
# configure.in:830
set (perfctr "0")
set (force_perfctr no)

option (WITH_PERFCTR "Specify perfctr as the performance interface." OFF)
if (WITH_PERFCTR)
    set (USER_SPECIFIED_INTERFACE "perfctr")
    set (force_perfctr yes)
    if (ARCH STREQUAL "ppc64")
        set (perfctr "7")
    else (ARCH STREQUAL "ppc64")
        set (perfctr "6")
    endif (perfctr "7")
else (WITH_PERFCTR)
    set (perfctr "0")
endif (WITH_PERFCTR)

# configure.in:844
if (NOT CROSS_COMPILING)
    find_path (HAVE_PERFCTR_SYS_CLASS perfctr
        PATHS /sys/class)
    if (HAVE_PERFCTR_SYS_CLASS)
        set (perfctr "7")
    else (HAVE_PERFCTR_SYS_CLASS)
        find_path (HAVE_PERFCTR_DEV perfctr
            PATHS /dev)
        if (HAVE_PERFCTR_DEV)
            set (perfctr "6")
        endif (HAVE_PERFCTR_DEV)
    endif (HAVE_PERFCTR_SYS_CLASS)
endif (NOT CROSS_COMPILING)


################################################
# Components
# Discovering components subdirectories
set (force_perf_events no CACHE INTERNAL "sdfvfv")
set (perf_events no CACHE INTERNAL "sdfhb")
set (force_perfmon2 no CACHE INTERNAL "adsrhg")
set (perfmon2 "0" CACHE INTERNAL "aerhgrt")
set (force_perfctr no CACHE INTERNAL "zSDgsda")
set (perfctr "0" CACHE INTERNAL "adfrghesrd")
set (disable_uncore    yes CACHE INTERNAL "uytgf")

set (COMPONENTS_LIST)
set (SUBCOMPONENTS_LIST)

# Examples installation target, do before discovery
set (EXAMPLES_DIR          ${CMAKE_INSTALL_PREFIX}/share)
add_custom_target (install-examples)

add_subdirectory(components)

set (COMPONENT_RULES components/Rules.components)

if (TARGET papi_shared)
 message(STATUS "Adding properties to papi_shared ${COMPONENTS_LIST}")
  set_target_properties(papi_shared
    PROPERTIES
    "COMPONENTS"    "${COMPONENTS_LIST}" #without quotes, only the first element is copied
    "SUBCOMPONENTS" "${SUBCOMPONENTS_LIST}")
  set_target_properties(papi_shared
    PROPERTIES
    EXPORT_PROPERTIES "COMPONENTS;SUBCOMPONENTS")
endif (TARGET papi_shared)

if (TARGET papi_static)
  set_target_properties(papi_static
    PROPERTIES
    "COMPONENTS"    "${COMPONENTS_LIST}"
    "SUBCOMPONENTS" "${SUBCOMPONENTS_LIST}")
  set_target_properties(papi_static
    PROPERTIES
    EXPORT_PROPERTIES "COMPONENTS;SUBCOMPONENTS")
endif (TARGET papi_static)


# TODO : add custom commands for the custom target of the makefile
# > showconf, etc... (see $PAPI_ROOT/Makefile.inc for the list)


##############################################################
# configure.in: 1143
if (_OS STREQUAL "aix")
    set (makever "${_OS}-${_CPU}")
endif (_OS STREQUAL "aix")

if (_OS STREQUAL "bgp")
    set (makever "bgp")
endif (_OS STREQUAL "bgp")

if (_OS STREQUAL "bgq")
    set (makever "bgq")
endif (_OS STREQUAL "bgq")

if (_OS STREQUAL "CLE")
    if (perfmon2 STREQUAL "yes")
        set (makever "${_OS}-perfmon2")
    else (perfmon2 STREQUAL "yes")
        set (makever "${_OS}-pe")
    endif (perfmon2 STREQUAL "yes")
endif (_OS STREQUAL "CLE")

if (_OS STREQUAL "freebsd")
    set (makever "freebsd")
    target_link_libraries(papi PRIVATE pmc)
    # HWPMC driver is available for FreeBSD >= 6
    #FIXME: I went offroad here, see configure.in:1165 for reference, change dot by dash
    execute_process (COMMAND "uname -r | cut -d'-' -f1" OUTPUT_VARIABLE FREEBSD_VERSION)
    if (FREEBSD_VERSION VERSION_LESS 6.0.0)
        message (FATAL_ERROR "PAPI requires FreeBSD 6.0 or later")
    endif (FREEBSD_VERSION VERSION_LESS 6.0.0)

    # Determine if HWPMC module is on the kernel
    execute_process(COMMAND "dmesg | grep hwpmc 2> /dev/null > /dev/null"
        OUTPUT_VARIABLE hwpmc_logs)
    if (NOT hwpmc_logs)
        message (FATAL_ERROR "HWPMC module not found. (see INSTALL.TXT)")
    endif (NOT hwpmc_logs)

    # Determine the number of counters
    file (WRITE freebsd-config.h
        "/* Automatically generated file by configure */
        #ifndef _FREEBSD_CONFIG_H_
        #define _FREEBSD_CONFIG_H_")

    check_c_source_compiles("#include <unistd.h>
        #include <pmc.h>
        main() { int i = pmc_init(); }" pmc_pmc_init_linked)

    if (NOT pmc_pmc_init_linked)
        message(FATAL_ERROR "Failed to link hwpmc example")
    endif (NOT pmc_pmc_init_linked)

    try_run (HAVE_FREEBSD_NUM_COUNTERS_RUN HAVE_FREEBSD_NUM_COUNTERS_COMPILE
        ${CMAKE_CURRENT_BINARY_DIR}/platform
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake_files/check_freebsd_num_counters.c)

    if (NOT HAVE_FREEBSD_NUM_COUNTERS_RUN OR HAVE_FREEBSD_NUM_COUNTERS_RUN STREQUAL "0")
        message (FATAL_ERROR "pmc_npmc info returned 0. Determine if the HWPMC module is loaded (see hwpmc(4))")
    endif (NOT HAVE_FREEBSD_NUM_COUNTERS_RUN OR HAVE_FREEBSD_NUM_COUNTERS_RUN STREQUAL "0")

    file (APPEND freebsd-config.h
        "#define HWPMC_NUM_COUNTERS ${num_counters}
#endif")
endif (_OS STREQUAL "freebsd")

#message (STATUS "force_perf_events ${force_perf_events}")

if (_OS STREQUAL "linux")
    # configure.in:1202
    if (force_perf_events STREQUAL "yes")
        set (makever "${_OS}-pe")
    elseif (force_perfmon2 STREQUAL "yes")
        set (makever "${_OS}-perfmon2")
    elseif (force_perfctr STREQUAL "yes")
        set (makever "${_OS}-perfctr-x86")
        if (_CPU STREQUAL itanium2 OR _CPU STREQUAL montecito)
            if (BITMODE STREQUAL "32")
                message (FATAL_ERROR "The bitmode you specified is not supported")
            endif (BITMODE STREQUAL "32")
            set (makever "${_OS}-pfm-${_CPU}")
        elseif (_CPU STREQUAL "POWER5" OR _CPU STREQUAL "POWER5+" OR _CPU STREQUAL "POWER6" OR _CPU STREQUAL "POWER7" OR _CPU STREQUAL "PPC970")
            set (makever "${_OS}-perfctr-${_CPU}")
        endif ()
    elseif (perf_events STREQUAL "yes")
        set (makever "${_OS}-pe")
    elseif (perfmon2 STREQUAL "yes")
        set (makever "${_OS}-perfmon2")
    elseif (old_pfmv2 STREQUAL "y")
        set (makever "${_OS}-pfm-${_CPU}")
    elseif (perfctr)
        if (_CPU STREQUAL itanium2 OR _CPU STREQUAL montecito)
            if (BITMODE STREQUAL "32")
                message (FATAL_ERROR "The bitmode you specified is not supported")
            endif (BITMODE STREQUAL "32")
            set (makever "${_OS}-pfm-${_CPU}")
        elseif (_CPU STREQUAL x86)
            set (makever "${_OS}-perfctr-x86")
        elseif (_CPU STREQUAL "POWER5" OR _CPU STREQUAL "POWER5+" OR _CPU STREQUAL "POWER6" OR _CPU STREQUAL "POWER7" OR _CPU STREQUAL "PPC970")
            set (makever "${_OS}-perfctr-${_CPU}")
        else ()
            set (makever "${_OS}-generic")
        endif ()
    endif ()
endif (_OS STREQUAL "linux")

if (_OS STREQUAL "solaris")
    # configure.in:1244
    execute_process (COMMAND "isainfo -v | grep \"64\"" OUTPUT_VARIABLE isainfo)
    if (BITMODE STREQUAL "64" AND NOT isainfo)
        message (FATAL_ERROR "The bitmode you specified is not supported")
    endif (BITMODE STREQUAL "64" AND NOT isainfo)
    set (makever "${_OS}-${_CPU}")
endif (_OS STREQUAL "solaris")

if (_OS STREQUAL "darwin")
    set (makever "${_OS}")
endif (_OS STREQUAL "darwin")

if (NOT makever)
    message (WARNING "This platform is not supported so a generic build without CPU counters will be used")
    set (makever "generic_platform")
endif (NOT makever)

#message (STATUS "makever = ${makever}")

##############################################################
# configure.in: 1261
set (PAPISOVER "${PAPI_VERSION_MAJOR}.${PAPI_VERSION_MINOR}.${PAPI_VERSION_PATCH}" CACHE INTERNAL "" FORCE)

#ompflags = -fopenmp
#cc_r = $cc -pthread
#cc_shr = $cc  -fPIC -DPIC -shared -Wl,-soname -Wl,$(VLIB) -Xlinker "-rpath" -Xlinker "$(LIBDIR)"

if (CC_COMMON_NAME STREQUAL "gcc")
    if (BITMODE STREQUAL "32")
        target_compile_options (papi PRIVATE -m32)
    elseif (BITMODE STREQUAL "64")
        target_compile_options (papi PRIVATE -m64)
    endif (BITMODE STREQUAL "64")
endif (CC_COMMON_NAME STREQUAL "gcc")

target_compile_definitions (papi PRIVATE _REENTRANT _GNU_SOURCE linux)

if (CC_COMMON_NAME STREQUAL "gcc")
    target_compile_options (papi PRIVATE -Wall)
endif (CC_COMMON_NAME STREQUAL "gcc")

##############################################################
# configure.in:1288

if (_OS STREQUAL aix)
    set (OSFILESRC ${CMAKE_CURRENT_SOURCE_DIR}/aix-memory.c)
    set (OSLOCK    ${CMAKE_CURRENT_SOURCE_DIR}/aix-lock.h)
    set (OSCONTEXT ${CMAKE_CURRENT_SOURCE_DIR}/aix-context.h)

elseif (_OS STREQUAL bgp)
    set (OSFILESRC ${CMAKE_CURRENT_SOURCE_DIR}/linux-bgp-memory.c)
    set (OSLOCK    ${CMAKE_CURRENT_SOURCE_DIR}/linux-bgp-lock.h)
    set (OSCONTEXT ${CMAKE_CURRENT_SOURCE_DIR}/linux-bgp-context.h)

elseif (_OS STREQUAL bgq)
    set (OSFILESRC ${CMAKE_CURRENT_SOURCE_DIR}/linux-bgq-memory.c)
    set (OSLOCK    ${CMAKE_CURRENT_SOURCE_DIR}/linux-bgq-lock.h)
    set (OSCONTEXT ${CMAKE_CURRENT_SOURCE_DIR}/linux-context.h)

elseif (_OS STREQUAL freebsd)
    set (OSFILESRC ${CMAKE_CURRENT_SOURCE_DIR}/freebsd-memory.c)
    set (OSLOCK    ${CMAKE_CURRENT_SOURCE_DIR}/freebsd-lock.h)
    set (OSCONTEXT ${CMAKE_CURRENT_SOURCE_DIR}/freebsd-context.h)

elseif (_OS STREQUAL linux)
    set (OSFILESRC
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-memory.c
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-timer.c
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-common.c)
    set (OSFILEHDR
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-memory.h
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-timer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-common.h)
    set (OSLOCK    ${CMAKE_CURRENT_SOURCE_DIR}/linux-lock.h)
    set (OSCONTEXT ${CMAKE_CURRENT_SOURCE_DIR}/linux-context.h)

elseif (_OS STREQUAL solaris)
    set (OSFILESRC
        ${CMAKE_CURRENT_SOURCE_DIR}/solaris-memory.c
        ${CMAKE_CURRENT_SOURCE_DIR}/solaris-common.c)
    set (OSFILEHDR
        ${CMAKE_CURRENT_SOURCE_DIR}/solaris-memory.h
        ${CMAKE_CURRENT_SOURCE_DIR}/solaris-common.h)
    set (OSLOCK    ${CMAKE_CURRENT_SOURCE_DIR}/solaris-lock.h)
    set (OSCONTEXT ${CMAKE_CURRENT_SOURCE_DIR}/solaris-context.h)

elseif (_OS STREQUAL darwin)
    set (OSFILESRC
        ${CMAKE_CURRENT_SOURCE_DIR}/darwin-memory.c
        ${CMAKE_CURRENT_SOURCE_DIR}/darwin-common.c)
    set (OSFILEHDR
        ${CMAKE_CURRENT_SOURCE_DIR}/darwin-memory.h
        ${CMAKE_CURRENT_SOURCE_DIR}/darwin-common.h)
    set (OSLOCK    ${CMAKE_CURRENT_SOURCE_DIR}/darwin-lock.h)
    set (OSCONTEXT ${CMAKE_CURRENT_SOURCE_DIR}/darwin-context.h)

endif (_OS STREQUAL aix)

target_compile_definitions(papi PRIVATE OSLOCK=\"${OSLOCK}\" OSCONTEXT=\"${OSCONTEXT}\")
target_sources (papi PRIVATE ${OSFILESRC})

##############################################################
# configure.in:1324

if (makever STREQUAL aix-power5 OR makever STREQUAL aix-power6 OR makever STREQUAL aix-power7)
    if (BITMODE STREQUAL 64)
        set_target_properties (papi_static PROPERTIES OUTPUT_NAME papi64)
        set_target_properties (papi_shared PROPERTIES OUTPUT_NAME papi64)
        target_compile_options (papi PRIVATE -q64 -bmaxdata:0x07000000000000)
        set (arg64 "-X64" CACHE INTERNAL "" FORCE)
    else (BITMODE STREQUAL 64)
        set_target_properties (papi_shared PROPERTIES OUTPUT_NAME papi)
        target_compile_options (papi PRIVATE -bmaxdata:0x80000000/dsa)
    endif (BITMODE STREQUAL 64)

    set (CPUCOMPONENT_NAME aix CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_C ${CMAKE_CURRENT_SOURCE_DIR}/aix.c)
    set (VECTOR _aix_vector CACHE INTERNAL "" FORCE)
    set (PAPI_EVENTS_CSV "papi_events.csv" CACHE INTERNAL "" FORCE)
    # TODO: we make perfctr_ppc mandatory on power, here
    set (MISCHDRS
        ${CMAKE_CURRENT_SOURCE_DIR}/aix.h
        ${CMAKE_CURRENT_SOURCE_DIR}/components/perfctr_ppc/ppc64_events.h
        ${CMAKE_CURRENT_SOURCE_DIR}/papi_events_table.h)
    set (MISCSRCS ${CMAKE_CURRENT_SOURCE_DIR}/aix.c)
    target_compile_options (papi PRIVATE -qenum=4 -qmaxmem=-1 -qarch=${cpu_option} -qlanglvl=extended)
    target_include_directories (papi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/components/perfctr_ppc
        ${PMAPI}/include)
    target_compile_definitions (papi PRIVATE NO_VARARG_MACRO _AIX _${CPU_MODEL} NEED_FFSLL ARCH_EVTS="${ARCH_EVENTS}.h" COMP_VECTOR=_ppc64_vectors STATIC_PAPI_EVENTS_TABLE)
    if (CMAKE_BUILD_TYPE MATCHES DEBUG)
        set (OPTFLAGS "-O3 -qstrict ${PMINIT}")
    else (CMAKE_BUILD_TYPE MATCHES DEBUG)
        set (OPTFLAGS "${PMINIT}")
    endif (CMAKE_BUILD_TYPE MATCHES DEBUG)
    set (SMPCFLGS "-qsmp" CACHE INTERNAL "" FORCE)
    set (OMPCFLGS "-qsmp=omp" CACHE INTERNAL "" FORCE)
    # configure.in:1362 FIXME: that out-of-nowhere detection of mpicc or mpcc
    set (CC_SHR_FLGS "-G -bnoentry" CACHE INTERNAL "Those werd flags if xlc" FORCE)
    set (CPP_E "-E")
    if (makever STREQUAL aix-power5)
        set (ARCH_EVENTS power5_events CACHE INTERNAL "" FORCE)
        set (CPU_MODEL POWER5 CACHE INTERNAL "" FORCE)
        set (cpu_option pwr5 CACHE INTERNAL "" FORCE)
        set (DESC "AIX 5.1.0 or greater with POWER5" CACHE INTERNAL "" FORCE)
    endif (makever STREQUAL aix-power5)

    if (makever STREQUAL aix-power6)
        set (ARCH_EVENTS power6_events CACHE INTERNAL "" FORCE)
        set (CPU_MODEL POWER6 CACHE INTERNAL "" FORCE)
        set (cpu_option pwr6 CACHE INTERNAL "" FORCE)
        set (DESC "AIX 5.1.0 or greater with POWER6" CACHE INTERNAL "" FORCE)
        set (CPPFLAGS "-qlanglvl=extended")
    endif (makever STREQUAL aix-power6)

    if (makever STREQUAL aix-power7)
        set (ARCH_EVENTS power7_events CACHE INTERNAL "" FORCE)
        set (CPU_MODEL POWER7 CACHE INTERNAL "" FORCE)
        set (cpu_option pwr7 CACHE INTERNAL "" FORCE)
        set (DESC "AIX 5.1.0 or greater with POWER7" CACHE INTERNAL "" FORCE)
        set (CPPFLAGS "-qlanglvl=extended")
    endif (makever STREQUAL aix-power7)

    if (BITMIDE STREQUAL 64)
        set (DESC "${DESC} 64 bit build" CACHE INTERNAL "" FORCE)
    endif (BITMIDE STREQUAL 64)

elseif (makever STREQUAL bgp)
    set (CPP_E "-E")
    set (CPUCOMPONENT_NAME linux-bgp CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_C ${CMAKE_CURRENT_SOURCE_DIR}/linux-bgp.c)
    set (VECTOR _bgp_vectors CACHE INTERNAL "" FORCE)
    set (PAPI_EVENTS_CSV "papi_events.csv" CACHE INTERNAL "" FORCE)
    set (CMAKE_C_FLAGS "-g -gdwarf-2 -O2 -Wall -DCOMP_VECTOR=_bgp_vectors")
    target_include_directories (papi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} /bgsys/drivers/ppcfloor/arch/include)
    set (BGP_SYSDIR "/bgsys/drivers/ppcfloor" CACHE INTERNAL "" FORCE)
    set (SHOW_CONF show_bgp_conf CACHE INTERNAL "" FORCE)
    set (BGP_GNU_LINUX_PATH ${BGP_SYSDIR}/gnu-linux CACHE INTERNAL "" FORCE)
    find_library (SPI_LIB SPI.cna
        PATHS ${BGP_SYSDIR} PATH_SUFFIXES runtime/SPI)
    if (SPI_LIB)
        target_link_library (papi PRIVATE ${SPI_LIB})
    endif (SPI_LIB)
    set (CMAKE_Fortran_FLAGS "-g -gdwarf-2 -O2 -Wall")
    target_compile_definitions (papi PRIVATE linux)
    set (OPTFLAGS "-g -Wall -O3")
    set (TOPTFLAGS "-g -Wall -O0")
    set (DESC "Linux for BlueGene/P" CACHE INTERNAL "" FORCE)
    #FIXME, what do I do with that: LIBS=static
    set (CC_SHR_FLGS " -shared -Xlinker \"-soname\" -Xlinker \"${SHLIB}\" -Xlinker \"-rpath\" -Xlinker" CACHE INTERNAL "" FORCE)
    set (OMPCFLGS "" CACHE INTERNAL "" FORCE)

elseif (makever STREQUAL bgq)
    set (CPUCOMPONENT_NAME linux-bgq CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_C ${CMAKE_CURRENT_SOURCE_DIR}/linux-bgq.c)
    set (VECTOR _bgq_vectors CACHE INTERNAL "" FORCE)
    set (FILENAME Rules.bgpm CACHE INTERNAL "" FORCE)
    set (PAPI_EVENTS_CSV "papi_events.csv" CACHE INTERNAL "" FORCE)
    set (MISCSRCS ${CMAKE_CURRENT_SOURCE_DIR}/linux-bgq-common.c)
    set (OPTFLAGS "-g -Wall -O3")
    set (TOPTFLAGS "-g -Wall -O0")
    set (DESC "Linux for BlueGene/Q" CACHE INTERNAL "" FORCE)
    set (CC_SHR_FLGS " -fPIC -DPIC -shared -Wl,-soname -Wl,${SHLIB} -Xlinker \"-rpath\" -Xlinker \"${LIBDIR}\"")
    set (OMPCFLGS "" CACHE INTERNAL "" FORCE)

elseif (makever STREQUAL CLE-perfmon2)
    set (FILENAME Rules.perfmon2 CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_NAME perfmon CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_C ${CMAKE_CURRENT_SOURCE_DIR}/perfmon.c)
    set (VECTOR _papi_pfm_vector CACHE INTERNAL "" FORCE)
    target_compile_definitions (papi PRIVATE __crayxt)
    set (CMAKE_Fortran_FLAGS "")

elseif (makever STREQUAL freebsd)
    set (CPP_E "-E")
    set (CPUCOMPONENT_NAME freebsd CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_C ${CMAKE_CURRENT_SOURCE_DIR}/freebsd.c)
    set (VECTOR _papi_freebsd_vectors CACHE INTERNAL "" FORCE)
    set (PAPI_EVENTS_CSV "freebsd_events.csv" CACHE INTERNAL "" FORCE)
    set (MISCHDRS
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-unknown.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6-m.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6-3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6-2.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6-c.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-k7.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-k8.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p4.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-atom.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-core.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-core2.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-core2-extreme.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-i7.h
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-westmere.h)
    set (MISCSRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-unknown.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6-m.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6-3.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6-2.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p6-c.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-k7.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-k8.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-p4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-atom.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-core.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-core2.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-core2-extreme.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-i7.c
        ${CMAKE_CURRENT_SOURCE_DIR}/freebsd/map-westmere.c)
    set (DESC "FreeBSD -over libpmc- " CACHE INTERNAL "" FORCE)
    set_target_properties (papi PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions (papi PRIVATE PIC)
    target_include_directories (papi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/freebsd)
    set (CC_SHR_FLGS " -shared -Xlinker \"-soname\" -Xlinker \"-rpath\" -Xlinker \"${LIBDIR}\" ")

elseif (makever STREQUAL linux-generic)
    # configure.in:1455
    set (CPP_E "-E")
    set (CPUCOMPONENT_NAME linux-generic CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_C ${CMAKE_CURRENT_SOURCE_DIR}/linux-generic.c)
    set (PAPI_EVENTS_CSV "papi_events.csv" CACHE INTERNAL "" FORCE)
    set (VECTOR _papi_dummy_vector CACHE INTERNAL "" FORCE)

elseif (makever STREQUAL linux-pe)
    # configure.in:1462
    set (CPP_E "-E")
    set (FILENAME Rules.pfm4_pe CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_NAME perf_event CACHE INTERNAL "" FORCE)
    list (APPEND components perf_event)
    # FIXME: find how to force a component
    set (WITH_PERF_EVENTS ON CACHE INTERNAL "" FORCE)
    if (disable_uncore STREQUAL no)
        list (APPEND components perf_event_uncore)
        set (WITH_PERF_EVENTS_UNCORE ON CACHE INTERNAL "" FORCE)
   endif (disable_uncore STREQUAL no)

elseif (makever STREQUAL linux-perfctr-x86)
    # configure.in:1470
    set (CPP_E "-E")
    set (FILENAME Rules.perfctr-pfm CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_NAME perfctr-x86 CACHE INTERNAL "" FORCE)
    set (version 2.6.x CACHE INTERNAL "" FORCE)
    list (APPEND components perfctr)
    set (WITH_perfctr ON CACHE INTERNAL "" FORCE)

elseif (makever STREQUAL linux-perfctr-POWER5 OR makever STREQUAL linux-perfctr-POWER5+ OR makever STREQUAL linux-perfctr-POWER6 OR makever STREQUAL linux-perfctr-POWER7 OR makever STREQUAL linux-perfctr-PPC970)
    # configure.in:1476
    set (FILENAME Rules.perfctr CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_NAME perfctr-ppc CAHCE INTERNAL "" FORCE)
    set (version 2.7.x CACHE INTERNAL "" FORCE)
    list (APPEND components perfctr_ppc)
    set (WITH_perfctr_ppc ON CACHE INTERNAL "" FORCE)
    set (_CPU ppc64 CACHE INTERNAL "" FORCE)
    if (makever STREQUAL linux-perfctr-POWER5)
        set (CPU_MODEL POWER5 CACHE INTERNAL "" FORCE)
        set (ARCH_EVENTS power5_events CACHE INTERNAL "" FORCE)
        set (ARCH_SPEC_EVTS ${CMAKE_CURRENT_SOURCE_DIR}/power5_events_map.c)
    elseif (makever STREQUAL linux-perfctr-POWER5+)
        set (CPU_MODEL POWER5p CACHE INTERNAL "" FORCE)
        set (ARCH_EVENTS power5+_events CACHE INTERNAL "" FORCE)
        set (ARCH_SPEC_EVTS ${CMAKE_CURRENT_SOURCE_DIR}/power5+_events_map.c)
    elseif (makever STREQUAL linux-perfctr-POWER6)
        set (CPU_MODEL POWER6 CACHE INTERNAL "" FORCE)
        set (ARCH_EVENTS power6_events CACHE INTERNAL "" FORCE)
        set (ARCH_SPEC_EVTS ${CMAKE_CURRENT_SOURCE_DIR}/power6_events_map.c)
    elseif (makever STREQUAL linux-perfctr-POWER7)
        set (CPU_MODEL POWER7 CACHE INTERNAL "" FORCE)
        set (ARCH_EVENTS power7_events CACHE INTERNAL "" FORCE)
        set (ARCH_SPEC_EVTS ${CMAKE_CURRENT_SOURCE_DIR}/power7_events_map.c)
    elseif (makever STREQUAL linux-perfctr-PPC970)
        set (CPU_MODEL PPC970 CACHE INTERNAL "" FORCE)
        set (ARCH_EVENTS ppc970_events CACHE INTERNAL "" FORCE)
        set (ARCH_SPEC_EVTS ${CMAKE_CURRENT_SOURCE_DIR}/ppc970_events_map.c)
    endif ()
    target_compile_definitions(papi PRIVATE PPC64
        _${CPU_MODEL} __perfctr__ ARCH_EVTS=${ARCH_EVENTS}.h COMP_VECTOR=_ppc64_vectors)
    #set (CFLAGS "${CFLAGS} -DPPC64 -D_${CPU_MODEL} -D__perfctr__ -DARCH_EVTS=\"$(ARCH_EVENTS).h\" -DCOMP_VECTOR=_ppc64_vectors"
    #FFLAGS='-D_$(CPU_MODEL)'

elseif (makever STREQUAL linux-perfmon2)
    # configure.in:1506
    set (FILENAME Rules.perfmon2 CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_NAME perfmon2 CACHE INTERNAL "" FORCE)
    list (APPEND components perfmon2)
    set (WITH_perfmon2 ON CACHE INTERNAL "" FORCE)

elseif (makever STREQUAL linux-pfm-ia64 OR makever STREQUAL linux-pfm-itanium2 OR makever STREQUAL linux-pfm-montecito)
    # configure.in:1511
    set (FILENAME Rules.pfm CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_NAME perfmon-ia64 CACHE INTERNAL "" FORCE)
    list (APPEND components perfmon_ia64)
    set (version 3.y CACHE INTERNAL "" FORCE)
    if (makever STREQUAL linux-pfm-itanium2)
        set (_CPU 2 CACHE INTERNAL "" FORCE)
    else ()
        set (_CPU 3 CACHE INTERNAL "" FORCE)
    endif ()
    target_compile_definitions(papi PRIVATE ITANIUM${_CPU})
    set (CC_SHR "$(CC) -fPIC -DPIC -shared -Wl,-soname -Wl,$(SHLIB) -Xlinker \"-rpath\" -Xlinker \"$(LIBDIR)\"")

elseif (makever STREQUAL solaris-ultra)
    # configure.in:1526

    # droping solaris platform until we need them
    # configure.in:1528-1589

elseif (makever STREQUAL darwin)
    # we do not really support mac anyway...
    set (DESC "Darwin" CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_NAME darwin CACHE INTERNAL "" FORCE)
    set (CPUCOMPONENT_C ${CMAKE_CURRENT_SOURCE_DIR}/linux-generic.c)
    # configure.in:1596 overwriting of source file seems very wrong
    set (CC_SHR "$(CC) -fPIC -DPIC -shared -Wl,-dylib -Xlinker \"-rpath\" -Xlinker \"$(LIBDIR)\"")
    target_compile_definitions (papi PRIVATE NEED_FFSLL)
    # .dylib is handled by cmake, allegedly

elseif (makever STREQUAL generic_platform)
    set (DESC "Generic platform" CACHE INTERNAL "" FORCE)
endif()

if (CMAKE_Fortran_COMPILER_ID STREQUAL PGI)
    set (CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} -Minform=inform -Mextend)
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
    set (CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} -warn all)
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL XL)
    set (CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} -ffixed-line-length-132)
endif()

if (CC_COMMON_NAME STREQUAL icc)
    set (OMPCFLAGS -qopenmp)
endif (CC_COMMON_NAME STREQUAL icc)


#message (STATUS "VECTOR = >${VECTOR}<")

# configure.in:1620
# BEWARE: That first call destroys the file, don't put anything before
file (WRITE ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "/* Automatically generated by configure */\n")
file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "#ifndef COMPONENTS_CONFIG_H\n")
file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "#define COMPONENTS_CONFIG_H\n")
file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "\n")

if (VECTOR)
    if (VECTOR STREQUAL "_papi_dummy_vector")
        list (LENGTH components nb_components)
        if (nb_components EQUAL 0)
            file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "papi_vector_t ${VECTOR} = {\n")
            file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "
            .size = {
            .context = sizeof ( int ),
            .control_state = sizeof ( int ),
            .reg_value = sizeof ( int ),
            .reg_alloc = sizeof ( int ),
        },
        .cmp_info = {
            .num_native_events = 0,
            .num_preset_events = 0,
            .num_cntrs = 0,
            .name = \"Your system is unsupported! \",
            .short_name = \"UNSUPPORTED!\"
        },
        .dispatch_timer = NULL,
        .get_overflow_address = NULL,
        .start = NULL,
        .stop = NULL,
        .read = NULL,
        .reset = NULL,
        .write = NULL,
        .cleanup_eventset = NULL,
        .stop_profiling = NULL,
        .init_component = NULL,
        .init_thread = NULL,
        .init_control_state = NULL,
        .update_control_state = NULL,
        .ctl = NULL,
        .set_overflow = NULL,
        .set_profile	 = NULL,
        .set_domain = NULL,
        .ntv_enum_events = NULL,
        .ntv_name_to_code = NULL,
        .ntv_code_to_name = NULL,
        .ntv_code_to_descr = NULL,
        .ntv_code_to_bits = NULL,
        .ntv_code_to_info = NULL,
        .allocate_registers = NULL,
        .shutdown_thread = NULL,
        .shutdown_component = NULL,
        .user = NULL,
    };\n")

        else (nb_components EQUAL 0)
            set (VECTOR "")
        endif (nb_components EQUAL 0)
    elseif (NOT VECTOR STREQUAL "")
        file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "extern papi_vector_t ${VECTOR};\n")
    endif ()
endif (VECTOR)

foreach (sub ${SUBCOMPONENTS_LIST})
    #configure.in:1685
    #message(STATUS "component: ${component}")
    file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "extern papi_vector_t _${sub}_vector;\n")
endforeach (sub ${SUBCOMPONENTS_LIST})

#configure.in 1727

file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "struct papi_vectors *_papi_hwd[] = {\n")
if (VECTOR AND NOT VECTOR STREQUAL "")
    file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "   &${VECTOR},\n")
endif (VECTOR AND NOT VECTOR STREQUAL "")

foreach (sub ${SUBCOMPONENTS_LIST})
    file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "    &_${sub}_vector,\n")
endforeach (sub ${SUBCOMPONENTS_LIST})

file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "    NULL\n")
file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "};\n")
file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/components_config.h "#endif\n")

foreach (comp ${COMPONENTS_LIST})
    if (comp STREQUAL "sde")
        #TODO: configure.in:1757, set flags for SDE
    endif (comp STREQUAL "sde")
endforeach (comp ${COMPONENTS_LIST})

configure_file(${CMAKE_SOURCE_DIR}/config.h.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/config.h)

if (TARGET papi)
    target_include_directories(papi
        PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endif (TARGET papi)

if (TARGET papi_static)
    target_include_directories(papi_static
        PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endif (TARGET papi_static)

if (TARGET papi_shared)
    target_include_directories(papi_shared
        PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endif (TARGET papi_shared)


macro (add_object obj_source cflags definitions)
    add_library (${obj_source}_tgt OBJECT ${obj_source})
    target_compile_definitions (${obj_source}_tgt PRIVATE definitions)
    target_link_libraries (papi PRIVATE ${obj_source}_tgt)
endmacro (add_object obj_source cflag definitions)

add_custom_command (
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/papi_fwrappers_.c
    ${CMAKE_CURRENT_BINARY_DIR}/papi_fwrappers__.c
    ${CMAKE_CURRENT_BINARY_DIR}/upper_PAPI_FWRAPPERS.c
    COMMAND ${CMAKE_C_COMPILER} ${CPP_E} -DFORTRANUNDERSCORE       papi_fwrappers.c > ${CMAKE_CURRENT_BINARY_DIR}/papi_fwrappers_.c
    COMMAND ${CMAKE_C_COMPILER} ${CPP_E} -DFORTRANDOUBLEUNDERSCORE papi_fwrappers.c > ${CMAKE_CURRENT_BINARY_DIR}/papi_fwrappers__.c
    COMMAND ${CMAKE_C_COMPILER} ${CPP_E} -DFORTRANCAPS             papi_fwrappers.c > ${CMAKE_CURRENT_BINARY_DIR}/upper_PAPI_FWRAPPERS.c

    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

set (PAPI_GENERATED_WRAPPERS
    ${CMAKE_CURRENT_BINARY_DIR}/papi_fwrappers_.c
    ${CMAKE_CURRENT_BINARY_DIR}/papi_fwrappers__.c
    ${CMAKE_CURRENT_BINARY_DIR}/upper_PAPI_FWRAPPERS.c)

set (PAPI_SRCS
    ${MISCSRCS}
    ${CPUCOMPONENT_C}
    ${CMAKE_CURRENT_SOURCE_DIR}/papi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/papi_internal.c
    ${CMAKE_CURRENT_SOURCE_DIR}/high-level/papi_hl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/extras.c
    ${CMAKE_CURRENT_SOURCE_DIR}/sw_multiplex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/papi_fwrappers.c
    ${CMAKE_CURRENT_SOURCE_DIR}/threads.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cpus.c
    ${CMAKE_CURRENT_SOURCE_DIR}/papi_preset.c
    ${CMAKE_CURRENT_SOURCE_DIR}/papi_vector.c
    ${CMAKE_CURRENT_SOURCE_DIR}/papi_memory.c)

target_sources (papi PRIVATE
    ${PAPI_GENERATED_WRAPPERS}
    ${PAPI_SRCS})

add_executable (genpapifdef_exe ${CMAKE_CURRENT_SOURCE_DIR}/genpapifdef.c)
target_include_directories (genpapifdef_exe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command (
	OUTPUT
	${CMAKE_CURRENT_BINARY_DIR}/papi_events_table.h
	COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/papi_events_table.sh  ${CMAKE_CURRENT_SOURCE_DIR}/papi_events.csv > ${CMAKE_CURRENT_BINARY_DIR}/papi_events_table.h
	)

add_custom_target (papi_events_table DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/papi_events_table.h)

add_dependencies (papi papi_events_table)

add_custom_command (
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/fpapi.h
    ${CMAKE_CURRENT_BINARY_DIR}/f77papi.h
    ${CMAKE_CURRENT_BINARY_DIR}/f90papi.h
    COMMAND genpapifdef_exe -c   > ${CMAKE_CURRENT_BINARY_DIR}/fpapi.h
    COMMAND genpapifdef_exe -f77 > ${CMAKE_CURRENT_BINARY_DIR}/f77papi.h
    COMMAND genpapifdef_exe -f90 > ${CMAKE_CURRENT_BINARY_DIR}/f90papi.h)

add_custom_target (papi_fortran_headers
       DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fpapi.h)

add_dependencies (papi_fortran_headers genpapifdef_exe)
add_dependencies (papi papi_fortran_headers)

set (PAPI_FRAMEWORK_SOLARIS_SRCS
    solaris-memory.c
    solaris-ultra.c
    solaris-common.c
    solaris-niagara2.c)

set (PAPI_FRAMEWORK_POWERPC_SRCS
    aix.c
    aix-memory.c)

set (PAPI_FRAMEWORK_MACOSX_SRCS
    darwin-memory.c
    darwin-common.c)

set (PAPI_FRAMEWORK_FREEBSD_SRCS
    freebsd.c
    freebsd-memory.c)

target_sources(papi PRIVATE ${PAPI_FRAMEWORK_SRCS})

if (${freebsd}) # how do we determine that?
    add_subdirectory(freebsd)
endif (${freebsd})


# high-level interface specific sources
set(high_level_extra_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/high-level/papi_hl.c)

target_sources(papi PRIVATE ${high_level_extra_srcs})

set (PAPI_TESTLIB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testlib)
set (PAPI_UTILS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set (PAPI_VALIDATION_TESTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/validation_tests)
set (PAPI_CAT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/counter_analysis_toolkit)
set (PAPI_EXAMPLES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples)


#include testlib if testing enabled
#include ctests if testing enabled, see ctests/CMakeLists.txt
add_subdirectory (testlib)
#add_subdirectory (examples)
#add_subdirectory (ctests)
#add_subdirectory (ftests)
add_subdirectory (utils)
add_subdirectory (validation_tests)
#add_subdirectory (counter_analysis_toolkit)


# Installation
# root is CMAKE_INSTALL_PREFIX
set (PAPI_INSTALL_LIB ${CMAKE_INSTALL_PREFIX}/lib)
set (PAPI_INSTALL_BIN ${CMAKE_INSTALL_PREFIX}/bin)
set (PAPI_INSTALL_SHARE ${CMAKE_INSTALL_PREFIX}/share)
set (PAPI_INSTALL_INC ${CMAKE_INSTALL_PREFIX}/include)


set (PAPI_INSTALL_PKGCONFIG ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)
#configure_file ()

if (TARGET papi_static)
    install (TARGETS papi_static
        DESTINATION lib
        EXPORT papi-targets)
endif (TARGET papi_static)

if (TARGET papi_shared)
  install (TARGETS papi_shared
        DESTINATION lib
        EXPORT papi-targets)
endif (TARGET papi_shared)

# Targets Export
install (EXPORT papi-targets
    FILE PAPITargets.cmake
    NAMESPACE PAPI::
    DESTINATION ${PAPI_INSTALL_SHARE})

include (CMakePackageConfigHelpers)

set (INCLUDE_INSTALL_DIR   include)
set (LIB_INSTALL_DIR       lib)
set (RUNTIME_INSTALL_DIR   bin)
set (SYSCONFIG_INSTALL_DIR share)
set (CONFIG_INSTALL_DIR    lib/cmake)

configure_package_config_file (cmake_files/PAPIConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_files/PAPIConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake
    PATH_VARS LIB_INSTALL_DIR SYSCONFIG_INSTALL_DIR
    INCLUDE_INSTALL_DIR
    RUNTIME_INSTALL_DIR CONFIG_INSTALL_DIR)

write_basic_package_version_file (${CMAKE_CURRENT_BINARY_DIR}/PAPIConfigVersion.cmake
    VERSION ${PAPI_VERSION_MAJOR}.${PAPI_VERSION_MINOR}.${PAPI_VERSION_PATCH}
    COMPATIBILITY SameMajorVersion)

configure_file(cmake_files/PAPIComponents.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_files/PAPIComponents.cmake)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_files/PAPIConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_files/PAPIComponents.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/PAPIConfigVersion.cmake
    DESTINATION ${CONFIG_INSTALL_DIR})

# Pkg-Config file configuration + installation
configure_file(cmake_files/papi.pc.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/papi.pc)

install (FILES ${CMAKE_CURRENT_BINARY_DIR}/papi.pc
    DESTINATION lib/pkgconfig)

# Generated headers + Lib headers
set (PAPI_HDRS
    ${CMAKE_CURRENT_BINARY_DIR}/f77papi.h
    ${CMAKE_CURRENT_BINARY_DIR}/f90papi.h
    ${CMAKE_CURRENT_BINARY_DIR}/fpapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/papi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/papiStdEventDefs.h)

install (FILES ${PAPI_HDRS}
    DESTINATION include)

# Utility binaries
# > each one of them has its own rule in utils/CMakeLists.txt

# Libs


# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "TBZ2")
#set (CPACK_RESOURCE_FILE_LICENSE
#    "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${PAPI_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${PAPI_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${PAPI_VERSION_PATCH}")
include (CPack)


