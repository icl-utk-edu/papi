
if (NOT TARGET papi)
    message(FATAL "Target papi is missing, components are compiled through the framework")
endif (NOT TARGET papi)

option(WITH_LMSENSORS "Build lm-sensors components" ON)

if (WITH_LMSENSORS)
    message(STATUS "  - component ${component}")

    set (PAPI_LMSENSORS_MAIN "" CACHE FILEPATH "Absolute path to libsensors.so")
    if (NOT PAPI_LMSENSORS_MAIN STREQUAL "")
        set (LMSENSORS_LIB ${PAPI_LMSENSORS_MAIN})
      endif (NOT PAPI_LMSENSORS_MAIN STREQUAL "")

    set(LMSENSORS_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/linux-lmsensors.c)

    if (NOT LMSENSORS_LIB)
        find_library(LMSENSORS_LIB sensors
            HINTS
            $ENV{PAPI_LMSENSORS_ROOT}
            ${PAPI_LMSENSORS_ROOT}
            /usr
            PATH_SUFFIXES
            lib
            lib64)
    endif (NOT LMSENSORS_LIB)

    if (NOT LMSENSORS_LIB)
        message (STATUS "Missing lm-sensors library, provide PAPI_LMSENSORS_ROOT")
        set (WITH_LMSENSORS OFF)
        return ()
    endif (NOT LMSENSORS_LIB)

    find_path (SENSORS_H sensors.h
        PATH_SUFFIXES sensors
        HINTS
        $ENV{PAPI_LMSENSORS_ROOT}/include
        ${PAPI_LMSENSORS_ROOT}/include
        /usr)

    if (NOT SENSORS_H)
        message (STATUS "Missing lm-sensors header, provide PAPI_LMSENSORS_ROOT")
        set (WITH_LMSENSORS OFF)
        return ()
    endif (NOT SENSORS_H)

    if (SENSORS_H AND LMSENSORS_LIB)
        message (STATUS "include ${SENSORS_H}")
        message (STATUS "lib ${LMSENSORS_LIB}")

        # list are duplicated when entering a new scope
        list (APPEND SUBCOMPONENTS_LIST_LCL "lmsensors")
        set (SUBCOMPONENTS_LIST_LCL ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)
        list (APPEND COMPONENTS_LIST_LCL lmsensors)
        set (COMPONENTS_LIST_LCL ${COMPONENTS_LIST_LCL} PARENT_SCOPE)
	#message(STATUS "pushing component lmsensors into list")

        target_sources (papi PRIVATE ${LMSENSORS_SRCS})
        target_include_directories (papi
            PRIVATE
            ${SENSORS_H}
            ${SENSORS_H}/sensors)
        target_compile_definitions (papi
            PRIVATE
            PAPI_LMSENSORS_MAIN=\"${LMSENSORS_LIB}\")

    endif (SENSORS_H AND LMSENSORS_LIB)
endif (WITH_LMSENSORS)
