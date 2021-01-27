set(freebsd_extra_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/map-atom.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-core2-extreme.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-i7.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-k7.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-k8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-p4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-p6-2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-p6-3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-p6.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-p6-m.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-unknown.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-westmere.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-core2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-core.c
    ${CMAKE_CURRENT_SOURCE_DIR}/map-p6-c.c)

target_sources(papi PRIVATE ${freebsd_extra_srcs})
target_include_directories(papi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
