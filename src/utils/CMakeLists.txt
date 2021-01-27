

set (util_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/cost_utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/print_header.c
    ${CMAKE_CURRENT_SOURCE_DIR}/papi_sde_interface.c)

add_library (util_objects OBJECT ${util_srcs})

target_include_directories (util_objects PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/..)

macro (add_util util_name)
    add_executable(${util_name} ${util_name}.c)
    #includes should be covered by both targets
	target_sources (${util_name}
		PRIVATE ${util_srcs})

	target_link_libraries (${util_name}
        PRIVATE
        # we pull the target validation_objects that contains clockcore and doloops
        $<TARGET_OBJECTS:testlib>
        papi_static)
    target_include_directories (${util_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${PAPI_TESTLIB_SOURCE_DIR})

install (TARGETS ${util_name}
	    #CONFIGURATIONS Debug/Release
	    RUNTIME DESTINATION bin)
endmacro (add_util util_name)

add_util(papi_avail)
add_util(papi_clockres)
add_util(papi_command_line)
add_util(papi_component_avail)
add_util(papi_cost)
add_util(papi_decode)
add_util(papi_error_codes)
add_util(papi_event_chooser)
#add_util(papi_hybrid_native_avail)
add_util(papi_mem_info)
add_util(papi_multiplex_cost)
add_util(papi_native_avail)
add_util(papi_version)
add_util(papi_xml_event_info)
