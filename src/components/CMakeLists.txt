
macro (add_papi_component_test test_name)
    add_executable(${test_name}
        ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.c)
    target_link_libraries (${test_name}
        PRIVATE papi)
    add_test(NAME test_${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endmacro (add_papi_component_test test_name)


set (component_dir ${CMAKE_CURRENT_SOURCE_DIR})
file (GLOB components RELATIVE ${component_dir} ${component_dir}/*)

message (STATUS "framework, components discovery:")

set (COMPONENTS_LIST_LCL)
set (COMPONENTS_RULES_LIST_LCL)

foreach (component ${components})
    if (EXISTS "${component_dir}/${component}/CMakeLists.txt")
        # let the component fill out the blanks
        add_subdirectory(${component})
    endif (EXISTS "${component_dir}/${component}/CMakeLists.txt")
endforeach (component ${components})

set (COMPONENT_LIST_FORMATTED "")

foreach (component_path ${COMPONENT_PATHS_LIST})
    set (COMPONENT_LIST_FORMATTED "${COMPONENT_LIST_FORMATTED} \"${component_path}\", ")
endforeach (component_path ${COMPONENT_PATHS_LIST})
set (COMPONENT_LIST_FORMATTED "${COMPONENT_LIST_FORMATTED} NULL" PARENT_SCOPE)

set (COMPONENTS_LIST ${COMPONENTS_LIST_LCL} PARENT_SCOPE)
set (SUBCOMPONENTS_LIST ${SUBCOMPONENTS_LIST_LCL} PARENT_SCOPE)

foreach (c ${COMPONENTS_LIST_LCL})
    #message (STATUS "propagating ${c} upper")
endforeach ()

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/static_component.h.in
#    ${CMAKE_CURRENT_BINARY_DIR}/static_component.h)
