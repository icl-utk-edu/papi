if (NOT WITH_TESTS)
      #    return ()
endif (NOT WITH_TESTS)

#Needs to add cpp test macro
#add_papi_component_test (rocm_all)
add_papi_component_test (rocm_command_line)
#add_papi_component_test (rocm_failure_demo)
#add_papi_component_test (rocm_standalone)
