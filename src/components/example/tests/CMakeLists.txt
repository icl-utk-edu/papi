if (NOT WITH_TESTS)
      #    return ()
endif (NOT WITH_TESTS)

add_papi_component_test (example_basic)
add_papi_component_test (example_multiple_components)
