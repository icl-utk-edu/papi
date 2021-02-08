if (NOT WITH_TESTS)
    return ()
endif (NOT WITH_TESTS)

add_papi_component_test (infiniband_list_events)
add_papi_component_test (infiniband_values_by_code)
