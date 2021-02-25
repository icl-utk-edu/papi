if (NOT WITH_TESTS)
      #    return ()
endif (NOT WITH_TESTS)

add_papi_component_test (powercap_ppc_basic)
add_papi_component_test (powercap_ppc_limit)
