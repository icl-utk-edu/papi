
macro (add_example example_name)
    add_executable(${example_name} ${example_name}.c)
    target_link_libraries(${example_name}
        testlib
        papi)
    #papi has pthread as a dependency
endmacro (add_example example_name)

add_example(high_level)
add_example(multiplex)
add_example(PAPI_add_remove_event)
add_example(PAPI_add_remove_events)
add_example(PAPI_epc)
add_example(PAPI_flips)
add_example(PAPI_flops)
add_example(PAPI_get_executable_info)
add_example(PAPI_get_opt)
add_example(PAPI_get_real_cyc)
add_example(PAPI_get_virt_cyc)
add_example(PAPI_hw_info)
add_example(PAPI_ipc)
add_example(PAPI_mix_hl_ll)
add_example(PAPI_mix_hl_rate)
add_example(PAPI_mix_ll_rate)
add_example(PAPI_overflow)
add_example(PAPI_perror)
add_example(PAPI_profil)
add_example(PAPI_reset)
add_example(PAPI_set_domain)
add_example(PAPI_state)
add_example(sprofile)

# Pthread examples.
# TODO: branch them out, based on pthread option
add_example(locks_pthreads)
add_example(overflow_pthreads)

