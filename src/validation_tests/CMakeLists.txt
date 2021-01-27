
set (validation_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/branches_testcode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/busy_work.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cache_helper.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cache_testcode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/display_error.c
    ${CMAKE_CURRENT_SOURCE_DIR}/flops_testcode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/instructions_testcode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_multiply.c)

add_library (validation_objects OBJECT ${validation_srcs})

target_include_directories (validation_objects PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/..
    ${PAPI_TESTLIB_SOURCE_DIR})

macro (add_validation valid_name dep_list)
    add_executable(${valid_name} ${valid_name}.c)

    foreach (dep ${dep_list})
        target_sources (${valid_name}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/${dep}.c)
    endforeach (dep dep_list)

    target_link_libraries (${valid_name}
        PRIVATE
        testlib
        papi_shared
        ${CMAKE_DL_LIBS}
        m)
    target_include_directories (${valid_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${PAPI_TESTLIB_SOURCE_DIR})

endmacro (add_validation valid_name)

add_validation(cycles_validation "display_error;instructions_testcode" )
add_validation(papi_tot_ins "display_error;instructions_testcode")

add_validation(fp_validation_hl "flops_testcode")

add_validation(memleak_check "display_error;branches_testcode")
add_validation(papi_br_cn "display_error;branches_testcode")
add_validation(papi_br_ins "display_error;branches_testcode")
add_validation(papi_br_msp "display_error;branches_testcode")
add_validation(papi_br_ntk "display_error;branches_testcode")
add_validation(papi_br_prc "display_error;branches_testcode")
add_validation(papi_br_tkn "display_error;branches_testcode")
add_validation(papi_br_ucn "display_error;branches_testcode")

add_validation(flops_validation "display_error;branches_testcode;flops_testcode")
add_validation(papi_dp_ops "display_error;branches_testcode;flops_testcode")
add_validation(papi_fp_ops "display_error;branches_testcode;flops_testcode")
add_validation(papi_sp_ops "display_error;branches_testcode;flops_testcode")

add_validation(papi_hw_int "display_error")

add_validation(papi_l1_dca "cache_testcode;display_error;matrix_multiply")

add_validation(papi_l1_dcm "cache_helper;cache_testcode;display_error;matrix_multiply")
add_validation(papi_l2_dca "cache_helper;cache_testcode;display_error;matrix_multiply")
add_validation(papi_l2_dcm "cache_helper;cache_testcode;display_error;matrix_multiply")
add_validation(papi_l2_dcr "cache_helper;cache_testcode;display_error;matrix_multiply")
add_validation(papi_l2_dcw "cache_helper;cache_testcode;display_error;matrix_multiply")

add_validation(papi_ld_ins "display_error;matrix_multiply")
add_validation(papi_sr_ins "display_error;matrix_multiply")
add_validation(papi_tot_cyc "display_error;matrix_multiply")

add_validation(papi_ref_cyc "display_error;flops_testcode")

#TODO: That file is not referenced in PAPI itself
#add_validation(vector_testcode)

