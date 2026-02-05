/**
* @file test_cuda_edelay_init.c
* @brief Verify that the PAPI_EDELAY_INIT functionality is working properly with the
*        cuda component. The test does this by:
*
*        1. Verifying that before accessing cuda native events the disabled member
*           variable of PAPI_component_info_t is set to -26 (PAPI_EDELAY_INIT) and
*           num_native_events is equal to -1.
*        2. Verfiying that after accessing cuda native events the disabled member
*           variable of PAPI_comopnent_info_t is set to 0 (PAPI_OK) and num_native_events
*           is > 0. 
*
*/

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"
#include "papi_test.h"

int test_cuda_is_edelay_init()
{
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed: %d\n", retval);
        return retval;    
    }

    const char *component_name = "cuda";
    int cidx = PAPI_get_component_index(component_name);
    if (cidx < 0) {
        fprintf(stderr, "PAPI_get_component_index failed: %d\n", retval);
        return retval;
    }

    const PAPI_component_info_t *cmpinfo = PAPI_get_component_info(cidx);
    if (cmpinfo == NULL) {
        fprintf(stderr, "PAPI_get_component_info failed.\n");
        return PAPI_ENOMEM;
    }

    if (cmpinfo->disabled != PAPI_EDELAY_INIT) {
        fprintf(stderr, "The cuda component should be PAPI_EDELAY_INIT.\n");
        return PAPI_ECOMBO;
    }   

    if (cmpinfo->num_native_events == -1) {
        fprintf(stderr, "The cuda component should have the value of -1 set for num_native_events.\n");
        return PAPI_ECOMBO;
    }

    PAPI_shutdown();


    return PAPI_OK;
}

int test_cuda_is_not_edelay_init_via_enum_cmp_event()
{
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed: %d\n", retval);
        return retval;
    }
    
    const char *component_name = "cuda";
    int cidx = PAPI_get_component_index(component_name);
    if (cidx < 0) {
        fprintf(stderr, "PAPI_get_component_index failed: %d\n", retval);
        return retval;
    }
    
    const PAPI_component_info_t *cmpinfo = PAPI_get_component_info(cidx);
    if (cmpinfo == NULL) {
        fprintf(stderr, "PAPI_get_component_info failed.\n");
        return PAPI_ENOMEM;
    } 

    int eventcode = 0 | PAPI_NATIVE_MASK;
    int modifier = PAPI_ENUM_FIRST;
    retval = PAPI_enum_cmp_event(&eventcode, modifier, cidx);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_enum_cmp_event failed: %d\n", retval);
        return retval;
    }

    if (cmpinfo->disabled != PAPI_OK) {
        fprintf(stderr, "The cuda component failed to be initialized.\n");
        return PAPI_ECOMBO;
    }

    if (cmpinfo->num_native_events < 0) {
        fprintf(stderr, "The cuda component should have a value greater than -1 set for num_native_events.\n");
        return PAPI_ECOMBO;
    }

    PAPI_shutdown();

    return PAPI_OK;
}

int main()
{
    int retval = test_cuda_is_edelay_init();
    if (retval != PAPI_OK) {
        return retval;
    }

    retval = test_cuda_is_not_edelay_init_via_enum_cmp_event();
    if (retval != PAPI_OK) {
        return retval;
    }

    return PAPI_OK;
}
