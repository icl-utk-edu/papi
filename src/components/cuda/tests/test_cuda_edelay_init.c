#include <stdio.h>
#include <stdlib.h>

#include "papi.h"

int main()
{
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "Failed to initialize the PAPI library.\n");
        exit(1);
    }

    const char *component_name = "cuda";
    int cidx = PAPI_get_component_index(component_name);
    if (cidx < 0) {
        fprintf(stderr, "Failed to get the cuda component index.\n");
        exit(1);
    }

    const PAPI_component_info_t *cmpinfo = PAPI_get_component_info(cidx);
    if (cmpinfo == NULL) {
        fprintf(stderr, "Failed to get the component info.\n");
        exit(1);
    }

    // Test 1
    if (cmpinfo->disabled != PAPI_EDELAY_INIT) {
        fprintf(stderr, "The cuda component should be PAPI_EDELAY_INIT.\n");
        exit(1);
    }

    // Test 2
    if (cmpinfo->num_native_events != -1) {
        fprintf(stderr, "The cuda component should have -1 events set.\n");
        exit(1);
    }

    int eventcode = 0 | PAPI_NATIVE_MASK;
    retval = PAPI_enum_cmp_event(&eventcode, PAPI_ENUM_FIRST, cidx);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Failed to PAPI_enum_cmp_event.\n");
        exit(1);
    }

    // Test 3
    if (cmpinfo->disabled != PAPI_OK) {
        fprintf(stderr, "Failed to initialize the Cuda component.\n");
        exit(1);
    }

    // Test 4
    if (cmpinfo->num_native_events <= 0) {
        fprintf(stderr, "Num native events should be greater than 0.\n");
        exit(1);
    }

    printf("The test passed.\n");

    return 0;
}
