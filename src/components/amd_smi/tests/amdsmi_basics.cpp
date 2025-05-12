//-----------------------------------------------------------------------------
// amd_smi_single_event_test.cpp
// Enumerates every native AMD-SMI event exposed through PAPI and measures
// them **one at a time**.  This isolates each counter in its own EventSet so
// that you can verify the event works independently of the others.
// Designed for C++17 / hipcc builds.
//
// Build example:
//   make -f ROCM_SMI_Makefile amd_smi_single_event_test.out
//-----------------------------------------------------------------------------

#define __HIP_PLATFORM_HCC__

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "papi.h"



// ---------------------------------------------------------------------------
// Simple helper for PAPI error handling
// ---------------------------------------------------------------------------
#define CALL_PAPI_OK(call)                                                         \
    do {                                                                          \
        int _ret = (call);                                                        \
        if (_ret != PAPI_OK) {                                                    \
            fprintf(stderr, "%s:%d: PAPI error in '" #call "': %s\n",            \
                    __FILE__, __LINE__, PAPI_strerror(_ret));                    \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

int main(int argc, char *argv[])
{
    //-------------------------------------------------------------------
    // 1.  Initialise PAPI
    //-------------------------------------------------------------------
    int ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed: %s\n", PAPI_strerror(ret));
        return EXIT_FAILURE;
    }

    //-------------------------------------------------------------------
    // 2.  Locate the AMD-SMI component
    //-------------------------------------------------------------------
    int cid = -1;
    const int ncomps = PAPI_num_components();
    for (int i = 0; i < ncomps && cid < 0; ++i) {
        const PAPI_component_info_t* cinfo = PAPI_get_component_info(i);
        if (cinfo && std::strcmp(cinfo->name, "amd_smi") == 0) {
            cid = i;
        }
    }
    if (cid < 0) {
        fprintf(stderr, "Unable to locate the amd_smi component ? is PAPI built with ROCm support?\n");
        return EXIT_FAILURE;
    }
    printf("Using AMD-SMI component id %d\n\n", cid);

    //-------------------------------------------------------------------
    // 3.  Enumerate every native event
    //-------------------------------------------------------------------
    int ev_code = PAPI_NATIVE_MASK;
    if (PAPI_enum_cmp_event(&ev_code, PAPI_ENUM_FIRST, cid) != PAPI_OK) {
        fprintf(stderr, "No native events found for AMD-SMI component.\n");
        return EXIT_SUCCESS; // Nothing more to do
    }

    int event_index = 0;
    do {
        char ev_name[PAPI_MAX_STR_LEN]{};
        if (PAPI_event_code_to_name(ev_code, ev_name) != PAPI_OK) {
            // Should not happen, but skip if it does.
            continue;
        }

        printf("[%4d] Testing %s...\n", event_index++, ev_name);

        //-------------------------------------------------------------------
        // 4-7.  Create a fresh EventSet, read the event, print, cleanup
        //-------------------------------------------------------------------
        int eventSet = PAPI_NULL;
        CALL_PAPI_OK(PAPI_create_eventset(&eventSet));
        CALL_PAPI_OK(PAPI_assign_eventset_component(eventSet, cid));

        ret = PAPI_add_event(eventSet, ev_code);
        if (ret != PAPI_OK) {
            fprintf(stderr, "  ?  Could not add %s (%s)\n\n", ev_name, PAPI_strerror(ret));
            CALL_PAPI_OK(PAPI_destroy_eventset(&eventSet));
            continue;
        }

        long long value = 0;
        CALL_PAPI_OK(PAPI_start(eventSet));
        CALL_PAPI_OK(PAPI_stop(eventSet, &value));

        printf("      %-60s = %lld\n\n", ev_name, value);

        CALL_PAPI_OK(PAPI_cleanup_eventset(eventSet));
        CALL_PAPI_OK(PAPI_destroy_eventset(&eventSet));

    } while (PAPI_enum_cmp_event(&ev_code, PAPI_ENUM_EVENTS, cid) == PAPI_OK);

    //-------------------------------------------------------------------
    // 8.  Shutdown
    //-------------------------------------------------------------------
    PAPI_shutdown();
    return EXIT_SUCCESS;
}
