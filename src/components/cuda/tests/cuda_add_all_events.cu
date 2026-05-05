/**
* @file cuda_add_all_events.cu
* @brief This test attempts to enumerate and add all base cuda native event names i.e cuda:::dram__bytes.
*        To see cuda native events for specific NVIDIA GPUs use either the --device argument or set
*        CUDA_VISIBLE_DEVICES.
*
*        Note 1: The cuda component supports being partially disabled, meaning that certain devices
*        will not be "enabled" to profile on. If PAPI_CUDA_API is not set, then devices with
*        CC's >= 7.0 will be used and if PAPI_CUDA_API is set to LEGACY then devices with
*        CC's <= 7.0 will be used.
*
*        Note 2: If the Perfworks Metrics API is used (CC's >=7.0) this test may take 12 minutes of wall clock time to
*        finish running.
*/

// Standard library headers
#include <stdio.h>

// Internal headers
#include "cuda_tests_helper.h"
#include "papi.h"
#include "papi_test.h"

static void print_help_message(void)
{
    printf("./cuda_add_all_events --devices [list of nvidia device indexes separated by a comma].\n"
           "Notes:\n"
           "1. CUDA_VISIBLE_DEVICES will be set IF device indexes have been provided to the arg --devices.\n");
}

static void parse_and_assign_args(int argc, char *argv[], char **device_indexes)
{
    int i;
    for (i = 1; i < argc; ++i)
    {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0)
        {
            print_help_message();
            exit(EXIT_SUCCESS);
        }
        else if (strcmp(arg, "--devices") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add a nvidia device index.\n");
                exit(EXIT_FAILURE);
            }
            *device_indexes = argv[i + 1];
            i++;
        }
        else
        {
            print_help_message();
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char **argv)
{
    // Determine the number of Cuda capable devices
    int num_devices = 0;
    check_cuda_runtime_api_call( cudaGetDeviceCount(&num_devices) );

    // No devices detected on the machine, exit
    if (num_devices < 1) {
        fprintf(stderr, "No NVIDIA devices found on the machine. This is required for the test to run. Exiting.\n");
        exit(EXIT_FAILURE);
    }

    char *device_indexes = NULL;
    parse_and_assign_args(argc, argv, &device_indexes);
    if (device_indexes != NULL) {
        int overwrite = 0;
        int setenv_errno = setenv("CUDA_VISIBLE_DEVICES", device_indexes, overwrite);
        if (setenv_errno == -1) {
            fprintf(stderr, "Failed to set CUDA_VISIBLE_DEVICES to %s. Proceeding.\n", device_indexes);
        }
    }

    int suppress_output = 0;
    char *user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    if (user_defined_suppress_output) {
        suppress_output = (int) strtol(user_defined_suppress_output, (char**) NULL, 10);
    }
    PRINT(suppress_output, "Running the cuda component test cudaOpenMP.cu\n");

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__,__LINE__, "PAPI_library_init()", papi_errno);
    }
    PRINT(suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION));

    const char *component_name = "cuda";
    int cidx = PAPI_get_component_index(component_name);
    if (cidx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", PAPI_ENOCMP);
    }

    int eventCode = 0 | PAPI_NATIVE_MASK;
    int modifier = PAPI_ENUM_FIRST;
    check_papi_api_call( PAPI_enum_cmp_event(&eventCode, modifier, cidx) );

    modifier = PAPI_ENUM_EVENTS;
    int baseEventIndex = 0;
    do {
        long long counterValue = 0;
        PAPI_event_info_t eventInfo;
        check_papi_api_call( PAPI_get_event_info(eventCode, &eventInfo) );
        int isWithoutSubmetric = verify_event_is_without_submetric(eventInfo.symbol);
        // Skip native events that have submetrics appended to them
        if (isWithoutSubmetric == 0) {
            continue;
        }

        int eventSet = PAPI_NULL;
        check_papi_api_call( PAPI_create_eventset(&eventSet) );

        papi_errno = PAPI_add_named_event(eventSet, eventInfo.symbol);
        if (papi_errno != PAPI_OK) {
            // PAPI does not support multiple pass events. Skipping.
            if (papi_errno == PAPI_EMULPASS) {
                goto cleanup;
            }
        }

        check_papi_api_call( PAPI_start(eventSet) );

        check_papi_api_call( PAPI_stop(eventSet, &counterValue) );

        printf("Base Event#: %d\n", baseEventIndex++);
        printf("PAPI Eventcode: %#x\n", eventInfo.event_code);
        printf("PAPI Eventname: %s\n", eventInfo.symbol);
        printf("PAPI Longdescr: %s\n", eventInfo.long_descr);
        printf("PAPI Countervalue: %lld\n\n", counterValue);

      cleanup:
        check_papi_api_call( PAPI_cleanup_eventset(eventSet) );
        check_papi_api_call( PAPI_destroy_eventset(&eventSet) );
    } while (PAPI_enum_cmp_event(&eventCode, modifier, cidx) == PAPI_OK);

    return 0;
}
