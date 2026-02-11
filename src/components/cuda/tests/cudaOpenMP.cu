/**
* @file cudaOpenMP.cu
* @brief For all NVIDIA devices detected on the machine create a matching thread
*        for it using OpenMP. Even though a thread is created for all NVIDIA devices,
*        cuCtxCreate will be called only for enabled devices.
*
*        Note: The cuda component supports being partially disabled, meaning that certain devices
*        will not be "enabled" to profile on. If PAPI_CUDA_API is not set, then devices with
*        CC's >= 7.0 will be used and if PAPI_CUDA_API is set to LEGACY then devices with
*        CC's <= 7.0 will be used.
*
*        For each enabled device, their matching thread will have a workflow of:
*            1. Creating an EventSet
*            2. Adding events to the EventSet
*            3. Starting the EventSet
*            4. Stopping the EventSet
*
*        Finally, a compiler that supports OpenMP 2.0 is needed.
*/

// Standard library headers
#include <omp.h>
#include <stdio.h>

// Internal headers
#include "cuda_tests_helper.h"
#include "gpu_work.h"
#include "papi.h"
#include "papi_test.h"

#define MAX_THREADS (32)

static void print_help_message(void)
{
    printf("./cudaOpenMP --cuda-native-event-names [list of cuda native event names separated by a comma].\n"
           "Notes:\n"
           "1. Native event names must not have the device qualifier appended.\n");
}

static void parse_and_assign_args(int argc, char *argv[], char ***cuda_native_event_names, int *total_event_count)
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
        else if (strcmp(arg, "--cuda-native-event-names") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! --cuda-native-event-names given, but no events listed.\n");
                exit(EXIT_FAILURE);
            }

            char **cmd_line_native_event_names = NULL;
            const char *cuda_native_event_name = strtok(argv[i+1], ",");
            while (cuda_native_event_name != NULL)
            {
                if (strstr(cuda_native_event_name, ":device")) {
                    fprintf(stderr, "Cuda native event name must not have a device qualifier appended for this test, i.e. no :device=#.\n");
                    print_help_message();
                    exit(EXIT_FAILURE);
                }

                cmd_line_native_event_names = (char **) realloc(cmd_line_native_event_names, ((*total_event_count) + 1) * sizeof(char *));
                check_memory_allocation_call(cmd_line_native_event_names);

                cmd_line_native_event_names[(*total_event_count)] = (char *) malloc(PAPI_2MAX_STR_LEN * sizeof(char));
                check_memory_allocation_call(cmd_line_native_event_names[(*total_event_count)]);

                int strLen = snprintf(cmd_line_native_event_names[(*total_event_count)], PAPI_2MAX_STR_LEN, "%s", cuda_native_event_name);
                if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN) {
                    fprintf(stderr, "Failed to fully write cuda native event name.\n");
                    exit(EXIT_FAILURE);
                }

                (*total_event_count)++;
                cuda_native_event_name = strtok(NULL, ",");
            }
            i++;
            *cuda_native_event_names = cmd_line_native_event_names;
        }
        else
        {
            print_help_message();
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char *argv[])
{
    // Determine the number of Cuda capable devices
    int num_devices = 0;
    check_cuda_runtime_api_call( cudaGetDeviceCount(&num_devices) );

    // No devices detected on the machine, exit
    if (num_devices < 1) {
        fprintf(stderr, "No NVIDIA devices found on the machine. This is required for the test to run.\n");
        exit(EXIT_FAILURE);
    }

    int suppress_output = 0;
    char *user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    if (user_defined_suppress_output) {
        suppress_output = (int) strtol(user_defined_suppress_output, (char**) NULL, 10);
    }   
    PRINT(suppress_output, "Running the cuda component test cudaOpenMP.cu\n");

    char **cuda_native_event_names = NULL;
    // If command line arguments are provided then get their values.
    int total_event_count = 0;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &cuda_native_event_names, &total_event_count);
    }

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if ( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed()", papi_errno);
    }
    PRINT(suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION));

    // Initialize thread support in PAPI
    check_papi_api_call( PAPI_thread_init((unsigned long (*)(void)) omp_get_thread_num) );

    // Verify the cuda component has been compiled in
    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", cuda_cmp_idx);
    }
    PRINT(suppress_output, "The cuda component is assigned to component index: %d\n", cuda_cmp_idx);  

    // Initialize the Cuda component
    int cuda_eventcode = 0 | PAPI_NATIVE_MASK;
    check_papi_api_call( PAPI_enum_cmp_event(&cuda_eventcode, PAPI_ENUM_FIRST, cuda_cmp_idx) );

    // If we have not gotten an event via the command line, use the event obtained from PAPI_enum_cmp_event
    if (total_event_count == 0) {
        int num_spaces_to_allocate = 1;
        cuda_native_event_names = (char **) malloc(num_spaces_to_allocate * sizeof(char *));
        check_memory_allocation_call( cuda_native_event_names );

        cuda_native_event_names[total_event_count] = (char *) malloc(PAPI_2MAX_STR_LEN * sizeof(char));
        check_memory_allocation_call( cuda_native_event_names[total_event_count] );

        check_papi_api_call( PAPI_event_code_to_name(cuda_eventcode, cuda_native_event_names[total_event_count++]) );
    }

    const PAPI_component_info_t *cmpInfo = PAPI_get_component_info(cuda_cmp_idx);
    if (cmpInfo == NULL) {
        fprintf(stderr, "Call to PAPI_get_component_info failed.\n");
        exit(EXIT_FAILURE);
    }   

    // Check to see if the Cuda component is partially disabled
    if (cmpInfo->partially_disabled) {
        const char *cc_support = (getenv("PAPI_CUDA_API") != NULL) ? "<=7.0" : ">=7.0";
        PRINT(suppress_output, "\033[33mThe cuda component is partially disabled. Only support for CC's %s are enabled.\033[0m\n", cc_support);
    }

    // Determine the number of threads we will launch based off the number of
    // Cuda devices on the machine (max of 32).
    int num_threads_and_devs = (num_devices > MAX_THREADS) ? MAX_THREADS : num_devices;
    omp_set_num_threads(num_threads_and_devs);
    PRINT(suppress_output, "Total number of threads to be launched: %d\n", num_threads_and_devs);
    int i, thread_and_dev_idx, event_idx, numMultipassEvents = 0;
    #pragma omp parallel for
    for (thread_and_dev_idx = 0; thread_and_dev_idx < num_threads_and_devs; thread_and_dev_idx++) {
        const PAPI_component_info_t *cmpInfo = PAPI_get_component_info(cuda_cmp_idx);
        if (cmpInfo == NULL) {
            fprintf(stderr, "Call to PAPI_get_component_info failed.\n");
            exit(EXIT_FAILURE);
        }

        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(thread_and_dev_idx) == 0) {
                continue;
            }   
        } 

        CUcontext ctx;
        int flags = 0;
        CUdevice device = thread_and_dev_idx;
#if defined(CUDA_TOOLKIT_GE_13)
        check_cuda_driver_api_call( cuCtxCreate(&ctx, (CUctxCreateParams*)0, flags, device) );
#else
        check_cuda_driver_api_call( cuCtxCreate(&ctx, flags, device) );
#endif
        int EventSet = PAPI_NULL;
        check_papi_api_call( PAPI_create_eventset(&EventSet) );

        int num_events_successfully_added = 0;
        char **events_successfully_added = (char **) malloc(total_event_count * sizeof(char *));
        check_memory_allocation_call(events_successfully_added);

        for (event_idx = 0; event_idx < total_event_count; event_idx++) {
            char tmp_event_name[PAPI_MAX_STR_LEN];
            int strLen = snprintf(tmp_event_name, PAPI_MAX_STR_LEN, "%s:device=%d", cuda_native_event_names[event_idx], thread_and_dev_idx);
            if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                fprintf(stderr, "Failed to fully write event name with appended device qualifier.\n");
                exit(EXIT_FAILURE);
            }

            events_successfully_added[event_idx] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
            check_memory_allocation_call(events_successfully_added[event_idx]);

            add_cuda_native_events(EventSet, tmp_event_name, &num_events_successfully_added, events_successfully_added, &numMultipassEvents);
        }

        // Only multiple pass events were provided on the command line
        if (num_events_successfully_added == 0) {
            fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
            exit(EXIT_FAILURE);
        }

        check_papi_api_call( PAPI_start(EventSet) );

        // Work for the device
        VectorAddSubtract(50000 * (thread_and_dev_idx + 1), suppress_output);

        long long cuda_counter_values[MAX_THREADS];
        check_papi_api_call( PAPI_stop(EventSet, cuda_counter_values) );

        printf("num_events_successfully: %d\n", num_events_successfully_added);
        for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++) {
            PRINT(suppress_output, "Event %s on thread and device id %d produced the value:\t\t%lld\n", events_successfully_added[event_idx], thread_and_dev_idx, cuda_counter_values[event_idx]);
        }

        // Free allocated memory
        for (event_idx = 0; i < num_events_successfully_added; event_idx++) {
            free(events_successfully_added[event_idx]);
        }
        free(events_successfully_added);

        check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

        check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

        check_cuda_driver_api_call( cuCtxDestroy(ctx) );
    } // End omp parallel for loop region

    // Output a note that a multiple pass event was provided on the command line
    if (numMultipassEvents > 0) {
        PRINT(suppress_output, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    // Free allocated memory
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        free(cuda_native_event_names[event_idx]);
    }
    free(cuda_native_event_names);

    PAPI_shutdown();

    test_pass(__FILE__); 

    return 0;
}
