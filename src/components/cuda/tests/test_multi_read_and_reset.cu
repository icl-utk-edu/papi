/**
* @file test_multi_read_and_reset.cu
* @brief This test has three function calls that will be executed:
*        1. multi_reset - Performs multiple PAPI_reset's.
*        2. multi_read - Performs multiple PAPI_read's. 
*        3. single_read - Performs a single PAPI_stop, which internally calls PAPI_read.
*
*        Note: The cuda component supports being partially disabled, meaning that certain devices
*        will not be "enabled" to profile on. If PAPI_CUDA_API is not set, then devices with
*        CC's >= 7.0 will be used and if PAPI_CUDA_API is set to LEGACY then devices with
*        CC's <= 7.0 will be used.
*/

// Standard library headers
#include <stdio.h>
#include <stdlib.h>

// Internal headers
#include "cuda_tests_helper.h"
#include "gpu_work.h"
#include "papi.h"
#include "papi_test.h"

#define MAX_EVENT_COUNT (32)
int suppress_output;

static void print_help_message(void)
{
    printf("./test_multi_read_and_reset --device [nvidia device index] --cuda-native-event-names [list of cuda native event names separated by a comma].\n"
           "Notes:\n"
           "1. The device index must match the device qualifier if provided.\n");
}

static void parse_and_assign_args(int argc, char *argv[], int *device_index, char ***cuda_native_event_names, int *total_event_count)
{
    int num_device_indices = 0, *event_device_indices = NULL;
    int i, device_arg_found = 0, cuda_native_event_name_arg_found = 0;
    for (i = 1; i < argc; ++i)
    {   
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0)
        {   
            print_help_message();
            exit(EXIT_SUCCESS);
        }   
        else if (strcmp(arg, "--device") == 0)
        {   
            if (!argv[i + 1]) 
            {   
                printf("ERROR!! Add a nvidia device index.\n");
                exit(EXIT_FAILURE);
            }   
            *device_index = atoi(argv[i + 1]);
            device_arg_found++;
            i++;
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
                const char *device_substring = strstr(cuda_native_event_name, ":device=");
                if (device_substring != NULL) {
                    event_device_indices = (int *) realloc(event_device_indices, (num_device_indices + 1) *  sizeof(int));
                    event_device_indices[num_device_indices++] = atoi(device_substring + strlen(":device=")); 
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
                cuda_native_event_name_arg_found++; 
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

    if (device_arg_found == 0 || cuda_native_event_name_arg_found == 0) {
        fprintf(stderr, "You must use both the --device arg and --cuda-native-event-names arg in conjunction.\n");
        exit(EXIT_FAILURE);
    }   

    for (i = 0; i < num_device_indices; i++) {
        if ((*device_index) != event_device_indices[i]) {
            fprintf(stderr, "The device qualifier index %d does not match the index %d provided by --device.\n", event_device_indices[i], *device_index);
            exit(EXIT_FAILURE);
        }
    }

    free(event_device_indices);
}

int approx_equal(long long v1, long long v2)
{
    double err = fabs(v1 - v2) / v1;
    if (err < 0.1)
        return 1;
    return 0;
}

// Globals for successfully added and multiple pass events
int global_num_events_successfully_added = 0, global_num_multipass_events = 0;

void multi_reset(int total_event_count, char **cuda_native_event_names, long long *cuda_counter_values, int cuda_device_index)
{
    CUcontext ctx;
    int flags = 0;
    CUdevice device = cuda_device_index;
#if defined(CUDA_TOOLKIT_GE_13)
    check_cuda_driver_api_call( cuCtxCreate(&ctx, (CUctxCreateParams*)0, flags, device) ); 
#else
    check_cuda_driver_api_call( cuCtxCreate(&ctx, flags, device) );
#endif

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    // Handle the events from the command line
    global_num_events_successfully_added = 0;
    global_num_multipass_events = 0;
    char **events_successfully_added = (char **) malloc(total_event_count * sizeof(char *));
    check_memory_allocation_call(events_successfully_added);

    int event_idx;
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        events_successfully_added[event_idx] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        check_memory_allocation_call(events_successfully_added[event_idx]);

        add_cuda_native_events(EventSet, cuda_native_event_names[event_idx], &global_num_events_successfully_added, events_successfully_added, &global_num_multipass_events);
    }

    // Only multiple pass events were provided on the command line
    if (global_num_events_successfully_added == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    check_papi_api_call( PAPI_start(EventSet) );

    int iter;
    for (iter = 0; iter < 10; iter++) {
        VectorAddSubtract(100000, suppress_output);
       
        check_papi_api_call( PAPI_read(EventSet, cuda_counter_values) );

        for (event_idx = 0; event_idx < global_num_events_successfully_added; event_idx++) {
            PRINT(suppress_output, "Event %s for iter %d produced the value:\t\t%lld\n", events_successfully_added[event_idx], iter, cuda_counter_values[event_idx]);
        }

        check_papi_api_call( PAPI_reset(EventSet) );
    }

    check_papi_api_call( PAPI_stop(EventSet, cuda_counter_values) );

    check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    check_cuda_driver_api_call( cuCtxDestroy(ctx) );

    // Free allocated memory
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        free(events_successfully_added[event_idx]);
    }
    free(events_successfully_added);
}

void multi_read(int total_event_count, char **cuda_native_event_names, long long *cuda_counter_values, int cuda_device_index)
{
    CUcontext ctx;
    int flags = 0;
    CUdevice device = cuda_device_index;
#if defined(CUDA_TOOLKIT_GE_13)
    check_cuda_driver_api_call( cuCtxCreate(&ctx, (CUctxCreateParams*)0, flags, device) );
#else
    check_cuda_driver_api_call( cuCtxCreate(&ctx, flags, device) );
#endif

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    // Handle the events from the command line
    global_num_events_successfully_added = 0;
    global_num_multipass_events = 0;
    char **events_successfully_added = (char **) malloc(total_event_count * sizeof(char *));
    check_memory_allocation_call(events_successfully_added);

    int event_idx;
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        events_successfully_added[event_idx] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        check_memory_allocation_call(events_successfully_added[event_idx]);

        add_cuda_native_events(EventSet, cuda_native_event_names[event_idx], &global_num_events_successfully_added, events_successfully_added, &global_num_multipass_events);
    }

    // Only multiple pass events were provided on the command line
    if (global_num_events_successfully_added == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    check_papi_api_call( PAPI_start(EventSet) );

    int iter;
    for (iter = 0; iter < 10; iter++) {
        VectorAddSubtract(100000, suppress_output);

        check_papi_api_call( PAPI_read(EventSet, cuda_counter_values) );

        for (event_idx = 0; event_idx < global_num_events_successfully_added; event_idx++) {
            PRINT(suppress_output, "Event %s for iter %d produced the value:\t\t%lld\n", events_successfully_added[event_idx], iter, cuda_counter_values[event_idx]);
        }
    }

    check_papi_api_call( PAPI_stop(EventSet, cuda_counter_values) );

    check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    check_cuda_driver_api_call( cuCtxDestroy(ctx) );

    // Free allocated memory
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        free(events_successfully_added[event_idx]);
    }
    free(events_successfully_added);
}

void single_read(int total_event_count, char **cuda_native_event_names, long long *cuda_counter_values, char ***addedEvents, int cuda_device_index)
{
    CUcontext ctx;
    int flags = 0;
    CUdevice device = cuda_device_index;
#if defined(CUDA_TOOLKIT_GE_13)
    check_cuda_driver_api_call( cuCtxCreate(&ctx, (CUctxCreateParams*)0, flags, device) );
#else
    check_cuda_driver_api_call( cuCtxCreate(&ctx, flags, device) );
#endif

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    // Handle the events from the command line
    global_num_events_successfully_added = 0;
    global_num_multipass_events = 0;
    char **events_successfully_added = (char **) malloc(total_event_count * sizeof(char *));
    check_memory_allocation_call(events_successfully_added);

    int event_idx;
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        events_successfully_added[event_idx] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        check_memory_allocation_call(events_successfully_added[event_idx]);

        add_cuda_native_events(EventSet, cuda_native_event_names[event_idx], &global_num_events_successfully_added, events_successfully_added, &global_num_multipass_events);
    }

    // Only multiple pass events were provided on the command line
    if (global_num_events_successfully_added == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    check_papi_api_call( PAPI_start(EventSet) );

    int iter;
    for (iter = 0; iter < 10; iter++) {
        VectorAddSubtract(100000, suppress_output);
    }

    check_papi_api_call( PAPI_stop(EventSet, cuda_counter_values) );

    for (event_idx = 0; event_idx < global_num_events_successfully_added; event_idx++) {
        PRINT(suppress_output, "Event %s for a single read produced the value:\t\t%lld\n", events_successfully_added[event_idx], cuda_counter_values[event_idx]);
    }

    check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    check_cuda_driver_api_call( cuCtxDestroy(ctx) );

    *addedEvents = events_successfully_added;
}

int main(int argc, char **argv)
{
    check_cuda_driver_api_call( cuInit(0) );

    // Determine the number of Cuda capable devices
    int num_devices = 0;
    check_cuda_runtime_api_call( cudaGetDeviceCount(&num_devices) );
    // No devices detected on the machine, exit
    if (num_devices < 1) {
        fprintf(stderr, "No NVIDIA devices found on the machine. This is required for the test to run.\n");
        exit(EXIT_FAILURE);
    }

    suppress_output = 0;
    char *user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    if (user_defined_suppress_output) {
        suppress_output = (int) strtol(user_defined_suppress_output, (char**) NULL, 10);
    } 
    PRINT(suppress_output, "Running the cuda component test test_multi_read_and_reset.cu\n");

    int cuda_device_index = -1; 
    char **cuda_native_event_names = NULL;
    // If command line arguments are provided then get their values.
    int total_event_count = 0;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &cuda_device_index, &cuda_native_event_names, &total_event_count);
    }

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init()", papi_errno);
    }
    PRINT(suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION));

    // Verify the cuda component has been compiled in
    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", cuda_cmp_idx);
    } 
    PRINT(suppress_output, "The cuda component is assigned to component index: %d\n", cuda_cmp_idx); 

    // If a user does not provide an event or events, then we go get an event to add
    if (total_event_count == 0) {
        enumerate_and_store_cuda_native_events(&cuda_native_event_names, &total_event_count, &cuda_device_index);
    }

    PRINT(suppress_output, "Running Multi Reset\n");
    PRINT(suppress_output, "----------------------------------------\n");
    long long cuda_counter_values_multi_reset[MAX_EVENT_COUNT];
    multi_reset(total_event_count, cuda_native_event_names, cuda_counter_values_multi_reset, cuda_device_index);
    PRINT(suppress_output, "----------------------------------------\n");

    PRINT(suppress_output, "\nRunning Multi Read\n");
    PRINT(suppress_output, "----------------------------------------\n");
    long long cuda_counter_values_multi_read[MAX_EVENT_COUNT];
    multi_read(total_event_count, cuda_native_event_names, cuda_counter_values_multi_read, cuda_device_index);
    PRINT(suppress_output, "----------------------------------------\n");

    PRINT(suppress_output, "\nRunning Single Read\n");
    PRINT(suppress_output, "----------------------------------------\n");
    long long cuda_counter_values_single_read[MAX_EVENT_COUNT];
    char **events_successfully_added = { 0 };
    single_read(total_event_count, cuda_native_event_names, cuda_counter_values_single_read, &events_successfully_added, cuda_device_index);
    PRINT(suppress_output, "----------------------------------------\n");

    int event_idx;
    PRINT(suppress_output, "\nFinal Measured Cuda Counter Values\n");
    PRINT(suppress_output, "----------------------------------------\n");
    PRINT(suppress_output, "Event Name\t\t\t\t\t\tMulti Read\tSingle Read\n");
    for (event_idx = 0; event_idx < global_num_events_successfully_added; event_idx++) {
        PRINT(suppress_output, "%s\t\t\t%lld\t\t%lld\n", events_successfully_added[event_idx], cuda_counter_values_multi_read[event_idx], cuda_counter_values_single_read[event_idx]);
        if ( !approx_equal(cuda_counter_values_multi_read[event_idx], cuda_counter_values_single_read[event_idx]) )
            printf("\033[33mWARNING: Multi read and single read do not match for %s\033[0m\n", events_successfully_added[event_idx]);
    }

    // Output a note that a multiple pass event was provided on the command line
    if (global_num_multipass_events > 0) {
        PRINT(suppress_output, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    // Free allocated memory
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        free(cuda_native_event_names[event_idx]);
    }
    free(cuda_native_event_names);

    for (event_idx = 0; event_idx < global_num_events_successfully_added; event_idx++) {
        free(events_successfully_added[event_idx]);
    }
    free(events_successfully_added);

    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
