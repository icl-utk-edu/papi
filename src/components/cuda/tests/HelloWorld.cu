/**
* @file HelloWorld.cu
* @brief This test serves as a very simple hello world c example where the string
*        "Hello World!" is mangled and then restored. cuCtxCreate is used for context
*        creation.
*
*        Note: The cuda component supports being partially disabled, meaning that certain devices
*        will not be "enabled" to profile on. If PAPI_CUDA_API is not set, then devices with
*        CC's >= 7.0 will be used and if PAPI_CUDA_API is set to LEGACY then devices with
*        CC's <= 7.0 will be used.
*/

// Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Cuda Toolkit headers
#include <cuda.h>

// Internal headers
#include "cuda_tests_helper.h"
#include "papi.h"
#include "papi_test.h"

// Aid in debugging Cuda contexts
#define STEP_BY_STEP_DEBUG 0

static void print_help_message(void)
{
    printf("./HelloWorld --device [nvidia device index] --cuda-native-event-names [list of cuda native event names separated by a comma].\n"
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

// Device kernel
__global__ void helloWorld(char* str)
{
        // determine where in the thread grid we are
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // unmangle output
        str[idx] += idx;
}

// Host function
int main(int argc, char** argv)
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

    int suppress_output = 0;
    char *user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    if (user_defined_suppress_output) {
        suppress_output = (int) strtol(user_defined_suppress_output, (char**) NULL, 10);
    }
    PRINT(suppress_output, "Running the cuda component test HelloWorld.cu\n");

    int cuda_device_index = -1;
    char **cuda_native_event_names = NULL;
    // If command line arguments are provided then get their values.
    int total_event_count = 0;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &cuda_device_index, &cuda_native_event_names, &total_event_count);
    }

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__,__LINE__, "PAPI_library_init()", papi_errno);
    }
    PRINT(suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION));

    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", cuda_cmp_idx);
    }
    PRINT(suppress_output, "The cuda component is assigned to component index: %d\n", cuda_cmp_idx);

    // If a user does not provide an event or events, then we go get an event to add
    if (total_event_count == 0) {
        enumerate_and_store_cuda_native_events(&cuda_native_event_names, &total_event_count, &cuda_device_index);
    }

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset( &EventSet ) );

    // If multiple GPUs/contexts were being used, you'd need to
    // create contexts for each device. See, for example,
    // simpleMultiGPU.cu.
    CUcontext sessionCtx = NULL;
    int flags = 0;
    CUdevice device = cuda_device_index;
#if defined(CUDA_TOOLKIT_GE_13)
    check_cuda_driver_api_call( cuCtxCreate(&sessionCtx, (CUctxCreateParams*)0, flags, device) );
#else
    check_cuda_driver_api_call( cuCtxCreate(&sessionCtx, flags, device) );
#endif

    CUcontext getCtx;
    if (STEP_BY_STEP_DEBUG) {
        check_cuda_driver_api_call( cuCtxGetCurrent(&getCtx) );
        fprintf(stderr, "Address of Cuda context after call to cuCtxCreate is %p\n", getCtx);
    }

    int num_events_successfully_added = 0, numMultipassEvents = 0;
    char **events_successfully_added = (char **) malloc(total_event_count * sizeof(char *));
    check_memory_allocation_call( events_successfully_added );

    int event_idx;
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        events_successfully_added[event_idx] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        check_memory_allocation_call( events_successfully_added[event_idx] );

        add_cuda_native_events(EventSet, cuda_native_event_names[event_idx], &num_events_successfully_added, events_successfully_added, &numMultipassEvents);
    }

    // Only multiple pass events were provided on the command line
    if (num_events_successfully_added == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        exit(EXIT_FAILURE);
    }

    if (STEP_BY_STEP_DEBUG) {
        check_cuda_driver_api_call( cuCtxGetCurrent(&getCtx) );
        fprintf(stderr, "Address of Cuda context after events have been added is %p\n", getCtx);
    }

    check_papi_api_call( PAPI_start(EventSet) );

    if (STEP_BY_STEP_DEBUG) {
        check_cuda_driver_api_call( cuCtxGetCurrent(&getCtx) );
        fprintf(stderr, "Address of Cuda context after call to PAPI_start is %p\n", getCtx);
    }

    // Mangle contents of output
    // The null character is left intact for simplicity
    char str[] = "Hello World!";
    int i;
    for (i = 0; i < strlen(str); i++) {
        str[i] -= i;
    }
    PRINT(suppress_output, "mangled str=%s\n", str);

    // Allocate memory on the device
    char *d_str;
    size_t size = sizeof(str);
    check_cuda_runtime_api_call( cudaMalloc((void**)&d_str, size) );
    check_memory_allocation_call( d_str );

    // Copy the string to the device
    check_cuda_runtime_api_call( cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice) );

    // Set the grid and block sizes
    dim3 dimGrid(2); // One block per word
    dim3 dimBlock(6); // One thread per character

    // Invoke the kernel
    helloWorld<<< dimGrid, dimBlock >>>(d_str);
    check_cuda_runtime_api_call( cudaGetLastError() );

    // Retrieve the results from the device
    check_cuda_runtime_api_call( cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost) );

    // free up the allocated memory on the device
    check_cuda_runtime_api_call( cudaFree(d_str) );


    long long *cuda_counter_values = (long long *) calloc(total_event_count, sizeof (long long));
    check_memory_allocation_call(cuda_counter_values);

    check_papi_api_call( PAPI_read(EventSet, cuda_counter_values) );
    for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++ ) {
        PRINT(suppress_output, "After PAPI_read, the event %s produced the value: \t\t%lld\n", events_successfully_added[event_idx], cuda_counter_values[event_idx]);
    }

    if (STEP_BY_STEP_DEBUG) {
        check_cuda_driver_api_call( cuCtxGetCurrent(&getCtx) );
        fprintf(stderr, "Address of Cuda context after call to PAPI_read is %p\n", getCtx);
    }

    check_papi_api_call( PAPI_stop(EventSet, cuda_counter_values) );
    for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++ ) {
        PRINT(suppress_output, "After PAPI_stop, the event %s produced the value: \t\t%lld\n", events_successfully_added[event_idx], cuda_counter_values[event_idx]);
    }

    if (STEP_BY_STEP_DEBUG) {
        check_cuda_driver_api_call( cuCtxGetCurrent(&getCtx) );
        fprintf(stderr, "Address of Cuda context after call to PAPI_stop is %p\n", getCtx);
    }

    check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    if (STEP_BY_STEP_DEBUG) {

        fprintf(stderr, "%s:%s:%i before cuCtxDestroy sessionCtx=%p.\n", __FILE__, __func__, __LINE__, sessionCtx);
    }

    // Destroy the context used for this test
    check_cuda_driver_api_call( cuCtxDestroy(sessionCtx) );

    // Free allocated memory
    free(cuda_counter_values);

    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        free(cuda_native_event_names[event_idx]);
    }
    free(cuda_native_event_names);

    for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++) {
        free(events_successfully_added[event_idx]);
    }
    free(events_successfully_added);

    PAPI_shutdown();

    // Output a note that a multiple pass event was provided on the command line
    if (numMultipassEvents > 0) {
        PRINT(suppress_output, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    test_pass(__FILE__);

    return 0;
}
