/**
* @file HelloWorld_noCuCtx.cu
* @brief This test serves as a very simple hello world c example where the string
*        "Hello World!" is mangled and then restored.
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

static void print_help_message(void)
{
    printf("./HelloWorld_noCuCtx --cuda-native-event-names [list of cuda native event names separated by a comma].\n"
           "Notes:\n"
           "1. A device qualifier must be provided otherwise a context will not be created.\n");
}

static void parse_and_assign_args(int argc, char *argv[], int *device_index, char ***cuda_native_event_names, int *total_event_count)
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



// Device kernel
__global__ void
helloWorld(char* str)
{
        // determine where in the thread grid we are
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // unmangle output
        str[idx] += idx;
}

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
    PRINT(suppress_output, "Running the cuda component test HelloWorld_noCuCtx.cu\n");
    
    int cuda_device_index = -1;
    char **cuda_native_event_names = NULL;
    // If command line arguments are provided then get their values.
    int total_event_count = 0;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &cuda_device_index, &cuda_native_event_names, &total_event_count);
    }

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if(papi_errno != PAPI_VER_CURRENT) {
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
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    // Handle the events from the command line
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
        test_skip(__FILE__, __LINE__, "", 0);
    }

    check_papi_api_call( PAPI_start(EventSet) );

    // mangle contents of output
    // the null character is left intact for simplicity
    char str[] = "Hello World!"; // Destired Output
    int i;
    for (i = 0; i < strlen(str); i++) {
        str[i] -= i;
    }
    PRINT(suppress_output, "mangled str=%s\n", str);

    // allocate memory on the device
    char *d_str;
    size_t size = sizeof(str);
    check_cuda_runtime_api_call( cudaMalloc((void**)&d_str, size) );
    check_memory_allocation_call( d_str );

    // Copy the string to the device
    check_cuda_runtime_api_call( cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice) );

    // Set the grid and block sizes
    dim3 dimGrid(2); // One block per word
    dim3 dimBlock(6); // One thread per character

    // invoke the kernel
    helloWorld<<< dimGrid, dimBlock >>>(d_str);
    check_cuda_runtime_api_call( cudaGetLastError() );

    // Retrieve the results from the device
    check_cuda_runtime_api_call( cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost) );

    // Free up the allocated memory on the device
    check_cuda_runtime_api_call( cudaFree(d_str) );

    long long *cuda_counter_values = (long long *) calloc(total_event_count, sizeof (long long));
    check_memory_allocation_call( cuda_counter_values );

    check_papi_api_call( PAPI_read(EventSet, cuda_counter_values) );

    for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++ ) {
        PRINT(suppress_output, "After PAPI_read, the event %s produced the value: \t\t%lld\n", events_successfully_added[event_idx], cuda_counter_values[event_idx]);
    }

    check_papi_api_call( PAPI_stop(EventSet, cuda_counter_values) );

    for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++) {
        PRINT(suppress_output, "After PAPI_stop, the event %s produced the value: \t\t%lld\n", events_successfully_added[event_idx], cuda_counter_values[event_idx]);
    } 

    check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    // Output a note that a multiple pass event was provided on the command line
    if (numMultipassEvents > 0) {
        PRINT(suppress_output, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    // Free allocated memory
    free(cuda_counter_values);
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        free(cuda_native_event_names[event_idx]);
        free(events_successfully_added[event_idx]);
    }
    free(cuda_native_event_names);
    free(events_successfully_added);

    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
