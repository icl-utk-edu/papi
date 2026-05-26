/**
* @file HelloWorld.cu
* @brief This test serves as a very simple hello world c example where the string
*        "Hello World!" is mangled and then restored. cudaSetDevice is used for context
*        creation.
*/


// Standard library headers
#include <stdio.h>

// Cuda Toolkit headers
#include <cuda_runtime.h>

// Internal headers
#include "papi.h"
#include "papi_test.h"
#include "nvml_tests_helper.h"

static void print_help_message(void)
{
	printf("./HelloWorld --device [nvidia device index] --nvml-native-event-names [list of nvml native event names separated by a comma].\n"
		"Notes:\n"
		"1. Both args (--device and --nvml-native-event-names) must be provided.\n"
		"2. The # in the nvml native event name's device_# must match the device index provided to --device.\n");
}

static void parse_and_assign_args(int argc, char *argv[], int *device_index_arg, char ***nvml_native_event_names_arg, int *total_event_count_arg)
{
    int num_device_indices = 0, *event_device_indices = NULL;
    int i, device_arg_found = 0, nvml_native_event_name_arg_found = 0;
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
                printf("ERROR!! Add a nvidia device index. Exiting.\n");
                exit(EXIT_FAILURE);
            }
            *device_index_arg = atoi(argv[i + 1]);
            device_arg_found++;
            i++;
        }
        else if (strcmp(arg, "--nvml-native-event-names") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! --nvml-native-event-names given, but no events listed. Exiting.\n");
                exit(EXIT_FAILURE);
            }

            char **cmd_line_native_event_names = NULL;
            char *nvml_native_event_name = strtok(argv[i+1], ",");
            while (nvml_native_event_name != NULL)
            {
            	// Get the device + state substring, for the nvml component it is always present
                const char *needle = ":device_";
            	char *device_and_state_substring = strstr(nvml_native_event_name, needle);
                if (device_and_state_substring == NULL) {
                    fprintf(stderr, "The substring (device_) is not present in the nvml native event name and should be. Exiting.\n");
                    exit(EXIT_FAILURE);
                }

                // Move past needle
                device_and_state_substring += strlen(needle);
                // Count the number of decimals after needle
                int c, num_decimals = 0;
                for (c = 0; device_and_state_substring[c] != '\0'; c++) {
                    // We have hit state 
                    if (device_and_state_substring[c] == ':') {
                        break;
                    }
       
                    num_decimals++; 
                }

                char device_index[PAPI_MAX_STR_LEN] = { 0 };
                int strLen = snprintf(device_index, sizeof(device_index), "%.*s", num_decimals, device_and_state_substring);
                if (strLen < 0 || (size_t) strLen >= sizeof(device_index)) {
                    fprintf(stderr, "Failed to fully write decimals into buffer.\n");
                    exit(EXIT_FAILURE);
                }

            	event_device_indices = (int *) realloc(event_device_indices, (num_device_indices + 1) *  sizeof(int));
            	check_memory_allocation_call(event_device_indices);
            	event_device_indices[num_device_indices++] = atoi(device_index);

                cmd_line_native_event_names = (char **) realloc(cmd_line_native_event_names, ((*total_event_count_arg) + 1) * sizeof(char *));
                check_memory_allocation_call(cmd_line_native_event_names);

                cmd_line_native_event_names[(*total_event_count_arg)] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
                check_memory_allocation_call(cmd_line_native_event_names[(*total_event_count_arg)]);

                strLen = snprintf(cmd_line_native_event_names[(*total_event_count_arg)], PAPI_MAX_STR_LEN, "%s", nvml_native_event_name);
                if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                    fprintf(stderr, "Failed to fully write nvml native event name into buffer.\n");
                    exit(EXIT_FAILURE);
                }

                (*total_event_count_arg)++;
                nvml_native_event_name_arg_found++;
                nvml_native_event_name = strtok(NULL, ",");
            }
            i++;
            *nvml_native_event_names_arg = cmd_line_native_event_names;
        }
        else
        {
            print_help_message();
            exit(EXIT_FAILURE);
        }
    }

    if (device_arg_found == 0 || nvml_native_event_name_arg_found == 0) {
        fprintf(stderr, "You must use both the --device arg and --nvml-native-event-names arg in conjunction.\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < num_device_indices; i++) {
        if ((*device_index_arg) != event_device_indices[i]) {
            fprintf(stderr, "The device_# (%d) does not match the index (%d) provided by --device.\n", event_device_indices[i], *device_index_arg);
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

int main(int argc, char **argv)
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
    PRINT(suppress_output, "Running the nvml component test HelloWorld.cu\n");
    
    int nvidia_device_index = -1;
    char **nvml_native_event_names = NULL;
    // If command line arguments are provided then get their values.
    int total_event_count = 0;
    if (argc > 1) {
    	parse_and_assign_args(argc, argv, &nvidia_device_index, &nvml_native_event_names, &total_event_count);
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
    
    int nvml_cmp_idx = PAPI_get_component_index("nvml");
    if (nvml_cmp_idx < 0) {
    	test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", nvml_cmp_idx);
    }
    PRINT(suppress_output, "The nvml component is assigned to component index: %d\n", nvml_cmp_idx);

    // If a user does not provide an event or events, then we go get an event to add
    if (total_event_count == 0) {
        enumerate_and_store_nvml_native_events(&nvml_native_event_names, &total_event_count, &nvidia_device_index);
    }
    
    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset( &EventSet ) );

    int event_idx;
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        check_papi_api_call( PAPI_add_named_event(EventSet, nvml_native_event_names[event_idx]) );
    }
 
    check_cuda_runtime_api_call( cudaSetDevice(nvidia_device_index) );
    
    check_papi_api_call( PAPI_start(EventSet) );
    
    char str[] = "Hello World!";
    PRINT(suppress_output, "\033[0;33m\nStarting string:\n\033[0m%s\n\n", str);
    
    // Mangle contents of output
    // The null character is left intact for simplicity
    PRINT(suppress_output, "\033[0;33mProceeding to mangle the starting string.\n\033[0m");
    int i;
    for(i = 0; i < strlen(str); i++) {
    	str[i] -= i;
    }
    PRINT(suppress_output, "The mangled string is: %s\n\n", str);
    
    // Allocate memory on the device
    char *d_str;
    size_t size = sizeof(str);
    check_cuda_runtime_api_call( cudaMalloc((void**)&d_str, size) );
    
    // Copy the string to the device
    check_cuda_runtime_api_call( cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice) );
    
    // Set the grid and block sizes
    dim3 dimGrid(2); // One block per word
    dim3 dimBlock(6); // One thread per character

    PRINT(suppress_output, "\033[0;33mProceeding to unmangle the mangled string.\n\033[0m");
    // Invoke the kernel
    helloWorld<<< dimGrid, dimBlock >>>(d_str);
    check_cuda_runtime_api_call( cudaGetLastError() );
    
    // Retrieve the results from the device
    check_cuda_runtime_api_call( cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost) );
    
    printf("The unmangled string is: %s\n\n", str);
    
    long long *counter_values = (long long *) malloc(total_event_count * sizeof(long long));
    check_memory_allocation_call(counter_values);
    check_papi_api_call( PAPI_stop(EventSet, counter_values) );

    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        PRINT(suppress_output, "After PAPI_stop, the event %s produced the value \t\t%lld\n",
              nvml_native_event_names[event_idx], counter_values[event_idx]);
    }

    // Free allocated memory
    check_cuda_runtime_api_call( cudaFree(d_str) );

    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        free(nvml_native_event_names[event_idx]);
    }
    free(nvml_native_event_names);

    return 0;
}

