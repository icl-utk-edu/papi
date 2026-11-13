/**
* @file  test_cuda_range_hello_world.cu
* @brief This test serves as a very simple hello world example where the string
*        "Hello World!" is mangled and then restored. cuCtxCreate is used for context
*        creation.
*/

// C++ STL headers.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// CTK headers.
#include <cuda.h>

// Internal headers.
#include "cuda_range_tests_helper.hpp"
#include "papi.h"
#include "papi_test.h"

/** 
  * @brief Kernel to manipulate the passed in char *.
  *
  * @param str
  *   String to manipulate.
*/
__global__ void hello_world(char *str)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        str[idx] += idx;
}

/** 
  * @brief A function wrapper to print out the help message for test_cuda_range_hello_world.cu.
*/
static void print_help_message(void)
{
    std::cout << "./test_cuda_range_hello_world --device [NVIDIA device index] --cuda-range-native-event-names [list of cuda_range native event names separated by a comma]" << std::endl
              << "Notes:" << std::endl
              << "1. Both args (--device and --cuda-range-native-event-names) must be provided." << std::endl
              << "2. If the device qualifier is provided then it must match the device index provided to --device." << std::endl;

    return;
}

/** 
  * @brief Parse the command line arguments provided by the user.
  *
  * @param argc
  *   Number of user passed arguments on the command line.
  * @param *argv
  *   Argument vector.
  * @param &device_index
  *   Stores the device index passed by the user to --device.
  * @param &cuda_range_native_event_names
  *   Stores the cuda_range native event names passed by the user to --cuda-range-native-event-names.
*/
static void parse_and_assign_args(int argc, char *argv[], int &device_index, std::vector<std::string> &cuda_range_native_event_names)
{
    std::vector<int> device_qualifier_indices {};
    int device_arg_found = 0, cuda_range_native_event_name_arg_found = 0;
    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_help_message();
            exit(EXIT_SUCCESS);
        }
        else if(strcmp(arg, "--device") == 0) {
            if (!argv[i + 1]) {
                std::cout << "ERROR!! Add a NVIDIA device index." << std::endl;
                exit(EXIT_FAILURE);
            }
            device_index = std::stoi(argv[i + 1]);
            device_arg_found++;
            i++;
        }
        else if (strcmp(arg, "--cuda-range-native-event-names") == 0) {
            if (!argv[i + 1]) {
                std::cout << "ERROR!! --cuda-range-native-event-names given, but no events listed." << std::endl;
                exit(EXIT_FAILURE);
            }

            std::stringstream ss(argv[i + 1]);
            std::string name;
            while (std::getline(ss, name, ',')) {
                std::string qualifier = ":device=";
                size_t device_qualifier_position = name.find(qualifier);
                if (device_qualifier_position != std::string::npos) {
                    std::string device_index_str = name.substr(device_qualifier_position + qualifier.size(), 1);
                    device_qualifier_indices.push_back(std::stoi(device_index_str)); 
                }

                cuda_range_native_event_names.push_back(name);
            }
            cuda_range_native_event_name_arg_found++;
            i++;
        }
        else {
            print_help_message();
            exit(EXIT_FAILURE);
        }
    }

    if (device_arg_found == 0 || cuda_range_native_event_name_arg_found == 0) {
        std::cout << "You must use both the --device arg and --cuda-range-native-event-names arg in conjunction." << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int device_qualifier_index : device_qualifier_indices) {
        if (device_qualifier_index != device_index) {
            std::cout << "The device qualifier index " << device_qualifier_index << " does not match the index " << device_index << " provided to --device." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return;
}

int main(int argc, char **argv)
{
    std::cout << "Running the cuda_range component test -- test_cuda_range_hello_world.cu." << std::endl;
    CHECK_CUDA_DRIVER_API_CALL( cuInit(0) );

    int device_index = -1; 
    std::vector<std::string> cuda_range_native_event_names {}; 
    // If a user provided command line arguments then parse them.
    if (argc > 1) {
        parse_and_assign_args(argc, argv, device_index, cuda_range_native_event_names);
    }

    // Determine the number of compute-capable devices.
    int number_of_devices_on_system = 0;
    CHECK_CUDA_RUNTIME_API_CALL( cudaGetDeviceCount(&number_of_devices_on_system) );
    // No compute-capable devices on the machine. Exiting.
    if (number_of_devices_on_system < 1) {
        std::cout << "No compute-capable devices found on the machine. This is required for the test to run." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize the PAPI library.
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) { 
        test_fail(__FILE__,__LINE__, "PAPI_library_init", papi_errno);
    }   
    std::cout << "The PAPI version being used for this test is: "
              << PAPI_VERSION_MAJOR(PAPI_VERSION) << "." 
              << PAPI_VERSION_MINOR(PAPI_VERSION) << "." 
              << PAPI_VERSION_REVISION(PAPI_VERSION) << "." 
              << std::endl;

    // Confirm the cuda_range component exists e.g. compiled in.
    int cuda_range_cmp_index = PAPI_get_component_index("cuda_range");
    if (cuda_range_cmp_index < 0) {
        std::cout << "The cuda_range component does not exist. This is often due to cuda_range not being passed to --with-components at configure." << std::endl;
        exit(EXIT_FAILURE);
    }

    // The user did not provide the command line argument --cuda-range-native-event-names.
    if (cuda_range_native_event_names.size() == 0) {
        get_cuda_range_native_event_name(cuda_range_cmp_index, device_index, cuda_range_native_event_names);
    }

    // Verify the device_index has been updated before proceeding.
    if (device_index == -1) {
        std::cout << "The device index is still -1; therefore, there is a bug in the internal code or test code." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create a CUDA context to be used for the entire application code.
    // Note: If multiple devices/contexts were being used, you'd need to create CUDA contexts for each device.
    CUcontext context = nullptr;
    unsigned int flags = 0;
    CUdevice device = device_index;
    CHECK_CUDA_DRIVER_API_CALL( cuCtxCreate(&context, (CUctxCreateParams*)0, flags, device) );

    // Create a PAPI event set.
    int event_set = PAPI_NULL;
    CHECK_PAPI_API_CALL( PAPI_create_eventset(&event_set) );

    // Add cuda_range native event names to the event set.
    for (std::string name : cuda_range_native_event_names) {
        CHECK_PAPI_API_CALL( PAPI_add_named_event(event_set, name.c_str()) );
    }

    // Start profiling.
    CHECK_PAPI_API_CALL( PAPI_start(event_set) );

    // Mangle the "Hello World!" string (the null character is left intact for simplicity).
    char str[] = "Hello World!";
    std::cout << "Proceeding to mangle the string: " << "'" << str << "'" << "." << std::endl;
    for (size_t i = 0; i < strlen(str); i++) {
        str[i] -= i;
    }
    std::cout << "Completed mangling the string: " << str << "." << std::endl;

    // Allocate memory on the device.
    char *d_str;
    size_t size = sizeof(str);
    CHECK_CUDA_RUNTIME_API_CALL( cudaMalloc((void**)&d_str, size) );
    CHECK_MEMORY_ALLOCATION_CALL( d_str );

    // Copy the string to the device.
    CHECK_CUDA_RUNTIME_API_CALL( cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice) );

    // Set the grid and block sizes.
    dim3 dimGrid(2); // One block per word.
    dim3 dimBlock(6); // One thread per character.

    // Launch kernel.
    hello_world<<< dimGrid, dimBlock >>>(d_str);
    CHECK_CUDA_RUNTIME_API_CALL( cudaGetLastError() );

    // Retrieve the results from the device.
    CHECK_CUDA_RUNTIME_API_CALL( cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost) );
    std::cout << "The string has been unmangled: " << "'" << str << "'" << "." << std::endl;

    // Read profiling data.
    std::vector<long long> cuda_range_counter_values(cuda_range_native_event_names.size());
    CHECK_PAPI_API_CALL( PAPI_read(event_set, cuda_range_counter_values.data()) );
    // Print profiling data.
    for (size_t i = 0; i < cuda_range_native_event_names.size(); i++) {
        std::cout << "After PAPI_read, the event " << cuda_range_native_event_names[i] << " produced the value: " << cuda_range_counter_values[i] << std::endl;
    }

    // Stop profiling.
    CHECK_PAPI_API_CALL( PAPI_stop(event_set, cuda_range_counter_values.data()) );
    // Print profiling data.
    for (size_t i = 0; i < cuda_range_native_event_names.size(); i++) {
        std::cout << "After PAPI_stop, the event " << cuda_range_native_event_names[i] << " produced the value: " << cuda_range_counter_values[i] << std::endl;
    }

    // Cleanup the PAPI event set. 
    CHECK_PAPI_API_CALL( PAPI_cleanup_eventset(event_set) );

    // Destroy the PAPI event set.
    CHECK_PAPI_API_CALL( PAPI_destroy_eventset(&event_set) );

    // Free the allocated memory on the device.
    CHECK_CUDA_RUNTIME_API_CALL( cudaFree(d_str) );

    // Destroy the CUDA context.
    CHECK_CUDA_DRIVER_API_CALL( cuCtxDestroy(context) );

    // Shutdown the PAPI library.
    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
