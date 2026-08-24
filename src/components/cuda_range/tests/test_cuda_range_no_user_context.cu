/**
* @file  test_cuda_range_no_user_context.cu
* @brief This test verifies that CUDA contexts are being created internally
*        if one is not created by the user on the calling CPU thread.
*/

// C++ STL headers.
#include <cstring>
#include <iostream>
#include <string>

// CTK headers.
#include <cuda.h>

// Internal headers.
#include "cuda_range_tests_helper.hpp"
#include <gpu_work.h>
#include "papi.h"
#include "papi_test.h"

#define KERNEL_QUIET 1

/** 
  * @brief A function wrapper to print out the help message for test_cuda_range_no_user_context.cu.
*/
static void print_help_message(void)
{
    std::cout << "./test_cuda_range_no_user_context --device [NVIDIA device index]" << std::endl
              << "Notes:" << std::endl
              << "1. Only a single NVIDIA device index will be used." << std::endl
              << "2. The NVIDIA device index will be used to append the device qualifier (i.e. :device=#)." << std::endl;

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
*/
static void parse_and_assign_args(int argc, char *argv[], int &device_index)
{
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
            i++;
        }
        else {
            print_help_message();
            exit(EXIT_FAILURE);
        }
    }

    return;
}

int main(int argc, char **argv)
{
    std::cout << "Running the cuda_range component test -- test_cuda_range_no_user_context.cu." << std::endl;
    CHECK_CUDA_DRIVER_API_CALL( cuInit(0) );

    int device_index = 0;
    // If a user provided command line arguments then parse them.
    if (argc > 1) {
        parse_and_assign_args(argc, argv, device_index);
    }

    // Determine the number of compute-capable devices.
    int number_of_devices_on_system = 0;
    CHECK_CUDA_RUNTIME_API_CALL( cudaGetDeviceCount(&number_of_devices_on_system) );
    // No compute-capable devices on the machine. Exiting.
    if (number_of_devices_on_system < 1) {
        std::cout << "No compute-capable devices found on the machine. This is required for the test to run." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Verify the device_index is valid.
    if (device_index < 0 || device_index > number_of_devices_on_system) {
        std::cout << "The device index is invalid. The number of NVIDIA devices on the system is "
                  << number_of_devices_on_system << "." << std::endl;
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

    // Set the internal environment variable to verify CUDA context creation.
    int status = setenv("PAPI_CUDA_RANGE_INTERNAL_VERIFY_CONTEXTS", "TRUE", 1);
    if (status != 0) {
        std::cout << "Failed to set PAPI_CUDA_RANGE_INTERNAL_VERIFY_CONTEXTS environment variable." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get a cuda_range native event name.
    std::string cuda_range_native_event_name {};
    get_cuda_range_native_event_name(cuda_range_cmp_index, cuda_range_native_event_name);
    cuda_range_native_event_name += ":device=" + std::to_string(device_index);

    // Create a PAPI event set.
    int event_set = PAPI_NULL;
    CHECK_PAPI_API_CALL( PAPI_create_eventset(&event_set) );

    // Add the cuda_range native event names to the event set.
    CHECK_PAPI_API_CALL( PAPI_add_named_event(event_set, cuda_range_native_event_name.c_str()) );

    // Formatting.
    std::cout << "Verifying internally created CUDA context:" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    // Verify a CUDA context was created.
    CUcontext context;
    CHECK_CUDA_DRIVER_API_CALL( cuCtxGetCurrent(&context) );
    // CUDA context creation success.
    if (context != nullptr) {
        std::cout << "- CUDA context was successfully created internally." << std::endl;
    }
    // CUDA context creation failed.
    else {
        std::cout << "- CUDA context was not successfully created internally. Exiting" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Verify a CUDA context was created for the correct device qualifier.
    CUdevice device;
    CHECK_CUDA_DRIVER_API_CALL( cuCtxGetDevice_v2(&device, context) );
    // Correct device qualifier.
    if (device == device_index) {
        std::cout << "- CUDA context was correctly created for the qualifier :device=" << device_index << "." << std::endl;
    }
    // Incorrect device qualifier.
    else {
        std::cout << "- CUDA context was not correctly created for the qualifier :device=" << device_index << ". Exiting." << std::endl;
        exit(EXIT_FAILURE); 
    }

    // Start profiling.
    CHECK_PAPI_API_CALL( PAPI_start(event_set) );

    // Launch kernel.
    int number_of_iterations = 50000; 
    VectorAddSubtract(number_of_iterations, KERNEL_QUIET);

    // Stop profiling.
    long long cuda_range_counter_value = 0;
    CHECK_PAPI_API_CALL( PAPI_stop(event_set, &cuda_range_counter_value) );

    // Print profiling data.
    std::cout << std::endl
              << "Profiling results:" << std::endl
              << "------------------------------------------" << std::endl
              << "- " << cuda_range_native_event_name << " -- " << cuda_range_counter_value << std::endl;

    // Cleanup the PAPI event set. 
    CHECK_PAPI_API_CALL( PAPI_cleanup_eventset(event_set) );

    // Destroy the PAPI event set.
    CHECK_PAPI_API_CALL( PAPI_destroy_eventset(&event_set) );

    // Shutdown the PAPI library. 
    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
