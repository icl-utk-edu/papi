/**
* @file  test_cuda_range_multiple_pass_events.cu.
* @brief This test verifies that multiple pass event support is functional.
*/

// C++ STL headers.
#include <cstring>
#include <iostream>
#include <map>
#include <string>

// CTK headers.
#include <cuda.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

// Internal headers.
#include "cuda_range_tests_helper.hpp"
#include "gpu_work.h"
#include "papi.h"
#include "papi_test.h"

#define KERNEL_QUIET 1

typedef struct native_event_data_t
{
    const char *cuda_range_native_event_name;
    size_t expected_number_of_passes_for_collection;
} native_event_data_t;

std::map<std::string, native_event_data_t> multiple_pass_events = {
    { "GA100", {"cuda_range:::sm__memory_throughput:stat=max:submetric=pct_of_peak_sustained_active", 4} },
    { "GH100", {"cuda_range:::sm__memory_throughput:stat=max:submetric=pct_of_peak_sustained_active", 5} },
    { "GB202", {"cuda_range:::sm__memory_throughput:stat=max:submetric=pct_of_peak_sustained_active", 3} },
}; 

/** 
  * @brief A function wrapper to print out the help message for test_cuda_range_multiple_pass_events.cu.
*/
static void print_help_message(void)
{
    std::cout << "./test_cuda_range_multiple_pass_events --device [NVIDIA device index]" << std::endl
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
    std::cout << "Running the cuda_range component test -- test_cuda_range_multiple_pass_events." << std::endl;
    CHECK_CUDA_DRIVER_API_CALL( cuInit(0) );

    CUpti_Profiler_Initialize_Params profiler_initialize_params {}; 
    profiler_initialize_params.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
    profiler_initialize_params.pPriv = nullptr;
    CHECK_CUPTI_API_CALL( cuptiProfilerInitialize(&profiler_initialize_params) );
   
    int device_index = 0; 
    // If a user provided command line arguments then print the help message.
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

    // Get the chip name for the current device index.
    CUpti_Device_GetChipName_Params get_chip_name_params {};
    get_chip_name_params.structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
    get_chip_name_params.pPriv = nullptr;
    get_chip_name_params.deviceIndex = device_index;
    CHECK_CUPTI_API_CALL( cuptiDeviceGetChipName(&get_chip_name_params) );
    std::cout << "Proceeding to test multiple pass event support for device " << device_index
              << " (" << get_chip_name_params.pChipName << ")" << "." << std::endl;

    // Create a PAPI event set.
    int event_set = PAPI_NULL;
    CHECK_PAPI_API_CALL( PAPI_create_eventset(&event_set) );

    // Add the cuda_range native event names to the event set.
    CHECK_PAPI_API_CALL( PAPI_add_named_event(event_set, multiple_pass_events[get_chip_name_params.pChipName].cuda_range_native_event_name) );

    // Start profiling.
    CHECK_PAPI_API_CALL( PAPI_start(event_set) );

    // Launch kernel.
    int number_of_iterations = 50000;
    VectorAddSubtract(number_of_iterations, KERNEL_QUIET);

    size_t actual_number_of_passes_for_collection = 0;
    long long cuda_range_counter_value = 0;
    do {
        papi_errno = PAPI_read(event_set, &cuda_range_counter_value);
        VectorAddSubtract(number_of_iterations, KERNEL_QUIET);
        actual_number_of_passes_for_collection++;
    } while(papi_errno == PAPI_EALLPASSES_NOT_SUBMITTED);

    std::cout << "Profiling results:" << std::endl;
    std::cout << "---------------------" << std::endl;
    if (actual_number_of_passes_for_collection == multiple_pass_events[get_chip_name_params.pChipName].expected_number_of_passes_for_collection) {
        std::cout << "-> The actual number of passes for collection (" << actual_number_of_passes_for_collection << ")"
                  << " equals the expected number of passes (" << multiple_pass_events[get_chip_name_params.pChipName].expected_number_of_passes_for_collection
                  << ")." << std::endl;

        std::cout << "-> " << multiple_pass_events[get_chip_name_params.pChipName].cuda_range_native_event_name << " -- " << cuda_range_counter_value << std::endl;
    }
    else {
        std::cout << "-> The actual number of passes for collection (" << actual_number_of_passes_for_collection << ")"
                  << " does not equal the expected number of passes (" << multiple_pass_events[get_chip_name_params.pChipName].expected_number_of_passes_for_collection
                  << ")." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Stop profiling.
    CHECK_PAPI_API_CALL( PAPI_stop(event_set, NULL) );

    // Cleanup the PAPI event set. 
    CHECK_PAPI_API_CALL( PAPI_cleanup_eventset(event_set) );

    // Destroy the PAPI event set.
    CHECK_PAPI_API_CALL( PAPI_destroy_eventset(&event_set) );

    // Shutdown the PAPI library. 
    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
