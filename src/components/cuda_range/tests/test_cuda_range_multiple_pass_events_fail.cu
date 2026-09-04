/**
* @file  test_cuda_range_multiple_pass_events_fail.cu.
* @brief This test verifies multiple pass events fail unless
*        a user has exported PAPI_CUDA_RANGE_ENABLE_MULTIPASSES.
*/

// C++ STL headers.
#include <iostream>

// Internal headers.
#include <cuda_range_tests_helper.hpp>
#include "papi.h"
#include "papi_test.h"

/** 
  * @brief A function wrapper to print out the help message for test_cuda_range_multiple_pass_events_fail.cu.
*/
static void print_help_message(void)
{
    std::cout << "./test_cuda_range_multiple_pass_events_fail" << std::endl
              << "Notes:" << std::endl
              << "1. No command line arguments are available for this test as stands." << std::endl;

    return;
}

int main(int argc, char **argv)
{
    std::cout << "Running the cuda_range component test -- test_cuda_range_multiple_pass_events_fail." << std::endl;
    CHECK_CUDA_DRIVER_API_CALL( cuInit(0) );

    // If a user provided command line arguments then print the help message.
    if (argc > 1) {
        print_help_message();
        exit(EXIT_FAILURE);
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

    // Initialize the cuda_range component/Get the first cuda_range event.
    int modifier = PAPI_ENUM_FIRST;
    int cuda_range_eventcode = 0 | PAPI_NATIVE_MASK;
    CHECK_PAPI_API_CALL( PAPI_enum_cmp_event(&cuda_range_eventcode, modifier, cuda_range_cmp_index) );

    // Search for a cuda_range native event that requires multiple passes for collection.
    PAPI_event_info_t event_info;
    modifier = PAPI_ENUM_EVENTS;
    do {
        CHECK_PAPI_API_CALL( PAPI_get_event_info(cuda_range_eventcode, &event_info) );
    } while( event_info.num_passes_for_collection == 1 &&
             PAPI_enum_cmp_event(&cuda_range_eventcode, modifier, cuda_range_cmp_index) == PAPI_OK);

    // Create a PAPI event set. 
    int event_set = PAPI_NULL;
    CHECK_PAPI_API_CALL( PAPI_create_eventset(&event_set) );

    // Attempt to add the multiple pass event. 
    papi_errno = PAPI_add_named_event(event_set, event_info.symbol);
    // PAPI_EMULPASS correctly returned.
    if (papi_errno == PAPI_EMULPASS) {
        std::cout << "-> " << event_info.symbol << " requires " << event_info.num_passes_for_collection
                  << " passes for collection -- PAPI_EMULPASS correctly returned" << std::endl;
    }
    // PAPI_EMULPASS not correctly returned.
    else {
        std::cout << "PAPI_EMULPASS not returned and should have been" << std::endl;
        exit(EXIT_FAILURE);
    }

    CHECK_PAPI_API_CALL( PAPI_destroy_eventset(&event_set) );

    test_pass(__FILE__);

    return 0;
}
