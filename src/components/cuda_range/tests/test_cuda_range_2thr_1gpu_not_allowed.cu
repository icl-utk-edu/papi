/**
* @file  test_cuda_range_2thr_1gpu_not_allowed.cu
* @brief Verify that we do not allow multiple threads on a single device. PAPI_ECNFLCT
*        should be returned if this occurs.
*/

// C++ STL headers.
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// POSIX standard headers.
#include <pthread.h>

// CTK headers.
#include <cuda.h>

// Internal headers.
#include "cuda_range_tests_helper.hpp"
#include "gpu_work.h"
#include "papi.h"
#include "papi_test.h"

#define KERNEL_QUIET 1
#define MAXIMUM_NUMBER_OF_THREADS 2

/** 
  * @brief An enum containing the available precisions.
*/
typedef struct pthread_params_t
{
    CUcontext context;
    int thread_num;
    char *cuda_range_native_event_name;
    pthread_t thread_id;
    int papi_errno;
} pthread_params_t;

/** 
  * @brief The pthread_create start routine.
  *   
  * @param *arg
  *   An instance of the structure pthread_params_t.
*/
void *thread_start(void *arg)
{
    pthread_params_t *tinfo = (pthread_params_t *) arg;

    CHECK_CUDA_DRIVER_API_CALL( cuCtxSetCurrent(tinfo->context) );

    // Create a PAPI event set.
    int event_set = PAPI_NULL;
    CHECK_PAPI_API_CALL( PAPI_create_eventset(&event_set) );

    // Add cuda_range native event names to the event set.
    tinfo->papi_errno = PAPI_add_named_event(event_set, tinfo->cuda_range_native_event_name);
    if (tinfo->papi_errno != PAPI_OK) {
        // PAPI_ECNFLCT correctly returned.
        if (tinfo->papi_errno == PAPI_ECNFLCT) {
            std::cout << "Thread " << tinfo->thread_num << ": Not allowed to profile." << std::endl;
            return NULL;
        }
        // PAPI_ENCFLCT not correctly returned.
        else {
            std::cout << "Thread " << tinfo->thread_num << ": Not allowed to profile. However, PAPI_ECNFLCT not returned." << std::endl;
            return NULL;
        }
    }

    // Start profiling.
    CHECK_PAPI_API_CALL( PAPI_start(event_set) );

    // Launch kernel.
    int number_of_iterations = 50000;
    VectorAddSubtract(number_of_iterations * (tinfo->thread_num + 1), KERNEL_QUIET);
    CHECK_CUDA_RUNTIME_API_CALL( cudaGetLastError() ); 

    // Stop profiling.
    long long cuda_range_counter_value = 0;
    CHECK_PAPI_API_CALL( PAPI_stop(event_set, &cuda_range_counter_value) );

    // Print profiling data.
    std::cout << "Thread " << tinfo->thread_num << ": " << tinfo->cuda_range_native_event_name
              << " produced the counter value -- " << cuda_range_counter_value << "." << std::endl;

    // Cleanup the PAPI event set.
    CHECK_PAPI_API_CALL( PAPI_cleanup_eventset(event_set) );

    // Destroy the PAPI event set.
    CHECK_PAPI_API_CALL( PAPI_destroy_eventset(&event_set) );

    return NULL;
}

/** 
  * @brief A function wrapper to print out the help message for test_cuda_range_2thr_1gpu_not_allowed.cu.
*/
static void print_help_message(void)
{
    std::cout << "./test_cuda_range_2thr_1gpu_not_allowed" << std::endl
              << "Notes:" << std::endl
              << "1. No command line arguments are available for this test as stands." << std::endl;

    return;
}

int main(int argc, char **argv)
{
    std::cout << "Running the cuda_range component test -- test_cuda_range_2thr_1gpu_not_allowed.cu." << std::endl;
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

    // Initialize thread support in the PAPI library.
    CHECK_PAPI_API_CALL( PAPI_thread_init((unsigned long (*)(void)) pthread_self) );

    // Confirm the cuda_range component exists e.g. compiled in.
    int cuda_range_cmp_index = PAPI_get_component_index("cuda_range");
    if (cuda_range_cmp_index < 0) {
        std::cout << "The cuda_range component does not exist. This is often due to cuda_range not being passed to --with-components at configure." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<char *> cuda_range_native_event_names {};
    cuda_range_native_event_names.reserve(MAXIMUM_NUMBER_OF_THREADS);
    // Get the event code for the first cuda_range native event.
    int modifier = PAPI_ENUM_FIRST;
    int cuda_range_eventcode = 0 | PAPI_NATIVE_MASK;
    CHECK_PAPI_API_CALL( PAPI_enum_cmp_event(&cuda_range_eventcode, modifier, cuda_range_cmp_index) );

    // Convert the event code to a cuda_range native event name.
    char first_cuda_range_native_event_name[PAPI_2MAX_STR_LEN] = "";
    CHECK_PAPI_API_CALL( PAPI_event_code_to_name(cuda_range_eventcode, first_cuda_range_native_event_name) );
    cuda_range_native_event_names.push_back(first_cuda_range_native_event_name);
    
    // Get the position of the device qualifier in the first enumerated cuda_range native event name and
    // subsequently get the device id.
    const char *qualifier = ":device=";
    const char *first_position = std::strstr(first_cuda_range_native_event_name, qualifier);
    if (first_position == nullptr) {
        std::cout << "The first enumerated cuda_range native event name lacks a device qualifier. Exiting" << std::endl;
        exit(EXIT_FAILURE); 
    }
    int device_index = std::stoi( first_position + std::strlen(qualifier) );  
    
    // Get the next cuda_range native event codes.
    char second_cuda_range_native_event_name[PAPI_2MAX_STR_LEN] = ""; 
    modifier = PAPI_ENUM_EVENTS;
    while (PAPI_enum_cmp_event(&cuda_range_eventcode, modifier, cuda_range_cmp_index) == PAPI_OK) {
        CHECK_PAPI_API_CALL( PAPI_event_code_to_name(cuda_range_eventcode, second_cuda_range_native_event_name) );
        const char *second_position = std::strstr(second_cuda_range_native_event_name, qualifier);
        if (second_position == nullptr) {
            std::cout << "The second enumerated cuda_range native event name lacks a device qualifier. Exiting" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (std::stoi( second_position + std::strlen(qualifier) ) == device_index) {
            cuda_range_native_event_names.push_back(second_cuda_range_native_event_name);
            break;
        }
    }

    // Verify we indeed have only 2 cuda_range native event names.
    if (cuda_range_native_event_names.size() != 2) {
        std::cout << "Two native event names are needed and we have " << cuda_range_native_event_names.size() << ". Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Total number of threads to be launched: " << MAXIMUM_NUMBER_OF_THREADS << "." << std::endl;
    pthread_params_t tinfo[MAXIMUM_NUMBER_OF_THREADS];
    // Create threads.
    for(size_t tnum = 0; tnum < MAXIMUM_NUMBER_OF_THREADS; tnum++) {
        CUcontext context;
        unsigned int flags = 0;
        CUdevice device = device_index;
        CHECK_CUDA_DRIVER_API_CALL( cuCtxCreate(&context, (CUctxCreateParams*)0, flags, device) );
        CHECK_CUDA_DRIVER_API_CALL( cuCtxPopCurrent(&context) );

        tinfo[tnum].context = context;
        tinfo[tnum].thread_num = tnum;
        tinfo[tnum].cuda_range_native_event_name = cuda_range_native_event_names[tnum];

        int status = pthread_create(&tinfo[tnum].thread_id, NULL, thread_start, &tinfo[tnum]);
        if(status != 0) {
            std::cout << "Call to pthread_create failed for thread " << tnum << " with error code " << status << "." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    int papi_ecnflct_returned = 0;
    // Join with each thread.
    for (size_t tnum = 0; tnum < MAXIMUM_NUMBER_OF_THREADS; tnum++) {
        int status = pthread_join(tinfo[tnum].thread_id, NULL);
        if (status != 0) {
            std::cout << "Call to pthread_join failed for thread " << tnum << " with error code " << status << "." << std::endl;
            exit(EXIT_FAILURE);
        }

        // Destroy the CUDA context.
        CHECK_CUDA_DRIVER_API_CALL( cuCtxDestroy(tinfo[tnum].context) );

        // See if thread returned PAPI_ECNFLCT.
        if (tinfo[tnum].papi_errno == PAPI_ECNFLCT) {
            papi_ecnflct_returned = 1; 
        }
    }

    // Shutdown the PAPI library.
    PAPI_shutdown();

    // Test succeeded -- PAPI_ECNFLCT was correctly returned.
    if (papi_ecnflct_returned) {
        test_pass(__FILE__);
    }
    // Test failed -- PAPI_ECNFLCT was not returned.
    else {
        test_fail(__FILE__, __LINE__, "PAPI_ECNFLCT was not returned and should have been.", 0);
    }

    return 0;
}
