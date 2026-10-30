/**
* @file  test_cuda_range_pthreads.cu
* @brief For each enabled NVIDIA device detected on the machine a matching thread will be created
*        using pthread_create. For each thread, cuCtxCreate will be called which will
*        create a Cuda context.
*
*        For each enabled device, their matching thread will have a workflow of:
*        1. Creating a PAPI event set.
*        2. Adding events to the PAPI event set.
*        3. Starting the PAPI event set.
*        4. Stopping the PAPI event set.
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

/** 
  * @brief An enum containing member variables for pthreads.
*/
typedef struct pthread_params_t
{
    int thread_num;
    std::vector<std::string> cuda_range_native_event_names;
    char *cuda_range_native_event_name;
    pthread_t thread_id;
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

    // Create a CUDA context for the current thread.
    CUcontext context;
    unsigned int flags = 0;
    CUdevice device = tinfo->thread_num;
    CHECK_CUDA_DRIVER_API_CALL( cuCtxCreate(&context, (CUctxCreateParams*)0, flags, device) );

    // Create a PAPI event set.
    int event_set = PAPI_NULL;
    CHECK_PAPI_API_CALL( PAPI_create_eventset(&event_set) );

    // Add cuda_range native event names to the event set.
    for (std::string name : tinfo->cuda_range_native_event_names) {
        std::string cuda_range_native_event_name_device = name + ":device=" + std::to_string(tinfo->thread_num);
        CHECK_PAPI_API_CALL( PAPI_add_named_event(event_set, cuda_range_native_event_name_device.c_str()) );
    }

    // Start profiling.
    CHECK_PAPI_API_CALL( PAPI_start(event_set) );

    // Launch kernel.
    int number_of_iterations = 50000;
    VectorAddSubtract(number_of_iterations * (tinfo->thread_num + 1), KERNEL_QUIET);

    // Stop profiling.
    std::vector<long long> cuda_range_counter_values(tinfo->cuda_range_native_event_names.size());
    CHECK_PAPI_API_CALL( PAPI_stop(event_set, cuda_range_counter_values.data()) );

    // Print profiling data.
    for (size_t i = 0; i < tinfo->cuda_range_native_event_names.size(); i++) {
        std::cout << "Thread " << tinfo->thread_num << ": " << tinfo->cuda_range_native_event_names[i]
                  << " produced the counter value -- " << cuda_range_counter_values[i] << "." << std::endl;
    }

    // Cleanup the PAPI event set.
    CHECK_PAPI_API_CALL( PAPI_cleanup_eventset(event_set) );

    // Destroy the PAPI event set.
    CHECK_PAPI_API_CALL( PAPI_destroy_eventset(&event_set) );

    // Destroy the CUDA context created on the thread.
    CHECK_CUDA_DRIVER_API_CALL( cuCtxDestroy(context) );

    return NULL;
}

/** 
  * @brief A function wrapper to print out the help message for test_cuda_range_pthreads.cu.
*/
static void print_help_message(void)
{
    std::cout << "./test_cuda_range_pthreads --cuda-range-native-event-names [list of cuda_range native event names separated by a comma]." << std::endl
              << "Notes:" << std::endl
              << "1. The cuda_range native event names must not have the device qualifier appended (i.e. no :device=#)." << std::endl;

    return;
}

/** 
  * @brief Parse the command line arguments provided by the user.
  *
  * @param argc
  *   Number of user passed arguments on the command line.
  * @param *argv
  *   Argument vector.
  * @param &cuda_range_native_event_names
  *   Stores the cuda_range native event names passed by the user to --cuda-range-native-event-names.
*/
static void parse_and_assign_args(int argc, char *argv[], std::vector<std::string> &cuda_range_native_event_names)
{
    for (int i = 1; i < argc; ++i) {   
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {   
            print_help_message();
            exit(EXIT_SUCCESS);
        }   
        else if (strcmp(arg, "--cuda-range-native-event-names") == 0) {   
            if (!argv[i + 1]) {   
                printf("ERROR!! --cuda-range-native-event-names given, but no events listed.\n");
                exit(EXIT_FAILURE);
            }   

            const char *cuda_range_native_event_name = strtok(argv[i + 1], ",");
            while (cuda_range_native_event_name != NULL) {   
                if (strstr(cuda_range_native_event_name, ":device")) {
                    std::cout << "The cuda_range native event name " << cuda_range_native_event_name << " has a device qualifier appended. This is not allowed." << std::endl;
                    exit(EXIT_FAILURE);
                }

                cuda_range_native_event_names.push_back(cuda_range_native_event_name); 

                cuda_range_native_event_name = strtok(NULL, ",");
            }
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
    std::cout << "Running the cuda_range component test -- test_cuda_range_pthreads.cu." << std::endl;
    CHECK_CUDA_DRIVER_API_CALL( cuInit(0) );

    // If a user provided command line arguments then print the help message.
    std::vector<std::string> cuda_range_native_event_names {};
    if (argc > 1) {
        parse_and_assign_args(argc, argv, cuda_range_native_event_names);
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

    // The user did not provide the command line argument --cuda-range-native-event-names.
    if (cuda_range_native_event_names.size() == 0) {
        cuda_range_native_event_names.resize(1);
        get_cuda_range_native_event_name(cuda_range_cmp_index, cuda_range_native_event_names[0]);
    } 

    pthread_params_t *tinfo = (pthread_params_t *) calloc(number_of_devices_on_system, sizeof(pthread_params_t));
    CHECK_MEMORY_ALLOCATION_CALL(tinfo);

    std::cout << "Total number of threads to be launched: " << number_of_devices_on_system << "." << std::endl;
    // Create threads.
    for(size_t tnum = 0; tnum < number_of_devices_on_system; tnum++) {
        tinfo[tnum].thread_num = tnum;
        tinfo[tnum].cuda_range_native_event_names = cuda_range_native_event_names;

        int status = pthread_create(&tinfo[tnum].thread_id, NULL, thread_start, &tinfo[tnum]);
        if(status != 0) {
            std::cout << "Call to pthread_create failed for thread " << tnum << " with error code " << status << "." << std::endl;
            exit(EXIT_FAILURE);
        }   
    }

    // Join with each thread.
    for (size_t tnum = 0; tnum < number_of_devices_on_system; tnum++) {
        int status = pthread_join(tinfo[tnum].thread_id, NULL);
        if (status != 0) {
            std::cout << "Call to pthread_join failed for thread " << tnum << " with error code " << status << "." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    free(tinfo);

    // Shutdown the PAPI library.
    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
