/**
* @file test_2thr_1gpu_not_allowed.cu
* @brief Verify that we do not allow multiple threads on a single device. PAPI_ECNFLCT
*        should be returned if this occurs.
*
*        Note: The cuda component supports being partially disabled, meaning that certain devices
*        will not be "enabled" to profile on. If PAPI_CUDA_API is not set, then devices with
*        CC's >= 7.0 will be used and if PAPI_CUDA_API is set to LEGACY then devices with
*        CC's <= 7.0 will be used.
*/

// Standard library headers
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Cuda Toolkit headers
#include <cuda.h>

// Internal headers
#include "cuda_tests_helper.h"
#include "gpu_work.h"
#include "papi.h"
#include "papi_test.h"

#define NUM_THREADS 2

int global_suppress_output;
int global_num_devices;
int global_total_event_count;
char **global_cuda_native_event_names = NULL;

typedef struct pthread_params_s {
    pthread_t tid;
    CUcontext cuCtx;
    int idx;
    int retval;
} pthread_params_t;

static void print_help_message(void)
{
    printf("./test_2thr_1gpu_not_allowed --device [nvidia device index] --cuda-native-event-names [list of cuda native event names separated by a comma].\n"
           "Notes:\n"
           "1. Must provide exactly two native event names on the command line with matching device qualifiers.\n");
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

void *thread_gpu(void * ptinfo)
{
    pthread_params_t *tinfo = (pthread_params_t *) ptinfo;
    int thread_idx = tinfo->idx;
    unsigned long gettid = (unsigned long) pthread_self();

    check_cuda_driver_api_call( cuCtxSetCurrent(tinfo->cuCtx) );

    CUdevice deviceId;
    check_cuda_driver_api_call( cuCtxGetDevice(&deviceId) );
    PRINT(global_suppress_output, "Attempting to run on thread %d (%lu) with device %d.\n", thread_idx, gettid, deviceId);

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    int papi_errno = PAPI_add_named_event(EventSet, global_cuda_native_event_names[thread_idx]);
    if (papi_errno != PAPI_OK) {
        if (papi_errno == PAPI_EMULPASS) {
            fprintf(stderr, "Event %s requires multiple passes and cannot be added to an EventSet. Two single pass events are needed for this test see utils/papi_native_avail for more Cuda native events.\n", global_cuda_native_event_names[thread_idx]);
            test_skip(__FILE__, __LINE__, "", 0);
        }
        else {
            fprintf(stderr, "Unable to add event %s to the EventSet with error code %d.\n", global_cuda_native_event_names[thread_idx], papi_errno);
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }

    papi_errno = PAPI_start(EventSet);
    if (papi_errno == PAPI_ECNFLCT) {
        PRINT(global_suppress_output, "\033[0;32mThread %d was not allowed to start profiling on the same GPU.\n\n\033[0m", thread_idx);
        tinfo->retval = papi_errno;
        return NULL;
    }

    VectorAddSubtract(5000000 * (thread_idx + 1), global_suppress_output);  // gpu work

    long long cuda_counter_value;
    check_papi_api_call( PAPI_stop(EventSet, &cuda_counter_value) );

    PRINT(global_suppress_output, "User measured values in thread id %d.\n", thread_idx);
    PRINT(global_suppress_output, "%s\t\t%lld\n\n", global_cuda_native_event_names[thread_idx], cuda_counter_value);
    tinfo->retval = PAPI_OK;

    check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    return NULL;
}

int main(int argc, char **argv)
{
    // Determine the number of Cuda capable devices
    global_num_devices = 0;
    check_cuda_runtime_api_call( cudaGetDeviceCount(&global_num_devices) );
    // No devices detected on the machine, exit
    if (global_num_devices < 1) {
        fprintf(stderr, "No NVIDIA devices found on the machine. This is required for the test to run.\n");
        exit(EXIT_FAILURE);
    }

    global_suppress_output = 0;
    char *user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    if (user_defined_suppress_output) {
        global_suppress_output = (int) strtol(user_defined_suppress_output, (char**) NULL, 10);
    }   
    PRINT(global_suppress_output, "Running the cuda component test test_2thr_1gpu_not_allowed.cu\n");

    int cuda_device_index = -1;
    // If command line arguments are provided then get their values.
    global_total_event_count = 0;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &cuda_device_index, &global_cuda_native_event_names, &global_total_event_count);
        if (global_total_event_count != 2) {
            fprintf(stderr, "Must provide two single pass Cuda native events on the command line for this test to run properoly.\n");
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init()", papi_errno);
    }
    PRINT(global_suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION));

    // Verify the cuda component has been compiled in
    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0 ) { 
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", cuda_cmp_idx);
    }
    PRINT(global_suppress_output, "The cuda component is assigned to component index: %d\n", cuda_cmp_idx);

    // No events were provided on the command line
    if (global_total_event_count == 0) {
        int num_spaces_to_allocate = 2;
        global_cuda_native_event_names = (char **) malloc(num_spaces_to_allocate * sizeof(char *));
        check_memory_allocation_call(global_cuda_native_event_names);

        int modifier = PAPI_ENUM_FIRST;
        int cuda_eventcode = 0 | PAPI_NATIVE_MASK;
        // Enumerate until we get two Cuda native events
        while (PAPI_enum_cmp_event(&cuda_eventcode, modifier, cuda_cmp_idx) == PAPI_OK && global_total_event_count < num_spaces_to_allocate) {
            global_cuda_native_event_names[global_total_event_count] = (char *) malloc(PAPI_2MAX_STR_LEN * sizeof(char));
            check_memory_allocation_call( global_cuda_native_event_names[global_total_event_count] );

            // Convert the first cuda native event code to a name, the name will
            // be in the format of cuda:::basename with no qualifiers appended.
            char basename[PAPI_MAX_STR_LEN];
            check_papi_api_call( PAPI_event_code_to_name(cuda_eventcode, basename) );

            // Begin reconstructing the Cuda native event name with qualifiers
            int strLen = snprintf(global_cuda_native_event_names[global_total_event_count], PAPI_2MAX_STR_LEN, "%s", basename);
            if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                fprintf(stderr, "Failed to fully write event name.");
                exit(EXIT_FAILURE);
            }

            // Enumerate through the available default qualifiers.
            // The Legacy API only has the device qualifiers
            // while the Perfworks Metrics API has a stat and device
            // qualifier.
            modifier = PAPI_NTV_ENUM_UMASKS;
            check_papi_api_call( PAPI_enum_cmp_event(&cuda_eventcode, modifier, cuda_cmp_idx) );

            do {
                PAPI_event_info_t info;
                papi_errno = PAPI_get_event_info(cuda_eventcode, &info);
                check_papi_api_call( PAPI_get_event_info(cuda_eventcode, &info) );

                char *qualifier = strstr(info.symbol + strlen("cuda:::"), ":");
                if (strncmp(qualifier, ":device=", 8) == 0) {
                    cuda_device_index = strtol(qualifier + strlen(":device="), NULL, 10);
                }   

                int strLen = snprintf(global_cuda_native_event_names[global_total_event_count] + strlen(global_cuda_native_event_names[global_total_event_count]), PAPI_2MAX_STR_LEN - strlen(global_cuda_native_event_names[global_total_event_count]), "%s", qualifier);
                if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN - strlen(global_cuda_native_event_names[global_total_event_count])) {
                    fprintf(stderr, "Unable to construct cuda native event name.\n");
                     exit(EXIT_FAILURE);
                }

            } while (PAPI_enum_cmp_event(&cuda_eventcode, modifier, cuda_cmp_idx) == PAPI_OK);

            global_total_event_count++;
            // Change modifier for the outer loop
            modifier = PAPI_ENUM_EVENTS;
        }

        // Safety net, this should never be triggered
        if (cuda_device_index == -1) {
            fprintf(stderr, "A device qualifier is needed to continue or a device index must be provided on the command line.\n");
            exit(EXIT_FAILURE);
        } 

    }

    // Initialize PAPI thread support
    check_papi_api_call( PAPI_thread_init((unsigned long (*)(void)) pthread_self) );
    
    // Launch the threads
    pthread_params_t data[NUM_THREADS];    
    int thread_idx, thread_errno;
    for(thread_idx = 0; thread_idx < NUM_THREADS; thread_idx++)
    {
        data[thread_idx].idx = thread_idx;

        int flags = 0;
        CUdevice device = cuda_device_index;
#if defined(CUDA_TOOLKIT_GE_13)
        check_cuda_driver_api_call( cuCtxCreate(&(data[thread_idx].cuCtx), (CUctxCreateParams*)0, flags, device) );
#else
        check_cuda_driver_api_call( cuCtxCreate(&(data[thread_idx].cuCtx), flags, device) );
#endif
        check_cuda_driver_api_call( cuCtxPopCurrent(&(data[thread_idx].cuCtx)) );

        thread_errno = pthread_create(&data[thread_idx].tid, NULL, thread_gpu, &(data[thread_idx]));
        if(thread_errno != 0) {
            fprintf(stderr, "Call to pthread_create failed for thread %d with error code %d.\n", thread_idx, thread_errno);
            exit(EXIT_FAILURE);
        }
    }

    // Join all threads when complete
    for (thread_idx = 0; thread_idx < NUM_THREADS; thread_idx++) {
        thread_errno = pthread_join(data[thread_idx].tid, NULL);
        if (thread_errno != 0) {
            fprintf(stderr, "Call to pthread_join failed for thread %d with error code %d.\n", thread_idx, thread_errno);
            exit(EXIT_FAILURE);
        }
        PRINT(global_suppress_output, "Thread %d (%lu) successfully joined main thread.\n", thread_idx, (unsigned long)data[thread_idx].tid);
    }

    // Destroy all CUDA contexts for all threads/GPUs
    for (thread_idx = 0; thread_idx < NUM_THREADS; thread_idx++) {
        check_cuda_driver_api_call( cuCtxDestroy(data[thread_idx].cuCtx) );
    }

    // Free allocated memory
    int event_idx;
    for (event_idx = 0; event_idx < global_total_event_count; event_idx++) {
        free(global_cuda_native_event_names[event_idx]);
    }
    free(global_cuda_native_event_names);

    PAPI_shutdown();

    // Verify that we returned PAPI_ECNFLCT
    int papi_ecnflct_found = 0;
    for (thread_idx = 0; thread_idx < NUM_THREADS; thread_idx++) {
        if (data[thread_idx].retval == PAPI_ECNFLCT) {
            papi_ecnflct_found = 1;
            break;
        }
    }

    if (papi_ecnflct_found) {
        test_pass(__FILE__);
    }
    else {
        test_fail(__FILE__, __LINE__, "PAPI_ECNFLCT was not returned and should have been.", 0);
    }

    return 0;
}
