/**
* @file pthreads_noCuCtx.cu
* @brief For each enabled NVIDIA device detected on the machine a matching thread will be created
*        using pthread_create. For each thread, cudaSetDevice will be called which determines which
*        device executions will be done on.
*
*        Note: The cuda component supports being partially disabled, meaning that certain devices
*        will not be "enabled" to profile on. If PAPI_CUDA_API is not set, then devices with
*        CC's >= 7.0 will be used and if PAPI_CUDA_API is set to LEGACY then devices with
*        CC's <= 7.0 will be used.
*
*        For each enabled device, their matching thread will have a workflow of:
*            1. Creating an EventSet
*            2. Adding events to the EventSet
*            3. Starting the EventSet
*            4. Stopping the EventSet
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

#define MAX_THREADS (32)

int global_suppress_output;
int global_total_event_count;
char **global_cuda_native_event_names = NULL;
int global_num_multipass_events = 0;

static void print_help_message(void)
{
    printf("./pthreads_noCuCtx --cuda-native-event-names [list of cuda native event names separated by a comma].\n"
           "Notes:\n"
           "1. Native event names must not have the device qualifier appended.\n");
}

static void parse_and_assign_args(int argc, char *argv[], char ***cuda_native_event_names, int *total_event_count)
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
                if (strstr(cuda_native_event_name, ":device")) {
                    fprintf(stderr, "Cuda native event name must not have a device qualifier appended for this test, i.e. no :device=#.\n");
                    print_help_message();
                    exit(EXIT_FAILURE);
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

void *thread_gpu(void *thread_and_dev_idx)
{
    int curr_thread_and_dev_idx = *(int *) thread_and_dev_idx;

    check_cuda_runtime_api_call( cudaSetDevice(curr_thread_and_dev_idx) );
    check_cuda_runtime_api_call( cudaFree(NULL) );

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    int num_events_successfully_added = 0;
    char **events_successfully_added = (char **) malloc(global_total_event_count * sizeof(char *));
    check_memory_allocation_call( events_successfully_added  );

    int event_idx;
    for (event_idx = 0; event_idx < global_total_event_count; event_idx++) {
        char tmp_event_name[PAPI_MAX_STR_LEN];
        int strLen = snprintf(tmp_event_name, PAPI_MAX_STR_LEN, "%s:device=%d", global_cuda_native_event_names[event_idx], curr_thread_and_dev_idx);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Failed to fully write event name with appended device qualifier.\n");
            exit(EXIT_FAILURE);
        }
        
        events_successfully_added[event_idx] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        check_memory_allocation_call(events_successfully_added[event_idx]);
        
        add_cuda_native_events(EventSet, tmp_event_name, &num_events_successfully_added, events_successfully_added, &global_num_multipass_events);
    }

    // Only multiple pass events were provided on the command line
    if (num_events_successfully_added == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        exit(EXIT_FAILURE);
    }

    check_papi_api_call( PAPI_start(EventSet) );

    // Work for the device
    VectorAddSubtract(50000 * (curr_thread_and_dev_idx + 1), global_suppress_output);

    long long cuda_counter_values[MAX_THREADS];
    check_papi_api_call( PAPI_stop(EventSet, cuda_counter_values) );

    for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++) {
        PRINT(global_suppress_output, "Event %s on thread and device id %d produced the value:\t\t%lld\n", events_successfully_added[event_idx], curr_thread_and_dev_idx, cuda_counter_values[event_idx]);
    }

    // Free allocated memory
    for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++) {
        free(events_successfully_added[event_idx]);
    }
    free(events_successfully_added);

    check_papi_api_call(PAPI_cleanup_eventset(EventSet));

    check_papi_api_call(PAPI_destroy_eventset(&EventSet));

    return NULL;
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

    global_suppress_output = 0;
    char *global_user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    if (global_user_defined_suppress_output) {
        global_suppress_output = (int) strtol(global_user_defined_suppress_output, (char**) NULL, 10);
    } 
    PRINT(global_suppress_output, "Running the cuda component test pthreads_noCuCtx.cu\n");

    // If command line arguments are provided then get their values.
    global_total_event_count = 0;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &global_cuda_native_event_names, &global_total_event_count);
    }

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init()", papi_errno);
    }
    PRINT(global_suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION));

    // Initialize thread support in PAPI
    check_papi_api_call(PAPI_thread_init((unsigned long (*)(void)) pthread_self));

    // Verify the cuda component is compiled in
    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", cuda_cmp_idx);
    }
    PRINT(global_suppress_output, "The cuda component is assigned to component index: %d\n", cuda_cmp_idx);

    // Initialize the Cuda component
    int cuda_eventcode = 0 | PAPI_NATIVE_MASK;
    check_papi_api_call( PAPI_enum_cmp_event(&cuda_eventcode, PAPI_ENUM_FIRST, cuda_cmp_idx) );

    // If we have not gotten an event via the command line, use the event obtained from PAPI_enum_cmp_event
    if (global_total_event_count == 0) {
        int num_spaces_to_allocate = 1;
        global_cuda_native_event_names = (char **) malloc(num_spaces_to_allocate * sizeof(char *));
        check_memory_allocation_call( global_cuda_native_event_names );

        global_cuda_native_event_names[global_total_event_count] = (char *) malloc(PAPI_2MAX_STR_LEN * sizeof(char));
        check_memory_allocation_call( global_cuda_native_event_names[global_total_event_count] );

        check_papi_api_call( PAPI_event_code_to_name(cuda_eventcode, global_cuda_native_event_names[global_total_event_count++]) );
    }

    const PAPI_component_info_t *cmpInfo = PAPI_get_component_info(cuda_cmp_idx);
    if (cmpInfo == NULL) {
        fprintf(stderr, "Call to PAPI_get_component_info failed.\n");
        exit(EXIT_FAILURE);
    }

    // Check to see if the Cuda component is partially disabled
    if (cmpInfo->partially_disabled) {
        const char *cc_support = (getenv("PAPI_CUDA_API") != NULL) ? "<=7.0" : ">=7.0";
        PRINT(global_suppress_output, "\033[33mThe cuda component is partially disabled. Only support for CC's %s are enabled.\033[0m\n", cc_support);
    }

    // Cap the number of devices to the max allowed number of threads
    if (num_devices > MAX_THREADS) {
        num_devices = MAX_THREADS;
    } 

    // Allocate memory for all the gpus found on the machine to keep track of threads and thread args
    pthread_t *tinfo = (pthread_t *) calloc(num_devices, sizeof(pthread_t));
    check_memory_allocation_call( tinfo );

    int *thread_args = (int *) calloc(num_devices, sizeof(int));
    check_memory_allocation_call( thread_args );

    PRINT(global_suppress_output, "Total number of threads to be launched: %d\n", num_devices);
    // For the number of devices detected on the machine, launch a thread
    int thread_and_dev_idx, thread_errno, global_num_multipass_events = 0;
    for(thread_and_dev_idx = 0; thread_and_dev_idx < num_devices; thread_and_dev_idx++)
    {
        const PAPI_component_info_t *cmpInfo = PAPI_get_component_info(cuda_cmp_idx);
        if (cmpInfo == NULL) {
            fprintf(stderr, "Call to PAPI_get_component_info failed.\n");
            exit(EXIT_FAILURE);
        }

        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(thread_and_dev_idx) == 0) {
                continue;
            }   
        } 

        // Store thread information to later use pthread_join
        tinfo[thread_and_dev_idx] = thread_and_dev_idx;
        // Store thread args so we do not increment the looping variable while in thread_gpu
        thread_args[thread_and_dev_idx] = thread_and_dev_idx;

        thread_errno = pthread_create(&tinfo[thread_and_dev_idx], NULL, thread_gpu, &thread_args[thread_and_dev_idx]);
        if(thread_errno != 0) {
            fprintf(stderr, "Call to pthread_create failed for thread %d with error code %d.\n", thread_and_dev_idx, thread_errno);
            exit(EXIT_FAILURE);
        }
    }

    // Now join each thread
    for (thread_and_dev_idx = 0; thread_and_dev_idx < num_devices; thread_and_dev_idx++) {
        const PAPI_component_info_t *cmpInfo = PAPI_get_component_info(cuda_cmp_idx);
        if (cmpInfo == NULL) {
            fprintf(stderr, "Call to PAPI_get_component_info failed.\n");
            exit(EXIT_FAILURE);
        } 

        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(thread_and_dev_idx) == 0) {
                continue;
            }   
        } 

        thread_errno = pthread_join(tinfo[thread_and_dev_idx], NULL);
        if (thread_errno != 0) {
             fprintf(stderr, "Call to pthread_join failed for thread %d with error code %d.\n", thread_and_dev_idx, thread_errno);
             exit(EXIT_FAILURE);
        }
    }

    // Output a note that a multiple pass event was provided on the command line
    if (global_num_multipass_events > 0) {
        PRINT(global_suppress_output, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    // Free allocated memory
    int event_idx;
    for (event_idx = 0; event_idx < global_total_event_count; event_idx++) {
        free(global_cuda_native_event_names[event_idx]);
    }
    free(global_cuda_native_event_names);
    free(tinfo);
    free(thread_args);

    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
