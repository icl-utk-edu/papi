/**
* @file test_multipass_event_fail.cu
* @brief Test to see if a cuda native event requires multiple passes to profile.
*        If it does PAPI_EMULPASS will be returned.
*
*        Note: The cuda component supports being partially disabled, meaning that certain devices
*        will not be "enabled" to profile on. If PAPI_CUDA_API is not set, then devices with
*        CC's >= 7.0 will be used and if PAPI_CUDA_API is set to LEGACY then devices with
*        CC's <= 7.0 will be used.
*/

// Standard library headers
#include <stdio.h>
#include <stdlib.h>

// Internal headers
#include "cuda_tests_helper.h"
#include "papi.h"
#include "papi_test.h"

static void print_help_message(void)
{
    printf("./test_multipass_event_fail --cuda-native-event-names [list of cuda native event names separated by a comma].\n"
           "Notes:\n"
           "1. This test is designed to see if a cuda native event requires multiple passes.\n");
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
                cmd_line_native_event_names = (char **) realloc(cmd_line_native_event_names, ((*total_event_count) + 1) * sizeof(char *));
                check_memory_allocation_call(cmd_line_native_event_names);

                cmd_line_native_event_names[(*total_event_count)] = (char *) malloc(PAPI_2MAX_STR_LEN * sizeof(char));
                check_memory_allocation_call(cmd_line_native_event_names[(*total_event_count)]);

                int strLen = snprintf(cmd_line_native_event_names[(*total_event_count)], PAPI_2MAX_STR_LEN, "%s", cuda_native_event_name);
                if (strLen < 0 || strLen >= PAPI_2MAX_STR_LEN) {
                    fprintf(stderr, "Failed to fully write event name %s.\n", cuda_native_event_name);
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
    PRINT(suppress_output, "Running the cuda component test test_multipass_event_fail.cu\n");

    char **cuda_native_event_names = NULL;
    // See if a metric was passed on the command line
    int total_event_count = 0;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &cuda_native_event_names, &total_event_count);
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

    // Verify the cuda component has been compiled in
    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0 ) { 
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", cuda_cmp_idx);
    }
    PRINT(suppress_output, "The cuda component is assigned to component index: %d\n", cuda_cmp_idx);

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    // An event has been added on the command line.
    int event_idx;
    if (total_event_count > 0) {
        for (event_idx = 0; event_idx < total_event_count; event_idx++) {
            // First check if the cuda native event even requires multiple passes
            papi_errno = PAPI_add_named_event(EventSet, cuda_native_event_names[event_idx]);
            if (papi_errno != PAPI_OK && papi_errno != PAPI_EMULPASS) {
                test_fail(__FILE__, __LINE__, "PAPI_add_named_event()", papi_errno);
            }
            else if (papi_errno == PAPI_EMULPASS) {
                PRINT(suppress_output, "%s requires multiple passes.\n", cuda_native_event_names[event_idx]);
            }
            else {
                PRINT(suppress_output, "%s does not require multiple passes.\n", cuda_native_event_names[event_idx]);

                check_papi_api_call( PAPI_remove_named_event(EventSet, cuda_native_event_names[event_idx]) );
            }
        }
    }
    // No event has been added on the command line.
    else {
        int modifier = PAPI_ENUM_FIRST;
        int cuda_eventcode = 0 | PAPI_NATIVE_MASK;
        check_papi_api_call( PAPI_enum_cmp_event(&cuda_eventcode, modifier, cuda_cmp_idx); );

        int multipass_event_found = 0;
        modifier = PAPI_ENUM_EVENTS;
        do {
           char cuda_eventname[PAPI_2MAX_STR_LEN];
           check_papi_api_call( PAPI_event_code_to_name(cuda_eventcode, cuda_eventname) );

           papi_errno = PAPI_add_named_event(EventSet, cuda_eventname);
           if (papi_errno != PAPI_OK && papi_errno != PAPI_EMULPASS) {
               test_fail(__FILE__, __LINE__, "PAPI_add_named_event()", papi_errno);
           }
           else if (papi_errno == PAPI_EMULPASS) {
               multipass_event_found++;
               PRINT(suppress_output, "%s requires multiple passes.\n", cuda_eventname);
           }
           else {
               check_papi_api_call( PAPI_remove_named_event(EventSet, cuda_eventname) );
           }

        } while (PAPI_enum_cmp_event(&cuda_eventcode, modifier, cuda_cmp_idx) == PAPI_OK && multipass_event_found  == 0);

        if (multipass_event_found == 0) {
            PRINT(suppress_output, "\033[33mNo multipass event found for this architecture. Verify that this indeed holds true.\033[0m\n");
        }
    }

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    // Free allocated memory
    if (cuda_native_event_names != NULL) {
        for (event_idx = 0; event_idx < total_event_count; event_idx++) {
            free(cuda_native_event_names[event_idx]);
        }
        free(cuda_native_event_names);
    }

    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
