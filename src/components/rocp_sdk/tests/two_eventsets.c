// Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Internal headers
#include <papi.h>
#include <papi_test.h>

extern int launch_kernel(int device_id);
extern void enumerate_and_store_rocp_sdk_native_events(char ***rocp_sdk_native_event_names, int *total_event_count);
extern void add_rocp_sdk_native_events(int eventSet, int maxNativeEventsToAdd, char **nativeEventsToAdd);

static void print_help_message(void)
{
    printf("./two_eventsets --first-eventset-native-eventnames [list of rocp_sdk native event names separated by a comma] --second-eventset-native-eventnames [list of rocp_sdk native event names separated by a comma].\n");
}

static void parse_and_assign_args(int argc, char *argv[], char ***rocp_sdk_native_event_names_eventset1, int *total_event_count_eventset1, char ***rocp_sdk_native_event_names_eventset2, int *total_event_count_eventset2)
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
        else if (strcmp(arg, "--first-eventset-native-eventnames") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! --first-eventset-native-eventnames given, but no events listed.\n");
                exit(EXIT_FAILURE);
            }

            char **cmd_line_native_event_names_eventset1 = NULL;
            const char *rocp_sdk_native_event_name = strtok(argv[i+1], ",");
            while (rocp_sdk_native_event_name != NULL)
            {
                cmd_line_native_event_names_eventset1 = (char **) realloc(cmd_line_native_event_names_eventset1, ((*total_event_count_eventset1) + 1) * sizeof(char *));
                if (cmd_line_native_event_names_eventset1 == NULL) {
                    fprintf(stderr, "%s:%d: Error: Memory allocation failed.\n", __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

                cmd_line_native_event_names_eventset1[(*total_event_count_eventset1)] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
                if (cmd_line_native_event_names_eventset1[(*total_event_count_eventset1)] == NULL) {
                    fprintf(stderr, "Failed to allocate memory for index %d in rocp_sdk_native_event_names.\n", (*total_event_count_eventset1));
                    exit(EXIT_FAILURE);
                }

                int strLen = snprintf(cmd_line_native_event_names_eventset1[(*total_event_count_eventset1)], PAPI_MAX_STR_LEN, "%s", rocp_sdk_native_event_name);
                if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                    fprintf(stderr, "Failed to fully write rocp_sdk native event name.\n");
                    exit(EXIT_FAILURE);
                }

                (*total_event_count_eventset1)++;
                rocp_sdk_native_event_name = strtok(NULL, ",");
            }
            *rocp_sdk_native_event_names_eventset1 = cmd_line_native_event_names_eventset1;
            i++;
        }
        else if (strcmp(arg, "--second-eventset-native-eventnames") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! --second-eventset-native-eventnames given, but no events listed.\n");
                exit(EXIT_FAILURE);
            }

            char **cmd_line_native_event_names_eventset2 = NULL;
            const char *rocp_sdk_native_event_name = strtok(argv[i+1], ",");
            while (rocp_sdk_native_event_name != NULL)
            {
                cmd_line_native_event_names_eventset2 = (char **) realloc(cmd_line_native_event_names_eventset2, ((*total_event_count_eventset2) + 1) * sizeof(char *));
                if (cmd_line_native_event_names_eventset2 == NULL) {
                    fprintf(stderr, "%s:%d: Error: Memory allocation failed.\n", __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

                cmd_line_native_event_names_eventset2[(*total_event_count_eventset2)] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
                if (cmd_line_native_event_names_eventset2[(*total_event_count_eventset2)] == NULL) {
                    fprintf(stderr, "Failed to allocate memory for index %d in rocp_sdk_native_event_names.\n", (*total_event_count_eventset2));
                    exit(EXIT_FAILURE);
                }

                int strLen = snprintf(cmd_line_native_event_names_eventset2[(*total_event_count_eventset2)], PAPI_MAX_STR_LEN, "%s", rocp_sdk_native_event_name);
                if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                    fprintf(stderr, "Failed to fully write rocp_sdk native event name.\n");
                    exit(EXIT_FAILURE);
                }

                (*total_event_count_eventset2)++;
                rocp_sdk_native_event_name = strtok(NULL, ",");
            }
            *rocp_sdk_native_event_names_eventset2 = cmd_line_native_event_names_eventset2;
            i++;
        }
        else
        {
            print_help_message();
            exit(EXIT_FAILURE);
        }
    }
}

#define NUM_EVENTS (5)

int main(int argc, char *argv[])
{
    int total_event_count_eventset1 = 0, total_event_count_eventset2 = 0;
    char **rocp_sdk_native_event_names_eventset1 = NULL, **rocp_sdk_native_event_names_eventset2 = NULL;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &rocp_sdk_native_event_names_eventset1, &total_event_count_eventset1, &rocp_sdk_native_event_names_eventset2, &total_event_count_eventset2);
    }

    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

    /* ---------- Setup for eventset1 ---------- */
    int eventset1 = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    // If a user does not provide an event or events, then we go get an event to add
    if (total_event_count_eventset1 == 0) {
        enumerate_and_store_rocp_sdk_native_events(&rocp_sdk_native_event_names_eventset1, &total_event_count_eventset1);
    }

    // Add the native events via command line or enumeration to evnetset1
    add_rocp_sdk_native_events(eventset1, total_event_count_eventset1, rocp_sdk_native_event_names_eventset1);

    /* ---------- Setup for eventset2 ---------- */
    int eventset2 = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    // If a user does not provide an event or events, then we go get an event to add
    if (total_event_count_eventset2 == 0) {
        enumerate_and_store_rocp_sdk_native_events(&rocp_sdk_native_event_names_eventset2, &total_event_count_eventset2);
    }

    // Add the native events via command line or enumeration to eventset2
    add_rocp_sdk_native_events(eventset2, total_event_count_eventset2, rocp_sdk_native_event_names_eventset2);

    printf("==================== FIRST EVENTSET - DEVICE 1 ====================\n");

    papi_errno = PAPI_start(eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

    long long *counters1 = (long long *) malloc(total_event_count_eventset1 * sizeof(long long));
    if (counters1 == NULL) {
        fprintf(stderr, "%s:%d: Error: Memory allocation failed.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    int rep, i;
    for(rep = 0; rep <= 3; ++rep){

        papi_errno = launch_kernel(1);
        if (papi_errno != 0) {
            test_fail(__FILE__, __LINE__, "launch_kernel(1)", papi_errno);
        }

        usleep(1000);

        papi_errno = PAPI_read(eventset1, counters1);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (i = 0; i < total_event_count_eventset1; ++i) {
            printf("%s: %lld (%.2lf)\n", rocp_sdk_native_event_names_eventset1[i], counters1[i], 1.0*counters1[i]/(1.0+rep));
        }
    }

    papi_errno = PAPI_stop(eventset1, counters1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (i = 0; i < total_event_count_eventset1; ++i) {
        printf("%s: %lld (%.2lf)\n", rocp_sdk_native_event_names_eventset1[i], counters1[i], 1.0*counters1[i]/(1.0+3));
    }

    printf("==================== SECOND EVENTSET - DEVICE 1 ====================\n");


    papi_errno = PAPI_start(eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

    long long *counters2 = (long long *) malloc(total_event_count_eventset2 * sizeof(long long));
    if (counters2 == NULL) {
        fprintf(stderr, "%s:%d: Error: Memory allocation failed.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    for(rep = 0; rep <= 3; ++rep){

        papi_errno = launch_kernel(1);
        if (papi_errno != 0) {
            test_fail(__FILE__, __LINE__, "launch_kernel(1)", papi_errno);
        }

        usleep(1000);

        papi_errno = PAPI_read(eventset2, counters2);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (i = 0; i < total_event_count_eventset2; ++i) {
            printf("%s: %lld (%.2lf)\n", rocp_sdk_native_event_names_eventset2[i], counters2[i], 1.0*counters2[i]/(1.0+rep));
        }
    }

    papi_errno = PAPI_stop(eventset2, counters2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (i = 0; i < total_event_count_eventset2; ++i) {
        printf("%s: %lld (%.2lf)\n", rocp_sdk_native_event_names_eventset2[i], counters2[i], 1.0*counters2[i]/(1.0+3));
    }

    printf("==================== SECOND EVENTSET - DEVICE 0 ====================\n");

    papi_errno = PAPI_start(eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

    for(rep = 0; rep <= 2; ++rep){

        papi_errno = launch_kernel(0);
        if (papi_errno != 0) {
            test_fail(__FILE__, __LINE__, "launch_kernel(0)", papi_errno);
        }

        usleep(1000);

        papi_errno = PAPI_read(eventset2, counters2);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
        }
        printf("---------------------  PAPI_read()\n");

        for (i = 0; i < total_event_count_eventset2; ++i) {
            printf("%s: %lld (%.2lf)\n", rocp_sdk_native_event_names_eventset2[i], counters2[i], 1.0*counters2[i]/(1.0+rep));
        }
    }

    papi_errno = PAPI_stop(eventset2, counters2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (i = 0; i < total_event_count_eventset2; ++i) {
        printf("%s: %lld (%.2lf)\n", rocp_sdk_native_event_names_eventset2[i], counters2[i], 1.0*counters2[i]/(1.0+2));
    }

    /* * * Cleanup * * */

    papi_errno = PAPI_cleanup_eventset(eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&eventset1);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    papi_errno = PAPI_cleanup_eventset(eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&eventset2);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    // Free memory allocation
    free(counters1);
    free(counters2);

    for (i = 0; i < total_event_count_eventset1; i++) {
        free(rocp_sdk_native_event_names_eventset1[i]);
    }
    free(rocp_sdk_native_event_names_eventset1);

    for (i = 0; i < total_event_count_eventset2; i++) {
        free(rocp_sdk_native_event_names_eventset2[i]);
    }
    free(rocp_sdk_native_event_names_eventset2);

    PAPI_shutdown();
    test_pass(__FILE__);
    return 0;
}
