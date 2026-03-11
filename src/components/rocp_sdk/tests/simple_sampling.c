// Standard library headers
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Internal headers
#include <papi.h>
#include <papi_test.h>

int eventset = PAPI_NULL;
int total_event_count = 0;
char **rocp_sdk_native_event_names = NULL;

volatile int gv=0;

extern int launch_kernel(int device_id);
extern void enumerate_and_store_rocp_sdk_native_events(char ***rocp_sdk_native_event_names, int *total_event_count);
extern void add_rocp_sdk_native_events(int eventSet, int maxNativeEventsToAdd, char **nativeEventsToAdd);

static void print_help_message(void)
{
    printf("./simple_sampling --rocp-sdk-native-event-names [list of rocp_sdk native event names separated by a comma].\n");
}

static void parse_and_assign_args(int argc, char *argv[], char ***rocp_sdk_native_event_names, int *total_event_count)
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
        else if (strcmp(arg, "--rocp-sdk-native-event-names") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! --rocp-sdk-event-names given, but no events listed.\n");
                exit(EXIT_FAILURE);
            }

            char **cmd_line_native_event_names = NULL;
            const char *rocp_sdk_native_event_name = strtok(argv[i+1], ",");
            while (rocp_sdk_native_event_name != NULL)
            {
                cmd_line_native_event_names = (char **) realloc(cmd_line_native_event_names, ((*total_event_count) + 1) * sizeof(char *));
                if (cmd_line_native_event_names == NULL) {
                    fprintf(stderr, "%s:%d: Error: Memory allocation failed.\n", __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

                cmd_line_native_event_names[(*total_event_count)] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
                if (cmd_line_native_event_names[(*total_event_count)] == NULL) {
                    fprintf(stderr, "Failed to allocate memory for index %d in rocp_sdk_native_event_names.\n", (*total_event_count));
                    exit(EXIT_FAILURE);
                }

                int strLen = snprintf(cmd_line_native_event_names[(*total_event_count)], PAPI_MAX_STR_LEN, "%s", rocp_sdk_native_event_name);
                if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                    fprintf(stderr, "Failed to fully write rocp_sdk native event name.\n");
                    exit(EXIT_FAILURE);
                }

                (*total_event_count)++;
                rocp_sdk_native_event_name = strtok(NULL, ",");
            }
            *rocp_sdk_native_event_names = cmd_line_native_event_names;
            i++;
        }
        else
        {
            print_help_message();
            exit(EXIT_FAILURE);
        }
    }
}


void *thread_main(void *arg)
{
    long long *counters = (long long *) malloc(total_event_count * sizeof(long long));
    if (counters == NULL) {
        fprintf(stderr, "%s:%d: Error: Memory allocation failed.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    while( 0 == gv) {;}
    usleep(150*1000);

    int i;
    for( i = 0; i < 30; i++) {
        printf("Sample: %2d\n", gv);
        fflush(stdout);
        PAPI_read(eventset, counters);
        int j;
        for (j = 0; j < total_event_count; ++j) {
            printf("%s: %.2lfM\n", rocp_sdk_native_event_names[j], (double)counters[j]/1e6);
            fflush(stdout);
        }
        printf("\n");
	fflush(stdout);
        usleep(30*1000);
        ++gv;
    }
    free(counters);

    return NULL;
}


int main(int argc, char *argv[])
{
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &rocp_sdk_native_event_names, &total_event_count);
    }

    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

    // If a user does not provide an event or events, then we go get an event to add
    if (total_event_count == 0) {
        enumerate_and_store_rocp_sdk_native_events(&rocp_sdk_native_event_names, &total_event_count);
    }

    int eventset = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    // Add the native events via command line or enumeration to the EventSet
    add_rocp_sdk_native_events(eventset, total_event_count, rocp_sdk_native_event_names);

    papi_errno = PAPI_start(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

    pthread_t tid;
    pthread_create(&tid, NULL, thread_main, NULL);

    printf("---------------------  launch_kernel(0)\n");
    gv = 1;
    papi_errno = launch_kernel(0);
    if (papi_errno != 0) {
        test_fail(__FILE__, __LINE__, "launch_kernel(0)", papi_errno);
    }

    usleep(20000);

    long long *counters = (long long *) malloc(total_event_count * sizeof(long long));
    if (counters == NULL) {
        fprintf(stderr, "%s:%d: Error: Memory allocation failed.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    papi_errno = PAPI_read(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
    }
    printf("---------------------  PAPI_read()\n");

    int i;
    for (i = 0; i < total_event_count; ++i) {
        printf("%s: %.2lfM\n", rocp_sdk_native_event_names[i], (double)counters[i]/1e6);
    }

    papi_errno = PAPI_stop(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (i = 0; i < total_event_count; ++i) {
        printf("%s: %.2lfM\n", rocp_sdk_native_event_names[i], (double)counters[i]/1e6);
    }
    
    papi_errno = PAPI_cleanup_eventset(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    //Free memory allocation
    free(counters);
    for (i = 0; i < total_event_count; i++) {
        free(rocp_sdk_native_event_names[i]);
    }
    free(rocp_sdk_native_event_names);


    PAPI_shutdown();
    test_pass(__FILE__);
    return 0;
}
