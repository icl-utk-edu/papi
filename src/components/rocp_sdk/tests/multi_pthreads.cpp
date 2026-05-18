// Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> 
#include <pthread.h>

// Internal headers
#include "kernel.h"

#define ONT 2
#define NUMGEMMS 5

typedef struct {
  int id;
} workerdata_t;

int main_id = 0;
int total_event_count = 0;
char **rocp_sdk_native_event_names = NULL;
int status[ONT];
pthread_t tid[ONT];
workerdata_t wkrdata[ONT];
pthread_barrier_t barrier;
pthread_mutex_t exclusive;

extern "C" void enumerate_and_store_rocp_sdk_native_events(char ***rocp_sdk_native_event_names, int *total_event_count);
extern "C" void add_rocp_sdk_native_events(int eventSet, int maxNativeEventsToAdd, char **nativeEventsToAdd);

static void print_help_message(char *argv[]);
static void parse_and_assign_args(int argc, char *argv[], char ***rocp_sdk_native_event_names, int *total_event_count);
void *thread_work(void *data);

int main(int argc, char *argv[]) {

    int overall_status = 0;

    // Parse command-line arguments.
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &rocp_sdk_native_event_names, &total_event_count);
    }

    // PAPI front matter.
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if( retval != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init() failed", retval);
    }

    PAPI_CALL(PAPI_thread_init(pthread_self));

    // If a user does not provide events, add them by enumerating available ones.
    if( 0 == total_event_count ) {
        enumerate_and_store_rocp_sdk_native_events(&rocp_sdk_native_event_names, &total_event_count);
    }

    // Initialize barrier and lock.
    pthread_barrier_init(&barrier, NULL, ONT);
    pthread_mutex_init(&exclusive, NULL);

    // Set thread function's arguments.
    for(int thdIdx = 0; thdIdx < ONT; ++thdIdx) {
        wkrdata[thdIdx].id = thdIdx;
        status[thdIdx] = 0;
    }

    // Fork the threads.
    for(int thdIdx = 0; thdIdx < ONT; ++thdIdx) {
        pthread_create(&(tid[thdIdx]), NULL, thread_work, &(wkrdata[thdIdx]));
    }

    // Join the threads.
    for(int thdIdx = 0; thdIdx < ONT; ++thdIdx) {
        pthread_join(tid[thdIdx], NULL);
        if( status[thdIdx] != 0 ) {
            overall_status = -1;
        }
    }

    // Check if there was an error in any of the threads.
    if( 0 != overall_status ) {
        test_fail(__FILE__, __LINE__, "Encountered an error in one of the threads", PAPI_EMISC);
    }

    // Free dynamically allocated memory.
    for (int i = 0; i < total_event_count; i++) {
        free(rocp_sdk_native_event_names[i]);
    }
    free(rocp_sdk_native_event_names);

    // Destroy pthread barrier and attributes.
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&exclusive);

    PAPI_shutdown();
    test_pass(__FILE__);
    return 0;
}

static void print_help_message(char *argv[]) {
    fprintf(stdout, "%s --rocp-sdk-native-event-names [list of rocp_sdk native event names separated by a comma]\n", argv[0]);
}

static void parse_and_assign_args(int argc, char *argv[], char ***rocp_sdk_native_event_names, int *total_event_count) {
    int i;
    for (i = 1; i < argc; ++i)
    {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0)
        {
            print_help_message(argv);
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
                    fprintf(stdout, "%s:%d: Error: Memory allocation failed.\n", __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

                cmd_line_native_event_names[(*total_event_count)] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
                if (cmd_line_native_event_names[(*total_event_count)] == NULL) {
                    fprintf(stdout, "Failed to allocate memory for index %d in rocp_sdk_native_event_names.\n", (*total_event_count));
                    exit(EXIT_FAILURE);
                }

                int strLen = snprintf(cmd_line_native_event_names[(*total_event_count)], PAPI_MAX_STR_LEN, "%s", rocp_sdk_native_event_name);
                if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                    fprintf(stdout, "Failed to fully write rocp_sdk native event name.\n");
                    exit(EXIT_FAILURE);
                }

                (*total_event_count)++;
                rocp_sdk_native_event_name = strtok(NULL, ",");
            }
            *rocp_sdk_native_event_names = cmd_line_native_event_names;
            i++;
        }
        else if (strcmp(arg, "TESTS_QUIET") == 0) {
            // TESTS_QUIET comes from running run_tests.sh.
            // As stands, we do not silence any printf's in
            // this application code. Therefore we acknowledge the case
            // and continue.
            continue;
        }
        else
        {
            print_help_message(argv);
            exit(EXIT_FAILURE);
        }
    }
}

void *thread_work(void *data) {

    // Get thread's ID.
    workerdata_t *wkrdata = (workerdata_t*)data;
    int id = wkrdata->id;

    int EventSet = PAPI_NULL;
    long long *values = (long long*)malloc(total_event_count/ONT*sizeof(long long));
    if( NULL == values ) {
        status[id] = PAPI_ENOMEM;
    }

    // Create event set.
    int stat;
    pthread_barrier_wait(&barrier);
    pthread_mutex_lock(&exclusive);
    stat = PAPI_create_eventset(&EventSet);
    if( PAPI_OK != stat ) {
        fprintf(stdout, "Thread %d: Failed to create event set.\n", id);
        status[id] = stat;
    }
    pthread_mutex_unlock(&exclusive);
    pthread_barrier_wait(&barrier);

    // Add event to set.
    pthread_barrier_wait(&barrier);
    pthread_mutex_lock(&exclusive);
    for(int i = id; i < total_event_count; i+=ONT) {
        stat = PAPI_add_named_event(EventSet, rocp_sdk_native_event_names[i]);
        if( PAPI_OK != stat ) {
            fprintf(stdout, "Thread %d: Failed to add event %s to set.\n", id, rocp_sdk_native_event_names[i]);
            status[id] = stat;
        }
    }
    pthread_mutex_unlock(&exclusive);
    pthread_barrier_wait(&barrier);

    // HIP front matter.
    int N = 16;
    dim3 threads_per_block( 1, 1, 1 );
    dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(N) / threads_per_block.y ), 1 );
    size_t probSize = N*N*sizeof(double);

    double *hostA=NULL,  *hostB=NULL,  *hostC=NULL;
    double  *devA=NULL,   *devB=NULL,   *devC=NULL;

    // Allocate host arrays.
    HIP_CALL_THD(hipHostMalloc(&hostA, probSize, 0));
    HIP_CALL_THD(hipHostMalloc(&hostB, probSize, 0));
    HIP_CALL_THD(hipHostMalloc(&hostC, probSize, 0));

    // Allocate device arrays.
    HIP_CALL_THD(hipMalloc(&devA, probSize));
    HIP_CALL_THD(hipMalloc(&devB, probSize));
    HIP_CALL_THD(hipMalloc(&devC, probSize));

    // Initialize arrays.
    srandom(1);
    for( int i = 0; i < N; i++ ) {
        for( int j = 0; j < N; j++ ) {
            hostA[i*N + j] = ((double)random())/RAND_MAX + 1.1;
            hostB[i*N + j] = ((double)random())/RAND_MAX + 1.1;
            devC[i*N + j] = 0.0;
        }
    }

    // Data transfer from host to device.
    HIP_CALL_THD(hipMemcpy(devA, hostA, probSize, hipMemcpyHostToDevice));
    HIP_CALL_THD(hipMemcpy(devB, hostB, probSize, hipMemcpyHostToDevice));
    HIP_CALL_THD(hipMemcpy(devC, hostC, probSize, hipMemcpyHostToDevice));

    // Call PAPI_start().
    pthread_barrier_wait(&barrier);
    pthread_mutex_lock(&exclusive);
    stat = PAPI_start(EventSet);
    if( PAPI_OK != stat ) {
        fprintf(stdout, "Thread %d: Failed to start counting. [%d]\n", id, stat);
        status[id] = stat;
    }
    pthread_mutex_unlock(&exclusive);
    pthread_barrier_wait(&barrier);

    // Launch the GEMM five times.
    for(int g = 0; g < NUMGEMMS; ++g) {
        pthread_barrier_wait(&barrier);
        pthread_mutex_lock(&exclusive);
        if( main_id == id ) {
            fprintf(stdout, "[New GEMM Phase]\n");
        }
        pthread_mutex_unlock(&exclusive);
        pthread_barrier_wait(&barrier);

        pthread_mutex_lock(&exclusive);
        fprintf(stdout, "------ Thread %d ------  Launch GEMM\n", id);
        hipLaunchKernelGGL(gemm, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
        HIP_CALL_THD(hipGetLastError());
        HIP_CALL_THD(hipDeviceSynchronize());
        pthread_mutex_unlock(&exclusive);
        pthread_barrier_wait(&barrier);

        // Call PAPI_read() after the GEMM.
        pthread_barrier_wait(&barrier);
        pthread_mutex_lock(&exclusive);
        stat = PAPI_read(EventSet, values);
        fprintf(stdout, "------ Thread %d ------  PAPI_read()\n", id);
        if( PAPI_OK != stat ) {
            fprintf(stdout, "\t<Error: Failed to read! %d>\n", stat);
            status[id] = stat;
        } else {
            for(int i = id; i < total_event_count; i+=ONT) {
                fprintf(stdout, "\t%s : %lld\n", rocp_sdk_native_event_names[i], values[i]);
            }
        }
        pthread_mutex_unlock(&exclusive);
        pthread_barrier_wait(&barrier);
    }

    // Call PAPI_stop() after the GEMM.
    pthread_barrier_wait(&barrier);
    pthread_mutex_lock(&exclusive);
    stat = PAPI_stop(EventSet, values);
    if( PAPI_OK != stat ) {
        fprintf(stdout, "Thread %d: Failed to stop. [%d]\n", id, stat);
        status[id] = stat;
    }
    pthread_mutex_unlock(&exclusive);
    pthread_barrier_wait(&barrier);

    // HIP back matter.
    HIP_CALL_THD(hipMemcpy(hostC, devC, probSize, hipMemcpyDeviceToHost));
    HIP_CALL_THD(hipFree(devA));
    HIP_CALL_THD(hipFree(devB));
    HIP_CALL_THD(hipFree(devC));
    HIP_CALL_THD(hipFree(hostA));
    HIP_CALL_THD(hipFree(hostB));
    HIP_CALL_THD(hipFree(hostC));

    // PAPI back matter.
    pthread_mutex_lock(&exclusive);
    stat = PAPI_cleanup_eventset( EventSet );
    if( PAPI_OK != stat ) {
        fprintf(stdout, "Thread %d: Failed to cleanup. [%d]\n", id, stat);
        status[id] = stat;
    }
    stat = PAPI_destroy_eventset( &EventSet );
    if( PAPI_OK != stat ) {
        fprintf(stdout, "Thread %d: Failed to destroy. [%d]\n", id, stat);
        status[id] = stat;
    }
    pthread_mutex_unlock(&exclusive);

    // Free other dynamically allocated memory.
    free(values);

    return NULL;
}
