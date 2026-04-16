// Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> 

// ROCm headers
#include <rocprofiler-sdk-roctx/roctx.h>

// Internal headers
#include "kernel.h"

int pauseResume = 0;

extern "C" void enumerate_and_store_rocp_sdk_native_events(char ***rocp_sdk_native_event_names, int *total_event_count);
extern "C" void add_rocp_sdk_native_events(int eventSet, int maxNativeEventsToAdd, char **nativeEventsToAdd);

static void print_help_message(char *argv[])
{
    fprintf(stdout, "%s --rocp-sdk-native-event-names [list of rocp_sdk native event names separated by a comma]\n" \
                       "--pause                       [enable ROCTX pause and resume calls (optional)]\n", argv[0]);
}

static void parse_and_assign_args(int argc, char *argv[], char ***rocp_sdk_native_event_names, int *total_event_count)
{
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
        else if (strcmp(arg, "--pause") == 0) {
            pauseResume = 1;
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

void copy_vals(long long *out, long long *in, int size);

int main(int argc, char *argv[]) {

    // Parse command-line arguments.
    int total_event_count = 0;
    char **rocp_sdk_native_event_names = NULL;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &rocp_sdk_native_event_names, &total_event_count);
    }

    // PAPI front matter.
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if( retval != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init() failed.", retval);
    }

    // If a user does not provide events, add them by enumerating available ones.
    if (total_event_count == 0) {
        enumerate_and_store_rocp_sdk_native_events(&rocp_sdk_native_event_names, &total_event_count);
    }

    int EventSet = PAPI_NULL;
    long long *values = (long long*)malloc(total_event_count*sizeof(long long));
    long long *prevValues = (long long*)malloc(total_event_count*sizeof(long long));
    if( NULL == values || NULL == prevValues ) {
        test_fail(__FILE__, __LINE__, "Failed to allocate memory for counter values.", PAPI_ENOMEM);
    }
    PAPI_CALL(PAPI_create_eventset(&EventSet));
    add_rocp_sdk_native_events(EventSet, total_event_count, rocp_sdk_native_event_names);
    PAPI_CALL(PAPI_start(EventSet));

    // HIP front matter.
    int N = 16;
    dim3 threads_per_block( 1, 1, 1 );
    dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(N) / threads_per_block.y ), 1 );
    hipError_t status;
    size_t probSize = N*N*sizeof(double);

    double *hostA=NULL,  *hostB=NULL,  *hostC=NULL;
    double  *devA=NULL,   *devB=NULL,   *devC=NULL;

    // Allocate host arrays.
    HIP_CALL(hipHostMalloc(&hostA, probSize, 0));
    HIP_CALL(hipHostMalloc(&hostB, probSize, 0));
    HIP_CALL(hipHostMalloc(&hostC, probSize, 0));

    // Allocate device arrays.
    HIP_CALL(hipMalloc(&devA, probSize));
    HIP_CALL(hipMalloc(&devB, probSize));
    HIP_CALL(hipMalloc(&devC, probSize));

    // Initialize arrays.
    int i, j;
    srandom(1);
    for( i = 0; i < N; i++ ) {
        for( j = 0; j < N; j++ ) {
            hostA[i*N + j] = ((double)random())/RAND_MAX + 1.1;
            hostB[i*N + j] = ((double)random())/RAND_MAX + 1.1;
            devC[i*N + j] = 0.0;
        }
    }

    // Data transfer from host to device.
    HIP_CALL(hipMemcpy(devA, hostA, probSize, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(devB, hostB, probSize, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(devC, hostC, probSize, hipMemcpyHostToDevice));

    // Launch the GEMM once.
    hipLaunchKernelGGL(gemm, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    // Call PAPI_read() after the 1st GEMM.
    PAPI_CALL(PAPI_read(EventSet, values));
    fprintf(stdout, "---------------------  PAPI_read()\n");
    for(i = 0; i < total_event_count; ++i) {
        fprintf(stdout, "%s : %lld\n", rocp_sdk_native_event_names[i], values[i]);
    }

    // Pause the profiler for the 2nd GEMM.
    roctx_thread_id_t tid;
    if( pauseResume ) {
        fprintf(stdout, "---------------------  roctxProfilerPause()\n");
        ROCTX_CALL(roctxGetThreadId(&tid));
        ROCTX_CALL(roctxProfilerPause(tid));
        copy_vals(prevValues, values, total_event_count);
    }

    // Launch the GEMM again.
    hipLaunchKernelGGL(gemm, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    // Call PAPI_read() while the profiler is paused.
    PAPI_CALL(PAPI_read(EventSet, values));
    fprintf(stdout, "---------------------  PAPI_read()\n");
    if( pauseResume ) {
        for(i = 0; i < total_event_count; ++i) {
            fprintf(stdout, "%s : %lld (expected %lld)\n", rocp_sdk_native_event_names[i], values[i], prevValues[i]);
            if( values[i] != prevValues[i] ) {
                test_fail(__FILE__, __LINE__, "Counting did not stop after roctxProfilerPause()!", PAPI_EMISC);
            }
        }
    } else {
        for(i = 0; i < total_event_count; ++i) {
            fprintf(stdout, "%s : %lld\n", rocp_sdk_native_event_names[i], values[i]);
        }
    }

    // Resume the profiler.
    if( pauseResume ) {
        fprintf(stdout, "---------------------  roctxProfilerResume()\n");
        ROCTX_CALL(roctxProfilerResume(tid));
    }

    // Launch the GEMM again.
    hipLaunchKernelGGL(gemm, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    // Call PAPI_read() after the 3rd GEMM.
    PAPI_CALL(PAPI_read(EventSet, values));
    fprintf(stdout, "---------------------  PAPI_read()\n");
    for(i = 0; i < total_event_count; ++i) {
        fprintf(stdout, "%s : %lld\n", rocp_sdk_native_event_names[i], values[i]);
    }

    // Pause the profiler for the 4th GEMM.
    if( pauseResume ) {
        fprintf(stdout, "---------------------  roctxProfilerPause()\n");
        ROCTX_CALL(roctxProfilerPause(tid));
        copy_vals(prevValues, values, total_event_count);
    }

    // Launch the GEMM again.
    hipLaunchKernelGGL(gemm, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    // Call PAPI_read() while the profiler is paused.
    PAPI_CALL(PAPI_read(EventSet, values));
    fprintf(stdout, "---------------------  PAPI_read()\n");
    if( pauseResume ) {
        for(i = 0; i < total_event_count; ++i) {
            fprintf(stdout, "%s : %lld (expected %lld)\n", rocp_sdk_native_event_names[i], values[i], prevValues[i]);
            if( values[i] != prevValues[i] ) {
                test_fail(__FILE__, __LINE__, "Counting did not stop after roctxProfilerPause()!", PAPI_EMISC);
            }
        }
    } else {
        for(i = 0; i < total_event_count; ++i) {
            fprintf(stdout, "%s : %lld\n", rocp_sdk_native_event_names[i], values[i]);
        }
    }

    // Resume the profiler.
    if( pauseResume ) {
        fprintf(stdout, "---------------------  roctxProfilerResume()\n");
        ROCTX_CALL(roctxProfilerResume(tid));
    }

    // Launch the GEMM again.
    hipLaunchKernelGGL(gemm, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    // Call PAPI_stop() for final counter reading.
    PAPI_CALL(PAPI_stop(EventSet, values));
    fprintf(stdout, "---------------------  PAPI_read()\n");
    for(i = 0; i < total_event_count; ++i) {
        fprintf(stdout, "%s : %lld\n", rocp_sdk_native_event_names[i], values[i]);
    }

    // HIP back matter.
    HIP_CALL(hipMemcpy(hostC, devC, probSize, hipMemcpyDeviceToHost));
    HIP_CALL(hipFree(devA));
    HIP_CALL(hipFree(devB));
    HIP_CALL(hipFree(devC));
    HIP_CALL(hipFree(hostA));
    HIP_CALL(hipFree(hostB));
    HIP_CALL(hipFree(hostC));

    // Free other dynamically allocated memory.
    free(values);
    free(prevValues);
    for (i = 0; i < total_event_count; i++) {
        free(rocp_sdk_native_event_names[i]);
    }
    free(rocp_sdk_native_event_names);

    // PAPI back matter.
    PAPI_CALL(PAPI_cleanup_eventset( EventSet ));
    PAPI_CALL(PAPI_destroy_eventset( &EventSet ));
    PAPI_shutdown();
    test_pass(__FILE__);
    return 0;
}

void copy_vals(long long *out, long long *in, int size) {

    int i;
    for(i = 0; i < size; ++i) {
        out[i] = in[i];
    }

    return;
}
