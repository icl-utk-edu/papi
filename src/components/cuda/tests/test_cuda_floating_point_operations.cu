/**
* @file cuda_floating_point_operations
* @brief This test verifies the counters collected for cuda native events that deal with
*        floating point operations.
*/

// Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Cuda headers
#include <cuda.h>

// Internal Headers
#include "cuda_tests_helper.h"
#include "papi.h"
#include "papi_test.h"

#define MAX_ITERATION_ENTRIES 3

typedef enum {
    ADD = 0,
    MULTIPLY,
    FUSED_MULTIPLY_ADD,
} operation_e;

typedef enum {
    SINGLE = 0,
    DOUBLE,
} precision_e;

__global__ void add(int num_iterations_arg, void *x_arg, void *y_arg, precision_e precision_arg) {
    int i;
    for (i = 0; i < num_iterations_arg; i++) {
        if (precision_arg == SINGLE) {
            ((float *) y_arg)[i] = ((float *) x_arg)[i] + ((float *) y_arg)[i];
        }
        else {
            ((double *) y_arg)[i] = ((double *) x_arg)[i] + ((double *) y_arg)[i];
        }
    }

    return;
}

__global__ void multiply(int num_iterations_arg, void *x_arg, void *y_arg, precision_e precision_arg)
{
    int i;
    for (i = 0; i < num_iterations_arg; i++) {
        if (precision_arg == SINGLE) {
            ((float *) y_arg)[i] = ((float *) x_arg)[i] * ((float *) y_arg)[i];
        }
        else {
            ((double *) y_arg)[i] = ((double *) x_arg)[i] * ((double *) y_arg)[i];
        }
    }

    return;
}

__global__ void fused_multiply_add(int num_iterations_arg, void *x_arg, void *y_arg, precision_e precision_arg)
{
    int i;
    for (i = 0; i < num_iterations_arg; i++) {
        if (precision_arg == SINGLE) {
            ((float *) y_arg)[i] = ((float *) x_arg)[i] * ((float *) y_arg)[i] + 1.0f;
        }
        else {
            ((double *) y_arg)[i] = ((double *) x_arg)[i] * ((double *) y_arg)[i] + 1.0;
        }
    }

    return;
}

void launch_kernel(int iterations_arg, void *x_arg, void *y_arg, precision_e precision_arg, operation_e operation_arg)
{
    switch(operation_arg) {
        case ADD:
            add<<<1, 1>>>(iterations_arg, x_arg, y_arg, precision_arg);
            break;
        case MULTIPLY:
            multiply<<<1,1>>>(iterations_arg, x_arg, y_arg, precision_arg);
            break;
        case FUSED_MULTIPLY_ADD:
            fused_multiply_add<<<1, 1>>>(iterations_arg, x_arg, y_arg, precision_arg);
            break;
        default:
            break;
    }
    check_cuda_runtime_api_call( cudaGetLastError() );
    check_cuda_runtime_api_call( cudaDeviceSynchronize() );

    return;
}

static void print_help_message(void)
{
    printf("./cuda_floating_point_operations --device [nvidia device index]"
           " --number-of-iterations [list of iterations to perform (must be <= 3) separated by a comma]"
           " --precision [options include single (default) or double]"
           " --operation [options include add (default), multiply, fused_add_multiply].\n"
           "Notes:\n"
           "1. The default precision is single and the default operation is add.\n"
           "2. If the number of iterations listed is greater than 3 then the iteration value will not be stored,"
           " BUT the test will proceed.\n");
}

static void parse_and_assign_args(int argc, char *argv[], int *device_index_arg, precision_e *precision_arg,
                                  operation_e *operation_arg, long long *numberOfIterations_arg)
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
        else if (strcmp(arg, "--device") == 0)
        {
            if (!argv[i + 1])
            {
                printf("ERROR!! Add a nvidia device index.\n");
                exit(EXIT_FAILURE);
            }
            *device_index_arg = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(arg, "--number-of-iterations") == 0)
        {
            if (!argv[i + 1])
            {
                fprintf(stderr, "ERROR!! --number-of-iterations given, but no numbers listed. Exiting.\n");
                exit(EXIT_FAILURE);
            }

            int countOfIterationsListed = 0;
            const char *iterationNumber = strtok(argv[i+1], ",");
            // As the passed in array numberOfIterations_arg is already set to have 3 default entries, we guard
            // against a user providing a 4
            while (iterationNumber != NULL && countOfIterationsListed <= 2)
            {
                numberOfIterations_arg[countOfIterationsListed] = atoll(iterationNumber);

                countOfIterationsListed++;
                iterationNumber= strtok(NULL, ",");
            }
            i++;
        }
        else if (strcmp(arg, "--precision") == 0)
        {
            if (!argv[i + 1])
            {
                fprintf(stderr, "ERROR!! --precision given, but no precision listed. Exiting.\n");
                exit(EXIT_FAILURE);
            }

            if (strcmp(argv[i+1], "single") == 0) {
                *precision_arg = SINGLE;
            }
            else if (strcmp(argv[i+1], "double") == 0) {
                *precision_arg = DOUBLE;
            }
            else {
                fprintf(stderr, "Provided precision is not valid. Exiting.\n");
                print_help_message();
                exit(EXIT_FAILURE);
            }
            i++;
        }
        else if (strcmp(arg, "--operation") == 0)
        {
            if (!argv[i + 1])
            {
                fprintf(stderr, "ERROR!! --operation given, but no operation listed. Exiting.\n");
                exit(EXIT_FAILURE);
            }

            if (strcmp(argv[i+1], "add") == 0) {
                *operation_arg = ADD;
            }
            else if (strcmp(argv[i+1], "multiply") == 0) {
                *operation_arg = MULTIPLY;
            }
            else if (strcmp(argv[i+1], "fused_multiply_add") == 0) {
                *operation_arg = FUSED_MULTIPLY_ADD;
            }
            else {
                fprintf(stderr, "Provided operation is not valid. Exiting.\n");
                print_help_message();
                exit(EXIT_FAILURE);
            }
            i++;
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
    char *papi_cuda_api = getenv("PAPI_CUDA_API");
    if (papi_cuda_api != NULL) {
        fprintf(stderr, "test_cuda_floating_point_operations.cu only works with the Perfworks Metrics API (CC's >= 7.0). Unset the environment variable PAPI_CUDA_API.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    // Determine the number of Cuda capable devices
    int num_devices = 0;
    check_cuda_runtime_api_call( cudaGetDeviceCount(&num_devices) );
    // No devices detected on the machine, exit
    if (num_devices < 1) {
        fprintf(stderr, "No NVIDIA devices found on the machine. This is required for the test to run. Skipping.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    int suppress_output = 0;
    char *user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    if (user_defined_suppress_output) {
        suppress_output = (int) strtol(user_defined_suppress_output, (char**) NULL, 10);
    }
    PRINT(suppress_output, "Running the cuda component test cuda_floating_point_operations.cu\n");

    int cudaDeviceIndex = 0;
    precision_e precision = SINGLE;
    operation_e operation = ADD;
    long long numberOfIterations[MAX_ITERATION_ENTRIES] = {2, 4, 16};
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &cudaDeviceIndex, &precision, &operation, numberOfIterations);
    }

    int dev_idx;
    for (dev_idx = 0; dev_idx < num_devices; dev_idx++) {
        int major;
        check_cuda_runtime_api_call( cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_idx) );
        if (major < 7 && cudaDeviceIndex == dev_idx) {
            fprintf(stderr, "test_cuda_floating_point_operations.cu only works with the Perfworks Metrics API (CC's >= 7.0)."
                            " The device index provided (%d) does not support the Perfworks Metrics API."
                            " Skipping.\n", cudaDeviceIndex);
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }

    check_cuda_runtime_api_call( cudaSetDevice(cudaDeviceIndex) );

    const char *nativeEventName = NULL;
    // Native event name which corresponds to single precision and an operation of addition
    if (precision == SINGLE && operation == ADD) {
        nativeEventName = "cuda:::smsp__sass_thread_inst_executed_op_fadd_pred_on:stat=sum";
    }
    // Native event name which corresponds to single precision and an operation of multiply
    else if (precision == SINGLE && operation == MULTIPLY) {
        nativeEventName = "cuda:::smsp__sass_thread_inst_executed_op_fmul_pred_on:stat=sum";
    }
    // Native event name which corresponds to single precision and an operation of multiply + add
    else if (precision == SINGLE && operation == FUSED_MULTIPLY_ADD) {
        nativeEventName = "cuda:::smsp__sass_thread_inst_executed_op_ffma_pred_on:stat=sum";
    }
    // Native event name which corresponds to double precision and an operation of add
    else if (precision == DOUBLE && operation == ADD) {
        nativeEventName = "cuda:::smsp__sass_thread_inst_executed_op_dadd_pred_on:stat=sum";
    }
    // Native event name which corresponds to double precision and an operation of multiply
    else if (precision == DOUBLE && operation == MULTIPLY) {
        nativeEventName = "cuda:::smsp__sass_thread_inst_executed_op_dmul_pred_on:stat=sum";
    }
    // Native event name which corresponds to double precision and an operation of multiply + add
    else if (precision == DOUBLE && operation == FUSED_MULTIPLY_ADD) {
        nativeEventName = "cuda:::smsp__sass_thread_inst_executed_op_dfma_pred_on:stat=sum";
    }
    // Error occurred
    else {
        fprintf(stderr, "The combination of precision and operation do not correspond to a cuda native event. Exiting.\n");
        exit(EXIT_FAILURE);
    }

    // To properly allocate the below arrays with enough space determine the max iteration number
    long long maxIteration = 0;
    int i;
    for (i = 0; i < MAX_ITERATION_ENTRIES; i++) {
        if (maxIteration < numberOfIterations[i]) {
            maxIteration = numberOfIterations[i];
        }
    }

    // Allocate memory for x and y arrays
    float *x, *y;
    unsigned int flags = cudaMemAttachGlobal; // Allows memory to be accessible from any stream on any device
    check_cuda_runtime_api_call( cudaMallocManaged(&x, maxIteration * sizeof(float), flags ) );
    check_cuda_runtime_api_call( cudaMallocManaged(&y, maxIteration * sizeof(float), flags ) );

    // Initialize x and y arrays on the host
    for (i = 0; i < maxIteration; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int papiErrno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papiErrno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init()", papiErrno);
    }
    PRINT(suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
      PAPI_VERSION_MAJOR(PAPI_VERSION),
      PAPI_VERSION_MINOR(PAPI_VERSION),
      PAPI_VERSION_REVISION(PAPI_VERSION));

    papiErrno = PAPI_query_named_event(nativeEventName);
    if (papiErrno != PAPI_OK) {
        fprintf(stderr, "The cuda native event (%s) does not exist on the machine. Skipping.\n", nativeEventName);
        test_skip(__FILE__, __LINE__, "", 0);
    }

    int eventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&eventSet) );

    check_papi_api_call( PAPI_add_named_event(eventSet, nativeEventName) );

    check_papi_api_call( PAPI_start(eventSet) );

    long long expectedCounterValue = numberOfIterations[0];
    launch_kernel(numberOfIterations[0], x, y, precision, operation);
    long long counterValue = 0;
    check_papi_api_call( PAPI_read(eventSet, &counterValue) );
    if (counterValue == expectedCounterValue) {
        printf("1st PAPI_read: Correct count -- expected was %lld and actual is %lld.\n", expectedCounterValue, counterValue);
    }
    else {
        fprintf(stderr, "\033[0;31m1st PAPI_read: Incorrect count -- expected was %lld and actual is %lld.\n\033[0m", expectedCounterValue, counterValue);
        exit(EXIT_FAILURE);
    }

    expectedCounterValue += numberOfIterations[1];
    launch_kernel(numberOfIterations[1], x, y, precision, operation);
    check_papi_api_call( PAPI_read(eventSet, &counterValue) );
    if (counterValue == expectedCounterValue) {
        printf("2nd PAPI_read: Correct count -- expected was %lld and actual is %lld.\n", expectedCounterValue, counterValue);
    }
    else {
        fprintf(stderr, "\033[0;31m2nd PAPI_read: Incorrect count -- expected was %lld and actual is %lld.\n\033[0m", expectedCounterValue, counterValue);
        exit(EXIT_FAILURE);
    }

    expectedCounterValue += numberOfIterations[2];
    launch_kernel(numberOfIterations[2], x, y, precision, operation);
    check_papi_api_call( PAPI_read(eventSet, &counterValue) );
    if (counterValue == expectedCounterValue) {
        printf("3rd PAPI_read: Correct count -- expected was %lld and actual is %lld.\n", expectedCounterValue, counterValue);
    }
    else {
        fprintf(stderr, "\033[0;31m3rd PAPI_read: Correct count -- expected was  %lld and actual is %lld.\n\033[0m", expectedCounterValue, counterValue);
        exit(EXIT_FAILURE);
    }

    // No work is occurring; therefore, PAPI_read here SHOULD give back the counter value obtained in the 3rd PAPI_read
    check_papi_api_call( PAPI_read(eventSet, &counterValue) );
    if (counterValue == expectedCounterValue) {
        printf("Final PAPI_read: Correct count -- expected was %lld and actual is %lld.\n", expectedCounterValue, counterValue);
    }   
    else {
        fprintf(stderr, "\033[0;31mFinal PAPI_read: Incorrect count -- expected was %lld and actual is %lld.\n\033[0m", expectedCounterValue, counterValue);
        exit(EXIT_FAILURE);
    }  

    // No work is occurring; therefore, PAPI_stop here SHOULD give back the counter value obtained in the 3rd PAPI_read
    check_papi_api_call( PAPI_stop(eventSet, &counterValue) );
    if (counterValue == expectedCounterValue) {
        printf("PAPI_stop: Correct count -- expected was %lld and actual is %lld.\n", expectedCounterValue, counterValue);
    }
    else {
        fprintf(stderr, "\033[0;31mPAPI_stop: Incorrect count -- expected was %lld and actual is %lld.\n\033[0m", expectedCounterValue, counterValue);
        exit(EXIT_FAILURE);
    }

    check_cuda_runtime_api_call( cudaFree(x) );
    check_cuda_runtime_api_call( cudaFree(y) );

    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
