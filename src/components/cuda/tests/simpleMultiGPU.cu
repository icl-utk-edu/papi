/**
* @file simpleMultiGPU.cu
* @brief For all enabled NVIDIA devices detected on the machine a matching Cuda context
*        will be created and work will be done on that device. 
*
*        Note: The cuda component supports being partially disabled, meaning that certain devices
*        will not be "enabled" to profile on. If PAPI_CUDA_API is not set, then devices with
*        CC's >= 7.0 will be used and if PAPI_CUDA_API is set to LEGACY then devices with
*        CC's <= 7.0 will be used.
*/

// Standard library headers
#include <stdio.h>
#include <timer.h>

// Cuda Toolkit headers
#include <cuda.h>

// Internal headers
#include "cuda_tests_helper.h"
#include "papi.h"
#include "papi_test.h"
#include "simpleMultiGPU.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// //////////////////////////////////////////////////////////////////////////////
// Data configuration
// //////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
const int DATA_N = 48576 * 32;
const int MAX_NUM_EVENTS = 32;

// //////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA SDK sample describing
// reduction optimization strategies
// //////////////////////////////////////////////////////////////////////////////
__global__ static void reduceKernel( float *d_Result, float *d_Input, int N )
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;
    float sum = 0;

    for( int pos = tid; pos < N; pos += threadN )
        sum += d_Input[pos];

    d_Result[tid] = sum;
}

static void print_help_message(void)
{
    printf("./simpleMultiGPU --cuda-native-event-names [list of cuda native event names separated by a comma].\n"
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

int main(int argc, char **argv)
{
    // Determine the number of Cuda capable devices
    int num_devices = 0;
    check_cuda_runtime_api_call( cudaGetDeviceCount(&num_devices) );

    // No devices detected on the machine, exit
    if (num_devices < 1) {
        fprintf(stderr, "No NVIDIA devices found on the machine. This is required for the test to run.\n");
        test_skip(__FILE__, __LINE__, "", 0); 
    }

    char *user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    int suppress_output = 0;
    if (user_defined_suppress_output) {
        suppress_output = (int) strtol(user_defined_suppress_output, (char**) NULL, 10);
    }
    PRINT(suppress_output, "Running the Cuda component test simpleMultiGPU.cu\n");

    char **cuda_native_event_names = NULL;
    // If command line arguments are provided then get their values.
    int total_event_count = 0;
    if (argc > 1) {
        parse_and_assign_args(argc, argv, &cuda_native_event_names, &total_event_count);
    }

    // Initialize PAPI library
    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init()", papi_errno);
    }
    PRINT(suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION));

    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", cuda_cmp_idx);
    }
    PRINT(suppress_output, "The cuda component is assigned to component index: %d\n", cuda_cmp_idx);  

    // Initialize the Cuda component
    int cuda_eventcode = 0 | PAPI_NATIVE_MASK;
    check_papi_api_call( PAPI_enum_cmp_event(&cuda_eventcode, PAPI_ENUM_FIRST, cuda_cmp_idx) );

    // If we have not gotten an event via the command line, use the event obtained from PAPI_enum_cmp_event
    if (total_event_count == 0) {
        int num_spaces_to_allocate = 1;
        cuda_native_event_names = (char **) malloc(num_spaces_to_allocate * sizeof(char *));
        check_memory_allocation_call( cuda_native_event_names );

        cuda_native_event_names[total_event_count] = (char *) malloc(PAPI_2MAX_STR_LEN * sizeof(char));
        check_memory_allocation_call( cuda_native_event_names[total_event_count] );

        check_papi_api_call( PAPI_event_code_to_name(cuda_eventcode, cuda_native_event_names[total_event_count++]) );
    }   

    const PAPI_component_info_t *cmpInfo = PAPI_get_component_info(cuda_cmp_idx);
    if (cmpInfo == NULL) {
        fprintf(stderr, "Call to PAPI_get_component_info failed.\n");
        exit(EXIT_FAILURE);
    }

    // Check to see if the Cuda component is partially disabled
    if (cmpInfo->partially_disabled) {
        const char *cc_support = (getenv("PAPI_CUDA_API") != NULL) ? "<=7.0" : ">=7.0";
        PRINT(suppress_output, "\033[33mThe cuda component is partially disabled. Only support for CC's %s are enabled.\033[0m\n", cc_support);
    }

    check_cuda_runtime_api_call( cudaGetDeviceCount( &num_devices ) );
    CUdevice device[MAX_GPU_COUNT];
    // Create one context per device. This can be delayed
    // to as late as PAPI_start(), but they are needed to
    // create streams, alloc memory, etc.
    TGPUplan plan[MAX_GPU_COUNT];
    float h_SumGPU[MAX_GPU_COUNT];
    int gpuBase = 0;
    CUcontext ctx[MAX_GPU_COUNT], poppedCtx;
    const int BLOCK_N = 32; 
    const int THREAD_N = 256;
    const int ACCUM_N = BLOCK_N * THREAD_N;
    // Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
    int j;
    int dev_idx;
    for (dev_idx = 0; dev_idx < num_devices; dev_idx++) {
        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(dev_idx) == 0) {
                continue;
            }
        }

        int flags = 0;
        check_cuda_driver_api_call( cuDeviceGet(&device[dev_idx], dev_idx) );
#if defined(CUDA_TOOLKIT_GE_13)
        check_cuda_driver_api_call( cuCtxCreate(&(ctx[dev_idx]), (CUctxCreateParams*)0, flags, device[dev_idx]) );
#else
        check_cuda_driver_api_call( cuCtxCreate(&(ctx[dev_idx]), flags, device[dev_idx]) );
#endif
        plan[dev_idx].dataN = DATA_N / num_devices;
        // Take into account odd data sizes and increment
        if (plan[dev_idx].dataN % 2) {
            plan[dev_idx].dataN++;
        }

        plan[dev_idx].h_Sum = h_SumGPU + dev_idx; // point within h_SumGPU array
        gpuBase += plan[dev_idx].dataN;

        // Create an asynchronous stream
        check_cuda_runtime_api_call( cudaStreamCreate( &plan[dev_idx].stream ) );
        // Allocate memory on the device
        check_cuda_runtime_api_call( cudaMalloc((void **) &plan[dev_idx].d_Data, plan[dev_idx].dataN * sizeof(float)) );
        check_cuda_runtime_api_call( cudaMalloc((void **) &plan[dev_idx].d_Sum, ACCUM_N * sizeof(float)) );
        // Allocates page locked memory on the host
        check_cuda_runtime_api_call( cudaMallocHost((void **) &plan[dev_idx].h_Sum_from_device, ACCUM_N * sizeof(float)) );
        check_cuda_runtime_api_call( cudaMallocHost((void **) &plan[dev_idx].h_Data, plan[dev_idx].dataN * sizeof(float)) );

        for (j = 0; j < plan[dev_idx].dataN; j++) {
            plan[dev_idx].h_Data[j] = ( float ) rand() / ( float ) RAND_MAX;
        }
        check_cuda_driver_api_call( cuCtxPopCurrent(&poppedCtx) );
    }

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    // Handle the events from the command line
    int num_events_successfully_added = 0, numMultipassEvents = 0;
    int NUM_EVENTS = MAX_GPU_COUNT * MAX_NUM_EVENTS;
    char **events_successfully_added = (char **) malloc(NUM_EVENTS * sizeof(char *));
    check_memory_allocation_call( events_successfully_added );

    int event_idx;
    for (dev_idx = 0; dev_idx < num_devices; dev_idx++) {
        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(dev_idx) == 0) {
                continue;
            }
        }

        for (event_idx = 0; event_idx < total_event_count; event_idx++) {
            char tmp_event_name[PAPI_MAX_STR_LEN];
            int strLen = snprintf(tmp_event_name, PAPI_MAX_STR_LEN, "%s:device=%d", cuda_native_event_names[event_idx], dev_idx);
            if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
                fprintf(stderr, "Failed to fully write event name with appended device qualifier.\n");
                exit(EXIT_FAILURE);
            }

            events_successfully_added[num_events_successfully_added] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
            check_memory_allocation_call( events_successfully_added[event_idx] );

            // We must change contexts to the appropriate device to add events to inform PAPI of the context that will run the kernels
            check_cuda_driver_api_call( cuCtxSetCurrent(ctx[dev_idx]) );
            add_cuda_native_events(EventSet, tmp_event_name, &num_events_successfully_added, events_successfully_added, &numMultipassEvents); 
        }
    }

    // Only multiple pass events were provided on the command line
    if (num_events_successfully_added == 0) {
        fprintf(stderr, "Events provided on the command line could not be added to an EventSet as they require multiple passes.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    // Invoke PAPI_start().
    check_papi_api_call( PAPI_start(EventSet) );

    // Start timing
    StartTimer();

    // Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (dev_idx = 0; dev_idx < num_devices; dev_idx++) {
        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(dev_idx) == 0) {
                continue;
            }
        }
        // Pushing a context implicitly sets the device for which it was created.
        check_cuda_driver_api_call( cuCtxPushCurrent(ctx[dev_idx]) );
        // Copy input data from CPU
        check_cuda_runtime_api_call( cudaMemcpyAsync( plan[dev_idx].d_Data, plan[dev_idx].h_Data, plan[dev_idx].dataN * sizeof( float ), cudaMemcpyHostToDevice, plan[dev_idx].stream ) );
        // Perform GPU computations
        reduceKernel <<< BLOCK_N, THREAD_N, 0, plan[dev_idx].stream >>> ( plan[dev_idx].d_Sum, plan[dev_idx].d_Data, plan[dev_idx].dataN );
        check_cuda_runtime_api_call( cudaGetLastError() );
        // Read back GPU results
        check_cuda_runtime_api_call( cudaMemcpyAsync( plan[dev_idx].h_Sum_from_device, plan[dev_idx].d_Sum, ACCUM_N * sizeof( float ), cudaMemcpyDeviceToHost, plan[dev_idx].stream ) );
        // Popping a context can change the device to match the previous context.
        check_cuda_driver_api_call( cuCtxPopCurrent(&(ctx[dev_idx])) );
    }

    // Process GPU results
    PRINT(suppress_output, "Process GPU results...\n");
    for(dev_idx = 0; dev_idx < num_devices; dev_idx++) {
        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(dev_idx) == 0) {
                continue;
            }
        }
        float sum;
        // Pushing a context implicitly sets the device for which it was created.
        check_cuda_driver_api_call( cuCtxPushCurrent(ctx[dev_idx]) );
        // Wait for all operations to finish
        cudaStreamSynchronize( plan[dev_idx].stream );
        // Finalize GPU reduction for current subvector
        sum = 0;
        for (j = 0; j < ACCUM_N; j++) {
            sum += plan[dev_idx].h_Sum_from_device[j];
        }
        *( plan[dev_idx].h_Sum ) = ( float ) sum;
        // Popping a context can change the device to match the previous context.
        check_cuda_driver_api_call( cuCtxPopCurrent(&(ctx[dev_idx])) );
    }
    double gpuTime = GetTimer();


    for (dev_idx=0; dev_idx < num_devices; dev_idx++) {
        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(dev_idx) == 0) {
                continue;
            }
        }
        // Pushing a context implicitly sets the device for which it was created.
        check_cuda_driver_api_call( cuCtxPushCurrent(ctx[dev_idx]) );
        check_cuda_driver_api_call( cuCtxSynchronize( ) );
        // Popping a context may change the current device to match the new current context.
        check_cuda_driver_api_call( cuCtxPopCurrent(&(ctx[dev_idx])) );
    }

    long long cuda_counter_values[NUM_EVENTS];
    check_papi_api_call( PAPI_stop(EventSet, cuda_counter_values) );

    for(event_idx = 0; event_idx < num_events_successfully_added; event_idx++) {
        PRINT(suppress_output, "Event %s produced the value:\t\t%lld\n", events_successfully_added[event_idx], cuda_counter_values[event_idx]);
    }

    check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    PAPI_shutdown();

    float sumGPU = 0.0;
    for(dev_idx = 0; dev_idx < num_devices; dev_idx++) {
        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(dev_idx) == 0) {
                continue;
            }
        }
        sumGPU += h_SumGPU[dev_idx];
    }
    PRINT(suppress_output, "  GPU Processing time: %f (ms)\n", gpuTime);

    // Compute on Host CPU
    PRINT(suppress_output, "Computing the same result with Host CPU...\n");
    StartTimer();
    double sumCPU = 0.0;
    for(dev_idx = 0; dev_idx < num_devices; dev_idx++) {
        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(dev_idx) == 0) {
                continue;
            }
        }

        for (j = 0; j < plan[dev_idx].dataN; j++) {
            sumCPU += plan[dev_idx].h_Data[j];
        }
    }

    double cpuTime = GetTimer();
    if (gpuTime > 0) {
        PRINT(suppress_output, "  CPU Processing time: %f (ms) (speedup %.2fX)\n", cpuTime, (cpuTime/gpuTime));
    } else {
        PRINT(suppress_output, "  CPU Processing time: %f (ms)\n", cpuTime);
    }

    // Compare GPU and CPU results
    PRINT(suppress_output, "Comparing GPU and Host CPU results...\n");
    double diff = fabs( sumCPU - sumGPU ) / fabs( sumCPU );
    PRINT(suppress_output, "  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
    PRINT(suppress_output, "  Relative difference: %E \n", diff);

    // Output a note that a multiple pass event was provided on the command line
    if (numMultipassEvents > 0) {
        PRINT(suppress_output, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    // Cleanup and shutdown
    for(dev_idx = 0; dev_idx < num_devices; dev_idx++ ) {
        if (cmpInfo->partially_disabled) {
            // Device is not enabled continue
            if (determine_if_device_is_enabled(dev_idx) == 0) {
                continue;
            }
        }
        // Free page-locked memory
        check_cuda_runtime_api_call( cudaFreeHost(plan[dev_idx].h_Sum_from_device) );
        check_cuda_runtime_api_call( cudaFreeHost(plan[dev_idx].h_Data) );
        // Free memory on the device
        check_cuda_runtime_api_call( cudaFree(plan[dev_idx].d_Sum) );
        check_cuda_runtime_api_call( cudaFree(plan[dev_idx].d_Data) );
        // Destroys and cleans up asynchronous stream
        check_cuda_runtime_api_call( cudaStreamDestroy(plan[dev_idx].stream) );
        // Destroy Cuda context
        check_cuda_driver_api_call( cuCtxDestroy(ctx[dev_idx]) );
    }

    //Free allocated memory
    for (event_idx = 0; event_idx < total_event_count; event_idx++) {
        free(cuda_native_event_names[event_idx]);
    }
    free(cuda_native_event_names);

    for (event_idx = 0; event_idx < num_events_successfully_added; event_idx++) {
        free(events_successfully_added[event_idx]);
    }   
    free(events_successfully_added);

    if (diff < 1e-5) {
        test_pass(__FILE__);
    }
    else {
        test_fail(__FILE__, __LINE__, "Result of GPU calculation doesn't match CPU.", PAPI_EINVAL);
    }

    return 0;
}
