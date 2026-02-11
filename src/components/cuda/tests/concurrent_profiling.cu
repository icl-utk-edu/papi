/**
* @file concurrent_profiling.cu
* @brief This test utilizes concurrent kernels by taking the ration of runtimes for a consecutive series of kernels,
*        compared to a series of concurrent kernels, one can definitively demonstrate that concurrent kernels
*        were running while native events were gathered and the user replay mechanism was in use. Cuda contexts are
*        created with calls to cuCtxCreate.
*
*        Example: 4 kernel launches, with 1x, 2x, 3x, and 4x amounts of work, each sized to one SM (one warp
*        of threads, one thread block). When run synchronously, this comes to 10x amount of work. When run
*        concurrently, the longest (4x) kernel should be the only measured time (it hides the others).
*        Thus w/4 kernels, the concurrent : consecutive time ration should be 4:10.
*        On test hardware this does simplify to 3.998:10. As the test is affected by memory layout, this
*        may not hold for certain architectures where, for example, cache sizes may optimize certain kernel
*        calls. 
*
*        After demonstrating concurrent usign multiple streams, this test then demonstrates using multiple devices.
*        In this 3rd configuration, the same concurrent workflow with streams is then duplicated and run
*        on each device concurrently using streams. In this, case, the wallclock time to launch, run, and join
*        threads should be roughly the same as the wallclock time to run the single device case. If concurrency
*        was not working, the wallclock time would be (number of deivces) times the single device concurrent case.
*
*        Notes:
*        - This test only works with CC's >= 7.0 which follows exactly what is done
*          for the concurrent_profiling.cu test in extras/CUPTI/samples/concurrent_profiling.
*
*        - If the multiple devices have different performance, the runtime may be significantly different between
*          devices, but this does not mean concurrent profiling is not happening.
*/

// Standard library headers
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
using ::std::string;
#include <thread>
using ::std::thread;
#include <vector>
using ::std::vector;
#include <algorithm>
using ::std::find;

// Cuda Toolkit headers
#include <cuda.h>
#include <cupti_profiler_target.h>

// Internal headers
extern "C" {
    #include "cuda_tests_helper.h"
    #include "papi.h"
    #include "papi_test.h"
}

// Currently we are only adding cuda:::sm__cycles_active:stat=sum and cuda:::sm__cycles_elapsed:stat=max 
#define MAX_EVENTS_TO_ADD 2

int global_suppress_output;

typedef struct
{
    int device;                //!< compute device number
    CUcontext context;         //!< CUDA driver context, or NULL if default context has already been initialized
} profilingConfig;

// Per-device configuration, buffers, stream and device information, and device pointers
typedef struct
{
    // For each device, store the range name
    vector<string> range_name;
    // For each device, store the successfully added events
    vector<string> events_successfully_added;
    // For each device (or each context) store its CUPTI profiling config
    profilingConfig config;
    // For each device, store its streams
    vector<cudaStream_t> streams;
    // For each device, allocate memory
    vector<double *> d_x;
    vector<double *> d_y;
    // For each event for a device, store PAPI counter values
    vector<vector<long long>> cuda_counter_values;
} perDeviceData;

#define DAXPY_REPEAT 32768
// Loop over array of elements performing daxpy multiple times
// To be launched with only one block (artificially increasing serial time to better demonstrate overlapping replay)
__global__ void daxpyKernel(int elements, double a, double * x, double * y)
{
    for (int i = threadIdx.x; i < elements; i += blockDim.x)
        // Artificially increase kernel runtime to emphasize concurrency
        for (int j = 0; j < DAXPY_REPEAT; j++)
            y[i] = a * x[i] + y[i]; // daxpy
}

// Initialize kernel values
double a = 2.5;

// Normally you would want multiple warps, but to emphasize concurrency with streams and multiple devices
// we run the kernels on a single warp.
int threadsPerBlock = 32;
int threadBlocks = 1;

// Configurable number of kernels (streams, when running concurrently)
int const numKernels = 4;
int const num_streams = numKernels;
vector<size_t> elements(numKernels);

// Each kernel call allocates and computes (call number) * (blockSize) elements
// For 4 calls, this is 4k elements * 2 arrays * (1 + 2 + 3 + 4 stream mul) * 8B/elem =~ 640KB
int const blockSize = 4 * 1024;

// Globals for successfully added and multiple pass events
int global_num_multipass_events;

/** @class add_cuda_native_events
  * @brief Try and add each event provided on the command line by the user.
  *
  * @param EventSet
  *   A PAPI eventset.
  * @param cuda_native_event_name
  *   Event to add to the EventSet.
  * @param &device_data
  *   Per device configuration.
  * @param *numMultipassEvents
  *   Counter to see if a multiple pass event was provided on the command line.
*/
static void add_cuda_native_events_concurrent(int EventSet, string cuda_native_event_name, perDeviceData &device_data, int *numMultipassEvents)
{  
   int papi_errno = PAPI_add_named_event(EventSet, cuda_native_event_name.c_str());
   if (papi_errno != PAPI_OK) {
       if (papi_errno != PAPI_EMULPASS) {
           fprintf(stderr, "Unable to add event %s to the EventSet with error code %d.\n", cuda_native_event_name, papi_errno);
           exit(EXIT_FAILURE);
       }
       
       // Handle multiple pass events
       (*numMultipassEvents)++;
   }

   // Handle successfully added events
   if (find(device_data.events_successfully_added.begin(), device_data.events_successfully_added.end(), cuda_native_event_name.c_str()) == device_data.events_successfully_added.end()) {
       device_data.events_successfully_added.push_back(cuda_native_event_name.c_str());
   }   
    
    return;
}

// Wrapper which will launch numKernel kernel calls on a single device
// The device streams vector is used to control which stream each call is made on
// If 'serial' is non-zero, the device streams are ignored and instead the default stream is used
void profileKernels(perDeviceData &d,
                    vector<string> const &base_cuda_native_event_names_with_stat_qual,
                    char const * const rangeName, bool serial)
{
    // Switch to desired device
    check_cuda_runtime_api_call( cudaSetDevice(d.config.device) );  // Orig code has mistake here

    check_cuda_driver_api_call( cuCtxSetCurrent(d.config.context) );

    int EventSet = PAPI_NULL;
    check_papi_api_call( PAPI_create_eventset(&EventSet) );

    global_num_multipass_events = 0;
    int event_idx;
    for (event_idx = 0; event_idx < base_cuda_native_event_names_with_stat_qual.size(); event_idx++) {
        string tmp_event_name = base_cuda_native_event_names_with_stat_qual[event_idx] + ":device=" + std::to_string(d.config.device);
        add_cuda_native_events_concurrent(EventSet, tmp_event_name, d, &global_num_multipass_events);
    }

    //add_cuda_native_events(d, EventSet, base_cuda_native_event_names_with_stat_qual, &global_num_multipass_events);

    // Only multiple pass events were provided on the command line
    if (d.events_successfully_added.size() == 0) {
        fprintf(stderr, "Both cuda:::sm__cycles_active:stat=sum and cuda:::sm__cycles_elapsed:stat=max were unable to be added. This may be due to the architecture you are running on.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    // Internally at PAPI_start we push a range; therefore, users do not push a range
    check_papi_api_call( PAPI_start(EventSet) );

    unsigned int stream;
    for (stream = 0; stream < d.streams.size(); stream++)
    {
        cudaStream_t streamId = (serial ? 0 : d.streams[stream]);
        daxpyKernel <<<threadBlocks, threadsPerBlock, 0, streamId>>> (elements[stream], a, d.d_x[stream], d.d_y[stream]);
    }

    // After launching all work, synchronize all streams
    if (serial == false)
    {
        for (stream = 0; stream < d.streams.size(); stream++)
        {
            check_cuda_runtime_api_call( cudaStreamSynchronize(d.streams[stream]) );
        }
    }
    else
    {
        check_cuda_runtime_api_call( cudaStreamSynchronize(0) );
    }

    // Internally at PAPI_stop we pop the range; therefore, users do not pop a range
    long long values[MAX_EVENTS_TO_ADD];
    check_papi_api_call( PAPI_stop(EventSet, values) );

    for (event_idx = 0; event_idx < d.events_successfully_added.size(); event_idx++) {
        d.cuda_counter_values[event_idx].push_back(values[event_idx]);
    }

    check_papi_api_call( PAPI_cleanup_eventset(EventSet) );

    check_papi_api_call( PAPI_destroy_eventset(&EventSet) );

    // Keep track of the range name, again PAPI internally has defined a range name, but
    // for the sake of following the CUPTI test, we will use their range names.
    d.range_name.push_back(rangeName);
}

void print_measured_values(perDeviceData &d)
{
    PRINT(global_suppress_output, "%s\n", std::string(200, '-').c_str());
    int event_idx;
    for (event_idx = 0; event_idx < d.events_successfully_added.size(); event_idx++) {
        int range_idx;
        for (range_idx = 0; range_idx < d.range_name.size(); range_idx++) {
            PRINT(global_suppress_output, "Range %s with event %s produced the value:\t\t%lld\n", d.range_name[range_idx].c_str(), d.events_successfully_added[event_idx].c_str(), d.cuda_counter_values[event_idx][range_idx]);
        }
    }
}

static void print_help_message(void)
{
    printf("./concurrent_profiling\n");
    printf("Notes:\n"
           "1. This test is specifically designed to use devices that support CUPTI Profiling i.e. devices with CCs >= 7.0.\n"
           "2. No events are accepted from the command line as cuda:::sm_cycles_active:stat=sum, cuda:::sm__cycles_elapsed:stat=max,\n"
           "   and cuda:::smsp__sass_thread_inst_executed_op_dfma_pred_on:stat=sum are required events.\n");
}

int main(int argc, char **argv)
{
    char *papi_cuda_api = getenv("PAPI_CUDA_API");
    if (papi_cuda_api != NULL) {
        fprintf(stderr, "The concurrent_profiling test only works with the Perfworks Metrics API. Unset the environment variable PAPI_CUDA_API.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }    

    // Determine the number of Cuda capable devices
    int num_devices = 0;
    check_cuda_runtime_api_call( cudaGetDeviceCount(&num_devices) );

    // No devices detected on the machine, exit
    if (num_devices < 1) {
        fprintf(stderr, "No NVIDIA devices found on the machine. This is required for the test to run.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    global_suppress_output = 0;
    char *user_defined_suppress_output = getenv("PAPI_CUDA_TEST_QUIET");
    if (user_defined_suppress_output) {
        global_suppress_output = (int) strtol(user_defined_suppress_output, (char**) NULL, 10);
    }
    PRINT(global_suppress_output, "Running the cuda component test concurrent_profiling.cu\n")

    // User either provided --help or an argument that would not be useful to this test
    if (argc > 1) {
        print_help_message();
        exit(EXIT_SUCCESS);
    }

    // Initialize the PAPI library
    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init()", papi_errno);
    }
    PRINT(global_suppress_output, "PAPI version being used for this test: %d.%d.%d\n",
          PAPI_VERSION_MAJOR(PAPI_VERSION),
          PAPI_VERSION_MINOR(PAPI_VERSION),
          PAPI_VERSION_REVISION(PAPI_VERSION));

    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", cuda_cmp_idx);
    }   
    PRINT(global_suppress_output, "The cuda component is assigned to component index: %d\n", cuda_cmp_idx);  

    // Per-device information
    vector<int> device_ids;
    // Find all devices capable of running CUPTI Profiling (CC >= 7.0)
    int dev_idx;
    for (dev_idx = 0; dev_idx < num_devices; dev_idx++)
    {
        // Obtain major compute capability
        int major;
        check_cuda_runtime_api_call( cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_idx) );
        if (major >= 7) {
            PRINT(global_suppress_output, "--> Device %d is compatible with the concurrent_profiling test\n", dev_idx);
            device_ids.push_back(dev_idx);
        }
    }
    if (device_ids.size() == 0) {
        fprintf(stderr, "No devices on the machine detected that have CC >= 7.0 and support CUPTI Profiling.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
    
    // Overwrite num_devices with the number of devices this test actually supports
    num_devices = device_ids.size();

    // Initialize kernel input to some known numbers
    vector<double> h_x(blockSize * numKernels);
    vector<double> h_y(blockSize * numKernels);
    size_t i;
    for (i = 0; i < blockSize * numKernels; i++) {
        h_x[i] = 1.5 * i;
        h_y[i] = 2.0 * (i - 3000);
    }

    // Initialize a vector of 'default stream' values to demonstrate serialized kernels
    vector<cudaStream_t> default_streams(num_streams);
    int stream;
    for (stream = 0; stream < num_streams; stream++) {
        default_streams[stream] = 0;
    }

    // Scale per-kernel work by stream number
    for (stream = 0; stream < num_streams; stream++) {
        elements[stream] = blockSize * (stream + 1);
    }

    // These metrics below are hardcoded in the CUPTI provided sample code and therefore this test was
    // written with them in mind. No command line arguments will be accepted. The test will be skipped
    // if none of them can be successfully added.
    vector<string> base_cuda_native_event_names_with_stat_qual;
    // The below two metrics will demonstrate whether kernels within a Range were run serially or concurrently.
    base_cuda_native_event_names_with_stat_qual.push_back("cuda:::sm__cycles_active:stat=sum");
    base_cuda_native_event_names_with_stat_qual.push_back("cuda:::sm__cycles_elapsed:stat=max");
    // This metric shows that the same number of flops were executed on each run.
    //base_cuda_native_event_names_with_stat_qual.push_back("cuda:::smsp__sass_thread_inst_executed_op_dfma_pred_on:stat=sum");

    // For each device, configure profiling, set up buffers, copy kernel data
    vector<perDeviceData> device_data(num_devices);
    int device;
    for (device = 0; device < num_devices; device++)
    {
        int device_id = device_ids[device];
        check_cuda_runtime_api_call( cudaSetDevice(device_id) );
        PRINT(global_suppress_output, "--> Configuring device %d\n", device_id);

        // Required CUPTI Profiling configuration & initialization
        // Can be done ahead of time or immediately before startSession() call
        // Initialization & configuration images can be generated separately, then passed to later calls
        // For simplicity's sake, in this sample, a single config struct is created per device and passed to each CUPTI Profiler API call
        // For more complex cases, each combination of CUPTI Profiler Session and Config requires additional initialization
        profilingConfig config;
        // Device ID, used to get device name for metrics enumeration
        config.device = device_id;
        // config.maxLaunchesPerPass = 1;     // Must be >= maxRangesPerPass.  Set this to the largest count of kernel launches which may be encountered in any Pass in this Session

        // // Device 0 has max of 3 passes; other devices only run one pass in this sample code
        int flags = 0;
#if defined(CUDA_TOOLKIT_GE_13)
        check_cuda_driver_api_call( cuCtxCreate(&(config.context), (CUctxCreateParams*)0, flags, device) );
#else
        check_cuda_driver_api_call( cuCtxCreate(&(config.context), flags, device) );
#endif
        device_data[device].config = config;// Save this device config

        // Initialize CUPTI Profiling structures
        // targetInitProfiling(device_data[device], base_cuda_native_event_names_with_stat_qual);

        // Per-stream initialization & memory allocation - copy from constant host array to each device array
        device_data[device].streams.resize(num_streams);
        device_data[device].d_x.resize(num_streams);
        device_data[device].d_y.resize(num_streams);
        // Resize the vector of vectors for the number of events we have
        device_data[device].cuda_counter_values.resize( base_cuda_native_event_names_with_stat_qual.size());
        for (stream = 0; stream < num_streams; stream++)
        {
            // Create an asynchronous stream
            check_cuda_runtime_api_call( cudaStreamCreate(&(device_data[device].streams[stream])) );

            // Each kernel does (stream #) * blockSize work on doubles
            size_t size = elements[stream] * sizeof(double);

            check_cuda_runtime_api_call( cudaMalloc(&(device_data[device].d_x[stream]), size) );
            check_memory_allocation_call( device_data[device].d_x[stream] );
            check_cuda_runtime_api_call( cudaMemcpy(device_data[device].d_x[stream], h_x.data(), size, cudaMemcpyHostToDevice) );

            check_cuda_runtime_api_call( cudaMalloc(&(device_data[device].d_y[stream]), size) );
            check_memory_allocation_call( device_data[device].d_y[stream] );
            check_cuda_runtime_api_call( cudaMemcpy(device_data[device].d_y[stream], h_x.data(), size, cudaMemcpyHostToDevice) );
        }
    }

    // Formatting print statement
    PRINT(global_suppress_output, "%s\n", std::string(200, '-').c_str());

    ////////////////////////////////////////////////////////////////////////////////
    // First Version - single device, kernel calls serialized on default stream. //
    //////////////////////////////////////////////////////////////////////////////
       
    // Use wallclock time to measure performance
    auto begin_time = ::std::chrono::high_resolution_clock::now();

    // Run on first device and use default streams, which run serially
    profileKernels(device_data[0], base_cuda_native_event_names_with_stat_qual, "single_device_serial", true);

    auto end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_serial_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time).count();
    int numBlocks = 0;
    for (i = 1; i <= numKernels; i++)
    {
        numBlocks += i;
    }
    PRINT(global_suppress_output, "It took %d ms on the host to profile %d kernels in serial.\n", elapsed_serial_ms, numKernels);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Second version - same kernel calls as before on the same device, but now using separate streams for concurrency, //
    // Should be limited by the longest running kernel.                                                                //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // User wallclock time to measure performance
    begin_time = ::std::chrono::high_resolution_clock::now();

    // Still only use first device, but this time use its allocated streams for parallelism
    profileKernels(device_data[0], base_cuda_native_event_names_with_stat_qual, "single_device_async", false);

    end_time = ::std::chrono::high_resolution_clock::now();
    int elapsed_single_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time).count();
    PRINT(global_suppress_output, "It took %d ms on the host to profile %d kernels on a single device on separate streams.\n", elapsed_single_device_ms, numKernels);
    PRINT(global_suppress_output, "--> If the separate stream wallclock time is less than the serial version, the streams were profiling concurrently.\n");

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Third version - same as the second case, but duplicates the concurrent work across devices to show cross-device concurrency. //
    // This is done using devices so no serialization is needed between devices.                                                   //
    // Should have roughly the same wallclock time as the second case if the devices have similar performance.                    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // The third version can only be ran if we have more than one compatible device found
    if (device_ids.size() == 1)
    {
        PRINT(global_suppress_output, "Only one compatible device found; skipping the multi-threaded test.\n");
    }
    else
    {
        // Initialize PAPI thread support
        check_papi_api_call( PAPI_thread_init((unsigned long (*)(void)) std::this_thread::get_id) );

        // Formatting print statement
        PRINT(global_suppress_output, "\n");
        PRINT(global_suppress_output, "Running on %d devices, one thread per device.\n", num_devices);

        // Time creation of the same multiple streams (on multiple devices, if possible)
        vector<::std::thread> threads;
        begin_time = ::std::chrono::high_resolution_clock::now();

        // Now launch parallel thread work, duplicated on one thread per device
        int thread;
        for (thread = 0; thread < num_devices; thread++)
        {
            threads.push_back(::std::thread(profileKernels, ::std::ref(device_data[thread]), base_cuda_native_event_names_with_stat_qual, "multi_device_async", false));
        }

        // Wait for all threads to finish
        for (auto &t: threads)
        {
            t.join();
        }

        // Record time used when launching on multiple devices
        end_time = ::std::chrono::high_resolution_clock::now();
        int elapsed_multiple_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time).count();
        PRINT(global_suppress_output, "It took %d ms on the host to profile the same %d kernels on each of the %d devices in parallel\n", elapsed_multiple_device_ms, numKernels, num_devices);
        PRINT(global_suppress_output, "--> Wallclock ratio of parallel device launch to single device launch is %f\n", elapsed_multiple_device_ms / (double) elapsed_single_device_ms);
        PRINT(global_suppress_output, "--> If the ratio is close to 1, that means there was little overhead to profile in parallel on multiple devices compared to profiling on a single device.\n");
        PRINT(global_suppress_output, "--> If the devices have different performance, the ratio may not be close to one, and this should be limited by the slowest device.\n");
    }

    // Free stream memory for each device
    for (i = 0; i < num_devices; i++)
    {
        int j;
        for (j = 0; j < numKernels; j++)
        {
            check_cuda_runtime_api_call( cudaFree(device_data[i].d_x[j]) );
            check_cuda_runtime_api_call( cudaFree(device_data[i].d_y[j]) );
        }
    }

    // Display metric values
    PRINT(global_suppress_output, "\nMetrics for device #0:\n");
    PRINT(global_suppress_output, "Look at the cuda:::sm__cycles_elapsed:stat=max values for each test.\n");
    PRINT(global_suppress_output, "This value represents the time spent on device to run the kernels in each case, and should be longest for the serial range, and roughly equal for the single and multi device concurrent ranges.\n");
    print_measured_values(device_data[0]);

    // Only display next device info if needed
    if (num_devices > 1)
    {
        PRINT(global_suppress_output, "\nMetrics for the remaining devices only display the multi device async case and should all be similar to the first device's values if the device has similar performance characteristics.\n");
        PRINT(global_suppress_output, "If devices have different performance characteristics, the runtime cycles calculation may vary by device.\n");
    }

    for (i = 1; i < num_devices; i++)
    {
        PRINT(global_suppress_output, "\nMetrics for device #%d:\n", i);
        print_measured_values(device_data[i]);
    }

    // Output a note that a multiple pass event was provided on the command line
    if (global_num_multipass_events > 0) {
        PRINT(global_suppress_output, "\033[0;33mNOTE: From the events provided on the command line, an event or events requiring multiple passes was detected and not added to the EventSet. Check your events with utils/papi_native_avail.\n\033[0m");
    }

    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
