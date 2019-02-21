/* 
 * This example is taken from the NVIDIA documentation (Copyright 1993-2013
 * NVIDIA Corporation) and has been adapted to show the use of CUPTI in
 * collecting event counters for multiple GPU contexts.
 */

/*
 * This software contains source code provided by NVIDIA Corporation
 *
 * According to the Nvidia EULA (compute 5.5 version)
 * http://developer.download.nvidia.com/compute/cuda/5_5/rel/docs/EULA.pdf
 *
 * Chapter 2. NVIDIA CORPORATION CUDA SAMPLES END USER LICENSE AGREEMENT
 * 2.1.1. Source Code
 * Developer shall have the right to modify and create derivative works with the Source
 * Code. Developer shall own any derivative works ("Derivatives") it creates to the Source
 * Code, provided that Developer uses the Materials in accordance with the terms and
 * conditions of this Agreement. Developer may distribute the Derivatives, provided that
 * all NVIDIA copyright notices and trademarks are propagated and used properly and
 * the Derivatives include the following statement: “This software contains source code
 * provided by NVIDIA Corporation.”
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs,
 * with an emphasis on simple illustration of the techniques (not on performance).
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <timer.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#include "simpleMultiGPU.h"

// //////////////////////////////////////////////////////////////////////////////
// Data configuration
// //////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
const int DATA_N = 48576 * 32;
char *NameToCollect = NULL;
int  RunKernel = 1;                                              // presume we run the kernel.
int  ForceInit = 0;                                              // presume we do not init.


#define CHECK_CU_ERROR(err, cufunc)                                     \
    if (err != CUDA_SUCCESS) { printf ("Error %d for CUDA Driver API function '%s'\n", err, cufunc); return -1; }

#define CHECK_CUDA_ERROR(err)                                           \
    if (err != cudaSuccess) { printf ("%s:%i Error %d for CUDA [%s]\n", __FILE__, __LINE__, err, cudaGetErrorString(err) ); return -1; }

#define CUPTI_CALL(call)                                                      \
do {                                                                          \
    CUptiResult _status = call;                                               \
    if (_status != CUPTI_SUCCESS) {                                           \
        const char *errstr;                                                   \
        cuptiGetResultString(_status, &errstr);                               \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
                __FILE__, __LINE__, #call, errstr);                           \
        exit(-1);                                                             \
    }                                                                         \
} while (0)

#define CHECK_ALLOC_ERROR(var)                                                 \
do {                                                                           \
    if (var == NULL) {                                                         \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",           \
                __FILE__, __LINE__);                                           \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

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
} // END CUDA KERNEL.

static void printUsage() {
    printf("usage: Perform a CUPTI only test of an event or metric.\n");
    printf("       EVENT_NAME      : or Metric, must be the FIRST argument.                \n");
    printf("       -noKernel       : Do all the memory work, but skip invoking the kernel. \n");
    printf("                       : It will not get the correct answer, the point is that \n");
    printf("                       : branch_efficiency (at least) will sometimes segfault  \n");
    printf("                       : if it is read with no kernel being invoked. However,  \n");
    printf("                       : if succeeds once, it will not segfault again.         \n");
    printf("       -forceInit      : Force init of cuda, and resetDevice on each device.   \n");
    printf("                       : This does not suffice to force the segfault we can get\n");
    printf("                       : with -noKernel; We have yet to find a way to force it.\n");
    printf("Note the PAPI prefix of 'cuda:::event:' or 'cuda:::metric:' should be left off,\n");
    printf("also any ':device=n' suffix. Those are PAPI added elements for disambiguation. \n");
    printf("Example: cudaTest_cupti_only branch_efficiency                                 \n");
    printf("Example: cudaTest_cupti_only branch_efficiency -nokernel                       \n");
    printf("Example: cudaTest_cupti_only branch_efficiency -nokernel -forceInit            \n");
}

void parseCommandLineArgs(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Invalid number of options\n");
        printUsage();
        exit(-1);
    }
    
    NameToCollect = argv[1];                                // Record name to collect. 
    
    if (argc < 3) return;                                   // Done if no more args.

    int i;
    for (i=2; i<argc; i++) {
        if (strcmp(argv[i], "-noKernel") == 0) {                // If we found the flag to skip kernel, 
            RunKernel= 0;                                       // .. remember it.
            continue;                                           // .. loop to next.
        }
        
        if (strcmp(argv[i], "-forceInit") == 0) {
            ForceInit = 1;
            continue;
        }

        printf("Argument ('%s') not understood.\n", argv[i]);
        printUsage();
        exit(-1);
    }
} // end routine.

//-----------------------------------------------------------------------------
// Return a text version with B, KB, MB, GB or TB. 
//-----------------------------------------------------------------------------
#define DIM(x) (sizeof(x)/sizeof(*(x)))
void calculateSize(char *result, uint64_t size)
{
    int i;

    const char *sizes[] = { "TB", "GB", "MB", "KB", "B" };
    uint64_t exbibytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;

    uint64_t multiplier = exbibytes;

    for(i = 0; (unsigned) i < DIM(sizes); i++, multiplier /= (uint64_t) 1024) {
        if(size < multiplier)
            continue;
        sprintf(result, "%.1f %s", (float) size / multiplier, sizes[i]);
        return;
    }
    strcpy(result, "0");
    return;
} // end routine


//-------------------------------------------------------------------------------------------------
// Returns the values in the event groups. Caller must know the number of events, and eventValues
// must be large enough to hold that many. eventIDArray must be large enough to hold that many 
// event IDs.
//-------------------------------------------------------------------------------------------------
void readEventGroup(CUpti_EventGroup eventGroup,
                    CUdevice dev, 
                    uint32_t numEvents,
                    CUpti_EventID *eventIdArray,
                    uint64_t *eventValues) {

    size_t bufferSizeBytes, numCountersRead;
    size_t eventIdArrayBytes= sizeof(CUpti_EventID) * numEvents;
    size_t numTotalInstancesSize = 0;
    uint64_t numTotalInstances = 0;
    uint32_t i = 0, j = 0;
    CUpti_EventDomainID domainId;
    size_t domainSize;

    domainSize = sizeof(CUpti_EventDomainID);

    CUPTI_CALL(cuptiEventGroupGetAttribute(eventGroup, 
                                           CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, 
                                           &domainSize, 
                                           (void *)&domainId));

    numTotalInstancesSize = sizeof(uint64_t);

    CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(dev, 
                                              domainId, 
                                              CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, 
                                              &numTotalInstancesSize, 
                                              (void *)&numTotalInstances));

    bufferSizeBytes = sizeof(uint64_t) * numEvents * numTotalInstances;
    uint64_t *eventValueArray = (uint64_t *) malloc(bufferSizeBytes);
    CHECK_ALLOC_ERROR(eventValueArray);

    for (i=0; i<numEvents; i++) eventValues[i]=0;                               // init the values.
    
    CUPTI_CALL(cuptiEventGroupReadAllEvents(eventGroup, 
                                            CUPTI_EVENT_READ_FLAG_NONE,
                                            &bufferSizeBytes, 
                                            eventValueArray, 
                                            &eventIdArrayBytes, 
                                            eventIdArray, 
                                            &numCountersRead));

    if (numCountersRead != numEvents) {
        if (numCountersRead > numEvents) exit(-1);
    }
    
    // Arrangement of 2-d Array returned in eventValueArray:
    //    domain instance 0: event0 event1 ... eventN
    //    domain instance 1: event0 event1 ... eventN
    //    ...
    //    domain instance M: event0 event1 ... eventN
    // But we accumulate by column, event[0], event[1], etc.

    for (i = 0; i < numEvents; i++) {                   // outer loop column traversal.
        for (j = 0; j < numTotalInstances; j++) {       // inner loop row traversal.
            eventValues[i] += eventValueArray[i + numEvents * j];
        }
    }

    free(eventValueArray);                              // Done with this.
} // end routine. 


//-------------------------------------------------------------------------------------------------
// For reading a metric. This requires a group set; we loop through the groups in the set and 
// read each one to accumulate a table of events. 
//-------------------------------------------------------------------------------------------------
void readMetricValue(CUpti_EventGroupSet *pEventGroupSet, 
                    CUdevice dev, CUpti_MetricID metricId,
                    uint64_t ns_timeDuration,
                    CUpti_MetricValue *metricValue) {
    int i;
    uint64_t *eventValues = NULL;
    CUpti_EventID *eventIDs = NULL;
    size_t sizeInt = sizeof(int);
    size_t size64  = sizeof(uint64_t);
    size_t sizeEvId= sizeof(CUpti_EventID);

    int numEventGroups = pEventGroupSet->numEventGroups;                // collect groups in the set.
    int *numGroupEvents = (int*) calloc(numEventGroups, sizeInt);       // Make room for event counts.
    int totEvents = 0;

    // Compute total number of events so we can allocate spaces.
    for (i=0; i<numEventGroups; i++) {        
        CUPTI_CALL(cuptiEventGroupGetAttribute(
            pEventGroupSet->eventGroups[i], 
            CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
            &sizeInt, &numGroupEvents[i]));                             // read # of events for this group.
        totEvents += numGroupEvents[i];                                 // Add to total.
    }

    size_t szValues =  size64  * totEvents;
    size_t szEvIds  = sizeEvId * totEvents;

    // Make space.
    eventValues = (uint64_t*) calloc(totEvents, size64);
    CHECK_ALLOC_ERROR(eventValues);

    eventIDs = (CUpti_EventID*) calloc(totEvents, sizeEvId);
    CHECK_ALLOC_ERROR(eventIDs);

    // Read each event group into its segment of the result arrays.
    int idx=0;
    for (i=0; i<numEventGroups; i++) {
        readEventGroup(
            pEventGroupSet->eventGroups[i],                             // The event group to read. 
            dev,                                                        // The device.
            numGroupEvents[i],                                          // Number of events in this group.
            &eventIDs[idx],                                             // Where to store event IDs. 
            &eventValues[idx]);                                         // Where to store event Values.

        idx += numGroupEvents[i];                                       // Add to the index.
    }

    for (i=0; i<totEvents; i++) {
        printf("   readMetricValue: EventID %lu=read %lu.\n", eventIDs[i], eventValues[i]);
    }

    CUPTI_CALL(cuptiMetricGetValue(dev, metricId,                       // Compute the metric with these values.
        szEvIds, eventIDs,                                              // events,
        szValues, eventValues,                                          // corresponding values read,
        ns_timeDuration, metricValue));                                 // duration, and storage pointer.

    free(numGroupEvents);
    free(eventValues);
    free(eventIDs);
} // end routine.


  // Print metric value, we format based on the value kind
int printMetricValue(CUpti_MetricID metricId, CUpti_MetricValue metricValue, 
        const char *metricName) {

    CUpti_MetricValueKind valueKind;
    char str[64];
    size_t valueKindSize = sizeof(valueKind);

    CUPTI_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
                                       &valueKindSize, &valueKind));
    switch (valueKind) {

    case CUPTI_METRIC_VALUE_KIND_DOUBLE:
        printf("%s = %f\n", metricName, metricValue.metricValueDouble);
        break;

    case CUPTI_METRIC_VALUE_KIND_UINT64:
        printf("%s = ", metricName);
        calculateSize(str, (uint64_t)metricValue.metricValueUint64);
        printf("%s\n", str);
        break;

    case CUPTI_METRIC_VALUE_KIND_INT64:
        printf("%s = ", metricName);
        calculateSize(str, (uint64_t)metricValue.metricValueInt64);
        printf("%s\n", str);
        break;

    case CUPTI_METRIC_VALUE_KIND_PERCENT:
        printf("%s = %.2f%%\n", metricName, metricValue.metricValueDouble);
        break;

    case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
        printf("%s = ", metricName);
        calculateSize(str, (uint64_t)metricValue.metricValueThroughput);
        printf("%s\n", str);
        break;

    case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
        printf("%s = %i\n", metricName, metricValue.metricValueUtilizationLevel);
        break;

    default:
        fflush(stdout);
        fprintf(stderr, "error: unknown value kind = %li\n", valueKind);
        return -1;                                                      // indicate failure.
    }

    return 0;                                                           // indicate success.
} // end routine.


// //////////////////////////////////////////////////////////////////////////////
// Program main
// //////////////////////////////////////////////////////////////////////////////
int main( int argc, char **argv )
{
    // Solver config
    TGPUplan plan[MAX_GPU_COUNT];
    // GPU reduction results
    float h_SumGPU[MAX_GPU_COUNT];
    float sumGPU;
    double sumCPU, diff;
    int i, j, gpuBase, GPU_N;

    const int BLOCK_N  = 32;
    const int THREAD_N = 256;
    const int ACCUM_N  = BLOCK_N * THREAD_N;

    CUcontext ctx[MAX_GPU_COUNT];
    
    printf( "Starting cudaTest_cupti_only.\n" ); fflush(stdout);

    // Parse command line arguments
    parseCommandLineArgs(argc, argv);
    
    // Report on the available CUDA devices
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    int runtimeVersion = 0, driverVersion = 0;
    char deviceName[64];
    CUdevice device[MAX_GPU_COUNT];

    if (ForceInit) {                                        // flag to force init.
        int callersDevice;                            
        CHECK_CU_ERROR(cuInit(0), "cuInit");                // Do it.
        CHECK_CUDA_ERROR(cudaGetDeviceCount(&GPU_N));       // Get the device count.
        CHECK_CUDA_ERROR(cudaGetDevice(&callersDevice));    // get the caller's current cuda device.

        for (i=0; i<GPU_N; i++) {                           // For each device, 
            CHECK_CUDA_ERROR(cudaSetDevice(i));             // .. point at device.
            CHECK_CUDA_ERROR(cudaDeviceReset());            // .. Reset it.
            printf("cudaDeviceReset accomplished on Device %i.\n", i); 
        }
        CHECK_CU_ERROR(cuInit(0), "cuInit");                // Re-init the driver.
        CHECK_CUDA_ERROR(cudaSetDevice(callersDevice));     // Back to caller's device.
    } // end optional system reset.

    CHECK_CUDA_ERROR( cudaGetDeviceCount( &GPU_N ) );
    if( GPU_N > MAX_GPU_COUNT ) GPU_N = MAX_GPU_COUNT;
    printf( "CUDA-capable device count: %i\n", GPU_N ); fflush(stdout);
    for ( i=0; i<GPU_N; i++ ) {
        CHECK_CU_ERROR( cuDeviceGet( &device[i], i ), "cuDeviceGet" );
        CHECK_CU_ERROR( cuDeviceGetName( deviceName, 64, device[i] ), "cuDeviceGetName" );
        CHECK_CU_ERROR( cuDeviceGetAttribute( &computeCapabilityMajor, 
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device[i]), "cuDeviceGetAttribute");
        CHECK_CU_ERROR( cuDeviceGetAttribute( &computeCapabilityMinor, 
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device[i]), "cuDeviceGetAttribute");
        cudaRuntimeGetVersion( &runtimeVersion );
        cudaDriverGetVersion( &driverVersion );
        printf( "CUDA Device %d: %s : computeCapability %d.%d runtimeVersion %d.%d driverVersion %d.%d\n", 
            i, deviceName, computeCapabilityMajor, computeCapabilityMinor, 
            runtimeVersion/1000, (runtimeVersion%100)/10, driverVersion/1000, (driverVersion%100)/10 );
        if ( computeCapabilityMajor < 2 ) {
            printf( "CUDA Device %d compute capability is too low... will not add any more GPUs\n", i );
            GPU_N = i;
            break;
        }
    } // end for each device.

    fflush(stdout);
    uint32_t cupti_linked_version;
    cuptiGetVersion( &cupti_linked_version );
    printf("CUPTI version: Compiled against version %d; Linked against version %d\n", 
            CUPTI_API_VERSION, cupti_linked_version );
    
    // create one context per device
    for (i = 0; i < GPU_N; i++) {
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );
        CHECK_CU_ERROR( cuCtxCreate( &(ctx[i]), 0, device[i] ), "cuCtxCreate" );
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }

    printf("Searching for '%s'.\n", NameToCollect); fflush(stdout);
    CUptiResult     myCURes;
    CUpti_EventID   eventId;
    CUpti_MetricID  metricId;
    
    int isMetric = 0;                                           // Presume this is not a metric.

    myCURes = cuptiEventGetIdFromName(0, NameToCollect, &eventId);
    if (myCURes == CUPTI_SUCCESS) {
        printf("Found '%s' as an event.\n", NameToCollect);
    } else {
        myCURes = cuptiMetricGetIdFromName(0, NameToCollect, &metricId);
        if (myCURes == CUPTI_SUCCESS) {
            isMetric = 1;                                       // remember we found a metric.
            printf("Found '%s' as a metric.\n", NameToCollect); fflush(stdout);
        } else {
            printf("'%s' not found, as event or as metric.\n", NameToCollect);
            fprintf(stderr, "%s Not Found as Event or Metric.\n", NameToCollect);
            exit(-1);
        }
    }

    printf( "Generating input data...\n" );
    
    // Subdividing input data across GPUs
    // Get data sizes for each GPU
    for( i = 0; i < GPU_N; i++ )
        plan[i].dataN = DATA_N / GPU_N;
    // Take into account "odd" data sizes
    for( i = 0; i < DATA_N % GPU_N; i++ )
        plan[i].dataN++;
    
    // Assign data ranges to GPUs
    gpuBase = 0;
    for( i = 0; i < GPU_N; i++ ) {
        plan[i].h_Sum = h_SumGPU + i; // point within h_SumGPU array
        gpuBase += plan[i].dataN;
    }

  
    // Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
    for( i = 0; i < GPU_N; i++ ) {
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CUDA_ERROR( cudaStreamCreate( &plan[i].stream ) );
        CHECK_CUDA_ERROR( cudaMalloc( ( void ** ) &plan[i].d_Data, plan[i].dataN * sizeof( float ) ) );
        CHECK_CUDA_ERROR( cudaMalloc( ( void ** ) &plan[i].d_Sum, ACCUM_N * sizeof( float ) ) );
        CHECK_CUDA_ERROR( cudaMallocHost( ( void ** ) &plan[i].h_Sum_from_device, ACCUM_N * sizeof( float ) ) );
        CHECK_CUDA_ERROR( cudaMallocHost( ( void ** ) &plan[i].h_Data, plan[i].dataN * sizeof( float ) ) );
        for( j = 0; j < plan[i].dataN; j++ ) {
            plan[i].h_Data[j] = ( float ) rand() / ( float ) RAND_MAX;
        }
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }
    
    // Create the group(s) needed to read the metric or event.
    CUpti_EventGroup eg[MAX_GPU_COUNT];                                 // event group only.
    CUpti_EventGroupSets* egs[MAX_GPU_COUNT];                           // need event group sets for metric.
    
    if (isMetric) {                                                     // If it is a metric, need a set.
        printf("Setup CUPTI counters internally for metric '%s'.\n", NameToCollect);
        for ( i=0; i<GPU_N; i++ ) {                                         // For every device, 
            CHECK_CUDA_ERROR( cudaSetDevice( i ) );
            CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
            CUPTI_CALL(cuptiSetEventCollectionMode(ctx[i], 
                CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));   // note: CONTINUOUS v. KERNEL made no difference in result.

            CUPTI_CALL(cuptiMetricCreateEventGroupSets(ctx[i], 
                sizeof(CUpti_MetricID), &metricId, &egs[i]));               // Get the pointer to sets.
            
            printf("%s Metric device %i requires %i sets, the first has %i Groups.\n", 
                NameToCollect, i, egs[i]->numSets, egs[i]->sets[0].numEventGroups);
            if (egs[i]->numSets > 1) {
                fprintf(stderr, "%s Aborted requires %i sets.\n", NameToCollect, egs[i]->numSets);
                exit(-1);
            }

            CUPTI_CALL(cuptiEventGroupSetEnable(&egs[i]->sets[0]));                 // Enable all groups in set.

            CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), 
                "cuCtxPopCurrent" );
        } // end of devices.
    } else {                                                            // If it is an event, just need one group. 
        printf("Setup CUPTI counters internally for event '%s' (CUPTI_ONLY)\n", NameToCollect);

        for ( i=0; i<GPU_N; i++ ) {                                     // For every device, 
            CHECK_CUDA_ERROR( cudaSetDevice( i ) );
            CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
            CUPTI_CALL(cuptiSetEventCollectionMode(ctx[i], 
                CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));
            CUPTI_CALL( cuptiEventGroupCreate( ctx[i], &eg[i], 0 ));
            CUPTI_CALL( cuptiEventGroupAddEvent(eg[i], eventId));
            CUPTI_CALL( cuptiEventGroupEnable( eg[i] )); 
            CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), 
                "cuCtxPopCurrent" );
        } // end of devices.
    } // end of if metric else event.
    
    // Start timing and compute on GPU(s)
    printf( "Preparing to compute with %d GPUs...\n", GPU_N ); fflush(stdout);
   
    uint64_t ns_timeDuration;                                                   // cuda device time elapsed. 
    uint64_t startTimestamp, endTimestamp;
    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));                             // We need time in ns for metrics.

    // Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (i = 0; i < GPU_N; i++) {
        // Set device
        CHECK_CUDA_ERROR( cudaSetDevice( i ));
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        // Copy input data from CPU
        CHECK_CUDA_ERROR( cudaMemcpyAsync( plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof( float ), cudaMemcpyHostToDevice, plan[i].stream ) );
        // Perform GPU computations. Sums are kept in plan[i].d_Sum.
        if (RunKernel) {                                // Can be skipped to analyze an event or metric.
            printf("Executing GPU Kernel.\n"); fflush(stdout);
            reduceKernel <<< BLOCK_N, THREAD_N, 0, plan[i].stream >>> ( plan[i].d_Sum, plan[i].d_Data, plan[i].dataN );
            if ( cudaGetLastError() != cudaSuccess ) { printf( "reduceKernel() execution failed (GPU %d).\n", i ); exit(EXIT_FAILURE); }
        } else {
            printf("Skipping execution of GPU Kernel (expect zeros for result).\n"); fflush(stdout);
        }

        // Read back GPU results; from plan[i].d_Sum into plan[i].h_Sum_from_device.
        CHECK_CUDA_ERROR( cudaMemcpyAsync( plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N * sizeof( float ), cudaMemcpyDeviceToHost, plan[i].stream ) );
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }
    
    // Process GPU results
    printf( "Process GPU results on %d GPUs...\n", GPU_N ); fflush(stdout);
    for( i = 0; i < GPU_N; i++ ) {
        float sum;
        // Set device
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        // Wait for all operations to finish
        cudaStreamSynchronize( plan[i].stream );
        // Finalize GPU reduction for current subvector
        sum = 0;
        for( j = 0; j < ACCUM_N; j++ ) {
            sum += plan[i].h_Sum_from_device[j];
        }

        if (!RunKernel) sum = 0.0;                  // Don't take NAN if we didn't run the kernel.
        *( plan[i].h_Sum ) = ( float ) sum;
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }

    CUPTI_CALL(cuptiGetTimestamp(&endTimestamp));
    ns_timeDuration = endTimestamp - startTimestamp;

    double gpuTime = (ns_timeDuration/((double) 1000000.0));                    // convert to ms.

    // Now, we must read the metric/event. 
    size_t size = 1024;
    uint64_t buffer[size];

    for ( i=0; i<GPU_N; i++ ) {                                                 // for each device,
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );                                 // point at it.
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CU_ERROR( cuCtxSynchronize( ), "cuCtxSynchronize" );              // wait for all to finish.

        if (isMetric) {                                                         // If we have a metric,
            fprintf(stderr, "Reading metric %s.\n", NameToCollect);
            CUpti_MetricValue metricValue;
            readMetricValue(&(egs[i]->sets[0]),                                 // Pointer to the single set.
            device[i], metricId,
            ns_timeDuration, &metricValue);
            printf("Device %i, Metric: ",i);                                    // prefix the printing...
            printMetricValue(metricId, metricValue, NameToCollect);             // Print "name = value\n".
        } else {                                                                // If we have just an event.
            readEventGroup(eg[i], device[i], 
                1, &eventId,                                                    // just 1 event.
                &buffer[i]);
            printf( "CUPTI %s device %d counterValue %u (on one domain, "
                    "may need to be multiplied by num of domains)\n", 
                    NameToCollect, i, buffer[i] );
        } // end if metric else event.

        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }

    sumGPU = 0;
    for( i = 0; i < GPU_N; i++ ) {
        sumGPU += h_SumGPU[i];
    }
    printf( "GPU Processing time (includes setup and memcpy overheads): %f (ms)\n", gpuTime );

    // Compute on Host CPU
    printf( "Computing the same result with Host CPU...\n" );
    StartTimer();
    sumCPU = 0;
    for( i = 0; i < GPU_N; i++ ) {
        for( j = 0; j < plan[i].dataN; j++ ) {
            sumCPU += plan[i].h_Data[j];
        }
    }
    double cpuTime = GetTimer();
    if (gpuTime > 0) {
        printf( "  CPU Processing time: %f (ms) (speedup %.2fX)\n", cpuTime, (cpuTime/gpuTime) );
    } else {
        printf( "  CPU Processing time: %f (ms)\n", cpuTime);
    }

    // Compare GPU and CPU results
    printf( "Comparing GPU and Host CPU results...\n" );
    diff = fabs( sumCPU - sumGPU ) / fabs( sumCPU );
    printf( "  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU );
    printf( "  Relative difference: %E \n", diff );

    // Cleanup and shutdown
    for( i = 0; i < GPU_N; i++ ) {
        CHECK_CUDA_ERROR( cudaSetDevice(i) );
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CUDA_ERROR( cudaStreamSynchronize(plan[i].stream) ); 
        CHECK_CUDA_ERROR( cudaFreeHost( plan[i].h_Sum_from_device ) );
        CHECK_CUDA_ERROR( cudaFree( plan[i].d_Sum ) );
        CHECK_CUDA_ERROR( cudaFree( plan[i].d_Data ) );
        CHECK_CUDA_ERROR( cudaStreamDestroy( plan[i].stream ) );
        CHECK_CUDA_ERROR( cudaFreeHost( plan[i].h_Data ) );
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }

    exit( ( diff < 1e-5 ) ? EXIT_SUCCESS : EXIT_FAILURE );
}

