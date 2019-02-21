/* 
 * Copyright 2015-2016 NVIDIA Corporation. All rights reserved.
 *
 * Sample to demonstrate use of NVlink CUPTI APIs
 * 
 * This version is significantly changed to use PAPI and the CUDA component to
 * handle access and reporting. As of 10/05/2018, I have deleted all CUPTI_ONLY
 * references, for clarity. The file nvlink_bandwidth_cupti_only.cu contains
 * the cupti-only code.  I also deleted the #if PAPI; there is no option
 * without PAPI.  Also, before my changes, the makefile did not even have a
 * build option that set CUPTI_ONLY for this file.
 *
 * -TonyC. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cupti.h>
#include "papi.h"

// THIS MACRO EXITS if the papi call does not return PAPI_OK. Do not use for routines that
// return anything else; e.g. PAPI_num_components, PAPI_get_component_info, PAPI_library_init.
#define CALL_PAPI_OK(papi_routine)                                                        \
    do {                                                                                  \
        int _papiret = papi_routine;                                                      \
        if (_papiret != PAPI_OK) {                                                        \
            fprintf(stderr, "%s:%d: PAPI Error: function %s failed with ret=%d [%s].\n",  \
                    __FILE__, __LINE__, #papi_routine, _papiret, PAPI_strerror(_papiret));\
            exit(-1);                                                                     \
        }                                                                                 \
    } while (0);


#define CUPTI_CALL(call)                                                \
    do {                                                                \
        CUptiResult _status = call;                                     \
        if (_status != CUPTI_SUCCESS) {                                 \
            const char *errstr;                                         \
            cuptiGetResultString(_status, &errstr);                     \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                 \
            exit(-1);                                                   \
        }                                                               \
    } while (0);  

#define DRIVER_API_CALL(apiFuncCall)                                    \
    do {                                                                \
        CUresult _status = apiFuncCall;                                 \
        if (_status != CUDA_SUCCESS) {                                  \
            const char *errName=NULL, *errStr=NULL;                     \
            CUresult _e1 = cuGetErrorName(_status, &errName);           \
            CUresult _e2 = cuGetErrorString(_status, &errStr);          \
            fprintf(stderr, "%s:%d: error: function %s failed with error %d [%s]='%s'.\n", \
                    __FILE__, __LINE__, #apiFuncCall, _status, errName, errStr);           \
            exit(-1);                                                   \
        }                                                               \
    } while (0);  

#define RUNTIME_API_CALL(apiFuncCall)                                   \
    do {                                                                \
        cudaError_t _status = apiFuncCall;                              \
        if (_status != cudaSuccess) {                                   \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
            exit(-1);                                                   \
        }                                                               \
    } while (0);  

#define MEMORY_ALLOCATION_CALL(var)                                     \
    do {                                                                \
        if (var == NULL) {                                              \
            fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n", \
                    __FILE__, __LINE__);                                \
            exit(-1);                                                   \
        }                                                               \
    } while (0);  


#define MAX_DEVICES    (32)
#define BLOCK_SIZE     (1024)
#define GRID_SIZE      (512)
#define BUF_SIZE       (32 * 1024)
#define ALIGN_SIZE     (8)
#define SUCCESS        (0)
#define MAX_SIZE       (64*1024*1024)   // 64 MB

int Streams;                            // Number of physical copy engines to use; taken from Device Properties asyncEngineCount.
int cpuToGpu = 0;
int gpuToGpu = 0;


//-----------------------------------------------------------------------------
// This is the GPU routine to move a block from dst (on one GPU) to src (on
// another GPU. This is no longer used in this code; we use cudaMemcpyAsync().
// Typical invocation (depends on #defines above):
//
//  for(i = 0; i < Streams; i++) {
//      test_nvlink_bandwidth <<< GRID_SIZE, BLOCK_SIZE >>> ((float *) pDevBuffer1[i], (float *) pDevBuffer0[i]);
//      printf("test_nvlink_bandwidth stream %d \n", i);
//  }
//-----------------------------------------------------------------------------
extern "C" __global__ void test_nvlink_bandwidth(float *src, float *dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2) dst[idx] = src[idx] * 2.0f;
    else         dst[idx] = src[idx] * 1.5f;
//  dst[idx] = src[idx] * 2.0f;
} // end routine

#define DIM(x) (sizeof(x)/sizeof(*(x)))


//-----------------------------------------------------------------------------
// Return a text version with B, KB, MB, GB or TB. 
//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
// We use Async copies (returns while operation is still in progress) with 
// multiple streams; cudaDeviceSynchronize waits for them to complete.
//-----------------------------------------------------------------------------
void testCpuToGpu(CUpti_EventGroup * eventGroup, 
      CUdeviceptr * pDevBuffer, float **pHostBuffer, size_t bufferSize, 
      cudaStream_t * cudaStreams, uint64_t * timeDuration, 
      int numEventGroup)
{
    int i;
    fprintf(stderr, "Streams = %d.\n", Streams); 
    // Unidirectional copy H2D (Host to Device).
    for(i = 0; i < Streams; i++) {
//      RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
        test_nvlink_bandwidth <<< GRID_SIZE, BLOCK_SIZE >>> ((float *) pDevBuffer[i], (float *) pHostBuffer[i]);
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Unidirectional copy D2H (Device to Host).
    for(i = 0; i < Streams; i++) {
//      RUNTIME_API_CALL(cudaMemcpyAsync(pHostBuffer[i], (void *) pDevBuffer[i], bufferSize, cudaMemcpyDeviceToHost, cudaStreams[i]));
        test_nvlink_bandwidth <<< GRID_SIZE, BLOCK_SIZE >>> ((float *) pHostBuffer[i], (float *) pDevBuffer[i]);
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Bidirectional copy
    for(i = 0; i < Streams; i += 2) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
        RUNTIME_API_CALL(cudaMemcpyAsync(pHostBuffer[i + 1], (void *) pDevBuffer[i + 1], bufferSize, cudaMemcpyDeviceToHost, cudaStreams[i + 1]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());
} // end routine.


//-----------------------------------------------------------------------------
// Copy buffers from the host to each device, in preparation for a transfer
// between devices.
// We use Async copies (returns while operation is still in progress) with 
// multiple streams; cudaDeviceSynchronize waits for them to complete.
//-----------------------------------------------------------------------------
void testGpuToGpu_part1(CUpti_EventGroup * eventGroup, 
      CUdeviceptr * pDevBuffer0, CUdeviceptr * pDevBuffer1, 
      float **pHostBuffer, size_t bufferSize, 
      cudaStream_t * cudaStreams, uint64_t * timeDuration, 
      int numEventGroup)
{
    int i;

    RUNTIME_API_CALL(cudaSetDevice(0));
    RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(1, 0));
    RUNTIME_API_CALL(cudaSetDevice(1));
    RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(0, 0));

    // Unidirectional copy H2D
    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer0[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
    }

    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer1[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
    }

    RUNTIME_API_CALL(cudaDeviceSynchronize());
} // end routine.


//-----------------------------------------------------------------------------
// Copy from device zero to device 1, then from device 1 to device 0.
//-----------------------------------------------------------------------------
void testGpuToGpu_part2(CUpti_EventGroup * eventGroup, 
      CUdeviceptr * pDevBuffer0, CUdeviceptr * pDevBuffer1, 
      float **pHostBuffer, size_t bufferSize, 
      cudaStream_t * cudaStreams, uint64_t * timeDuration, 
      int numEventGroup)
{
    int i;

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer0[i], (void *) pDevBuffer1[i], bufferSize, cudaMemcpyDeviceToDevice, cudaStreams[i]));
        printf("Copy %zu stream %d to devBuffer0 from devBuffer1 \n", bufferSize, i);
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer1[i], (void *) pDevBuffer0[i], bufferSize, cudaMemcpyDeviceToDevice, cudaStreams[i]));
        printf("Copy %zu stream %d to devBuffer1 from devBuffer0 \n", bufferSize, i);
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());
} // end routine.


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
static void printUsage()
{
    printf("usage: Demonstrate use of NVlink CUPTI APIs\n");
    printf("       -help           : display help message\n");
    printf("       --cpu-to-gpu    : Show results for data transfer between CPU and GPU \n");
    printf("       --gpu-to-gpu    : Show results for data transfer between two GPUs \n");
} // end routine.


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void parseCommandLineArgs(int argc, char *argv[])
{
    if(argc != 2) {
        printf("Invalid number of options\n");
        exit(0);
    }

    if(strcmp(argv[1], "--cpu-to-gpu") == 0) {
        cpuToGpu = 1;
    } else if(strcmp(argv[1], "--gpu-to-gpu") == 0) {
        gpuToGpu = 1;
    } else if((strcmp(argv[1], "--help") == 0) || 
              (strcmp(argv[1], "-help") == 0)  || 
              (strcmp(argv[1], "-h") == 0)) {
        printUsage();
        exit(0);
    } else {
        cpuToGpu = 1;
    }
} // end routine.


//-----------------------------------------------------------------------------
// Main program.
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int deviceCount = 0, i = 0, numEventGroup = 0, eventsRead=0;
    size_t bufferSize = 0, freeMemory = 0, totalMemory = 0;
    char str[64];

    CUdeviceptr *pDevBuffer0  = NULL;
    CUdeviceptr *pDevBuffer1  = NULL;
    float       **pHostBuffer = NULL;

    cudaStream_t *cudaStreams = NULL;
    cudaDeviceProp prop[MAX_DEVICES];
    uint64_t timeDuration;
    CUpti_EventGroup eventGroup[32];

    // This contains the original set of 'bandwidth' events to read.  This
    // program attempts to read them together within a single EventSet. PAPI
    // allows them all, and the component does not complain, but it always
    // returns 0 for the two "receive" events below. That is not the result
    // when we read each of these separately for the same program; which is the
    // approach taken in the program nvlink_all.cu. I suspect nvlink has a
    // conflict of some sort, but haven't tracked down the documentation to
    // prove that.  -Tony C.

#define NUM_METRIC     ( 4)
    const char *MetricBase[NUM_METRIC] = {
        "cuda:::metric:nvlink_total_data_transmitted"        , // okay Group NVLINK.
        "cuda:::metric:nvlink_transmit_throughput"           , // okay Group NVLINK.
        "cuda:::metric:nvlink_total_data_received"           , // okay Group NVLINK.
        "cuda:::metric:nvlink_receive_throughput"            , // okay Group NVLINK.
//      "cuda:::metric:inst_per_warp"                        , // okay group A.
//      "cuda:::metric:warp_execution_efficiency"            , // okay Group A.
//      "cuda:::metric:warp_nonpred_execution_efficiency"    , // okay Group A.
//      "cuda:::metric:shared_load_transactions_per_request" , // okay Group A.
//      "cuda:::metric:shared_store_transactions_per_request", // okay Group A.
//      "cuda:::metric:shared_store_transactions"            , // okay Group A.
//      "cuda:::metric:shared_load_transactions"             , // okay Group A.
//      "cuda:::metric:inst_replay_overhead"                 , // Group B
//      "cuda:::metric:local_load_transactions"              , // Group B.
//      "cuda:::metric:local_load_transactions_per_request"  , // Group NONE. Bad Combo, even by itself requires 2 passes.
//      "cuda:::metric:local_store_transactions_per_request" , // Group NONE. Bad Combo, even by itself.
//      "cuda:::metric:gld_transactions_per_request"         , // Group NONE. Bad Combo, even by itself.
//      "cuda:::metric:gst_transactions_per_request"         , // Group NONE. Bad Combo, even by itself.
//      "cuda:::event:active_cycles"                         ,
//      "cuda:::event:active_warps"                          ,
//      "cuda:::event:active_cycles"                         ,
//      "cuda:::event:active_warps"                          ,
//      "cuda:::event:inst_executed"                         ,
//      "cuda:::event:warps_launched"                        ,
//      "cuda:::metric:branch_efficiency"                    , // Even by itself, causes signal 11 (seg fault) on SECOND read.
    };
    
    // Parse command line arguments
    parseCommandLineArgs(argc, argv);
    if (cpuToGpu) printf("TEST: CPU to GPU transfer.\n");
    else          printf("TEST: GPU to GPU transfer.\n");

    DRIVER_API_CALL(cuInit(0));
    RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));
    printf("There are %d devices.\n", deviceCount);

    if(deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(-1);
    }

    Streams = 1;                                            // Always use at least ONE stream.
    for(i = 0; i < deviceCount; i++) {
        RUNTIME_API_CALL(cudaGetDeviceProperties(&prop[i], i));
        printf("CUDA Device %d Name: %s", i, prop[i].name);
        printf(", AsyncEngineCount=%i", prop[i].asyncEngineCount);
        printf(", MultiProcessors=%i", prop[i].multiProcessorCount);
        printf(", MaxThreadsPerMP=%i", prop[i].maxThreadsPerMultiProcessor);
        printf("\n");
        if (prop[i].asyncEngineCount > Streams) {           // If a new high,
            Streams = prop[i].asyncEngineCount;             // Always use the maximum.
        }
    }

    printf("Streams to use: %i (= max Copy Engines).\n", Streams);

    // allocate space
    pDevBuffer0 = (CUdeviceptr*)    calloc(Streams, sizeof(CUdeviceptr));
    pDevBuffer1 = (CUdeviceptr*)    calloc(Streams, sizeof(CUdeviceptr));
    pHostBuffer = (float **)        calloc(Streams, sizeof(float*));
    cudaStreams = (cudaStream_t*)   calloc(Streams, sizeof(cudaStream_t));

    // Set memcpy size based on available device memory
    RUNTIME_API_CALL(cudaMemGetInfo(&freeMemory, &totalMemory));
    printf("Total Device Memory available : ");
    calculateSize(str, (uint64_t) totalMemory);
    printf("%s\n", str);

    bufferSize = MAX_SIZE < (freeMemory / 4) ? MAX_SIZE : (freeMemory / 4);
    bufferSize = bufferSize/2;
    printf("Memcpy size is set to %llu B (%llu MB)\n", (unsigned long long) bufferSize, (unsigned long long) bufferSize / (1024 * 1024));

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaStreamCreate(&cudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Nvlink-topology Records are generated even before cudaMemcpy API is called.
    CUPTI_CALL(cuptiActivityFlushAll(0x7fffffff)); // flag covers every kind of record.

    fprintf(stderr, "Setup PAPI counters internally (PAPI)\n");
    int EventSet = PAPI_NULL;
    long long values[MAX_DEVICES * NUM_METRIC];
    char *EventName[MAX_DEVICES * NUM_METRIC];
    int eventCount;
    int retval, ee;
    int k, cid=-1;

    /* PAPI Initialization */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if(retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed, ret=%i [%s]\n", 
            retval, PAPI_strerror(retval));
        exit(-1);
    }

    fprintf(stderr, "PAPI version: %d.%d.%d\n", 
        PAPI_VERSION_MAJOR(PAPI_VERSION), 
        PAPI_VERSION_MINOR(PAPI_VERSION), 
        PAPI_VERSION_REVISION(PAPI_VERSION));

    // Find cuda component index.
    k = PAPI_num_components();                                          // get number of components.
    for (i=0; i<k && cid<0; i++) {                                      // while not found,
        PAPI_component_info_t *aComponent = 
            (PAPI_component_info_t*) PAPI_get_component_info(i);        // get the component info.     
        if (aComponent == NULL) {                                       // if we failed,
            fprintf(stderr,  "PAPI_get_component_info(%i) failed, "
                "returned NULL. %i components reported.\n", i,k);
            exit(-1);    
        }

       if (strcmp("cuda", aComponent->name) == 0) cid=i;                // If we found our match, record it.
    } // end search components.

    if (cid < 0) {                                                      // if no PCP component found,
        fprintf(stderr, "Failed to find pcp component among %i "
            "reported components.\n", k);
        exit(-1); 
    }

    fprintf(stderr, "Found CUDA Component at id %d\n",cid);

    CALL_PAPI_OK(PAPI_create_eventset(&EventSet)); 
    CALL_PAPI_OK(PAPI_assign_eventset_component(EventSet, cid)); 

    // ===== Allocate Memory =====================================

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMalloc((void **) &pDevBuffer0[i], bufferSize));

        pHostBuffer[i] = (float *) malloc(bufferSize);
        MEMORY_ALLOCATION_CALL(pHostBuffer[i]);
    }
    
    // Add events at a GPU specific level ... eg cuda:::metric:nvlink_total_data_transmitted:device=0
    char tmpEventName[1024];
    eventCount = 0;
    for(i = 0; i < deviceCount; i++) {                                  // Profile all devices.
        fprintf(stderr, "Set device to %d\n", i);
        for(ee = 0; ee < NUM_METRIC; ee++) {
            snprintf(tmpEventName, 1024, "%s:device=%d\0", MetricBase[ee], i);
            retval = PAPI_add_named_event(EventSet, tmpEventName);      // Don't want to fail program if name not found...
            if(retval == PAPI_OK) {
                EventName[eventCount] = strdup(tmpEventName);
                eventCount++;
            } else {
                fprintf(stderr, "Failed to add event %s to GPU %i; ret=%d [%s].\n", tmpEventName, i, retval, PAPI_strerror(retval));
            }
        }
    }

    if (eventCount > 0) {                                              // If we have events...
        for(i = 0; i < eventCount; i++) values[i] = -1;                // init.

        if(cpuToGpu) {
            RUNTIME_API_CALL(cudaSetDevice(1));
            for(i = 0; i < Streams; i++) 
                RUNTIME_API_CALL(cudaMalloc((void **) &pDevBuffer1[i], bufferSize));
            CALL_PAPI_OK(PAPI_start(EventSet));                             // Start event counters.
            RUNTIME_API_CALL(cudaSetDevice(0));
            testCpuToGpu(eventGroup, pDevBuffer0, pHostBuffer, bufferSize, cudaStreams, &timeDuration, numEventGroup);
            RUNTIME_API_CALL(cudaSetDevice(1));
            testCpuToGpu(eventGroup, pDevBuffer1, pHostBuffer, bufferSize, cudaStreams, &timeDuration, numEventGroup);
            RUNTIME_API_CALL(cudaSetDevice(0));
            CALL_PAPI_OK(PAPI_stop(EventSet, values));                      // Stop and read values.
        } else if(gpuToGpu) {
            RUNTIME_API_CALL(cudaSetDevice(1));
            for(i = 0; i < Streams; i++) 
                RUNTIME_API_CALL(cudaMalloc((void **) &pDevBuffer1[i], bufferSize));

            //  Prepare the copy, load up buffers on each device from the host.
            testGpuToGpu_part1(eventGroup, pDevBuffer0, pDevBuffer1, pHostBuffer, bufferSize, cudaStreams, &timeDuration, numEventGroup);

            // Copy from device 0->1, then device 1->0.
            CALL_PAPI_OK(PAPI_start(EventSet));                             // Start event counters.
            testGpuToGpu_part2(eventGroup, pDevBuffer0, pDevBuffer1, pHostBuffer, bufferSize, cudaStreams, &timeDuration, numEventGroup);
            CALL_PAPI_OK(PAPI_stop(EventSet, values));                      // Stop and read values.
        }

        // report each event counted.
        for(i = 0; i < eventCount; i++) {
            if (values[i] >= 0) {                                           // If not still -1,
                eventsRead++;                                               // .. count and report.
                calculateSize(str, (uint64_t) values[i] );
                printf("PAPI %64s: %s \n", EventName[i], str);
            }
        }
    }

    // Program cleanup.
    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // Delete all events in set.
    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // Release PAPI memory.
    PAPI_shutdown();                                                    // Has no return.
        
    if (eventsRead > 0) {                                               // If we succeeded with any, report. 
        printf("%i bandwidth events successfully reported.\n", eventsRead);
        return(0);                                                      // exit OK.
    }

    printf("Failed to read any bandwidth events.\n");                   // report a failure.
        
    return (-1);                                                        // Exit with error.
} // end MAIN.
