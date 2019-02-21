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
            fprintf(stderr, "%s:%d macro: PAPI Error: function " #papi_routine " failed with ret=%d [%s].\n", \
                    __FILE__, __LINE__, _papiret, PAPI_strerror(_papiret));               \
            exit(-1);                                                                     \
        }                                                                                 \
    } while (0);


#define CUPTI_CALL(call)                                                \
    do {                                                                \
        CUptiResult _status = call;                                     \
        if (_status != CUPTI_SUCCESS) {                                 \
            const char *errstr;                                         \
            cuptiGetResultString(_status, &errstr);                     \
            fprintf(stderr, "%s:%d: error: function %s failed with message '%s'.\n", \
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
            fprintf(stderr, "%s:%d: error: function %s failed with error [%s]='%s'.\n", \
                    __FILE__, __LINE__, #apiFuncCall, errName, errStr); \
            exit(-1);                                                   \
        }                                                               \
    } while (0);  

#define RUNTIME_API_CALL(apiFuncCall)                                   \
    do {                                                                \
        cudaError_t _status = apiFuncCall;                              \
        if (_status != cudaSuccess) {                                   \
            fprintf(stderr, "%s:%d: error: function %s failed with message'%s'.\n", \
                    __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
            exit(-1);                                                   \
        }                                                               \
    } while (0);  

#define MEMORY_ALLOCATION_CALL(var)                                     \
    do {                                                                \
        if (var == NULL) {                                              \
            fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",\
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
#define NUM_METRIC     (18)
#define NUM_EVENTS     (2)
#define MAX_SIZE       (64*1024*1024)   // 64 MB

typedef union
{
    long long ll;
    unsigned long long ull;
    double    d;
    void *vp;
    unsigned char ch[8];
} convert_64_t;

typedef struct {
    char name[128];
    long long value;
} eventStore_t;

int eventsFoundCount = 0;               // occupants of the array.
int eventsFoundMax;                     // Size of the array.
int eventsFoundAdd = 32;                // Blocksize for increasing the array.
eventStore_t *eventsFound = NULL;       // The array.

int Streams;                            // Gets asyncEngineCount (number of physical copy engines).
int cpuToGpu = 0;
int gpuToGpu = 0;
size_t bufferSize = 0;

int         *deviceEvents = NULL;
CUdeviceptr *pDevBuffer0  = NULL;
CUdeviceptr *pDevBuffer1  = NULL;
float       **pHostBuffer = NULL;
cudaStream_t *cudaStreams = NULL;

//-----------------------------------------------------------------------------
// This is the GPU routine to move a block from 'source' (on one GPU) to 'dest'
// on another GPU. 
//-----------------------------------------------------------------------------
extern "C" __global__ void test_nvlink_bandwidth(float *source, float *dest)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dest[idx] = source[idx] * 2.0f;
} // end routine

#define DIM(x) (sizeof(x)/sizeof(*(x))) /* compute elements in an array */


//-----------------------------------------------------------------------------
// FreeGlobals: Frees globally allocated memories.
//-----------------------------------------------------------------------------
void FreeGlobals(void) 
{
    int i;
    free(deviceEvents);
    
    for(i=0; i<Streams; i++) {
        RUNTIME_API_CALL(cudaSetDevice(0));                     // device 0 for pDevBuffer0. 
        RUNTIME_API_CALL(cudaFree((void **) &pDevBuffer0[i]));  // Free allocated space.
        free(pHostBuffer[i]);                                   // Just locally allocateed.
    }

    free(pDevBuffer0);              // all contents freed by above.
    free(pHostBuffer);              // Free the pointers.
    free(pDevBuffer1);              // contents freed by the way the tests work.
    for (i=0; i<Streams; i++) {     // Destroy all streams.
        if (cudaStreams[i] != NULL) {
            RUNTIME_API_CALL(cudaStreamDestroy(cudaStreams[i]));
        }
    }
    
    free(cudaStreams);              // Free the memory for pointers.
} // end routine.


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
// Copy buffers from host to device, vice versa, both simultaneously.
//-----------------------------------------------------------------------------
void testCpuToGpu(CUdeviceptr * pDevBuffer, float **pHostBuffer, size_t bufferSize, 
      cudaStream_t * cudaStreams)
{
    int i;
    // Unidirectional copy H2D (Host to Device).
    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Unidirectional copy D2H (Device to Host).
    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync(pHostBuffer[i], (void *) pDevBuffer[i], bufferSize, cudaMemcpyDeviceToHost, cudaStreams[i]));
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
// Copy buffers from the host to each device, in preperation for a transfer
// between devices.
//-----------------------------------------------------------------------------
void testGpuToGpu_part1(CUdeviceptr * pDevBuffer0, CUdeviceptr * pDevBuffer1, 
      float **pHostBuffer, size_t bufferSize, cudaStream_t * cudaStreams) 
{
    int i;

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
void testGpuToGpu_part2(CUdeviceptr * pDevBuffer0, CUdeviceptr * pDevBuffer1, 
      float **pHostBuffer, size_t bufferSize, cudaStream_t * cudaStreams) 
{
    int i;

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer0[i], (void *) pDevBuffer1[i], bufferSize, cudaMemcpyDeviceToDevice, cudaStreams[i]));
        //printf("Copy %zu stream %d to devBuffer0 from devBuffer1 \n", bufferSize, i);
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *) pDevBuffer1[i], (void *) pDevBuffer0[i], bufferSize, cudaMemcpyDeviceToDevice, cudaStreams[i]));
        // printf("Copy %zu stream %d to devBuffer0 from devBuffer1 \n", bufferSize, i);
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for(i = 0; i < Streams; i++) {
        test_nvlink_bandwidth <<< GRID_SIZE, BLOCK_SIZE >>> ((float *) pDevBuffer1[i], (float *) pDevBuffer0[i]);
        // printf("test_nvlink_bandwidth stream %d \n", i);
    }
} // end routine.


//-----------------------------------------------------------------------------
// conducts test CpuToGpu. This is mostly a shortcut for readability, 
// decisions must be made about the device buffers.
//-----------------------------------------------------------------------------
void conductCpuToGpu(int EventSet, int device, long long *values) {
    int i;
    if (device == 0) { 
        CALL_PAPI_OK(PAPI_start(EventSet));                         // Start event counters.
        testCpuToGpu(pDevBuffer0, pHostBuffer, bufferSize, 
            cudaStreams);
    } else {
        RUNTIME_API_CALL(cudaSetDevice(device));
        for(i = 0; i < Streams; i++) {
            RUNTIME_API_CALL(cudaMalloc((void **) &pDevBuffer1[i], bufferSize));
        }

        CALL_PAPI_OK(PAPI_start(EventSet));                         // Start event counters.
        testCpuToGpu(pDevBuffer1, pHostBuffer, bufferSize, 
            cudaStreams);

        for (i=0; i<Streams; i++) {
            RUNTIME_API_CALL(cudaFree((void **) pDevBuffer1[i]));
        }
    } // end testing device other than 0.

    CALL_PAPI_OK(PAPI_stop(EventSet, values));                      // Stop and read any values.
} // end routine. 



//-----------------------------------------------------------------------------
// conducts test GpuToGpu. This is mostly a shortcut for readability, 
// decisions must be made about the device buffers.
//-----------------------------------------------------------------------------
void conductGpuToGpu(int EventSet, int device, long long *values) {
    int i;
    // Need to target another GPU. I already have pDevBuffer0 on device 0.
    int partner=device;                                         // Presume event is not on zero.
    if (device == 0) partner=1;                                 // If it is on zero, make partner 1.

    RUNTIME_API_CALL(cudaSetDevice(0));                         // Device 0 must 
    RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(partner, 0));   // access partner.  
    
    RUNTIME_API_CALL(cudaSetDevice(partner));                   // The partner device must access 0.
    RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(0, 0));         // Let non-zero device access 0.

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMalloc((void **) &pDevBuffer1[i], bufferSize));
    }

    //  Prepare the copy, load up buffers on each device from the host.
    testGpuToGpu_part1(pDevBuffer0, pDevBuffer1, pHostBuffer, 
        bufferSize, cudaStreams);

    // What we want to time: Copy from device 0->1, then device 1->0.
    CALL_PAPI_OK(PAPI_start(EventSet));                         // Start event counters.
    testGpuToGpu_part2(pDevBuffer0, pDevBuffer1, pHostBuffer, 
         bufferSize, cudaStreams);
    CALL_PAPI_OK(PAPI_stop(EventSet, values));                  // Stop and read value.

    // Disable peer access.
    RUNTIME_API_CALL(cudaSetDevice(0));
    RUNTIME_API_CALL(cudaDeviceDisablePeerAccess(partner)); // Kill connection to device i.
    
    RUNTIME_API_CALL(cudaSetDevice(partner));
    RUNTIME_API_CALL(cudaDeviceDisablePeerAccess(0));       // Kill access to device 0.

    // Now free the pointers on device 'partner' (never 0). 
    for (i=0; i<Streams; i++) {
        RUNTIME_API_CALL(cudaFree((void **) pDevBuffer1[i]));
    }

    RUNTIME_API_CALL(cudaSetDevice(0));                     // return to default pointer.
} // end routine.


//-----------------------------------------------------------------------------
// Show help.
//-----------------------------------------------------------------------------
static void printUsage()
{
    printf("usage: Demonstrate use of NVlink CUPTI APIs\n");
    printf("       -help           : display help message\n");
    printf("       --cpu-to-gpu    : Show results for data transfer between CPU and GPU \n");
    printf("       --gpu-to-gpu    : Show results for data transfer between two GPUs \n");
} // end routine.


//-----------------------------------------------------------------------------
// Interpret command line flags.
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
// Add an entry to the eventsFound array. On entry we always have room.
//-----------------------------------------------------------------------------
void addEventsFound(char *eventName, long long value) {
    strncpy(eventsFound[eventsFoundCount].name, eventName, 127);    // Copy up to 127 chars.
    eventsFound[eventsFoundCount].value = value;                    // Copy the value.

    if (++eventsFoundCount >= eventsFoundMax) {                     // bump count, if too much, make room.
        eventsFoundMax += eventsFoundAdd;                           // Add.
        eventsFound = (eventStore_t*) realloc(eventsFound, eventsFoundMax*sizeof(eventStore_t));    // Make new room.
        memset(eventsFound+(eventsFoundMax-eventsFoundAdd), 0, eventsFoundAdd*sizeof(eventStore_t));    // zero it.
    }
} // end routine.

//-----------------------------------------------------------------------------
// Main program.
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int device, deviceCount = 0, i = 0;
    size_t freeMemory = 0, totalMemory = 0;
    char str[64];

    eventsFoundMax = eventsFoundAdd;                            // space allocated.
    eventsFound = (eventStore_t*) calloc(eventsFoundMax, sizeof(eventStore_t)); // make some space.

    cudaDeviceProp prop[MAX_DEVICES];

    // Parse command line arguments
    parseCommandLineArgs(argc, argv);

    DRIVER_API_CALL(cuInit(0));
    RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));
    printf("There are %d devices.\n", deviceCount);

    if(deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(-1);
    }

    Streams = 1;                                            // Always use at least ONE stream.
    for(device = 0; device < deviceCount; device++) {
        RUNTIME_API_CALL(cudaGetDeviceProperties(&prop[device], device));
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
    deviceEvents= (int*)            calloc(deviceCount, sizeof(int));
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
    CUPTI_CALL(cuptiActivityFlushAll(0));

    // fprintf(stderr, "Setup PAPI counters internally (PAPI)\n");
    int EventSet = PAPI_NULL;
    int eventCount;
    int retval;
    int k, m, cid=-1;

    /* PAPI Initialization */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if(retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed, ret=%i [%s]\n", 
            retval, PAPI_strerror(retval));
        FreeGlobals();
        exit(-1);
    }

    printf("PAPI version: %d.%d.%d\n", 
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
            FreeGlobals();
            exit(-1);    
        }

       if (strcmp("cuda", aComponent->name) == 0) cid=i;                // If we found our match, record it.
    } // end search components.

    if (cid < 0) {                                                      // if no PCP component found,
        fprintf(stderr, "Failed to find cuda component among %i "
            "reported components.\n", k);
        FreeGlobals();
        PAPI_shutdown();
        exit(-1); 
    }

    printf("Found CUDA Component at id %d\n", cid);

    // Add events at a GPU specific level ... eg cuda:::metric:nvlink_total_data_transmitted:device=0
    // Just profile devices to match the CUPTI example
    eventCount = 0;
    int eventsRead=0;

    for(i = 0; i < Streams; i++) {
        RUNTIME_API_CALL(cudaMalloc((void **) &pDevBuffer0[i], bufferSize));
    
        pHostBuffer[i] = (float *) malloc(bufferSize);
        MEMORY_ALLOCATION_CALL(pHostBuffer[i]);
    }
            
   // Begin enumeration of all events.
   if (cpuToGpu) printf("Experiment timing memory copy from host to GPU.\n");
   if (gpuToGpu) printf("Experiment timing memory copy between GPU 0 and each other GPU.\n");

   printf("Events with numeric values were read; if they are zero, they may not  \n"
          "be operational, or the exercises performed by this code do not affect \n"
          "them. We report all 'nvlink' events presented by the cuda component.  \n"
          "\n"
          "---------------------------Event Name---------------------------:---Value---\n");

    PAPI_event_info_t info;                                             // To get event enumeration info.
    m=PAPI_NATIVE_MASK;                                                 // Get the PAPI NATIVE mask.
    CALL_PAPI_OK(PAPI_enum_cmp_event(&m,PAPI_ENUM_FIRST,cid));          // Begin enumeration of ALL papi counters.
    do {                                                                // Enumerate all events.
        memset(&info,0,sizeof(PAPI_event_info_t));                      // Clear event info.
        k=m;                                                            // Make a copy of current code.

        // enumerate sub-events, with masks. For this test, we do not
        // have any! But we do this to test our enumeration works as
        // expected. First time through is guaranteed, of course.

        do {                                                            // enumerate masked events. 
            CALL_PAPI_OK(PAPI_get_event_info(k,&info));                 // get name of k symbol.
            if (strstr(info.symbol, "nvlink") == NULL) continue;        // skip if not an nvlink event.
            char *devstr = strstr(info.symbol, "device=");              // look for device enumerator.
            if (devstr == NULL) continue;                               // Skip if no device present. 
            device=atoi(devstr+7);                                      // Get the device id, for info.
            // fprintf(stderr, "Found nvlink symbol '%s', device=%i.\n", info.symbol , device);
            if (device < 0 || device >= deviceCount) continue;          // skip any not in range.
            deviceEvents[device]++;                                     // Add to count of events on this device.

            CALL_PAPI_OK(PAPI_create_eventset(&EventSet)); 
            CALL_PAPI_OK(PAPI_assign_eventset_component(EventSet, cid)); 

            retval = PAPI_add_named_event(EventSet, info.symbol);       // Don't want to fail program if name not found...
            if(retval == PAPI_OK) {
                eventCount++;                                           // Bump number of events we could test.
            } else {
                CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));          // Delete all events in set.
                CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));         // destroy the event set.
                continue; 
            }

            long long value=-1;                                         // The only value we read.
            
            // ===== Allocate Memory =====================================
            
            if(cpuToGpu) {
                conductCpuToGpu(EventSet, device, &value);              // Just one value for now.
            } else if(gpuToGpu) {
                conductGpuToGpu(EventSet, device, &value);              // Just one value for now.
            }

            addEventsFound(info.symbol, value);                         // Add to events we were able to read.
            
            CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));              // Delete all events in set.
            CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));             // destroy the event set.

            // report each event counted.
            if (value >= 0) {                                           // If not still -1,
                eventsRead++;                                           // .. count and report.
                calculateSize(str, value);
                if (value == 0) {
                    printf("%-64s: %9s (not exercised by current test code.)\n", info.symbol, str);
                } else {
                    printf("%-64s: %9s\n", info.symbol, str);
                }
            } else {
                printf("%-64s: Failed to read.\n", info.symbol);
            }
        } while(PAPI_enum_cmp_event(&k,PAPI_NTV_ENUM_UMASKS,cid)==PAPI_OK); // Get next umask entry (bits different) (should return PAPI_NOEVNT).
    } while(PAPI_enum_cmp_event(&m,PAPI_ENUM_EVENTS,cid)==PAPI_OK);         // Get next event code.

    if (eventCount < 1) {                                                   // If we failed on all of them,
        fprintf(stderr, "Unable to add any NVLINK events; they are not present in the component.\n");
        fprintf(stderr, "Unable to proceed with this test.\n");
        FreeGlobals();
        PAPI_shutdown();                                                    // Returns no value.
        exit(-1);                                                           // exit no matter what.
    }
        
    if (eventsRead < 1) {                                                   // If failed to read any,
    printf("\nFailed to read any nvlink events.\n");                        // report a failure.
        fprintf(stderr, "Unable to proceed with this test.\n");
        FreeGlobals();
        PAPI_shutdown();                                                    // Returns no value.
        exit(-1);                                                           // exit no matter what.
    }

    printf("\nTotal nvlink events identified: %i.\n\n", eventsFoundCount);
    if (eventsFoundCount < 2) {                                             // If failed to get counts on any,
        printf("Insufficient events are exercised by the current test code to perform pair testing.\n"); // report a failure.
        FreeGlobals();
        PAPI_shutdown();                                                    // Returns no value.
        exit(0);                                                            // exit no matter what.
    }

    for (i=0; i<deviceCount; i++) {
        printf("Device %i has %i events. %i potential pairings per device.\n", i, deviceEvents[i], deviceEvents[i]*(deviceEvents[i]-1)/2);
    }

    // Begin pair testing. We consider every possible pairing of events
    // that, tested alone, returned a value greater than zero.

    int mainEvent, pairEvent, mainDevice, pairDevice;
    long long saveValues[2];
    long long readValues[2];
    int  goodOnSame=0, failOnDiff=0, badSameCombo=0, pairProblems=0;        // Some counters.
    int type;                                                               // 0 succeed on same device, 1 = fail across devices.
    for (type=0; type<2; type++) {
        if (type == 0) {
            printf("List of Pairings on SAME device:\n");
            printf("* means value changed by more than 10%% when paired (vs measured singly, above).\n");
            printf("^ means a pair was rejected as an invalid combo.\n");
        } else {
            printf("List of Failed Pairings on DIFFERENT devices:\n");
        }

        for (mainEvent = 0; mainEvent<eventsFoundCount-1; mainEvent++) {                // Through all but one events.
             char *devstr = strstr(eventsFound[mainEvent].name, "device=");             // look for device enumerator.
             mainDevice=atoi(devstr+7);                                                 // Get the device id.
            
            for (pairEvent = mainEvent+1; pairEvent<eventsFoundCount; pairEvent++) {    // Through all possible pairs,
                devstr = strstr(eventsFound[pairEvent].name, "device=");                // look for device enumerator.
                pairDevice=atoi(devstr+7);                                              // Get the device id.

                if (type == 0 && mainDevice != pairDevice) continue;                    // Skip if we need same device.
                if (type == 1 && mainDevice == pairDevice) continue;                    // Skip if we need different devices.

                CALL_PAPI_OK(PAPI_create_eventset(&EventSet)); 
                CALL_PAPI_OK(PAPI_assign_eventset_component(EventSet, cid)); 
                CALL_PAPI_OK(PAPI_add_named_event(EventSet, eventsFound[mainEvent].name));
                // Here we must examine the return code.
                int ret = PAPI_add_named_event(EventSet, eventsFound[pairEvent].name);
                if (type == 0 && ret == PAPI_ECOMBO) {                                  // A bad combination when looking for valid combos.
                    printf("%c %64s + %-64s [Invalid Combo]\n", '^',                    // report it.
                        eventsFound[mainEvent].name, eventsFound[pairEvent].name);
                    badSameCombo++;                                                     // .. count an explicit rejection.
                    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // .. done with event set.
                    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // ..
                    continue;                                                           // .. try the next combo.
                }

                if (type == 1 && ret == PAPI_ECOMBO) {                                  // A bad  combination when we are looking for that.
                    printf("%64s + %-64s BAD COMBINATION ACROSS DEVICES.\n", 
                        eventsFound[mainEvent].name, eventsFound[pairEvent].name);      // report it.
                    failOnDiff++;                                                       // count the bad combos.
                    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // .. don't need to go further.
                    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // ..
                    continue;                                                           // .. try the next combo.
                }

                if (ret != PAPI_OK) {                                                   // If it failed for some other reason,
                    fprintf(stderr, "%s:%d Attempt to add event '%s' to set "
                            "with event '%s' produced an unexpected error: "
                            "[%s]. Ignoring this pair.\n", 
                        __FILE__, __LINE__, eventsFound[pairEvent], 
                        eventsFound[mainEvent], PAPI_strerror(ret));
                    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // .. didn't work.
                    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // ..
                    continue;                                                           // .. try the next combo.
                }

                // We were able to add the pair. In type 1, we just skip it,
                // because we presume a single event on a device isn't changed
                // by any event on another device.
                if (type == 1) {
                    CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                      // .. worked fine; don't measure it.
                    CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                     // ..
                    continue;                                                           // .. try the next combo.
                }

                // We were able to add the pair, in type 0, get a measurement. 
                readValues[0]= -1; readValues[1] = -1;

                if(cpuToGpu) {
                    conductCpuToGpu(EventSet, mainDevice, readValues);                  // conduct for main.
                    saveValues[0] = readValues[0];
                    saveValues[1] = readValues[1];
                } else if(gpuToGpu) {
                    conductGpuToGpu(EventSet, mainDevice, readValues);                  // conduct for main.
                    saveValues[0] = readValues[0];
                    saveValues[1] = readValues[1];
                }

                goodOnSame++;                                                           // Was accepted by cuda as a valid pairing.

                // For the checks, we add 2 (so -1 becomes +1) to avoid any
                // divide by zeros. It won't make a significant difference 
                // in the ratios. (none if readings are the same). 
                double mainSingle = (2.0 + eventsFound[mainEvent].value);               // Get value when read alone.
                double pairSingle = (2.0 + eventsFound[pairEvent].value);               // ..
                double mainCheck  = mainSingle/(2.0 + saveValues[0]);                   // Get ratio when paired.
                double pairCheck  = pairSingle/(2.0 + saveValues[1]);                   // ..

                char flag=' ', flag1=' ', flag2=' ';                                    // Presume all okay.
                if (mainCheck < 0.90 || mainCheck > 1.10) flag1='*';                    // Flag as significantly different for main.
                if (pairCheck < 0.90 || pairCheck > 1.10) flag2='*';                    // Flag as significantly different for pair.
                if (flag1 == '*' || flag2 == '*') {
                    pairProblems++;                                                     // Remember number of problems.
                    flag = '*';                                                         // set global flag.
                }

                printf("%c %64s + %-64s [", flag, eventsFound[mainEvent].name, eventsFound[pairEvent].name);
                calculateSize(str, saveValues[0]);                                      // Do some pretty formatting,
                printf("%c%9s,", flag1, str);
                calculateSize(str, saveValues[1]);
                printf("%c%9s]\n", flag2, str);

                CALL_PAPI_OK(PAPI_cleanup_eventset(EventSet));                          // Delete all events in set.
                CALL_PAPI_OK(PAPI_destroy_eventset(&EventSet));                         // destroy the event set.
            }
        } // end loop on all events.

        if (type == 0) {                                                                // For good pairings on same devices,
            if (goodOnSame == 0) {
                printf("NO valid pairings of above events if both on the SAME device.\n");
            } else {
                printf("%i valid pairings of above events if both on the SAME device.\n", goodOnSame);
            }

            printf("%i unique pairings on SAME device were rejected as bad combinations.\n", badSameCombo);
            
            if (pairProblems > 0) {
                printf("%i pairings resulted in a change of one or both event values > 10%%.\n", pairProblems);
            } else {
                printf("No significant change in event values read for any pairings.\n");
            }
        } else {                                                                        // Must be reporting bad pairings across devies.
            if (failOnDiff == 0) printf("NO failed pairings of above events if each on a DIFFERENT device.\n");
            else printf("%i failed pairings of above events with each on a DIFFERENT device.\n", failOnDiff);
        }
    } // end loop on type.

    PAPI_shutdown();                                                                    // Returns no value.
    return(0);                                                                          // exit OK.
} // end MAIN.
