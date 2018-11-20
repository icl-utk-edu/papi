 /*
 * Copyright 2015-2016 NVIDIA Corporation. All rights reserved.
 *
 * Sample to demonstrate use of NVlink CUPTI APIs
 * Note: this is nearly identical to the NVDIA provided version, which can be
 * found in $CUDAPATH/extras/CUPTI/sample/nvlink_bandwidth/nvlink_bandwidth.cu.
 * The differences are white space and the addition of 'timeDuration' to the 
 * printMetricValue routine; some printf() and output text changes. -TonyC.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cupti.h>

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

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        const char *errstr;                                                    \
        cuGetErrorString (_status, &errstr);                                   \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d [%s].\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status, errstr);            \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define MEMORY_ALLOCATION_CALL(var)                                            \
do {                                                                           \
    if (var == NULL) {                                                         \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",           \
                __FILE__, __LINE__);                                           \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define MAX_DEVICES    (32)
#define BLOCK_SIZE     (1024)
#define GRID_SIZE      (512)
#define BUF_SIZE       (32 * 1024)
#define ALIGN_SIZE     (8)
#define SUCCESS        (0)
#define NUM_METRIC     (4)
#define NUM_EVENTS     (2)
#define MAX_SIZE       (64*1024*1024)   // 64 MB
#define NUM_STREAMS    (4)   // Should be = to DeviceProperties.asyncEngineCount.

CUpti_ActivityNvLink *nvlinkRec = NULL;
int cpuToGpu = 0;
int gpuToGpu = 0;
int cpuToGpuAccess = 0;
int gpuToGpuAccess = 0;
int DeviceToReport = 0;

extern "C" __global__ void test_nvlink_bandwidth(float *src, float *dst)
{
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     dst[idx] = src[idx] * 2.0f;
}

static void printActivity(CUpti_Activity *record)
{
     if (record->kind == CUPTI_ACTIVITY_KIND_NVLINK) {
         nvlinkRec = (CUpti_ActivityNvLink *)record;

         fprintf(stderr, "CUPTI callback on activity record: typeDev0 %d, typeDev1 %d, sysmem %d, peer %d, physical links %d, portdev0 %d, %d, %d, %d, portDev1 %d, %d, %d, %d, bandwidth %llu\n",
           nvlinkRec->typeDev0,
           nvlinkRec->typeDev1,
           ((nvlinkRec->flag & CUPTI_LINK_FLAG_SYSMEM_ACCESS) ? 1 : 0),
           ((nvlinkRec->flag & CUPTI_LINK_FLAG_PEER_ACCESS) ? 1 : 0),
           nvlinkRec->physicalNvLinkCount,
           nvlinkRec->portDev0[0], nvlinkRec->portDev0[1], nvlinkRec->portDev0[2], nvlinkRec->portDev0[3],
           nvlinkRec->portDev1[0], nvlinkRec->portDev1[1], nvlinkRec->portDev1[2], nvlinkRec->portDev1[3],
           (long long unsigned int)nvlinkRec->bandwidth);
         cpuToGpuAccess |= (nvlinkRec->flag & CUPTI_LINK_FLAG_SYSMEM_ACCESS);
         gpuToGpuAccess |= (nvlinkRec->flag & CUPTI_LINK_FLAG_PEER_ACCESS);
     }
     else {
         printf("CUPTI callback on activity record: Error. Unexpected CUPTI activity kind, expected CUPTI_ACTIVITY_KIND_NVLINK\n");
     }
}

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)

{
     *size = BUF_SIZE + ALIGN_SIZE;
     *buffer = (uint8_t*) calloc(1, *size);
     MEMORY_ALLOCATION_CALL(*buffer);
     *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                                 uint8_t *buffer, size_t size,
                                 size_t validSize)
{
     CUptiResult status;
     CUpti_Activity *record = NULL;
     do {
         status = cuptiActivityGetNextRecord(buffer, validSize, &record);
         if(status == CUPTI_SUCCESS) {
             printActivity(record);
         }
         else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
             break;
         }
         else {
             CUPTI_CALL(status);
         }
     } while (1);

     size_t dropped;
     CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
     if (dropped != 0) {
         printf("Dropped %u activity records\n", (unsigned int)dropped);
     }
 }

#define DIM(x) (sizeof(x)/sizeof(*(x)))

void calculateSize(char *result, uint64_t size)
{
    int i;

    const char *sizes[]   = { "TB", "GB", "MB", "KB", "B" };
    uint64_t  exbibytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;

    uint64_t  multiplier = exbibytes;

    for (i = 0; (unsigned)i < DIM(sizes); i++, multiplier /= (uint64_t)1024)
    {
        if (size < multiplier)
            continue;
        sprintf(result, "%.1f %s", (float) size / multiplier, sizes[i]);
        return;
    }
    strcpy(result, "0");
    return;
}

void readMetricValue(CUpti_EventGroup eventGroup, uint32_t numEvents,
                    CUdevice dev, CUpti_MetricID *metricId,
                    uint64_t timeDuration,
                    CUpti_MetricValue *metricValue) {

    size_t bufferSizeBytes, numCountersRead;
    uint64_t *eventValueArray = NULL;
    CUpti_EventID *eventIdArray;
    size_t arraySizeBytes = 0;
    size_t numTotalInstancesSize = 0;
    uint64_t numTotalInstances = 0;
    uint64_t *aggrEventValueArray = NULL;
    size_t aggrEventValueArraySize;
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

    printf("LINE %i, DeviceEventDomainAttribute numTotalInstances=%llu.\n", __LINE__, numTotalInstances);

    arraySizeBytes = sizeof(CUpti_EventID) * numEvents;
    bufferSizeBytes = sizeof(uint64_t) * numEvents * numTotalInstances;

    eventValueArray = (uint64_t *) malloc(bufferSizeBytes);
    MEMORY_ALLOCATION_CALL(eventValueArray);

    eventIdArray = (CUpti_EventID *) malloc(arraySizeBytes);
    MEMORY_ALLOCATION_CALL(eventIdArray);

    aggrEventValueArray = (uint64_t *) calloc(numEvents, sizeof(uint64_t));
    MEMORY_ALLOCATION_CALL(aggrEventValueArray);

    aggrEventValueArraySize = sizeof(uint64_t) * numEvents;

    CUPTI_CALL(cuptiEventGroupReadAllEvents(eventGroup, 
                                                CUPTI_EVENT_READ_FLAG_NONE,
                                                &bufferSizeBytes, 
                                                eventValueArray, 
                                                &arraySizeBytes, 
                                                eventIdArray, 
                                                &numCountersRead));

    printf("LINE %i, numCountersRead=%d.\n", __LINE__, numCountersRead);

    // Arrangement of 2-d Array returned in eventValueArray:
    //    domain instance 0: event0 event1 ... eventN
    //    domain instance 1: event0 event1 ... eventN
    //    ...
    //    domain instance M: event0 event1 ... eventN
    // But we accumulate by column, event[0], event[1], etc.

    for (i = 0; i < numEvents; i++) {                   // outer loop column traversal.
        for (j = 0; j < numTotalInstances; j++) {       // inner loop row traversal.
            aggrEventValueArray[i] += eventValueArray[i + numEvents * j];
            printf("For event %d (id %d) instance %d value %llu aggregate %d = %llu\n", i, eventIdArray[i], j,  eventValueArray[i + numEvents * j], i, aggrEventValueArray[i]);
        }
    }

    for (i = 0; i < NUM_METRIC; i++) {
        CUPTI_CALL(cuptiMetricGetValue(dev, metricId[i], arraySizeBytes,
                              eventIdArray, aggrEventValueArraySize,
                              aggrEventValueArray, timeDuration,
                              &metricValue[i]));
    }

    free(eventValueArray);
    free(eventIdArray);
}

  // Print metric value, we format based on the value kind
int printMetricValue(CUpti_MetricID metricId, CUpti_MetricValue metricValue, const char *metricName, uint64_t timeDuration) {

    CUpti_MetricValueKind valueKind;
    char str[64];
    size_t valueKindSize = sizeof(valueKind);

    CUPTI_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
                                       &valueKindSize, &valueKind));
    switch (valueKind) {

    case CUPTI_METRIC_VALUE_KIND_DOUBLE:
        printf("%s = ", metricName);
        calculateSize(str, (uint64_t)metricValue.metricValueDouble);
        // printf("%s   (val %lu  %lu nsec)\n", str, metricValue.metricValueUint64, timeDuration);
        printf("%s\n", str);
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

    case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
        printf("%s = ", metricName);
        calculateSize(str, (uint64_t)metricValue.metricValueThroughput);
        printf("%s\n", str);
        break;

    default:
        fprintf(stderr, "error: unknown value kind\n");
        return -1;
    }
    return 0;
  }

void testCpuToGpu(CUpti_EventGroup *eventGroup, CUdeviceptr *pDevBuffer,
                    float** pHostBuffer, size_t bufferSize,
                    cudaStream_t *cudaStreams,
                    uint64_t *timeDuration, int numEventGroup)
{
    int i;
    uint32_t value = 1;
    uint64_t startTimestamp, endTimestamp;

    for (i = 0; i < numEventGroup; i++) {
            CUPTI_CALL(cuptiEventGroupEnable(eventGroup[i]));
            CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup[i],
                                CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                sizeof(uint32_t), (void*)&value));
    }

    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));

    //Unidirectional copy H2D
    for (i = 0; i < NUM_STREAMS; i++)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBuffer[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    //Unidirectional copy D2H
    for (i = 0; i < NUM_STREAMS; i++)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync(pHostBuffer[i], (void *)pDevBuffer[i], bufferSize, cudaMemcpyDeviceToHost, cudaStreams[i]));}

    RUNTIME_API_CALL(cudaDeviceSynchronize());

    //Bidirectional copy
    for (i = 0; i < NUM_STREAMS; i+=2)
    {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBuffer[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
        RUNTIME_API_CALL(cudaMemcpyAsync(pHostBuffer[i+1], (void *)pDevBuffer[i+1], bufferSize, cudaMemcpyDeviceToHost, cudaStreams[i+1]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    CUPTI_CALL(cuptiGetTimestamp(&endTimestamp));
    *timeDuration = endTimestamp - startTimestamp;
}

void testGpuToGpu(CUpti_EventGroup *eventGroup, CUdeviceptr *pDevBuffer0, CUdeviceptr *pDevBuffer1,
                    float** pHostBuffer, size_t bufferSize,
                    cudaStream_t *cudaStreams,
                    uint64_t *timeDuration, int numEventGroup)
{
    int i, access;
    uint32_t value = 1;
    uint64_t startTimestamp, endTimestamp;

    CUPTI_CALL(cuptiActivityFlushAll(0));

    // Note the commented out cudaDeviceEnablePeerAccess() calls fail on peak;
    // the compute mode is set wrong and we don't have SUDO access to change it
    // or allow these calls. 
    
    RUNTIME_API_CALL(cudaDeviceCanAccessPeer(&access, 0, 1)); 
    if (access) {
        printf("Device 0 can access Device 1.\n");
//      RUNTIME_API_CALL(cudaSetDevice(0));
//      RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(1, 0));
    } else { 
        printf("Device 0 cannot access Device 1. GPU to GPU Test aborted.\n"); 
        return; 
    }

    RUNTIME_API_CALL(cudaDeviceCanAccessPeer(&access, 1, 0));
    if (access) {
        printf("Device 1 can access Device 0.\n");
//      RUNTIME_API_CALL(cudaSetDevice(1));
//      RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(0, 0));
    } else {
        printf("Device 1 cannot access Device 0. GPU to GPU Test aborted.\n");
        return;
    }

    //Unidirectional copy H2D
    for (i = 0; i < NUM_STREAMS; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBuffer0[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for (i = 0; i < NUM_STREAMS; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBuffer1[i], pHostBuffer[i], bufferSize, cudaMemcpyHostToDevice, cudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());


    for (i = 0; i < numEventGroup; i++) {
        printf("cuptiEventGroupEnable(eventGroup[%d])\n", i);
        CUPTI_CALL(cuptiEventGroupEnable(eventGroup[i]));
        CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup[i],
                            CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                            sizeof(uint32_t), (void*)&value));
    }
    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));

    for (i = 0; i < NUM_STREAMS; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBuffer0[i], (void *)pDevBuffer1[i], bufferSize, cudaMemcpyDeviceToDevice, cudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for (i = 0; i < NUM_STREAMS; i++) {
        RUNTIME_API_CALL(cudaMemcpyAsync((void *)pDevBuffer1[i], (void *)pDevBuffer0[i], bufferSize, cudaMemcpyDeviceToDevice, cudaStreams[i]));
    }
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    for (i = 0; i < NUM_STREAMS; i++) {
        test_nvlink_bandwidth<<<GRID_SIZE, BLOCK_SIZE>>>((float*)pDevBuffer1[i], (float*)pDevBuffer0[i]);
    }

    CUPTI_CALL(cuptiGetTimestamp(&endTimestamp));
    *timeDuration = endTimestamp - startTimestamp;
}

static void printUsage() {
    printf("usage: Demonstrate use of NVlink CUPTI APIs\n");
    printf("       -help           : display help message\n");
    printf("       --cpu-to-gpu    : Show results for data transfer between CPU and GPU \n");
    printf("       --gpu-to-gpu    : Show results for data transfer between two GPUs \n");
    printf("       --Dn            : Must follow the above argument; n=device to report; 0,1,... \n");
}

void parseCommandLineArgs(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Invalid number of options\n");
        exit(0);
    }

    if (strcmp(argv[1], "--cpu-to-gpu") == 0) {
        cpuToGpu = 1;
    }
    else if (strcmp(argv[1], "--gpu-to-gpu") == 0) {
        gpuToGpu = 1;
    }
    else if ((strcmp(argv[1], "--help") == 0) ||
             (strcmp(argv[1], "-help") == 0) ||
             (strcmp(argv[1], "-h") == 0)) {
        printUsage();
        exit(0);
    }
    else {
        cpuToGpu = 1;
    }

    if (strncmp(argv[2], "--D", 3) == 0) {
        DeviceToReport = atoi(argv[2]+3);
    } else {
        printUsage();
        exit(0);
    }
}

int main(int argc, char *argv[])
{
    int deviceCount = 0, i = 0, j = 0, numEventGroup = 0;
    size_t bufferSize = 0, freeMemory = 0, totalMemory = 0;
    CUpti_EventGroupSets *passes = NULL;
    CUcontext ctx;
    char str[64];

    CUdeviceptr pDevBuffer0[NUM_STREAMS];
    CUdeviceptr pDevBuffer1[NUM_STREAMS];
    float* pHostBuffer[NUM_STREAMS];

    cudaStream_t cudaStreams[NUM_STREAMS] = {0};

    CUpti_EventGroup eventGroup[32];
    CUpti_MetricID metricId[NUM_METRIC];
    uint32_t numEvents[NUM_METRIC];
    CUpti_MetricValue metricValue[NUM_METRIC];
    cudaDeviceProp prop[MAX_DEVICES];
    uint64_t timeDuration;

    // Adding nvlink Metrics.
    const char *metricName[NUM_METRIC] = {"nvlink_total_data_transmitted",
                                    "nvlink_transmit_throughput",
                                    "nvlink_total_data_received",
                                    "nvlink_receive_throughput"};

    // Parse command line arguments
    parseCommandLineArgs(argc, argv);

    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NVLINK));
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    DRIVER_API_CALL(cuInit(0));

    RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));

    printf("There are %d devices.\n", deviceCount);

    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(-1);
    }

   if (DeviceToReport < 0 || DeviceToReport >= deviceCount) {
        printf("The device given to report (%i) does not exist.\n", DeviceToReport);
        exit(-1);
   }

   CUresult cu_res = cuCtxCreate(&ctx, 0, 0);          // create context here.
   const char *cu_errstr;
   cuGetErrorString(cu_res, &cu_errstr);
   fprintf(stderr, "deviceCount=%i, cuCtxCreate result='%s'.\n", deviceCount, cu_errstr);


    for (i = 0; i < deviceCount; i++) {
       RUNTIME_API_CALL(cudaGetDeviceProperties(&prop[i], i));
       printf("CUDA Device %d Name: %s\n", i, prop[i].name);
       printf("CUDA Device %d asyncEngineCount: %i\n", i, prop[i].asyncEngineCount);
   }

   // Set memcpy size based on available device memory
   RUNTIME_API_CALL(cudaMemGetInfo(&freeMemory, &totalMemory));
   bufferSize = MAX_SIZE < (freeMemory/4) ? MAX_SIZE : (freeMemory/4);

   printf("Total Device Memory available : ");
   calculateSize(str, (uint64_t)totalMemory);
   printf("%s\n", str);

   printf("Memcpy size is set to %llu B (%llu MB)\n",
   (unsigned long long)bufferSize, (unsigned long long)bufferSize/(1024*1024));

   fflush(stdout);

   for(i = 0; i < NUM_STREAMS; i++) {
      RUNTIME_API_CALL(cudaStreamCreate(&cudaStreams[i]));
   }
   RUNTIME_API_CALL(cudaDeviceSynchronize());

   // Nvlink-topology Records are generated even before cudaMemcpy API is called.
   CUPTI_CALL(cuptiActivityFlushAll(0));

   // Transfer Data between Host And Device, if Nvlink is Present
   // Check condition : nvlinkRec->flag & CUPTI_LINK_FLAG_SYSMEM_ACCESS
   // True : Nvlink is present between CPU & GPU
   // False : Nvlink is not present.
   if ((nvlinkRec) && (((cpuToGpu) && (cpuToGpuAccess)) || ((gpuToGpu) && (gpuToGpuAccess)))) {
       for (i = 0; i < NUM_METRIC; i++) {
           CUPTI_CALL(cuptiMetricGetIdFromName(DeviceToReport, metricName[i], &metricId[i]));
           CUPTI_CALL(cuptiMetricGetNumEvents(metricId[i], &numEvents[i]));
           printf("LINE %i: device %i metricId[%i]=%d [%s], numEvents[%i]=%d.\n", __LINE__, DeviceToReport, i, metricId[i], metricName[i], i, numEvents[i]);
       }

       printf("LINE %i: sizeof metricId=%i\n", __LINE__, (sizeof metricId));
       CUPTI_CALL(cuptiMetricCreateEventGroupSets(ctx, (sizeof metricId) ,metricId, &passes));

       // EventGroups required to profile Nvlink metrics.
       for (i = 0; i < (signed)passes->numSets; i++) {
           for (j = 0; j < (signed)passes->sets[i].numEventGroups; j++) {
               eventGroup[numEventGroup] = passes->sets[i].eventGroups[j];

               if (!eventGroup[numEventGroup]) {
                   fprintf(stderr, "\n eventGroup initialization failed \n");
                   exit(-1);
               }

               numEventGroup++;
           }
       }

       printf("There will be %i passes required to profile the requested NVLINK metrics.\n", passes->numSets);

       CUPTI_CALL(cuptiSetEventCollectionMode(ctx, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));

       // ===== Allocate Memory =====================================

       for(i = 0; i < NUM_STREAMS; i++) {
           RUNTIME_API_CALL(cudaMalloc((void**)&pDevBuffer0[i], bufferSize));

           pHostBuffer[i] = (float *)malloc(bufferSize);
           MEMORY_ALLOCATION_CALL(pHostBuffer[i]);
       }

       RUNTIME_API_CALL(cudaDeviceSynchronize());

       if (cpuToGpu) {
           testCpuToGpu(eventGroup, pDevBuffer0, pHostBuffer, bufferSize, cudaStreams, &timeDuration, numEventGroup);
           printf("Data transferred between CPU & Device%d : \n", (int)nvlinkRec->typeDev0);
       }
       else if(gpuToGpu) {
           RUNTIME_API_CALL(cudaSetDevice(1));
           for(i = 0; i < NUM_STREAMS; i++) {
               RUNTIME_API_CALL(cudaMalloc((void**)&pDevBuffer1[i], bufferSize));
           }
           
           RUNTIME_API_CALL(cudaDeviceSynchronize());
           testGpuToGpu(eventGroup, pDevBuffer0, pDevBuffer1, pHostBuffer, bufferSize, cudaStreams, &timeDuration, numEventGroup);
           printf("Data transferred between Device 0 & Device 1 : %.3f MB\n", ((double) bufferSize) / (1<<20));
       }

       // Collect Nvlink Metric values for the data transfer via Nvlink for all the eventGroups.
       for (i = 0; i < numEventGroup; i++) {
           readMetricValue(eventGroup[i], NUM_EVENTS, DeviceToReport, metricId, timeDuration, metricValue);

           CUPTI_CALL(cuptiEventGroupDisable(eventGroup[i]));
           CUPTI_CALL(cuptiEventGroupDestroy(eventGroup[i]));

           for (i = 0; i < NUM_METRIC; i++) {
               if (printMetricValue(metricId[i], metricValue[i], metricName[i], timeDuration) != 0) {
                   printf("\n printMetricValue failed \n");
                   exit(-1);
               }
           }
       }
   }
   else {
       printf("No Nvlink supported device found\n");
   }

   return 0;
}
