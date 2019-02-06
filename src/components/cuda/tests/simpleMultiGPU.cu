/* 
 * PAPI Multiple GPU example.  This example is taken from the NVIDIA
 * documentation (Copyright 1993-2013 NVIDIA Corporation) and has been
 * adapted to show the use of CUPTI and PAPI in collecting event
 * counters for multiple GPU contexts.  PAPI Team (2015)
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

#include "papi.h"
#include "papi_test.h"

#if not defined PAPI
#undef PAPI
#endif

#if not defined CUPTI_ONLY
#undef CUPTI_ONLY
#endif

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#include "simpleMultiGPU.h"

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
// //////////////////////////////////////////////////////////////////////////////
// Data configuration
// //////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 32;
const int DATA_N = 48576 * 32;
#ifdef PAPI
const int MAX_NUM_EVENTS = 32;
#endif

#define CHECK_CU_ERROR(err, cufunc)                                     \
    if (err != CUDA_SUCCESS) { printf ("Error %d for CUDA Driver API function '%s'\n", err, cufunc); return -1; }

#define CHECK_CUDA_ERROR(err)                                           \
    if (err != cudaSuccess) { printf ("%s:%i Error %d for CUDA [%s]\n", __FILE__, __LINE__, err, cudaGetErrorString(err) ); return -1; }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                               \
    if (err != CUPTI_SUCCESS) { const char *errStr; cuptiGetResultString(err, &errStr); \
       printf ("%s:%i Error %d [%s] for CUPTI API function '%s'\n", __FILE__, __LINE__, err, errStr, cuptifunc); return -1; }


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
    
    const int BLOCK_N = 32;
    const int THREAD_N = 256;
    const int ACCUM_N = BLOCK_N * THREAD_N;

    CUcontext ctx[MAX_GPU_COUNT];
    
    printf( "Starting simpleMultiGPU\n" );
    
    // Report on the available CUDA devices
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    int runtimeVersion = 0, driverVersion = 0;
    char deviceName[64];
    CUdevice device[MAX_GPU_COUNT];
    CHECK_CUDA_ERROR( cudaGetDeviceCount( &GPU_N ) );
    if( GPU_N > MAX_GPU_COUNT ) GPU_N = MAX_GPU_COUNT;
    printf( "CUDA-capable device count: %i\n", GPU_N );
    for ( i=0; i<GPU_N; i++ ) {
        CHECK_CU_ERROR( cuDeviceGet( &device[i], i ), "cuDeviceGet" );
        CHECK_CU_ERROR( cuDeviceGetName( deviceName, 64, device[i] ), "cuDeviceGetName" );
        CHECK_CU_ERROR( cuDeviceGetAttribute( &computeCapabilityMajor, 
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device[i]), "cuDeviceGetAttribute");
        CHECK_CU_ERROR( cuDeviceGetAttribute( &computeCapabilityMinor, 
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device[i]), "cuDeviceGetAttribute");
        cudaRuntimeGetVersion( &runtimeVersion );
        cudaDriverGetVersion( &driverVersion );
        printf( "CUDA Device %d: %s : computeCapability %d.%d runtimeVersion %d.%d driverVersion %d.%d\n", i, deviceName, computeCapabilityMajor, computeCapabilityMinor, runtimeVersion/1000, (runtimeVersion%100)/10, driverVersion/1000, (driverVersion%100)/10 );
        if ( computeCapabilityMajor < 2 ) {
            printf( "CUDA Device %d compute capability is too low... will not add any more GPUs\n", i );
            GPU_N = i;
            break;
        }
    }
    uint32_t cupti_linked_version;
    cuptiGetVersion( &cupti_linked_version );
    printf("CUPTI version: Compiled against version %d; Linked against version %d\n", CUPTI_API_VERSION, cupti_linked_version );
    
    // create one context per device
    for (i = 0; i < GPU_N; i++) {
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );
        CHECK_CU_ERROR( cuCtxCreate( &(ctx[i]), 0, device[i] ), "cuCtxCreate" );
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
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
    
    
#ifdef CUPTI_ONLY
//  char const *cuptiEventName = "elapsed_cycles_sm"; // "elapsed_cycles_sm" "inst_executed"; "inst_issued0";
//  char const *cuptiEventName = "inst_executed";     // "elapsed_cycles_sm" "inst_executed"; "inst_issued0";
    char const *cuptiEventName = "inst_per_warp";     // "elapsed_cycles_sm" "inst_executed"; "inst_issued0";
    printf("Setup CUPTI counters internally for event '%s' (CUPTI_ONLY)\n", cuptiEventName);
    CUpti_EventGroup eg[MAX_GPU_COUNT];
    CUpti_EventID *myevent = (CUpti_EventID*) calloc(GPU_N, sizeof(CUpti_EventID));   // Make space for event ids.
    for ( i=0; i<GPU_N; i++ ) {
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CUPTI_ERROR(cuptiSetEventCollectionMode(ctx[i], CUPTI_EVENT_COLLECTION_MODE_KERNEL), "cuptiSetEventCollectionMode" );
        CHECK_CUPTI_ERROR( cuptiEventGroupCreate( ctx[i], &eg[i], 0 ), "cuptiEventGroupCreate" );
        cuptiEventGetIdFromName ( device[i], cuptiEventName, &myevent[i] );
        printf("GPU %i %s=%u.\n", i, cuptiEventName, myevent[i]);
        CHECK_CUPTI_ERROR( cuptiEventGroupAddEvent( eg[i], myevent[i] ), "cuptiEventGroupAddEvent" );
        CHECK_CUPTI_ERROR( cuptiEventGroupEnable( eg[i] ), "cuptiEventGroupEnable" );
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }
#endif
    
#ifdef PAPI
    printf("Setup PAPI counters internally (PAPI)\n");
    int EventSet = PAPI_NULL;
    int NUM_EVENTS = MAX_GPU_COUNT*MAX_NUM_EVENTS;
    long long values[NUM_EVENTS];
    int eventCount;
    int cid=-1;
    int retval, ee;
    
    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) fprintf( stderr, "PAPI_library_init failed\n" );
    printf( "PAPI version: %d.%d.%d\n", PAPI_VERSION_MAJOR( PAPI_VERSION ), PAPI_VERSION_MINOR( PAPI_VERSION ), PAPI_VERSION_REVISION( PAPI_VERSION ) );
    
    // Find cuda component index.
    int k = PAPI_num_components();                                      // get number of components.
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
        fprintf(stderr, "Failed to find cuda component among %i "
            "reported components.\n", k);
        PAPI_shutdown();
        exit(-1); 
    }

    printf("Found CUDA Component at id %d\n", cid);

    CALL_PAPI_OK(PAPI_create_eventset(&EventSet)); 
    CALL_PAPI_OK(PAPI_assign_eventset_component(EventSet, cid)); 
    
    // In this example measure events from each GPU
    int numEventEndings = 2;
    char const *EventEndings[] = { 
        "cuda:::metric:nvlink_total_data_transmitted",
        "cuda:::metric:nvlink_total_data_received",
    };

    // Add events at a GPU specific level ... eg cuda:::device:2:elapsed_cycles_sm
    char *EventName[NUM_EVENTS];
    char tmpEventName[64];
    eventCount = 0;
    for( i = 0; i < GPU_N; i++ ) {
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );         // Set device
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CUPTI_ERROR(cuptiSetEventCollectionMode(ctx[i], CUPTI_EVENT_COLLECTION_MODE_KERNEL), "cuptiSetEventCollectionMode" );
        for ( ee=0; ee<numEventEndings; ee++ ) {
            snprintf( tmpEventName, 64, "%s:device=%d\0", EventEndings[ee], i );
            // printf( "Trying to add event %s to GPU %d in PAPI...", tmpEventName , i ); fflush(NULL);
            retval = PAPI_add_named_event( EventSet, tmpEventName );
            if (retval==PAPI_OK) {
                printf( "Add event success: '%s' GPU %i\n", tmpEventName, i );
                EventName[eventCount] = (char *)calloc( 64, sizeof(char) );
                snprintf( EventName[eventCount], 64, "%s", tmpEventName );
                eventCount++;
            } else {
                printf( "Add event failure: '%s' GPU %i error=%s\n", tmpEventName, i, PAPI_strerror(retval));
            }
        }
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }
    
    // Start PAPI event measurement
    retval = PAPI_start( EventSet );
    if( retval != PAPI_OK )  fprintf( stderr, "PAPI_start failed, retval=%i [%s].\n", retval, PAPI_strerror(retval));
#endif
    
    // Start timing and compute on GPU(s)
    printf( "Computing with %d GPUs...\n", GPU_N );
    StartTimer();
    
    // Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (i = 0; i < GPU_N; i++) {
        // Set device
        CHECK_CUDA_ERROR( cudaSetDevice( i ));
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        // Copy input data from CPU
        CHECK_CUDA_ERROR( cudaMemcpyAsync( plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof( float ), cudaMemcpyHostToDevice, plan[i].stream ) );
        // Perform GPU computations
        reduceKernel <<< BLOCK_N, THREAD_N, 0, plan[i].stream >>> ( plan[i].d_Sum, plan[i].d_Data, plan[i].dataN );
        if ( cudaGetLastError() != cudaSuccess ) { printf( "reduceKernel() execution failed (GPU %d).\n", i ); exit(EXIT_FAILURE); }
        // Read back GPU results
        CHECK_CUDA_ERROR( cudaMemcpyAsync( plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N * sizeof( float ), cudaMemcpyDeviceToHost, plan[i].stream ) );
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }
    
    // Process GPU results
    printf( "Process GPU results on %d GPUs...\n", GPU_N );
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
        *( plan[i].h_Sum ) = ( float ) sum;
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }
    double gpuTime = GetTimer();


#ifdef CUPTI_ONLY
    size_t size = 1024;
    size_t sizeBytes = size*sizeof(uint64_t);
    uint64_t buffer[size];
    uint64_t tmp[size];     for (int jj=0; jj<1024; jj++) tmp[jj]=0;
    for ( i=0; i<GPU_N; i++ ) {
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CU_ERROR( cuCtxSynchronize( ), "cuCtxSynchronize" );
        CHECK_CUPTI_ERROR( cuptiEventGroupReadEvent ( eg[i], CUPTI_EVENT_READ_FLAG_NONE, myevent[i], &sizeBytes, &tmp[0] ), "cuptiEventGroupReadEvent" );
        buffer[i] = tmp[0];
        printf( "CUPTI %s device %d counterValue %u (on one domain, may need to be multiplied by num of domains)\n", cuptiEventName, i, buffer[i] );
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }
#endif

#ifdef PAPI
    for ( i=0; i<GPU_N; i++ ) {
        CHECK_CUDA_ERROR( cudaSetDevice( i ) );
        CHECK_CU_ERROR(cuCtxPushCurrent(ctx[i]), "cuCtxPushCurrent");
        CHECK_CU_ERROR( cuCtxSynchronize( ), "cuCtxSynchronize" );
        CHECK_CU_ERROR( cuCtxPopCurrent(&(ctx[i])), "cuCtxPopCurrent" );
    }

    retval = PAPI_stop( EventSet, values );                                         // Stop (will read values).
    if( retval != PAPI_OK )  fprintf( stderr, "PAPI_stop failed\n" );
    for( i = 0; i < eventCount; i++ )
        printf( "PAPI counterValue %12lld \t\t --> %s \n", values[i], EventName[i] );

    retval = PAPI_cleanup_eventset( EventSet );
    if( retval != PAPI_OK )  fprintf( stderr, "PAPI_cleanup_eventset failed\n" );
    retval = PAPI_destroy_eventset( &EventSet );
    if( retval != PAPI_OK ) fprintf( stderr, "PAPI_destroy_eventset failed\n" );
    PAPI_shutdown();
#endif

    sumGPU = 0;
    for( i = 0; i < GPU_N; i++ ) {
        sumGPU += h_SumGPU[i];
    }
    printf( "  GPU Processing time: %f (ms)\n", gpuTime );

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
        CHECK_CUDA_ERROR( cudaFreeHost( plan[i].h_Sum_from_device ) );
        CHECK_CUDA_ERROR( cudaFreeHost( plan[i].h_Data ) );
        CHECK_CUDA_ERROR( cudaFree( plan[i].d_Sum ) );
        CHECK_CUDA_ERROR( cudaFree( plan[i].d_Data ) );
        // Shut down this GPU
        CHECK_CUDA_ERROR( cudaStreamDestroy( plan[i].stream ) );
    }

#ifdef CUPTI_ONLY
    free(myevent);
#endif 
    
    exit( ( diff < 1e-5 ) ? EXIT_SUCCESS : EXIT_FAILURE );
}

