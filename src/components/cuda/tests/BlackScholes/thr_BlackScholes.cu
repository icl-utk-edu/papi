/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

//-----------------------------------------------------------------------------
// thr_BlackScholes.cu
// This is a modification of a cuda sample code; the original was found in
// /sw/cuda/10.1/samples/4_Finance/BlackScholes/. It requires a modification to
// the Makefile, ensure OSCONTEXT and OSLOCK are defined as given in
// config.log. The point is to test thread safety in the CUDA component. 
// 
// As written, the CUDA component only allows a single EventSet, so threads
// cannot create multiple EventSets of their own, but they can all READ the
// same eventset, which can contain multiple events. We use valgrind with tool
// helgrind to search for race conditions. A run on the ICL Saturn cluster:

// srun -N1 -wa04 valgrind --tool=helgrind ./thr_BlackScholes cuda:::metric:inst_fp_32:device=0 2>helgrind-out.txt

// The arguments are cuda events as given by PAPI/src/utils/papi_native_avail.
// Note that helgrind reports *potential* race conditions and does not parse
// _papi_hwi_lock() or _papi_hwi_unlock(); so reports must be parsed by hand to
// ensure any conflict lines are locked by the same lock; e.g. papi's
// THREADS_LOCK, or COMPONENT_LOCK.
//
// IMPORTANT: Do not use PAPI_register_thread() and PAPI_unregister_thread()
// with the CUDA component; PAPI_unregister_thread() can delete the EventSet
// still being used by other threads; resulting in segfaults and PAPI errors.
// Nor can you create 'local' EventSets within the thread. The CUDA component
// only allows ONE context, because the GPUs are shared by ALL threads and
// cores. CUDA is already thread-aware on Kernel Execution, it serializes
// those.  However, helgrind exposes potential race conditions in the cuda
// library routines (alloc, free, copy), and we cannot look at the source to
// see if they are thread safe or not. So we surround those with  PAPI locks.
// NOTE that in this code, PAPI_thread_init() is necessary, or the locks
// don't do anything.
//
// Different threads can all READ the EventSet, but it is up to the user to
// force any kind of ordering on the threads or the reads. In experiments, even
// though cuda serializes the kernel executions, we don't know in what order, and
// even then the reads, on different threads, may be executed in any order. It 
// is not unusual for all three reads to occur after all the kernels execute and
// report the same counter value.
// 
// Mods by Tony Castaldo; 09/16/2020.
//-----------------------------------------------------------------------------

#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include <unistd.h>             // sleep function.
#include <pthread.h>
#include <time.h>
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"

#define REPORT if (0)  /* '0' for quiet operation. */
#define NUM_THREADS 3

#define CUDA_CALL( call, handleerror )                                              \
    do {                                                                            \
        cudaError_t _status = (call);                                               \
        if (_status != cudaSuccess) {                                               \
            printf("\"%s:%s:%i CUDA error: function %s failed with error %d.\"\n", __FILE__, __func__, __LINE__, #call, _status);   \
            fflush(stdout);                                                         \
            handleerror;                                                            \
        }                                                                           \
    } while (0)

//--- threads wait until spinwait == their ID; which is set from 0 to (NUM_THREADS-1)
//--- and then increment it, to automatically launch the next thread.
//--- If you want them all to run at once, set spinWait = NUM_THREADS.
static int spinWait = -1;

typedef struct threadInfo {
    int myId;               // my thread ID.
    int numOptions;         // number of BlackSholes options to run.
    int EventSet;           // eventset created by main.
    int rc_read;            // any return code from read.
    long long *myRead;      // Array to read values, calloc() by master.
    int order;              // Order of PAPI_read execution.
} threadInfo_t;

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

int runBlackScholes(int opt_n, int doLocks);
static threadInfo_t *threadInfo = NULL;      // Array of threadInfo_t[NUM_THREADS].

//-----------------------------------------------------------------------------
// black_scholes thread. 
//-----------------------------------------------------------------------------
static void* blackScholesThread(void *given) {
    threadInfo_t *myInfo = (threadInfo_t*) given;
    int ret;

//  fprintf(stderr, "%s:%s:%i thread id %d is waiting.\n", __FILE__, __func__, __LINE__, myInfo->myId);

    while (1) {
        _papi_hwi_lock( THREADS_LOCK );
        // Unlock and try again if it is not my turn. 
        if (spinWait != myInfo->myId && spinWait != NUM_THREADS) {
            _papi_hwi_unlock( THREADS_LOCK );
            continue;
        }

        // It is my turn. Still locked. 
        if (spinWait != NUM_THREADS) spinWait++;
        fprintf(stderr, "Thread %d is running.\n", myInfo->myId);
        _papi_hwi_unlock( THREADS_LOCK );

        ret = runBlackScholes(myInfo->numOptions, 0);
        if (ret < 0) {
            switch(ret) {
                case -1: // cudaMalloc
                    fprintf(stderr, "%s:%s:%i runBlackScholes failed; cudaMalloc() failed.\n", __FILE__, __func__, __LINE__); break;
                case -2: // cudaMemcpy host->device
                    fprintf(stderr, "%s:%s:%i runBlackScholes failed; cudaMemcpy() host->device failed.\n", __FILE__, __func__, __LINE__); break;
                case -3: // cudaDeviceSynchronize
                    fprintf(stderr, "%s:%s:%i runBlackScholes failed; first cudaDeviceSynchronize() failed.\n", __FILE__, __func__, __LINE__); break;
                case -4: // cudaDeviceSynchronize
                    fprintf(stderr, "%s:%s:%i runBlackScholes failed; second cudaDeviceSynchronize() failed.\n", __FILE__, __func__, __LINE__); break;
                case -5: // cudaMemcpy device->host
                    fprintf(stderr, "%s:%s:%i runBlackScholes failed; cudaMemcpy() device->host failed.\n", __FILE__, __func__, __LINE__); break;
                case -6: // cudaFree 
                    fprintf(stderr, "%s:%s:%i runBlackScholes failed; cudaFree() failed.\n", __FILE__, __func__, __LINE__); break;
                default: // Unknown error.
                    fprintf(stderr, "%s:%s:%i runBlackScholes failed; Unkown error=%d.\n", __FILE__, __func__, __LINE__, ret); break;
            } 

            myInfo->rc_read = PAPI_EMISC;   // Record miscellaneous error. 
        } else { // success.
            _papi_hwi_lock( THREADS_LOCK );
            fprintf(stderr, "%s:%i Thread %d papi_hwd_lock_data[THREADS_LOCK]=%i order=[%d,%d,%d].\n", __FILE__, __LINE__, 
                myInfo->myId, _papi_hwd_lock_data[THREADS_LOCK],
                threadInfo[0].order, threadInfo[1].order, threadInfo[2].order);
            int i, max=-2;
            for (i=0; i<NUM_THREADS; i++) if (threadInfo[i].order > max) max=threadInfo[i].order;
            myInfo->order = (max+1);
            fprintf(stderr, "Thread %i read order %i\n", myInfo->myId, myInfo->order);
            myInfo->rc_read = PAPI_read( myInfo->EventSet, myInfo->myRead );

            _papi_hwi_unlock( THREADS_LOCK );
            fprintf(stderr, "Thread %d read %lli.\n", myInfo->myId, myInfo->myRead[0]);

            if( myInfo->rc_read != PAPI_OK ) {
                fprintf(stderr, "%s:%s:%i Thread %d, PAPI_read failed; rc=%d='%s'\n", __FILE__, __func__, __LINE__, 
                    myInfo->myId, myInfo->rc_read, PAPI_strerror(myInfo->rc_read));
            }
        }

        // Exit infinite loop.
        break; 
    } // END WHILE.

    return(NULL);
} // end blackScholesThread()


////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
int MAX_N = 4000000; // Set MAXIMUM here.
const int  NUM_ITERATIONS = 512;


const int          OPT_SZ = MAX_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

// This takes a second operand; DoLocks, which will execute lock/unlock around cuda memory operations.
// otherwise, it assumes the whole routine is locked.
int runBlackScholes(int opt_n, int doLocks) {
    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    double L1norm=0., gpuTime;

    StopWatchInterface *hTimer = NULL;
    int i;

    sdkCreateTimer(&hTimer);

    REPORT printf("Initializing data...\n");
    REPORT printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);

    REPORT printf("...allocating GPU memory for options.\n");
   
    // Explanation: valgrind --tools=helgrind indicates race conditions on the CPU side
    // for cudaMalloc() and cudaFree(); without the source we cannot be sure if these 
    // are real. So we do our own lock, just in case. 
    if (doLocks) _papi_hwi_lock( THREADS_LOCK );
    CUDA_CALL(cudaMalloc((void **)&d_CallResult,   OPT_SZ), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-1));
    CUDA_CALL(cudaMalloc((void **)&d_PutResult,    OPT_SZ), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-1));
    CUDA_CALL(cudaMalloc((void **)&d_StockPrice,   OPT_SZ), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-1));
    CUDA_CALL(cudaMalloc((void **)&d_OptionStrike, OPT_SZ), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-1));
    CUDA_CALL(cudaMalloc((void **)&d_OptionYears,  OPT_SZ), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-1));
    if (doLocks) _papi_hwi_unlock( THREADS_LOCK );

    REPORT printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    for (i = 0; i < opt_n; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    REPORT printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    if (doLocks) _papi_hwi_lock( THREADS_LOCK );
    CUDA_CALL(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-2));
    CUDA_CALL(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-2));
    CUDA_CALL(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-2));
    if (doLocks) _papi_hwi_unlock( THREADS_LOCK );
    REPORT printf("Data init done.\n\n");


    REPORT printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    if (doLocks) _papi_hwi_lock( THREADS_LOCK );
    CUDA_CALL(cudaDeviceSynchronize(), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-3));
    if (doLocks) _papi_hwi_unlock( THREADS_LOCK );
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<DIV_UP((opt_n/2), 128), 128/*480, 128*/>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            opt_n
        );
        getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    if (doLocks) _papi_hwi_lock( THREADS_LOCK );
    CUDA_CALL(cudaDeviceSynchronize(), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-4));
    if (doLocks) _papi_hwi_unlock( THREADS_LOCK );
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

    //Both call and put is calculated
    REPORT printf("Options count             : %i     \n", 2 * opt_n);
    REPORT printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    REPORT printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * opt_n * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    REPORT printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * opt_n) * 1E-9) / (gpuTime * 1E-3));

    REPORT printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * opt_n) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * opt_n), 1, 128);

    REPORT printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    if (doLocks) _papi_hwi_lock( THREADS_LOCK );
    CUDA_CALL(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-5));
    CUDA_CALL(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-5));
    if (doLocks) _papi_hwi_unlock( THREADS_LOCK );

#if 0 // Whether to check Results 
    double delta, ref, sum_delta, sum_ref, max_delta;
    REPORT printf("Checking the results...\n");
    REPORT printf("...running CPU calculations.\n\n");
    //Calculate options values on CPU
    BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        opt_n
    );

    REPORT printf("Comparing the results...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < opt_n; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    REPORT printf("L1 norm: %E\n", L1norm);
    REPORT printf("Max absolute error: %E\n\n", max_delta);
#endif 

    REPORT printf("Shutting down...\n");
    REPORT printf("...releasing GPU memory.\n");
    if (doLocks) _papi_hwi_lock( THREADS_LOCK );
    CUDA_CALL(cudaFree(d_OptionYears), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-6));
    CUDA_CALL(cudaFree(d_OptionStrike), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-6));
    CUDA_CALL(cudaFree(d_StockPrice), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-6));
    CUDA_CALL(cudaFree(d_PutResult), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-6));
    CUDA_CALL(cudaFree(d_CallResult), if (doLocks) _papi_hwi_unlock( THREADS_LOCK ); return (-6));
    if (doLocks) _papi_hwi_unlock( THREADS_LOCK );

    REPORT printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    sdkDeleteTimer(&hTimer);
    REPORT printf("Shutdown done.\n");

    REPORT printf("\n[BlackScholes] - Test Summary\n");

    if (L1norm > 1e-6)
    {
        REPORT printf("Test failed!\n");
        return(-1);
    }

    REPORT printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
    return(0);
} // end runBlackScholes

////////////////////////////////////////////////////////////////////////////////
// Main program
// Here we are threaded; we launch several threads; they all wait for a global
// signal. 
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    // Start logs
    int ret, i, numEvents;
    int EventSet = PAPI_NULL;
    int       *events;          // [numEvents] papi event ids.
    pthread_t *threadHandles = NULL;    // All my thread handles.
 
    // From original, looks for "device=" command line arg, sets device to use. 
    // findCudaDevice(argc, (const char **)argv); 
   checkCudaErrors(cudaSetDevice(0));

    REPORT printf("[%s] - Starting...\n", argv[0]);

    if (argc < 2) {
        fprintf(stderr, "You must have at least one argument, the event(s) to test.\n");
        exit(-1);
    }

    numEvents = argc-1; 
    events = (int *) calloc(numEvents, sizeof(int));
    long long *stopread = (long long*) calloc(numEvents, sizeof(long long));
    threadHandles = (pthread_t *) calloc(NUM_THREADS, sizeof(pthread_t));
    threadInfo    = (threadInfo_t *) calloc(NUM_THREADS, sizeof(threadInfo_t));

    /* PAPI Initialization */
    ret = PAPI_library_init( PAPI_VER_CURRENT );
    if( ret != PAPI_VER_CURRENT ) {
        fprintf(stderr, "PAPI_library_init failed; ret=%d='%s'\n", ret, PAPI_strerror(ret));
            if (numEvents==1) {printf("PAPI_library_init failed; name='%s' ret=%d='%s'\n", argv[1], ret, PAPI_strerror(ret)); fflush(stdout);}
        exit (-2);
    }

   ret = PAPI_thread_init(pthread_self);
    if( ret != PAPI_OK ) {
        fprintf(stderr, "PAPI_thread_init failed; ret=%d='%s'\n", ret, PAPI_strerror(ret));
        exit (-2);
    }

    /* convert PAPI native events to PAPI code */
    for( i = 0; i < numEvents; i++ ){
        ret = PAPI_event_name_to_code( argv[i+1], &events[i] );
        if( ret != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failed; name='%s' ret=%d='%s'\n", argv[i+1], ret, PAPI_strerror(ret));
            if (numEvents==1) {printf("PAPI_event_name_to_code failed; name='%s' ret=%d='%s'\n", argv[i+1], ret, PAPI_strerror(ret)); fflush(stdout);}
            exit (-2);
        }   

        REPORT printf("Name %s --- Code: %#x\n", argv[i+1], events[i]);
    }

    ret = PAPI_create_eventset( &EventSet );
    if( ret != PAPI_OK ) {
        fprintf(stderr, "PAPI_create_eventset failed; ret=%d='%s'\n", ret, PAPI_strerror(ret));
        if (numEvents==1) {printf("PAPI_create_eventset failed; ret=%d='%s'\n", ret, PAPI_strerror(ret)); fflush(stdout);}
        exit (-2);
    }    

    // If multiple GPUs/contexts were being used, 
    // you need to switch to each device before adding its events
    // e.g. cudaSetDevice( 0 );
    ret = PAPI_add_events( EventSet, events, numEvents );
    if( ret != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events failed; ret=%d='%s'\n", ret, PAPI_strerror(ret));
        if (numEvents==1) {printf("PAPI_add_events failed; ret=%d='%s'\n", ret, PAPI_strerror(ret)); fflush(stdout);}
        exit (-2);
    }

    ret = PAPI_start( EventSet );
    if( ret != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failed; ret=%d='%s'\n", ret, PAPI_strerror(ret));
        if (numEvents==1) {printf("PAPI_start failed; ret=%d='%s'\n", ret, PAPI_strerror(ret)); fflush(stdout);}
        exit (-2);
    } 

    //-------------------------------------------------------------------------
    // Deal with threads. 
    //-------------------------------------------------------------------------
    for (i=0; i<NUM_THREADS; i++) {
        threadInfo[i].myId = i;
        threadInfo[i].numOptions = MAX_N/(i+1);
        threadInfo[i].EventSet = EventSet;
        threadInfo[i].rc_read = -1;
        threadInfo[i].order = -1;
        threadInfo[i].myRead = (long long*) calloc(numEvents, sizeof(long long));
   }

    for (i=0; i<NUM_THREADS; i++) {
        ret = pthread_create( &(threadHandles[i]), NULL, blackScholesThread, (void*) &(threadInfo[i]));
        if (ret != 0) fprintf(stderr, "pthread_create() for thread %d, error=%i.\n", i, ret);
    }

    spinWait = 0;   // Open the gate.

    // Wait on all threads.
    for (i=0; i<NUM_THREADS; i++) {
        ret = pthread_join(threadHandles[i], NULL);
        if (ret != 0) {
            printf("Thread %1d: pthread_join() error=%d.\n",i, ret);
            threadInfo[i].rc_read = PAPI_EMISC;
        }
    }

    for (i=0; i<NUM_THREADS; i++) {
        printf("Thread %1d: threadInfo[i].ret_read=%i, readOrder=%i, read values=[", 
          threadInfo[i].myId, threadInfo[i].rc_read, threadInfo[i].order);

        int j;
        for (j=0; j<numEvents; j++) {
            if (j < (numEvents-1)) printf("%lli,", threadInfo[i].myRead[j]);
            else                   printf("%lli]\n", threadInfo[i].myRead[j]);
        }

        free(threadInfo[i].myRead);  
    } 

    ret = PAPI_stop(EventSet, stopread); // Ignore values on Stop.
    if( ret != PAPI_OK ) {
        fprintf(stderr, "PAPI_stop failed; EventSet=%d, ret=%d='%s'\n", EventSet, ret, PAPI_strerror(ret));
        exit (-2);
    }

    ret = PAPI_cleanup_eventset(EventSet);
    if( ret != PAPI_OK ) {
        fprintf(stderr, "PAPI_cleanup_eventset failed; EventSet=%d, ret=%d='%s'\n", EventSet, ret, PAPI_strerror(ret));
        exit (-2);
    }

    ret = PAPI_destroy_eventset(&EventSet);
    if( ret != PAPI_OK ) {
        fprintf(stderr, "PAPI_destroy_eventset failed; EventSet=%d, ret=%d='%s'\n", EventSet, ret, PAPI_strerror(ret));
        exit (-2);
    }

    ret = PAPI_unregister_thread();

    if (ret != PAPI_OK) 
        fprintf(stderr, "%s:%s:%i PAPI_unregister_thread() failed.\n", __FILE__, __func__, __LINE__);

    free(events);

    PAPI_shutdown();
    return 0;
}
