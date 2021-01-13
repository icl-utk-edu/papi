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


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include <unistd.h>             // sleep function.
#include "papi.h"

#define REPORT if (0)  /* '0' for quiet operation. */

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

int runBlackScholes(int opt_n) {
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
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));

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
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
    REPORT printf("Data init done.\n\n");


    REPORT printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());
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

    checkCudaErrors(cudaDeviceSynchronize());
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
    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));

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
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));

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
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    // Start logs
    int retval, i, numEvents;
    int EventSet = PAPI_NULL;
    long long *values;          // [numEvents] read results.
    long long *testResults;     // [4*numEvents] all results for 4 experiments.
    int       *events;          // [numEvents] papi event ids.
 
    // From original, looks for "device=" command line arg, sets device to use. 
    // findCudaDevice(argc, (const char **)argv); 
   checkCudaErrors(cudaSetDevice(0));

    REPORT printf("[%s] - Starting...\n", argv[0]);

    if (argc < 2) {
        fprintf(stderr, "You must have at least one argument, the event(s) to test.\n");
        exit(-1);
    }

    numEvents = argc-1; 
    values = (long long *) calloc(numEvents, sizeof(long long));
    testResults = (long long *) calloc(5*numEvents, sizeof(long long));
    events = (int *) calloc(numEvents, sizeof(int));

    if (numEvents==1) {printf("%s,", argv[1]); fflush(stdout);} // Begin output here.

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) {
        fprintf(stderr, "PAPI_library_init failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
            if (numEvents==1) {printf("PAPI_library_init failed; name='%s' retval=%d='%s'\n", argv[1], retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    }

    /* convert PAPI native events to PAPI code */
    for( i = 0; i < numEvents; i++ ){
        retval = PAPI_event_name_to_code( argv[i+1], &events[i] );
        if( retval != PAPI_OK ) {
            fprintf(stderr, "PAPI_event_name_to_code failed; name='%s' retval=%d='%s'\n", argv[i+1], retval, PAPI_strerror(retval));
            if (numEvents==1) {printf("PAPI_event_name_to_code failed; name='%s' retval=%d='%s'\n", argv[i+1], retval, PAPI_strerror(retval)); fflush(stdout);}
            exit (-2);
        }   

        REPORT printf("Name %s --- Code: %#x\n", argv[i+1], events[i]);
    }

    retval = PAPI_create_eventset( &EventSet );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_create_eventset failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        if (numEvents==1) {printf("PAPI_create_eventset failed; retval=%d='%s'\n", retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    }    

    // If multiple GPUs/contexts were being used, 
    // you need to switch to each device before adding its events
    // e.g. cudaSetDevice( 0 );
    retval = PAPI_add_events( EventSet, events, numEvents );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_add_events failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        if (numEvents==1) {printf("PAPI_add_events failed; retval=%d='%s'\n", retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    }

    retval = PAPI_start( EventSet );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        if (numEvents==1) {printf("PAPI_start failed; retval=%d='%s'\n", retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    } 

    // invoke the kernel, run 0: A full run of algorithm.
    int exp=0;
    retval = runBlackScholes(MAX_N);
    if (retval < 0) { 
        fprintf(stderr, "Kernel execution failed.\n");
        if (numEvents==1) {printf("Run 0 BlackScholes kernel execution failed; retval=%d\n", retval); fflush(stdout);}
        exit (-2);
    }

    retval = PAPI_read( EventSet, values );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "1st PAPI_read failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        if (numEvents==1) {printf("PAPI_read failed; retval=%d='%s'\n", retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    }

    for (i=0; i<numEvents; i++) testResults[exp*numEvents+i]=values[i];

    // Run 1: at 1/4 as many, to ensure we are work dependent, not just run dependent.
    exp++;
    retval = runBlackScholes(MAX_N>>2);
    if (retval < 0) { 
        fprintf(stderr, "Kernel execution failed.\n");
        if (numEvents==1) {printf("Run 1 BlackScholes kernel execution failed; retval=%d\n", retval); fflush(stdout);}
        exit (-2);
    }

    retval = PAPI_read( EventSet, values );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_read failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        if (numEvents==1) {printf("2nd PAPI_read failed; retval=%d='%s'\n", retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    }

    for (i=0; i<numEvents; i++) testResults[exp*numEvents+i]=values[i];

    // Run 2: No runs, should be static. 
    exp++;
    retval = PAPI_read( EventSet, values );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_read failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        if (numEvents==1) {printf("3rd PAPI_read failed; retval=%d='%s'\n", retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    }

    for (i=0; i<numEvents; i++) testResults[exp*numEvents+i]=values[i];

    // Run 3: Sleep 1 second and run, no invocations. See if time related.
    exp++;
    sleep(1); 
    retval = PAPI_read( EventSet, values );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_read failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        if (numEvents==1) {printf("4th PAPI_read failed; retval=%d='%s'\n", retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    }

    for (i=0; i<numEvents; i++) testResults[exp*numEvents+i]=values[i];

    // Run 4: Invoke the kernel five times (total will be 7 times, but one at 1/4, so expect 6.25 ratio);
    exp++;
    retval  = runBlackScholes(MAX_N);
    retval += runBlackScholes(MAX_N);
    retval += runBlackScholes(MAX_N);
    retval += runBlackScholes(MAX_N);
    retval += runBlackScholes(MAX_N);

    if (retval < 0) { 
        fprintf(stderr, "6 Kernel executions failed.\n");
        if (numEvents==1) {printf("Run 3 BlackScholes 5 kernel executions failed; retval=%d\n", retval); fflush(stdout);}
        exit (-2);
    }

    retval = PAPI_stop( EventSet, values );
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_stop failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        if (numEvents==1) {printf("PAPI_stop failed; retval=%d='%s'\n", retval, PAPI_strerror(retval)); fflush(stdout);}
        exit (-2);
    }

    for (i=0; i<numEvents; i++) testResults[exp*numEvents+i]=values[i];

    // Produce a final report. Final testResult format:
    // [exp0Event0, exp0Event1, ... exp1Event0, exp1Event1, ...]
    // So Event0 is at 0, numEvents, 2*numEvents, ...
    //    Event1 is at 1, numEvents+1, 2*numEvents+1, ...
    for (i=0; i<numEvents; i++) {
        int j;
        if (numEvents>1) printf("%s,", argv[i+1]);
        for (j=0; j<=exp; j++) printf("%8lld,", testResults[i+j*numEvents]);
        if (testResults[i] == 0) printf("Not Computed,Not Computed\n");
        else {
            printf("%2.3f,", ((double) testResults[i+    numEvents])/((double) testResults[i])); // .25 runs vs 1 run.
            printf("%2.3f\n", ((double) testResults[i+exp*numEvents])/((double) testResults[i])); // 5 runs vs 1 run.
        }
    }

    retval = PAPI_cleanup_eventset(EventSet);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        exit (-2);
    }

    retval = PAPI_destroy_eventset(&EventSet);
    if( retval != PAPI_OK ) {
        fprintf(stderr, "PAPI_start failed; retval=%d='%s'\n", retval, PAPI_strerror(retval));
        exit (-2);
    }

    free(values);
    free(testResults);
    free(events);

    PAPI_shutdown();
    return 0;
}
