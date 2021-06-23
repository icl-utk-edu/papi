/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    HelloWorld.c
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	<your name here>
 *			<your email address>
 * test case for Example component 
 * 
 *
 * @brief
 *  This file is a very simple HelloWorld C example which serves (together
 *	with its Makefile) as a guideline on how to add tests to components.
 *  The papi configure and papi Makefile will take care of the compilation
 *	of the component tests (if all tests are added to a directory named
 *	'tests' in the specific component dir).
 *	See components/README for more details.
 *
 *	The string "Hello World!" is mangled and then restored.
 */

#include <cuda.h>
#include <stdio.h>

#include "papi.h"
#include "papi_test.h"

#define NUM_EVENTS 1
#define PAPI 1
#define STEP_BY_STEP_DEBUG 0

// Prototypes
__global__ void helloWorld(char*);


// Host function
int main(int argc, char** argv)
{
	int quiet = 0;
    CUcontext getCtx=NULL, sessionCtx=NULL, primaryCtx=NULL;
    cudaError_t cudaError;
    CUresult cuError; (void) cuError;

    // We must do either a cuInit(), or get it done implicitly by cudaSetDevice().
    cudaError = cudaSetDevice(0);   // Set device to zero, implicit cuInit().
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i cudaSetDevice(0) Return Code: %s.\n", __FILE__, __func__, __LINE__, cudaGetErrorString(cudaError));

    cuCtxGetCurrent(&primaryCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i after cudaSetDevice(0), primaryCtx=%p.\n", __FILE__, __func__, __LINE__, primaryCtx);

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );
	
#ifdef PAPI
	int i, retval;
	int EventSet = PAPI_NULL;
	long long values[NUM_EVENTS];
	/* REPLACE THE EVENT NAME 'PAPI_FP_OPS' WITH A CUDA EVENT 
	   FOR THE CUDA DEVICE YOU ARE RUNNING ON.
	   RUN papi_native_avail to get a list of CUDA events that are 
	   supported on your machine */
        //char *EventName[] = { "PAPI_FP_OPS" };
        // char const *EventName[] = { "cuda:::fe__cycles_elapsed.sum:device=0"};
        char const *EventName[] = { "cuda:::dram__bytes_read.sum:device=0"};
	int events[NUM_EVENTS];
	int eventCount = 0;

    cuCtxCreate(&sessionCtx, 0, 0); // Create a context.
	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if( retval != PAPI_VER_CURRENT ) {
		if (!quiet) printf("PAPI init failed\n");
		test_fail(__FILE__,__LINE__,
			"PAPI_library_init failed", 0 );
	}

	if (!quiet) {
		printf( "PAPI_VERSION     : %4d %6d %7d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ) );
	}

//  cuCtxCreate(&sessionCtx, 0, 0); // Create a context.
    
    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i after PAPI_library_init() and cuCtxCreate() sessionCtx=%p, getCtx=%p.\n", __FILE__, __func__, __LINE__, sessionCtx, getCtx);
  
	/* convert PAPI native events to PAPI code */
	for( i = 0; i < NUM_EVENTS; i++ ){
                retval = PAPI_event_name_to_code( (char *)EventName[i], &events[i] );
		if( retval != PAPI_OK ) {
			fprintf(stderr, "%s:%s:%i PAPI_event_name_to_code failed for '%s'\n", __FILE__, __func__, __LINE__, EventName[i] );
			continue;
		}
		eventCount++;
		if (!quiet) printf( "Name %s --- Code: %#x\n", EventName[i], events[i] );
	}

	/* if we did not find any valid events, just report test failed. */
	if (eventCount == 0) {
		if (!quiet) printf( "Test FAILED: no valid events found.\n");
		test_skip(__FILE__,__LINE__,"No events found",0);
		return 1;
	}
	
    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	retval = PAPI_create_eventset( &EventSet );
	if( retval != PAPI_OK ) {
		if (!quiet) printf( "PAPI_create_eventset failed\n" );
		test_fail(__FILE__,__LINE__,"Cannot create eventset",retval);
	}	

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

        // If multiple GPUs/contexts were being used, 
        // you need to switch to each device before adding its events
        // e.g. cudaSetDevice( 0 );
	retval = PAPI_add_events( EventSet, events, eventCount );
	if( retval != PAPI_OK ) {
		fprintf( stderr, "PAPI_add_events failed\n" );
	}

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	retval = PAPI_start( EventSet );
	if( retval != PAPI_OK ) {
		fprintf( stderr, "PAPI_start failed\n" );
	}

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

#endif


	int j;
	
	// desired output
	char str[] = "Hello World!";

	// mangle contents of output
	// the null character is left intact for simplicity
	for(j = 0; j < 12; j++) {
		str[j] -= j;
	}

    printf("mangled str=%s\n", str);

	// allocate memory on the device
	char *d_str;
	size_t size = sizeof(str);
	cudaMalloc((void**)&d_str, size);
	
    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	// copy the string to the device
	cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);
	
    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	// set the grid and block sizes
	dim3 dimGrid(2); // one block per word
	dim3 dimBlock(6); // one thread per character

	// invoke the kernel
	helloWorld<<< dimGrid, dimBlock >>>(d_str);

    cudaError = cudaGetLastError();
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i Kernel Return Code: %s.\n", __FILE__, __func__, __LINE__, cudaGetErrorString(cudaError));

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	// retrieve the results from the device
	cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);
	
    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	// free up the allocated memory on the device
	cudaFree(d_str);
	
    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	if (!quiet) printf("END: %s\n", str);

	
#ifdef PAPI
	retval = PAPI_read( EventSet, values );
	if( retval != PAPI_OK )
		fprintf(stderr, "PAPI_read failed\n" );

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i after PAPI_read getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	for( i = 0; i < eventCount; i++ )
		if (!quiet) printf( "read: %12lld \t\t --> %s \n", values[i], EventName[i] );

	retval = PAPI_stop( EventSet, values );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_stop failed\n" );

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i after PAPI_stop getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	retval = PAPI_cleanup_eventset(EventSet);
	if( retval != PAPI_OK )
		fprintf(stderr, "PAPI_cleanup_eventset failed\n");

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i after PAPI_cleanup_eventset getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	retval = PAPI_destroy_eventset(&EventSet);
	if (retval != PAPI_OK)
		fprintf(stderr, "PAPI_destroy_eventset failed\n");

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i after PAPI_destroy_eventset getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	PAPI_shutdown();

    cuCtxGetCurrent(&getCtx);
    if (STEP_BY_STEP_DEBUG) fprintf(stderr, "%s:%s:%i after PAPI_shutdown getCtx=%p.\n", __FILE__, __func__, __LINE__, getCtx);

	for( i = 0; i < eventCount; i++ )
		if (!quiet) printf( "stop: %12lld \t\t --> %s \n", values[i], EventName[i] );
#endif

	test_pass(__FILE__);

    cuCtxDestroy(sessionCtx);

	return 0;
}


// Device kernel
__global__ void
helloWorld(char* str)
{
	// determine where in the thread grid we are
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// unmangle output
	str[idx] += idx;
}

