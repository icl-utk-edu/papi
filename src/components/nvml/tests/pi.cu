/*****************************************************************************
* This example shows how to use PAPI_add_event, PAPI_start, PAPI_read,       *
*  PAPI_stop, PAPI_remove_event and Nvidia's Management Library and uses CUDA device to calculate pi.        *
******************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <papi.h> /* This needs to be included every time you use PAPI */
#include <cuda.h>
#define NUM_EVENTS 1
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }
#define NBIN 1000  // Number of bins
#define NUM_BLOCK  1  // Number of thread blocks
#define NUM_THREAD  1  // Number of threads per block
int tid;
float pi = 0;

// CUDA kernel
__global__ void cal_pi(float *sum, int nbin, float step, int nthreads, int nblocks) {
    int i;
    float x;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
    for (i=idx; i< nbin; i+=nthreads*nblocks) {
        x = (i+0.5)*step;
        sum[idx] += 4.0/(1.0+x*x);
    }
}

int main()
{
   int EventSet = PAPI_NULL;
   /*must be initialized to PAPI_NULL before calling PAPI_create_event*/

   long long  values[NUM_EVENTS];
   /*This is where we store the values we read from the eventset */
    
   int retval,Events[NUM_EVENTS];
   /* We use number to keep track of the number of events in the EventSet */ 
   dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
    dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
    float *sumHost, *sumDev;  // Pointer to host & device arrays
    float step = 1.0/NBIN;  // Step size
    size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size
    sumHost = (float *)malloc(size);  //  Allocate array on host
    cudaMalloc((void **) &sumDev, size);  // Allocate array on device
    // Initialize array in device to 0
    cudaMemset(sumDev, 0, size);

   /*************************************************************************** 
   *  This part initializes the library and compares the version number of the*
   * header file, to the version of the library, if these don't match then it *
   * is likely that PAPI won't work correctly.If there is an error, retval    *
   * keeps track of the version number.                                       *
   ***************************************************************************/
   if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
      ERROR_RETURN(retval);
     
/* replace PAPI_TOT_CYC with the NVML event from papi_native_avail */
	if ( (retval = PAPI_event_name_to_code( "PAPI_TOT_CYC", &(Events[0]) )) < PAPI_OK)		fprintf( stderr, "PAPI_event_name_to_code %d\n", retval );

   /* Creating the eventset */              
   if ( (retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      ERROR_RETURN(retval);
	 if ( ( retval = PAPI_add_events( EventSet, ( int * ) Events, NUM_EVENTS ) ) < PAPI_OK )
    fprintf( stderr, "PAPI_library_add_eventset_failed %d\n", retval );

   /* Start counting */
   if ( (retval = PAPI_start(EventSet)) != PAPI_OK)
	{
		printf("code failed\n");
      ERROR_RETURN(retval);
    }
   /* you can replace your code here */
    // Do calculation on device
    cal_pi <<<dimGrid, dimBlock>>> (sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); 
//PAPI stop to stop counting
	if ( (retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      ERROR_RETURN(retval);

	// Retrieve result from device and store it in host array
    cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
    for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
        pi += sumHost[tid];
    pi *= step;

    // Print results
    printf("PI = %f\n",pi);

	//print PAPI counts 
   printf("Values returned by PAPI are %lld \n", values[0] );

   /* free the resources used by PAPI */
   PAPI_shutdown();
   // Cleanup to free device and host resources
    free(sumHost);
    cudaFree(sumDev);
   exit(0);
}











