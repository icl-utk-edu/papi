/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
 * @file    test_nvml.c
 *
 *
 * @author  Rizwan Ashraf
 *          rizwan@icl.utk.edu
 *
 * Mods:    <your name here>
 *          <your email address>
 *
 *
 * Test all events in the NVML component 
 *        
 * @brief
 *      This code does simple vector addition on a *single* GPU.
 *      It automatically checks whether NVML component is enabled
 *      and correspondingly adds all available PAPI events for all GPUs.
 *      Afterwards, the event values are listed (if enabled) before and 
 *      after the GPU computation. The event values need to be checked 
 *      manually. The code does a simple test on the energy event 
 *      if available and will simply report whether after compute
 *      energy is higher than before compute energy. This code will 
 *      report passed once all PAPI functions have exited normally. 
 *          
 *      NOTE: 1. This code uses Unified Memory and thus requires
 *               CUDA version 6 or greater.
 *            2. In case multiple GPUs are available, this code 
 *               will only use a single GPU.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>

#include "papi.h"
#include "papi_test.h"

#define N 1<<25
#define PAPI
#define MAX_NVML_EVENTS 150

/* GPU Compute Kernel */
__global__ void vector_add (float *a, float *b, int n) {
   
   int i, index, stride;
   index = blockIdx.x * blockDim.x + threadIdx.x;
   stride = blockDim.x * gridDim.x;

   for (i = index; i < n; i += stride)
        a[i] = a[i] + b[i];
}

int main (int argc, char **argv) {

   int i;

   /* Set TESTS_QUIET variable */
   tests_quiet( argc, argv );

#ifdef PAPI
   int retVal, r, code;
   int ComponentID, NumComponents, NVML_ID;
   int EventSet = PAPI_NULL;
   int eventCount = 0;
   int eventNum = 0;
   long long *values;
   long long startEnergy = 0, endEnergy = 0;

   /* Note: these are fixed length arrays */ 
   char eventNames[MAX_NVML_EVENTS][PAPI_MAX_STR_LEN];
   char units[MAX_NVML_EVENTS][PAPI_MIN_STR_LEN];

   const PAPI_component_info_t *cmpInfo = NULL;
   PAPI_event_info_t eventInfo;
   
   /* for timing */
   long long startTime, endTime;
   double elapsedTime;

   /* PAPI Initialization */
   retVal = PAPI_library_init( PAPI_VER_CURRENT );
   if ( retVal != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__,"PAPI_library_init failed\n",retVal);
   }
   
   /* Get total components detected by PAPI */ 
   NumComponents = PAPI_num_components();

   /* Check if NVML component exists */
   for ( ComponentID = 0; ComponentID < NumComponents; ComponentID++ ) {
      
        if ( (cmpInfo = PAPI_get_component_info(ComponentID)) == NULL) {
            test_fail(__FILE__, __LINE__,"PAPI_get_component_info failed\n",-1);
        }

        if ( strstr(cmpInfo->name, "nvml") == NULL ) {
            continue;
        }
        
        /* if we are here, NVML component is found */
        if (!TESTS_QUIET) {
            printf("Component %d (%d) - %d events - %s\n",
                ComponentID, cmpInfo->CmpIdx,
                cmpInfo->num_native_events, cmpInfo->name);
        }

        if (cmpInfo->disabled) {
            test_skip(__FILE__,__LINE__,"NVML Component is disabled\n", 0);
            break;
        }

        eventCount = cmpInfo->num_native_events;
        NVML_ID = ComponentID;
        break;
   }
   
   /* if we did not find any valid events, skip the test. */
   if ( eventCount==0 ) {
        test_skip(__FILE__,__LINE__,"No events found for the NVML component\n", 0);
   }

   /* create EventSet */
   retVal = PAPI_create_eventset ( &EventSet );
   if (retVal != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed\n", retVal);
   }

   /* add all events to EventSet */
   code = PAPI_NATIVE_MASK;

   r = PAPI_enum_cmp_event ( &code, PAPI_ENUM_FIRST, NVML_ID );

   while ( r == PAPI_OK ) {
      
      retVal = PAPI_event_code_to_name (code, eventNames[eventNum]);
      if (retVal != PAPI_OK ) {
          test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name failed\n", retVal );
      }   
     
      retVal = PAPI_get_event_info (code, &eventInfo);
      if (retVal != PAPI_OK ) {
          test_fail(__FILE__, __LINE__, "Error getting event info\n", retVal);
      }

      strncpy(units[eventNum], eventInfo.units, sizeof(units[0])-1);
      units[eventNum][sizeof(units[0])-1] = '\0';

      retVal = PAPI_add_event (EventSet, code);
      if (retVal != PAPI_OK ) {
          break; /* all events have been added */
      } 
      eventNum++;

      /* go get next event */
      r = PAPI_enum_cmp_event (&code, PAPI_ENUM_EVENTS, NVML_ID);
      
   }   

   values = (long long*) calloc (eventNum, sizeof(long long));
   if (values == NULL ) {
       test_fail (__FILE__, __LINE__, "Memory allocation failed\n", -1);
   }
#endif

   float *a, *b;

   // allocate unified memory - this is visible in both host and GPU
   cudaMallocManaged ((void**) &a, sizeof(float) * N);
   cudaMallocManaged ((void**) &b, sizeof(float) * N);
       
   // initialize
   for (i = 0; i < N; i++) {
        a[i] = 10.0;
        b[i] = 2.0;
   }

#ifdef PAPI        
   /* Start Recording PAPI events */
   startTime = PAPI_get_real_nsec();
   retVal = PAPI_start (EventSet);
   if (retVal != PAPI_OK ) {
       test_fail(__FILE__, __LINE__, "PAPI start failed\n:", retVal);
   }
   
   /* read initial event values */
   retVal = PAPI_read (EventSet, values);
   if (retVal != PAPI_OK) {
       test_fail(__FILE__, __LINE__, "PAPI read failed\n", retVal);
   }
   
   if (!TESTS_QUIET) {
       printf("BEFORE GPU COMPUTE EVENT VALUES>>>\n");
       for ( i = 0; i < eventNum; i++) {
            printf("\t\t %12lld %s \t\t --> %s \n", values[i], units[i], eventNames[i]); 
            
            if (strstr(units[i], "J")) /* record last energy reading */
                startEnergy = values[i];
       } 
   }  
#endif
    
   int numBlocks, threadsPerBlock;      
   
   // Number of threads in a block
   threadsPerBlock = 1024;
   
   // total thread blocks in grid
   numBlocks = (N + threadsPerBlock - 1 )/threadsPerBlock;

   // Do GPU compute
   vector_add<<<numBlocks, threadsPerBlock>>>(a, b, N);
   
   // wait for GPU to finish 
   cudaDeviceSynchronize();

#ifdef PAPI
   /* Stop Recording PAPI events */
   endTime = PAPI_get_real_nsec();
   retVal = PAPI_stop (EventSet, values);
   if (retVal != PAPI_OK ) {
       test_fail(__FILE__, __LINE__, "PAPI stop failed:",retVal);
   }
   
   /* Done, clean up */
   retVal = PAPI_cleanup_eventset( EventSet );
   if (retVal != PAPI_OK) {
       test_fail(__FILE__, __LINE__,
                              "PAPI_cleanup_eventset failed:",retVal);
   }

   retVal = PAPI_destroy_eventset( &EventSet );
   if (retVal != PAPI_OK) {
       test_fail(__FILE__, __LINE__,
                              "PAPI_destroy_eventset failed:",retVal);
   }

   elapsedTime = ((double) (endTime-startTime))/1.0e9;

   if (!TESTS_QUIET) {
       printf("\nStopping PAPI measurements, the test took %.3fs...\n", elapsedTime);
        
       printf("ALL GPU EVENTS POST MEASUREMENT>>>\n"); 
       for (i=0; i < eventNum; i++) {
            printf("\t\t %12lld %s \t\t --> %s \n", values[i], units[i], eventNames[i]);
            
            /* if there are multiple devices, 
               this will record reading of the last device */ 
            if (strstr(units[i], "J")) 
                endEnergy = values[i];
       }

       if (endEnergy > startEnergy) { 
           printf("\nTEST PASSED:: Event values for energy measurements checks out.\n"
                  "Energy consumption post GPU compute is higher than initial reading.\n\n");
       }    
   }
#endif

   // perform sanity check on computation
   float maxErr = 0.0;
   for (i=0; i < N; i++) {
        maxErr = fmax (maxErr, fabs(a[i] - 12.0));
   }

   if (!TESTS_QUIET) {
       if (maxErr > 0.001) 
           printf("WARNING:: GPU Compute fails sanity check.\n"
                  "Max. error from GPU compute was %f.\n\n", maxErr);
   }   
 
   // free memory
   cudaFree (a); cudaFree (b); 
 
   /* assume SUCCESS if you made it here */
   test_pass( __FILE__ );

   return 0;

} // end main
