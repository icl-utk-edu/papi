/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    mpi_net_test.c
 *
 * @author  Rizwan Ashraf
 *          rizwan@icl.utk.edu
 * 
 * MPI-based test case for the infiniband, infiniband_umad and net components.
 * 
 * @brief
 *   This MPI-based application tests various events in the infiniband, infiniband_umad 
 *   and net components. In this application, the master process distributes workload to
 *   all other processes (NumProcs-1) and then receives the results of the corresponding 
 *   sub-computations. The expected behavior of this application is as follows:
 *   1. Master TX event ~= Sum of all RX events across all workers (NumProcs-1),
 *   2. Master RX event ~= Sum of all TX events across all workers (NumProcs-1).
 *   Usage: mpirun -n <NumProcs> ./mpi_net_test
 *   Warning: make sure NSIZE is divisible by NumProcs provided. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

#include "papi.h"
#include "papi_test.h"

#define NUM_EVENTS 2
#define NSIZE 400000000
#define PAPI

int main (int argc, char **argv)
{
    
    int i;

#ifdef PAPI
    int retval, ComponentID, NumComponents;
    long long Values[NUM_EVENTS];
    int EventSet = PAPI_NULL;
    /* infiniband component events: 
     * REPLACE THE EVENT NAMES 'infiniband:::mlx5_0_1:port_xmit_packets' 
     * 'infiniband:::mlx5_0_1:port_rcv_packets' WITH EVENTS REPORTED ON 
     * YOUR IB NETWORK/SYSTEM.
     * RUN papi_native_avail TO GET A LIST OF infiniband EVENTS THAT ARE 
     * SUPPORTED. */
    // e.g., mlx5_0_1 is the name of the IB port in the event name, 
    // which will most likely be different on your machine.      
    //char *EventName[] = { "infiniband:::mlx5_0_1:port_xmit_packets", "infiniband:::mlx5_0_1:port_rcv_packets" };  
    
    /* infiniband_umad component events:
     * REPLACE THE EVENT NAMES AS DIRECTED ABOVE */
    // Note: this component is known to produce erroneous results 
    //       in some cases. The component supports IB devices with
    //       OFED version 1.4 and below.
    //char *EventName[] = { "infiniband_umad:::mlx5_0_1_send", "infiniband_umad:::mlx5_0_1_recv" };

    /* net component events:
     * REPLACE/ADD EVENT NAMES BASED ON NUMBER OF ETHERNET INTERFACES */
    // Note: The use of net component to monitor IB interfaces is not recommended.
    char *EventName[] = { "net:::eth0:tx:packets", "net:::eth0:rx:packets" }; 
	
    int Events[NUM_EVENTS];
    int EventCount = 0;
 
    const PAPI_component_info_t *cmpinfo = NULL; 

    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__,"PAPI_library_init failed\n",retval);
    }

    NumComponents = PAPI_num_components();

    /* Check if any of the following component exists: infiniband, infiniband_umad, net */
    for(ComponentID=0; ComponentID<NumComponents; ComponentID++) {

        if ( (cmpinfo = PAPI_get_component_info(ComponentID)) == NULL) {
            test_fail(__FILE__, __LINE__,"PAPI_get_component_info failed\n",-1);
        }

        if ( (strstr(cmpinfo->name, "infiniband") == NULL) && (strstr(cmpinfo->name, "net") == NULL) ) {
            continue;
        }

        if (!TESTS_QUIET) {
            printf("Component %d (%d) - %d events - %s\n",
                ComponentID, cmpinfo->CmpIdx,
                cmpinfo->num_native_events, cmpinfo->name);
        }
        
        if (cmpinfo->disabled) {
            test_skip(__FILE__,__LINE__,"Component infiniband is disabled", 0);
            continue;
        }

        EventCount = cmpinfo->num_native_events;
    }

    /* if we did not find any valid events, just report test failed. */
    if (EventCount==0) {
        test_skip(__FILE__,__LINE__,"No events found for any of the following: infiniband, infiniband_umad, net.", 0);
    }

    /* convert PAPI native events to PAPI code */
    EventCount = 0; // re-initialize EventCount
    for( i = 0; i < NUM_EVENTS; i++ ) {
        retval = PAPI_event_name_to_code( EventName[i], &Events[i] );
        if ( retval != PAPI_OK ) {
            test_fail(__FILE__, __LINE__,"PAPI_event_name_to_code failed\n",retval);
        }
        EventCount++;
    }

    /* create eventset */
    retval = PAPI_create_eventset( &EventSet );
    if ( retval != PAPI_OK ) {
        test_fail(__FILE__, __LINE__,"PAPI_create_eventset failed\n",retval);
    }

    /* add all events to eventset */
    retval = PAPI_add_events( EventSet, Events, EventCount );
    if ( retval != PAPI_OK ) {
        test_fail(__FILE__, __LINE__,"PAPI_add_events failed\n",retval);
    }

#endif
    
    /* Set TESTS_QUIET variable */
    tests_quiet( argc, argv );

    if (!TESTS_QUIET) {
        printf("This test should trigger some network events\n");
    }
    
    int N = NSIZE;
    int NumProcs, Rank, N_per_Proc;

    /* Initialize MPI environment */
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &NumProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &Rank);

    /* check perfect divisibility based on provided input */
    if ( N % NumProcs != 0 ) {
        test_fail(__FILE__, __LINE__, "Please provide number of MPI processes to have perfect divisibility.\n", 0);
    }

    double * a;
    double * b;
    double * out;

    /* Master */
    if (Rank == 0) {
        a = (double *) malloc (sizeof(double) * N);
        b = (double *) malloc (sizeof(double) * N);
        out = (double *) malloc (sizeof(double) * N);

        if (!TESTS_QUIET) 
            printf("Master is initializing data...\n");
        
        /* do some FLOPS at the master */
        for (i = 0; i < N; i++) {
            a[i] = i*0.25;
            b[i] = i*0.75;
        }

        if (!TESTS_QUIET)
            printf("Master has successfully initialized arrays.\n");
    }

    double * aN;
    double * bN;
    double * outN;

    N_per_Proc = N/NumProcs;

    aN = (double *) malloc (sizeof(double)*N_per_Proc);
    bN = (double *) malloc (sizeof(double)*N_per_Proc);
    outN = (double *) malloc (sizeof(double)*N_per_Proc);
    
#ifdef PAPI
    retval = PAPI_start( EventSet );
    if ( retval != PAPI_OK ) {
        test_fail(__FILE__, __LINE__,"PAPI_start failed\n",retval);
    } 
#endif
    
    if ((!TESTS_QUIET) & (Rank == 0))
        printf("Master is scattering data to other workers.\n");

    MPI_Scatter (a, N_per_Proc, MPI_DOUBLE, aN, N_per_Proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter (b, N_per_Proc, MPI_DOUBLE, bN, N_per_Proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if ((!TESTS_QUIET) & (Rank == 0))
        printf("Master has successfully scattered data to other workers.\n");
    
    /* do FLOPS */
    for (i = 0; i < N_per_Proc; i++)
        outN [i] = aN [i] + bN [i];

    MPI_Gather (outN, N_per_Proc, MPI_DOUBLE, out, N_per_Proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if ((!TESTS_QUIET) & (Rank == 0))
        printf("Master has successfully gathered data from other workers.\n");

#ifdef PAPI
    retval = PAPI_stop( EventSet, Values );
    if ( retval != PAPI_OK ) {
        test_fail(__FILE__, __LINE__,"PAPI_stop failed\n",retval);
    }
    if (!TESTS_QUIET) {
        /* report event values */
        for( i = 0; i < EventCount; i++ )
            printf( "On rank %d: %12lld \t\t --> %s \n", Rank, Values[i], EventName[i] );
    }
    
    /* perform check on event values:
     * UNCOMMENT THE CODE BELOW ONLY WHEN USING 2 EVENTS, 
     * WHERE ONE EVENT CHECKS TX PACKET COUNT,
     * AND THE OTHER EVENT CHECKS RX PACKET COUNT. */ 
    /*
    int j;
    long long event_sum;
    long long *total = (long long*) malloc(sizeof(long long) * NumProcs);
    for ( i = 0; i < EventCount; i++ ) {
         MPI_Gather (&Values[i], 1, MPI_LONG_LONG, total, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
         if (Rank == 0) { // do check of event values on Master 
             event_sum = 0;
             for ( j = 1; j < NumProcs; j++ ) 
                  event_sum += total[j];

             if (event_sum == 0) {
                 printf( "Warning: %s event values at workers have zero values.\n", EventName[i]); 
	         continue;
             }
             if ( ( (i == 0) && (abs(event_sum - Values[1]) < 500) ) || 
                  ( (i == 1) && (abs(event_sum - Values[0]) < 500) ) ) 
                 printf( "The tx/rx event values at master/workers is as expected.\n"); 
	     else
                 printf( "Warning: tx/rx event values at master/workers do not have expected values.\n");
         }
    }     
    free (total);
    */
 
    retval = PAPI_cleanup_eventset (EventSet);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__,"PAPI_cleanup_eventset failed\n",retval);
    }
    retval = PAPI_destroy_eventset (&EventSet);
    if (retval != PAPI_OK) {
        test_fail(__FILE__, __LINE__,"PAPI_destroy_eventset failed\n",retval);
    }
#endif

   
    if (Rank == 0) {
        /* check results */
        for (i = 0; i < N; i++) {
            if ( fabs(out[i] - (a[i] + b[i])) < 0.00001 ) 
                continue; 
            else
                test_fail(__FILE__, __LINE__, "Master Node: Sanity check failed on floating point computation.\n", 0);
        }
        free (a); free (b); free (out);
    }
    free (aN); free (bN); free (outN);

    /* finialize MPI */
    MPI_Finalize();

    /* assume SUCCESS if you made it here */
    if ( Rank == 0 )
        test_pass( __FILE__ );

    return 0;
}
