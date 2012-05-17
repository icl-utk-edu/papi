/** 
 * @author  Vince Weaver
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "papi.h"

#define NUM_EVENTS 6

char events[NUM_EVENTS][BUFSIZ]={
  "PACKAGE_ENERGY:PACKAGE0",
  "PACKAGE_ENERGY:PACKAGE1",
  "DRAM_ENERGY:PACKAGE0",
  "DRAM_ENERGY:PACKAGE1",
  "PP0_ENERGY:PACKAGE0",
  "PP0_ENERGY:PACKAGE1",
};

char filenames[NUM_EVENTS][BUFSIZ]={
  "results.PACKAGE_ENERGY_PACKAGE0",
  "results.PACKAGE_ENERGY_PACKAGE1",
  "results.DRAM_ENERGY_PACKAGE0",
  "results.DRAM_ENERGY_PACKAGE1",
  "results.PP0_ENERGY_PACKAGE0",
  "results.PP0_ENERGY_PACKAGE1",
};

FILE *fff[NUM_EVENTS];


int main (int argc, char **argv)
{

    int retval,cid,rapl_cid=-1,numcmp;
    int EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];
    int i;
    const PAPI_component_info_t *cmpinfo = NULL;
    long long start_time,before_time,after_time;
    double elapsed_time,total_time;

    

	/* PAPI Initialization */
     retval = PAPI_library_init( PAPI_VER_CURRENT );
     if ( retval != PAPI_VER_CURRENT ) {
        fprintf(stderr,"PAPI_library_init failed\n");
	exit(1);
     }

     numcmp = PAPI_num_components();

     for(cid=0; cid<numcmp; cid++) {

	if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
	   fprintf(stderr,"PAPI_get_component_info failed\n");
	   exit(1);
	}

	if (strstr(cmpinfo->name,"linux-rapl")) {
	   rapl_cid=cid;
	   printf("Found rapl component at cid %d\n", rapl_cid);

           if (cmpinfo->num_native_events==0) {
	     fprintf(stderr,"No rapl events found\n");
	     exit(1);
           }
	   break;
	}
     }

     /* Component not found */
     if (cid==numcmp) {
        fprintf(stderr,"No rapl component found\n");
        exit(1);
     }


     /* Open output files */
     for(i=0;i<NUM_EVENTS;i++) {
        fff[i]=fopen(filenames[i],"w");
	if (fff[i]==NULL) {
	   fprintf(stderr,"Could not open %s\n",filenames[i]);
	   exit(1);
	}
     }
				   

     /* Create EventSet */
     retval = PAPI_create_eventset( &EventSet );
     if (retval != PAPI_OK) {
        fprintf(stderr,"Error creating eventset!\n");
     }

     for(i=0;i<NUM_EVENTS;i++) {
	
        retval = PAPI_add_named_event( EventSet, events[i] );
        if (retval != PAPI_OK) {
	   fprintf(stderr,"Error adding event %s\n",events[i]);
	}
     }

  

     start_time=PAPI_get_real_nsec();

     while(1) {

        /* Start Counting */
        before_time=PAPI_get_real_nsec();
        retval = PAPI_start( EventSet);
        if (retval != PAPI_OK) {
           fprintf(stderr,"PAPI_start() failed\n");
	   exit(1);
        }


        usleep(100000);

        /* Stop Counting */
        after_time=PAPI_get_real_nsec();
        retval = PAPI_stop( EventSet, values);
        if (retval != PAPI_OK) {
           fprintf(stderr, "PAPI_start() failed\n");
        }

        total_time=((double)(after_time-start_time))/1.0e9;
        elapsed_time=((double)(after_time-before_time))/1.0e9;

        for(i=0;i<NUM_EVENTS;i++) {

	   fprintf(fff[i],"%.4f %.1f (* Average Power for %s *)\n",
		   total_time,
		   ((double)values[i]/1.0e9)/elapsed_time,
		   events[i]);
	   fflush(fff[i]);
        }
     }
		
     return 0;
}

