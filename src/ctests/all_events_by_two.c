/* This file tries to add,start,stop all events in a substrate it
 * is meant not to test the accuracy of the mapping but to make sure
 * that all events in the substrate will at least start (Helps to
 * catch typos.
 *
 * Author: Kevin London
 *         london@cs.utk.edu
 * Mod: Rick Kufrin (added a second event to test all combinations)
 */
#include "papi_test.h"
extern int TESTS_QUIET; /* Declared in test_utils.c */

#define NUMEVENTS 2

int main(int argc, char **argv)
{
  int retval, i, j;
  int EventSet,count=0;
  long_long values[NUMEVENTS];
  const PAPI_preset_info_t *info = NULL;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ((info = PAPI_query_all_events_verbose()) == NULL)
        test_fail(__FILE__, __LINE__, "PAPI_query_all_events_verbose", 1);

  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  printf("Start\n");
  fflush(stdout);
  retval = PAPI_add_event(&EventSet,PAPI_L1_DCM);
  retval = PAPI_add_event(&EventSet,PAPI_L2_TCM);
  retval = PAPI_start(EventSet);
  printf("Start\n");
  fflush(stdout);
  retval = PAPI_stop(EventSet, values);
  printf("Stop\n");
  fflush(stdout);
  retval = PAPI_rem_event(&EventSet, PAPI_L1_DCM);
  retval = PAPI_rem_event(&EventSet, PAPI_L2_TCM);
  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++){ 
	if ( !(info[i].avail) ) continue;
	retval = PAPI_add_event(&EventSet,info[i].event_code);
  	if (retval != PAPI_OK) {
	    if ( !TESTS_QUIET ) 
		printf("Error adding %s\n", info[i].event_name );
 	    test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
	}
    for (j=i+1;j<PAPI_MAX_PRESET_EVENTS;j++) {
	  if ( !(info[j].avail) ) continue;
	  if ( !TESTS_QUIET ){
	    printf("Adding %s and %s: ", info[i].event_name, info[j].event_name );
        fflush(stdout);
 	  }
	  retval = PAPI_add_event(&EventSet,info[j].event_code);
  	  if (retval != PAPI_OK) {
	     if ( !TESTS_QUIET ) {
	            printf("succesful\n" );
                    fflush(stdout);
		    printf("Error adding %s\n", info[j].event_name );
 	     }
	     continue;
 	     test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);
	  }
	  else if ( !TESTS_QUIET ){
	        printf("succesful\n" );
            fflush(stdout);
	        count++;
	  }
  	  retval = PAPI_start(EventSet);
  	  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_start", retval);
      do_flops(NUM_FLOPS);
  	  retval = PAPI_stop(EventSet, values);
  	  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

      retval = PAPI_rem_event(&EventSet, info[j].event_code);
  	  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_rem_event", retval);
        }
      retval = PAPI_rem_event(&EventSet, info[i].event_code);
  }
  if ( !TESTS_QUIET ) 
	printf("Succesfully added,started and stopped %d event combinations.\n",count);
  test_pass(__FILE__, NULL, 0);
  exit(1);
}

