/* 
* File:    overflow_single_event.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* This file performs the following test: overflow dispatch of an eventset
   with just a single event. 

     The Eventset contains:
     + PAPI_FP_INS (overflow monitor)

   - Start eventset 1
   - Do flops
   - Stop and measure eventset 1
   - Set up overflow on eventset 1
   - Start eventset 1
   - Do flops
   - Stop eventset 1
*/

#include "papi_test.h"

#undef NUM_FLOPS

#define NUM_FLOPS 10000000
#define THRESHOLD  5000000

#ifdef _CRAYT3E
	#define OVER_FMT	"handler(%d, %x, %d, %lld, %d, %x) Overflow at %x!\n"
	#define OUT_FMT		"%-12s : %16lld%16lld\n"
#elif defined(_WIN32)
	#define OVER_FMT	"handler(%d, %x, %d, %I64d, %d, %p) Overflow at %p!\n"
	#define OUT_FMT		"%-12s : %16I64d%16I64d\n"
#else
	#define OVER_FMT	"handler(%d, %x, %d, %lld, %d, %p) Overflow at %p!\n"
	#define OUT_FMT		"%-12s : %16lld%16lld\n"
#endif

int total = 0;		/* total overflows */
extern int TESTS_QUIET; /* Declared in test_utils.c */

void handler(int EventSet, int EventCode, int EventIndex, long_long *values, int *threshold, void *context)
{
  if ( !TESTS_QUIET )
    fprintf(stderr,OVER_FMT,EventSet,EventCode,EventIndex,
	  values[EventIndex],*threshold,context,PAPI_get_overflow_address(context));
  total++;
}

int main(int argc, char **argv) 
{
  int EventSet;
  long_long values[2] = { 0, 0 };
  long_long min, max;
  int num_flops, retval;
  int PAPI_event;
  char event_name[PAPI_MAX_STR_LEN];

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

/* Skip Itanium till we fix the overflow problem -KSL */
#if defined(linux) && defined(__ia64__)
  test_pass(__FILE__, NULL, 0);
#endif
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

  /* query and set up the right instruction to monitor */
  if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK) PAPI_event = PAPI_FP_INS;
  else PAPI_event = PAPI_TOT_INS;

  retval = PAPI_create_eventset(&EventSet);
  if ( retval != PAPI_OK)  test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  retval = PAPI_add_event(&EventSet, PAPI_event);
  if ( retval != PAPI_OK)  test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

  retval = PAPI_start(EventSet);
  if ( retval != PAPI_OK)  test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS*10);

  retval = PAPI_stop(EventSet, &values[0]);
  if ( retval != PAPI_OK)  test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  retval = PAPI_overflow(EventSet, PAPI_event, THRESHOLD, 0, handler);
  if ( retval != PAPI_OK)  test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);

  retval = PAPI_start(EventSet);
  if ( retval != PAPI_OK)  test_fail(__FILE__, __LINE__, "PAPI_start", retval);

  do_flops(NUM_FLOPS*10);

  retval = PAPI_stop(EventSet, &values[1]);
  if ( retval != PAPI_OK)  test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  num_flops = 10*NUM_FLOPS;
#if defined(linux) || defined(__ia64__) || defined(_WIN32) || defined(_CRAYT3E)
  num_flops *= 2;
#endif

  if ( !TESTS_QUIET ) {
	if ((retval=PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
		test_fail(__FILE__,__LINE__,"PAPI_event_code_to_name",retval);

	printf("Test case: Overflow dispatch of 1st event in set with 1 event.\n");
	printf("--------------------------------------------------------------\n");
	printf("Threshold for overflow is: %d\n",THRESHOLD);
	printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
	printf("-----------------------------------------------\n");

	printf("Test type    : %16d%16d\n",1,2);
	printf(OUT_FMT, event_name,
	 values[0],values[1]);
	printf("Overflows    : %16d\n",total);
	printf("-----------------------------------------------\n");

	printf("Verification:\n");
	if (PAPI_event == PAPI_FP_INS)
		printf("Row 1 approximately equals %d %d\n", num_flops, num_flops);
	printf("Column 1 approximately equals column 2\n");
	printf(TAB1, "Row 3 approximately equals",(values[0])/THRESHOLD);
  }

  if (PAPI_event == PAPI_FP_INS) {
	  min = (long_long)(num_flops*.9);
	  max = (long_long)(num_flops*1.1);
	  if ( values[0] > max || values[0] < min || values[1] < min || values[1] >max)
		test_fail(__FILE__, __LINE__, event_name, 1);
  }
  min = (long_long)(values[0]*.9);
  max = (long_long)(values[0]*1.1);
  if ( values[1] > max || values[1] < min )
  	test_fail(__FILE__, __LINE__, event_name, 1);

  min = (long_long)((values[0]*.85)/THRESHOLD);
  max = (long_long)((values[0]*1.15)/THRESHOLD);
  if ( total > max || total < min )
  	test_fail(__FILE__, __LINE__, "Overflows", 1);

  test_pass(__FILE__,NULL,0);
  exit(1);
}
