#define ITERS 1000
#define INTLEN 100000 /* check vs. do_loops.c */
#define STRIDE 33

/* This file performs the following test: start, stop and timer functionality for 
   PAPI_L1_TCM derived event

   - They are counted in the default counting domain and default
     granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

int main() 
{
  int retval, num_tests = 3, tmp;
  int EventSet1;
  int EventSet2;
  int EventSet3;
  int mask1 = 0x10; /* PAPI_L1_TCM */
  int mask2 = 0x20; /* PAPI_L1_ICM */
  int mask3 = 0x40; /* PAPI_L1_DCM */
  int num_events1;
  int num_events2;
  int num_events3;
  long long **values;

  EventSet1 = add_test_events(&num_events1,&mask1);
  EventSet2 = add_test_events(&num_events2,&mask2);
  EventSet3 = add_test_events(&num_events3,&mask3);

  values = allocate_test_space(num_tests, 1);

  /* Warm me up */

  do_l1misses(ITERS,INTLEN,STRIDE);

  retval = PAPI_start(EventSet1);
  assert(retval >= PAPI_OK);

  do_l1misses(ITERS,INTLEN,STRIDE);
  
  retval = PAPI_stop(EventSet1, values[0]);
  assert(retval >= PAPI_OK);

  retval = PAPI_start(EventSet2);
  assert(retval >= PAPI_OK);

  do_l1misses(ITERS,INTLEN,STRIDE);
  
  retval = PAPI_stop(EventSet2, values[1]);
  assert(retval >= PAPI_OK);

  retval = PAPI_start(EventSet3);
  assert(retval >= PAPI_OK);

  do_l1misses(ITERS,INTLEN,STRIDE);
  
  retval = PAPI_stop(EventSet3, values[2]);
  assert(retval >= PAPI_OK);

  remove_test_events(&EventSet1, mask1);
  remove_test_events(&EventSet2, mask2);
  remove_test_events(&EventSet3, mask3);

  printf("Test case 9: start, stop for derived event PAPI_FLOPS.\n");
  printf("------------------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Using %d iterations of c = a*b\n",NUM_FLOPS);
  printf("-------------------------------------------------------------------------\n");

  printf("Test type   : \t1\t\t2\t\t3\n");
  printf("PAPI_L1_TCM : \t%lld\t0\t\t0\n",
	 (values[0])[0]);
  printf("PAPI_L1_ICM : \t0\t\t%lld\t\t0\n",
	 (values[1])[0]);
  printf("PAPI_L1_DCM : \t0\t\t0\t\t%lld\n",
	 (values[2])[0]);
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("First number row 1 approximately equals (1,1) + (2,2) or %lld\n",(values[1])[0]+(values[2])[0]);

  free_test_space(values, num_tests);

  PAPI_shutdown();

  exit(0);
}
