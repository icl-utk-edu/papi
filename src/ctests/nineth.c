/* This file performs the following test: start, stop and timer functionality for derived events

   - It tests the derived metric FLOPS using the following two counters.
     They are counted in the default counting domain and default
     granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS
     + PAPI_TOT_CYC
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
  int retval, num_tests = 2, tmp;
  int EventSet1;
  int EventSet2;
  int mask1 = 0x5; /* FP_INS and TOT_CYC */
  int mask2 = 0x8; /* FLOPS */
  int num_events1;
  int num_events2;
  long long **values;
  int clockrate;
  double test_flops;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  assert(retval >= PAPI_OK);

  retval = PAPI_thread_init(NULL, 0);
  assert(retval >= PAPI_OK);

  EventSet1 = add_test_events(&num_events1,&mask1);
  EventSet2 = add_test_events(&num_events2,&mask2);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  clockrate = PAPI_get_opt(PAPI_GET_CLOCKRATE,NULL);
  assert(clockrate > 0);

  retval = PAPI_start(EventSet1);
  assert(retval >= PAPI_OK);

  do_flops(NUM_FLOPS);
  
  retval = PAPI_stop(EventSet1, values[0]);
  assert(retval >= PAPI_OK);

  retval = PAPI_start(EventSet2);
  assert(retval >= PAPI_OK);

  do_flops(NUM_FLOPS);
  
  retval = PAPI_stop(EventSet2, values[1]);
  assert(retval >= PAPI_OK);

  remove_test_events(&EventSet1, mask1);
  remove_test_events(&EventSet2, mask2);

  printf("Test case 9: start, stop for derived event PAPI_FLOPS.\n");
  printf("------------------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-------------------------------------------------------------------------\n");

  printf("Test type   : \t1\t\t2\n");
  printf("PAPI_FP_INS : \t%lld\t0\n",
	 (values[0])[0]);
  printf("PAPI_TOT_CYC: \t%lld\t0\n",
	 (values[0])[1]);
  printf("PAPI_FLOPS  : \t0\t\t%lld\n",
	 (values[1])[0]);
  printf("-------------------------------------------------------------------------\n");

  test_flops = (double)(values[0])[0]*(double)clockrate*(double)1000000.0;
  test_flops = test_flops / (double)(values[0])[1];
  printf("Verification:\n");
  printf("Last number in row 3 approximately equals %f\n",test_flops);

  free_test_space(values, num_tests);

  PAPI_shutdown();

  exit(0);
}
