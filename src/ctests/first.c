/* This file performs the following test: start, read, stop and again functionality

   - It attempts to use the following three counters. It may use less depending on
     hardware counter resource limitations. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS
     + PAPI_TOT_CYC
   - Start counters
   - Do flops
   - Read counters
   - Reset counters
   - Do flops
   - Read counters
   - Do flops
   - Read counters
   - Do flops
   - Stop and read counters
   - Read counters
*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

int main() 
{
  int retval, num_tests = 5, num_events, mask = 0x5, tmp;
  long long **values;
  int EventSet;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  retval = PAPI_start(EventSet);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);

  retval = PAPI_read(EventSet, values[0]);
  if (retval != PAPI_OK)
    exit(1);

  retval = PAPI_reset(EventSet);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);

  retval = PAPI_read(EventSet, values[1]);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);

  retval = PAPI_read(EventSet, values[2]);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet, values[3]);
  if (retval != PAPI_OK)
    exit(1);

  retval = PAPI_read(EventSet, values[4]);
  if (retval != PAPI_OK)
    exit(1);

  remove_test_events(&EventSet, mask);

  printf("Test case 1: Non-overlapping start, stop, read.\n");
  printf("-----------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-------------------------------------------------------------------------\n");

  printf("Test type   : \t1\t\t2\t\t3\t\t4\t\t5\n");
  printf("PAPI_FP_INS : \t%lld\t%lld\t%lld\t%lld\t%lld\n",
	 (values[0])[0],(values[1])[0],(values[2])[0],(values[3])[0],(values[4])[0]);
  printf("PAPI_TOT_CYC: \t%lld\t%lld\t%lld\t%lld\t%lld\n",
	 (values[0])[1],(values[1])[1],(values[2])[1],(values[3])[1],(values[4])[1]);
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Column 1 approximately equals column 2\n");
  printf("Column 3 approximately equals 2 * column 2\n");
  printf("Column 4 approximately equals 3 * column 2\n");
  printf("Column 4 exactly equals column 5\n");

  free_test_space(values, num_tests);

  PAPI_shutdown();

  exit(0);
}
