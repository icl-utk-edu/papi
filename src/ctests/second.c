/* This file performs the following test: counter domain testing

   - It attempts to use the following two counters. It may use less depending on
     hardware counter resource limitations. 
     + PAPI_FP_INS
     + PAPI_TOT_CYC
   - Start system domain counters
   - Do flops
   - Stop and read system domain counters
   - Start kernel domain counters
   - Do flops
   - Stop and read kernel domain counters
   - Start user domain counters
   - Do flops
   - Stop and read user domain counters
*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <memory.h>
#include <sys/types.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

int main(int argc, char **argv) 
{
  int retval, num_tests = 3, tmp;
  long long **values;
  int EventSet1, EventSet2, EventSet3;
  int num_events1, num_events2, num_events3;
  int mask1 = 0x5, mask2 = 0x5, mask3 = 0x5;
  PAPI_option_t options;

  memset(&options,0x0,sizeof(options));

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  assert(retval >= PAPI_OK);

  EventSet1 = add_test_events(&num_events1,&mask1);
  EventSet2 = add_test_events(&num_events2,&mask2);
  EventSet3 = add_test_events(&num_events3,&mask3);

  /* num_events1 is equal to num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  options.domain.eventset=EventSet1;
  options.domain.domain=PAPI_DOM_ALL;
  retval = PAPI_set_opt(PAPI_SET_DOMAIN, &options);
  assert(retval >= PAPI_OK);

  options.domain.eventset=EventSet2;
  options.domain.domain=PAPI_DOM_KERNEL;
  retval = PAPI_set_opt(PAPI_SET_DOMAIN, &options);
  assert(retval >= PAPI_OK);

  options.domain.eventset=EventSet3;
  options.domain.domain=PAPI_DOM_USER;
  retval = PAPI_set_opt(PAPI_SET_DOMAIN, &options);
  assert(retval >= PAPI_OK);

  retval = PAPI_start(EventSet1);

  do_flops(NUM_FLOPS);

  if (retval == PAPI_OK)
    retval = PAPI_stop(EventSet1, values[0]);
  else
    { values[0][0] = retval; values[0][1] = retval; }

  retval = PAPI_start(EventSet2);

  do_flops(NUM_FLOPS);

  if (retval == PAPI_OK)
    retval = PAPI_stop(EventSet2, values[1]);
  else
    { values[1][0] = retval; values[1][1] = retval; }

  retval = PAPI_start(EventSet3);

  do_flops(NUM_FLOPS);

  if (retval == PAPI_OK)
    retval = PAPI_stop(EventSet3, values[2]);
  else
    { values[2][0] = retval; values[2][1] = retval; }

  remove_test_events(&EventSet1, mask1);
  remove_test_events(&EventSet2, mask2);
  remove_test_events(&EventSet3, mask3);

  printf("Test case 2: Non-overlapping start, stop, read for all 3 domains.\n");
  printf("-----------------------------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-------------------------------------------------------------\n");

  printf("Test type   : \tPAPI_DOM_ALL\tPAPI_DOM_KERNEL\tPAPI_DOM_USER\n");
  printf("PAPI_FP_INS : \t%lld\t%lld\t\t%lld\n",
	 (values[0])[0],(values[1])[0],(values[2])[0]);
  printf("PAPI_TOT_CYC: \t%lld\t%lld\t\t%lld\n",
	 (values[0])[1],(values[1])[1],(values[2])[1]);
  printf("-------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Row 1 approximately equals %d %d %d\n",2*NUM_FLOPS,0,2*NUM_FLOPS);
  printf("Column 1 approximately equals column 2 plus column 3\n");

  free_test_space(values, num_tests);

  PAPI_shutdown();

  exit(0);
}
