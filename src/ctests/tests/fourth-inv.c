/* This file performs the following test: nested eventsets that share 
   all counter values, but added in reverse order.

   - It attempts to use two eventsets simultaneously. These are counted 
   in the default counting domain and default granularity, depending on 
   the platform. Usually this is the user domain (PAPI_DOM_USER) and 
   thread context (PAPI_GRN_THR).

     Eventset 1 and 2 both have:
     + PAPI_FP_INS
     + PAPI_TOT_CYC

   - Start eventset 1
   - Do flops
   - Stop eventset 1
   - Start eventset 1
   - Do flops
   - Start eventset 2
   - Do flops
   - Stop and read eventset 2
   - Do flops
   - Stop and read eventset 1
   - Start eventset 2
   - Do flops
   - Stop eventset 2
*/

#include <stdlib.h>
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

int main(int argc, char **argv) 
{
  int retval, num_tests = 4, tmp;
  long long **values;
  int EventSet1, EventSet2;
  int num_events = 2;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  if (PAPI_create_eventset(&EventSet1) != PAPI_OK)
    exit(1);
  if (PAPI_create_eventset(&EventSet2) != PAPI_OK)
    exit(1);

  if (PAPI_add_event(&EventSet1, PAPI_TOT_CYC) != PAPI_OK)
    exit(1);
  if (PAPI_add_event(&EventSet1, PAPI_FP_INS) != PAPI_OK)
    exit(1);
  if (PAPI_add_event(&EventSet2, PAPI_FP_INS) != PAPI_OK)
    exit(1);
  if (PAPI_add_event(&EventSet2, PAPI_TOT_CYC) != PAPI_OK)
    exit(1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events);

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);
  
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK)
    exit(1);

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);
  
  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet2, values[1]);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet1, values[2]);
  if (retval != PAPI_OK)
    exit(1);

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);
  
  retval = PAPI_stop(EventSet2, values[3]);
  if (retval != PAPI_OK)
    exit(1); 

  printf("Test case 4: Overlapping start and stop of 2 eventsets.\n");
  printf("-------------------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-------------------------------------------------------------------------\n");

  printf("Test type   : \t1\t\t2\t\t3\t\t4\n");
  printf("PAPI_FP_INS : \t%lld\t%lld\t%lld\t%lld\n",
	 (values[0])[1],(values[1])[0],(values[2])[1],(values[3])[0]);
  printf("PAPI_TOT_CYC: \t%lld\t%lld\t%lld\t%lld\n",
	 (values[0])[0],(values[1])[1],(values[2])[0],(values[3])[1]);
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Column 1 approximately equals column 2\n");
  printf("Column 3 approximately equals three times column 2\n");
  printf("Column 4 approximately equals column 2\n");

  free_test_space(values, num_tests);

  PAPI_shutdown();
  
  exit(0);
}
