/* This file performs the following test: overflow dispatch

   - This tests the dispatch of overflow calls from PAPI. These are counted 
   in the default counting domain and default granularity, depending on 
   the platform. Usually this is the user domain (PAPI_DOM_USER) and 
   thread context (PAPI_GRN_THR).

     The Eventset contains:
     + PAPI_FP_INS (overflow monitor)
     + PAPI_TOT_CYC

   - Set up overflow
   - Start eventset 1
   - Do flops
   - Stop eventset 1
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

#define THRESHOLD 1000000

int total = 0;

void handler(int EventSet, int EventCode, int EventIndex, long long *values, int *threshold, void *context)
{
#ifdef _CRAYT3E
  fprintf(stderr,"handler(%d, %x, %d, %lld, %d, %x) Overflow at %x!\n",
	  EventSet,EventCode,EventIndex,values[EventIndex],*threshold,context,PAPI_get_overflow_address(context));
#else
  fprintf(stderr,"handler(%d, %x, %d, %lld, %d, %p) Overflow at %p!\n",
	  EventSet,EventCode,EventIndex,values[EventIndex],*threshold,context,PAPI_get_overflow_address(context));
#endif
  total++;
}

int main(int argc, char **argv) 
{
  int EventSet, num_tests = 2, tmp, num_events, mask = 0x5;
  long long **values;
  int retval;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  if ((mask & 0x4) == 0x0)
    exit(1);

  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS*10);

  if (PAPI_stop(EventSet, values[0]) != PAPI_OK)
    exit(1);

  if (PAPI_overflow(EventSet, PAPI_FP_INS, THRESHOLD, 0, handler) != PAPI_OK)
    exit(1);

  if (PAPI_start(EventSet) != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS*10);

  if (PAPI_stop(EventSet, values[1]) != PAPI_OK)
    exit(1);

  printf("Test case 6: Overflow dispatch.\n");
  printf("-----------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Threshold for overflow is: %d\n",THRESHOLD);
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-----------------------------------------\n");

  printf("Test type   : \t1\t\t2\n");
  printf("PAPI_FP_INS : \t%lld\t%lld\n",
	 (values[0])[0],(values[1])[0]);
  printf("PAPI_TOT_CYC: \t%lld\t%lld\n",
	 (values[0])[1],(values[1])[1]);
  printf("Overflows   : \t%d\n",total);
  printf("-----------------------------------------\n");

  printf("Verification:\n");
  printf("Row 1 approximately equals %d %d\n",20*NUM_FLOPS,20*NUM_FLOPS);
  printf("Column 1 approximately equals column 2\n");
  printf("Row 3 approximate equals %d\n",(20*NUM_FLOPS)/THRESHOLD);

  remove_test_events(&EventSet, mask);

  free_test_space(values, num_tests);

  PAPI_shutdown();

  exit(0);
}
