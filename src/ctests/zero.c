/* This file performs the following test: start, stop and timer functionality

   - It attempts to use the following two counters. It may use less depending on
     hardware counter resource limitations. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is 
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
  int retval, num_tests = 1, tmp;
  int EventSet1;
  int mask1 = 0x5;
  int num_events1;
  long long **values;
  long long elapsed_us, elapsed_cyc;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  assert(retval >= PAPI_OK);

  retval = PAPI_thread_init(NULL, 0);
  assert(retval >= PAPI_OK);

  EventSet1 = add_test_events(&num_events1,&mask1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  retval = PAPI_start(EventSet1);
  assert(retval >= PAPI_OK);

  do_flops(NUM_FLOPS);
  
  retval = PAPI_accum(EventSet1, values[0]);
  assert(retval >= PAPI_OK);

  PAPI_shutdown();

  exit(0);
}
