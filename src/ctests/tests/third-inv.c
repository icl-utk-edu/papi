/* This file performs the following test: nested eventsets that do not share any counter values

   - It attempts to use two eventsets simultaneously. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is the user domain 
     (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).

     Eventset 1 has:
     + PAPI_TOT_CYC

     EventSet 2 has:
     + PAPI_FP_INS

   - Start eventset 1
   - Do flops
   - Stop eventset 1
   - Start eventset 1
   - Do flops
   - Start eventset 2
   - Read eventset 1
   - Do flops
   - Stop and read eventset 2
   - Do flops
   - Stop and read eventset 1
   - Start eventset 2
   - Do flops
   - Stop eventset 2
*/

#include "papi_test.h"

int main(int argc, char **argv) 
{
  int retval, num_tests = 5, tmp;
  long_long **values;
  int EventSet1, EventSet2;
  int mask1 = 0x4, mask2 = 0x1;
  int num_events1, num_events2;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  EventSet1 = add_test_events(&num_events1,&mask1);
  EventSet2 = add_test_events(&num_events2,&mask2);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

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

  retval = PAPI_read(EventSet1, values[1]);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet2, values[2]);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);

  retval = PAPI_stop(EventSet1, values[3]);
  if (retval != PAPI_OK)
    exit(1);

  retval = PAPI_start(EventSet2);
  if (retval != PAPI_OK)
    exit(1);

  do_flops(NUM_FLOPS);
  
  retval = PAPI_stop(EventSet2, values[4]);
  if (retval != PAPI_OK)
    exit(1); 

  remove_test_events(&EventSet1, mask1);
  remove_test_events(&EventSet2, mask2);

  printf("Test case 3i: Overlapping start and stop of 2 eventsets with different counters.\n");
  printf("-------------------------------------------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-------------------------------------------------------------------------\n");

  printf("Test type   : \t1\t\t2\t\t3\t\t4\t\t5\n");
  printf(TAB5, "PAPI_TOT_CYC: ",
	 (long_long)0,(long_long)0,(values[2])[0],(long_long)0,(values[4])[0]);
  printf(TAB5, "PAPI_FP_INS : ",
	 (values[0])[0],(values[1])[0],(long_long)0,(values[3])[0],(long_long)0);
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Row 1 approximately equals %d %d N %d N\n",0,0,0);
  printf("Row 2 approximately equals X X %d X %d\n",0,0);
  printf("Column 1 approximately equals column 2\n");
  printf("Column 4 approximately equals three times column 1\n");
  printf("Column 5 approximately equals column 3\n");

  free_test_space(values, num_tests);

  PAPI_shutdown();
  
  exit(0);
}
