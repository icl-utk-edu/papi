/* This file performs the following test: start, read, stop and again functionality

   - It attempts to use the following three counters. It may use less depending on
     hardware counter resource limitations. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS or PAPI_TOT_INS if PAPI_FP_INS doesn't exist
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

#include "papi_test.h"

int main(int argc, char **argv)
{
  int retval, num_tests = 6, num_events, tmp, i;
   long_long **values;
   int EventSet=PAPI_NULL;
   int PAPI_event, mask;
   char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_MAX_STR_LEN];
   const PAPI_hw_info_t *hw_info;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   hw_info = PAPI_get_hardware_info();
   if (hw_info == NULL)
     test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   /* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
      PAPI_TOT_INS, depending on the availability of the event on the
      platform */
   EventSet = add_two_events(&num_events, &PAPI_event, hw_info, &mask);

   retval = PAPI_event_code_to_name(PAPI_event, event_name);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
   sprintf(add_event_str, "PAPI_add_event[%s]", event_name);

   values = allocate_test_space(num_tests, num_events);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, values[0]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, values[1]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   retval = PAPI_reset(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_reset", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, values[2]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS/2);

   retval = PAPI_read(EventSet, values[3]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_read", retval);

   retval = PAPI_reset(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_reset", retval);

   do_flops(NUM_FLOPS/2);

   retval = PAPI_stop(EventSet, values[4]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   retval = PAPI_reset(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_reset", retval);

   retval = PAPI_read(EventSet, values[5]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_read", retval);

   remove_test_events(&EventSet, mask);

   printf("Test case: Start/Stop/Reset.\n");
   printf("----------------------------------------------------------------\n");
   printf("1 start,ops,stop\n");   
   printf("2 start,ops,stop of same eventset (should start from 0)\n");
   printf("3 reset,start,ops,stop (should start from 0, reset is redundant)\n");
   printf("4 start,ops/2,read \n");
   printf("5 reset,ops/2,stop (reset should affect running counters)\n");
   printf("6 reset (reset should affect stopped counters)\n");
   printf("----------------------------------------------------------------\n");
   tmp = PAPI_get_opt(PAPI_DEFDOM, NULL);
   printf("Default domain is: %d (%s)\n", tmp, stringify_all_domains(tmp));
   tmp = PAPI_get_opt(PAPI_DEFGRN, NULL);
   printf("Default granularity is: %d (%s)\n", tmp, stringify_granularity(tmp));
   printf("Using %d iterations of c += a*b\n", NUM_FLOPS);
   printf
     ("-------------------------------------------------------------------------\n");

      printf("Test type   :        1           2           3           4           5\n");
      sprintf(add_event_str, "%s:", event_name);
      printf(TAB5, add_event_str,
             (values[0])[0], (values[1])[0], (values[2])[0], (values[3])[0],
             (values[4])[0]);
      printf(TAB5, "PAPI_TOT_CYC:", (values[0])[1], (values[1])[1], (values[2])[1],
             (values[3])[1], (values[4])[1]);
      printf ("-------------------------------------------------------------------------\n"); 
   printf("Verification:\n");
   printf("Column 1 approximately equals column 2 and 3 \n");
   printf("Column 4 approximately equals 1/2 of column 3\n");
   printf("Column 5 approximately equals column 4\n");
   printf("%% difference between %s 1 & 2: %.2f\n",add_event_str,100.0*(float)(values[0])[0]/(float)(values[1])[0]);
   printf("%% difference between %s 1 & 2: %.2f\n","PAPI_TOT_CYC",100.0*(float)(values[0])[1]/(float)(values[1])[1]);

   for (i=0;i<=1;i++)
     {
       if (!approx_equals(values[0][i],values[1][i]))
         test_fail(__FILE__, __LINE__, ((i == 0) ? add_event_str : "PAPI_TOT_CYC"), 1);
       if (!approx_equals(values[1][i],values[2][i]))
         test_fail(__FILE__, __LINE__, ((i == 0) ? add_event_str : "PAPI_TOT_CYC"), 1);
       if (!approx_equals(values[3][i],values[4][i]))
         test_fail(__FILE__, __LINE__, ((i == 0) ? add_event_str : "PAPI_TOT_CYC"), 1);
       if (!approx_equals(values[2][i],values[3][i]*2.0))
	 test_fail(__FILE__, __LINE__, ((i == 0) ? add_event_str : "PAPI_TOT_CYC"), 1);
       if (values[5][i] != 0LL)
         test_fail(__FILE__, __LINE__, ((i == 0) ? add_event_str : "PAPI_TOT_CYC"), 1);
     }
	   
   test_pass(__FILE__, values, num_tests);
   exit(1);
}
