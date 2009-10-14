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

extern int TESTS_QUIET;         /* Declared in test_utils.c */

int main(int argc, char **argv)
{
   int retval, num_tests = 5, num_events, tmp;
   long long **values;
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

   retval = PAPI_read(EventSet, values[0]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_read", retval);

   retval = PAPI_reset(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_reset", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_read(EventSet, values[1]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_read", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_read(EventSet, values[2]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_read", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, values[3]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   retval = PAPI_read(EventSet, values[4]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_read", retval);

   remove_test_events(&EventSet, mask);

   if (!TESTS_QUIET) {
      printf("Test case 1: Non-overlapping start, stop, read.\n");
      printf("-----------------------------------------------\n");
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
      printf("Row 1 Column 1 at least 90%% of %d\n", 2*NUM_FLOPS);
      printf("%% difference between %s 1 & 2: %.2f\n",add_event_str,100.0*(float)(values[0])[0]/(float)(values[1])[0]);
      printf("%% difference between %s 1 & 2: %.2f\n","PAPI_TOT_CYC",100.0*(float)(values[0])[1]/(float)(values[1])[1]);
      printf("Column 1 approximately equals column 2\n");
      printf("Column 3 approximately equals 2 * column 2\n");
      printf("Column 4 approximately equals 3 * column 2\n");
      printf("Column 4 exactly equals column 5\n");
   }

   {
      long long min, max;
      min = (long long) (values[1][0] * .9);
      max = (long long) (values[1][0] * 1.1);

      if (values[0][0] > max || values[0][0] < min || values[2][0] > (2 * max)
          || values[2][0] < (2 * min) || values[3][0] > (3 * max)
          || values[3][0] < (3 * min)
          || values[3][0] != values[4][0]
          || values[0][0] < (long long)(1.8*NUM_FLOPS)) {
/*
         printf("min: ");
         printf(LLDFMT, min);
         printf("max: ");
         printf(LLDFMT, max);
         printf("1st: ");
         printf(LLDFMT, values[0][0]);
         printf("2nd: ");
         printf(LLDFMT, values[1][0]);
         printf("3rd: ");
         printf(LLDFMT, values[2][0]);
         printf("4th: ");
         printf(LLDFMT, values[3][0]);
         printf("5th: ");
         printf(LLDFMT, values[4][0]);
         printf("\n");
*/         test_fail(__FILE__, __LINE__, event_name, 1);
      }

      min = (long long) (values[1][1] * .8);
      max = (long long) (values[1][1] * 1.2);
      if (values[0][1] > max || values[0][1] < min || values[2][1] > (2 * max)
          || values[2][1] < (2 * min) || values[3][1] > (3 * max)
          || values[3][1] < (3 * min)
          || values[3][1] != values[4][1]) {
         test_fail(__FILE__, __LINE__, "PAPI_TOT_CYC", 1);
      }
   }
   test_pass(__FILE__, values, num_tests);
   exit(1);
}
