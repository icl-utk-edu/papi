/* This file performs the following test: counter domain testing

   - It attempts to use the following two counters. It may use less depending on
     hardware counter resource limitations. 
     + PAPI_TOT_INS
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

#include "papi_test.h"

#ifdef _WIN32
#define TAB_DOM	"%s%12I64d%15I64d%17I64d\n"
#else
#define TAB_DOM	"%s%12lld%15lld%17lld\n"
#endif

extern int TESTS_QUIET;         /* Declared in test_utils.c */

int main(int argc, char **argv)
{
   int retval, num_tests = 3, tmp;
   long_long **values;
   int EventSet1, EventSet2, EventSet3;
   int num_events1, num_events2, num_events3;
   int mask1 = 0x3, mask2 = 0x3, mask3 = 0x3;
   PAPI_option_t options;
   char event_name[PAPI_MAX_STR_LEN], add_event_str[PAPI_MAX_STR_LEN];
#if defined(sgi) && defined(host_mips)
   uid_t id;
   id = getuid();
#endif

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET) {
      retval = PAPI_set_debug(PAPI_VERB_ECONT);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   }

   if ((retval = PAPI_query_event(PAPI_TOT_INS)) != PAPI_OK)
      test_skip(__FILE__, __LINE__, "PAPI_query_event", retval);

   retval = PAPI_event_code_to_name(PAPI_TOT_INS, event_name);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
   sprintf(add_event_str, "PAPI_add_event[%s]", event_name);


   memset(&options, 0x0, sizeof(options));

   EventSet1 = add_test_events(&num_events1, &mask1);
   EventSet2 = add_test_events(&num_events2, &mask2);
   EventSet3 = add_test_events(&num_events3, &mask3);

   /* num_events1 is equal to num_events2 so don't worry. */

   values = allocate_test_space(num_tests, num_events1);

   options.domain.eventset = EventSet1;
   options.domain.domain = PAPI_DOM_ALL;

   retval = PAPI_set_opt(PAPI_DOMAIN, &options);
   if (retval != PAPI_OK && retval != PAPI_ESBSTR)
      test_fail(__FILE__, __LINE__, "PAPI_set_opt", retval);

   options.domain.eventset = EventSet2;
   options.domain.domain = PAPI_DOM_KERNEL;

   retval = PAPI_set_opt(PAPI_DOMAIN, &options);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_opt", retval);

   options.domain.eventset = EventSet3;
   options.domain.domain = PAPI_DOM_USER;

   retval = PAPI_set_opt(PAPI_DOMAIN, &options);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_opt", retval);

   retval = PAPI_start(EventSet1);

   do_flops(NUM_FLOPS);

   if (retval == PAPI_OK) {
      retval = PAPI_stop(EventSet1, values[0]);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   } else {
      values[0][0] = retval;
      values[0][1] = retval;
   }

   retval = PAPI_start(EventSet2);

   do_flops(NUM_FLOPS);

   if (retval == PAPI_OK) {
      retval = PAPI_stop(EventSet2, values[1]);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   } else {
      values[1][0] = retval;
      values[1][1] = retval;
   }

   retval = PAPI_start(EventSet3);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet3, values[2]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   remove_test_events(&EventSet1, mask1);
   remove_test_events(&EventSet2, mask2);
   remove_test_events(&EventSet3, mask3);

   if (!TESTS_QUIET) {
      printf("Test case 2: Non-overlapping start, stop, read for all 3 domains.\n");
      printf("-----------------------------------------------------------------\n");
      tmp = PAPI_get_opt(PAPI_DEFDOM, NULL);
      printf("Default domain is: %d (%s)\n", tmp, stringify_domain(tmp));
      tmp = PAPI_get_opt(PAPI_DEFGRN, NULL);
      printf("Default granularity is: %d (%s)\n", tmp, stringify_granularity(tmp));
      printf("Using %d iterations of c += a*b\n", NUM_FLOPS);
      printf("-------------------------------------------------------------\n");

      printf("Test type   :   PAPI_DOM_ALL  PAPI_DOM_KERNEL  PAPI_DOM_USER\n");
      sprintf(add_event_str, "%s : ", event_name);
      printf(TAB_DOM, add_event_str, (values[0])[0], (values[1])[0], (values[2])[0]);
      printf(TAB_DOM, "PAPI_TOT_CYC: ", (values[0])[1], (values[1])[1], (values[2])[1]);
      printf("-------------------------------------------------------------\n");

      printf("Verification:\n");
      printf("Row 1 approximately equals N %d N\n", 0);
      printf("Column 1 approximately equals column 2 plus column 3\n");

#if defined(sgi) && defined(host_mips)
      printf("\n* IRIX requires root for PAPI_DOM_KERNEL and PAPI_DOM_ALL.\n");
      printf("* The first two columns will be -3 if not run as root for IRIX.\n");
#endif
   }
   {
#if defined(sgi) && defined(host_mips)
      long_long min, max;
      if (id != 0) {
         min = NUM_FLOPS * (1.0 - TOLERANCE);
         max = NUM_FLOPS * (1.0 + TOLERANCE);
         if (values[2][0] < min || values[2][0] > max)
            test_fail(__FILE__, __LINE__, event_name, 1);
      } else {
         min = (long_long) (values[2][0] * (1.0 - TOLERANCE));
         max = (long_long) (values[2][0] * (1.0 + TOLERANCE));
         if (values[0][0] > max || values[0][0] < min)
            test_fail(__FILE__, __LINE__, event_name, 1);

         min = (long_long) (values[0][1] * (1.0 - TOLERANCE));
         max = (long_long) (values[0][1] * (1.0 + TOLERANCE));
         if ((values[1][1] + values[2][1]) > max || (values[1][1] + values[2][1]) < min)
            test_fail(__FILE__, __LINE__, "PAPI_TOT_CYC", 1);
      }
#else
      long_long min, max;
      min = (long_long) (values[2][0] * (1.0 - TOLERANCE));
      max = (long_long) (values[2][0] * (1.0 + TOLERANCE));
      if (values[0][0] > max || values[0][0] < min)
         test_fail(__FILE__, __LINE__, event_name, 1);

      min = (long_long) (values[0][1] * (1.0 - TOLERANCE));
      max = (long_long) (values[0][1] * (1.0 + TOLERANCE));
      if ((values[1][1] + values[2][1]) > max || (values[1][1] + values[2][1]) < min)
         test_fail(__FILE__, __LINE__, "PAPI_TOT_CYC", 1);
#endif
   }
   test_pass(__FILE__, values, num_tests);
   exit(1);
}
