/*
 * $Id$
 *
 * Test example for multiplex functionality, originally 
 * provided by Timothy Kaiser, SDSC. It was modified to fit the 
 * PAPI test suite by Nils Smeds, <smeds@pdc.kth.se>.
 *
 * This example verifies the accuracy of multiplexed events
 */

#include "papi_test.h"
#include <stdio.h>
#include <math.h>

#define REPEATS 5
#define MAXEVENTS 9
#define SLEEPTIME 100
#define MINCOUNTS 100000

static double dummy3(double x, int iters);

int main(int argc, char **argv)
{
   PAPI_event_info_t info;
   int i, j, retval;
   int iters = NUM_FLOPS;
   double x = 1.1, y;
   long_long t1, t2;
   long_long values[MAXEVENTS], refvalues[MAXEVENTS];
   int sleep_time = SLEEPTIME;
   double valsqsum[MAXEVENTS];
   double valsum[MAXEVENTS];
   double spread[MAXEVENTS];
   int nevents = MAXEVENTS;
   int eventset = PAPI_NULL;
   int events[MAXEVENTS];

   events[0] = PAPI_FP_INS;
   events[1] = PAPI_TOT_INS;
   events[2] = PAPI_INT_INS;
   events[3] = PAPI_TOT_CYC;
   events[4] = PAPI_STL_CCY;
   events[5] = PAPI_BR_INS;
   events[6] = PAPI_SR_INS;
   events[7] = PAPI_LD_INS;
   events[8] = PAPI_TOT_IIS;

   for (i = 0; i < MAXEVENTS; i++) {
      values[i] = 0;
      valsqsum[i] = 0;
      valsum[i] = 0;
   }

   if (argc > 1) {
      if (!strcmp(argv[1], "TESTS_QUIET"))
         tests_quiet(argc, argv);
      else {
         sleep_time = atoi(argv[1]);
         if (sleep_time <= 0)
            sleep_time = SLEEPTIME;
      }
   }

   if (!TESTS_QUIET) {
      printf("\nAccuracy check of multiplexing routines.\n");
      printf("Comparing a multiplex measurement with separate measurements.\n\n");
   }

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if ((retval = PAPI_create_eventset(&eventset)))
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);
#ifdef MPX
   if ((retval = PAPI_multiplex_init()))
      test_fail(__FILE__, __LINE__, "PAPI_multiplex_init", retval);

   /* In Component PAPI, EventSets must be assigned a component index
      before you can fiddle with their internals.
      0 is always the cpu component */
   retval = PAPI_assign_eventset_component(eventset, 0);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_assign_eventset_component", retval);

   if ((retval = PAPI_set_multiplex(eventset)))
      test_fail(__FILE__, __LINE__, "PAPI_set_multiplex", retval);
#endif

   nevents = MAXEVENTS;
   for (i = 0; i < nevents; i++) {
      if ((retval = PAPI_add_event(eventset, events[i]))) {
         for (j = i; j < (MAXEVENTS - 1); j++)
            events[j] = events[j + 1];
         nevents--;
         i--;
      }
   }
   if (nevents < 2)
      test_skip(__FILE__, __LINE__, "Not enough events left...", 0);

   /* Find a reasonable number of iterations (each 
    * event active 20 times) during the measurement
    */
   t2 = 10000 * 20 * nevents;   /* Target: 10000 usec/multiplex, 20 repeats */
   if (t2 > 30e6)
      test_skip(__FILE__, __LINE__, "This test takes too much time", retval);

   /* Measure one run */
   t1 = PAPI_get_real_usec();
   y = dummy3(x, iters);
   t1 = PAPI_get_real_usec() - t1;

   if (t2 > t1)                 /* Scale up execution time to match t2 */
      iters = iters * ((int)(t2 / t1));
   else if (t1 > 30e6)          /* Make sure execution time is < 30s per repeated test */
      test_skip(__FILE__, __LINE__, "This test takes too much time", retval);

   x = 1.0;

   if (!TESTS_QUIET)
      printf("\nFirst run: Multiplexed.\n");

   t1 = PAPI_get_real_usec();
   if ((retval = PAPI_start(eventset)))
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   y = dummy3(x, iters);
   if ((retval = PAPI_stop(eventset, values)))
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   t2 = PAPI_get_real_usec();

   if (!TESTS_QUIET) {
      printf("\tOperations= %.1f Mflop", y * 1e-6);
      printf("\t(%g Mflop/s)\n\n", ((float) y / (t2 - t1)));
      printf("PAPI multiplexed measurement:\n");
   }
   for (j = 0; j < nevents; j++) {
      PAPI_get_event_info(events[j], &info);
      if (!TESTS_QUIET) {
         printf("%20s = ", info.short_descr);
         printf(LLDFMT, values[j]);
         printf("\n");
      }
   }
   if (!TESTS_QUIET)
      printf("\n");


   if ((retval = PAPI_remove_events(eventset, events, nevents)))
      test_fail(__FILE__, __LINE__, "PAPI_remove_events", retval);
   if ((retval = PAPI_destroy_eventset(&eventset)))
      test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);
   eventset = PAPI_NULL;
   if ((retval = PAPI_create_eventset(&eventset)))
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   for (i = 0; i < nevents; i++) {

      if ((retval = PAPI_cleanup_eventset(eventset)))
         test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", retval);
      if ((retval = PAPI_add_event(eventset, events[i])))
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

      x = 1.0;

      if (!TESTS_QUIET)
         printf("\nReference measurement %d (of %d):\n", i + 1, nevents);

      t1 = PAPI_get_real_usec();
      if ((retval = PAPI_start(eventset)))
         test_fail(__FILE__, __LINE__, "PAPI_start", retval);
      y = dummy3(x, iters);
      if ((retval = PAPI_stop(eventset, &refvalues[i])))
         test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
      t2 = PAPI_get_real_usec();

      if (!TESTS_QUIET) {
         printf("\tOperations= %.1f Mflop", y * 1e-6);
         printf("\t(%g Mflop/s)\n\n", ((float) y / (t2 - t1)));
      }
      PAPI_get_event_info(events[i], &info);
      if (!TESTS_QUIET) {
         printf("PAPI results:\n%20s = ", info.short_descr);
         printf(LLDFMT, refvalues[i]);
         printf("\n");
      }
   }
   if (!TESTS_QUIET)
      printf("\n");


   if (!TESTS_QUIET) {
      printf("\n\nRelative accuracy:\n");
      for (j = 0; j < nevents; j++)
         printf("   Event %.2d", j);
      printf("\n");
   }

   for (j = 0; j < nevents; j++) {
      spread[j] = abs((int)(refvalues[j] - values[j]));
      if (values[j])
         spread[j] /= (double) values[j];
      if (!TESTS_QUIET)
         printf("%10.3g ", spread[j]);
      /* Make sure that NaN get counted as errors */
      if (spread[j] < MPX_TOLERANCE)
         i--;
      else if (refvalues[j] < MINCOUNTS)        /* Neglect inprecise results with low counts */
         i--;
   }
   if (!TESTS_QUIET) {
      printf("\n\n");
      for (j = 0; j < nevents; j++) {
         PAPI_get_event_info(events[j], &info);
         printf("Event %.2d: ref=", j);
         printf(LLDFMT10, refvalues[j]);
         printf(", diff/ref=%7.2g  -- %s\n", spread[j], info.short_descr);
         printf("\n");
      }
      printf("\n");
   }

   if (i)
      test_fail(__FILE__, __LINE__, "Values outside threshold", i);
   else
      test_pass(__FILE__, NULL, 0);

   return 0;
}

static double dummy3(double x, int iters)
{
   int i;
   double w, y, z, a, b, c, d, e, f, g, h;
   double one;
   one = 1.0;
   w = x;
   y = x;
   z = x;
   a = x;
   b = x;
   c = x;
   d = x;
   e = x;
   f = x;
   g = x;
   h = x;
   for (i = 1; i <= iters; i++) {
      w = w * 1.000000000001 + one;
      y = y * 1.000000000002 + one;
      z = z * 1.000000000003 + one;
      a = a * 1.000000000004 + one;
      b = b * 1.000000000005 + one;
      c = c * 0.999999999999 + one;
      d = d * 0.999999999998 + one;
      e = e * 0.999999999997 + one;
      f = f * 0.999999999996 + one;
      g = h * 0.999999999995 + one;
      h = h * 1.000000000006 + one;
   }
   return 2.0 * (a + b + c + d + e + f + w + x + y + z + g + h);
}
