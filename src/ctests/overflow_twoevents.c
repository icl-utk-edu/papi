/* 
* File:    overflow.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: overflow dispatch

     The Eventset contains:
     + PAPI_TOT_CYC
     + PAPI_FP_INS (overflow monitor)

   - Start eventset 1
   - Do flops
   - Stop and measure eventset 1
   - Set up overflow on eventset 1
   - Start eventset 1
   - Do flops
   - Stop eventset 1
*/

#include "papi_test.h"

#ifdef _CRAYT3E
#define OVER_FMT	"handler(%d, %x, %d, %lld, %d, %x) Overflow at %x!\n"
#define OUT_FMT		"%-12s : %16lld%16lld%16lld\n"
#elif defined(_WIN32)
#define OVER_FMT	"handler(%d, %x, %d, %I64d, %d, %p) Overflow at %p!\n"
#define OUT_FMT		"%-12s : %16I64d%16I64d%16I64d\n"
#else
#define OVER_FMT	"handler(%d) Overflow at %p! vector=0x%llx\n"
#define OUT_FMT		"%-12s : %16lld%16lld%16lld\n"
#endif

int total = 0;                  /* total overflows */
extern int TESTS_QUIET;         /* Declared in test_utils.c */

void handler(int EventSet, void *address, long_long overflow_vector)
{

   if (!TESTS_QUIET) {
      fprintf(stderr, OVER_FMT, EventSet, address, overflow_vector);
   }

   total++;
}

int main(int argc, char **argv)
{
   int EventSet;
   long_long(values[3])[2];
/*
  long_long min, max;
*/
   int num_flops, retval;
   int PAPI_event;
   char event_name[PAPI_MAX_STR_LEN];

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET) {
      retval = PAPI_set_debug(PAPI_VERB_ECONT);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   }

/*
#if (defined(sun) && defined(sparc)) || (defined(mips) && defined(sgi) && defined(unix))
*/
#if (defined(sun) && defined(sparc)) || defined(POWER3)
   /* query and set up the right instruction to monitor */
   if (PAPI_query_event(PAPI_TOT_INS) == PAPI_OK)
      PAPI_event = PAPI_TOT_INS;
   else
      test_fail(__FILE__, __LINE__, "PAPI_TOT_INS not available on this Sun platform!",
                0);
#else
   /* query and set up the right instruction to monitor */
   if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK)
      PAPI_event = PAPI_FP_INS;
   else
      PAPI_event = PAPI_TOT_INS;
#endif

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

   retval = PAPI_add_event(EventSet, PAPI_event);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, values[0]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   retval = PAPI_overflow(EventSet, PAPI_event, THRESHOLD, 0, handler);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);
   retval = PAPI_overflow(EventSet, PAPI_TOT_CYC, THRESHOLD, 0, handler);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, values[1]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   retval = PAPI_overflow(EventSet, PAPI_event, 0, 0, handler);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);
   retval = PAPI_overflow(EventSet, PAPI_TOT_CYC, 0, 0, handler);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, values[2]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);
   num_flops = NUM_FLOPS;
#if defined(linux) || defined(__ia64__) || defined(_WIN32) || defined(_CRAYT3E) || defined(_POWER4)
   num_flops *= 2;
#endif

   if (!TESTS_QUIET) {
      if ((retval = PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

      printf("Test case: Overflow dispatch of 2nd event in set with 2 events.\n");
      printf("---------------------------------------------------------------\n");
      printf("Threshold for overflow is: %d\n", THRESHOLD);
      printf("Using %d iterations of c += a*b\n", NUM_FLOPS);
      printf("-----------------------------------------------\n");

      printf("Test type    : %16d%16d\n", 1, 2);
      printf(OUT_FMT, "PAPI_TOT_CYC", (values[0])[0], (values[1])[0], (values[2])[0]);
      printf(OUT_FMT, event_name, (values[0])[1], (values[1])[1], (values[2])[1]);
      printf("Overflows    : %16s%16d\n", "", total);
      printf("-----------------------------------------------\n");

      printf("Verification:\n");
      if (PAPI_event == PAPI_FP_INS)
         printf("Row 2 approximately equals %d %d\n", num_flops, num_flops);
      printf("Column 1 approximately equals column 2\n");
      printf("Row 3 approximately equals %u +- %u %%\n",
             (unsigned) ((values[0])[1] / (long_long) THRESHOLD),
             (unsigned) (OVR_TOLERANCE * 100.0));
   }

/*
  min = (long_long)((values[0])[1]*(1.0-TOLERANCE));
  max = (long_long)((values[0])[1]*(1.0+TOLERANCE));
  if ( (values[0])[1] > max || (values[0])[1] < min )
  	test_fail(__FILE__, __LINE__, event_name, 1);

  min = (long_long)(((values[0])[1]*(1.0-OVR_TOLERANCE))/(long_long)THRESHOLD);
  max = (long_long)(((values[0])[1]*(1.0+OVR_TOLERANCE))/(long_long)THRESHOLD);
  if ( total > max || total < min )
  	test_fail(__FILE__, __LINE__, "Overflows", 1);
*/

   test_pass(__FILE__, NULL, 0);
   exit(1);
}
