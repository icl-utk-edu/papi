/* 
* File:    overflow_single_event.c
* CVS:     $Id$
* Author:  Kevin London
*          london@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: overflow dispatch of an eventset
   with just a single event. Using both Hardware and software overflows

     The Eventset contains:
     + PAPI_FP_INS (overflow monitor)

   - Start eventset 1
   - Do flops
   - Stop and measure eventset 1
   - Set up overflow on eventset 1
   - Start eventset 1
   - Do flops
   - Stop eventset 1
   - Set up forced software overflow on eventset 1
   - Start eventset 1
   - Do flops
   - Stop eventset 1
*/

#include "papi_test.h"

#ifdef _CRAYT3E
#define OVER_FMT	"handler(%d ) Overflow at %x! overflow_vector=0x%x!\n"
#define OUT_FMT		"%-12s : %16lld%16d%16lld\n"
#elif defined(_WIN32)
#define OVER_FMT	"handler(%d ) Overflow at %p! overflow_vector=0x%x!\n"
#define OUT_FMT		"%-12s : %16I64d%16d%16I64d\n"
#else
#define OVER_FMT	"handler(%d ) Overflow at %p overflow_vector=0x%llx!\n"
#define OUT_FMT		"%-12s : %16lld%16d%16lld\n"
#endif

static int total[2] = {0,0};                  /* total overflows */
static int use_total=0;

void handler(int EventSet, void *address, long_long overflow_vector, void *context)
{
   if (!TESTS_QUIET) {
      fprintf(stderr, OVER_FMT, EventSet, address, overflow_vector);
   }

   if ( use_total )
      total[1]++;
   else
      total[0]++;
}

int main(int argc, char **argv)
{
   int EventSet=PAPI_NULL;
   long_long values[3] = { 0, 0, 0 };
   long_long min, max;
   int save_debug_setting;
   int num_flops, retval;
   int PAPI_event=0, mythreshold;
   char event_name[PAPI_MAX_STR_LEN];
   PAPI_option_t  opt;
   PAPI_event_info_t info;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET) {
      retval = PAPI_set_debug(PAPI_VERB_ECONT);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   }

   retval = PAPI_get_opt(PAPI_SUBSTRATE_SUPPORT, &opt);
   if ( !opt.sub_info.supports_hw_overflow )
      test_skip(__FILE__, __LINE__, "Platform does not support Hardware overflow", 0);

   /* query and set up the right instruction to monitor */
  if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK) {
     if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK) {
        PAPI_get_event_info(PAPI_FP_INS, &info);
        if ( info.count == 1 )
           PAPI_event = PAPI_FP_INS;
     }
  } 
  if ( PAPI_event == 0 ) {
     if (PAPI_query_event(PAPI_FP_OPS) == PAPI_OK) {
        PAPI_get_event_info(PAPI_FP_OPS, &info);
        if ( info.count == 1 )
           PAPI_event = PAPI_FP_OPS;
     }
  }
  if ( PAPI_event == 0 ) 
      PAPI_event = PAPI_TOT_INS;

   if ( PAPI_event == 0 )
      test_fail(__FILE__, __LINE__, "No suitable event for this test found!", 0);
   

   if (PAPI_event == PAPI_FP_INS )
      mythreshold = THRESHOLD;
   else if ( PAPI_event == PAPI_TOT_INS )
      mythreshold = THRESHOLD * 10;
   else 
      mythreshold = THRESHOLD * 2;

   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   retval = PAPI_add_event(EventSet, PAPI_event);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, &values[0]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   retval = PAPI_overflow(EventSet, PAPI_event, mythreshold, 0, handler);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, &values[1]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   use_total = 1;

   retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

   retval = PAPI_overflow(EventSet, PAPI_event, 0, PAPI_OVERFLOW_FORCE_SW, handler);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);

   retval = PAPI_overflow(EventSet, PAPI_event, mythreshold, PAPI_OVERFLOW_FORCE_SW, handler);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow", retval);

   /* expecting error code, so let's suppress printout of error message */
   PAPI_get_opt(PAPI_DEBUG, &opt);
   save_debug_setting = opt.debug.level;
   opt.debug.level = PAPI_QUIET;
   PAPI_set_opt(PAPI_DEBUG, &opt);
   retval = PAPI_overflow(EventSet, PAPI_TOT_CYC, mythreshold, 0, handler);
   if (retval == PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow: allowed hardware and software overflow", -1);
   opt.debug.level = save_debug_setting;
   PAPI_set_opt(PAPI_DEBUG, &opt);

   retval = PAPI_overflow(EventSet, PAPI_TOT_CYC, mythreshold, PAPI_OVERFLOW_FORCE_SW, handler);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_overflow: didn't allow 2 software overflow", -1);

   retval = PAPI_remove_event(EventSet, PAPI_TOT_CYC);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_remove_event", retval);

   retval = PAPI_start(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   do_flops(NUM_FLOPS);

   retval = PAPI_stop(EventSet, &values[2]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   num_flops = NUM_FLOPS;
#if defined(linux) || defined(__ia64__) || defined(_WIN32) || defined(_CRAYT3E) || defined(_POWER4)
   num_flops *= 2;
#endif

   if (!TESTS_QUIET) {
      if ((retval = PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

      printf("Test case: Overflow dispatch of 1st event in set with 1 event.\n");
      printf("--------------------------------------------------------------\n");
      printf("Threshold for overflow is: %d\n", mythreshold);
      printf("Using %d iterations of c += a*b\n", NUM_FLOPS);
      printf("-----------------------------------------------\n");

      printf("Test type    : %16d%16d%16d\n", 1, 2, 3);
      printf(OUT_FMT, event_name, values[0], 0, values[2]);
      printf("Overflows    : %16s%16d%16d\n", "", total[0], total[1]);
      printf("-----------------------------------------------\n");

      printf("Verification:\n");
     
      printf("Overflow in Column 2 approximately equals overflows in column 3\n");
   }

   min = (long_long) ((values[0] * (1.0 - OVR_TOLERANCE)) / (long_long) mythreshold);
   max = (long_long) ((values[0] * (1.0 + OVR_TOLERANCE)) / (long_long) mythreshold);
   if (total[0] > max || total[0] < min)
      test_fail(__FILE__, __LINE__, "Hardware Overflows", 1);

   if (total[1] > max || total[1] < min)
      test_fail(__FILE__, __LINE__, "Software Overflows", 1);

   test_pass(__FILE__, NULL, 0);
   exit(1);
}
