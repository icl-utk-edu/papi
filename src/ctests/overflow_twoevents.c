/* 
* File:    overflow.c
* CVS:     $Id$
* Author:  min@cs.utk.edu
*          Min Zhou
* Mods:    Philip Mucci
*          mucci@cs.utk.edu
*/

/* This file performs the following test: overflow dispatch on 2 counters. */

#include "papi_test.h"

#ifdef _CRAYT3E
#define OVER_FMT	"handler(%d) Overflow at %x! overflow_vector=0x%x!\n"
#define OUT_FMT		"%-12s : %16lld%16lld\n"
#define VEC_FMT	"Overflows at vector 0x%x: %d\n"
#elif defined(_WIN32)
#define OVER_FMT	"handler(%d) Overflow at %p! overflow_vector=0x%x!\n"
#define OUT_FMT		"%-12s : %16I64d%16I64d\n"
#define VEC_FMT	"Overflows at vector %I64d: %d\n"
#else
#define OVER_FMT	"handler(%d) Overflow at %p! vector=0x%llx\n"
#define OUT_FMT		"%-12s : %16lld%16lld\n"
#define VEC_FMT	"Overflows at vector 0x%llx: %d\n"
#endif

typedef struct {
  long long mask;
  int count;
} ocount_t;

/* there are three possible vectors, one counter overflows, the other 
   counter overflows, both overflow */
ocount_t overflow_counts[3] = {{0,0}, {0,0}, {0,0}};
int total_unknown = 0;

static const PAPI_hw_info_t *hw_info = NULL;

void handler(int EventSet, void *address, long long overflow_vector, void *context)
{
  int i;

   if (!TESTS_QUIET) {
      fprintf(stderr, OVER_FMT, EventSet, address, overflow_vector);
   }

   /* Look for the overflow_vector entry */

   for (i=0;i<3;i++)
   {
      if (overflow_counts[i].mask == overflow_vector)
      {
	      overflow_counts[i].count++;
	      return;
      }
   }

   /* Didn't find it so add it. */

   for (i=0;i<3;i++)
   {
      if (overflow_counts[i].mask == (long long)0)
      {
         overflow_counts[i].mask = overflow_vector;
         overflow_counts[i].count = 1;
         return;
      }
   }

   /* Unknown entry!?! */

   total_unknown++;
}

int main(int argc, char **argv)
{
   int EventSet=PAPI_NULL;
   long long(values[3])[2];
   int k, retval;
   long long PAPI_event;
   char event_name[PAPI_MAX_STR_LEN];
   int num_events1, mask1;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   hw_info = PAPI_get_hardware_info(); 
   if (hw_info == NULL)
     test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);
      
    /* add PAPI_TOT_CYC and one of the events in PAPI_FP_INS, PAPI_FP_OPS or
      PAPI_TOT_INS, depends on the availability of the event on the
      platform */ 
   EventSet = add_two_nonderived_events(&num_events1, &PAPI_event, hw_info, &mask1);

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

   retval = PAPI_cleanup_eventset(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   if ((retval = PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
     test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

   printf("Test case: Overflow dispatch of 2nd event in set with 2 events.\n");
   printf("---------------------------------------------------------------\n");
   printf("Threshold for overflow is: %d\n", THRESHOLD);
   printf("Using %d iterations of c += a*b\n", NUM_FLOPS);
   printf("-----------------------------------------------\n");

   printf("Test type    : %16d%16d\n", 1, 2);
   printf(OUT_FMT, event_name, (values[0])[0], (values[1])[0]);
   printf(OUT_FMT, "PAPI_TOT_CYC", (values[0])[1], (values[1])[1]);

   if (overflow_counts[0].count == 0 && overflow_counts[1].count==0)
      test_fail(__FILE__, __LINE__, "one counter had no overflows", 1);

   for(k=0; k<3; k++ )
   {
     if (overflow_counts[k].mask) 
         printf(VEC_FMT, (long long)overflow_counts[k].mask, overflow_counts[k].count);
   }
   printf("Case 2 %s Overflows: %d\n", "Unknown", total_unknown);
   printf("-----------------------------------------------\n");

   if (total_unknown > 0)
      test_fail(__FILE__, __LINE__, "Unknown counter had overflows", 1);

   test_pass(__FILE__, NULL, 0);
   exit(1);
}
