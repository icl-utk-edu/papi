/* 
* File:    overflow_index.c
* CVS:     $Id$
* Author:  min@cs.utk.edu
*          Min Zhou
*/

/* This file performs the following test: overflow dispatch on 2 counters. */

#include "papi_test.h"

#ifdef _CRAYT3E
#define OVER_FMT	"handler(%d) Overflow at %x! vector=0x%llx\n"
#define OUT_FMT		"%-12s : %16lld%16lld\n"
#elif defined(_WIN32)
#define OVER_FMT	"handler(%d) Overflow at %p! vector=0x%llx\n"
#define OUT_FMT		"%-12s : %16I64d%16I64d\n"
#define INDEX_FMT   "Overflows vector 0x%llx: \n"
#else
#define OVER_FMT	"handler(%d) Overflow at %p! vector=0x%llx\n"
#define OUT_FMT		"%-12s : %16lld%16lld\n"
#define INDEX_FMT   "Overflows vector 0x%llx: \n"
#endif

typedef struct {
  long_long mask;
  int count;
} ocount_t;

/* there are three possible vectors, one counter overflows, the other 
   counter overflows, both overflow */
ocount_t overflow_counts[3] = {{0,0}, {0,0}, {0,0}};
int total_unknown = 0;

void handler(int EventSet, void *address, long_long overflow_vector, void *context)
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
       if (overflow_counts[i].mask == (long_long)0)
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
   int EventSet;
   long_long(values[3])[2];
   int retval;
   int PAPI_event,k ,i;
   char event_name[PAPI_MAX_STR_LEN];
   int index_array[2], number;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET) {
      retval = PAPI_set_debug(PAPI_VERB_ECONT);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   }

#if (defined(sun) && defined(sparc)) || defined(POWER3)
   /* query and set up the right instruction to monitor */
   if (PAPI_query_event(PAPI_TOT_INS) == PAPI_OK)
      PAPI_event = PAPI_TOT_INS;
   else
      test_fail(__FILE__, __LINE__, "PAPI_TOT_INS not available on this platform!",
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

   if ((retval = PAPI_event_code_to_name(PAPI_event, event_name)) != PAPI_OK)
     test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

   printf("Test case: Overflow dispatch of 2nd event in set with 2 events.\n");
   printf("---------------------------------------------------------------\n");
   printf("Threshold for overflow is: %d\n", THRESHOLD);
   printf("Using %d iterations of c += a*b\n", NUM_FLOPS);
   printf("-----------------------------------------------\n");

   printf("Test type    : %16d%16d\n", 1, 2);
   printf(OUT_FMT, "PAPI_TOT_CYC", (values[0])[0], (values[1])[0]);
   printf(OUT_FMT, event_name, (values[0])[1], (values[1])[1]);

   if (overflow_counts[0].count == 0 && overflow_counts[1].count==0)
      test_fail(__FILE__, __LINE__, "one counter had no overflows", 1);

   for(k=0; k<3; k++ )
   {
      if (overflow_counts[k].mask) 
      { 
         number=2;
         retval = PAPI_get_overflow_event_index(EventSet,
                          overflow_counts[k].mask, index_array, &number);
         if (retval != PAPI_OK)
            test_fail(__FILE__, __LINE__, 
                   "PAPI_get_overflow_event_index", retval);
         printf(INDEX_FMT, (long_long)overflow_counts[k].mask);
         printf(" counts: %d ", overflow_counts[k].count);
         for(i=0; i<number; i++)
            printf(" Event Index %d ", index_array[i]);
         printf("\n");
      }
   }
   printf("Case 2 %s Overflows: %d\n", "Unknown", total_unknown);
   printf("-----------------------------------------------\n");

   if (total_unknown > 0)
      test_fail(__FILE__, __LINE__, "Unknown counter had overflows", 1);

   retval = PAPI_cleanup_eventset(EventSet);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   test_pass(__FILE__, NULL, 0);
   exit(1);
}
