/* From Dave McNamara at PSRV. Thanks! */

/* If an event is countable but you've exhausted the counter resources
and you try to add an event, it seems subsequent PAPI_start and/or
PAPI_stop will causes a Seg. Violation.

   I got around this by calling PAPI to get the # of countable events,
then making sure that I didn't try to add more than these number of
events. I still have a problem if someone adds Level 2 cache misses
and then adds FLOPS 'cause I didn't count FLOPS as actually requiring
2 counters. */

#include <stdio.h>
#include "papiStdEventDefs.h"
#include "papi.h"

int main()
{
   double c,a = 0.999,b = 1.001;
   int n = 1000;
   int EventSet;
   int retval;
   int j = 0,i;
   long long int g1[3];

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
     exit(1);
   
   if (PAPI_query_event(PAPI_L2_TCM) == PAPI_OK)
     j++;
   retval = PAPI_add_event(&EventSet, PAPI_L2_TCM);
   if ( retval != PAPI_OK ) printf("Error adding L2 TCM (OK)\n");

   if (PAPI_query_event(PAPI_TOT_CYC) == PAPI_OK)
     j++;
   retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC);
   if ( retval != PAPI_OK ) printf("Error adding TOT_CYC (OK)\n");

   if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK)
     j++;
   retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
   if ( retval != PAPI_OK ) printf("Error adding FP_INS (OK)\n");

   if (j)
     {
       PAPI_start(EventSet);
       for ( i = 0; i < n; i++ )
	 {
	   c = a * b;
	 }
       PAPI_stop(EventSet, g1);
     }
   exit(0);
}
