#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include "papi.h"
#include "test_utils.h"

int main()
{
   int i, EventSet = PAPI_NULL, CostEventSet = PAPI_NULL;
   long long int totcyc, values[2], readvalues[2];

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
     exit(1);

   if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
     exit(1);

   if (PAPI_query_event(PAPI_TOT_CYC) != PAPI_OK)
     exit(1);

   if (PAPI_query_event(PAPI_TOT_INS) != PAPI_OK)
     exit(1);

   if (PAPI_create_eventset(&EventSet) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&EventSet, PAPI_TOT_CYC) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&EventSet, PAPI_TOT_INS) != PAPI_OK)
     exit(1);

   if (PAPI_create_eventset(&CostEventSet) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&CostEventSet, PAPI_TOT_CYC) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&CostEventSet, PAPI_TOT_INS) != PAPI_OK)
     exit(1);

   /* Make sure no errors */

   if (PAPI_start(CostEventSet) != PAPI_OK)
     exit(1);
   if (PAPI_start(EventSet) != PAPI_OK)
     exit(1);
   if (PAPI_stop(EventSet, NULL) != PAPI_OK)
     exit(1);
   if (PAPI_stop(CostEventSet, NULL) != PAPI_OK)
     exit(1);

   /* Start the start/stop eval */

   if (PAPI_reset(CostEventSet) != PAPI_OK)
     exit(1);

   totcyc = PAPI_get_real_cyc();
   if (PAPI_start(CostEventSet) != PAPI_OK)
     exit(1);

   for (i=0;i<1000000;i++)
     {
       PAPI_start(EventSet);
       PAPI_stop(EventSet, values);
     }

   if (PAPI_stop(CostEventSet, values) != PAPI_OK)
     exit(1);
   totcyc = PAPI_get_real_cyc() - totcyc;

   printf("User level cost for PAPI_start/stop(2 counters) over 1000000 iterations\n");
   printf("%lld total cyc, %lld total ins, %f cyc/call pair, %f ins/call pair\n",
	  values[0],values[1],((float)values[0])/1000000.0,((float)values[1])/1000000.0);
   printf("\nTotal cost for PAPI_start/stop(2 counters) over 1000000 iterations\n");
   printf("%lld total cyc, %f cyc/call pair\n",
	  totcyc,((float)totcyc)/1000001.0);

   /* Start the read eval */

   if (PAPI_start(EventSet) != PAPI_OK)
     exit(1);
   if (PAPI_reset(CostEventSet) != PAPI_OK)
     exit(1);
   totcyc = PAPI_get_real_cyc();
   if (PAPI_start(CostEventSet) != PAPI_OK)
     exit(1);

   for (i=0;i<1000000;i++)
     {
       PAPI_read(EventSet, values);
     }

   if (PAPI_stop(CostEventSet, readvalues) != PAPI_OK)
     exit(1);
   totcyc = PAPI_get_real_cyc() - totcyc;

   printf("\nUser level cost for PAPI_read(2 counters) over 1000000 iterations\n");
   printf("%lld total cyc, %lld total ins, %f cyc/call, %f ins/call\n",
	  values[0],values[1],((float)values[0])/1000000.0,((float)values[1])/1000000.0);
   printf("\nTotal cost for PAPI_read(2 counters) over 1000000 iterations\n");
   printf("%lld total cyc, %f cyc/call\n",
	  totcyc,((float)totcyc)/1000001.0);

   exit(0);
}
