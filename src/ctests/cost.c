#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "papi.h"
#include "test_utils.h"

int err_exit(int code, char *str)
{
  char out[PAPI_MAX_STR_LEN];
  PAPI_perror(code, out, PAPI_MAX_STR_LEN);
  printf("Error in %s: %s\n",str,out);
  exit(1);
}

int main()
{
   int i, retval, EventSet = PAPI_NULL, CostEventSet = PAPI_NULL;
   long long int totcyc, values[2], readvalues[2];

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
     err_exit(retval, "PAPI_library_init(PAPI_VER_CURRENT)");

   if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
     err_exit(retval, "PAPI_set_debug(PAPI_VERB_ECONT)");

   if ((retval = PAPI_query_event(PAPI_TOT_CYC)) != PAPI_OK)
     err_exit(retval, "PAPI_query_event(PAPI_TOT_CYC)");

   if ((retval = PAPI_query_event(PAPI_TOT_INS)) != PAPI_OK)
     err_exit(retval, "PAPI_query_event(PAPI_TOT_INS)");

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_create_eventset(&EventSet)");

   if ((retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC)) != PAPI_OK)
     err_exit(retval, "PAPI_add_event(&EventSet, PAPI_TOT_CYC)");

   if ((retval = PAPI_add_event(&EventSet, PAPI_TOT_INS)) != PAPI_OK)
     err_exit(retval, "PAPI_add_event(&EventSet, PAPI_TOT_INS)");

   if ((retval = PAPI_create_eventset(&CostEventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_create_eventset(&CostEventSet)");

   if ((retval = PAPI_add_event(&CostEventSet, PAPI_TOT_CYC)) != PAPI_OK)
     err_exit(retval, "PAPI_add_event(&CostEventSet, PAPI_TOT_CYC)");

   if ((retval = PAPI_add_event(&CostEventSet, PAPI_TOT_INS)) != PAPI_OK)
     err_exit(retval, "PAPI_add_event(&CostEventSet, PAPI_TOT_INS)");

   /* Make sure no errors */

   if ((retval = PAPI_start(CostEventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_start(CostEventSet)");
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_start(EventSet)");
   if ((retval = PAPI_stop(EventSet, NULL)) != PAPI_OK)
     err_exit(retval, "PAPI_stop(EventSet, NULL)");
   if ((retval = PAPI_stop(CostEventSet, NULL)) != PAPI_OK)
     err_exit(retval, "PAPI_stop(CostEventSet, NULL)");

   /* Start the start/stop eval */

   if ((retval = PAPI_reset(CostEventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_reset(CostEventSet)");

   totcyc = PAPI_get_real_cyc();
   if ((retval = PAPI_start(CostEventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_start(CostEventSet)");

   for (i=0;i<1000000;i++)
     {
       PAPI_start(EventSet);
       PAPI_stop(EventSet, values);
     }

   if ((retval = PAPI_stop(CostEventSet, values)) != PAPI_OK)
     err_exit(retval, "PAPI_stop(CostEventSet, values)");
   totcyc = PAPI_get_real_cyc() - totcyc;

   printf("User level cost for PAPI_start/stop(2 counters) over 1000000 iterations\n");
   printf("%lld total cyc, %lld total ins, %f cyc/call pair, %f ins/call pair\n",
	  values[0],values[1],((float)values[0])/1000000.0,((float)values[1])/1000000.0);
   printf("\nTotal cost for PAPI_start/stop(2 counters) over 1000000 iterations\n");
   printf("%lld total cyc, %f cyc/call pair\n",
	  totcyc,((float)totcyc)/1000001.0);

   /* Start the read eval */

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_start(EventSet)");
   if( (retval = PAPI_reset(CostEventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_reset(CostEventSet)");
   totcyc = PAPI_get_real_cyc();
   if ((retval = PAPI_start(CostEventSet)) != PAPI_OK)
     err_exit(retval, "PAPI_start(CostEventSet)");

   for (i=0;i<1000000;i++)
     {
       PAPI_read(EventSet, values);
     }

   if ((retval = PAPI_stop(CostEventSet, readvalues)) != PAPI_OK)
     err_exit(retval, "PAPI_stop(CostEventSet, readvalues)");
   totcyc = PAPI_get_real_cyc() - totcyc;

   printf("\nUser level cost for PAPI_read(2 counters) over 1000000 iterations\n");
   printf("%lld total cyc, %lld total ins, %f cyc/call, %f ins/call\n",
	  values[0],values[1],((float)values[0])/1000000.0,((float)values[1])/1000000.0);
   printf("\nTotal cost for PAPI_read(2 counters) over 1000000 iterations\n");
   printf("%lld total cyc, %f cyc/call\n",
	  totcyc,((float)totcyc)/1000001.0);

   exit(0);
}
