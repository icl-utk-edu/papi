#include "papi_test.h"

#ifdef _WIN32
	char format_string1[] = {"%I64d total cyc,\n%I64d total ins,\n%f %s,\n%f %s\n"};
	char format_string2[] = {"%I64d total cyc,\n%f %s\n"};
#else
	char format_string1[] = {"%lld total cyc,\n%lld total ins,\n%f %s,\n%f %s\n"};
	char format_string2[] = {"%lld total cyc,\n%f %s\n"};
#endif

void err_exit(int code, char *str)
{
  char out[PAPI_MAX_STR_LEN];
  PAPI_perror(code, out, PAPI_MAX_STR_LEN);
  printf("Error in %s: %s\n",str,out);
  exit(1);
}

int main()
{
   int i, retval, EventSet = PAPI_NULL, CostEventSet = PAPI_NULL;
   long_long totcyc, values[2], readvalues[2];

	printf("Cost of execution for PAPI start/stop and PAPI read.\n");
	printf("This test takes a while. Please be patient...\n");

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

   printf("Performing start/stop test...\n");
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
   printf("\n");

   printf("User level cost for PAPI_start/stop(2 counters) over 1000000 iterations\n");
   printf(format_string1,values[0],values[1],((float)values[0])/1000000.0,"cyc/call pair",
	   ((float)values[1])/1000000.0,"ins/call pair");
   printf("\nTotal cost for PAPI_start/stop(2 counters) over 1000000 iterations\n");
   printf(format_string2,totcyc,((float)totcyc)/1000001.0,"cyc/call pair");

   /* Start the read eval */
   printf("\n\nPerforming read test...\n");
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
   printf(format_string1,values[0],values[1],((float)values[0])/1000000.0,"cyc/call",
	   ((float)values[1])/1000000.0,"ins/call");
   printf("\nTotal cost for PAPI_read(2 counters) over 1000000 iterations\n");
   printf(format_string2,totcyc,((float)totcyc)/1000001.0,"cyc/call");

   exit(0);
}
