#include "papi_test.h"
#define COST 50000

#ifdef _WIN32
	char format_string1[] = {"%I64d total cyc,\n%I64d total ins,\n%f %s,\n%f %s\n"};
	char format_string2[] = {"%I64d total cyc,\n%f %s\n"};
#else
	char format_string1[] = {"%lld total cyc,\n%lld total ins,\n%f %s,\n%f %s\n"};
	char format_string2[] = {"%lld total cyc,\n%f %s\n"};
#endif

extern int TESTS_QUIET; /* Declared in test_utils.c */

int main(int argc, char **argv)
{
   int i, retval, CostEventSet = PAPI_NULL;
   long_long totcyc, values[2];


  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ( !TESTS_QUIET ) {
   printf("Cost of execution for PAPI start/stop and PAPI read.\n");
   printf("This test takes a while. Please be patient...\n");
  }

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init", retval );
   if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug", retval );

   if ((retval = PAPI_query_event(PAPI_TOT_CYC)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_query_event", retval );
   if ((retval = PAPI_query_event(PAPI_TOT_INS)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_query_event", retval );

   if ((retval = PAPI_create_eventset(&CostEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_create_eventset", retval );
   if ((retval = PAPI_add_event(&CostEventSet, PAPI_TOT_CYC)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event", retval );
   if ((retval = PAPI_add_event(&CostEventSet, PAPI_TOT_INS)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event", retval );

   /* Make sure no errors */

   if ((retval = PAPI_start(CostEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start", retval );
   if ((retval = PAPI_stop(CostEventSet, NULL)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval );

   /* Start the start/stop eval */

   if ((retval = PAPI_reset(CostEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_reset",retval );

  if ( !TESTS_QUIET ) 
       printf("Performing start/stop test...\n");

   totcyc = PAPI_get_virt_cyc();
   for (i=0;i<COST;i++)
     {
       PAPI_start(CostEventSet);
       PAPI_stop(CostEventSet, values);
     }
   totcyc = PAPI_get_virt_cyc() - totcyc;

   if ( !TESTS_QUIET ) {
   printf("\n");

   printf("Accumulated Noise for PAPI_start/stop(2 counters) over %d iterations\n",COST);
   printf(format_string1,values[0],values[1],((float)values[0])/((float)COST),"cyc/call pair",
	   ((float)values[1])/((float)COST),"ins/call pair");
   printf("\nVirtual cost for PAPI_start/stop(2 counters) over %d iterations\n",COST);
   printf(format_string2,totcyc,((float)totcyc)/((float)COST),"cyc/call pair");

   /* Start the read eval */
   printf("\n\nPerforming read test...\n");
   }

   if( (retval = PAPI_reset(CostEventSet)) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_reset", retval );

   if ((retval = PAPI_start(CostEventSet)) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_start", retval );

   totcyc = PAPI_get_virt_cyc();
   for (i=0;i<50000;i++)
     {
       PAPI_read(CostEventSet, values);
     }
   totcyc = PAPI_get_virt_cyc() - totcyc;

   if ((retval = PAPI_stop(CostEventSet, NULL)) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_stop", retval );

   if ( !TESTS_QUIET ) {
   printf("\nUser level cost for PAPI_read(2 counters) over %d iterations\n",COST);
   printf(format_string1,values[0],values[1],((float)values[0])/((float)COST),"cyc/call",
	   ((float)values[1])/((float)COST),"ins/call");
   printf("\nVirtual cost for PAPI_read(2 counters) over %d iterations\n",COST);
   printf(format_string2,totcyc,((float)totcyc)/((float)COST),"cyc/call");
   }
   test_pass(__FILE__,NULL,0);
   exit(1);
}
