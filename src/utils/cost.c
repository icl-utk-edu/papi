#include "papi_test.h"

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
   int i, retval, EventSet = PAPI_NULL, CostEventSet = PAPI_NULL;
   long_long totcyc, values[2], readvalues[2];


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
   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_create_eventset", retval );

   if ((retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event", retval );

   if ((retval = PAPI_add_event(&EventSet, PAPI_TOT_INS)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event", retval );

   if ((retval = PAPI_create_eventset(&CostEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_create_eventset", retval );

   if ((retval = PAPI_add_event(&CostEventSet, PAPI_TOT_CYC)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event", retval );

   if ((retval = PAPI_add_event(&CostEventSet, PAPI_TOT_INS)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event", retval );

   /* Make sure no errors */

   if ((retval = PAPI_start(CostEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start", retval );
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval );
   if ((retval = PAPI_stop(EventSet, NULL)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval );
   if ((retval = PAPI_stop(CostEventSet, NULL)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval );

   /* Start the start/stop eval */

   if ((retval = PAPI_reset(CostEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_reset",retval );

  if ( !TESTS_QUIET ) 
       printf("Performing start/stop test...\n");
   totcyc = PAPI_get_real_cyc();
   if ((retval = PAPI_start(CostEventSet)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_start",retval );

     for (i=0;i<50000;i++)
     {
       PAPI_start(EventSet);
       PAPI_stop(EventSet, values);
     }

   if ((retval = PAPI_stop(CostEventSet, values)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_stop",retval );
   totcyc = PAPI_get_real_cyc() - totcyc;

   if ( !TESTS_QUIET ) {
   printf("\n");

   printf("User level cost for PAPI_start/stop(2 counters) over 50000 iterations\n");
   printf(format_string1,values[0],values[1],((float)values[0])/50000.0,"cyc/call pair",
	   ((float)values[1])/50000.0,"ins/call pair");
   printf("\nTotal cost for PAPI_start/stop(2 counters) over 50000 iterations\n");
   printf(format_string2,totcyc,((float)totcyc)/50001.0,"cyc/call pair");

   /* Start the read eval */
   printf("\n\nPerforming read test...\n");
   }
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_start", retval );
   if( (retval = PAPI_reset(CostEventSet)) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_reset", retval );
   totcyc = PAPI_get_real_cyc();
   if ((retval = PAPI_start(CostEventSet)) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_start", retval );

	for (i=0;i<50000;i++)
     {
       PAPI_read(EventSet, values);
     }

   if ((retval = PAPI_stop(CostEventSet, readvalues)) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_stop", retval );
   totcyc = PAPI_get_real_cyc() - totcyc;

   if ( !TESTS_QUIET ) {
   printf("\nUser level cost for PAPI_read(2 counters) over 50000 iterations\n");
   printf(format_string1,values[0],values[1],((float)values[0])/50000.0,"cyc/call",
	   ((float)values[1])/50000.0,"ins/call");
   printf("\nTotal cost for PAPI_read(2 counters) over 50000 iterations\n");
   printf(format_string2,totcyc,((float)totcyc)/50001.0,"cyc/call");
   }
   test_pass(__FILE__,NULL,0);
   exit(1);
}
