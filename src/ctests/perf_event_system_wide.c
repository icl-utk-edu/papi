/*
 * This tests the measuring of events using a system-wide granularity
 */

#include "papi_test.h"

int main( int argc, char **argv ) {

   int retval;
   int EventSet = PAPI_NULL;
   int EventSet2 = PAPI_NULL;
   long long values[1],local_values[1];

   /* Set TESTS_QUIET variable */
   tests_quiet( argc, argv );

   /* Init the PAPI library */
   retval = PAPI_library_init( PAPI_VER_CURRENT );
   if ( retval != PAPI_VER_CURRENT ) {
      test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
   }

   /* Create a Local eventset */
   retval = PAPI_create_eventset(&EventSet2);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   /* Add PAPI_TOT_CYC */
   retval = PAPI_add_named_event(EventSet2, "PAPI_TOT_CYC");
   if (retval != PAPI_OK) {
      if ( !TESTS_QUIET ) {
         fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
      }
      test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
   }


   /* Create a System-Wide eventset */
   retval = PAPI_create_eventset(&EventSet);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   /* Set a component for the EventSet */
   retval = PAPI_assign_eventset_component(EventSet, 0);

   /* we need to set to a certain cpu for uncore to work */

   PAPI_cpu_option_t cpu_opt;

   cpu_opt.eventset=EventSet;
   cpu_opt.cpu_num=0;

   retval = PAPI_set_opt(PAPI_CPU_ATTACH,(PAPI_option_t*)&cpu_opt);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_CPU_ATTACH",retval);
   }

   /* Set the granularity to system-wide */

   PAPI_granularity_option_t gran_opt;

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet;
   gran_opt.granularity=PAPI_GRN_SYS;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      test_skip( __FILE__, __LINE__,
		      "this test; trying to set PAPI_GRN_SYS",
		      retval);
   }

   /* we need to set domain to be as inclusive as possible */

   PAPI_domain_option_t domain_opt;

   domain_opt.def_cidx=0;
   domain_opt.eventset=EventSet;
   domain_opt.domain=PAPI_DOM_ALL;

   retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
   if (retval != PAPI_OK) {
      test_skip( __FILE__, __LINE__,
		      "this test; trying to set PAPI_DOM_ALL; need to run as root",
		      retval);
   }

   /* Add PAPI_TOT_CYC */
   retval = PAPI_add_named_event(EventSet, "PAPI_TOT_CYC");
   if (retval != PAPI_OK) {
      if ( !TESTS_QUIET ) {
         fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
      }
      test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
   }


   /* Start PAPI */
   retval = PAPI_start( EventSet );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }
   retval = PAPI_start( EventSet2 );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   /* our work code */
   do_flops( NUM_FLOPS );

   /* Stop PAPI */
   retval = PAPI_stop( EventSet, values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   /* Stop PAPI */
   retval = PAPI_stop( EventSet2, local_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !TESTS_QUIET ) {
      printf("System Wide test:\n");
      printf("\tSystem Wide %s: %lld\n","PAPI_TOT_CYC",values[0]);
      printf("\tLocal Value %s: %lld\n","PAPI_TOT_CYC",local_values[0]);
   }

   test_pass( __FILE__, NULL, 0 );

   return 0;
}
