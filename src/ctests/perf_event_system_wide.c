/*
 * This tests the measuring of events using a system-wide granularity
 */

#include "papi_test.h"

int main( int argc, char **argv ) {

   int retval;
   int EventSet1 = PAPI_NULL;
   int EventSet2 = PAPI_NULL;
   int EventSet3 = PAPI_NULL;
   int EventSet4 = PAPI_NULL;
   int EventSet5 = PAPI_NULL;
   int EventSet6 = PAPI_NULL;
   int EventSet7 = PAPI_NULL;
   int EventSet8 = PAPI_NULL;

   PAPI_domain_option_t domain_opt;
   PAPI_granularity_option_t gran_opt;

   long long user_values[1],userkernel_values[1],all_values[1];
   long long grn_thr_values[1],grn_proc_values[1];
   long long grn_sys_values[1],grn_sys_cpu_values[1];
   long long total_values[1];

   user_values[0]=0;
   userkernel_values[0]=0;
   all_values[0]=0;
   grn_thr_values[0]=0;
   grn_proc_values[0]=0;
   grn_sys_values[0]=0;
   grn_sys_cpu_values[0]=0;
   total_values[0]=0;

   /* Set TESTS_QUIET variable */
   tests_quiet( argc, argv );

   /* Init the PAPI library */
   retval = PAPI_library_init( PAPI_VER_CURRENT );
   if ( retval != PAPI_VER_CURRENT ) {
      test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
   }

   /***************************/
   /***************************/
   /* Default, user events    */
   /***************************/
   /***************************/

   retval = PAPI_create_eventset(&EventSet1);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_add_named_event(EventSet1, "PAPI_TOT_CYC");
   if (retval != PAPI_OK) {
      if ( !TESTS_QUIET ) {
         fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
      }
      test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
   }

   retval = PAPI_start( EventSet1 );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet1, user_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !TESTS_QUIET ) {
      printf("User PAPI_TOT_CYC: %lld\n",user_values[0]);
   }


   /***************************/
   /***************************/
   /* User+Kernel events      */
   /***************************/
   /***************************/

   retval = PAPI_create_eventset(&EventSet2);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet2, 0);

   /* we need to set domain to be as inclusive as possible */

   domain_opt.def_cidx=0;
   domain_opt.eventset=EventSet2;
   domain_opt.domain=PAPI_DOM_USER|PAPI_DOM_KERNEL;

   retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
   if (retval != PAPI_OK) {

      if (retval==PAPI_EPERM) {
         test_skip( __FILE__, __LINE__,
		    "this test; trying to set PAPI_DOM_ALL; need to run as root",
		    retval);
      }
      else {
         test_fail(__FILE__, __LINE__, "setting PAPI_DOM_ALL",retval);
      }
   }


   retval = PAPI_add_named_event(EventSet2, "PAPI_TOT_CYC");
   if (retval != PAPI_OK) {
      if ( !TESTS_QUIET ) {
         fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
      }
      test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
   }

   retval = PAPI_start( EventSet2 );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet2, userkernel_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !TESTS_QUIET ) {
      printf("User+Kernel PAPI_TOT_CYC: %lld\n",userkernel_values[0]);
   }



   /***************************/
   /***************************/
   /* DOMAIN_ALL  events      */
   /***************************/
   /***************************/

   retval = PAPI_create_eventset(&EventSet3);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet3, 0);

   /* we need to set domain to be as inclusive as possible */

   domain_opt.def_cidx=0;
   domain_opt.eventset=EventSet3;
   domain_opt.domain=PAPI_DOM_ALL;

   retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
   if (retval != PAPI_OK) {

      if (retval==PAPI_EPERM) {
         test_skip( __FILE__, __LINE__,
		    "this test; trying to set PAPI_DOM_ALL; need to run as root",
		    retval);
      }
      else {
         test_fail(__FILE__, __LINE__, "setting PAPI_DOM_ALL",retval);
      }
   }


   retval = PAPI_add_named_event(EventSet3, "PAPI_TOT_CYC");
   if (retval != PAPI_OK) {
      if ( !TESTS_QUIET ) {
         fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
      }
      test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
   }

   retval = PAPI_start( EventSet3 );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet3, all_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !TESTS_QUIET ) {
      printf("DOM_ALL PAPI_TOT_CYC: %lld\n",all_values[0]);
   }


   /***************************/
   /***************************/
   /* PAPI_GRN_THR  events */
   /***************************/
   /***************************/

   retval = PAPI_create_eventset(&EventSet4);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet4, 0);

   /* Set the granularity to system-wide */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet4;
   gran_opt.granularity=PAPI_GRN_THR;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      test_skip( __FILE__, __LINE__,
		      "this test; trying to set PAPI_GRN_THR",
		      retval);
   }


   retval = PAPI_add_named_event(EventSet4, "PAPI_TOT_CYC");
   if (retval != PAPI_OK) {
      if ( !TESTS_QUIET ) {
         fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
      }
      test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
   }

   retval = PAPI_start( EventSet4 );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   do_flops( NUM_FLOPS );

   retval = PAPI_stop( EventSet4, grn_thr_values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !TESTS_QUIET ) {
      printf("GRN_THR PAPI_TOT_CYC: %lld\n",grn_thr_values[0]);
   }


   /***************************/
   /***************************/
   /* PAPI_GRN_PROC  events   */
   /***************************/
   /***************************/

   retval = PAPI_create_eventset(&EventSet5);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet5, 0);

   /* Set the granularity to system-wide */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet5;
   gran_opt.granularity=PAPI_GRN_PROC;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      if (!TESTS_QUIET) {
         printf("Unable to set PAPI_GRN_PROC\n");
      }
   }
   else {
      retval = PAPI_add_named_event(EventSet5, "PAPI_TOT_CYC");
      if (retval != PAPI_OK) {
         if ( !TESTS_QUIET ) {
            fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
         }
         test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
      }

      retval = PAPI_start( EventSet5 );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_start", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_stop( EventSet5, grn_proc_values );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
      }

      if ( !TESTS_QUIET ) {
         printf("GRN_PROC PAPI_TOT_CYC: %lld\n",grn_proc_values[0]);
      }
   }



   /***************************/
   /***************************/
   /* PAPI_GRN_SYS  events    */
   /***************************/
   /***************************/

   retval = PAPI_create_eventset(&EventSet6);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet6, 0);

   /* Set the granularity to system-wide */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet6;
   gran_opt.granularity=PAPI_GRN_SYS;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      if (!TESTS_QUIET) {
         printf("Unable to set PAPI_GRN_SYS\n");
      }
   }
   else {
      retval = PAPI_add_named_event(EventSet6, "PAPI_TOT_CYC");
      if (retval != PAPI_OK) {
         if ( !TESTS_QUIET ) {
            fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
         }
         test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
      }

      retval = PAPI_start( EventSet6 );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_start", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_stop( EventSet6, grn_sys_values );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
      }

      if ( !TESTS_QUIET ) {
         printf("GRN_SYS PAPI_TOT_CYC: %lld\n",grn_sys_values[0]);
      }
   }


   /***************************/
   /***************************/
   /* PAPI_GRN_SYS_CPU  events    */
   /***************************/
   /***************************/

   retval = PAPI_create_eventset(&EventSet7);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet7, 0);

   /* Set the granularity to system-wide */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet7;
   gran_opt.granularity=PAPI_GRN_SYS_CPU;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      if (!TESTS_QUIET) {
         printf("Unable to set PAPI_GRN_SYS_CPU\n");
      }
   }
   else {
      retval = PAPI_add_named_event(EventSet7, "PAPI_TOT_CYC");
      if (retval != PAPI_OK) {
         if ( !TESTS_QUIET ) {
            fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
         }
         test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
      }

      retval = PAPI_start( EventSet7 );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_start", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_stop( EventSet7, grn_sys_cpu_values );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
      }

      if ( !TESTS_QUIET ) {
         printf("GRN_SYS_CPU PAPI_TOT_CYC: %lld\n",grn_sys_cpu_values[0]);
      }
   }


   /***************************/
   /***************************/
   /* BLARGH  events    */
   /***************************/
   /***************************/

   retval = PAPI_create_eventset(&EventSet8);
   if (retval != PAPI_OK) {
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
   }

   retval = PAPI_assign_eventset_component(EventSet8, 0);

   /* Set the granularity to system-wide */

   gran_opt.def_cidx=0;
   gran_opt.eventset=EventSet8;
   gran_opt.granularity=PAPI_GRN_SYS;

   retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
   if (retval != PAPI_OK) {
      if (!TESTS_QUIET) {
         printf("Unable to set PAPI_GRN_SYS\n");
      }
   }
   else {
      /* we need to set to a certain cpu for uncore to work */
      PAPI_cpu_option_t cpu_opt;

      cpu_opt.eventset=EventSet8;
      cpu_opt.cpu_num=0;

      retval = PAPI_set_opt(PAPI_CPU_ATTACH,(PAPI_option_t*)&cpu_opt);
      if (retval != PAPI_OK) {
         test_fail(__FILE__, __LINE__, "PAPI_CPU_ATTACH",retval);
      }

      retval = PAPI_add_named_event(EventSet8, "PAPI_TOT_CYC");
      if (retval != PAPI_OK) {
         if ( !TESTS_QUIET ) {
            fprintf(stderr,"Error trying to add PAPI_TOT_CYC\n");
         }
         test_fail(__FILE__, __LINE__, "adding PAPI_TOT_CYC ",retval);
      }

      retval = PAPI_start( EventSet8 );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_start", retval );
      }

      do_flops( NUM_FLOPS );

      retval = PAPI_stop( EventSet8, total_values );
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
      }

      if ( !TESTS_QUIET ) {
         printf("TOTAL PAPI_TOT_CYC: %lld\n",total_values[0]);
      }
   }


   test_pass( __FILE__, NULL, 0 );

   return 0;
}
