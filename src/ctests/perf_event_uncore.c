/* This file tests uncore events on perf_event kernels
*/

#include "papi_test.h"

#define MAX_CYCLE_ERROR 30


int
main( int argc, char **argv )
{
	int retval;
	int EventSet = PAPI_NULL;
	long long values[1];
	const PAPI_hw_info_t *hwinfo;
        char *uncore_event=NULL;
        char *cpu_type=NULL;
        char skip_message[BUFSIZ];

	/* Set TESTS_QUIET variable */
	tests_quiet( argc, argv );	

	/* Init the PAPI library */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
	   test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	hwinfo = PAPI_get_hardware_info();
	if ( hwinfo == NULL ) {
	   test_fail(__FILE__, __LINE__, 
		     "PAPI_get_hardware_info failed, THIS should not happen.", 
		     PAPI_ESYS);
	}
	if ( hwinfo->vendor != PAPI_VENDOR_INTEL ) {
	   test_skip( __FILE__, __LINE__, 
		      "This test is only for Intel Processors, for now.", 
		      PAPI_OK );
	}
   
        /* Find out what processor we are running on */
	if ( hwinfo->cpuid_family == 6) {
	   if (hwinfo->cpuid_model == 45) {
	      /* SandyBridge EP */
	      cpu_type=strdup("SandyBridge EP");
	      uncore_event=strdup("snbep_unc_imc0::UNC_M_CLOCKTICKS");
	   }
	   if (hwinfo->cpuid_model == 58) {
	      /* IvyBridge */
	      cpu_type=strdup("IvyBridge");
	   }	   	   
	}
   
        if (uncore_event==NULL) {
	   sprintf(skip_message,
		   "This test currently does not support family %d model %d CPUs",
		   hwinfo->cpuid_family, hwinfo->cpuid_model);
	   test_skip( __FILE__, __LINE__, skip_message, PAPI_OK );
	}

	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, "PAPI_create_eventset",retval);
	}

	/* we need to set a component for the EventSet */

	retval = PAPI_assign_eventset_component(EventSet, 0);

	/* we need to set to a certain cpu for uncore to work */

	PAPI_cpu_option_t cpu_opt;

	cpu_opt.eventset=EventSet;
	cpu_opt.cpu_num=0;

	retval = PAPI_set_opt(PAPI_CPU_ATTACH,(PAPI_option_t*)&cpu_opt);
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, "PAPI_CPU_ATTACH",retval);
	}

	/* we need to set the granularity to system-wide for uncore to work */

	PAPI_granularity_option_t gran_opt;
	
	gran_opt.def_cidx=0;
	gran_opt.eventset=EventSet;
	gran_opt.granularity=PAPI_GRN_SYS;

	retval = PAPI_set_opt(PAPI_GRANUL,(PAPI_option_t*)&gran_opt);
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, "PAPI_GRANUL",retval);
	}
	
	/* we need to set domain to be as inclusive as possible */
	
	PAPI_domain_option_t domain_opt;

	domain_opt.def_cidx=0;
	domain_opt.eventset=EventSet;
	domain_opt.domain=PAPI_DOM_ALL;

	retval = PAPI_set_opt(PAPI_DOMAIN,(PAPI_option_t*)&domain_opt);
	if (retval != PAPI_OK) {
	   test_fail(__FILE__, __LINE__, "PAPI_DOMAIN",retval);
	}

	/* Add our uncore event */

	retval = PAPI_add_named_event(EventSet, uncore_event);

	if (retval != PAPI_OK) {
	  test_fail(__FILE__, __LINE__, "Error with event \n",retval);
	}

	/* Start PAPI */
	retval = PAPI_start( EventSet );
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

	if ( !TESTS_QUIET ) {
	   printf("Uncore test:\n");
	   printf("Using event %s on %s\n",uncore_event,cpu_type);
	   printf("\t%s: %lld\n",uncore_event,values[0]);
	}

	test_pass( __FILE__, NULL, 0 );
	
	return 0;
}
