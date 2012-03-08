/* 
* File:    cmpinfo.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/
#include <stdlib.h>
#include <stdio.h>
#include "papi_test.h"

int
main( int argc, char **argv )
{
	int retval;

	const PAPI_component_info_t *cmpinfo;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	if ( ( cmpinfo = PAPI_get_component_info( 0 ) ) == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_component_info", retval );

	printf( "name: %s\n", cmpinfo->name );
	printf( "substrate_version: %s\n", cmpinfo->version );
	printf( "support_version: %s\n", cmpinfo->support_version );
	printf( "kernel_version: %s\n", cmpinfo->kernel_version );
	printf( "num_cntrs: %d\n", cmpinfo->num_cntrs );
	printf( "num_mpx_cntrs: %d\n", cmpinfo->num_mpx_cntrs );
	printf( "num_preset_events: %d\n", cmpinfo->num_preset_events );	/* Number of counters the substrate supports */
	printf( "num_native_events: %d\n", cmpinfo->num_native_events );	/* Number of counters the substrate supports */
	printf( "default_domain: 0x%x (%s)\n", cmpinfo->default_domain,
			stringify_all_domains( cmpinfo->default_domain ) );
	printf( "available_domains: 0x%x (%s)\n", cmpinfo->available_domains, stringify_all_domains( cmpinfo->available_domains ) );	/* Available domains */
	printf( "default_granularity: 0x%x (%s)\n", cmpinfo->default_granularity,
			stringify_granularity( cmpinfo->default_granularity ) );
	/* The default granularity when this substrate is used */
	printf( "available_granularities: 0x%x (%s)\n", cmpinfo->available_granularities, stringify_all_granularities( cmpinfo->available_granularities ) );	/* Available granularities */
	printf( "hardware_intr_sig: %d\n", cmpinfo->hardware_intr_sig );	/* Width of opcode matcher if exists, 0 if not */
	printf( "opcode_match_width: %d\n", cmpinfo->opcode_match_width );	/* Width of opcode matcher if exists, 0 if not */
/*   printf("reserved_ints[4]: %d\n",cmpinfo->reserved_ints[4]); */
	printf( "hardware_intr: %d\n", cmpinfo->hardware_intr );	/* Needs hw overflow intr to be emulated in software */
	printf( "precise_intr: %d\n", cmpinfo->precise_intr );	/* Performance interrupts happen precisely */
	printf( "posix1b_timers: %d\n", cmpinfo->posix1b_timers );	/* Performance interrupts happen precisely */
	printf( "kernel_profile: %d\n", cmpinfo->kernel_profile );	/* Needs kernel profile support (buffered interrupts) to be emulated */
	printf( "kernel_multiplex: %d\n", cmpinfo->kernel_multiplex );	/* In kernel multiplexing */
	printf( "data_address_range: %d\n", cmpinfo->data_address_range );	/* Supports data address range limiting */
	printf( "instr_address_range: %d\n", cmpinfo->instr_address_range );	/* Supports instruction address range limiting */
	printf( "fast_counter_read: %d\n", cmpinfo->fast_counter_read );	/* Has a fast counter read */
	printf( "fast_real_timer: %d\n", cmpinfo->fast_real_timer );	/* Has a fast real timer */
	printf( "fast_virtual_timer: %d\n", cmpinfo->fast_virtual_timer );	/* Has a fast virtual timer */
	printf( "attach: %d\n", cmpinfo->attach );	/* Supports attach */
	printf( "attach_must_ptrace: %d\n", cmpinfo->attach_must_ptrace );	/* */
	printf( "profile_ear: %d\n", cmpinfo->profile_ear );	/* Supports data/instr/tlb miss address sampling */
	printf( "cntr_groups: %d\n", cmpinfo->cntr_groups );	/* Underlying hardware uses counter groups */
	printf( "cntr_umasks: %d\n", cmpinfo->cntr_umasks );	/* counters have unit masks */
	printf( "cntr_IEAR_events: %d\n", cmpinfo->cntr_IEAR_events );	/* counters support instr event addr register */
	printf( "cntr_DEAR_events: %d\n", cmpinfo->cntr_DEAR_events );	/* counters support data event addr register */
	printf( "cntr_OPCM_events: %d\n", cmpinfo->cntr_OPCM_events );	/* counter events support opcode matching */

	test_pass( __FILE__, NULL, 0 );
	exit( 0 );
}
