/*
* File:    sprofile.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

/* These architectures use Function Descriptors as Function Pointers */

#if (defined(linux) && defined(__ia64__)) || (defined(_AIX)) \
  || ((defined(__powerpc64__) && (_CALL_ELF != 2)))
/* PPC64 Big Endian is ELF version 1 which uses function descriptors */
#define DO_READS (unsigned long)(*(void **)do_reads)
#define DO_FLOPS (unsigned long)(*(void **)do_flops)
#else
/* PPC64 Little Endian is ELF version 2 which does not use
 * function descriptors
 */
#define DO_READS (unsigned long)(do_reads)
#define DO_FLOPS (unsigned long)(do_flops)
#endif

/* This file performs the following test: sprofile */

#include "papi_test.h"
#include "prof_utils.h"

int
main( int argc, char **argv )
{
	int i, num_events, num_tests = 6, mask = 0x1;
	int EventSet = PAPI_NULL;
	unsigned short **buf = ( unsigned short ** ) profbuf;
	unsigned long length, blength;
	int num_buckets;
	PAPI_sprofil_t sprof[3];
	int retval;
	const PAPI_exe_info_t *prginfo;
	caddr_t start, end;

	prof_init( argc, argv, &prginfo );

	start = prginfo->address_info.text_start;
	end = prginfo->address_info.text_end;
	if ( start > end ) {
	   test_fail( __FILE__, __LINE__, "Profile length < 0!", PAPI_ESYS );
	}
	length = ( unsigned long ) ( end - start );
	prof_print_address
		( "Test case sprofile: POSIX compatible profiling over multiple regions.\n",
		  prginfo );
	blength =
		prof_size( length, FULL_SCALE, PAPI_PROFIL_BUCKET_16, &num_buckets );
	prof_alloc( 3, blength );

	/* First half */
	sprof[0].pr_base = buf[0];
	sprof[0].pr_size = ( unsigned int ) blength;
	sprof[0].pr_off = ( caddr_t ) DO_FLOPS;
#if defined(linux) && defined(__ia64__)
	if ( !TESTS_QUIET )
		fprintf( stderr, "do_flops is at %p %p\n", &do_flops, sprof[0].pr_off );
#endif
	sprof[0].pr_scale = FULL_SCALE;
	/* Second half */
	sprof[1].pr_base = buf[1];
	sprof[1].pr_size = ( unsigned int ) blength;
	sprof[1].pr_off = ( caddr_t ) DO_READS;
#if defined(linux) && defined(__ia64__)
	if ( !TESTS_QUIET )
		fprintf( stderr, "do_reads is at %p %p\n", &do_reads, sprof[1].pr_off );
#endif
	sprof[1].pr_scale = FULL_SCALE;
	/* Overflow bin */
	sprof[2].pr_base = buf[2];
	sprof[2].pr_size = 1;
	sprof[2].pr_off = 0;
	sprof[2].pr_scale = 0x2;

	EventSet = add_test_events( &num_events, &mask, 1 );

	values = allocate_test_space( num_tests, num_events );

	if ( ( retval = PAPI_sprofil( sprof, 3, EventSet, PAPI_TOT_CYC, THRESHOLD,
								  PAPI_PROFIL_POSIX | PAPI_PROFIL_BUCKET_16 ) )
		 != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_sprofil", retval );

	do_stuff(  );

	if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_stuff(  );

	if ( ( retval = PAPI_stop( EventSet, values[1] ) ) != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

	/* clear the profile flag before removing the event */
	if ( ( retval = PAPI_sprofil( sprof, 3, EventSet, PAPI_TOT_CYC, 0,
								  PAPI_PROFIL_POSIX | PAPI_PROFIL_BUCKET_16 ) )
		 != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_sprofil", retval );

	remove_test_events( &EventSet, mask );



	if ( !TESTS_QUIET ) {
		printf( "Test case: PAPI_sprofil()\n" );
		printf( "---------Buffer 1--------\n" );
		for ( i = 0; i < ( int ) length / 2; i++ ) {
			if ( buf[0][i] )
				printf( "%#lx\t%d\n", DO_FLOPS + 2 * ( unsigned long ) i,
						buf[0][i] );
		}
		printf( "---------Buffer 2--------\n" );
		for ( i = 0; i < ( int ) length / 2; i++ ) {
			if ( buf[1][i] )
				printf( "%#lx\t%d\n", DO_READS + 2 * ( unsigned long ) i,
						buf[1][i] );
		}
		printf( "-------------------------\n" );
		printf( "%u samples fell outside the regions.\n", *buf[2] );
	}
	retval = prof_check( 2, PAPI_PROFIL_BUCKET_16, num_buckets );

	for ( i = 0; i < 3; i++ ) {
		free( profbuf[i] );
	}
	if ( retval == 0 )
		test_fail( __FILE__, __LINE__, "No information in buffers", 1 );

	test_pass( __FILE__, values, num_tests );

	exit( 1 );
}
