/* 
* File:    profile.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Dan Terpstra
*          terpstra@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: profiling and program info option call

   - This tests the SVR4 profiling interface of PAPI. These are counted 
   in the default counting domain and default granularity, depending on 
   the platform. Usually this is the user domain (PAPI_DOM_USER) and 
   thread context (PAPI_GRN_THR).

     The Eventset contains:
     + PAPI_FP_INS (to profile)
     + PAPI_TOT_CYC

   - Set up profile
   - Start eventset 1
   - Do both (flops and reads)
   - Stop eventset 1
*/

#include "papi_test.h"
#include "prof_utils.h"
#define PROFILE_ALL

static int do_profile( caddr_t start, unsigned long plength, unsigned scale,
					   int thresh, int bucket );

int
main( int argc, char **argv )
{
	int num_tests = 6;
	long length;
	int mask;
	int retval;
	int mythreshold = THRESHOLD;
	const PAPI_exe_info_t *prginfo;
	caddr_t start, end;

	prof_init( argc, argv, &prginfo );
	mask = prof_events( num_tests );

#ifdef PROFILE_ALL
/* use these lines to profile entire code address space */
	start = prginfo->address_info.text_start;
	end = prginfo->address_info.text_end;
#else
/* use these lines to profile only do_flops address space */
	start = ( caddr_t ) do_flops;
	end = ( caddr_t ) fdo_flops;
/* Itanium and ppc64 processors return function descriptors instead of function addresses.
   You must dereference the descriptor to get the address.
*/
#if defined(ITANIUM1) || defined(ITANIUM2) || defined(__powerpc64__)
	start = ( caddr_t ) ( ( ( struct fdesc * ) start )->ip );
	end = ( caddr_t ) ( ( ( struct fdesc * ) end )->ip );
#endif
#endif

#if defined(linux)
	{
		char *tmp = getenv( "THRESHOLD" );
		if ( tmp )
			mythreshold = atoi( tmp );
	}
#endif

	length = end - start;
	if ( length < 0 )
		test_fail( __FILE__, __LINE__, "Profile length < 0!", ( int ) length );

	prof_print_address
		( "Test case profile: POSIX compatible profiling with hardware counters.\n",
		  prginfo );
	prof_print_prof_info( start, end, mythreshold, event_name );
	retval =
		do_profile( start, ( unsigned long ) length, FULL_SCALE, mythreshold,
					PAPI_PROFIL_BUCKET_16 );
	if ( retval == PAPI_OK )
		retval =
			do_profile( start, ( unsigned long ) length, FULL_SCALE,
						mythreshold, PAPI_PROFIL_BUCKET_32 );
	if ( retval == PAPI_OK )
		retval =
			do_profile( start, ( unsigned long ) length, FULL_SCALE,
						mythreshold, PAPI_PROFIL_BUCKET_64 );

	remove_test_events( &EventSet, mask );
	test_pass( __FILE__, values, num_tests );
	exit( 1 );
}

static int
do_profile( caddr_t start, unsigned long plength, unsigned scale, int thresh,
			int bucket )
{
	int i, retval;
	unsigned long blength;
	int num_buckets;

	char *profstr[5] = { "PAPI_PROFIL_POSIX",
		"PAPI_PROFIL_RANDOM",
		"PAPI_PROFIL_WEIGHTED",
		"PAPI_PROFIL_COMPRESS",
		"PAPI_PROFIL_<all>"
	};

	int profflags[5] = { PAPI_PROFIL_POSIX,
		PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM,
		PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED,
		PAPI_PROFIL_POSIX | PAPI_PROFIL_COMPRESS,
		PAPI_PROFIL_POSIX | PAPI_PROFIL_WEIGHTED |
			PAPI_PROFIL_RANDOM | PAPI_PROFIL_COMPRESS
	};

	do_no_profile(  );
	blength = prof_size( plength, scale, bucket, &num_buckets );
	prof_alloc( 5, blength );

	for ( i = 0; i < 5; i++ ) {
		if ( !TESTS_QUIET )
			printf( "Test type   : \t%s\n", profstr[i] );

#ifndef SWPROFILE
		if ( ( retval =
			   PAPI_profil( profbuf[i], ( unsigned int ) blength, start, scale,
							EventSet, PAPI_event, thresh,
							profflags[i] | bucket ) ) != PAPI_OK ) {
		   if (retval==PAPI_ENOSUPP) { 
		      char warning[BUFSIZ];
		    
		      sprintf(warning,"PAPI_profil %s not supported",
			      profstr[i]);
		      test_warn( __FILE__, __LINE__, warning, 1 );
		   }		   
		   else {
		      test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
		   }
		}
#else
		if ( ( retval =
			   PAPI_profil( profbuf[i], ( unsigned int ) blength, start, scale,
							EventSet, PAPI_event, thresh,
							profflags[i] | bucket | PAPI_PROFIL_FORCE_SW ) ) !=
			 PAPI_OK ) {
		   test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
		}
#endif

		if ( retval != PAPI_OK )
			break;

		if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_start", retval );

		do_flops( getenv( "NUM_FLOPS" ) ? atoi( getenv( "NUM_FLOPS" ) ) :
				  NUM_FLOPS );

		if ( ( retval = PAPI_stop( EventSet, values[1] ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_stop", retval );

		if ( !TESTS_QUIET ) {
			printf( TAB1, event_name, ( values[1] )[0] );
			printf( TAB1, "PAPI_TOT_CYC", ( values[1] )[1] );
		}
		if ( ( retval =
			   PAPI_profil( profbuf[i], ( unsigned int ) blength, start, scale,
							EventSet, PAPI_event, 0,
							profflags[i] ) ) != PAPI_OK )
			test_fail( __FILE__, __LINE__, "PAPI_profil", retval );
	}

	if ( retval == PAPI_OK ) {
		prof_head( blength, bucket, num_buckets,
				   "address\t\t\tflat\trandom\tweight\tcomprs\tall\n" );
		prof_out( start, 5, bucket, num_buckets, scale );
		retval = prof_check( 5, bucket, num_buckets );
	}

	for ( i = 0; i < 5; i++ ) {
		free( profbuf[i] );
	}

	return ( retval );
}
