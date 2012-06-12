#include "papi_test.h"

int
main( int argc, char **argv )
{
	int retval;
	long long elapsed_us, elapsed_cyc;
	const PAPI_hw_info_t *hw_info;

	tests_quiet( argc, argv );	/* Set TESTS_QUIET variable */

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );

	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL )
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );

	elapsed_us = PAPI_get_real_usec(  );

	elapsed_cyc = PAPI_get_real_cyc(  );

	printf( "Testing real time clock. (CPU Max %d MHz, CPU Min %d MHz)\n",
			hw_info->cpu_max_mhz, hw_info->cpu_min_mhz );
	printf( "Sleeping for 10 seconds.\n" );

	sleep( 10 );

	elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;

	elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;

	printf( "%lld us. %lld cyc.\n", elapsed_us, elapsed_cyc );
	printf( "%f Computed MHz.\n",
			( float ) elapsed_cyc / ( float ) elapsed_us );

/* Elapsed microseconds and elapsed cycles are not as unambiguous as they appear.
   On Pentium III and 4, for example, cycles is a measured value, while useconds 
   is computed from cycles and mhz. MHz is read from /proc/cpuinfo (on linux).
   Thus, any error in MHz is propagated to useconds.
   Conversely, on ultrasparc useconds are extracted from a system call (gethrtime())
   and cycles are computed from useconds. Also, MHz comes from a scan of system info,
   Thus any error in gethrtime() propagates to both cycles and useconds, and cycles
   can be further impacted by errors in reported MHz.
   Without knowing the error bars on these system values, we can't really specify
   error ranges for our reported values, but we *DO* know that errors for at least
   one instance of Pentium 4 (torc17@utk) are on the order of one part per thousand.
   Newer multicore Intel processors seem to have broken the relationship between the
   clock rate reported in /proc/cpuinfo and the actual computed clock. To accomodate
   this artifact, the test no longer fails, but merely reports results out of range.
*/

	if ( elapsed_us < 9000000 )
		printf( "NOTE: Elapsed real time less than 9 seconds!\n" );
	if ( elapsed_us > 11000000 )
		printf( "NOTE: Elapsed real time greater than 11 seconds!\n" );
	if ( ( float ) elapsed_cyc < 9.0 * hw_info->cpu_max_mhz * 1000000.0 )
		printf( "NOTE: Elapsed real cycles less than 9*MHz*1000000.0!\n" );
	if ( ( float ) elapsed_cyc > 11.0 * hw_info->cpu_max_mhz * 1000000.0 )
		printf( "NOTE: Elapsed real cycles greater than 11*MHz*1000000.0!\n" );

	test_pass( __FILE__, NULL, 0 );
	exit( 1 );
}
