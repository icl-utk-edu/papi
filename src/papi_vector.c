/*
 * File:    linux.c
 * CVS:     $Id$
 * Author:  Kevin London
 *          london@cs.utk.edu
 * Mods:    Haihang You
 *	       you@cs.utk.edu
 * Mods:    <Your name here>
 *          <Your email here>
 */

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

/* Prototypes */
int vec_int_ok_dummy(  );
int vec_int_one_dummy(  );
int vec_int_dummy(  );
void vec_void_dummy(  );
void *vec_void_star_dummy(  );
long long vec_long_long_dummy(  );
char *vec_char_star_dummy(  );
long vec_long_dummy(  );
long long vec_dummy_get_virt_cycles( const hwd_context_t * zero );
long long vec_dummy_get_virt_usec( const hwd_context_t * zero );
long long vec_dummy_get_real_usec( void );
long long vec_dummy_get_real_cycles( void );

extern papi_vector_t MY_VECTOR;
#ifdef HAVE_ACPI
extern papi_vector_t _acpi_vector;
#endif
#ifdef HAVE_MX
extern papi_vector_t _mx_vector;
#endif
#ifdef HAVE_NET
extern papi_vector_t _net_vector;
#endif
#ifdef HAVE_LMSENSORS
extern papi_vector_t _lmsensors_vector;
#endif


papi_vector_t *_papi_hwd[] = {
	&MY_VECTOR,
#ifdef HAVE_ACPI
	&_acpi_vector,
#endif
#ifdef HAVE_MX
	&_mx_vector,
#endif
#ifdef HAVE_NET
	&_net_vector,
#endif
#ifdef HAVE_LMSENSORS
	&_lmsensors_vector,
#endif
	NULL
};
int papi_num_components = ( sizeof ( _papi_hwd ) / sizeof ( *_papi_hwd ) ) - 1;

void
_vectors_error(  )
{
	SUBDBG( "function is not implemented in the component!\n" );
	exit( PAPI_ESBSTR );
}

long long
vec_dummy_get_real_usec( void )
{
#ifdef _WIN32
	LARGE_INTEGER PerformanceCount, Frequency;
	QueryPerformanceCounter( &PerformanceCount );
	QueryPerformanceFrequency( &Frequency );
	return ( ( PerformanceCount.QuadPart * 1000000 ) / Frequency.QuadPart );
#else
	struct timeval tv;
	gettimeofday( &tv, NULL );
	return ( ( tv.tv_sec * 1000000 ) + tv.tv_usec );
#endif
}

long long
vec_dummy_get_real_cycles( void )
{
	float usec, cyc;

	usec = ( float ) vec_dummy_get_real_usec(  );
	cyc = usec * _papi_hwi_system_info.hw_info.mhz;
	return ( ( long long ) cyc );
}

#if ((defined _BGL) || (defined _BGP))
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

long long
vec_dummy_get_virt_usec( const hwd_context_t * zero )
{
	( void ) zero;			 /*unused */
	long long retval;
#if ((defined _BGL) || (defined _BGP))
	struct rusage ruse;
	getrusage( RUSAGE_SELF, &ruse );
	retval =
		( long long ) ( ruse.ru_utime.tv_sec * 1000000 +
						ruse.ru_utime.tv_usec );
#elif _WIN32
	/* identical code is found in the windows substrate */
	HANDLE p;
	BOOL ret;
	FILETIME Creation, Exit, Kernel, User;
	long long virt;

	p = GetCurrentProcess(  );
	ret = GetProcessTimes( p, &Creation, &Exit, &Kernel, &User );
	if ( ret ) {
		virt =
			( ( ( long long ) ( Kernel.dwHighDateTime +
								User.dwHighDateTime ) ) << 32 )
			+ Kernel.dwLowDateTime + User.dwLowDateTime;
		retval = virt / 1000;
	} else
		return ( PAPI_ESBSTR );
#else
	struct tms buffer;
	times( &buffer );
	retval =
		( long long ) buffer.tms_utime * ( long long ) ( 1000000 /
														 sysconf
														 ( _SC_CLK_TCK ) );
#endif
	return ( retval );
}

long long
vec_dummy_get_virt_cycles( const hwd_context_t * zero )
{
	float usec, cyc;

	usec = ( float ) vec_dummy_get_virt_usec( zero );
	cyc = usec * _papi_hwi_system_info.hw_info.mhz;
	return ( ( long long ) cyc );
}

int
vec_int_ok_dummy(  )
{
	return PAPI_OK;
}

int
vec_int_one_dummy(  )
{
	return 1;
}

int
vec_int_dummy(  )
{
	return PAPI_ESBSTR;
}

void *
vec_void_star_dummy(  )
{
	return NULL;
}

void
vec_void_dummy(  )
{
	return;
}

long long
vec_long_long_dummy(  )
{
	return PAPI_ESBSTR;
}

char *
vec_char_star_dummy(  )
{
	return NULL;
}

long
vec_long_dummy(  )
{
	return PAPI_ESBSTR;
}

int
_papi_hwi_innoculate_vector( papi_vector_t * v )
{
	if ( !v )
		return ( PAPI_EINVAL );

	/* component function pointers */
#ifdef _WIN32				 /* Windows requires a different callback format */
	if ( !v->timer_callback )
		v->timer_callback =
			( void ( * )( UINT, UINT, DWORD, DWORD, DWORD ) ) vec_void_dummy;
#else
	if ( !v->dispatch_timer )
		v->dispatch_timer =
			( void ( * )( int, hwd_siginfo_t *, void * ) ) vec_void_dummy;
#endif
	if ( !v->get_overflow_address )
		v->get_overflow_address =
			( void *( * )( int, char *, int ) ) vec_void_star_dummy;
	if ( !v->start )
		v->start =
			( int ( * )( hwd_context_t *, hwd_control_state_t * ) )
			vec_int_dummy;
	if ( !v->stop )
		v->stop =
			( int ( * )( hwd_context_t *, hwd_control_state_t * ) )
			vec_int_dummy;
	if ( !v->read )
		v->read =
			( int ( * )
			  ( hwd_context_t *, hwd_control_state_t *, long long **,
				int ) ) vec_int_dummy;
	if ( !v->reset )
		v->reset =
			( int ( * )( hwd_context_t *, hwd_control_state_t * ) )
			vec_int_dummy;
	if ( !v->write )
		v->write =
			( int ( * )( hwd_context_t *, hwd_control_state_t *, long long[] ) )
			vec_int_dummy;
	if ( !v->get_real_cycles )
		v->get_real_cycles = ( long long ( * )(  ) ) vec_dummy_get_real_cycles;
	if ( !v->get_real_usec )
		v->get_real_usec = ( long long ( * )(  ) ) vec_dummy_get_real_usec;
	if ( !v->get_virt_cycles )
		v->get_virt_cycles = vec_dummy_get_virt_cycles;
	if ( !v->get_virt_usec )
		v->get_virt_usec = vec_dummy_get_virt_usec;
	if ( !v->stop_profiling )
		v->stop_profiling =
			( int ( * )( ThreadInfo_t *, EventSetInfo_t * ) ) vec_int_dummy;
	if ( !v->init_substrate )
		v->init_substrate = ( int ( * )( int ) ) vec_int_ok_dummy;
	if ( !v->init )
		v->init = ( int ( * )( hwd_context_t * ) ) vec_int_ok_dummy;
	if ( !v->init_control_state )
		v->init_control_state =
			( int ( * )( hwd_control_state_t * ptr ) ) vec_void_dummy;
	if ( !v->update_shlib_info )
		v->update_shlib_info = ( int ( * )( void ) ) vec_int_dummy;
	if ( !v->get_system_info )
		v->get_system_info = ( int ( * )(  ) ) vec_int_dummy;
	if ( !v->get_memory_info )
		v->get_memory_info =
			( int ( * )( PAPI_hw_info_t *, int ) ) vec_int_dummy;
	if ( !v->update_control_state )
		v->update_control_state =
			( int ( * )
			  ( hwd_control_state_t *, NativeInfo_t *, int,
				hwd_context_t * ) ) vec_int_dummy;
	if ( !v->ctl )
		v->ctl =
			( int ( * )( hwd_context_t *, int, _papi_int_option_t * ) )
			vec_int_dummy;
	if ( !v->set_overflow )
		v->set_overflow =
			( int ( * )( EventSetInfo_t *, int, int ) ) vec_int_dummy;
	if ( !v->set_profile )
		v->set_profile =
			( int ( * )( EventSetInfo_t *, int, int ) ) vec_int_dummy;
	if ( !v->add_prog_event )
		v->add_prog_event =
			( int ( * )
			  ( hwd_control_state_t *, unsigned int, void *,
				EventInfo_t * ) ) vec_int_dummy;
	if ( !v->set_domain )
		v->set_domain =
			( int ( * )( hwd_control_state_t *, int ) ) vec_int_dummy;
	if ( !v->ntv_enum_events )
		v->ntv_enum_events = ( int ( * )( unsigned int *, int ) ) vec_int_dummy;
	if ( !v->ntv_name_to_code )
		v->ntv_name_to_code =
			( int ( * )( char *, unsigned int * ) ) vec_int_dummy;
	if ( !v->ntv_code_to_name )
		v->ntv_code_to_name =
			( int ( * )( unsigned int, char *, int ) ) vec_int_dummy;
	if ( !v->ntv_code_to_descr )
		v->ntv_code_to_descr =
			( int ( * )( unsigned int, char *, int ) ) vec_int_ok_dummy;
	if ( !v->ntv_code_to_bits )
		v->ntv_code_to_bits =
			( int ( * )( unsigned int, hwd_register_t * ) ) vec_int_dummy;
	if ( !v->ntv_bits_to_info )
		v->ntv_bits_to_info =
			( int ( * )( hwd_register_t *, char *, unsigned int *, int, int ) )
			vec_int_dummy;
	if ( !v->allocate_registers )
		v->allocate_registers =
			( int ( * )( EventSetInfo_t * ) ) vec_int_one_dummy;
	if ( !v->bpt_map_avail )
		v->bpt_map_avail =
			( int ( * )( hwd_reg_alloc_t *, int ) ) vec_int_dummy;
	if ( !v->bpt_map_set )
		v->bpt_map_set =
			( void ( * )( hwd_reg_alloc_t *, int ) ) vec_void_dummy;
	if ( !v->bpt_map_exclusive )
		v->bpt_map_exclusive = ( int ( * )( hwd_reg_alloc_t * ) ) vec_int_dummy;
	if ( !v->bpt_map_shared )
		v->bpt_map_shared =
			( int ( * )( hwd_reg_alloc_t *, hwd_reg_alloc_t * ) ) vec_int_dummy;
	if ( !v->bpt_map_preempt )
		v->bpt_map_preempt =
			( void ( * )( hwd_reg_alloc_t *, hwd_reg_alloc_t * ) )
			vec_void_dummy;
	if ( !v->bpt_map_update )
		v->bpt_map_update =
			( void ( * )( hwd_reg_alloc_t *, hwd_reg_alloc_t * ) )
			vec_void_dummy;
	if ( !v->get_dmem_info )
		v->get_dmem_info = ( int ( * )( PAPI_dmem_info_t * ) ) vec_int_dummy;
	if ( !v->shutdown )
		v->shutdown = ( int ( * )( hwd_context_t * ) ) vec_int_dummy;
	if ( !v->shutdown_substrate )
		v->shutdown_substrate = ( int ( * )( void ) ) vec_int_ok_dummy;
	if ( !v->user )
		v->user = ( int ( * )( int, void *, void * ) ) vec_int_dummy;
	return PAPI_OK;
}

int
PAPI_user( int func_num, void *input, void *output, int cidx )
{
	return ( _papi_hwd[cidx]->user( func_num, input, output ) );
}

void *
vector_find_dummy( void *func, char **buf )
{
	void *ptr = NULL;

	if ( vec_int_ok_dummy == ( int ( * )(  ) ) func ) {
		ptr = ( void * ) vec_int_ok_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_int_ok_dummy" );
	} else if ( vec_int_one_dummy == ( int ( * )(  ) ) func ) {
		ptr = ( void * ) vec_int_one_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_int_one_dummy" );
	} else if ( vec_int_dummy == ( int ( * )(  ) ) func ) {
		ptr = ( void * ) vec_int_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_int_dummy" );
	} else if ( vec_void_dummy == ( void ( * )(  ) ) func ) {
		ptr = ( void * ) vec_void_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_void_dummy" );
	} else if ( vec_void_star_dummy == ( void *( * )(  ) ) func ) {
		ptr = ( void * ) vec_void_star_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_void_star_dummy" );
	} else if ( vec_long_long_dummy == ( long long ( * )(  ) ) func ) {
		ptr = ( void * ) vec_long_long_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_long_long_dummy" );
	} else if ( vec_char_star_dummy == ( char *( * )(  ) ) func ) {
		ptr = ( void * ) vec_char_star_dummy;
		*buf = papi_strdup( "vec_char_star_dummy" );
	} else if ( vec_long_dummy == ( long ( * )(  ) ) func ) {
		ptr = ( void * ) vec_long_dummy;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_long_dummy" );
	} else if ( vec_dummy_get_real_usec == ( long long ( * )( void ) ) func ) {
		ptr = ( void * ) vec_dummy_get_real_usec;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_dummy_get_real_usec" );
	} else if ( vec_dummy_get_real_cycles == ( long long ( * )( void ) ) func ) {
		ptr = ( void * ) vec_dummy_get_real_cycles;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_dummy_get_real_cycles" );
	} else if ( vec_dummy_get_virt_usec ==
				( long long ( * )( const hwd_context_t * ) ) func ) {
		ptr = ( void * ) vec_dummy_get_virt_usec;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_dummy_get_virt_usec" );
	} else if ( vec_dummy_get_virt_cycles ==
				( long long ( * )( const hwd_context_t * ) ) func ) {
		ptr = ( void * ) vec_dummy_get_virt_cycles;
		if ( buf != NULL )
			*buf = papi_strdup( "vec_dummy_get_virt_cycles" );
	} else {
		ptr = NULL;
	}
	return ( ptr );
}

void
vector_print_routine( void *func, char *fname, int pfunc )
{
	void *ptr = NULL;
	char *buf = NULL;

	ptr = vector_find_dummy( func, &buf );
	if ( ptr ) {
		printf( "%s: %s is mapped to %s.\n", ( ptr ? "DUMMY" : "function" ),
				fname, buf );
		papi_free( buf );
	} else if ( ( !ptr && pfunc ) )
		printf( "%s: %s is mapped to %p.\n", ( ptr ? "DUMMY" : "function" ),
				fname, func );
}

void
vector_print_table( papi_vector_t * v, int print_func )
{

	if ( !v )
		return;

#ifdef _WIN32				 /* Windows requires a different callback format */
	vector_print_routine( ( void * ) v->timer_callback,
						  "_papi_hwd_timer_callback", print_func );
#else
	vector_print_routine( ( void * ) v->dispatch_timer,
						  "_papi_hwd_dispatch_timer", print_func );
#endif
	vector_print_routine( ( void * ) v->get_overflow_address,
						  "_papi_hwd_get_overflow_address", print_func );
	vector_print_routine( ( void * ) v->start, "_papi_hwd_start", print_func );
	vector_print_routine( ( void * ) v->stop, "_papi_hwd_stop", print_func );
	vector_print_routine( ( void * ) v->read, "_papi_hwd_read", print_func );
	vector_print_routine( ( void * ) v->reset, "_papi_hwd_reset", print_func );
	vector_print_routine( ( void * ) v->write, "_papi_hwd_write", print_func );
	vector_print_routine( ( void * ) v->get_real_cycles,
						  "_papi_hwd_get_real_cycles", print_func );
	vector_print_routine( ( void * ) v->get_real_usec,
						  "_papi_hwd_get_real_usec", print_func );
	vector_print_routine( ( void * ) v->get_virt_cycles,
						  "_papi_hwd_get_virt_cycles", print_func );
	vector_print_routine( ( void * ) v->get_virt_usec,
						  "_papi_hwd_get_virt_usec", print_func );
	vector_print_routine( ( void * ) v->stop_profiling,
						  "_papi_hwd_stop_profiling", print_func );
	vector_print_routine( ( void * ) v->init_substrate,
						  "_papi_hwd_init_substrate", print_func );
	vector_print_routine( ( void * ) v->init, "_papi_hwd_init", print_func );
	vector_print_routine( ( void * ) v->init_control_state,
						  "_papi_hwd_init_control_state", print_func );
	vector_print_routine( ( void * ) v->ctl, "_papi_hwd_ctl", print_func );
	vector_print_routine( ( void * ) v->set_overflow, "_papi_hwd_set_overflow",
						  print_func );
	vector_print_routine( ( void * ) v->set_profile, "_papi_hwd_set_profile",
						  print_func );
	vector_print_routine( ( void * ) v->add_prog_event,
						  "_papi_hwd_add_prog_event", print_func );
	vector_print_routine( ( void * ) v->set_domain, "_papi_hwd_set_domain",
						  print_func );
	vector_print_routine( ( void * ) v->ntv_enum_events,
						  "_papi_hwd_ntv_enum_events", print_func );
	vector_print_routine( ( void * ) v->ntv_name_to_code,
						  "_papi_hwd_ntv_name_to_code", print_func );
	vector_print_routine( ( void * ) v->ntv_code_to_name,
						  "_papi_hwd_ntv_code_to_name", print_func );
	vector_print_routine( ( void * ) v->ntv_code_to_descr,
						  "_papi_hwd_ntv_code_to_descr", print_func );
	vector_print_routine( ( void * ) v->ntv_code_to_bits,
						  "_papi_hwd_ntv_code_to_bits", print_func );
	vector_print_routine( ( void * ) v->ntv_bits_to_info,
						  "_papi_hwd_ntv_bits_to_info", print_func );
	vector_print_routine( ( void * ) v->allocate_registers,
						  "_papi_hwd_allocate_registers", print_func );
	vector_print_routine( ( void * ) v->bpt_map_avail,
						  "_papi_hwd_bpt_map_avail", print_func );
	vector_print_routine( ( void * ) v->bpt_map_set, "_papi_hwd_bpt_map_set",
						  print_func );
	vector_print_routine( ( void * ) v->bpt_map_exclusive,
						  "_papi_hwd_bpt_map_exclusive", print_func );
	vector_print_routine( ( void * ) v->bpt_map_shared, "_papi_hwd_bpt_shared",
						  print_func );
	vector_print_routine( ( void * ) v->bpt_map_update,
						  "_papi_hwd_bpt_map_update", print_func );
	vector_print_routine( ( void * ) v->get_dmem_info,
						  "_papi_hwd_get_dmem_info", print_func );
	vector_print_routine( ( void * ) v->shutdown, "_papi_hwd_shutdown",
						  print_func );
	vector_print_routine( ( void * ) v->shutdown_substrate,
						  "_papi_hwd_shutdown_substrate", print_func );
	vector_print_routine( ( void * ) v->user, "_papi_hwd_user", print_func );
}
