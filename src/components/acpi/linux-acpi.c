/** @file linux-acpi.c
  */
#include <inttypes.h>
#include "papi.h"
#include "papi_internal.h"
#include "linux-acpi.h"
#include "papi_memory.h"

extern papi_vector_t _acpi_vector;
int init_presets(  );

enum native_name_acpi
{
	PNE_ACPI_STAT = 0x40000000,
	PNE_ACPI_TEMP,
};

ACPI_native_event_entry_t acpi_native_table[] = {
	{{1, {"/proc/stat"}},
	 "ACPI_STAT",
	 "kernel statistics",
	 },
	{{2, {"/proc/acpi"}},
	 "ACPI_TEMP",
	 "ACPI temperature",
	 },
	{{0, {NULL}}, NULL, NULL},
};

/*
 * Substrate setup and shutdown
 */

/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
ACPI_init_substrate(  )
{
	int retval = PAPI_OK;

	/* retval = _papi_hwi_setup_vector_table( vtable, &_acpi_vector); */

#ifdef DEBUG
	/* This prints out which functions are mapped to dummy routines
	 * and this should be taken out once the substrate is completed.
	 * The 0 argument will print out only dummy routines, change
	 * it to a 1 to print out all routines.
	 */
	vector_print_table( &_acpi_vector, 0 );
#endif
	/* Internal function, doesn't necessarily need to be a function */
	init_presets(  );

	return ( retval );
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * for the presets are setup here.
 */
hwi_search_t acpi_preset_map[] = {
	{0, {0, {PAPI_NULL, PAPI_NULL}
		 , {0,}
		 }
	 }
};


int
init_presets(  )
{
	return ( _papi_hwi_setup_all_presets( acpi_preset_map, NULL ) );
}


/**
 * This is called whenever a thread is initialized
 */
int
ACPI_init( hwd_context_t * ctx )
{
	( void ) ctx;			 /*unused */
	init_presets(  );
	return ( PAPI_OK );
}

int
ACPI_shutdown( hwd_context_t * ctx )
{
	( void ) ctx;			 /*unused */
	return ( PAPI_OK );
}

/**
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int
ACPI_init_control_state( hwd_control_state_t * ptr )
{
	( void ) ptr;			 /*unused */
	return PAPI_OK;
}

int
ACPI_update_control_state( hwd_control_state_t * ptr, NativeInfo_t * native,
						   int count, hwd_context_t * ctx )
{
	( void ) ptr;			 /*unused */
	( void ) ctx;			 /*unused */
	int i, index;

	for ( i = 0; i < count; i++ ) {
		index =
			native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position = acpi_native_table[index].resources.selector - 1;
	}
	return ( PAPI_OK );
}

int
ACPI_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;			 /*unused */
	( void ) ctrl;			 /*unused */
	return ( PAPI_OK );
}

int
get_load_value(  )
{
	char txt[256];
	char *p;
	static int ct[2][4] = { {0, 0, 0, 0}, {0, 0, 0, 0} };
	static int n = 0;
	int d[4];
	int i, t, fd;
	float v;
	static FILE *f = NULL;

	if ( !f && !( f = fopen( "/proc/stat", "r" ) ) ) {
		printf( "Unable to open kernel statistics file." );
		goto fail;
	}

	if ( !( p = fgets( txt, sizeof ( txt ), f ) ) ) {
		printf( "Unable to read from kernel statistics file." );
		goto fail;
	}

	fd = dup( fileno( f ) );
	fclose( f );
	f = fdopen( fd, "r" );
	assert( f );
	fseek( f, 0, SEEK_SET );

	if ( strlen( p ) <= 5 ) {
		printf( "Parse failure" );
		goto fail;
	}

	sscanf( p + 5, "%u %u %u %u", &ct[n][0], &ct[n][1], &ct[n][2], &ct[n][3] );

	t = 0;

	for ( i = 0; i < 4; i++ )
		t += ( d[i] = abs( ct[n][i] - ct[1 - n][i] ) );

	v = ( t - d[3] ) / ( float ) t;

	n = 1 - n;

	return ( int ) ( v * 100 );

  fail:
	if ( f ) {
		fclose( f );
		f = NULL;
	}

	return -1;
}

FILE *
fopen_first( const char *pfx, const char *sfx, const char *m )
{
	assert( pfx );
	assert( sfx );
	assert( m );

	DIR *dir;
	struct dirent *de;
	char fn[PATH_MAX];

	if ( !( dir = opendir( pfx ) ) )
		return NULL;

	while ( ( de = readdir( dir ) ) ) {
		if ( de->d_name[0] != '.' ) {
			FILE *f;
			snprintf( fn, sizeof ( fn ), "%s/%s/%s", pfx, de->d_name, sfx );

			if ( ( f = fopen( fn, m ) ) ) {
				closedir( dir );
				return f;
			}

			break;
		}
	}

	closedir( dir );
	return NULL;
}

int
get_temperature_value(  )
{
	char txt[256];
	char *p;
	int v, fd;
	static FILE *f = NULL;
	static int old_acpi = 0;

	if ( !f ) {
		if ( !
			 ( f =
			   fopen_first( "/proc/acpi/thermal_zone", "temperature",
							"r" ) ) ) {
			if ( !( f = fopen_first( "/proc/acpi/thermal", "status", "r" ) ) ) {
				printf( "Unable to open ACPI temperature file." );
				goto fail;
			}

			old_acpi = 1;
		}
	}

	if ( !( p = fgets( txt, sizeof ( txt ), f ) ) ) {
		printf( "Unable to read data from ACPI temperature file." );
		goto fail;
	}

	fd = dup( fileno( f ) );
	fclose( f );
	f = fdopen( fd, "r" );
	assert( f );
	fseek( f, 0, SEEK_SET );

	if ( !old_acpi ) {
		if ( strlen( p ) > 20 )
			v = atoi( p + 20 );
		else
			v = 0;
	} else {
		if ( strlen( p ) > 15 )
			v = atoi( p + 15 );
		else
			v = 0;
		v = ( ( v - 2732 ) / 10 );	/* convert from deciKelvin to degrees Celcius */
	}

	if ( v > 100 )
		v = 100;
	if ( v < 0 )
		v = 0;

	return v;

  fail:
	if ( f ) {
		fclose( f );
		f = NULL;
	}

	return -1;
}

int
ACPI_read( hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **events,
		   int flags )
{
	( void ) ctx;			 /*unused */
	( void ) flags;			 /*unused */
	static int failed = 0;

	if ( failed ||
		 ( ( ( ACPI_control_state_t * ) ctrl )->counts[0] =
		   ( long long ) get_load_value(  ) ) < 0 ||
		 ( ( ( ACPI_control_state_t * ) ctrl )->counts[1] =
		   ( long long ) get_temperature_value(  ) ) < 0 )
		goto fail;

	*events = ( ( ACPI_control_state_t * ) ctrl )->counts;
	return 0;

  fail:
	failed = 1;
	return -1;
}

int
ACPI_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;			 /*unused */
	( void ) ctrl;			 /*unused */
	return ( PAPI_OK );
}

int
ACPI_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;			 /*unused */
	( void ) ctrl;			 /*unused */
	return ( PAPI_OK );
}

int
ACPI_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long long *from )
{
	( void ) ctx;			 /*unused */
	( void ) ctrl;			 /*unused */
	( void ) from;			 /*unused */
	return ( PAPI_OK );
}

/*
 * Functions for setting up various options
 */

/** This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
ACPI_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;			 /*unused */
	( void ) code;			 /*unused */
	( void ) option;		 /*unused */
	return ( PAPI_OK );
}

/**
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
int
ACPI_set_domain( hwd_control_state_t * cntrl, int domain )
{
	( void ) cntrl;			 /*unused */
	int found = 0;
	if ( PAPI_DOM_USER & domain ) {
		found = 1;
	}
	if ( PAPI_DOM_KERNEL & domain ) {
		found = 1;
	}
	if ( PAPI_DOM_OTHER & domain ) {
		found = 1;
	}
	if ( !found )
		return ( PAPI_EINVAL );
	return ( PAPI_OK );
}

/* 
 * Timing Routines
 * These functions should return the highest resolution timers available.
 */
/*long long _papi_hwd_get_real_usec(void)
{
	return(1);
}

long long _papi_hwd_get_real_cycles(void)
{
	return(1);
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
	return(1);
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
	return(1);
}
*/
/**
 * Native Event functions
 */
int
ACPI_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int cidx = PAPI_COMPONENT_INDEX( *EventCode );

	if ( modifier == PAPI_ENUM_FIRST ) {
		/* assumes first native event is always 0x4000000 */
		*EventCode = PAPI_NATIVE_MASK | PAPI_COMPONENT_MASK( cidx );
		return ( PAPI_OK );
	}

	if ( modifier == PAPI_ENUM_EVENTS ) {
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if ( acpi_native_table[index + 1].resources.selector ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );
	} else
		return ( PAPI_EINVAL );
}

int
ACPI_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	strncpy( name,
			 acpi_native_table[EventCode & PAPI_NATIVE_AND_MASK &
							   PAPI_COMPONENT_AND_MASK].name, len );
	return ( PAPI_OK );
}

int
ACPI_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	strncpy( name,
			 acpi_native_table[EventCode & PAPI_NATIVE_AND_MASK &
							   PAPI_COMPONENT_AND_MASK].description, len );
	return ( PAPI_OK );
}

int
ACPI_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	memcpy( ( ACPI_register_t * ) bits, &( acpi_native_table[EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK].resources ), sizeof ( ACPI_register_t ) );	/* it is not right, different type */
	return ( PAPI_OK );
}

int
ACPI_ntv_bits_to_info( hwd_register_t * bits, char *names, unsigned int *values,
					   int name_len, int count )
{
	( void ) bits;			 /*unused */
	( void ) names;			 /*unused */
	( void ) values;		 /*unused */
	( void ) name_len;		 /*unused */
	( void ) count;			 /*unused */
	return ( 1 );
}


/*
 * Shared Library Information and other Information Functions
 */
/*int _papi_hwd_update_shlib_info(void){
	return(PAPI_OK);
}
*/

papi_vector_t _acpi_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name =
				 "$Id$",
				 .version = "$Revision$",
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .num_cntrs = ACPI_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_USER,
				 .default_granularity = PAPI_GRN_THR,
				 .available_granularities = PAPI_GRN_THR,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 0,
				 .fast_virtual_timer = 0,
				 .attach = 0,
				 .attach_must_ptrace = 0,
				 .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( ACPI_context_t ),
			 .control_state = sizeof ( ACPI_control_state_t ),
			 .reg_value = sizeof ( ACPI_register_t ),
			 .reg_alloc = sizeof ( ACPI_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init = ACPI_init,
	.init_substrate = ACPI_init_substrate,
	.init_control_state = ACPI_init_control_state,
	.start = ACPI_start,
	.stop = ACPI_stop,
	.read = ACPI_read,
	.shutdown = ACPI_shutdown,
	.ctl = ACPI_ctl,
	.update_control_state = ACPI_update_control_state,
	.set_domain = ACPI_set_domain,
	.reset = ACPI_reset,
/*	.set_overflow =		_p3_set_overflow,
	.stop_profiling =		_p3_stop_profiling,*/
	.ntv_enum_events = ACPI_ntv_enum_events,
	.ntv_code_to_name = ACPI_ntv_code_to_name,
	.ntv_code_to_descr = ACPI_ntv_code_to_descr,
	.ntv_code_to_bits = ACPI_ntv_code_to_bits,
	.ntv_bits_to_info = ACPI_ntv_bits_to_info
};
