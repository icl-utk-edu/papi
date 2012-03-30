/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
#include "linux-lmsensors.h"

papi_vector_t _lmsensors_vector;

/******************************************************************************
 ********  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT ********
 *****************************************************************************/
/*
 * Counts number of events available in this system
 */
static unsigned
detectSensors( void )
{
	unsigned id = 0;
	int chip_nr = 0;
	const sensors_chip_name *chip_name;

	/* Loop through all the chips, features, subfeatures found */
	while ( ( chip_name =
			  sensors_get_detected_chips( NULL, &chip_nr ) ) != NULL ) {
		int a = 0, b;
		const sensors_feature *feature;

		while ( ( feature = sensors_get_features( chip_name, &a ) ) ) {
			b = 0;
			while ( ( sensors_get_all_subfeatures( chip_name, feature,
												   &b ) ) ) {
				id++;
			}
		}
	}

	return id;
}


/*
 * Create the native events for particulare component (!= 0)
 */
static unsigned
createNativeEvents( void )
{
	unsigned id = 0;
	unsigned int count = 0;

	int chip_nr = 0;
	const sensors_chip_name *chip_name;

	/* component name and description */
	strcpy( _lmsensors_vector.cmp_info.short_name, "LM_SENSORS" );
	strcpy( _lmsensors_vector.cmp_info.description,
			"lm-sensors provides tools for monitoring the hardware health" );


	/* Loop through all the chips found */
	while ( ( chip_name =
			  sensors_get_detected_chips( NULL, &chip_nr ) ) != NULL ) {
	   int a, b;
	   const sensors_feature *feature;
	   const sensors_subfeature *sub;
	   char chipnamestring[PAPI_MIN_STR_LEN];

	   //	   lm_sensors_native_table[id].count = 0;

		/* get chip name from its internal representation */
	   sensors_snprintf_chip_name( chipnamestring,
					    PAPI_MIN_STR_LEN, chip_name );

	   a = 0;

	   /* Loop through all the features found */
	   while ( ( feature = sensors_get_features( chip_name, &a ) ) ) {
	      char *featurelabel;

	      if ( !( featurelabel = sensors_get_label( chip_name, feature ))) {
		 fprintf( stderr, "ERROR: Can't get label of feature %s!\n",
						 feature->name );
		 continue;
	      }

	      b = 0;

	      /* Loop through all the subfeatures found */
	      while ((sub=sensors_get_all_subfeatures(chip_name,feature,&b))) {

	         count = 0;

		 /* Save native event data */
		 sprintf( lm_sensors_native_table[id].name, "%s.%s.%s.%s",
			  _lmsensors_vector.cmp_info.short_name,
			  chipnamestring, featurelabel, sub->name );

		 strncpy( lm_sensors_native_table[id].description,
			  lm_sensors_native_table[id].name, PAPI_MAX_STR_LEN );

		 /* The selector has to be !=0 . Starts with 1 */
		 lm_sensors_native_table[id].resources.selector = id + 1;

		 /* Save the actual references to this event */
		 lm_sensors_native_table[id].resources.name = chip_name;
		 lm_sensors_native_table[id].resources.subfeat_nr = sub->number;

		 count = sub->number;

		 /* increment the table index counter */
		 id++;		 
	      }

	      //   lm_sensors_native_table[id].count = count + 1;
	      free( featurelabel );
	   }
	}

	/* Return the number of events created */
	return id;
}

/*
 * Returns the value of the event with index 'i' in lm_sensors_native_table
 * This value is scaled by 1000 to cope with the lack to return decimal numbers
 * with PAPI
 */

static long_long
getEventValue( unsigned event_id )
{
	double value;
	int res;

	res = sensors_get_value( lm_sensors_native_table[event_id].resources.name,
							 lm_sensors_native_table[event_id].resources.
							 subfeat_nr, &value );

	if ( res < 0 ) {
		fprintf( stderr, "libsensors(): Could not read event #%d!\n",
				 event_id );
		return -1;
	}

	return ( ( long_long ) ( value * 1000 ) );
}


/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * This is called whenever a thread is initialized
 */
int
LM_SENSORS_init( hwd_context_t * ctx )
{
    ( void ) ctx;
	return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
LM_SENSORS_init_substrate(  )
{
	int res;

	/* Initialize libsensors library */
	if ( ( res = sensors_init( NULL ) ) != 0 ) {
		return res;
	}

	/* Create dyanmic events table */
	NUM_EVENTS = detectSensors(  );
	//printf("Found %d sensors\n",NUM_EVENTS);

	if ( ( lm_sensors_native_table =
		   ( LM_SENSORS_native_event_entry_t * )
		   malloc( sizeof ( LM_SENSORS_native_event_entry_t ) *
				   NUM_EVENTS ) ) == NULL ) {
		perror( "malloc():Could not get memory for events table" );
		return EXIT_FAILURE;
	}

	if ( ( unsigned ) NUM_EVENTS != createNativeEvents(  ) ) {
		fprintf( stderr, "Number of LM_SENSORS events mismatch!\n" );
		return EXIT_FAILURE;
	}

	return PAPI_OK;
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int
LM_SENSORS_init_control_state( hwd_control_state_t * ctrl )
{
	int i;

	for ( i = 0; i < NUM_EVENTS; i++ )
		( ( LM_SENSORS_control_state_t * ) ctrl )->counts[i] =
			getEventValue( i );

	( ( LM_SENSORS_control_state_t * ) ctrl )->lastupdate =
		PAPI_get_real_usec(  );
	return PAPI_OK;
}


/*
 *
 */
int
LM_SENSORS_start( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
	( void ) ctx;
	( void ) ctrl;

	return PAPI_OK;
}


/*
 *
 */
int
LM_SENSORS_stop( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
    ( void ) ctx;
    ( void ) ctrl;

    return PAPI_OK;
}


/*
 *
 */
int
LM_SENSORS_read( hwd_context_t * ctx, hwd_control_state_t * ctrl,
				 long_long ** events, int flags )
{
    ( void ) ctx;
	( void ) flags;
	long long start = PAPI_get_real_usec(  );

	if ( start - ( ( LM_SENSORS_control_state_t * ) ctrl )->lastupdate > 200000 ) {	// cache refresh
		int i;

		for ( i = 0; i < NUM_EVENTS; i++ )
			( ( LM_SENSORS_control_state_t * ) ctrl )->counts[i] =
				getEventValue( i );

		( ( LM_SENSORS_control_state_t * ) ctrl )->lastupdate =
			PAPI_get_real_usec(  );
	}

	*events = ( ( LM_SENSORS_control_state_t * ) ctrl )->counts;	// return cached data
	return ( PAPI_OK );
}

/*
 *
 */
int
LM_SENSORS_shutdown( hwd_context_t * ctx )
{
    ( void ) ctx;
	/* Call the libsensors cleaning function before leaving */
	sensors_cleanup(  );

	return ( PAPI_OK );
}



/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
LM_SENSORS_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
    ( void ) ctx;
	( void ) code;
	( void ) option;
	return ( PAPI_OK );
}

//int             LM_SENSORS_ntv_code_to_bits ( unsigned int EventCode, hwd_register_t * bits );



/*
 *
 */
int
LM_SENSORS_update_control_state( hwd_control_state_t * ptr,
								 NativeInfo_t * native, int count,
								 hwd_context_t * ctx )
{
	int i, index;
    ( void ) ctx;
	( void ) ptr;

	for ( i = 0; i < count; i++ ) {
		index =
			native[i].ni_event & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;
		native[i].ni_position =
			lm_sensors_native_table[index].resources.selector - 1;
	}
	return ( PAPI_OK );
}


/*
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
LM_SENSORS_set_domain( hwd_control_state_t * cntrl, int domain )
{
	int found = 0;
    ( void ) cntrl;
	
	if ( PAPI_DOM_USER & domain )
		found = 1;

	if ( PAPI_DOM_KERNEL & domain )
		found = 1;

	if ( PAPI_DOM_OTHER & domain )
		found = 1;

	if ( !found )
		return ( PAPI_EINVAL );

	return ( PAPI_OK );
}


/*
 *
 */
int
LM_SENSORS_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{
    ( void ) ctx;
	( void ) ctrl;
	return ( PAPI_OK );
}


/*
 * Native Event functions
 */
int
LM_SENSORS_ntv_enum_events( unsigned int *EventCode, int modifier )
{

	switch ( modifier ) {
	case PAPI_ENUM_FIRST:
		*EventCode = PAPI_NATIVE_MASK;

		return PAPI_OK;
		break;

	case PAPI_ENUM_EVENTS:
	{
		int index = *EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

		if ( index < NUM_EVENTS - 1 ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else
			return ( PAPI_ENOEVNT );

		break;
	}
	default:
		return ( PAPI_EINVAL );
	}
	return ( PAPI_EINVAL );
}

/*
 *
 */
int
LM_SENSORS_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	strncpy( name, lm_sensors_native_table[index].name, len );
	return ( PAPI_OK );
}

/*
 *
 */
int
LM_SENSORS_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	strncpy( name, lm_sensors_native_table[index].description, len );
	return ( PAPI_OK );
}

/*
 *
 */
int
LM_SENSORS_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	int index = EventCode & PAPI_NATIVE_AND_MASK & PAPI_COMPONENT_AND_MASK;

	memcpy( ( LM_SENSORS_register_t * ) bits,
			&( lm_sensors_native_table[index].resources ),
			sizeof ( LM_SENSORS_register_t ) );

	return ( PAPI_OK );
}



/*
 *
 */
papi_vector_t _lmsensors_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name = "linux-lmsensors.c",
				 .version = "4.2.1",
				 .num_mpx_cntrs = PAPI_MPX_DEF_DEG,
				 .num_cntrs = LM_SENSORS_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_USER,
				 //.available_domains = PAPI_DOM_USER,
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
			 .context = sizeof ( LM_SENSORS_context_t ),
			 .control_state = sizeof ( LM_SENSORS_control_state_t ),
			 .reg_value = sizeof ( LM_SENSORS_register_t ),
			 .reg_alloc = sizeof ( LM_SENSORS_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init = LM_SENSORS_init,
	.init_substrate = LM_SENSORS_init_substrate,
	.init_control_state = LM_SENSORS_init_control_state,
	.start = LM_SENSORS_start,
	.stop = LM_SENSORS_stop,
	.read = LM_SENSORS_read,
	.shutdown = LM_SENSORS_shutdown,
	.ctl = LM_SENSORS_ctl,

	.update_control_state = LM_SENSORS_update_control_state,
	.set_domain = LM_SENSORS_set_domain,
	.reset = LM_SENSORS_reset,

	.ntv_enum_events = LM_SENSORS_ntv_enum_events,
	.ntv_code_to_name = LM_SENSORS_ntv_code_to_name,
	.ntv_code_to_bits = LM_SENSORS_ntv_code_to_bits,
};
