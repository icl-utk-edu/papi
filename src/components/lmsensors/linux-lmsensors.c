/** 
 * @file    linux-lmsensors.c
 * @author  Daniel Lucio
 * @author  Joachim Protze
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * @mods    Rizwan Ashraf
 *          rizwan@icl.utk.edu
 *
 * @ingroup papi_components
 *
 *
 * LM_SENSORS component 
 * 
 * Tested version of lm_sensors: 3.4.0
 *
 * @brief 
 *  This file has the source code for a component that enables PAPI-C to access
 *  hardware monitoring sensors through the libsensors library. This code will
 *  dynamically create a native events table for all the sensors that can be 
 *  accesed by the libsensors library.
 *  In order to learn more about libsensors, visit: (http://www.lm-sensors.org) 
 *
 * Notes: 
 *  - I used the ACPI and MX components to write this component. A lot of the
 *    code in this file mimics what other components already do. 
 *  - The return values are scaled by 1000 because PAPI can not return decimals.
 *  - A call of PAPI_read can take up to 2 seconds while using lm_sensors!
 *  - Please remember that libsensors uses the GPL license. 
 */


/* Headers required by libsensors */
#include <sensors.h>
#include <error.h>
#include <time.h>
#include <string.h>
#include <dlfcn.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/
// time in usecs
#define LM_SENSORS_REFRESHTIME 200000

/** Structure that stores private information of each event */
typedef struct _lmsensors_register
{
	/* This is used by the framework.It likes it to be !=0 to do somehting */
	unsigned int selector;
	/* These are the only information needed to locate a libsensors event */
	const sensors_chip_name *name;
	int subfeat_nr;
} _lmsensors_register_t;

/*
 * The following structures mimic the ones used by other components. It is more
 * convenient to use them like that as programming with PAPI makes specific
 * assumptions for them.
 */

/** This structure is used to build the table of events */
typedef struct _lmsensors_native_event_entry
{
	_lmsensors_register_t resources;
	char name[PAPI_MAX_STR_LEN];
	char description[PAPI_MAX_STR_LEN];
	unsigned int count;
} _lmsensors_native_event_entry_t;


typedef struct _lmsensors_reg_alloc
{
	_lmsensors_register_t ra_bits;
} _lmsensors_reg_alloc_t;


typedef struct _lmsensors_control_state
{
	long_long lastupdate;
} _lmsensors_control_state_t;


typedef struct _lmsensors_context
{
	_lmsensors_control_state_t state;
} _lmsensors_context_t;



/*************************  GLOBALS SECTION  ***********************************
 *******************************************************************************/
/* This table contains the LM_SENSORS native events */
static _lmsensors_native_event_entry_t *lm_sensors_native_table;
/* number of events in the table*/
static int num_events = 0;
static long_long *cached_counts = NULL;	// used for caching readings


static int (*sensors_initPtr)(FILE *input);
static void (*sensors_cleanupPtr)(void);
static int (*sensors_snprintf_chip_namePtr)(char *str, size_t size,
				  const sensors_chip_name *chip);
static char *(*sensors_get_labelPtr)(const sensors_chip_name *name, const sensors_feature *feature);
static int (*sensors_get_valuePtr)(const sensors_chip_name *name, int subfeat_nr,
		      double *value);
static const sensors_chip_name *(*sensors_get_detected_chipsPtr)(const sensors_chip_name
						       *match, int *nr);
static const sensors_feature *(*sensors_get_featuresPtr)(const sensors_chip_name *name, int *nr);
static const sensors_subfeature *(*sensors_get_all_subfeaturesPtr)(const sensors_chip_name *name,
			    const sensors_feature *feature, int *nr);
static const char *(*sensors_get_adapter_namePtr)(const sensors_bus_id *bus);  

// file handles used to access lmsensors libraries with dlopen
static void* dl1 = NULL;

// string macro defined within Rules.lmsensors
static char lmsensors_main[]=PAPI_LMSENSORS_MAIN;

//-----------------------------------------------------------------------------
// Using weak symbols (global declared without a value, so it defers to any
// other global declared in another file WITH a value) allows PAPI to be built
// with the component, but PAPI can still be installed in a system without the
// required library.
//-----------------------------------------------------------------------------

void (*_dl_non_dynamic_init)(void) __attribute__((weak));               // declare a weak dynamic-library init routine pointer.

static int link_lmsensors_libraries ();

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
			  sensors_get_detected_chipsPtr( NULL, &chip_nr ) ) != NULL ) {
		int a = 0, b;
		const sensors_feature *feature;

		while ( ( feature = sensors_get_featuresPtr( chip_name, &a ) ) ) {
			b = 0;
			while ( ( sensors_get_all_subfeaturesPtr( chip_name, feature,
												   &b ) ) ) {
				id++;
			}
		}
	}

	return id;
}

static char*
makeEventDescription(const char* featureLabel, const sensors_feature* feature, const sensors_subfeature* subFeature, const sensors_bus_id* bus)
{
 	char *desc = 0;
	char *sensor = 0;
	char *units = 0;

	desc = (char*) papi_calloc (PAPI_MAX_STR_LEN, 1);
	sensor = (char*) papi_calloc (PAPI_MAX_STR_LEN, 1);
	units = (char*) papi_calloc (PAPI_MIN_STR_LEN, 1);
	if (desc == 0 || sensor == 0 || units == 0) {
		PAPIERROR("cannot allocate memory for event description");
		return (0);
	}

	switch (feature->type) {
		case SENSORS_FEATURE_IN:
			switch (subFeature->type) {
				case SENSORS_SUBFEATURE_IN_INPUT:
					sensor = "Measured voltage";
					break;
				case SENSORS_SUBFEATURE_IN_MIN:
					sensor = "Minimum voltage limit";
					break;
				case SENSORS_SUBFEATURE_IN_MAX:
					sensor = "Maximum voltage limit";
					break;	
				case SENSORS_SUBFEATURE_IN_LCRIT:
					sensor = "Low critical voltage";
					break; 
				case SENSORS_SUBFEATURE_IN_CRIT:
					sensor = "High critical voltage";
					break;
				case SENSORS_SUBFEATURE_IN_AVERAGE:
					sensor = "Average measured voltage";
					break;
				case SENSORS_SUBFEATURE_IN_LOWEST:
					sensor = "Lowest measured voltage";
					break;
				case SENSORS_SUBFEATURE_IN_HIGHEST:
                                        sensor = "Highest measured voltage";
					break;
				case SENSORS_SUBFEATURE_IN_ALARM:
					sensor = "Voltage value which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_IN_MIN_ALARM:
					sensor = "Minimum voltage value which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_IN_MAX_ALARM:
					sensor = "Maximum voltage value which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_IN_BEEP:
					sensor = "Voltage value which will result in beeping";
					break;
				case SENSORS_SUBFEATURE_IN_LCRIT_ALARM:
					sensor = "Low critical voltage which will result in an alarm";
					break; 
				case SENSORS_SUBFEATURE_IN_CRIT_ALARM:
					sensor = "High critical voltage which will result in an alarm";
					break;
				default:
					sensor = "Voltage value with unknown attribute";
			}
			units = "Volts";
			break;
		case SENSORS_FEATURE_FAN:
			switch (subFeature->type) {
				case SENSORS_SUBFEATURE_FAN_INPUT:
					sensor = "Measured fan speed";
					break;
				case SENSORS_SUBFEATURE_FAN_MIN:
					sensor = "Minimum fan speed";
					break;
				case SENSORS_SUBFEATURE_FAN_MAX:
					sensor = "Maximum fan speed";
					break;
				case SENSORS_SUBFEATURE_FAN_ALARM:
					sensor = "Fan speed which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_FAN_FAULT:
					sensor = "Fan speed which will result in a fault";
					break;
				case SENSORS_SUBFEATURE_FAN_DIV:
					sensor = "Fan speed div";
					break;
				case SENSORS_SUBFEATURE_FAN_BEEP:
					sensor = "Fan speed which will result in beeping";
					break;
				case SENSORS_SUBFEATURE_FAN_PULSES:
					sensor = "Fan speed pulses";
					break;
				case SENSORS_SUBFEATURE_FAN_MIN_ALARM:
					sensor = "Minimum fan speed which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_FAN_MAX_ALARM:
					sensor = "Maximum fan speed which will result in an alarm";
					break;
				default:
					sensor = "Fan speed with unknown attribute";
			}
			units = "RPM";
			break;
		case SENSORS_FEATURE_TEMP:
			switch (subFeature->type) {
				case SENSORS_SUBFEATURE_TEMP_INPUT:
					sensor = "Measured temperature";
					break;
				case SENSORS_SUBFEATURE_TEMP_MAX:
					sensor = "Maximum temperature limit";
					break;
				case SENSORS_SUBFEATURE_TEMP_MAX_HYST:
					sensor = "Maximum temperature hysteresis limit. This is usually slightly lower than maximum limit and is used to safely clear maximum temperature condition";
					break;
				case SENSORS_SUBFEATURE_TEMP_MIN:
					sensor = "Minimum temperature limit";
					break;
				case SENSORS_SUBFEATURE_TEMP_CRIT:
					sensor = "Critical temperature limit";
					break;
				case SENSORS_SUBFEATURE_TEMP_CRIT_HYST:
					sensor = "Critical temperature hysteresis limit. This is usually slightly lower than critical limit and is used to safely clear critical temperature condition";
					break;
				case SENSORS_SUBFEATURE_TEMP_LCRIT:
					sensor = "Critical low temperature limit";
					break;
				case SENSORS_SUBFEATURE_TEMP_EMERGENCY:
					sensor = "Emergency temperature limit";
					break;
				case SENSORS_SUBFEATURE_TEMP_EMERGENCY_HYST:
					sensor = "Emergency temperature hysteresis limit. This is usually slightly lower than emergency limit and is used to safely clear emergency temperature condition";
					break;
				case SENSORS_SUBFEATURE_TEMP_LOWEST:
					sensor = "Lowest measured temperature";
					break;
				case SENSORS_SUBFEATURE_TEMP_HIGHEST:
					sensor = "Highest measured temperature";
					break;
				case SENSORS_SUBFEATURE_TEMP_MIN_HYST:
					sensor = "Minimum temperature hysteresis limit";
					break;
				case SENSORS_SUBFEATURE_TEMP_LCRIT_HYST:
					sensor = "Critical low temperature hysteresis limit";
					break;
				case SENSORS_SUBFEATURE_TEMP_TYPE:
					sensor = "Temperature sensor used for measurement. Possibilities: 1) PII/Celeron Diode, 2) 3904 transistor, 3) thermal diode, 4) thermistor, 5) AMD AMDSI, 6) Intel PECI"; 
					break;
				case SENSORS_SUBFEATURE_TEMP_OFFSET:
					sensor = "Temperature offset";
					break;
				case SENSORS_SUBFEATURE_TEMP_BEEP:
					sensor = "Temperature which will result in beeping";
					break;
				case SENSORS_SUBFEATURE_TEMP_EMERGENCY_ALARM:
					sensor = "Emergency temperature limit which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_TEMP_LCRIT_ALARM:
					sensor = "Critical low temperature limit which will result in an alarm";
					break;
				default:
					sensor = "Temperature with unknonwn attribute";
			}
			units = "Celsius";
			break;
		case SENSORS_FEATURE_POWER:
			switch (subFeature->type) {
				case SENSORS_SUBFEATURE_POWER_AVERAGE:
					sensor = "Average power over a time interval";
					break;
				case SENSORS_SUBFEATURE_POWER_AVERAGE_HIGHEST:
					sensor = "Highest average power";
					break;
				case SENSORS_SUBFEATURE_POWER_AVERAGE_LOWEST:
					sensor = "Lowest average power";
					break;
				case SENSORS_SUBFEATURE_POWER_INPUT:
					sensor = "Measured absolute power";
					break;
				case SENSORS_SUBFEATURE_POWER_INPUT_HIGHEST:
					sensor = "Highest measured power";
					break;
				case SENSORS_SUBFEATURE_POWER_INPUT_LOWEST:
					sensor = "Lowest measured power";
					break;
				case SENSORS_SUBFEATURE_POWER_CAP:
					sensor = "Power cap limit";
					break;
				case SENSORS_SUBFEATURE_POWER_CAP_HYST:
					sensor = "Power cap hysteresis limit. This is usually slightly lower than the power cap limit and is used to safely clear the violation of power capping limit";
					break;
				case SENSORS_SUBFEATURE_POWER_MAX:
					sensor = "Maximum power limit";
					break;
				case SENSORS_SUBFEATURE_POWER_CRIT:
					sensor = "Critical power limit";
					break;
				case SENSORS_SUBFEATURE_POWER_AVERAGE_INTERVAL:
					sensor = "Time interval over which average power is calculated/reported";
					break;
				case SENSORS_SUBFEATURE_POWER_ALARM:
					sensor = "Power limit which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_POWER_CAP_ALARM:
					sensor = "Power cap limit which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_POWER_MAX_ALARM:
					sensor = "Maximum power limit which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_POWER_CRIT_ALARM:
					sensor = "Critical power limit which will result in an alarm";
					break;
				default:
					sensor = "Power with unknown attribute";
			}
			units = "Watts";
			break;
		case SENSORS_FEATURE_ENERGY:
			sensor = "Energy consumed";
			units = "Joules";
			break;
		case SENSORS_FEATURE_CURR:
			switch (subFeature->type) {
				case SENSORS_SUBFEATURE_CURR_INPUT:
					sensor = "Measured current";
					break;
				case SENSORS_SUBFEATURE_CURR_MIN:
					sensor = "Minimum current limit";
					break;
				case SENSORS_SUBFEATURE_CURR_MAX:
					sensor = "Maximum current limit";
					break;
				case SENSORS_SUBFEATURE_CURR_LCRIT:
					sensor = "Critical low current limit";
					break;
				case SENSORS_SUBFEATURE_CURR_CRIT:
					sensor = "Critical current limit";
					break;
				case SENSORS_SUBFEATURE_CURR_AVERAGE:
					sensor = "Average current";
					break;
				case SENSORS_SUBFEATURE_CURR_LOWEST:
					sensor = "Lowest measured current";
					break;
				case SENSORS_SUBFEATURE_CURR_HIGHEST:
					sensor = "Highest measured current";
					break;
				case SENSORS_SUBFEATURE_CURR_ALARM:
					sensor = "Current reading which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_CURR_MIN_ALARM:
					sensor = "Minimum current limit which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_CURR_MAX_ALARM:
					sensor = "Maximum current limit which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_CURR_BEEP:
					sensor = "Current reading which will result in beeping";
					break;
				case SENSORS_SUBFEATURE_CURR_LCRIT_ALARM:
					sensor = "Critical low current limit which will result in an alarm";
					break;
				case SENSORS_SUBFEATURE_CURR_CRIT_ALARM:
					sensor = "Critical current limit which will result in an alarm";
					break;
				default:
					sensor = "Current with unknown attribute";
			}
			units = "Amperes";
			break;
		case SENSORS_FEATURE_HUMIDITY:
			sensor = "Humidity reading";
			units = "Percentage";
			break;
		case SENSORS_FEATURE_MAX_MAIN: // The description of this feature is unclear.
			sensor = "Feature: MAX_MAIN";
			units = "N/A";
			break;
		case SENSORS_FEATURE_VID: // voltage identification: value set in BIOS to set core voltage of processor
			sensor = "Voltage identification";
			units = "Volts";
			break;
		case SENSORS_FEATURE_INTRUSION: // chassis intrusion detection 
			switch (subFeature->type) {
				case SENSORS_SUBFEATURE_INTRUSION_ALARM:
					sensor = "Chassis intrusion detection alarm. Flag which indicates whether an alarm will result in case of intrusion";
					break;
				case SENSORS_SUBFEATURE_INTRUSION_BEEP:
					sensor = "Chassis intrusion detection beep. Flag which indicates whether beeping will occur in case of intrusion";
					break;
				default:
					sensor = "Chassis intrusion flag";
			} 
			units = "N/A";
			break;
		case SENSORS_FEATURE_MAX_OTHER: // The description of this feature is unclear.
			sensor = "Feature: MAX_OTHER";
			units = "N/A";
			break;
		case SENSORS_FEATURE_BEEP_ENABLE: // 
			sensor = "Flag which indicates whether beeping is enabled or not";
			units = "N/A";
			break;
		case SENSORS_FEATURE_MAX: // The description of this feature is unclear.
			sensor = "Feature: MAX";
			units = "N/A";
			break;
		default:
			sensor = "Feature: UNKNWON";
			units = "N/A";
	}

	/* compose the description with all available info */	
	if (subFeature->flags == SENSORS_MODE_R) { // readable sensor only
		snprintf (desc, PAPI_MAX_STR_LEN, "%s (read-only) for %s. Units: %s. Adapter: %s.",
		  	sensor, featureLabel, units, sensors_get_adapter_namePtr(bus));
	} else if ((subFeature->flags == SENSORS_MODE_W) || 
		   (subFeature->flags == (SENSORS_MODE_W + SENSORS_MODE_R))) { // writable value
		snprintf (desc, PAPI_MAX_STR_LEN, "%s (writeable) for %s. Units: %s. Adapter: %s.",
                        sensor, featureLabel, units, sensors_get_adapter_namePtr(bus));
	} else if (subFeature->flags == (SENSORS_MODE_R + SENSORS_COMPUTE_MAPPING)) {  // affected by the computation rules of the main feature
 		snprintf (desc, PAPI_MAX_STR_LEN, "%s (affected by computation rules) for %s. Units: %s. Adapter: %s.",
                        sensor, featureLabel, units, sensors_get_adapter_namePtr(bus));
	} else {
		snprintf (desc, PAPI_MAX_STR_LEN, "%s for %s. Units: %s. Adapter: %s.",
                        sensor, featureLabel, units, sensors_get_adapter_namePtr(bus));
	}

	return (desc);
}

/*
 * Create the native events for particulare component (!= 0)
 */
static unsigned
createNativeEvents( void )
{
	unsigned id = 0;
	unsigned int count;
	(void) count; // Ignore not used warnings.

	int chip_nr = 0;
	const sensors_chip_name *chip_name;

	/* component name and description */
	strcpy( _lmsensors_vector.cmp_info.short_name, "lm_sensors" );
	strcpy( _lmsensors_vector.cmp_info.description,
			"lm-sensors provides tools for monitoring the hardware health" );


	/* Loop through all the chips found */
	while ( ( chip_name =
			  sensors_get_detected_chipsPtr( NULL, &chip_nr ) ) != NULL ) {
	   int a, b;
	   const sensors_feature *feature;
	   const sensors_subfeature *sub;
	   char chipnamestring[PAPI_MIN_STR_LEN];

	   //	   lm_sensors_native_table[id].count = 0;

		/* get chip name from its internal representation */
	   sensors_snprintf_chip_namePtr( chipnamestring,
					    PAPI_MIN_STR_LEN, chip_name );

	   a = 0;

	   /* Loop through all the features found */
	   while ( ( feature = sensors_get_featuresPtr( chip_name, &a ) ) ) {
	      char *featurelabel;

	      if ( !( featurelabel = sensors_get_labelPtr( chip_name, feature ))) {
		 fprintf( stderr, "ERROR: Can't get label of feature %s!\n",
						 feature->name );
		 continue;
	      }

	      b = 0;

	      /* Loop through all the subfeatures found */
	      while ((sub=sensors_get_all_subfeaturesPtr(chip_name,feature,&b))) {

	         count = 0;

		 /* Save native event data */
		 sprintf( lm_sensors_native_table[id].name, "%s.%s.%s",
			  chipnamestring, featurelabel, sub->name );

		 strncpy( lm_sensors_native_table[id].description,
			  makeEventDescription(featurelabel, feature, sub, &chip_name->bus), 
			  PAPI_MAX_STR_LEN );
                 lm_sensors_native_table[id].description[PAPI_MAX_STR_LEN-1] = '\0';

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

	res = sensors_get_valuePtr( lm_sensors_native_table[event_id].resources.name,
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
static int
_lmsensors_init_thread( hwd_context_t *ctx )
{
    ( void ) ctx;
    return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int
_lmsensors_init_component( int cidx )
{
    int res;
    (void) cidx;

    /* link in all the lmsensor libraries and resolve the symbols we need to use */
    if (link_lmsensors_libraries() != PAPI_OK) {
	    SUBDBG ("Dynamic link of lmsensors libraries failed, component will be disabled.\n");
	    SUBDBG ("See disable reason in papi_component_avail output for more details.\n");
	    return (PAPI_ENOSUPP);
    }

    /* Initialize libsensors library */
    if ( ( res = sensors_initPtr( NULL ) ) != 0 ) {
       strncpy(_lmsensors_vector.cmp_info.disabled_reason,
	      "Cannot enable libsensors",PAPI_MAX_STR_LEN);
       return res;
    }

    /* Create dyanmic events table */
    num_events = detectSensors(  );
    SUBDBG("Found %d sensors\n",num_events);

    _lmsensors_vector.cmp_info.num_mpx_cntrs = num_events;
    _lmsensors_vector.cmp_info.num_cntrs = num_events;

    if ( ( lm_sensors_native_table =
	   calloc( num_events, sizeof ( _lmsensors_native_event_entry_t )))
				   == NULL ) {
       strncpy(_lmsensors_vector.cmp_info.disabled_reason,
	      "Could not malloc room",PAPI_MAX_STR_LEN);
       return PAPI_ENOMEM;
    }

    cached_counts = (long long*) calloc(num_events, sizeof(long long));

    if (cached_counts == NULL) {
        strncpy(_lmsensors_vector.cmp_info.disabled_reason,
               "Could not malloc room",PAPI_MAX_STR_LEN);
      free(lm_sensors_native_table);
      lm_sensors_native_table = NULL;
	   return PAPI_ENOMEM;
    }

    if ( ( unsigned ) num_events != createNativeEvents(  ) ) {
       strncpy(_lmsensors_vector.cmp_info.disabled_reason,
	      "LM_SENSOR number mismatch",PAPI_MAX_STR_LEN);
      free(cached_counts);
      cached_counts = NULL;
      free(lm_sensors_native_table);
      lm_sensors_native_table = NULL;
       return PAPI_ECMP;
    }

    _lmsensors_vector.cmp_info.num_native_events=num_events;
    _lmsensors_vector.cmp_info.num_cntrs=num_events;

    return PAPI_OK;
}

/*
 * Link the necessary lmsensors libraries to use the lmsensors
 * component.  If any of them can not be found, then the lmsensors
 * component will just be disabled.  This is done at runtime so that a
 * version of PAPI built with the Infiniband component can be
 * installed and used on systems which have the lmsensors libraries
 * installed and on systems where these libraries are not installed.
 */
static int
link_lmsensors_libraries ()
{
   if ( _dl_non_dynamic_init != NULL ) {
      // If weak var present, statically linked insted of dynamic.
      strncpy( _lmsensors_vector.cmp_info.disabled_reason, "The lmsensors component REQUIRES dynamic linking capabilities.", PAPI_MAX_STR_LEN-1);
      // EXIT not supported.
      return PAPI_ENOSUPP;
   }

   char path_name[1024];
   char *lmsensors_root = getenv("PAPI_LMSENSORS_ROOT"); 
   
   dl1 = NULL;
   // Step 1: Process override if given.   
   if (strlen(lmsensors_main) > 0) {                                  // If override given, it has to work.
      dl1 = dlopen(lmsensors_main, RTLD_NOW | RTLD_GLOBAL);           // Try to open that path.
      if (dl1 == NULL) {
         snprintf(_lmsensors_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "PAPI_LMSENSORS_MAIN override '%s' given in Rules.lmsensors not found.", lmsensors_main);
         return(PAPI_ENOSUPP);   // Override given but not found.
      }
   }

   // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
   if (dl1 == NULL) {                                           // No override,
      dl1 = dlopen("libsensors.so", RTLD_NOW | RTLD_GLOBAL);        // Try system paths.
   }

   // Step 3: Try the explicit install default. 
   if (dl1 == NULL && lmsensors_root != NULL) {                          // if root given, try it.
      snprintf(path_name, 1024, "%s/lib64/libsensors.so", lmsensors_root);   // PAPI Root check.
      dl1 = dlopen(path_name, RTLD_NOW | RTLD_GLOBAL);             // Try to open that path.
   }

   // Check for failure.
   if (dl1 == NULL) {
      snprintf(_lmsensors_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "libsensors.so not found.");
      return(PAPI_ENOSUPP);
   }

   // We have dl1. 

	sensors_initPtr = dlsym(dl1, "sensors_init");
	if (dlerror() != NULL)
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_init.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	sensors_cleanupPtr = dlsym(dl1, "sensors_cleanup");
	if (dlerror() != NULL)
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_cleanup.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	sensors_snprintf_chip_namePtr = dlsym(dl1, "sensors_snprintf_chip_name");
	if (dlerror() != NULL)
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_snprintf_chip_name.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	sensors_get_labelPtr = dlsym(dl1, "sensors_get_label");
	if (dlerror() != NULL)
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_get_label.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	sensors_get_valuePtr = dlsym(dl1, "sensors_get_value");
	if (dlerror() != NULL)
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_get_value.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	sensors_get_detected_chipsPtr = dlsym(dl1, "sensors_get_detected_chips");
	if (dlerror() != NULL)
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_get_detected_chips.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	sensors_get_featuresPtr = dlsym(dl1, "sensors_get_features");
	if (dlerror() != NULL)
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_get_features.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	sensors_get_all_subfeaturesPtr = dlsym(dl1, "sensors_get_all_subfeatures");
	if (dlerror() != NULL)
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_get_all_subfeatures.",PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}
	sensors_get_adapter_namePtr = dlsym(dl1, "sensors_get_adapter_name"); 
	if (dlerror() != NULL)  
	{
		strncpy(_lmsensors_vector.cmp_info.disabled_reason,
			"lmsensor function sensors_get_adapter_name not found.", PAPI_MAX_STR_LEN);
		return ( PAPI_ENOSUPP );
	}

	return ( PAPI_OK );
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
static int
_lmsensors_init_control_state( hwd_control_state_t *ctl )
{
	int i;

	for ( i = 0; i < num_events; i++ )
		cached_counts[i] = getEventValue( i );

	( ( _lmsensors_control_state_t * ) ctl )->lastupdate =
		PAPI_get_real_usec(  );
	return PAPI_OK;
}


/*
 *
 */
static int
_lmsensors_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	( void ) ctx;
	( void ) ctl;

	return PAPI_OK;
}


/*
 *
 */
static int
_lmsensors_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;
    ( void ) ctl;

    return PAPI_OK;
}


/*
 *
 */
static int
_lmsensors_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
		 long_long ** events, int flags )
{
    ( void ) ctx;
    ( void ) flags;
    long long start = PAPI_get_real_usec(  );
    int i;
 
    _lmsensors_control_state_t *control=(_lmsensors_control_state_t *)ctl;

    if ( start - control->lastupdate > 200000 ) {	// cache refresh
       
       for ( i = 0; i < num_events; i++ ) {
	   cached_counts[i] = getEventValue( i );
       }
       control->lastupdate = PAPI_get_real_usec(  );
    }

    *events = cached_counts;
    return PAPI_OK;
}

static int
_lmsensors_shutdown_component( void )
{
	if (cached_counts)   {
		free(cached_counts);
      cached_counts = NULL;
   }

	/* Call the libsensors cleaning function before leaving */
	sensors_cleanupPtr(  );

   if (lm_sensors_native_table) {
      free(lm_sensors_native_table);
      lm_sensors_native_table = NULL;
   }

	return PAPI_OK;
}

static int
_lmsensors_shutdown_thread( hwd_context_t *ctx )
{
    ( void ) ctx;

    return PAPI_OK;
}



/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
static int
_lmsensors_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    ( void ) ctx;
    ( void ) code;
    ( void ) option;
    return PAPI_OK;
}


static int
_lmsensors_update_control_state( hwd_control_state_t *ctl,
				 NativeInfo_t * native, 
				 int count,
				 hwd_context_t *ctx )
{
    int i, index;
    ( void ) ctx;
    ( void ) ctl;

    for ( i = 0; i < count; i++ ) {
	index = native[i].ni_event;
	native[i].ni_position =
			lm_sensors_native_table[index].resources.selector - 1;
    }
    return PAPI_OK;
}


/*
 * As I understand it, all data reported by these interfaces will be system wide
 */
static int
_lmsensors_set_domain( hwd_control_state_t *ctl, int domain )
{
	(void) ctl;
	if ( PAPI_DOM_ALL != domain )
		return ( PAPI_EINVAL );

	return ( PAPI_OK );
}


/*
 *
 */
static int
_lmsensors_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;
    ( void ) ctl;
    return PAPI_OK;
}


/*
 * Native Event functions
 */
static int
_lmsensors_ntv_enum_events( unsigned int *EventCode, int modifier )
{

	switch ( modifier ) {
	case PAPI_ENUM_FIRST:
		*EventCode = 0;

		return PAPI_OK;
		break;

	case PAPI_ENUM_EVENTS:
	{
		int index = *EventCode;

		if ( index < num_events - 1 ) {
			*EventCode = *EventCode + 1;
			return PAPI_OK;
		} else
			return PAPI_ENOEVNT;

		break;
	}
	default:
		return PAPI_EINVAL;
	}
	return PAPI_EINVAL;
}

/*
 *
 */
static int
_lmsensors_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	int index = EventCode;

	if (index>=0 && index<num_events) {
	   strncpy( name, lm_sensors_native_table[index].name, len );
	}

	return PAPI_OK;
}

/*
 *
 */
static int
_lmsensors_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	int index = EventCode;

	if (index>=0 && index<num_events) {
	   strncpy( name, lm_sensors_native_table[index].description, len );
	}
	return PAPI_OK;
}



/*
 *
 */
papi_vector_t _lmsensors_vector = {
   .cmp_info = {
        /* component information (unspecified values are initialized to 0) */
	.name = "lmsensors",
	.short_name = "lmsensors",
	.version = "5.0",
	.description = "Linux LMsensor statistics",
	.num_mpx_cntrs = 0,
	.num_cntrs = 0,
	.default_domain = PAPI_DOM_ALL,
	.default_granularity = PAPI_GRN_SYS,
	.available_granularities = PAPI_GRN_SYS,
	.hardware_intr_sig = PAPI_INT_SIGNAL,

	/* component specific cmp_info initializations */
	.fast_real_timer = 0,
	.fast_virtual_timer = 0,
	.attach = 0,
	.attach_must_ptrace = 0,
	.available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
  },

        /* sizes of framework-opaque component-private structures */
	.size = {
	   .context = sizeof ( _lmsensors_context_t ),
	   .control_state = sizeof ( _lmsensors_control_state_t ),
	   .reg_value = sizeof ( _lmsensors_register_t ),
	   .reg_alloc = sizeof ( _lmsensors_reg_alloc_t ),
  },
	/* function pointers in this component */
     .init_thread =          _lmsensors_init_thread,
     .init_component =       _lmsensors_init_component,
     .init_control_state =   _lmsensors_init_control_state,
     .start =                _lmsensors_start,
     .stop =                 _lmsensors_stop,
     .read =                 _lmsensors_read,
     .shutdown_thread =      _lmsensors_shutdown_thread,
     .shutdown_component =   _lmsensors_shutdown_component,
     .ctl =                  _lmsensors_ctl,
     .update_control_state = _lmsensors_update_control_state,
     .set_domain =           _lmsensors_set_domain,
     .reset =                _lmsensors_reset,
	
     .ntv_enum_events =      _lmsensors_ntv_enum_events,
     .ntv_code_to_name =     _lmsensors_ntv_code_to_name,
     .ntv_code_to_descr =    _lmsensors_ntv_code_to_descr,
};
