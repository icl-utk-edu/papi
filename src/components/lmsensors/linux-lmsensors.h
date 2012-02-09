/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-lmsensors.h
 * CVS:     $Id$
 * @author  Daniel Lucio
 * @author  Joachim Protze
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 *
 * LM_SENSORS component 
 * 
 * Tested version of lm_sensors: 3.1.1
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

#ifndef _PAPI_LMSENSORS_H
#define _PAPI_LMSENSORS_H

/* Headers required by libsensors */
#include <sensors.h>
#include <error.h>
#include <time.h>
#include <string.h>

/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/
/* this number assumes that there will never be more events than indicated */
#define LM_SENSORS_MAX_COUNTERS 512
// time in usecs
#define LM_SENSORS_REFRESHTIME 200000

/** Structure that stores private information of each event */
typedef struct LM_SENSORS_register
{
	/* This is used by the framework.It likes it to be !=0 to do somehting */
	unsigned int selector;
	/* These are the only information needed to locate a libsensors event */
	const sensors_chip_name *name;
	int subfeat_nr;
} LM_SENSORS_register_t;

/*
 * The following structures mimic the ones used by other components. It is more
 * convenient to use them like that as programming with PAPI makes specific
 * assumptions for them.
 */

/** This structure is used to build the table of events */
typedef struct LM_SENSORS_native_event_entry
{
	LM_SENSORS_register_t resources;
	char name[PAPI_MAX_STR_LEN];
	char description[PAPI_MAX_STR_LEN];
	unsigned int count;
} LM_SENSORS_native_event_entry_t;


typedef struct LM_SENSORS_reg_alloc
{
	LM_SENSORS_register_t ra_bits;
} LM_SENSORS_reg_alloc_t;


typedef struct LM_SENSORS_control_state
{
	long_long counts[LM_SENSORS_MAX_COUNTERS];	// used for caching
	long_long lastupdate;
} LM_SENSORS_control_state_t;


typedef struct LM_SENSORS_context
{
	LM_SENSORS_control_state_t state;
} LM_SENSORS_context_t;



/*************************  GLOBALS SECTION  ***********************************
 *******************************************************************************/
/* This table contains the LM_SENSORS native events */
static LM_SENSORS_native_event_entry_t *lm_sensors_native_table;
/* number of events in the table*/
static int NUM_EVENTS = 0;



#endif /* _PAPI_LMSENSORS_H */
