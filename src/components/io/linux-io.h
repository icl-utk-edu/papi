/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    linux-io.h
 * CVS:     $Id$
 *
 * @author  Kevin A. Huck
 *          khuck@uoregon.edu
 *
 * @ingroup papi_components
 *
 * @brief io component
 *  This file contains the source code for a component that enables
 *  PAPI-C to access I/O statistics through the /proc file system.
 *  This component will dynamically create a native events table for
 *  all the 7 counters in /proc/self/io.
 */

#ifndef _PAPI_IO_H
#define _PAPI_IO_H

#include <unistd.h>

/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/
/* this number assumes that there will never be more events than indicated */
#define IO_MAX_COUNTERS 7

/** Structure that stores private information of each event */
typedef struct IO_register
{
    /**< Signifies which counter slot is being used */
    /**< Indexed from 1 as 0 has a special meaning  */
    unsigned int selector;
} IO_register_t;


/*
 * The following structures mimic the ones used by other components. It is more
 * convenient to use them like that as programming with PAPI makes specific
 * assumptions for them.
 */


/** This structure is used to build the table of events */
typedef struct IO_native_event_entry
{
    IO_register_t resources;            /**< Per counter resources       */
    char name[PAPI_MAX_STR_LEN];	    /**< Name of the counter         */
    char description[PAPI_MAX_STR_LEN]; /**< Description of the counter  */
	int writable;                       /**< Whether counter is writable */
} IO_native_event_entry_t;

/** This structure is used when doing register allocation 
    it possibly is not necessary when there are no 
    register constraints */
typedef struct IO_reg_alloc
{
    IO_register_t ra_bits;
} IO_reg_alloc_t;

/** Holds control flags.  
 *    There's one of these per event-set.
 *    Use this to hold data specific to the EventSet, either hardware
 *      counter settings or things like counter start values 
 */
typedef struct IO_control_state
{
    int num_events;
    int domain;
    int multiplexed;
    int overflow;
    int inherit;
    int which_counter[IO_MAX_COUNTERS]; 
    long long counter[IO_MAX_COUNTERS]; // used for caching
} IO_control_state_t;


typedef struct IO_context
{
    IO_control_state_t state;
} IO_context_t;

/*************************  GLOBALS SECTION  ***********************************
 *******************************************************************************/

#endif /* _PAPI_IO_H */

/* vim:set ts=4 sw=4 sts=4 et: */
