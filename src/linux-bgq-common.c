/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-bgq-common.c
 * CVS:     $Id$
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	<your name here>
 *			<your email address>
 * BGPM component 
 * 
 * Tested version of bgpm (early access)
 *
 * @brief
 *  This file is part of the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for BG/Q through the bgpm library.
 */

#include "papi.h"
#include "linux-bgq-common.h"

/*******************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT **********
 ******************************************************************************/

void _check_BGPM_error( int err, char* bgpmfunc )
{
	if ( err < 0 ) {
#ifdef DEBUG_BGPM
		printf ( "Error: ret value is %d for BGPM API function '%s'.\n",
				 err, bgpmfunc);
#endif
	}
}


/*
 * Returns all event values from the BGPM eventGroup 
 */
long_long
_common_getEventValue( unsigned event_id, int EventGroup )
{	
	uint64_t value;
    int retval;
	
	retval = Bgpm_ReadEvent( EventGroup, event_id, &value );
	CHECK_BGPM_ERROR( retval, "Bgpm_ReadEvent" );
	
	return ( ( long_long ) value );	
}


/*
 * Delete BGPM eventGroup and create an new empty one
 */
void
_common_deleteRecreate( int *EventGroup_ptr )
{
#ifdef DEBUG_BGQ
	printf( _AT_ " _common_deleteRecreate: *EventGroup_ptr=%d\n", *EventGroup_ptr);
#endif
	int retval;
	
	// delete previous bgpm eventset
	retval = Bgpm_DeleteEventSet( *EventGroup_ptr );
	CHECK_BGPM_ERROR( retval, "Bgpm_DeleteEventSet" );
	
	// create a new empty bgpm eventset
	*EventGroup_ptr = Bgpm_CreateEventSet();
	CHECK_BGPM_ERROR( *EventGroup_ptr, "Bgpm_CreateEventSet" );

#ifdef DEBUG_BGQ
	printf( _AT_ " _common_deleteRecreate: *EventGroup_ptr=%d\n", *EventGroup_ptr);
#endif	
}


/*
 * Rebuild BGPM eventGroup with the events as it was prior to deletion 
 */
void
_common_rebuildEventgroup( int count, int *EventGroup_local, int *EventGroup_ptr )
{
#ifdef DEBUG_BGQ
	printf( "_common_rebuildEventgroup\n" );
#endif	
	int i, retval;
	
	// rebuild BGPM EventGroup
	for ( i = 0; i < count; i++ ) {
		retval = Bgpm_AddEvent( *EventGroup_ptr, EventGroup_local[i] );
		CHECK_BGPM_ERROR( retval, "Bgpm_AddEvent" );
	
#ifdef DEBUG_BGQ
		printf( "_common_rebuildEventgroup: After emptying EventGroup, event re-added: %d\n",
			    EventGroup_local[i] );
#endif
	}
}




