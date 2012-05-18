/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-bgq-common.h
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

/* Header required by BGPM */
#include "bgpm/include/bgpm.h"

/* Specific errors from BGPM lib */
#define CHECK_BGPM_ERROR(err, bgpmfunc) _check_BGPM_error( err, bgpmfunc );

// Define gymnastics to create a compile time AT string.
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define _AT_ __FILE__ ":" TOSTRING(__LINE__)

/* return EXIT_FAILURE;  \*/

#define MAX_COUNTERS ( PEVT_LAST_EVENT + 1 )
//#define DEBUG_BGQ


/*************************  COMMON PROTOTYPES  *********************************
 *******************************************************************************/

/* common prototypes for BGQ sustrate and BGPM components */
void        _check_BGPM_error( int err, char* bgpmfunc );
long_long	_common_getEventValue( unsigned event_id, int EventGroup );
void		_common_deleteRecreate( int *EventGroup_ptr );
void		_common_rebuildEventgroup( int count, int *EventGroup_local, int *EventGroup_ptr );

