/** @file cpus.h
 *  CVS: $Id$
 * Author:  Gary Mohr
 *          gary.mohr@bull.com
 *          - based on threads.h by unknown author -
 */

#ifndef PAPI_CPUS_H
#define PAPI_CPUS_H

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

typedef struct _CpuInfo
{
	unsigned int cpu_num;
	struct _CpuInfo *next;
	hwd_context_t **context;
	EventSetInfo_t **running_eventset;
	EventSetInfo_t *from_esi;          /* ESI used for last update this control state */
} CpuInfo_t;

/* The list of cpus, gets built as user apps set the cpu papi option on an event set */
extern volatile CpuInfo_t *_papi_hwi_cpu_head;

int _papi_hwi_initialize_cpu( CpuInfo_t ** dest, unsigned int cpu_num );
int _papi_hwi_shutdown_cpu( CpuInfo_t * cpu );

inline_static CpuInfo_t *
_papi_hwi_lookup_cpu( unsigned int cpu_num )
{
	THRDBG("Entry:\n");
	CpuInfo_t *tmp;

	_papi_hwi_lock( CPUS_LOCK );

	tmp = ( CpuInfo_t * ) _papi_hwi_cpu_head;
	while ( tmp != NULL ) {
		THRDBG( "Examining cpu 0x%x at %p\n", tmp->cpu_num, tmp );
		if ( tmp->cpu_num == cpu_num )
			break;
		tmp = tmp->next;
		if ( tmp == _papi_hwi_cpu_head ) {
			tmp = NULL;
			break;
		}
	}

	if ( tmp ) {
		_papi_hwi_cpu_head = tmp;
		THRDBG( "Found cpu 0x%x at %p\n", cpu_num, tmp );
	} else {
		THRDBG( "Did not find cpu 0x%x\n", cpu_num );
	}

	_papi_hwi_unlock( CPUS_LOCK );
	return ( tmp );
}

inline_static int
_papi_hwi_lookup_or_create_cpu( CpuInfo_t ** here, unsigned int cpu_num )
{
	THRDBG("Entry: here: %p\n", here);
	CpuInfo_t *tmp = NULL;
	int retval = PAPI_OK;

	tmp = _papi_hwi_lookup_cpu(cpu_num);
	if ( tmp == NULL )
		retval = _papi_hwi_initialize_cpu( &tmp, cpu_num );

	if ( retval == PAPI_OK )
		*here = tmp;

	return ( retval );
}

#endif
