/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    cpus.c
* CVS:     $Id$
* Author:  Gary Mohr
*          gary.mohr@bull.com
*          - based on threads.c by Philip Mucci -
*/

/* This file contains cpu allocation and bookkeeping functions */

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"
#include <string.h>
#include <unistd.h>

/*****************/
/* BEGIN GLOBALS */
/*****************/

/* The following globals get initialized and cleared by:
   extern int _papi_hwi_init_global_cpus(void);
   extern int _papi_hwi_shutdown_cpu(CpuInfo_t *cpu); */

volatile CpuInfo_t *_papi_hwi_cpu_head;

/*****************/
/*  END  GLOBALS */
/*****************/

static CpuInfo_t *
allocate_cpu( unsigned int cpu_num )
{
   THRDBG("Entry: cpu_num: %d\n", cpu_num);
	CpuInfo_t *cpu;
	int i;

	cpu = ( CpuInfo_t * ) papi_malloc( sizeof ( CpuInfo_t ) );
	if ( cpu == NULL )
		return ( NULL );
	memset( cpu, 0x00, sizeof ( CpuInfo_t ) );
	
	/* identify the cpu this info structure represents */
	cpu->cpu_num = cpu_num;

	cpu->context = ( hwd_context_t ** ) papi_malloc( sizeof ( hwd_context_t * ) *
										  ( size_t ) papi_num_components );
	if ( !cpu->context ) {
		papi_free( cpu );
		return ( NULL );
	}
	
	cpu->running_eventset =
		( EventSetInfo_t ** ) papi_malloc( sizeof ( EventSetInfo_t * ) *
										   ( size_t ) papi_num_components );
	if ( !cpu->running_eventset ) {
		papi_free( cpu->context );
		papi_free( cpu );
		return ( NULL );
	}

	for ( i = 0; i < papi_num_components; i++ ) {
		cpu->context[i] =
			( void * ) papi_malloc( ( size_t ) _papi_hwd[i]->size.context );
		cpu->running_eventset[i] = NULL;
		if ( cpu->context[i] == NULL ) {
			for ( i--; i >= 0; i-- )
				papi_free( cpu->context[i] );
			papi_free( cpu->context );
			papi_free( cpu );
			return ( NULL );
		}
		memset( cpu->context[i], 0x00,
				( size_t ) _papi_hwd[i]->size.context );
	}

	THRDBG( "Allocated CpuInfo: %p\n", cpu );
	return ( cpu );
}

static void
free_cpu( CpuInfo_t ** cpu )
{
   THRDBG( "Entry: *cpu: %p, cpu_num: 0x%x\n", *cpu, ( *cpu )->cpu_num);
	int i;
	for ( i = 0; i < papi_num_components; i++ ) {
		if ( ( *cpu )->context[i] )
			papi_free( ( *cpu )->context[i] );
	}

	if ( ( *cpu )->context )
		papi_free( ( *cpu )->context );

	if ( ( *cpu )->running_eventset )
		papi_free( ( *cpu )->running_eventset );

	memset( *cpu, 0x00, sizeof ( CpuInfo_t ) );
	papi_free( *cpu );
	*cpu = NULL;
}

static void
insert_cpu( CpuInfo_t * entry )
{
   THRDBG("Entry: entry: %p\n", entry);
	_papi_hwi_lock( CPUS_LOCK );

	if ( _papi_hwi_cpu_head == NULL ) {	/* 0 elements */
		THRDBG( "_papi_hwi_cpu_head is NULL\n" );
		entry->next = entry;
	} else if ( _papi_hwi_cpu_head->next == _papi_hwi_cpu_head ) {	/* 1 elements */
		THRDBG( "_papi_hwi_cpu_head was cpu %d at %p\n",
				_papi_hwi_cpu_head->cpu_num, _papi_hwi_cpu_head );
		_papi_hwi_cpu_head->next = entry;
		entry->next = ( CpuInfo_t * ) _papi_hwi_cpu_head;
	} else {				 /* 2+ elements */

		THRDBG( "_papi_hwi_cpu_head was cpu %d at %p\n",
				_papi_hwi_cpu_head->cpu_num, _papi_hwi_cpu_head );
		entry->next = _papi_hwi_cpu_head->next;
		_papi_hwi_cpu_head->next = entry;
	}

	_papi_hwi_cpu_head = entry;

	THRDBG( "_papi_hwi_cpu_head now cpu %d at %p\n",
			_papi_hwi_cpu_head->cpu_num, _papi_hwi_cpu_head );

	_papi_hwi_unlock( CPUS_LOCK );

}

static int
remove_cpu( CpuInfo_t * entry )
{
   THRDBG("Entry: entry: %p\n", entry);
	CpuInfo_t *tmp = NULL, *prev = NULL;

	_papi_hwi_lock( CPUS_LOCK );

	THRDBG( "_papi_hwi_cpu_head was cpu %d at %p\n",
			_papi_hwi_cpu_head->cpu_num, _papi_hwi_cpu_head );

	/* Find the preceding element and the matched element,
	   short circuit if we've seen the head twice */

	for ( tmp = ( CpuInfo_t * ) _papi_hwi_cpu_head;
		  ( entry != tmp ) || ( prev == NULL ); tmp = tmp->next ) {
		prev = tmp;
	}

	if ( tmp != entry ) {
		THRDBG( "Cpu %d at %p was not found in the cpu list!\n",
				entry->cpu_num, entry );
		return ( PAPI_EBUG );
	}

	/* Only 1 element in list */

	if ( prev == tmp ) {
		_papi_hwi_cpu_head = NULL;
		tmp->next = NULL;
		THRDBG( "_papi_hwi_cpu_head now NULL\n" );
	} else {
		prev->next = tmp->next;
		/* If we're removing the head, better advance it! */
		if ( _papi_hwi_cpu_head == tmp ) {
			_papi_hwi_cpu_head = tmp->next;
			THRDBG( "_papi_hwi_cpu_head now cpu %d at %p\n",
					_papi_hwi_cpu_head->cpu_num, _papi_hwi_cpu_head );
		}
		THRDBG( "Removed cpu %p from list\n", tmp );
	}

	_papi_hwi_unlock( CPUS_LOCK );

	return ( PAPI_OK );
}

int
_papi_hwi_initialize_cpu( CpuInfo_t ** dest, unsigned int cpu_num )
{
   THRDBG("Entry: dest: %p, *dest: %p, cpu_num: %d\n", dest, *dest, cpu_num);
	int retval;
	CpuInfo_t *cpu;
	int i;

	if ( ( cpu = allocate_cpu(cpu_num) ) == NULL ) {
		*dest = NULL;
		return ( PAPI_ENOMEM );
	}

	/* Call the substrate to fill in anything special. */
	for ( i = 0; i < papi_num_components; i++ ) {
		retval = _papi_hwd[i]->init( cpu->context[i] );
		if ( retval ) {
			free_cpu( &cpu );
			*dest = NULL;
			return ( retval );
		}
	}

	insert_cpu( cpu );

	*dest = cpu;
	return ( PAPI_OK );
}

int
_papi_hwi_shutdown_cpu( CpuInfo_t * cpu )
{
   THRDBG("Entry: cpu: %p, cpu_num: %d\n", cpu, cpu->cpu_num);
	int retval = PAPI_OK;
	int i, failure = 0;

	remove_cpu( cpu );
	THRDBG( "Shutting down cpu %d at %p\n", cpu->cpu_num, cpu );
	for ( i = 0; i < papi_num_components; i++ ) {
		retval = _papi_hwd[i]->shutdown( cpu->context[i] );
		if ( retval != PAPI_OK )
			failure = retval;
	}
	free_cpu( &cpu );
	return ( failure );
}
