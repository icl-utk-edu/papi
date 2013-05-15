/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    cpus.c
* Author:  Gary Mohr
*          gary.mohr@bull.com
*          - based on threads.c by Philip Mucci -
*/

/* This file contains cpu allocation and bookkeeping functions */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "cpus.h"
#include <string.h>
#include <unistd.h>

/* The list of cpus; this gets built as user apps set the cpu papi */
/* option on an event set */

static CpuInfo_t *_papi_hwi_cpu_head;


static CpuInfo_t *
_papi_hwi_lookup_cpu( unsigned int cpu_num )
{
   APIDBG("Entry:\n");

   CpuInfo_t *tmp;

   _papi_hwi_lock( CPUS_LOCK );

   tmp = ( CpuInfo_t * ) _papi_hwi_cpu_head;
   while ( tmp != NULL ) {
      THRDBG( "Examining cpu %#x at %p\n", tmp->cpu_num, tmp );
      if ( tmp->cpu_num == cpu_num ) {
	 break;
      }
      tmp = tmp->next;
      if ( tmp == _papi_hwi_cpu_head ) {
	 tmp = NULL;
	 break;
      }
   }

   if ( tmp ) {
      _papi_hwi_cpu_head = tmp;
      THRDBG( "Found cpu %#x at %p\n", cpu_num, tmp );
   } else {
      THRDBG( "Did not find cpu %#x\n", cpu_num );
   }

   _papi_hwi_unlock( CPUS_LOCK );
	
   return tmp;
}

int
_papi_hwi_lookup_or_create_cpu( CpuInfo_t **here, unsigned int cpu_num )
{
   APIDBG("Entry: here: %p\n", here);
	
   CpuInfo_t *tmp = NULL;
   int retval = PAPI_OK;

   tmp = _papi_hwi_lookup_cpu(cpu_num);
   if ( tmp == NULL ) {
      retval = _papi_hwi_initialize_cpu( &tmp, cpu_num );
   }

   if ( retval == PAPI_OK )
      *here = tmp;

   return retval;
}


static CpuInfo_t *
allocate_cpu( unsigned int cpu_num )
{
   APIDBG("Entry: cpu_num: %d\n", cpu_num);

   CpuInfo_t *cpu;
   int i;

   /* Allocate new CpuInfo structure */
   cpu = ( CpuInfo_t * ) papi_calloc( 1, sizeof ( CpuInfo_t ) );
   if ( cpu == NULL ) {
      goto allocate_error;
   }
	
   /* identify the cpu this info structure represents */
   cpu->cpu_num = cpu_num;
   cpu->context = ( hwd_context_t ** ) 
                  papi_calloc( ( size_t ) papi_num_components ,
			       sizeof ( hwd_context_t * ) );
   if ( !cpu->context ) {
      goto error_free_cpu;
   }
 
   /* Allocate an eventset per component per cpu?  Why? */
	
   cpu->running_eventset = ( EventSetInfo_t ** ) 
                           papi_calloc(( size_t ) papi_num_components, 
                                       sizeof ( EventSetInfo_t * ) );
   if ( !cpu->running_eventset ) {
      goto error_free_context;
   }

   for ( i = 0; i < papi_num_components; i++ ) {
       cpu->context[i] =
	 ( void * ) papi_calloc( 1, ( size_t ) _papi_hwd[i]->size.context );
       cpu->running_eventset[i] = NULL;
       if ( cpu->context[i] == NULL ) {
	  goto error_free_contexts;
       }
   }

   THRDBG( "Allocated CpuInfo: %p\n", cpu );

   return cpu;

error_free_contexts:
   for ( i--; i >= 0; i-- ) papi_free( cpu->context[i] );
error_free_context:
   papi_free( cpu->context );
error_free_cpu:
   papi_free( cpu );
allocate_error:
   return NULL;
}

static void
free_cpu( CpuInfo_t ** cpu )
{
   APIDBG( "Entry: *cpu: %p, cpu_num: %#x\n", *cpu, ( *cpu )->cpu_num);
	
   int i;
   for ( i = 0; i < papi_num_components; i++ ) {
      if ( ( *cpu )->context[i] ) {
	 papi_free( ( *cpu )->context[i] );
      }
   }

   if ( ( *cpu )->context ) {
      papi_free( ( *cpu )->context );
   }

   if ( ( *cpu )->running_eventset ) {
      papi_free( ( *cpu )->running_eventset );
   }

   /* why do we clear this? */
   memset( *cpu, 0x00, sizeof ( CpuInfo_t ) );
   papi_free( *cpu );
   *cpu = NULL;
}

static void
insert_cpu( CpuInfo_t * entry )
{
   APIDBG("Entry: entry: %p\n", entry);

   _papi_hwi_lock( CPUS_LOCK );

   if ( _papi_hwi_cpu_head == NULL ) {	
      /* 0 elements */
      THRDBG( "_papi_hwi_cpu_head is NULL\n" );
      entry->next = entry;
   } else if ( _papi_hwi_cpu_head->next == _papi_hwi_cpu_head ) {
      /* 1 element */
      THRDBG( "_papi_hwi_cpu_head was cpu %d at %p\n",
              _papi_hwi_cpu_head->cpu_num, _papi_hwi_cpu_head );
      _papi_hwi_cpu_head->next = entry;
      entry->next = ( CpuInfo_t * ) _papi_hwi_cpu_head;
   } else {
      /* 2+ elements */
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
   APIDBG("Entry: entry: %p\n", entry);
	
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
      return PAPI_EBUG;
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

   return PAPI_OK;
}

int
_papi_hwi_initialize_cpu( CpuInfo_t ** dest, unsigned int cpu_num )
{
   APIDBG("Entry: dest: %p, *dest: %p, cpu_num: %d\n", dest, *dest, cpu_num);

   int retval;
   CpuInfo_t *cpu;
   int i;

   if ( ( cpu = allocate_cpu(cpu_num) ) == NULL ) {
      *dest = NULL;
      return PAPI_ENOMEM;
   }

   /* Call the component to fill in anything special. */
   for ( i = 0; i < papi_num_components; i++ ) {
      retval = _papi_hwd[i]->init_thread( cpu->context[i] );
      if ( retval ) {
	 free_cpu( &cpu );
	 *dest = NULL;
	 return retval;
      }
   }

   insert_cpu( cpu );

   *dest = cpu;
   return PAPI_OK;
}

int
_papi_hwi_shutdown_cpu( CpuInfo_t *cpu )
{
   APIDBG("Entry: cpu: %p, cpu_num: %d\n", cpu, cpu->cpu_num);

   int retval = PAPI_OK;
   int i, failure = 0;

   remove_cpu( cpu );

   THRDBG( "Shutting down cpu %d at %p\n", cpu->cpu_num, cpu );

   for ( i = 0; i < papi_num_components; i++ ) {
      retval = _papi_hwd[i]->shutdown_thread( cpu->context[i] );
      if ( retval != PAPI_OK ) {
	 failure = retval;
      }
   }
   free_cpu( &cpu );

   return failure;
}
