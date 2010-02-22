/* 
* File:    linux-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"
#ifdef __LINUX__
#include <limits.h>
#endif
#include SUBSTRATE
#include <stdio.h>
int init_bgl( PAPI_mh_info_t * mem_info );
inline void cpuid( unsigned int *, unsigned int *, unsigned int *,
				   unsigned int * );

int
_papi_hwd_get_memory_info( PAPI_hw_info_t * hw_info, int cpu_type )
{
	int retval = 0;
	PAPI_mh_info_t *mem_info = &hw_info->mem_hierarchy;

	switch ( cpu_type ) {
	default:
		//fprintf(stderr,"CPU type unknown in %s (%d)\n",__FUNCTION__,__LINE__);
		retval = init_bgl( &hw_info->mem_hierarchy );
		break;
	}
/*  SUBDBG((stderr,"Detected L1: %d L2: %d  L3: %d\n",
       mem_info->total_L1_size, mem_info->L2_cache_size, 
       mem_info->L3_cache_size));
*/
	return retval;
}

/* Cache configuration for AMD AThlon/Duron */
int
init_bgl( PAPI_mh_info_t * mem_info )
{
	memset( mem_info, 0x0, sizeof ( *mem_info ) );
	//fprintf(stderr,"mem_info not est up [%s (%d)]\n",__FUNCTION__,__LINE__);
	return PAPI_OK;
}
