#include "papi.h"
#include <invent.h>

inventory_t *getinvent (void);
int get_memory_info( PAPI_mem_info_t * mem_info ){
	inventory_t *curr;

	while ( (curr=getinvent())!= NULL ) {
		if ((curr->inv_class == INV_MEMORY ) && (curr->inv_type==INV_DCACHE)) {	
			mem_info->L1_dcache_size = curr->inv_state /1024;
		}
		if ((curr->inv_class == INV_MEMORY ) && (curr->inv_type==INV_ICACHE)) {	
			mem_info->L1_icache_size = curr->inv_state /1024;
		}
		if ((curr->inv_class == INV_MEMORY ) && (curr->inv_type==INV_SIDCACHE))	
			mem_info->L2_cache_size = curr->inv_state / 1024 ;
	}
	mem_info->total_L1_size = mem_info->L1_dcache_size + mem_info->L1_icache_size;

/************** 
	mem_info->L1_dcache_linesize = 32;
	mem_info->L1_dcache_lines = mem_info->L1_dcache_size / mem_info->L1_dcache_linesize;
	mem_info->L1_dcache_assoc = 2;
	mem_info->L1_icache_linesize = 32;
	mem_info->L1_icache_lines = mem_info->L1_icache_size / mem_info->L1_icache_linesize;
	mem_info->L1_icache_assoc = 2;

	mem_info->L2_cache_assoc =2;
	mem_info->L2_cache_linesize = 128;
	mem_info->L2_cache_lines = mem_info->L2_cache_size / mem_info->L2_cache_linesize;
***************/

    return PAPI_OK;
}
