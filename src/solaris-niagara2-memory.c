/*******************************************************************************
 * >>>>>> "Development of a PAPI Backend for the Sun Niagara 2 Processor" <<<<<<
 * -----------------------------------------------------------------------------
 *
 * Fabian Gorsler <fabian.gorsler@smail.inf.h-bonn-rhein-sieg.de>
 *
 *       Hochschule Bonn-Rhein-Sieg, Sankt Augustin, Germany
 *       University of Applied Sciences
 *
 * -----------------------------------------------------------------------------
 *
 * File:   solaris-niagara2-memory.c
 * Author: fg215045
 *
 * Description: This file holds functions returning information about the memory
 * hiearchy and the memory available to this process. utils/papi_mem_info and
 * ctests/dmem_info are related to these functions.
 *
 *      ***** Feel free to convert this header to the PAPI default *****
 *
 * -----------------------------------------------------------------------------
 * Created on April 23, 2009, 3:18 PM
 ******************************************************************************/

#include "papi.h"
#include "papi_internal.h"

int
_niagara2_get_memory_info( PAPI_hw_info_t * hw, int id )
{
	PAPI_mh_level_t *mem = hw->mem_hierarchy.level;

#ifdef DEBUG
	SUBDBG( "ENTERING FUNCTION >>%s<< at %s:%d\n", __func__, __FILE__,
			__LINE__ );
#endif

	/* I-Cache -> L1$ instruction */
	/* FIXME: The policy used at this cache is unknown to PAPI. LSFR with random
	   replacement. */
	mem[0].cache[0].type = PAPI_MH_TYPE_INST;
	mem[0].cache[0].size = 16 * 1024;	// 16 Kb
	mem[0].cache[0].line_size = 32;
	mem[0].cache[0].num_lines =
		mem[0].cache[0].size / mem[0].cache[0].line_size;
	mem[0].cache[0].associativity = 8;

	/* D-Cache -> L1$ data */
	mem[0].cache[1].type =
		PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU;
	mem[0].cache[1].size = 8 * 1024;	// 8 Kb
	mem[0].cache[1].line_size = 16;
	mem[0].cache[1].num_lines =
		mem[0].cache[1].size / mem[0].cache[1].line_size;
	mem[0].cache[1].associativity = 4;

	/* ITLB -> TLB instruction */
	mem[0].tlb[0].type = PAPI_MH_TYPE_INST | PAPI_MH_TYPE_PSEUDO_LRU;
	mem[0].tlb[0].num_entries = 64;
	mem[0].tlb[0].associativity = 64;

	/* DTLB -> TLB data */
	mem[0].tlb[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_PSEUDO_LRU;
	mem[0].tlb[1].num_entries = 128;
	mem[0].tlb[1].associativity = 128;

	/* L2$ unified */
	mem[1].cache[0].type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_WB
		| PAPI_MH_TYPE_PSEUDO_LRU;
	mem[1].cache[0].size = 4 * 1024 * 1024;	// 4 Mb
	mem[1].cache[0].line_size = 64;
	mem[1].cache[0].num_lines =
		mem[1].cache[0].size / mem[1].cache[0].line_size;
	mem[1].cache[0].associativity = 16;

	/* Indicate we have two levels filled in the hierarchy */
	hw->mem_hierarchy.levels = 2;

#ifdef DEBUG
	SUBDBG( "LEAVING FUNCTION >>%s<< at %s:%d\n", __func__, __FILE__,
			__LINE__ );
#endif

	return PAPI_OK;
}

int
_niagara2_get_dmem_info( PAPI_dmem_info_t * d )
{
#ifdef DEBUG
	SUBDBG( "ENTERING FUNCTION >>%s<< at %s:%d\n", __func__, __FILE__,
			__LINE__ );
#endif

	/* More information needed to fill all fields */
	d->pagesize = sysconf( _SC_PAGESIZE );
	d->size = d->pagesize * sysconf( _SC_PHYS_PAGES );
	d->resident = PAPI_EINVAL;
	d->high_water_mark = PAPI_EINVAL;
	d->shared = PAPI_EINVAL;
	d->text = PAPI_EINVAL;
	d->library = PAPI_EINVAL;
	d->heap = PAPI_EINVAL;
	d->locked = PAPI_EINVAL;
	d->stack = PAPI_EINVAL;

#ifdef DEBUG
	SUBDBG( "LEAVING FUNCTION >>%s<< at %s:%d\n", __func__, __FILE__,
			__LINE__ );
#endif

	return PAPI_OK;
}
