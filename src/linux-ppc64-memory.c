/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    linux-ppc64-memory.c
* Author:  Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>

PAPI_mh_info_t sys_mem_info[2] = {
	{2,						 // 970 begin
	 {
	  {						 // level 1 begins
	   {					 // tlb's begin
		{PAPI_MH_TYPE_UNIFIED, 1024, 4}
		,
		{PAPI_MH_TYPE_EMPTY, -1, -1}
		}
	   ,
	   {					 // caches begin
		{PAPI_MH_TYPE_INST, 65536, 128, 512, 1}
		,
		{PAPI_MH_TYPE_DATA, 32768, 128, 256, 2}
		}
	   }
	  ,
	  {						 // level 2 begins
	   {					 // tlb's begin
		{PAPI_MH_TYPE_EMPTY, -1, -1}
		,
		{PAPI_MH_TYPE_EMPTY, -1, -1}
		}
	   ,
	   {					 // caches begin
		{PAPI_MH_TYPE_UNIFIED, 524288, 128, 4096, 8}
		,
		{PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
		}
	   }
	  ,
	  }
	 }
	,						 // 970 end
	{3,
	 {
	  {						 // level 1 begins
	   {					 // tlb's begin
		{PAPI_MH_TYPE_UNIFIED, 1024, 4}
		,
		{PAPI_MH_TYPE_EMPTY, -1, -1}
		}
	   ,
	   {					 // caches begin
		{PAPI_MH_TYPE_INST, 65536, 128, 512, 2}
		,
		{PAPI_MH_TYPE_DATA, 32768, 128, 256, 4}
		}
	   }
	  ,
	  {						 // level 2 begins
	   {					 // tlb's begin
		{PAPI_MH_TYPE_EMPTY, -1, -1}
		,
		{PAPI_MH_TYPE_EMPTY, -1, -1}
		}
	   ,
	   {					 // caches begin
		{PAPI_MH_TYPE_UNIFIED, 1966080, 128, 15360, 10}
		,
		{PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
		}
	   }
	  ,
	  {						 // level 3 begins
	   {					 // tlb's begin
		{PAPI_MH_TYPE_EMPTY, -1, -1}
		,
		{PAPI_MH_TYPE_EMPTY, -1, -1}
		}
	   ,
	   {					 // caches begin
		{PAPI_MH_TYPE_UNIFIED, 37748736, 256, 147456, 12}
		,
		{PAPI_MH_TYPE_EMPTY, -1, -1, -1, -1}
		}
	   }
	  ,
	  }
	 }						 // POWER5 end
};

#define                 SPRN_PVR                0x11F	/* Processor Version Register */
static unsigned int
mfpvr( void )
{
	unsigned long pvr;

  asm( "mfspr          %0,%1": "=r"( pvr ):"i"( SPRN_PVR ) );
	return pvr;

}

/* The second variable, cpu_type, is currently not used.
   I suspect it may perform the same function as mfpvr.
*/
int
_papi_hwd_get_memory_info( PAPI_hw_info_t * hw_info, int cpu_type )
{
	unsigned int pvr = mfpvr(  );

	int index;
	switch ( pvr ) {
	case 0x39:
	case 0x3C:
		index = 1;
		break;
	case 0x3A:
	case 0x3B:
		index = 1;
		break;
	default:
		index = -1;
		break;
	};

	if ( index != -1 ) {
		int cache_level;
		PAPI_mh_info_t sys_mh_inf = sys_mem_info[index];
		PAPI_mh_info_t *mh_inf = &hw_info->mem_hierarchy;
		mh_inf->levels = sys_mh_inf.levels;
		PAPI_mh_level_t *level = mh_inf->level;
		PAPI_mh_level_t sys_mh_level;
		for ( cache_level = 0; cache_level < sys_mh_inf.levels; cache_level++ ) {
			sys_mh_level = sys_mh_inf.level[cache_level];
			int cache_idx;
			for ( cache_idx = 0; cache_idx < 2; cache_idx++ ) {
				// process TLB info
				PAPI_mh_tlb_info_t curr_tlb = sys_mh_level.tlb[cache_idx];
				int type = curr_tlb.type;
				if ( type != PAPI_MH_TYPE_EMPTY ) {
					level[cache_level].tlb[cache_idx].type = type;
					level[cache_level].tlb[cache_idx].associativity =
						curr_tlb.associativity;
					level[cache_level].tlb[cache_idx].num_entries =
						curr_tlb.num_entries;
				}
			}
			for ( cache_idx = 0; cache_idx < 2; cache_idx++ ) {
				// process cache info
				PAPI_mh_cache_info_t curr_cache = sys_mh_level.cache[cache_idx];
				int type = curr_cache.type;
				if ( type != PAPI_MH_TYPE_EMPTY ) {
					level[cache_level].cache[cache_idx].type = type;
					level[cache_level].cache[cache_idx].associativity =
						curr_cache.associativity;
					level[cache_level].cache[cache_idx].size = curr_cache.size;
					level[cache_level].cache[cache_idx].line_size =
						curr_cache.line_size;
					level[cache_level].cache[cache_idx].num_lines =
						curr_cache.num_lines;
				}
			}
		}
	}
	return 0;
}

int
_papi_hwd_get_dmem_info( PAPI_dmem_info_t * d )
{
	char fn[PATH_MAX], tmp[PATH_MAX];
	FILE *f;
	int ret;
	long long sz = 0, lck = 0, res = 0, shr = 0, stk = 0, txt = 0, dat =
		0, dum = 0, lib = 0, hwm = 0;

	sprintf( fn, "/proc/%ld/status", ( long ) getpid(  ) );
	f = fopen( fn, "r" );
	if ( f == NULL ) {
		PAPIERROR( "fopen(%s): %s\n", fn, strerror( errno ) );
		return PAPI_ESBSTR;
	}
	while ( 1 ) {
		if ( fgets( tmp, PATH_MAX, f ) == NULL )
			break;
		if ( strspn( tmp, "VmSize:" ) == strlen( "VmSize:" ) ) {
			sscanf( tmp + strlen( "VmSize:" ), "%lld", &sz );
			d->size = sz;
			continue;
		}
		if ( strspn( tmp, "VmHWM:" ) == strlen( "VmHWM:" ) ) {
			sscanf( tmp + strlen( "VmHWM:" ), "%lld", &hwm );
			d->high_water_mark = hwm;
			continue;
		}
		if ( strspn( tmp, "VmLck:" ) == strlen( "VmLck:" ) ) {
			sscanf( tmp + strlen( "VmLck:" ), "%lld", &lck );
			d->locked = lck;
			continue;
		}
		if ( strspn( tmp, "VmRSS:" ) == strlen( "VmRSS:" ) ) {
			sscanf( tmp + strlen( "VmRSS:" ), "%lld", &res );
			d->resident = res;
			continue;
		}
		if ( strspn( tmp, "VmData:" ) == strlen( "VmData:" ) ) {
			sscanf( tmp + strlen( "VmData:" ), "%lld", &dat );
			d->heap = dat;
			continue;
		}
		if ( strspn( tmp, "VmStk:" ) == strlen( "VmStk:" ) ) {
			sscanf( tmp + strlen( "VmStk:" ), "%lld", &stk );
			d->stack = stk;
			continue;
		}
		if ( strspn( tmp, "VmExe:" ) == strlen( "VmExe:" ) ) {
			sscanf( tmp + strlen( "VmExe:" ), "%lld", &txt );
			d->text = txt;
			continue;
		}
		if ( strspn( tmp, "VmLib:" ) == strlen( "VmLib:" ) ) {
			sscanf( tmp + strlen( "VmLib:" ), "%lld", &lib );
			d->library = lib;
			continue;
		}
	}
	fclose( f );

	sprintf( fn, "/proc/%ld/statm", ( long ) getpid(  ) );
	f = fopen( fn, "r" );
	if ( f == NULL ) {
		PAPIERROR( "fopen(%s): %s\n", fn, strerror( errno ) );
		return PAPI_ESBSTR;
	}
	ret =
		fscanf( f, "%lld %lld %lld %lld %lld %lld %lld", &dum, &dum, &shr, &dum,
				&dum, &dat, &dum );
	if ( ret != 7 ) {
		PAPIERROR( "fscanf(7 items): %d\n", ret );
		return PAPI_ESBSTR;
	}
	d->pagesize = getpagesize(  );
	d->shared = ( shr * d->pagesize ) / 1024;
	fclose( f );

	return PAPI_OK;
}
