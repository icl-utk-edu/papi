/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:		linux-memory.c
* Author:	Harald Servat
*			redcrash@gmail.com
* Mod:		James Ralph 
*			ralph@cs.utk.edu
*/

#include "papi.h"
#include "papi_internal.h"

#define UNREFERENCED(x) (void)x

extern int x86_cache_info( PAPI_mh_info_t *mh_info );

int 
_freebsd_get_memory_info( PAPI_hw_info_t * hw_info, int id)
{
	UNREFERENCED(id);
	UNREFERENCED(hw_info);
	return PAPI_ESBSTR;
}

int _papi_freebsd_get_dmem_info(PAPI_dmem_info_t *d)
{
  /* TODO */
	d->pagesize = getpagesize();
	return PAPI_OK;
}
 
