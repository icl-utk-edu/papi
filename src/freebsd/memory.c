/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    memory.c
* CVS:     $Id$
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include "papi.h"
#include "papi_internal.h"

int _papi_hwd_get_memory_info(PAPI_hw_info_t *hw, int id)
{
   /* TODO */
   return PAPI_OK;
}

int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{
#if 0
   /* TODO */
   d->size = 1;
   d->resident = 2;
   d->high_water_mark = 3;
   d->shared = 4;
   d->text = 5;
   d->library = 6;
   d->heap = 7;
   d->locked = 8;
   d->stack = 9;
#endif

   d->pagesize = getpagesize();

   return PAPI_OK;
}
