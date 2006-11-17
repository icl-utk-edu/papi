/*
* File:    unicos-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

int _papi_hwd_get_memory_info(PAPI_hw_info_t * mem_info, int cpu)
{
   return PAPI_OK;
}

int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{
	return(PAPI_ESBSTR);
}
