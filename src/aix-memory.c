#include "papi.h"
#include <sys/systemcfg.h>


int get_memory_info( PAPI_mem_info_t * mem_info ) {
  if ( _system_configuration.tlb_attrib & (1<<30) ){
   mem_info->total_tlb_size=_system_configuration.itlb_size;
  }
  else{
   mem_info->itlb_size=_system_configuration.itlb_size;
   mem_info->itlb_assoc=_system_configuration.itlb_asc;
   mem_info->dtlb_size=_system_configuration.dtlb_size;
   mem_info->dtlb_assoc=_system_configuration.dtlb_asc;
   mem_info->total_tlb_size=_system_configuration.itlb_size+
	_system_configuration.dtlb_size;
  }
  if ( _system_configuration.cache_attrib & (1<<30) ) {
   mem_info->total_L1_size=_system_configuration.icache_size/1024;
   mem_info->L1_icache_assoc=_system_configuration.icache_asc;
   mem_info->L1_icache_linesize=_system_configuration.icache_line;
  }
  else {
   mem_info->L1_icache_size=_system_configuration.icache_size/1024;
   mem_info->L1_icache_assoc=_system_configuration.icache_asc;
   mem_info->L1_icache_linesize=_system_configuration.icache_line;
   mem_info->L1_dcache_size=_system_configuration.dcache_size/1024;
   mem_info->L1_dcache_assoc=_system_configuration.dcache_asc;
   mem_info->L1_dcache_linesize=_system_configuration.dcache_line;
   mem_info->total_L1_size=mem_info->L1_icache_size+mem_info->L1_dcache_size;
  }
 mem_info->L2_cache_size=_system_configuration.L2_cache_size/1024;
 mem_info->L2_cache_assoc=_system_configuration.L2_cache_asc;
 return PAPI_OK;
}

long _papi_hwd_get_dmem_info(int option){
}
