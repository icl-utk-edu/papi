/*
 * This file perfoms the following test:  memory info
 *
 * Author: Kevin London
 *	   london@cs.utk.edu
 */
#include "papi_test.h"
extern int TESTS_QUIET; /*Declared in test_utils.c */

int main(int argc, char **argv ) {
  const PAPI_mem_info_t *meminfo = NULL;
  int retval;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval!=PAPI_VER_CURRENT) 
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  if ((meminfo = PAPI_get_memory_info()) == NULL)
	test_fail(__FILE__,__LINE__,"PAPI_get_memory_info",2);

  if ( !TESTS_QUIET ) {
     printf("Test case:  Memory Information.\n");
     printf("------------------------------------------------------------------------\n");
     printf("Total TLB size:  %d KB.\n",meminfo->total_tlb_size );
     if ( meminfo->itlb_size )
     	printf("Instruction TLB: %d KB of %d-way associative.\n",meminfo->itlb_size, meminfo->itlb_assoc );
     if ( meminfo->dtlb_size )
     	printf("Data TLB: %d KB of %d-way associative.\n",meminfo->dtlb_size, meminfo->dtlb_assoc );
     printf("Total L1 cache: %d KB.\n",meminfo->total_L1_size );
     if ( meminfo->L1_icache_size )
     	printf("Instruction L1 cache: %d KB of %d-way associative and %d KB lines.\n", meminfo->L1_icache_size,meminfo->L1_icache_assoc,meminfo->L1_icache_linesize);
     if ( meminfo->L1_dcache_size )
     	printf("Data L1 cache: %d KB of %d-way associative and %d KB lines.\n", meminfo->L1_dcache_size,meminfo->L1_dcache_assoc,meminfo->L1_dcache_linesize);
     if ( meminfo->L2_cache_size )
     	printf("L2 cache: %d KB of %d-way associative and %d KB lines.\n", meminfo->L2_cache_size, meminfo->L2_cache_assoc,meminfo->L2_cache_linesize);
     if ( meminfo->L3_cache_size )
	printf("L3 cache: %d KB of %d-way associative and %d KB lines.\n", meminfo->L3_cache_size,meminfo->L3_cache_assoc,meminfo->L3_cache_linesize);
  }
  test_pass(__FILE__,NULL,0);
  exit(1);
}
