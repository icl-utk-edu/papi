/*
 * This file perfoms the following test:  memory info
 *
 * Author: Kevin London
 *         london@cs.utk.edu
 */
#include "papi_test.h"
extern int TESTS_QUIET;         /*Declared in test_utils.c */

int main(int argc, char **argv)
{
   const PAPI_hw_info_t *meminfo = NULL;
   PAPI_mh_level_t *L;
   int i,j,retval;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */
   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if ((meminfo = PAPI_get_hardware_info()) == NULL)
      test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   if (!TESTS_QUIET) {
      printf("Test case:  Memory Information.\n");
      printf
          ("------------------------------------------------------------------------\n");
      /* Extract and report the tlb and cache information */
      L = &(meminfo->mem_hierarchy.level[0]);
      /* Scan the TLB structures */
     for (i=0; i<meminfo->mem_hierarchy.levels; i++) {
         for (j=0; j<2; j++) {
            switch (L[i].tlb[j].type) {
               case PAPI_MH_TYPE_UNIFIED:
                  printf("L%d Unified TLB:", i+1);
                  break;
               case PAPI_MH_TYPE_DATA:
                  printf("L%d Data TLB:", i+1);
                  break;
               case PAPI_MH_TYPE_INST:
                  printf("L%d Instruction TLB:", i+1);
                  break;
            }
            if (L[i].tlb[j].type) {
               printf("  Number of Entries: %d;  Associativity: %d\n\n",
                  L[i].tlb[j].num_entries, L[i].tlb[j].associativity);
            }
         }
      }
      /* Scan the Cache structures */
      for (i=0; i<meminfo->mem_hierarchy.levels; i++) {
         for (j=0; j<2; j++) {
            switch (L[i].cache[j].type) {
               case PAPI_MH_TYPE_UNIFIED:
                  printf("L%d Unified Cache:\n", i+1);
                  break;
               case PAPI_MH_TYPE_DATA:
                  printf("L%d Data Cache:\n", i+1);
                  break;
               case PAPI_MH_TYPE_INST:
                  printf("L%d Instruction Cache:\n", i+1);
                  break;
            }
            if (L[i].cache[j].type) {
               printf("  Total size: %dKB\n  Line size: %dB\n  Number of Lines: %d\n  Associativity: %d\n\n",
                  (L[i].cache[j].size)>>10, L[i].cache[j].line_size, L[i].cache[j].num_lines, L[i].cache[j].associativity);
            }
         }
      }
   }
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
