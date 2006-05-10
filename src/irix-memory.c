/*
* File:    irix-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
* Author:  Zhou Min
*	   min@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

#define TLB_R4  96
#define TLB_R5  96
#define TLB_R8  384
#define TLB_R10  128
#define TLB_R12  128

inventory_t *getinvent(void);
int _papi_hwd_get_memory_info(PAPI_hw_info_t * mem_info, int cpu_type)
{
   inventory_t *curr;
   long count;
   int chiptype;
   char ptype[80];
   PAPI_mh_level_t *L = mem_info->mem_hierarchy.level;
   
   count = 80;
   while ((curr = getinvent()) != NULL) {
      if ((curr->inv_class == INV_MEMORY) && (curr->inv_type == INV_DCACHE)) {
         L[0].cache[1].type = PAPI_MH_TYPE_DATA;
         L[0].cache[1].size = curr->inv_state;
         L[0].cache[1].line_size = 32;
         L[0].cache[1].num_lines = 
                 L[0].cache[1].size /
                 L[0].cache[1].line_size ;
         L[0].cache[1].associativity = 2;
      }
      if ((curr->inv_class == INV_MEMORY) && (curr->inv_type == INV_ICACHE)) {
         L[0].cache[0].type = PAPI_MH_TYPE_INST;
         L[0].cache[0].size = curr->inv_state;
         L[0].cache[0].line_size = 64;
         L[0].cache[0].num_lines = 
                 L[0].cache[0].size /
                 L[0].cache[0].line_size ;
         L[0].cache[0].associativity = 2;
      }
      if ((curr->inv_class == INV_MEMORY) && (curr->inv_type == INV_SIDCACHE))
         L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
         L[1].cache[0].size = curr->inv_state;
         L[1].cache[0].line_size = 128;
         L[1].cache[0].num_lines = 
                 L[1].cache[0].size /
                 L[1].cache[0].line_size ;
         L[1].cache[0].associativity = 2;
   }

   sysinfo(_MIPS_SI_PROCESSORS, ptype, count);
   sscanf(ptype + 1, "%d", &chiptype);
   switch (chiptype) {
   case 4000:
      L[0].tlb[0].num_entries= TLB_R4 ;
      break;
   case 5000:
      L[0].tlb[0].num_entries = TLB_R5 ;
      break;
   case 8000:
      L[0].tlb[0].num_entries = TLB_R8 ;
      break;
   case 10000:
      L[0].tlb[0].num_entries = TLB_R10 ;
      break;
   case 12000:
      L[0].tlb[0].num_entries = TLB_R12 ;
      break;
   default:
      break;
   }
   if(L[0].tlb[0].num_entries != 0)
      L[0].tlb[0].type = PAPI_MH_TYPE_UNIFIED;

   mem_info->mem_hierarchy.levels = 2;
   return PAPI_OK;
}

int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{
	/* This function has been reimplemented 
		to conform to current interface.
		It has not been tested.
		Nor has it been confirmed for completeness.
		dkt 05-10-06
	*/

   pid_t pid = getpid();
   prpsinfo_t info;
   char pfile[256];
   int fd;

   sprintf(pfile, "/proc/%05d", (int) pid);
   if ((fd = open(pfile, O_RDONLY)) < 0) {
      SUBDBG("PAPI_get_dmem_info can't open /proc/%d\n", (int) pid);
      return (PAPI_ESYS);
   }
   if (ioctl(fd, PIOCPSINFO, &info) < 0) {
      return (PAPI_ESYS);
   }
   close(fd);

   d->size = info.pr_size;
   d->resident = piinfo.pr_rssize;
   d->high_water_mark = PAPI_EINVAL;
   d->shared = PAPI_EINVAL;
   d->text = PAPI_EINVAL;
   d->library = PAPI_EINVAL;
   d->heap = PAPI_EINVAL;
   d->locked = PAPI_EINVAL;
   d->stack = PAPI_EINVAL;
   d->pagesize = getpagesize();

   return (PAPI_OK);
}
