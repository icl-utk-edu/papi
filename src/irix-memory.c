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
#include <invent.h>
#include <sys/systeminfo.h>
#include SUBSTRATE

/* Are these values computed in bytes?
   Do they need to be divided by 32 to properly represent entries? */
#define TLB_R4  96*32
#define TLB_R5  96*32
#define TLB_R8  384*32
#define TLB_R10  128*32
#define TLB_R12  128*32

inventory_t *getinvent(void);
int get_memory_info(PAPI_hw_info_t * mem_info)
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
      }
      if ((curr->inv_class == INV_MEMORY) && (curr->inv_type == INV_ICACHE)) {
         L[0].cache[0].type = PAPI_MH_TYPE_INST;
         L[0].cache[0].size = curr->inv_state;
      }
      if ((curr->inv_class == INV_MEMORY) && (curr->inv_type == INV_SIDCACHE))
         L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
         L[1].cache[0].size = curr->inv_state;
   }

   L[0].cache[1].line_size = 32;
   L[0].cache[1].lines = L[0].cache[1].size / L[0].cache[1].line_size;
   L[0].cache[1].associativity = 2;
   L[0].cache[0].line_size = 64;
   L[0].cache[0].lines = L[0].cache[0].size / L[0].cache[0].line_size;
   L[0].cache[0].associativity = 2;

   L[1].cache[0].assoc = 2;
   L[1].cache[0].linesize = 128;
   L[1].cache[0].lines = L[1].cache[0].size / L[1].cache[0].line_size;

   sysinfo(_MIPS_SI_PROCESSORS, ptype, count);
   sscanf(ptype + 1, "%d", &chiptype);
   switch (chiptype) {
   case 4000:
      L[0].tlb[0].num_entries = TLB_R4;
      break;
   case 5000:
      L[0].tlb[0].num_entries = TLB_R5;
      break;
   case 8000:
      L[0].tlb[0].num_entries = TLB_R8;
      break;
   case 10000:
      L[0].tlb[0].num_entries = TLB_R10;
      break;
   case 12000:
      L[0].tlb[0].num_entries = TLB_R12;
      break;
   default:
      break;
/*
			mem_info->total_tlb_size = TLB_R4;
*/
   }
   if(L[0].tlb[0].num_entries != 0)
      L[0].tlb[0].type = PAPI_MH_TYPE_UNIFIED;
   return PAPI_OK;
}

long _papi_hwd_get_dmem_info(int option)
{
   pid_t pid = getpid();
   prpsinfo_t info;
   char pfile[256];
   int fd;

   sprintf(pfile, "/proc/%05d", (int) pid);
   if ((fd = open(pfile, O_RDONLY)) < 0) {
      DBG((stderr, "PAPI_get_dmem_info can't open /proc/%d\n", (int) pid));
      return (PAPI_ESYS);
   }
   if (ioctl(fd, PIOCPSINFO, &info) < 0) {
      return (PAPI_ESYS);
   }
   close(fd);
   switch (option) {
   case PAPI_GET_RESSIZE:
      return (info.pr_rssize);
   case PAPI_GET_SIZE:
      return (info.pr_size);
   default:
      return (PAPI_EINVAL);
   }
}
