/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    linux-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

static int init_amd(PAPI_mh_info_t * mh_info);
static short int init_amd_L2_assoc_inf(unsigned short int pattern);
static int init_intel(PAPI_mh_info_t * mh_info);
inline_static void cpuid(unsigned int *, unsigned int *, unsigned int *, unsigned int *);

int _papi_hwd_get_memory_info(PAPI_hw_info_t * hw_info, int cpu_type)
{
   int i,j;
   int retval = 0;

   /*
      if ( !check_cpuid() ) {
      return PAPI_ESBSTR;
      }
    */

   /* Defaults to Intel which is *probably* a safe assumption -KSL */
   switch (cpu_type) {
	   case PERFCTR_X86_AMD_K7:
		  retval = init_amd(&hw_info->mem_hierarchy);
		  break;
#ifdef PERFCTR_X86_AMD_K8 /* this is defined in perfctr 2.5.x, ff */
	   case PERFCTR_X86_AMD_K8:
#endif
#ifdef PERFCTR_X86_AMD_K8C  /* this is defined in perfctr 2.6.x */
	   case PERFCTR_X86_AMD_K8C:
#endif
		  retval = init_amd(&hw_info->mem_hierarchy);
		  break;
	   default:
		  retval = init_intel(&hw_info->mem_hierarchy);
		  break;
   }

   /* Do some post-processing */
   if (retval == PAPI_OK) {
      for (i=0; i<PAPI_MH_MAX_LEVELS; i++) {
         for (j=0; j<2; j++) {
            /* Compute the number of levels of hierarchy actually used */
            if (hw_info->mem_hierarchy.level[i].tlb[j].type != PAPI_MH_TYPE_EMPTY ||
               hw_info->mem_hierarchy.level[i].cache[j].type != PAPI_MH_TYPE_EMPTY)
               hw_info->mem_hierarchy.levels = i+1;
            /* Cache sizes were reported as KB; convert to Bytes by multipying by 2^10 */
            if (hw_info->mem_hierarchy.level[i].cache[j].size != 0)
               hw_info->mem_hierarchy.level[i].cache[j].size <<= 10;
            /* if line_size was reported without num_lines, compute it */
             if ((hw_info->mem_hierarchy.level[i].cache[j].line_size != 0) &&
                 (hw_info->mem_hierarchy.level[i].cache[j].size != 0))
               hw_info->mem_hierarchy.level[i].cache[j].num_lines = 
                  hw_info->mem_hierarchy.level[i].cache[j].size / hw_info->mem_hierarchy.level[i].cache[j].line_size;
        }
      }
   }

   /* This works only because an empty cache element is initialized to 0 */
   SUBDBG("Detected L1: %d L2: %d  L3: %d\n",
        hw_info->mem_hierarchy.level[0].cache[0].size + hw_info->mem_hierarchy.level[0].cache[1].size, 
        hw_info->mem_hierarchy.level[1].cache[0].size + hw_info->mem_hierarchy.level[1].cache[1].size, 
        hw_info->mem_hierarchy.level[2].cache[0].size + hw_info->mem_hierarchy.level[2].cache[1].size);
   return retval;
}

/* Cache configuration for AMD AThlon/Duron */
static int init_amd(PAPI_mh_info_t * mh_info)
{
   unsigned int reg_eax, reg_ebx, reg_ecx, reg_edx;
   unsigned short int pattern;
   PAPI_mh_level_t *L = mh_info->level;
   /*
    * Layout of CPU information taken from :
    * "AMD Processor Recognition Application Note", 20734W-1 November 2002
    *
    * ****Does this properly decode Opterons (K8)? Probably not...
    * See updated #20734 Rev 3.13, December 2005 for info on K7;
    * See "CPUID Specification" #25481 Rev 2.18, January 2006 for info on K8.
    */

   SUBDBG("Initializing AMD (K7) memory\n");
   /* AMD level 1 cache info */
   reg_eax = 0x80000005;
   cpuid(&reg_eax, &reg_ebx, &reg_ecx, &reg_edx);

   SUBDBG("eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
        reg_eax, reg_ebx, reg_ecx, reg_edx);
   /* TLB info in L1-cache */

   /* 2MB memory page information, 4MB pages has half the number of entries */
   /* Most people run 4k pages on Linux systems, don't they? */
   /*
    * L[0].tlb[0].type          = PAPI_MH_TYPE_INST;
    * L[0].tlb[0].num_entries   = (reg_eax&0xff);
    * L[0].tlb[0].associativity = ((reg_eax&0xff00)>>8);
    * L[0].tlb[1].type          = PAPI_MH_TYPE_DATA;
    * L[0].tlb[1].num_entries   = ((reg_eax&0xff0000)>>16);
    * L[0].tlb[1].associativity = ((reg_eax&0xff000000)>>24);
    */

   /* 4k page information */
   L[0].tlb[0].type          = PAPI_MH_TYPE_INST;
   L[0].tlb[0].num_entries   = ((reg_ebx & 0x000000ff));
   L[0].tlb[0].associativity = ((reg_ebx & 0x0000ff00) >> 8);
   switch (L[0].tlb[0].associativity) {
   case 0x00:                  /* Reserved */
      L[0].tlb[0].associativity = -1;
      break;
   case 0xff:
      L[0].tlb[0].associativity = SHRT_MAX;
      break;
   }
   L[0].tlb[1].type          = PAPI_MH_TYPE_DATA;
   L[0].tlb[1].num_entries          = ((reg_ebx & 0x00ff0000) >> 16);
   L[0].tlb[1].associativity = ((reg_ebx & 0xff000000) >> 24);
   switch (L[0].tlb[1].associativity) {
   case 0x00:                  /* Reserved */
      L[0].tlb[1].associativity = -1;
      break;
   case 0xff:
      L[0].tlb[1].associativity = SHRT_MAX;
      break;
   }

   SUBDBG("L1 TLB info (to be over-written by L2):\n");
   SUBDBG("\tI-num_entries %d,  I-assoc %d\n\tD-num_entries %d,  D-assoc %d\n",
        L[0].tlb[0].num_entries, L[0].tlb[0].associativity,
	  L[0].tlb[1].num_entries, L[0].tlb[1].associativity);

   /* L1 D-cache/I-cache info */

   L[0].cache[1].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WB | PAPI_MH_TYPE_PSEUDO_LRU;
   L[0].cache[1].size = ((reg_ecx & 0xff000000) >> 24);
   L[0].cache[1].associativity = ((reg_ecx & 0x00ff0000) >> 16);
   switch (L[0].cache[1].associativity) {
   case 0x00:                  /* Reserved */
      L[0].cache[1].associativity = -1;
      break;
   case 0xff:                  /* Fully assoc. */
      L[0].cache[1].associativity = SHRT_MAX;
      break;
   }
   /* Bit 15-8 is "Lines per tag" */
   /* L[0].cache[1].num_lines = ((reg_ecx & 0x0000ff00) >> 8); */
   L[0].cache[1].line_size = ((reg_ecx & 0x000000ff));

   L[0].cache[0].type = PAPI_MH_TYPE_INST;
   L[0].cache[0].size = ((reg_edx & 0xff000000) >> 24);
   L[0].cache[0].associativity = ((reg_edx & 0x00ff0000) >> 16);
   switch (L[0].cache[0].associativity) {
   case 0x00:                  /* Reserved */
      L[0].cache[0].associativity = -1;
      break;
   case 0xff:
      L[0].cache[0].associativity = SHRT_MAX;
      break;
   }
   /* Bit 15-8 is "Lines per tag" */
   /* L[0].cache[0].num_lines = ((reg_edx & 0x0000ff00) >> 8); */
   L[0].cache[0].line_size = ((reg_edx & 0x000000ff));

   reg_eax = 0x80000006;
   cpuid(&reg_eax, &reg_ebx, &reg_ecx, &reg_edx);

   SUBDBG("eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
        reg_eax, reg_ebx, reg_ecx, reg_edx);

   /* AMD level 2 cache info */
   L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_PSEUDO_LRU;
   L[1].cache[0].size = ((reg_ecx & 0xffff0000) >> 16);
   pattern = ((reg_ecx & 0x0000f000) >> 12);
   L[1].cache[0].associativity = init_amd_L2_assoc_inf(pattern);
   /*   L[1].cache[0].num_lines = ((reg_ecx & 0x00000f00) >> 8); */
   L[1].cache[0].line_size = ((reg_ecx & 0x000000ff));

   /* L2 cache TLB information. This over-writes the L1 cache TLB info */

   /* 2MB memory page information, 4MB pages has half the number of entris */
   /* Most people run 4k pages on Linux systems, don't they? */
   /*
    * mem_info->dtlb_size      = ((reg_eax&0x0fff0000)>>16);
    * pattern = ((reg_eax&0xf0000000)>>28);
    * mem_info->dtlb_assoc = init_amd_L2_assoc_inf(pattern);
    * mem_info->itlb_size      = (reg_eax&0xfff);
    * pattern = ((reg_eax&0xf000)>>12);
    * mem_info->itlb_assoc = init_amd_L2_assoc_inf(pattern);
    * if (!mem_info->dtlb_size) {
    *   mem_info->total_tlb_size = mem_info->itlb_size  ; mem_info->itlb_size = 0;
    * }
    */

   /* 4k page information */
   L[0].tlb[1].type = PAPI_MH_TYPE_DATA;
   L[0].tlb[1].num_entries = ((reg_ebx & 0x0fff0000) >> 16);
   pattern = ((reg_ebx & 0xf0000000) >> 28);
   L[0].tlb[1].associativity = init_amd_L2_assoc_inf(pattern);
   L[0].tlb[0].type = PAPI_MH_TYPE_INST;
   L[0].tlb[0].num_entries = ((reg_ebx & 0x00000fff));
   pattern = ((reg_ebx & 0x0000f000) >> 12);
   L[0].tlb[0].associativity = init_amd_L2_assoc_inf(pattern);

   if (!L[0].tlb[1].num_entries) {       /* The L2 TLB is a unified TLB, with the size itlb_size */
      L[0].tlb[0].num_entries = 0;
   }


   /* AMD doesn't have Level 3 cache yet..... */
   return PAPI_OK;
}

static short int init_amd_L2_assoc_inf(unsigned short int pattern)
{
   short int assoc;
   /* "AMD Processor Recognition Application Note", 20734W-1 November 2002 */
   switch (pattern) {
   case 0x0:
      assoc = 0;
      break;
   case 0x1:
   case 0x2:
   case 0x4:
      assoc = pattern;
      break;
   case 0x6:
      assoc = 8;
      break;
   case 0x8:
      assoc = 16;
      break;
   case 0xf:
      assoc = SHRT_MAX;         /* Full associativity */
      break;
   default:
      /* We've encountered a pattern marked "reserved" in my manual */
      assoc = -1;
      break;
   }
   return assoc;
}

static int init_intel(PAPI_mh_info_t * mh_info)
{
   unsigned int reg_eax, reg_ebx, reg_ecx, reg_edx, value;
   int i, j, k, count;
   PAPI_mh_level_t *L = mh_info->level;

   /*
    * "Intel® Processor Identification and the CPUID Instruction",
    * Application Note, AP-485, Nov 2002, 241618-022
    */
   for (i = 0; i < 3; i++) {
      L[i].tlb[0].type = PAPI_MH_TYPE_EMPTY;
      L[i].tlb[0].num_entries = 0;
      L[i].tlb[0].associativity = 0;
      L[i].tlb[1].type = PAPI_MH_TYPE_EMPTY;
      L[i].tlb[1].num_entries = 0;
      L[i].tlb[1].associativity = 0;
      L[i].cache[0].type = PAPI_MH_TYPE_EMPTY;
      L[i].cache[0].associativity = 0;
      L[i].cache[0].line_size = 0;
      L[i].cache[0].size = 0;
      L[i].cache[1].type = PAPI_MH_TYPE_EMPTY;
      L[i].cache[1].associativity = 0;
      L[i].cache[1].line_size = 0;
      L[i].cache[1].size = 0;
   }

   SUBDBG("Initializing Intel Memory\n");
   /* All of Intels cache info is in 1 call to cpuid
    * however it is a table lookup :(
    */
   reg_eax = 0x2;
   cpuid(&reg_eax, &reg_ebx, &reg_ecx, &reg_edx);
   SUBDBG("eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
        reg_eax, reg_ebx, reg_ecx, reg_edx);

   count = (0xff & reg_eax);
   for (j = 0; j < count; j++) {
      for (i = 0; i < 4; i++) {
         if (i == 0)
            value = reg_eax;
         else if (i == 1)
            value = reg_ebx;
         else if (i == 2)
            value = reg_ecx;
         else
            value = reg_edx;
         if (value & (1 << 31)) {       /* Bit 31 is 0 if information is valid */
            SUBDBG("Register %d does not contain valid information (skipped)\n",
                 i);
            continue;
         }
         for (k = 0; k <= 4; k++) {
            if (i == 0 && j == 0 && k == 0) {
               value = value >> 8;
               continue;
            }
            switch ((value & 0xff)) {
            case 0x01:
               L[0].tlb[0].num_entries = 32;
               L[0].tlb[0].associativity = 4;
               break;
            case 0x02:
               L[0].tlb[0].num_entries = 2;
               L[0].tlb[0].associativity = 1;
               break;
            case 0x03:
               L[0].tlb[1].num_entries = 8;
               L[0].tlb[1].associativity = 4;
               break;
            case 0x04:
               L[0].tlb[1].num_entries = 8;
               L[0].tlb[1].associativity = 4;
               break;
            case 0x05:
               L[0].tlb[1].num_entries = 32;
               L[0].tlb[1].associativity = 4;
               break;
            case 0x06:
               L[0].cache[0].size = 8;
               L[0].cache[0].associativity = 4;
               L[0].cache[0].line_size = 32;
               break;
            case 0x08:
               L[0].cache[0].size = 16;
               L[0].cache[0].associativity = 4;
               L[0].cache[0].line_size = 32;
               break;
            case 0x0A:
               L[0].cache[1].size = 8;
               L[0].cache[1].associativity = 2;
               L[0].cache[1].line_size = 32;
               break;
            case 0x0C:
               L[0].cache[1].size = 16;
               L[0].cache[1].associativity = 4;
               L[0].cache[1].line_size = 32;
               break;
            case 0x10:
               /* This value is not in my copy of the Intel manual */
               /* IA64 codes, can most likely be moved to the IA64 memory,
                * If we can't combine the two *Still Hoping ;) * -KSL
                * This is L1 data cache
                */
               L[0].cache[1].size = 16;
               L[0].cache[1].associativity = 4;
               L[0].cache[1].line_size = 32;
               break;
            case 0x15:
               /* This value is not in my copy of the Intel manual */
               /* IA64 codes, can most likely be moved to the IA64 memory,
                * If we can't combine the two *Still Hoping ;) * -KSL
                * This is L1 instruction cache
                */
               L[0].cache[0].size = 16;
               L[0].cache[0].associativity = 4;
               L[0].cache[0].line_size = 32;
               break;
            case 0x1A:
               /* This value is not in my copy of the Intel manual */
               /* IA64 codes, can most likely be moved to the IA64 memory,
                * If we can't combine the two *Still Hoping ;) * -KSL
                * This is L1 instruction AND data cache
                */
               L[1].cache[0].size = 96;
               L[1].cache[0].associativity = 6;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x22:
               L[2].cache[0].associativity = 4;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].size = 512;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x23:
               L[2].cache[0].associativity = 8;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].size = 1024;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x25:
               L[2].cache[0].associativity = 8;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].size = 2048;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x29:
               L[2].cache[0].associativity = 8;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].size = 4096;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x2C:
	       L[0].cache[1].associativity = 8;
               L[0].cache[1].line_size = 64;
               L[0].cache[1].size = 32;
               break;
            case 0x30:
	       L[0].cache[0].associativity = 8;
               L[0].cache[0].line_size = 64;
               L[0].cache[0].size = 32;
            case 0x39:
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 128;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x3A:
               L[1].cache[0].associativity = 6;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 192;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x3B:
               L[1].cache[0].associativity = 2;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 128;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x3C:
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 256;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x3D:
               L[1].cache[0].associativity = 6;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 384;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x3E:
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 512;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x40:
               if (L[1].cache[1].size) {
                  /* We have valid L2 cache, but no L3 */
                  L[2].cache[1].size = 0;
               } else {
                  /* We have no L2 cache */
                  L[1].cache[1].size = 0;
               }
               break;
            case 0x41:
               L[1].cache[0].size = 128;
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x42:
               L[1].cache[0].size = 256;
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x43:
               L[1].cache[0].size = 512;
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x44:
               L[1].cache[0].size = 1024;
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x45:
               L[1].cache[0].size = 2048;
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x46:
               L[2].cache[0].size = 4096;
               L[2].cache[0].associativity = 4;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x47:
               L[2].cache[0].size = 8192;
               L[2].cache[0].associativity = 8;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x49:
               L[1].cache[0].size = 4096;
               L[1].cache[0].associativity = 16;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               L[2].cache[0].size = 4096;
               L[2].cache[0].associativity = 16;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x4A:
               L[2].cache[0].size = 6144;
               L[2].cache[0].associativity = 12;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x4B:
               L[2].cache[0].size = 8192;
               L[2].cache[0].associativity = 16;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x4C:
               L[2].cache[0].size = 12288;
               L[2].cache[0].associativity = 12;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x4D:
               L[2].cache[0].size = 16384;
               L[2].cache[0].associativity = 16;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x50:
               L[0].tlb[0].num_entries = 64;
               L[0].tlb[0].associativity = 1;
               break;
            case 0x51:
               L[0].tlb[0].num_entries = 128;
               L[0].tlb[0].associativity = 1;
               break;
            case 0x52:
               L[0].tlb[0].num_entries = 256;
               L[0].tlb[0].associativity = 1;
               break;
            case 0x56:
               L[0].tlb[1].num_entries = 16;
               L[0].tlb[1].associativity = 4;
               break;
            case 0x57:
               L[0].tlb[1].num_entries = 16;
               L[0].tlb[1].associativity = 4;
               break;
            case 0x5B:
               L[0].tlb[1].num_entries = 64;
               L[0].tlb[1].associativity = 1;
               break;
            case 0x5C:
               L[0].tlb[1].num_entries = 128;
               L[0].tlb[1].associativity = 1;
               break;
            case 0x5D:
               L[0].tlb[1].num_entries = 256;
               L[0].tlb[1].associativity = 1;
               break;
	    case 0x60:
	       L[0].cache[1].associativity = 8;
               L[0].cache[1].line_size = 64;
               L[0].cache[1].size = 16;
               break;
            case 0x66:
               L[0].cache[1].associativity = 4;
               L[0].cache[1].line_size = 64;
               L[0].cache[1].size = 8;
               break;
            case 0x67:
               L[0].cache[1].associativity = 4;
               L[0].cache[1].line_size = 64;
               L[0].cache[1].size = 16;
               break;
            case 0x68:
               L[0].cache[1].associativity = 4;
               L[0].cache[1].line_size = 64;
               L[0].cache[1].size = 32;
               break;
	       /* Looks to me like these trace cache values
	       (0x70 - 0x73) will overwrite L1 I-cache info.
	       Should there be another slot in the cache table
	       for them? - dkt 05/14/07*/
            case 0x70:
               /* 12k-uops trace cache */
               L[0].cache[0].associativity = 8;
               L[0].cache[0].size = 12;
               L[0].cache[0].line_size = 0;
               L[0].cache[0].type = PAPI_MH_TYPE_TRACE;
               break;
            case 0x71:
               /* 16k-uops trace cache */
               L[0].cache[0].associativity = 8;
               L[0].cache[0].size = 16;
               L[0].cache[0].line_size = 0;
               L[0].cache[0].type = PAPI_MH_TYPE_TRACE;
               break;
            case 0x72:
               /* 32k-uops trace cache */
               L[0].cache[0].associativity = 8;
               L[0].cache[0].size = 32;
               L[0].cache[0].line_size = 0;
               L[0].cache[0].type = PAPI_MH_TYPE_TRACE;
               break;
            case 0x73:
               /* 64k-uops trace cache */
               L[0].cache[0].associativity = 8;
               L[0].cache[0].size = 64;
               L[0].cache[0].line_size = 0;
               L[0].cache[0].type = PAPI_MH_TYPE_TRACE;
               break;
            case 0x77:
               /* This value is not in my copy of the Intel manual */
               /* Once again IA-64 code, will most likely have to be moved */
               /* This is sectored */
               L[0].cache[0].size = 16;
               L[0].cache[0].associativity = 4;
               L[0].cache[0].line_size = 64;
               break;
            case 0x78:
               L[1].cache[0].size = 1024;
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x79:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 128;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x7A:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 256;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x7B:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 512;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x7C:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 1024;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x7D:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 2048;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x7E:
               /* This value is not in my copy of the Intel manual */
               /* IA64 value */
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 128;
               L[1].cache[0].size = 256;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x7F:
               L[1].cache[0].associativity = 2;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 512;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x81:
               /* This value is not in my copy of the Intel manual */
               /* This is not listed as IA64, but it might be, 
                * Perhaps it is in an errata somewhere, I found the
                * info at sandpile.org -KSL
                */
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].size = 128;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
            case 0x82:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].size = 256;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x83:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].size = 512;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x84:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].size = 1024;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x85:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 32;
               L[1].cache[0].size = 2048;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x86:
               L[1].cache[0].associativity = 4;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 512;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x87:
               L[1].cache[0].associativity = 8;
               L[1].cache[0].line_size = 64;
               L[1].cache[0].size = 1024;
               L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x88:
               /* This value is not in my copy of the Intel manual */
               /* IA64 */
               L[2].cache[0].associativity = 4;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].size = 2048;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x89:
               /* This value is not in my copy of the Intel manual */
               /* IA64 */
               L[2].cache[0].associativity = 4;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].size = 4096;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x8A:
               /* This value is not in my copy of the Intel manual */
               /* IA64 */
               L[2].cache[0].associativity = 4;
               L[2].cache[0].line_size = 64;
               L[2].cache[0].size = 8192;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x8D:
               /* This value is not in my copy of the Intel manual */
               /* IA64 */
               L[2].cache[0].associativity = 12;
               L[2].cache[0].line_size = 128;
               L[2].cache[0].size = 3096;
               L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED;
               break;
            case 0x90:
               L[0].tlb[0].associativity = 1;
               L[0].tlb[0].num_entries = 64;
               break;
            case 0x96:
               L[0].tlb[1].associativity = 1;
               L[0].tlb[1].num_entries = 32;
               break;
            case 0x9b:
               L[1].tlb[1].associativity = 1;
               L[1].tlb[1].num_entries = 96;
               break;
            case 0xB0:
               L[0].tlb[0].associativity = 4;
               L[0].tlb[0].num_entries = 128;
               break;
            case 0xB1:
		/* 4MB pages @ 4 way assoc
		or 2MB pages @ 8 way assoc */
               L[0].tlb[0].associativity = 4;
               L[0].tlb[0].num_entries = 4;
               break;
            case 0xB3:
               L[0].tlb[1].associativity = 4;
               L[0].tlb[1].num_entries = 128;
               break;
            case 0xB4:
               L[0].tlb[1].associativity = 4;
               L[0].tlb[1].num_entries = 256;
               break;
               /* Note, there are still various IA64 cases not mapped yet */
               /* I think I have them all now 9/10/04 */
            }
            value = value >> 8;
         }
      }
   }
   /* Scan memory hierarchy elements to look for non-zero structures.
      If a structure is not empty, it must be marked as type DATA or type INST.
      By convention, this routine always assumes {tlb,cache}[0] is INST and
      {tlb,cache}[1] is DATA. If Intel produces a unified TLB or cache, this
      algorithm will fail.
   */
  /* There are a bunch of Unified caches, changed slightly to support this 
   * Unified should be in slot 0
   */
   for (i = 0; i < 3; i++) {
      if( L[i].tlb[0].type == PAPI_MH_TYPE_EMPTY ) {
         if (L[i].tlb[0].num_entries) L[i].tlb[0].type = PAPI_MH_TYPE_INST;
         if (L[i].tlb[1].num_entries) L[i].tlb[1].type = PAPI_MH_TYPE_DATA;
      }
      if ( L[i].cache[0].type == PAPI_MH_TYPE_EMPTY) {
         if (L[i].cache[0].size) L[i].cache[0].type = PAPI_MH_TYPE_INST;
         if (L[i].cache[1].size) L[i].cache[1].type = PAPI_MH_TYPE_DATA;
      }
   }

   return PAPI_OK;
}

/* Checks to see if cpuid exists on this processor, if
 * it doesn't it is pre pentium K6 series that we don't
 * support.
 */
#if 0
/* This routine appears to no longer be called. */
static int check_cpuid()
{
   volatile unsigned long val;
#ifdef _WIN32
   __asm {
	   pushfd
	   pop eax
	   mov ebx, eax
	   xor eax, 0x00200000
	   push eax 
	   popfd 
	   pushfd 
	   pop eax 
	   cmp eax, ebx 
	   jz NO_CPUID 
	   mov val, 1 
	   jmp END 
NO_CPUID:  mov val, 0 
END:	  }
#elif defined(__x86_64__)
   __asm__ __volatile__("pushf;"
                        "pop %%eax;"
                        "mov %%eax, %%ebx;"
                        "xor $0x00200000,%%eax;"
                        "push %%eax;"
                        "popf;"
                        "pushf;"
                        "pop %%eax;"
                        "cmp %%eax, %%ebx;"
                        "jz NO_CPUID;"
                        "mov $1, %0;"
                        "jmp END;" "NO_CPUID:" "mov $0, %0;" "END:":"=r"(val));
#else
   __asm__ __volatile__("pushfl;"
                        "pop %%eax;"
                        "movl %%eax, %%ebx;"
                        "xor $0x00200000,%%eax;"
                        "push %%eax;"
                        "popfl;"
                        "pop %%eax;"
                        "cmp %%eax, %%ebx;"
                        "jz NO_CPUID;"
                        "movl $1, %0;"
                        "jmp END;" "NO_CPUID:" "movl $0, %0;" "END:":"=r"(val));
#endif
   return (int) val;
}
#endif
#ifdef _WIN32
static void cpuid(unsigned int *a, unsigned int *b,
                         unsigned int *c, unsigned int *d)
{
   volatile unsigned long tmp, tmp2, tmp3, tmp4;
   volatile unsigned long in_tmp;

   in_tmp = *a;
   __asm {
      mov eax, in_tmp;
      cpuid;
      mov tmp, eax;
      mov tmp2, ebx;
      mov tmp3, ecx;
      mov tmp4, edx;
   }
   *a = tmp;
   *b = tmp2;
   *c = tmp3;
   *d = tmp4;
}
#else
inline_static void cpuid(unsigned int *a, unsigned int *b,
                  unsigned int *c, unsigned int *d)
{
  unsigned int op = *a;
  __asm__ __volatile__ ("movl %%ebx, %%edi\n\tcpuid\n\tmovl %%ebx, %%esi\n\tmovl %%edi, %%ebx"
       : "=a" (*a),
	     "=S" (*b),
		 "=c" (*c),
		 "=d" (*d)
       : "a" (op)
       : "%edi" );
}
#endif

/* A pointer to the following is passed to PAPI_get_dmem_info() 
	typedef struct _dmem_t {
	  long_long size;
	  long_long resident;
	  long_long high_water_mark;
	  long_long shared;
	  long_long text;
	  long_long library;
	  long_long heap;
	  long_long locked;
	  long_long stack;
	  long_long pagesize;
	} PAPI_dmem_info_t;
*/


#ifdef _WIN32
#include <Psapi.h>
int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{

   HANDLE proc = GetCurrentProcess();
   PROCESS_MEMORY_COUNTERS cntr;
   SYSTEM_INFO SystemInfo;      // system information structure  

   GetSystemInfo(&SystemInfo);
   GetProcessMemoryInfo(proc, &cntr, sizeof(cntr));

   d->pagesize = SystemInfo.dwPageSize;
   d->size = (cntr.WorkingSetSize - cntr.PagefileUsage) / SystemInfo.dwPageSize;
   d->resident = cntr.WorkingSetSize / SystemInfo.dwPageSize;
   d->high_water_mark = cntr.PeakWorkingSetSize / SystemInfo.dwPageSize;
  
   return PAPI_OK;
}

#else
#ifdef __CATAMOUNT__
int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{
	return( PAPI_EINVAL );
}
#else
//int get_dmem_info(long_long *size, long_long *resident, long_long *shared, long_long *text, long_long *library, long_long *heap, long_long *locked, long_long *stack, long_long *ps, long_long *vmhwm)
int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{
  char fn[PATH_MAX], tmp[PATH_MAX];
  FILE *f;
  int ret;
  long_long sz = 0, lck = 0, res = 0, shr = 0, stk = 0, txt = 0, dat = 0, dum = 0, lib = 0, hwm = 0;

  sprintf(fn,"/proc/%ld/status",(long)getpid());
  f = fopen(fn,"r");
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s\n",fn,strerror(errno));
      return PAPI_ESBSTR;
    }
  while (1)
    {
      if (fgets(tmp,PATH_MAX,f) == NULL)
	break;
      if (strspn(tmp,"VmSize:") == strlen("VmSize:"))
	{
	  sscanf(tmp+strlen("VmSize:"),"%lld",&sz);
	  d->size = sz;
	  continue;
	}
      if (strspn(tmp,"VmHWM:") == strlen("VmHWM:"))
	{
	  sscanf(tmp+strlen("VmHWM:"),"%lld",&hwm);
	  d->high_water_mark = hwm;
	  continue;
	}
      if (strspn(tmp,"VmLck:") == strlen("VmLck:"))
	{
	  sscanf(tmp+strlen("VmLck:"),"%lld",&lck);
	  d->locked = lck;
	  continue;
	}
      if (strspn(tmp,"VmRSS:") == strlen("VmRSS:"))
	{
	  sscanf(tmp+strlen("VmRSS:"),"%lld",&res);
	  d->resident = res;
	  continue;
	}
      if (strspn(tmp,"VmData:") == strlen("VmData:"))
	{
	  sscanf(tmp+strlen("VmData:"),"%lld",&dat);
	  d->heap = dat;
	  continue;
	}
      if (strspn(tmp,"VmStk:") == strlen("VmStk:"))
	{
	  sscanf(tmp+strlen("VmStk:"),"%lld",&stk);
	  d->stack = stk;
	  continue;
	}
      if (strspn(tmp,"VmExe:") == strlen("VmExe:"))
	{
	  sscanf(tmp+strlen("VmExe:"),"%lld",&txt);
	  d->text = txt;
	  continue;
	}
      if (strspn(tmp,"VmLib:") == strlen("VmLib:"))
	{
	  sscanf(tmp+strlen("VmLib:"),"%lld",&lib);
	  d->library = lib;
	  continue;
	}
    }
  fclose(f);

  sprintf(fn,"/proc/%ld/statm",(long)getpid());
  f = fopen(fn,"r");
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s\n",fn,strerror(errno));
      return PAPI_ESBSTR;
    }
  ret = fscanf(f,"%lld %lld %lld %lld %lld %lld %lld",&dum,&dum,&shr,&dum,&dum,&dat,&dum);
  if (ret != 7)
    {
      PAPIERROR("fscanf(7 items): %d\n",ret);
      return PAPI_ESBSTR;
    }
  d->pagesize = getpagesize();
  d->shared = (shr * d->pagesize)/1024;
  fclose(f);

  return PAPI_OK;
}

#endif /* __CATAMOUNT__ */
#endif /* _WIN32 */

