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

#ifdef __LINUX__
#include <limits.h>
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

#include <stdio.h>
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
#ifdef _WIN32
   case AMD:
#else
   case PERFCTR_X86_AMD_K7:
#endif
      retval = init_amd(&hw_info->mem_hierarchy);
      break;
#ifdef __x86_64__
   case PERFCTR_X86_AMD_K8:
   case PERFCTR_X86_AMD_K8C:
      retval = init_amd(&hw_info->mem_hierarchy);
      break;
#endif
   default:
      retval = init_intel(&hw_info->mem_hierarchy);
      break;
   }

   /* Do some post-processing */
   if (retval == PAPI_OK) {
      for (i=0; i<PAPI_MAX_MEM_HIERARCHY_LEVELS; i++) {
         for (j=0; j<2; j++) {
            /* Compute the number of levels of hierarchy actually used */
            if (hw_info->mem_hierarchy.level[i].tlb[j].type != PAPI_MH_TYPE_EMPTY ||
               hw_info->mem_hierarchy.level[i].cache[j].type != PAPI_MH_TYPE_EMPTY)
               hw_info->mem_hierarchy.levels = i+1;
            /* Cache sizes were reported as KB; convert to Bytes by multipying by 2^10 */
            if (hw_info->mem_hierarchy.level[i].cache[j].size != 0)
               hw_info->mem_hierarchy.level[i].cache[j].size <<= 10;
         }
      }
   }

   /* This works only because an empty cache element is initialized to 0 */
   DBG((stderr, "Detected L1: %d L2: %d  L3: %d\n",
        hw_info->mem_hierarchy.level[0].cache[0].size + hw_info->mem_hierarchy.level[0].cache[1].size, 
        hw_info->mem_hierarchy.level[1].cache[0].size + hw_info->mem_hierarchy.level[1].cache[1].size, 
        hw_info->mem_hierarchy.level[2].cache[0].size + hw_info->mem_hierarchy.level[2].cache[1].size));
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
    */

   DBG((stderr, "Initializing AMD (K7) memory\n"));
   /* AMD level 1 cache info */
   reg_eax = 0x80000005;
   cpuid(&reg_eax, &reg_ebx, &reg_ecx, &reg_edx);

   DBG((stderr, "eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
        reg_eax, reg_ebx, reg_ecx, reg_edx));
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
   DBG((stderr, "L1 TLB info (to be over-written by L2:\n"
        "\tI-num_entries %d,  I-assoc %d\n\tD-num_entries %d,  D-assoc %d\n",
        L[0].tlb[0].num_entries, L[0].tlb[0].associativity,
        L[0].tlb[1].num_entries, L[0].tlb[1].associativity))

   /* L1 D-cache/I-cache info */

   L[0].cache[1].type = PAPI_MH_TYPE_DATA;
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
   L[0].cache[1].num_lines = ((reg_ecx & 0x0000ff00) >> 8);
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
   L[0].cache[0].num_lines = ((reg_edx & 0x0000ff00) >> 8);
   L[0].cache[0].line_size = ((reg_edx & 0x000000ff));

   reg_eax = 0x80000006;
   cpuid(&reg_eax, &reg_ebx, &reg_ecx, &reg_edx);

   DBG((stderr, "eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
        reg_eax, reg_ebx, reg_ecx, reg_edx));

   /* AMD level 2 cache info */
   L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED;
   L[1].cache[0].size = ((reg_ecx & 0xffff0000) >> 16);
   pattern = ((reg_ecx & 0x0000f000) >> 12);
   L[1].cache[0].associativity = init_amd_L2_assoc_inf(pattern);
   L[1].cache[0].num_lines = ((reg_ecx & 0x00000f00) >> 8);
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

   DBG((stderr, "Initializing Intel Memory\n"));
   /* All of Intels cache info is in 1 call to cpuid
    * however it is a table lookup :(
    */
   reg_eax = 0x2;
   cpuid(&reg_eax, &reg_ebx, &reg_ecx, &reg_edx);
   DBG((stderr, "eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
        reg_eax, reg_ebx, reg_ecx, reg_edx));

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
            DBG((stderr, "Register %d does not contain valid information (skipped)\n",
                 i));
            continue;
         }
         for (k = 0; k <= 4; k++) {
            if (i == 0 && j == 0 && k == 0) {
               value = value >> 8;
               continue;
            }
            switch ((value & 0xff)) {
            case 0x01:
               L[0].tlb[0].num_entries = 128;
               L[0].tlb[0].associativity = 4;
               break;
            case 0x02:
               L[0].tlb[0].num_entries = 8;
               L[0].tlb[0].associativity = 1;
               break;
            case 0x03:
               L[0].tlb[1].num_entries = 256;
               L[0].tlb[1].associativity = 4;
               break;
            case 0x04:
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
               L[1].cache[1].size = 96;
               L[1].cache[1].associativity = 6;
               L[1].cache[1].line_size = 64;
               break;
            case 0x22:
               L[2].cache[1].associativity = 4;
               L[2].cache[1].line_size = 64;
               L[2].cache[1].size = 512;
               break;
            case 0x23:
               L[2].cache[1].associativity = 8;
               L[2].cache[1].line_size = 64;
               L[2].cache[1].size = 1024;
               break;
            case 0x25:
               L[2].cache[1].associativity = 8;
               L[2].cache[1].line_size = 64;
               L[2].cache[1].size = 2048;
               break;
            case 0x29:
               L[2].cache[1].associativity = 8;
               L[2].cache[1].line_size = 64;
               L[2].cache[1].size = 4096;
               break;
            case 0x39:
               L[1].cache[1].associativity = 4;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 128;
               break;
            case 0x3B:
               L[1].cache[1].associativity = 2;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 128;
               break;
            case 0x3C:
               L[1].cache[1].associativity = 4;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 256;
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
               L[1].cache[1].size = 128;
               L[1].cache[1].associativity = 4;
               L[1].cache[1].line_size = 32;
               break;
            case 0x42:
               L[1].cache[1].size = 256;
               L[1].cache[1].associativity = 4;
               L[1].cache[1].line_size = 32;
               break;
            case 0x43:
               L[1].cache[1].size = 512;
               L[1].cache[1].associativity = 4;
               L[1].cache[1].line_size = 32;
               break;
            case 0x44:
               L[1].cache[1].size = 1024;
               L[1].cache[1].associativity = 4;
               L[1].cache[1].line_size = 32;
               break;
            case 0x45:
               L[1].cache[1].size = 2048;
               L[1].cache[1].associativity = 4;
               L[1].cache[1].line_size = 32;
               break;
               /* Events 0x50--0x5d: TLB size info */
               /*There is no way to determine
                * the size since the page size
                * can be 4K,2M or 4M and there
                * is no way to determine it
                * Sigh -KSL
                */
               /* I object, the size is 64, 128, 256 entries even
                * though the page size is unknown
                * Smile -smeds 
                */
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
            case 0x70:
               /* 12k-uops trace cache */
               L[0].cache[0].associativity = 8;
               L[0].cache[0].size = 12;
               L[0].cache[0].line_size = 0;
               break;
            case 0x71:
               /* 16k-uops trace cache */
               L[0].cache[0].associativity = 8;
               L[0].cache[0].size = 16;
               L[0].cache[0].line_size = 0;
               break;
            case 0x72:
               /* 32k-uops trace cache */
               L[0].cache[0].associativity = 8;
               L[0].cache[0].size = 32;
               L[0].cache[0].line_size = 0;
               break;
            case 0x77:
               /* This value is not in my copy of the Intel manual */
               /* Once again IA-64 code, will most likely have to be moved */
               /* This is sectored */
               L[0].cache[0].size = 16;
               L[0].cache[0].associativity = 4;
               L[0].cache[0].line_size = 64;
               break;
            case 0x79:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 128;
               break;
            case 0x7A:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 256;
               break;
            case 0x7B:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 512;
               break;
            case 0x7C:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 1024;
               break;
            case 0x7E:
               /* This value is not in my copy of the Intel manual */
               /* IA64 value */
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 128;
               L[1].cache[1].size = 256;
               break;
            case 0x81:
               /* This value is not in my copy of the Intel manual */
               /* This is not listed as IA64, but it might be, 
                * Perhaps it is in an errata somewhere, I found the
                * info at sandpile.org -KSL
                */
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 32;
               L[1].cache[1].size = 128;
            case 0x82:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 32;
               L[1].cache[1].size = 256;
               break;
            case 0x83:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 32;
               L[1].cache[1].size = 512;
               break;
            case 0x84:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 32;
               L[1].cache[1].size = 1024;
               break;
            case 0x85:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 32;
               L[1].cache[1].size = 2048;
               break;
            case 0x86:
               L[1].cache[1].associativity = 4;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 512;
               break;
            case 0x87:
               L[1].cache[1].associativity = 8;
               L[1].cache[1].line_size = 64;
               L[1].cache[1].size = 1024;
               break;
            case 0x88:
               /* This value is not in my copy of the Intel manual */
               /* IA64 */
               L[2].cache[1].associativity = 4;
               L[2].cache[1].line_size = 64;
               L[2].cache[1].size = 2048;
               break;
            case 0x89:
               /* This value is not in my copy of the Intel manual */
               /* IA64 */
               L[2].cache[1].associativity = 4;
               L[2].cache[1].line_size = 64;
               L[2].cache[1].size = 4096;
               break;
            case 0x8A:
               /* This value is not in my copy of the Intel manual */
               /* IA64 */
               L[2].cache[1].associativity = 4;
               L[2].cache[1].line_size = 64;
               L[2].cache[1].size = 8192;
               break;
            case 0x8D:
               /* This value is not in my copy of the Intel manual */
               /* IA64 */
               L[2].cache[1].associativity = 12;
               L[2].cache[1].line_size = 128;
               L[2].cache[1].size = 3096;
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
            case 0xb0:
               L[0].tlb[0].associativity = 4;
               L[0].tlb[0].num_entries = 512;
               break;
            case 0xb3:
               L[0].tlb[1].associativity = 4;
               L[0].tlb[1].num_entries = 512;
               break;
               /* Note, there are still various IA64 cases not mapped yet */
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
   for (i = 0; i < 3; i++) {
      if (L[i].tlb[0].num_entries) L[i].tlb[0].type = PAPI_MH_TYPE_INST;
      if (L[i].tlb[1].num_entries) L[i].tlb[1].type = PAPI_MH_TYPE_DATA;
      if (L[i].cache[0].size) L[i].cache[0].type = PAPI_MH_TYPE_INST;
      if (L[i].cache[1].size) L[i].cache[1].type = PAPI_MH_TYPE_DATA;
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
inline_static void cpuid(unsigned int *a, unsigned int *b,
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
#elif defined(__x86_64__)
inline_static void cpuid(unsigned int *eax, unsigned int *ebx,
                         unsigned int *ecx, unsigned int *edx)
{
 __asm__("cpuid":"+a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
 :
   );
}
#else
inline_static void cpuid(unsigned int *eax, unsigned int *ebx,
                         unsigned int *ecx, unsigned int *edx)
{
 __asm__("cpuid":"+a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
 :
   );
}
#endif

#ifdef _WIN32
#include <Psapi.h>
long _papi_hwd_get_dmem_info(int option)
{
   int tmp;
   HANDLE proc = GetCurrentProcess();
   PROCESS_MEMORY_COUNTERS cntr;

   GetProcessMemoryInfo(proc, &cntr, sizeof(cntr));

   tmp = getpagesize();
   if (tmp == 0) tmp = 1;

   switch (option) {
     case PAPI_GET_RESSIZE:
	return ((cntr.WorkingSetSize-cntr.PagefileUsage) / tmp);
     case PAPI_GET_SIZE:	    
	return (cntr.WorkingSetSize / tmp);
     default:
	return (PAPI_EINVAL);
   }
}
#else
long _papi_hwd_get_dmem_info(int option)
{
   pid_t pid = getpid();
   char pfile[256];
   FILE *fd;
   int tmp;
   unsigned int vsize, rss;

   sprintf(pfile, "/proc/%d/stat", pid);
   if ((fd = fopen(pfile, "r")) == NULL) {
      DBG((stderr, "PAPI_get_dmem_info can't open /proc/%d/stat\n", pid));
      return (PAPI_ESYS);
   }
   fgets(pfile, 256, fd);
   fclose(fd);

   /* Scan through the information */
   sscanf(pfile,
          "%d %s %c %d %d %d %d %d %u %u %u %u %u %d %d %d %d %d %d %d %d %d %u %u", &tmp,
          pfile, pfile, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp,
          &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &vsize, &rss);
   switch (option) {
   case PAPI_GET_RESSIZE:
      return (rss);
   case PAPI_GET_SIZE:
      tmp = getpagesize();
      if (tmp == 0)
         tmp = 1;
      return ((vsize / tmp));
   default:
      return (PAPI_EINVAL);
   }
}
#endif /* _WIN32 */

