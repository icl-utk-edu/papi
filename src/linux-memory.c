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

#ifdef _WIN32
/* Define SUBSTRATE to map to linux-perfctr.h
 * since we haven't figured out how to assign a value 
 * to a label at make inside the Windows IDE */
#define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

#include <stdio.h>
static int init_amd( PAPI_hw_info_t * mem_info );
static short int init_amd_L2_assoc_inf(unsigned short int pattern);
static int init_intel( PAPI_hw_info_t * mem_info );
inline_static void cpuid(unsigned int *, unsigned int *,
				 unsigned int *,unsigned int *);

int _papi_hwd_get_memory_info( PAPI_hw_info_t * mem_info, int cpu_type ){
  int retval = 0;

  /*
    if ( !check_cpuid() ) {
    return PAPI_ESBSTR;
    }
  */

  /* Defaults to Intel which is *probably* a safe assumption -KSL */
  switch ( cpu_type ) {
#ifdef _WIN32
  case AMD:
#else
  case PERFCTR_X86_AMD_K7:
#endif
    retval = init_amd(mem_info);
    break;
#ifdef __x86_64__
  case PERFCTR_X86_AMD_K8:
    retval = init_amd(mem_info);
    break;
#endif
  default:
    retval = init_intel(mem_info);
    break;
  }
  DBG((stderr,"Detected L1: %d L2: %d  L3: %d\n",
       mem_info->L1_size, mem_info->L2_cache_size, 
       mem_info->L3_cache_size));
  return retval;
}

/* Cache configuration for AMD AThlon/Duron */
static int init_amd( PAPI_hw_info_t * mem_info ) {
  unsigned int reg_eax,reg_ebx,reg_ecx,reg_edx;
  unsigned short int pattern;
  
  /*
   * Layout of CPU information taken from :
   * "AMD Processor Recognition Application Note", 20734W-1 November 2002 
   */

  DBG((stderr,"Initializing AMD (K7) memory\n"));
  /* AMD level 1 cache info */
  reg_eax = 0x80000005;
  cpuid(&reg_eax,&reg_ebx,&reg_ecx,&reg_edx);

  DBG((stderr,"eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
       reg_eax,reg_ebx,reg_ecx,reg_edx));
  /* TLB info in L1-cache */

  /* 2MB memory page information, 4MB pages has half the number of entris */
  /* Most people run 4k pages on Linux systems, don't they? */
  /*
   * mem_info->L1_itlb_size      = (reg_eax&0xff);
   * mem_info->L1_itlb_assoc     = ((reg_eax&0xff00)>>8);
   * mem_info->L1_dtlb_size      = ((reg_eax&0xff0000)>>16);
   * mem_info->L1_dtlb_assoc     = ((reg_eax&0xff000000)>>24);
   */

  /* 4k page information */
  mem_info->L1_itlb_size      = ((reg_ebx&0x000000ff));
  mem_info->L1_itlb_assoc     = ((reg_ebx&0x0000ff00)>>8);
  switch(mem_info->L1_itlb_assoc) {
  case 0x00: /* Reserved */
    mem_info->L1_itlb_assoc = -1;
    break;
  case 0xff:
    mem_info->L1_itlb_assoc = SHRT_MAX;
    break;
  }
  mem_info->L1_dtlb_size      = ((reg_ebx&0x00ff0000)>>16);
  mem_info->L1_dtlb_assoc     = ((reg_ebx&0xff000000)>>24);
  switch(mem_info->L1_dtlb_assoc) {
  case 0x00: /* Reserved */
    mem_info->L1_dtlb_assoc = -1;
    break;
  case 0xff:
    mem_info->L1_dtlb_assoc = SHRT_MAX;
    break;
  }
  DBG((stderr,"L1 TLB info (to be over-written by L2:\n"
       "\tI-size %d,  I-assoc %d\n\tD-size %d,  D-assoc %d\n",
       mem_info->L1_itlb_size, mem_info->L1_itlb_assoc,
       mem_info->L1_dtlb_size, mem_info->L1_dtlb_assoc))

  mem_info->L1_tlb_size = mem_info->L1_itlb_size + mem_info->L1_dtlb_size;

  /* L1 D-cache/I-cache info */

  mem_info->L1_dcache_size = ((reg_ecx&0xff000000)>>24);
  mem_info->L1_dcache_assoc= ((reg_ecx&0x00ff0000)>>16);
  switch(mem_info->L1_dcache_assoc) {
  case 0x00: /* Reserved */
    mem_info->L1_dcache_assoc = -1;
    break;
  case 0xff: /* Fully assoc. */
    mem_info->L1_dcache_assoc = SHRT_MAX;
    break;
  }
  /* Bit 15-8 is "Lines per tag" */
  mem_info->L1_dcache_lines= ((reg_ecx&0x0000ff00)>>8);
  mem_info->L1_dcache_linesize= ((reg_ecx&0x000000ff));

  mem_info->L1_icache_size = ((reg_edx&0xff000000)>>24);
  mem_info->L1_icache_assoc= ((reg_edx&0x00ff0000)>>16);
  switch(mem_info->L1_icache_assoc) {
  case 0x00: /* Reserved */
    mem_info->L1_icache_assoc = -1;
    break;
  case 0xff:
    mem_info->L1_icache_assoc = SHRT_MAX;
    break;
  }
  /* Bit 15-8 is "Lines per tag" */
  mem_info->L1_icache_lines= ((reg_edx&0x0000ff00)>>8);
  mem_info->L1_icache_linesize= ((reg_edx&0x000000ff));

  /* Why summing up these entries ? */
  mem_info->L1_size = mem_info->L1_icache_size+mem_info->L1_dcache_size;

  reg_eax = 0x80000006;
  cpuid(&reg_eax,&reg_ebx,&reg_ecx,&reg_edx);

  DBG((stderr,"eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
       reg_eax,reg_ebx,reg_ecx,reg_edx));

  /* AMD level 2 cache info */
  mem_info->L2_cache_size  = ((reg_ecx&0xffff0000)>>16);
  pattern = ((reg_ecx&0x0000f000)>>12);
  mem_info->L2_cache_assoc = init_amd_L2_assoc_inf(pattern);
  mem_info->L2_cache_lines = ((reg_ecx&0x00000f00)>>8);
  mem_info->L2_cache_linesize = ((reg_ecx&0x000000ff));

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
  mem_info->L1_dtlb_size      = ((reg_ebx&0x0fff0000)>>16);
  pattern = ((reg_ebx&0xf0000000)>>28);
  mem_info->L1_dtlb_assoc = init_amd_L2_assoc_inf(pattern);
  mem_info->L1_itlb_size      = ((reg_ebx&0x00000fff));
  pattern = ((reg_ebx&0x0000f000)>>12);
  mem_info->L1_itlb_assoc = init_amd_L2_assoc_inf(pattern);
  
  mem_info->L1_tlb_size += mem_info->L1_itlb_size + mem_info->L1_dtlb_size;
  if (!mem_info->L1_dtlb_size) { /* The L2 TLB is a unified TLB, with the size itlb_size */
    mem_info->L1_itlb_size = 0;  
  }
  

  /* AMD doesn't have Level 3 cache yet..... */
  return PAPI_OK;
}

static short int init_amd_L2_assoc_inf(unsigned short int pattern) {
  short int assoc;
  /* "AMD Processor Recognition Application Note", 20734W-1 November 2002 */
  switch(pattern) {
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
    assoc = SHRT_MAX; /* Full associativity */
    break;
  default:
    /* We've encountered a pattern marked "reserved" in my manual */
    assoc = -1;
    break;
  }
return assoc;
}

static int init_intel( PAPI_hw_info_t * mem_info ) {
  unsigned int reg_eax,reg_ebx,reg_ecx,reg_edx,value;
  int i,j,k,count;

  /*
   * "Intel® Processor Identification and the CPUID Instruction",
   * Application Note, AP-485, Nov 2002, 241618-022
   */

  DBG((stderr,"Initializing Intel Memory\n"));
  /* All of Intels cache info is in 1 call to cpuid
   * however it is a table lookup :(
   */
  reg_eax = 0x2;
  cpuid(&reg_eax,&reg_ebx,&reg_ecx,&reg_edx);
  DBG((stderr,"eax=0x%8.8x ebx=0x%8.8x ecx=0x%8.8x edx=0x%8.8x\n",
       reg_eax,reg_ebx,reg_ecx,reg_edx));

  count = (0xff&reg_eax);
  for ( j=0; j<count; j++ ) {
    for ( i=0;i<4;i++){
      if(i==0) value = reg_eax;
      else if (i==1) value = reg_ebx;
      else if (i==2) value = reg_ecx;
      else value = reg_edx;
      if(value & (1<<31)) { /* Bit 31 is 0 if information is valid */
	DBG((stderr,"Register %d does not contain valid information (skipped)\n",i));
	continue;
      }
      for (k=0;k<=4;k++){
	if(i==0&&j==0&&k==0) {
	  value=value>>8;
	  continue;
	}
	switch((value&0xff)){
	case 0x01:
	  mem_info->L1_itlb_size = 128;
	  mem_info->L1_itlb_assoc= 4;
	  break;
	case 0x02:
	  mem_info->L1_itlb_size = 8;
	  mem_info->L1_itlb_assoc= 1;
	  break;
	case 0x03:
	  mem_info->L1_dtlb_size = 256;
	  mem_info->L1_dtlb_assoc= 4;
	  break;
	case 0x04:
	  mem_info->L1_dtlb_size = 32;
	  mem_info->L1_dtlb_assoc= 4;
	  break;
	case 0x06:
	  mem_info->L1_icache_size = 8;
	  mem_info->L1_icache_assoc = 4;
	  mem_info->L1_icache_linesize = 32;
	  break;
	case 0x08:
	  mem_info->L1_icache_size = 16;
	  mem_info->L1_icache_assoc = 4;
	  mem_info->L1_icache_linesize = 32;
	  break;
	case 0x0A:
	  mem_info->L1_dcache_size = 8;
	  mem_info->L1_dcache_assoc = 2;
	  mem_info->L1_dcache_linesize = 32;
	  break;
	case 0x0C:
	  mem_info->L1_dcache_size = 16;
	  mem_info->L1_dcache_assoc = 4;
	  mem_info->L1_dcache_linesize = 32;
	  break;
	case 0x10:
	  /* This value is not in my copy of the Intel manual */
	  /* IA64 codes, can most likely be moved to the IA64 memory,
           * If we can't combine the two *Still Hoping ;) * -KSL
 	   * This is L1 data cache
           */
	  mem_info->L1_dcache_size = 16;
	  mem_info->L1_dcache_assoc = 4;
	  mem_info->L1_dcache_linesize = 32;
	  break;
	case 0x15:
	  /* This value is not in my copy of the Intel manual */
	  /* IA64 codes, can most likely be moved to the IA64 memory,
           * If we can't combine the two *Still Hoping ;) * -KSL
 	   * This is L1 instruction cache
           */
	  mem_info->L1_icache_size = 16;
	  mem_info->L1_icache_assoc = 4;
	  mem_info->L1_icache_linesize = 32;
	  break;
	case 0x1A:
	  /* This value is not in my copy of the Intel manual */
	  /* IA64 codes, can most likely be moved to the IA64 memory,
           * If we can't combine the two *Still Hoping ;) * -KSL
 	   * This is L1 instruction AND data cache
           */
	  mem_info->L2_cache_size = 96;
	  mem_info->L2_cache_assoc = 6;
	  mem_info->L2_cache_linesize = 64;
	  break;
	case 0x22:
	  mem_info->L3_cache_assoc = 4;
	  mem_info->L3_cache_linesize = 64;
	  mem_info->L3_cache_size = 512;
	  break;
	case 0x23:
	  mem_info->L3_cache_assoc = 8;
	  mem_info->L3_cache_linesize = 64;
	  mem_info->L3_cache_size = 1024;
	  break;
	case 0x25:
	  mem_info->L3_cache_assoc = 8;
	  mem_info->L3_cache_linesize = 64;
	  mem_info->L3_cache_size = 2048;
	  break;
	case 0x29:
	  mem_info->L3_cache_assoc = 8;
	  mem_info->L3_cache_linesize = 64;
	  mem_info->L3_cache_size = 4096;
	  break;
	case 0x39:
	  mem_info->L2_cache_assoc = 4;
	  mem_info->L2_cache_linesize = 64;
	  mem_info->L2_cache_size = 128;
	  break;
	case 0x3B:
	  mem_info->L2_cache_assoc = 2;
	  mem_info->L2_cache_linesize = 64;
	  mem_info->L2_cache_size = 128;
	  break;
	case 0x3C:
	  mem_info->L2_cache_assoc = 4;
	  mem_info->L2_cache_linesize = 64;
	  mem_info->L2_cache_size = 256;
	  break;
	case 0x40:
	  if(mem_info->L2_cache_size) {
	    /* We have valid L2 cache, but no L3 */
	    mem_info->L3_cache_size = 0;
	  } else {
	    /* We have no L2 cache */
	    mem_info->L2_cache_size = 0;
	  }
	  break;
	case 0x41:
	  mem_info->L2_cache_size = 128;
	  mem_info->L2_cache_assoc = 4;
	  mem_info->L2_cache_linesize = 32;
	  break;
	case 0x42:
	  mem_info->L2_cache_size = 256;
	  mem_info->L2_cache_assoc = 4;
	  mem_info->L2_cache_linesize = 32;
	  break;
	case 0x43:
	  mem_info->L2_cache_size = 512;
	  mem_info->L2_cache_assoc = 4;
	  mem_info->L2_cache_linesize = 32;
	  break;
	case 0x44:
	  mem_info->L2_cache_size = 1024;
	  mem_info->L2_cache_assoc = 4;
	  mem_info->L2_cache_linesize = 32;
	  break;
	case 0x45:
	  mem_info->L2_cache_size = 2048;
	  mem_info->L2_cache_assoc = 4;
	  mem_info->L2_cache_linesize = 32;
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
	  mem_info->L1_itlb_size=64;
	  mem_info->L1_itlb_assoc=1;
	  break;
	case 0x51:
	  mem_info->L1_itlb_size=128;
	  mem_info->L1_itlb_assoc=1;
	  break;
	case 0x52:
	  mem_info->L1_itlb_size=256;
	  mem_info->L1_itlb_assoc=1;
	  break;
	case 0x5B:
	  mem_info->L1_dtlb_size=64;
	  mem_info->L1_dtlb_assoc=1;
	  break;
	case 0x5C:
	  mem_info->L1_dtlb_size=128;
	  mem_info->L1_dtlb_assoc=1;
	  break;
	case 0x5D:
	  mem_info->L1_dtlb_size=128;
	  mem_info->L1_dtlb_assoc=1;
	  break;
	case 0x66:
	  mem_info->L1_dcache_assoc = 4;
	  mem_info->L1_dcache_linesize = 64;
	  mem_info->L1_dcache_size = 8;
	  break;
	case 0x67:
	  mem_info->L1_dcache_assoc = 4;
	  mem_info->L1_dcache_linesize = 64;
	  mem_info->L1_dcache_size = 16;
	  break;
	case 0x68:
	  mem_info->L1_dcache_assoc = 4;
	  mem_info->L1_dcache_linesize = 64;
	  mem_info->L1_dcache_size = 32;
	  break;
	case 0x70:
	  /* 12k-uops trace cache */
	  mem_info->L1_icache_assoc = 8;
	  mem_info->L1_icache_size = 12;
	  mem_info->L1_icache_linesize = 0;
	  break;
	case 0x71:
	  /* 16k-uops trace cache */
	  mem_info->L1_icache_assoc = 8;
	  mem_info->L1_icache_size = 16;
	  mem_info->L1_icache_linesize = 0;
	  break;
	case 0x72:
	  /* 32k-uops trace cache */
	  mem_info->L1_icache_assoc = 8;
	  mem_info->L1_icache_size = 32;
	  mem_info->L1_icache_linesize = 0;
	  break;
	case 0x77:
	  /* This value is not in my copy of the Intel manual */
  	  /* Once again IA-64 code, will most likely have to be moved */
	  /* This is sectored */
	  mem_info->L1_icache_size = 16;
	  mem_info->L1_icache_assoc = 4;
	  mem_info->L1_icache_linesize = 64;
	  break;
	case 0x79:
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 64;
	  mem_info->L2_cache_size = 128;
	  break;
	case 0x7A:
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 64;
	  mem_info->L2_cache_size = 256;
	  break;
	case 0x7B:
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 64;
	  mem_info->L2_cache_size = 512;
	  break;
	case 0x7C:
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 64;
	  mem_info->L2_cache_size = 1024;
	  break;
	case 0x7E:
	  /* This value is not in my copy of the Intel manual */
	  /* IA64 value */
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 128;
	  mem_info->L2_cache_size = 256;
	  break;
	case 0x81:
	  /* This value is not in my copy of the Intel manual */
          /* This is not listed as IA64, but it might be, 
	   * Perhaps it is in an errata somewhere, I found the
	   * info at sandpile.org -KSL
           */
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 32;
	  mem_info->L2_cache_size = 128;
	case 0x82:
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 32;
	  mem_info->L2_cache_size = 256;
	  break;
	case 0x83:
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 32;
	  mem_info->L2_cache_size = 512;
	  break;
	case 0x84:
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 32;
	  mem_info->L2_cache_size = 1024;
	  break;
	case 0x85:
	  mem_info->L2_cache_assoc = 8;
	  mem_info->L2_cache_linesize = 32;
	  mem_info->L2_cache_size = 2048;
	  break;
	case 0x88:
	  /* This value is not in my copy of the Intel manual */
	  /* IA64 */
	  mem_info->L3_cache_assoc = 4;
	  mem_info->L3_cache_linesize = 64;
	  mem_info->L3_cache_size = 2048;
	  break;
	case 0x89:
	  /* This value is not in my copy of the Intel manual */
	  /* IA64 */
	  mem_info->L3_cache_assoc = 4;
	  mem_info->L3_cache_linesize = 64;
	  mem_info->L3_cache_size = 4096;
	  break;
	case 0x8A:
	  /* This value is not in my copy of the Intel manual */
	  /* IA64 */
	  mem_info->L3_cache_assoc = 4;
	  mem_info->L3_cache_linesize = 64;
	  mem_info->L3_cache_size = 8192;
	  break;
	case 0x8D:
	  /* This value is not in my copy of the Intel manual */
	  /* IA64 */
	  mem_info->L3_cache_assoc = 12;
	  mem_info->L3_cache_linesize = 128;
	  mem_info->L3_cache_size = 3096;
	  break;
	  /* Note, there are still various IA64 cases not mapped yet */
	}
	value=value>>8;
      }
    }
  }
   
  /* I don't like summing this as when the cache is divided, but...  /smeds */
  mem_info->L1_size = mem_info->L1_icache_size+mem_info->L1_dcache_size;
  mem_info->L1_tlb_size = mem_info->L1_itlb_size + mem_info->L1_dtlb_size;

  return PAPI_OK;
}

/* Checks to see if cpuid exists on this processor, if
 * it doesn't it is pre pentium K6 series that we don't
 * support.
 */

static int  check_cpuid(){
  volatile unsigned long val;
#ifdef _WIN32
	__asm {
		pushfd
		pop eax
		mov ebx, eax
		xor eax, 00200000h
		push eax
		popfd
		pushfd
		pop eax
		cmp eax, ebx
		jz NO_CPUID
		mov val, 1
		jmp END
	NO_CPUID:
		mov val, 0
	END:
	}
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
		       "jmp END;"
		       "NO_CPUID:"
		       "mov $0, %0;"
		       "END:"
		       :"=r"(val));
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
		       "jmp END;"
		       "NO_CPUID:"
		       "movl $0, %0;"
		       "END:"
		       :"=r"(val));
#endif
  return (int) val;
}

#ifdef _WIN32
inline_static void cpuid(unsigned int *a, unsigned int *b,
			 unsigned int *c, unsigned int *d)
{
  volatile unsigned long tmp,tmp2,tmp3,tmp4;
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
  __asm__ ("cpuid"
	   : "+a" (*eax),"=b" (*ebx), "=c" (*ecx), "=d" (*edx)
	   :
	   );
}
#else
inline_static void cpuid(unsigned int *eax, unsigned int *ebx,
			 unsigned int *ecx, unsigned int *edx)
{
  __asm__ ("cpuid"
	   : "+a" (*eax),"=b" (*ebx), "=c" (*ecx), "=d" (*edx)
	   :
	   );
}
#endif

long _papi_hwd_get_dmem_info(int option){
  pid_t pid = getpid();
  char pfile[256];
  FILE * fd;
  int tmp;
  unsigned int vsize,rss;

  sprintf(pfile, "/proc/%d/stat", pid);
  if((fd=fopen(pfile,"r")) == NULL ) {
    DBG((stderr,"PAPI_get_dmem_info can't open /proc/%d/stat\n",pid));
    return(PAPI_ESYS);
  }
  fgets(pfile, 256, fd);
  fclose(fd);
  
  /* Scan through the information */
  sscanf(pfile,"%d %s %c %d %d %d %d %d %u %u %u %u %u %d %d %d %d %d %d %d %d %d %u %u", 
	 &tmp,pfile,pfile,&tmp,&tmp,&tmp,&tmp,&tmp,
	 &tmp,&tmp,&tmp,&tmp, &tmp,&tmp,&tmp,&tmp,
	 &tmp, &tmp,&tmp,&tmp,&tmp,&tmp, &vsize,&rss );
  switch(option){
  case PAPI_GET_RESSIZE:
    return(rss);
  case PAPI_GET_SIZE:
    tmp=getpagesize();
    if ( tmp == 0 ) tmp = 1;
    return((vsize/tmp));
  default:
    return(PAPI_EINVAL);
  }
}

