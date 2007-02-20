/*
* File:    perfmon.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*/

/* TODO LIST:
   - Events for all platforms
   - Derived events for all platforms
   - Latency profiling
   - BTB/IPIEAR sampling
   - Test on ITA2, Pentium 4
   - hwd_ntv_code_to_name
   - Make native map carry major events, not umasks
   - Enum event uses native_map not pfm()
   - Hook up globals to be freed to sub_info
   - Better feature bit support for IEAR
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

/* Globals declared extern elsewhere */

hwi_search_t *preset_search_map;

volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];

papi_svector_t _linux_pfm_table[] = {
  {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
  {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
  {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE},
  {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
  {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
  {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
  {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
  {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
  {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
  {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
  {(void (*)())_papi_hwd_allocate_registers,VEC_PAPI_HWD_ALLOCATE_REGISTERS},
  {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
  {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
  {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
  {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
  {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
  {(void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
  {(void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
  {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
  {(void (*)())_papi_hwd_get_memory_info, VEC_PAPI_HWD_GET_MEMORY_INFO},
  {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
  {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
  {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
  {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
  {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
  {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {NULL, VEC_PAPI_END}
};

/* Static locals */

static pfm_preset_search_entry_t *_perfmon2_pfm_preset_search_map;
static hwd_native_event_entry_t *_perfmon2_pfm_native_map;
static int _perfmon2_pfm_pmu_type;
static pfmlib_regmask_t _perfmon2_pfm_unavailable_pmcs;
static char _perfmon2_pfm_pmu_name[PAPI_MIN_STR_LEN];

/* Hardware clock functions */

#if defined(__ia64__)
inline_static unsigned long get_cycles(void)
{
   unsigned long tmp;
#ifdef HAVE_MMTIMER
   tmp = mmdev_clicks_per_tick * (*mmdev_timer_addr);
#elif defined(__INTEL_COMPILER)
   tmp = __getReg(_IA64_REG_AR_ITC);
   if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
     tmp = tmp * (unsigned long)4;
#else                           /* GCC */
   /* XXX: need more to adjust for Itanium itc bug */
   __asm__ __volatile__("mov %0=ar.itc":"=r"(tmp)::"memory");
   if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
     tmp = tmp * (unsigned long)4;
#endif
   return tmp ;
}
#elif defined(__i386__)||defined(__x86_64__)
inline_static long_long get_cycles(void) {
   long_long ret = 0;
#ifdef __x86_64__
   do {
      unsigned int a,d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      (ret) = ((long_long)a) | (((long_long)d)<<32);
   } while(0);
#else
   __asm__ __volatile__("rdtsc"
                       : "=A" (ret)
                       : );
#endif
   return ret;
}
#elif defined(mips)
inline_static long_long get_cycles(void) {
  struct timespec foo;
  double bar;
  
  syscall(__NR_clock_gettime,HAVE_CLOCK_GETTIME_REALTIME,&foo);
  SUBDBG("%lld,%lld\n",(long_long)foo.tv_sec,(long_long)foo.tv_nsec);
  bar = (double)foo.tv_nsec/1000000000.0 + (double)foo.tv_sec;
  SUBDBG("%f\n",bar);
  bar = bar * (double)_papi_hwi_system_info.hw_info.mhz * (double)1000000;
  SUBDBG("%lld\n",(long_long)bar);
  return((long_long)bar);
}
#else
#error "get_cycles undefined!"
#endif

/* The below function is stolen from libpfm from Stephane Eranian */
static int
detect_unavail_pmcs(pfmlib_regmask_t *r_pmcs, unsigned int *n_pmds)
{
  pfarg_setinfo_t	setf;
  pfmlib_regmask_t r_pmds, ipmds, ipmcs, tpmcs, tpmds;
  pfarg_ctx_t newctx;
  int ret, i, j, ctx_fd;
  unsigned int c;
  
  memset(r_pmcs, 0, sizeof(*r_pmcs));
  memset(&r_pmds, 0, sizeof(r_pmds));
  memset(&ipmds, 0, sizeof(ipmds));
  memset(&ipmcs, 0, sizeof(ipmcs));
  memset(&tpmds, 0, sizeof(tpmds));
  memset(&tpmcs, 0, sizeof(tpmcs));
  memset(&setf, 0, sizeof(setf));
  memset(&newctx, 0, sizeof(newctx));

  SUBDBG("PFM_CREATE_CONTEXT(%p,%p,%p,%d)\n",&newctx,NULL,NULL,0);
  if ((ret = pfm_create_context(&newctx, NULL, NULL, 0)) == -1)
    {
      PAPIERROR("pfm_create_context(): %s", pfm_strerror(ret));
      return(PAPI_ESYS);
    }
  SUBDBG("PFM_CREATE_CONTEXT returns fd %d\n",ret);
  ctx_fd = ret;

  if (pfm_get_impl_pmcs(&ipmcs) != PFMLIB_SUCCESS)
    return(PAPI_ESYS);

  if (pfm_get_impl_pmds(&ipmds) != PFMLIB_SUCCESS)
    return(PAPI_ESYS);
  
  if (pfm_get_num_counters(&c) != PFMLIB_SUCCESS)
    return(PAPI_ESYS);

  /*
   * retrieve available register bitmasks from set0
   * which is guaranteed to exist for every context
   */
  ret = pfm_getinfo_evtsets(ctx_fd, &setf, 1);
  if (ret == PFMLIB_SUCCESS) 
    {
      for(i=0; i < PFM_PMC_BV; i++) 
	{
	  for(j=0; j < 64; j++) 
	    {
	      if ((setf.set_avail_pmcs[i] & (1ULL << j)) == 0)
		{
		  pfm_regmask_set(&tpmcs, (i<<6)+j);
		  SUBDBG("Found unavailable PMC %d\n",(i<<6)+j);
		}
	    }
	}
      pfm_regmask_and(r_pmcs, &ipmcs, &tpmcs);
      for(i=0; i < PFM_PMC_BV; i++) 
	{
	  for(j=0; (j < 64) && ((i<<6)+j < c); j++) 
	    {
	      if ((setf.set_avail_pmds[i] & (1ULL << j)) == 0)
		{
		  pfm_regmask_set(&r_pmds, (i<<6)+j);
		  SUBDBG("Found unavailable PMD %d\n",(i<<6)+j);
		}
	    }
	}
      pfm_regmask_and(&tpmds, &ipmds, &r_pmds);
      pfm_regmask_weight(&r_pmds, n_pmds);
    }
  else
    PAPIERROR("pfm_getinfo_evtsets(%d,%p,%d): %s",ctx_fd, &setf, 1, pfm_strerror(ret));

  close(ctx_fd);
  return ret;
}

/* BEGIN COMMON CODE */

static void decode_vendor_string(char *s, int *vendor)
{
  if (strcasecmp(s,"GenuineIntel") == 0)
    *vendor = PAPI_VENDOR_INTEL;
  else if ((strcasecmp(s,"AMD") == 0) || (strcasecmp(s,"AuthenticAMD") == 0))
    *vendor = PAPI_VENDOR_AMD;
  else if (strcasecmp(s,"IBM") == 0)
    *vendor = PAPI_VENDOR_IBM;
  else if (strcasecmp(s,"MIPS") == 0)
    *vendor = PAPI_VENDOR_MIPS;
  else
    *vendor = PAPI_VENDOR_UNKNOWN;
}

static char *search_cpu_info(FILE * f, char *search_str, char *line)
{
   /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
   /* See the PCL home page for the German version of PAPI. */

   char *s;

   while (fgets(line, 256, f) != NULL) {
      if (strstr(line, search_str) != NULL) {
         /* ignore all characters in line up to : */
         for (s = line; *s && (*s != ':'); ++s);
         if (*s)
            return (s);
      }
   }
   return (NULL);

   /* End stolen code */
}

static int get_cpu_info(PAPI_hw_info_t *hw_info)
{
   int tmp, retval = PAPI_OK;
   char maxargs[PAPI_HUGE_STR_LEN], *t, *s;
   float mhz = 0.0;
   FILE *f;

   if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
     { PAPIERROR("fopen(/proc/cpuinfo) errno %d",errno); return(PAPI_ESYS); }

   /* All of this information maybe overwritten by the substrate */

   /* MHZ */

   rewind(f);
   s = search_cpu_info(f, "cpu MHz", maxargs);
   if (s)
     {
       sscanf(s + 1, "%f", &mhz);
       hw_info->mhz = mhz;
     }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "BogoMIPS", maxargs);
       if (s)
	 {
	   sscanf(s + 1, "%f", &mhz);
	   hw_info->mhz = mhz;
	 }
     }       

   /* Vendor Name and Vendor Code */

   rewind(f);
   s = search_cpu_info(f, "vendor_id", maxargs);
   if (s && (t = strchr(s + 2, '\n')))
     {
      *t = '\0';
      strcpy(hw_info->vendor_string, s + 2);
     }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) 
	 {
	   *t = '\0';
	   strcpy(hw_info->vendor_string, s + 2);
	 }
       else
	 {
	   rewind(f);
	   s = search_cpu_info(f, "system type", maxargs);
	   if (s && (t = strchr(s + 2, '\n'))) 
	     {
	       *t = '\0';
	       s = strtok(s+2," ");
	       strcpy(hw_info->vendor_string, s);
	     }
	 }
     }
   if (strlen(hw_info->vendor_string))
     decode_vendor_string(hw_info->vendor_string,
			  &hw_info->vendor);

   /* Revision */

   rewind(f);
   s = search_cpu_info(f, "stepping", maxargs);
   if (s)
      {
        sscanf(s + 1, "%d", &tmp);
        hw_info->revision = (float) tmp;
      }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "revision", maxargs);
       if (s)
         {
           sscanf(s + 1, "%d", &tmp);
           hw_info->revision = (float) tmp;
         }
     }

   /* Model Name */

   rewind(f);
   s = search_cpu_info(f, "model name", maxargs);
   if (s && (t = strchr(s + 2, '\n')))
     {
       *t = '\0';
       strcpy(hw_info->model_string, s + 2);
     }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "family", maxargs);
       if (s && (t = strchr(s + 2, '\n')))
         {
           *t = '\0';
           strcpy(hw_info->model_string, s + 2);
         }
       else
	 {
	   rewind(f);
	   s = search_cpu_info(f, "cpu model", maxargs);
	   if (s && (t = strchr(s + 2, '\n')))
	     {
	       *t = '\0';
	       s = strtok(s + 2," ");
	       s = strtok(NULL," ");
	       strcpy(hw_info->model_string, s);
	     }
	 }
     }

#if 0
   rewind(f);
   s = search_cpu_info(f, "model", maxargs);
   if (s)
      {
        sscanf(s + 1, "%d", &tmp);
        hw_info->model = tmp;
      }
#endif
   fclose(f);

   return (retval);
}

#if defined(__i386__)||defined(__x86_64__)
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

static void cpuid(unsigned int *a, unsigned int *b,
                         unsigned int *c, unsigned int *d)
{
  unsigned int op = *a;
  asm volatile ("movl %%ebx, %%edi\n\tcpuid\n\tmovl %%ebx, %%esi\n\tmovl %%edi, %%ebx"
       : "=a" (*a),
	 "=S" (*b),
	 "=c" (*c),
	 "=d" (*d)
       : "a" (op)
       : "%edi" );
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
            case 0xb0:
               L[0].tlb[0].associativity = 4;
               L[0].tlb[0].num_entries = 512;
               break;
            case 0xb3:
               L[0].tlb[1].associativity = 4;
               L[0].tlb[1].num_entries = 512;
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

/* Cache configuration for AMD AThlon/Duron */
static int init_amd(PAPI_mh_info_t * mh_info)
{
   unsigned int reg_eax, reg_ebx, reg_ecx, reg_edx;
   unsigned short int pattern;
   PAPI_mh_level_t *L = mh_info->level;
   /*
    * Layout of CPU information taken from :
    * "AMD Processor Recognition Application Note", 20734W-1 November 2002 
	* ****Does this properly decode Opterons (K8)?
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

static int x86_get_memory_info(PAPI_hw_info_t *hw_info)
{
   int retval = PAPI_OK;
   int i,j;

   /* Defaults to Intel which is *probably* a safe assumption -KSL */
   switch (hw_info->vendor) 
     {
     case PAPI_VENDOR_AMD:
       retval = init_amd(&hw_info->mem_hierarchy);
       break;
     case PAPI_VENDOR_INTEL:
       retval = init_intel(&hw_info->mem_hierarchy);
       break;
     default:
       PAPIERROR("Unknown vendor in memory information call for x86.");
       return(PAPI_ESBSTR);
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
#endif

/* 2.6.19 has this:
 VmPeak:     4588 kB
 VmSize:     4584 kB
 VmLck:         0 kB
 VmHWM:      1548 kB
 VmRSS:      1548 kB
 VmData:      312 kB
 VmStk:        88 kB
 VmExe:       684 kB
 VmLib:      1360 kB
 VmPTE:        20 kB
*/
int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{
  char fn[PATH_MAX], tmp[PATH_MAX];
  FILE *f;
  int ret;
  long_long vmpk = 0, sz = 0, lck = 0, res = 0, shr = 0, stk = 0, txt = 0, dat = 0, dum = 0, lib = 0, hwm = 0, pte = 0;

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
      if (strspn(tmp,"VmPeak:") == strlen("VmPeak:"))
	{
	  sscanf(tmp+strlen("VmPeak:"),"%lld",&vmpk);
	  d->peak = vmpk;
	  continue;
	}
      if (strspn(tmp,"VmSize:") == strlen("VmSize:"))
	{
	  sscanf(tmp+strlen("VmSize:"),"%lld",&sz);
	  d->size = sz;
	  continue;
	}
      if (strspn(tmp,"VmLck:") == strlen("VmLck:"))
	{
	  sscanf(tmp+strlen("VmLck:"),"%lld",&lck);
	  d->locked = lck;
	  continue;
	}
      if (strspn(tmp,"VmHWM:") == strlen("VmHWM:"))
	{
	  sscanf(tmp+strlen("VmHWM:"),"%lld",&hwm);
	  d->high_water_mark = hwm;
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
      if (strspn(tmp,"VmPTE:") == strlen("VmPTE:"))
	{
	  sscanf(tmp+strlen("VmPTE:"),"%lld",&pte);
	  d->pte = pte;
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
  d->pagesize = getpagesize() / 1024;
  d->shared = (shr * d->pagesize)/1024;
  fclose(f);

  return PAPI_OK;
}

#if defined(__ia64__)
static int get_number( char *buf ){
   char numbers[] = "0123456789";
   int num;
   char *tmp, *end;

   tmp = strpbrk(buf, numbers);
   if ( tmp != NULL ){
	end = tmp;
	while(isdigit(*end)) end++;
	*end='\0';
        num = atoi(tmp);
        return(num);
    }

   PAPIERROR("Number could not be parsed from %s",buf);
   return(-1);
}

static void fline ( FILE *fp, char *rline ) {
  char *tmp,*end,c;

  tmp = rline;
  end = &rline[1023];
   
  memset(rline, '\0', 1024);

  do {
    if ( feof(fp))  return;
    c = getc(fp);
  } while (isspace(c) || c == '\n' || c == '\r');

  ungetc( c, fp);

  for(;;) {
    if ( feof(fp) ) {
       return;
    }
    c = getc( fp);
    if ( c == '\n' || c == '\r' )
      break;
    *tmp++ = c;
    if ( tmp == end ) {
       *tmp = '\0';
       return;
    }
  }
  return;
} 

static int ia64_get_memory_info(PAPI_hw_info_t *hw_info)
{
   int retval = 0;
   FILE *f;
   int clevel = 0, cindex = -1;
   char buf[1024];
   int num, i, j;
   PAPI_mh_info_t *meminfo = &hw_info->mem_hierarchy;
   PAPI_mh_level_t *L = hw_info->mem_hierarchy.level;

   f = fopen("/proc/pal/cpu0/cache_info","r");

   if (!f)
     { PAPIERROR("fopen(/proc/pal/cpu0/cache_info) returned < 0"); return(PAPI_ESYS); }

   while (!feof(f)) {
      fline(f, buf);
      if ( buf[0] == '\0' ) break;
      if (  !strncmp(buf, "Data Cache", 10) ) {
         cindex = 1;
         clevel = get_number( buf );
         L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_DATA;
      }
      else if ( !strncmp(buf, "Instruction Cache", 17) ) {
         cindex = 0;
         clevel = get_number( buf );
         L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_INST;
      }
      else if ( !strncmp(buf, "Data/Instruction Cache", 22)) {
         cindex = 0;
         clevel = get_number( buf );
         L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_UNIFIED;
      }
      else {
         if ( (clevel == 0 || clevel > 3) && cindex >= 0)
	   { PAPIERROR("Cache type could not be recognized, please send /proc/pal/cpu0/cache_info"); return(PAPI_EBUG); }

         if ( !strncmp(buf, "Size", 4) ) {
            num = get_number( buf );
            L[clevel - 1].cache[cindex].size = num;
         }
         else if ( !strncmp(buf, "Associativity", 13) ) {
            num = get_number( buf );
            L[clevel - 1].cache[cindex].associativity = num;
         }
         else if ( !strncmp(buf, "Line size", 9) ) {
            num = get_number( buf );
            L[clevel - 1].cache[cindex].line_size = num;
            L[clevel - 1].cache[cindex].num_lines = L[clevel - 1].cache[cindex].size/num;
         }
      }
   } 

   fclose(f);

   f = fopen("/proc/pal/cpu0/vm_info","r");
   /* No errors on fopen as I am not sure this is always on the systems */
   if ( f != NULL ) {
      cindex = -1;
      clevel = 0;
      while (!feof(f)) {
         fline(f, buf);
         if ( buf[0] == '\0' ) break;
         if (  !strncmp(buf, "Data Translation", 16) ) {
            cindex = 1;
	         clevel = get_number( buf );
            L[clevel - 1].tlb[cindex].type = PAPI_MH_TYPE_DATA;
         }
         else if ( !strncmp(buf, "Instruction Translation", 23) ){
            cindex = 0;
            clevel = get_number( buf );
            L[clevel - 1].tlb[cindex].type = PAPI_MH_TYPE_INST;
         }
         else {
	         if ( (clevel == 0 || clevel > 2) && cindex >= 0)
		   { PAPIERROR("TLB type could not be recognized, send /proc/pal/cpu0/vm_info"); return(PAPI_EBUG); }

	         if ( !strncmp(buf, "Number of entries", 17) ){
	            num = get_number( buf );
               L[clevel - 1].tlb[cindex].num_entries = num;
	         }
  	         else if ( !strncmp(buf, "Associativity", 13) ) {
	            num = get_number( buf );
               L[clevel - 1].tlb[cindex].associativity = num;
	         }
         }
      } 
      fclose(f);
   }

   /* Compute and store the number of levels of hierarchy actually used */
   for (i=0; i<PAPI_MH_MAX_LEVELS; i++) {
      for (j=0; j<2; j++) {
         if (L[i].tlb[j].type != PAPI_MH_TYPE_EMPTY ||
            L[i].cache[j].type != PAPI_MH_TYPE_EMPTY)
            meminfo->levels = i+1;
      }
   }
   return retval;
}
#endif

#if defined(mips)
/* system type             : MIPS Malta
processor               : 0
cpu model               : MIPS 20Kc V2.0  FPU V2.0
BogoMIPS                : 478.20
wait instruction        : no
microsecond timers      : yes
tlb_entries             : 48 64K pages
icache size             : 32K sets 256 ways 4 linesize 32
dcache size             : 32K sets 256 ways 4 linesize 32
scache....
default cache policy    : cached write-back
extra interrupt vector  : yes
hardware watchpoint     : yes
ASEs implemented        : mips3d
VCED exceptions         : not available
VCEI exceptions         : not available
*/

static int mips_get_cache(char *entry, int *sizeB, int *assoc, int *lineB)
{
  int retval, dummy;

  retval = sscanf(entry,"%dK sets %d ways %d linesize %d",sizeB,&dummy,assoc,lineB);
  *sizeB *= 1024;

  if (retval != 4)
    PAPIERROR("Could not get 4 integers from %s\nPlease send this line to ptools-perfapi@cs.utk.edu",entry);

  SUBDBG("Got cache %d, %d, %d\n",*sizeB,*assoc,*lineB);      
  return(PAPI_OK);
}

static int mips_get_policy(char *s, int *cached, int *policy)
{
  if (strstr(s,"cached"))
    *cached = 1;
  if (strstr(s,"write-back"))
    *policy = PAPI_MH_TYPE_WB | PAPI_MH_TYPE_LRU;
  if (strstr(s,"write-through"))
    *policy = PAPI_MH_TYPE_WT | PAPI_MH_TYPE_LRU;

  if (*policy == 0)
    PAPIERROR("Could not get cache policy from %s\nPlease send this line to ptools-perfapi@cs.utk.edu",s);

  SUBDBG("Got policy 0x%x, cached 0x%x\n",*policy,*cached);
  return(PAPI_OK);
}

static int mips_get_tlb(char *s, int *u, int *size2)
{
  int retval;
  
  retval = sscanf(s,"%d %dK",u,size2);
  *size2 *= 1024;

  if (retval <= 0)
    PAPIERROR("Could not get tlb entries from %s\nPlease send this line to ptools-perfapi@cs.utk.edu",s);
  else if (retval >= 1)
    {
      if (*size2 == 0)
	*size2 = getpagesize();
    }
  SUBDBG("Got tlb %d %d pages\n",*u,*size2);
  return(PAPI_OK);
}

static int mips_get_memory_info(PAPI_hw_info_t *hw_info)
{
  char *s;
  int retval = PAPI_OK;
  int i = 0, cached = 0, policy = 0, num = 0, pagesize = 0, maxlevel = 0;
  int sizeB, assoc, lineB;
  char maxargs[PAPI_HUGE_STR_LEN];
  PAPI_mh_info_t *mh_info = &hw_info->mem_hierarchy;

   FILE *f;

   if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
     { PAPIERROR("fopen(/proc/cpuinfo) errno %d",errno); return(PAPI_ESYS); }

   /* All of this information maybe overwritten by the substrate */

   /* MHZ */

   rewind(f);
   s = search_cpu_info(f, "default cache policy", maxargs);
   if (s && strlen(s))
     {
       mips_get_policy(s+2,&cached,&policy);
       if (cached == 0)
	 {
	   SUBDBG("Uncached default policy detected, reporting zero cache entries.\n");
	   goto nocache;
	 }
     }
   else
     {
       PAPIERROR("Could not locate 'default cache policy' in /proc/cpuinfo\nPlease send the contents of this file to ptools-perfapi@cs.utk.edu");
     }

   rewind(f);
   s = search_cpu_info(f, "icache size", maxargs);
   if (s)
     {
       mips_get_cache(s + 2, &sizeB, &assoc, &lineB);
       mh_info->level[0].cache[i].size = sizeB;
       mh_info->level[0].cache[i].line_size = lineB;
       mh_info->level[0].cache[i].num_lines = sizeB / lineB;
       mh_info->level[0].cache[i].associativity = assoc;
       mh_info->level[0].cache[i].type = PAPI_MH_TYPE_INST | policy;
       i++;
       if (!maxlevel) maxlevel++;
     }
   else
     {
       PAPIERROR("Could not locate 'icache size' in /proc/cpuinfo\nPlease send the contents of this file to ptools-perfapi@cs.utk.edu");
     }       

   rewind(f);
   s = search_cpu_info(f, "dcache size", maxargs);
   if (s)
     {
       mips_get_cache(s + 2, &sizeB, &assoc, &lineB);
       mh_info->level[0].cache[i].size = sizeB;
       mh_info->level[0].cache[i].line_size = lineB;
       mh_info->level[0].cache[i].num_lines = sizeB / lineB;
       mh_info->level[0].cache[i].associativity = assoc;
       mh_info->level[0].cache[i].type = PAPI_MH_TYPE_DATA | policy;
       i++;
       if (!maxlevel) maxlevel++;
     }
   else
     {
       PAPIERROR("Could not locate 'dcache size' in /proc/cpuinfo\nPlease send the contents of this file to ptools-perfapi@cs.utk.edu");
     }       

   rewind(f);
   s = search_cpu_info(f, "scache size", maxargs);
   if (s)
     {
       mips_get_cache(s + 2, &sizeB, &assoc, &lineB);
       mh_info->level[1].cache[0].size = sizeB;
       mh_info->level[1].cache[0].line_size = lineB;
       mh_info->level[1].cache[0].num_lines = sizeB / lineB;
       mh_info->level[1].cache[0].associativity = assoc;
       mh_info->level[1].cache[0].type = PAPI_MH_TYPE_UNIFIED | policy;
       maxlevel++;
     }
   else
     {
       /* Hey, it's ok not to have an L2. Slow, but ok. */
     }       


   /* Currently only reports on the JTLB. This is in-fact missing the dual 4-entry uTLB
      that only works on systems with 4K pages. */

 nocache:

   rewind(f);
   s = search_cpu_info(f, "tlb_entries", maxargs);
   if (s && strlen(s))
     {
       mips_get_tlb(s+2,&num,&pagesize);
       mh_info->level[0].tlb[0].num_entries = num;
       mh_info->level[0].tlb[0].associativity = num;
       mh_info->level[0].tlb[0].type = PAPI_MH_TYPE_UNIFIED;
       if (!maxlevel) maxlevel++;
     }
   else
     {
       PAPIERROR("Could not locate 'tlb_entries' in /proc/cpuinfo\nPlease send the contents of this file to ptools-perfapi@cs.utk.edu");
     }

   fclose(f);

   mh_info->levels = maxlevel;
   return retval;
}
#endif

int _papi_hwd_get_memory_info(PAPI_hw_info_t * hwinfo, int unused)
{
  int retval = PAPI_OK;

#if defined(mips)
  mips_get_memory_info(hwinfo);
#elif defined(__i386__)||defined(__x86_64__)
  x86_get_memory_info(hwinfo);
#elif defined(__ia64__)
  ia64_get_memory_info(hwinfo);
#else
#error "No support for this architecture. Please modify perfmon.c"
#endif

  return(retval);
}

#ifdef DEBUG
void dump_smpl_arg(pfm_dfl_smpl_arg_t *arg)
{
  SUBDBG("SMPL_ARG.buf_size = %llu\n",(unsigned long long)arg->buf_size);
  SUBDBG("SMPL_ARG.buf_flags = %d\n",arg->buf_flags);
}

void dump_sets(pfarg_setdesc_t *set, int num_sets)
{
  int i;

  for (i=0;i<num_sets;i++)
    {
      SUBDBG("SET[%d]\n",i);
      SUBDBG("SET[%d].set_id = %d\n",i,set[i].set_id);
      SUBDBG("SET[%d].set_id_next = %d\n",i,set[i].set_id_next);
      SUBDBG("SET[%d].set_flags = %d\n",i,set[i].set_flags);
      SUBDBG("SET[%d].set_timeout = %llu\n",i,(unsigned long long)set[i].set_timeout);
      SUBDBG("SET[%d].set_mmap_offset = 0x%016llx\n",i,(unsigned long long)set[i].set_mmap_offset);
    }
}

void dump_setinfo(hwd_control_state_t * ctl)
{
  int i;
  pfarg_setinfo_t *setinfo = ctl->setinfo;

  for (i=0;i<ctl->num_sets;i++)
    {
      SUBDBG("SETINFO[%d]\n",i);
      SUBDBG("SETINFO[%d].set_id = %d\n",i,setinfo[i].set_id);
      SUBDBG("SETINFO[%d].set_id_next = %d\n",i,setinfo[i].set_id_next);
      SUBDBG("SETINFO[%d].set_flags = %d\n",i,setinfo[i].set_flags);
      SUBDBG("SETINFO[%d].set_ovfl_pmds[0] = 0x%016llx\n",i,(unsigned long long)setinfo[i].set_ovfl_pmds[0]);
      SUBDBG("SETINFO[%d].set_runs = %llu\n",i,(unsigned long long)setinfo[i].set_runs);
      SUBDBG("SETINFO[%d].set_timeout = %llu\n",i,(unsigned long long)setinfo[i].set_timeout);
      SUBDBG("SETINFO[%d].set_act_duration = %llu\n",i,(unsigned long long)setinfo[i].set_act_duration);
      SUBDBG("SETINFO[%d].set_mmap_offset = 0x%016llx\n",i,(unsigned long long)setinfo[i].set_mmap_offset);
      SUBDBG("SETINFO[%d].set_avail_pmcs[0] = 0x%016llx\n",i,(unsigned long long)setinfo[i].set_avail_pmcs[0]);
      SUBDBG("SETINFO[%d].set_avail_pmds[0] = 0x%016llx\n",i,(unsigned long long)setinfo[i].set_avail_pmds[0]);
    }
}

void dump_pmc(hwd_control_state_t * ctl)
{
  int i;
  pfarg_pmc_t *pc = ctl->pc;

  for (i=0;i<ctl->out.pfp_pmc_count;i++)
    {
      SUBDBG("PC[%d]\n",i);
      SUBDBG("PC[%d].reg_num = %d\n",i,pc[i].reg_num);
      SUBDBG("PC[%d].reg_set = %d\n",i,pc[i].reg_set);
      SUBDBG("PC[%d].reg_flags = 0x%08x\n",i,pc[i].reg_flags);
      SUBDBG("PC[%d].reg_value = 0x%016llx\n",i,(unsigned long long)pc[i].reg_value);
    }
}

void dump_pmd(hwd_control_state_t * ctl)
{
  int i;
  pfarg_pmd_t *pd = ctl->pd;

  for (i=0;i<ctl->in.pfp_event_count;i++)
    {
      SUBDBG("PD[%d]\n",i);
      SUBDBG("PD[%d].reg_num = %d\n",i,pd[i].reg_num);
      SUBDBG("PD[%d].reg_set = %d\n",i,pd[i].reg_set);
      SUBDBG("PD[%d].reg_flags = 0x%08x\n",i,pd[i].reg_flags);
      SUBDBG("PD[%d].reg_value = 0x%016llx\n",i,(unsigned long long)pd[i].reg_value);
      SUBDBG("PD[%d].reg_long_reset = %llu\n",i,(unsigned long long)pd[i].reg_long_reset);
      SUBDBG("PD[%d].reg_short_reset = %llu\n",i,(unsigned long long)pd[i].reg_short_reset);
      SUBDBG("PD[%d].reg_last_reset_val = %llu\n",i,(unsigned long long)pd[i].reg_last_reset_val);
      SUBDBG("PD[%d].reg_ovfl_switch_cnt = %llu\n",i,(unsigned long long)pd[i].reg_ovfl_switch_cnt);
      SUBDBG("PD[%d].reg_reset_pmds[0] = 0x%016llx\n",i,(unsigned long long)pd[i].reg_reset_pmds[0]);
      SUBDBG("PD[%d].reg_smpl_pmds[0] = 0x%016llx\n",i,(unsigned long long)pd[i].reg_smpl_pmds[0]);
      SUBDBG("PD[%d].reg_smpl_eventid = %llu\n",i,(unsigned long long)pd[i].reg_smpl_eventid);
      SUBDBG("PD[%d].reg_random_mask = %llu\n",i,(unsigned long long)pd[i].reg_random_mask);
      SUBDBG("PD[%d].reg_random_seed = %d\n",i,pd[i].reg_random_seed);
    }
}

void dump_smpl_hdr(pfm_dfl_smpl_hdr_t *hdr)
{
  SUBDBG("SMPL_HDR.hdr_count = %llu\n",(unsigned long long)hdr->hdr_count);
  SUBDBG("SMPL_HDR.hdr_cur_offs = %llu\n",(unsigned long long)hdr->hdr_cur_offs);
  SUBDBG("SMPL_HDR.hdr_overflows = %llu\n",(unsigned long long)hdr->hdr_overflows);
  SUBDBG("SMPL_HDR.hdr_buf_size = %llu\n",(unsigned long long)hdr->hdr_buf_size);
  SUBDBG("SMPL_HDR.hdr_min_buf_space = %llu\n",(unsigned long long)hdr->hdr_min_buf_space);
  SUBDBG("SMPL_HDR.hdr_version = %d\n",hdr->hdr_version);
  SUBDBG("SMPL_HDR.hdr_buf_flags = %d\n",hdr->hdr_buf_flags);
}

void dump_smpl(pfm_dfl_smpl_entry_t *entry)
{
  SUBDBG("SMPL.pid = %d\n",entry->pid);
  SUBDBG("SMPL.ovfl_pmd = %d\n",entry->ovfl_pmd);
  SUBDBG("SMPL.last_reset_val = %llu\n",(unsigned long long)entry->last_reset_val);
  SUBDBG("SMPL.ip = 0x%llx\n",(unsigned long long)entry->ip);
  SUBDBG("SMPL.tstamp = %llu\n",(unsigned long long)entry->tstamp);
  SUBDBG("SMPL.cpu = %d\n",entry->cpu);
  SUBDBG("SMPL.set = %d\n",entry->set);
  SUBDBG("SMPL.tgid = %d\n",entry->tgid);
}
#endif

int _papi_hwd_update_shlib_info(void)
{
   char fname[PAPI_HUGE_STR_LEN];
   unsigned long t_index = 0, d_index = 0, b_index = 0, counting = 1;
   PAPI_address_map_t *tmp = NULL;
   FILE *f;
                                                                                
   sprintf(fname, "/proc/%ld/maps", (long) _papi_hwi_system_info.pid);
   f = fopen(fname, "r");
                                                                                
   if (!f)
     {
         PAPIERROR("fopen(%s) returned < 0", fname);
         return(PAPI_OK);
     }
                                                                                
 again:
   while (!feof(f)) {
      char buf[PAPI_HUGE_STR_LEN+PAPI_HUGE_STR_LEN], perm[5], dev[6], mapname[PATH_MAX], lastmapname[PAPI_HUGE_STR_LEN];
      unsigned long begin, end, size, inode, foo;
                                                                                
      if (fgets(buf, sizeof(buf), f) == 0)
         break;
      if (strlen(mapname))
        strcpy(lastmapname,mapname);
      else
        lastmapname[0] = '\0';
      mapname[0] = '\0';
      sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm,
             &foo, dev, &inode, mapname);
      size = end - begin;
                                                                                
      /* the permission string looks like "rwxp", where each character can
       * be either the letter, or a hyphen.  The final character is either
       * p for private or s for shared. */
                                                                                
      if (counting)
        {
          if ((perm[2] == 'x') && (perm[0] == 'r') && (inode != 0))
            {
              if  (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) == 0)                {
                  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) begin;
                  _papi_hwi_system_info.exe_info.address_info.text_end =
                    (caddr_t) (begin + size);
                }
              t_index++;
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0) && (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) == 0))
            {
              _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) begin;
              _papi_hwi_system_info.exe_info.address_info.data_end =
                (caddr_t) (begin + size);
              d_index++;
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0) && (strcmp(_papi_hwi_system_info.exe_info.fullname,lastmapname) == 0))
            {
              _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) begin;
              _papi_hwi_system_info.exe_info.address_info.bss_end =
                (caddr_t) (begin + size);
              b_index++;
            }
        }
      else if (!counting)
        {
          if ((perm[2] == 'x') && (perm[0] == 'r') && (inode != 0))
            {
              if (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) != 0)
                {
              t_index++;
                  tmp[t_index-1 ].text_start = (caddr_t) begin;
                  tmp[t_index-1 ].text_end = (caddr_t) (begin + size);
                  strncpy(tmp[t_index-1 ].name, mapname, PAPI_MAX_STR_LEN);
                }
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode != 0))
            {
              if ( (strcmp(_papi_hwi_system_info.exe_info.fullname,mapname) != 0)
               && (t_index >0 ) && (tmp[t_index-1 ].data_start == 0))
                {
                  tmp[t_index-1 ].data_start = (caddr_t) begin;
                  tmp[t_index-1 ].data_end = (caddr_t) (begin + size);
                }
            }
          else if ((perm[0] == 'r') && (perm[1] == 'w') && (inode == 0))
            {
              if ((t_index > 0 ) && (tmp[t_index-1].bss_start == 0))
                {
                  tmp[t_index-1].bss_start = (caddr_t) begin;
                  tmp[t_index-1].bss_end = (caddr_t) (begin + size);
                }
            }
        }
   }
                                                                                
   if (counting) {
      /* When we get here, we have counted the number of entries in the map
         for us to allocate */
                                                                                
      tmp = (PAPI_address_map_t *) papi_calloc(t_index-1, sizeof(PAPI_address_map_t));
      if (tmp == NULL)
        { PAPIERROR("Error allocating shared library address map"); return(PAPI_ENOMEM); }
      t_index = 0;
      rewind(f);
      counting = 0;
      goto again;
   } else {
      if (_papi_hwi_system_info.shlib_info.map)
         papi_free(_papi_hwi_system_info.shlib_info.map);
      _papi_hwi_system_info.shlib_info.map = tmp;
      _papi_hwi_system_info.shlib_info.count = t_index;
                                                                                
      fclose(f);
   }
   return (PAPI_OK);
}
                                                                                
static int get_system_info(papi_mdi_t *mdi)
{
   int retval;
   char maxargs[PAPI_HUGE_STR_LEN];
   pid_t pid;

   /* Software info */

   /* Path and args */

   pid = getpid();
   if (pid < 0)
     { PAPIERROR("getpid() returned < 0"); return(PAPI_ESYS); }
   mdi->pid = pid;

   sprintf(maxargs, "/proc/%d/exe", (int) pid);
   if (readlink(maxargs, mdi->exe_info.fullname, PAPI_HUGE_STR_LEN) < 0)
     { PAPIERROR("readlink(%s) returned < 0", maxargs); return(PAPI_ESYS); }

   /* Careful, basename can modify it's argument */

   strcpy(maxargs,mdi->exe_info.fullname);
   strcpy(mdi->exe_info.address_info.name, basename(maxargs));
   SUBDBG("Executable is %s\n", mdi->exe_info.address_info.name);
   SUBDBG("Full Executable is %s\n", mdi->exe_info.fullname);

   /* Executable regions, may require reading /proc/pid/maps file */

   retval = _papi_hwd_update_shlib_info();
   SUBDBG("Text: Start %p, End %p, length %d\n",
          mdi->exe_info.address_info.text_start,
          mdi->exe_info.address_info.text_end,
          (int)(mdi->exe_info.address_info.text_end -
          mdi->exe_info.address_info.text_start));
   SUBDBG("Data: Start %p, End %p, length %d\n",
          mdi->exe_info.address_info.data_start,
          mdi->exe_info.address_info.data_end,
          (int)(mdi->exe_info.address_info.data_end -
          mdi->exe_info.address_info.data_start));
   SUBDBG("Bss: Start %p, End %p, length %d\n",
          mdi->exe_info.address_info.bss_start,
          mdi->exe_info.address_info.bss_end,
          (int)(mdi->exe_info.address_info.bss_end -
          mdi->exe_info.address_info.bss_start));

   /* PAPI_preload_option information */

   strcpy(mdi->preload_info.lib_preload_env, "LD_PRELOAD");
   mdi->preload_info.lib_preload_sep = ' ';
   strcpy(mdi->preload_info.lib_dir_env, "LD_LIBRARY_PATH");
   mdi->preload_info.lib_dir_sep = ':';

   /* Hardware info */

  mdi->hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  mdi->hw_info.nnodes = 1;
  mdi->hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);

  retval = get_cpu_info(&mdi->hw_info);
  if (retval)
    return(retval);

  retval = _papi_hwd_get_memory_info(&mdi->hw_info,mdi->hw_info.model);
  if (retval)
    return(retval);

   SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
          mdi->hw_info.totalcpus,
          mdi->hw_info.vendor_string,
          mdi->hw_info.vendor,
          mdi->hw_info.model_string,
          mdi->hw_info.model,
	  mdi->hw_info.mhz);
                                                                               
   return (PAPI_OK);
}

inline_static pid_t mygettid(void)
{
#ifdef SYS_gettid
  return(syscall(SYS_gettid));
#elif defined(__NR_gettid)
  return(syscall(__NR_gettid));
#else
  return(syscall(1105));  
#endif
}

#define PAPI_NATIVE_EVENT_AND_MASK 0x0000ffff
#define PAPI_NATIVE_EVENT_SHIFT 0
#define PAPI_NATIVE_UMASK_AND_MASK 0x00ff0000
#define PAPI_NATIVE_UMASK_SHIFT 16

static inline int encode_native_event_raw(unsigned int event, unsigned int mask)
{
  unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
  SUBDBG("Old native index was 0x%08x with 0x%08x mask\n",tmp,mask);
  tmp = tmp | (mask << PAPI_NATIVE_UMASK_SHIFT);
  SUBDBG("New encoding is 0x%08x\n",tmp|PAPI_NATIVE_MASK);
  return(tmp|PAPI_NATIVE_MASK);
}

static inline int encode_native_event(unsigned int event, unsigned int num_mask, unsigned int *mask_values)
{
  unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
  SUBDBG("Old native index was 0x%08x with %d masks\n",tmp,num_mask);
  if (num_mask)
    tmp = tmp | (((1 << num_mask) - 1) << PAPI_NATIVE_UMASK_SHIFT);
  SUBDBG("New encoding is 0x%08x\n",tmp|PAPI_NATIVE_MASK);
  return(tmp|PAPI_NATIVE_MASK);
}

static int setup_preset(hwi_search_t *entry, hwi_dev_notes_t *note_entry, pfmlib_event_t *event, unsigned int preset, unsigned int derived, int cntrs)
{
  pfmlib_regmask_t impl_cnt, evnt_cnt;
  char *findme;
  int n, j, ret, did = 0;

  /* find out which counters it lives on */

  if ((ret = pfm_get_event_counters(event->event,&evnt_cnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_event_counters(%d,%p): %s",event->event,&evnt_cnt,pfm_strerror(ret));
      return(PAPI_EBUG);
    }
  if ((ret = pfm_get_impl_counters(&impl_cnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_impl_counters(%p): %s", &impl_cnt, pfm_strerror(ret));
      return(PAPI_EBUG);
    }

  /* Initialize the data.native array to NULL because braindead
     middle PAPI code checks this value. */

  for (j=0;j<PFMLIB_MAX_PMCS;j++)
    entry->data.native[j] = PAPI_NULL;

  /* Make sure this event lives on some counter, if so, put in the description. If not, BUG */
  n = cntrs;
  for (j=0;n;j++)
    {
      if (pfm_regmask_isset(&impl_cnt, j))
	{
	  n--;
	  if (pfm_regmask_isset(&evnt_cnt,j))
	    {
	      entry->data.native[did] = encode_native_event(event->event,event->num_masks,event->unit_masks);
	      if ((ret = pfm_get_event_description(event->event,&findme)) != PFMLIB_SUCCESS)
		{
		  PAPIERROR("pfm_get_event_description(%d,%p): %s",event,&findme,pfm_strerror(ret));
		}
	      else 
		{
		  if ((findme) && strlen(findme))
		    note_entry->dev_note = strdup(findme);
		  free(findme);
		}
	      entry->event_code = preset;
	      entry->data.derived = derived;
	      return(PAPI_OK);
	    }
	}
    }

  PAPIERROR("PAPI preset 0x%08x PFM event %d did not have any available counters");
  return(PAPI_ENOEVNT);
}

static inline char *trim_string(char *in)
{
  int len, i = 0;
  char *start = in;

  if (in == NULL)
    return(in);
  len = strlen(in);
  if (len == 0)
    return(in);
  /* Trim left */
  while (i < len)
    {
      if (isblank(in[i]))
	{
	  in[i] = '\0';
	  start++;
	}
      else
	break;
      i++;
    }
  /* Trim right */
  i = strlen(start) - 1;
  while (i >= 0)
    {
      if (isblank(start[i]))
	start[i] = '\0';
      else
	break;
      i--;
    }
  return(start);
}

static inline int find_preset_code(char *tmp, int *code)
{
  int i = 0;
  extern hwi_presets_t _papi_hwi_presets;
  while (_papi_hwi_presets.info[i].symbol != NULL)
    {
      if (strcasecmp(tmp,_papi_hwi_presets.info[i].symbol) == 0)
	{
	  *code = i|PAPI_PRESET_MASK;
	  return(PAPI_OK);
	}
      i++;
    }
  return(PAPI_EINVAL);
}

static int load_preset_table(pfm_preset_search_entry_t **here)
{
  char name[PATH_MAX], line[LINE_MAX], *tmpn;
  FILE *table;
  int line_no = 1, get_presets = 0, derived = 0, insert = 2, preset = 0;
  static pfm_preset_search_entry_t tmp[PAPI_MAX_PRESET_EVENTS];

  SUBDBG("%p\n",here);
  memset(tmp,0x0,sizeof(tmp));
  tmp[0].preset = PAPI_TOT_CYC;
  tmp[0].derived = NOT_DERIVED;
  tmp[1].preset = PAPI_TOT_INS;
  tmp[1].derived = NOT_DERIVED;

  if ((tmpn = getenv("PAPI_PERFMON_EVENT_FILE")) && (strlen(tmpn) != 0))
    sprintf(name,"%s",tmpn);
  else
    sprintf(name,"%s/%s",PAPI_DATADIR,PERFMON_EVENT_FILE);
  SUBDBG("Opening %s\n",name);
  table = fopen(name,"r");
  if (table == NULL)
    {
      SUBDBG("Open %s failed, trying ./%s.\n",name,PERFMON_EVENT_FILE);
      table = fopen(PERFMON_EVENT_FILE,"r");
      if (table == NULL)
	{
	  PAPIERROR("fopen(%s): %s, please set the PAPI_PERFMON_EVENT_FILE env. variable",name,strerror(errno));
	  return(PAPI_ESYS);
	}
      strcpy(name,PERFMON_EVENT_FILE);
    }
  SUBDBG("Open %s succeeded.\n",name);
  while (fgets(line,LINE_MAX,table))
    {
      char *t;
      int i;
      i = strlen(line);
      if (line[i-1] == '\n')
	line[i-1] = '\0';
      t = trim_string(strtok(line,","));
      if ((t == NULL) || (strlen(t) == 0))
	continue;
      if (t[0] == '#')
	{
	  SUBDBG("Comment found on line %d\n",line_no);
	  goto nextline;
	}
      else if (strcasecmp(t,"CPU") == 0)
	{
	  SUBDBG("CPU token found on line %d\n",line_no);
	  if (get_presets != 0)
	    {
	      SUBDBG("Ending preset scanning at line %d of %s.\n",line_no,name);
	      goto done;
	    }
	  t = trim_string(strtok(NULL,","));
	  if ((t == NULL) || (strlen(t) == 0))
	    {
	      PAPIERROR("Expected name after CPU token at line %d of %s -- ignoring",line_no,name);
	      goto nextline;
	    }
	  SUBDBG("Examining CPU (%s) vs. (%s)\n",t,_perfmon2_pfm_pmu_name);
	  if (strcasecmp(t,_perfmon2_pfm_pmu_name) == 0)
	    {
	      int type;

	      SUBDBG("Found CPU %s at line %d of %s.\n",t,line_no,name);
	      t = trim_string(strtok(NULL,","));
	      if ((t == NULL) || (strlen(t) == 0))
		{
		  SUBDBG("No additional qualifier found, matching on string.\n");
		  get_presets = 1;
		}
	      else if ((sscanf(t,"%d",&type) == 1) && (type == _perfmon2_pfm_pmu_type))
		{
		  SUBDBG("Found CPU %s type %d at line %d of %s.\n",_perfmon2_pfm_pmu_name,type,line_no,name);
		  get_presets = 1;
		}
	      else
		SUBDBG("Additional qualifier match failed %d vs %d.\n",_perfmon2_pfm_pmu_type,type);
	    }
	}
      else if (strcasecmp(t,"PRESET") == 0)
	{
	  SUBDBG("PRESET token found on line %d\n",line_no);
	  if (get_presets == 0)
	    goto nextline;
	  t = trim_string(strtok(NULL,","));
	  if ((t == NULL) || (strlen(t) == 0))
	    {
	      PAPIERROR("Expected name after PRESET token at line %d of %s -- ignoring",line_no,name);
	      goto nextline;
	    }
	  SUBDBG("Examining preset %s\n",t);
	  if (find_preset_code(t,&preset) != PAPI_OK)
	    {
	      PAPIERROR("Invalid preset name %s after PRESET token at line %d of %s -- ignoring",t,line_no,name);
	      goto nextline;
	    }
	  SUBDBG("Found 0x%08x for %s\n",preset,t);
	  t = trim_string(strtok(NULL,","));
	  if ((t == NULL) || (strlen(t) == 0))
	    {
	      PAPIERROR("Expected derived type after PRESET token at line %d of %s -- ignoring",line_no,name);
	      goto nextline;
	    }
	  SUBDBG("Examining derived %s\n",t);
	  if (_papi_hwi_derived_type(t,&derived) != PAPI_OK)
	    {
	      PAPIERROR("Invalid derived name %s after PRESET token at line %d of %s -- ignoring",t,line_no,name);
	      goto nextline;
	    }
	  SUBDBG("Found %d for %s\n",derived,t);

	  /* Add derived support here */

	  t = trim_string(strtok(NULL,","));
	  if ((t == NULL) || (strlen(t) == 0))
	    {
	      PAPIERROR("Expected PFM event after DERIVED token at line %d of %s -- ignoring",line_no,name);
	      goto nextline;
	    }
	  SUBDBG("Adding 0x%x,%d,%s to preset search table.\n",preset,derived,t);
	  tmp[insert].preset = preset;
	  tmp[insert].derived = derived;
	  tmp[insert].findme[0] = strdup(t);
	  insert++;
	}
      else
	{
	  PAPIERROR("Unrecognized token %s at line %d of %s -- ignoring",t,line_no,name);
	  goto nextline;
	}
    nextline:
      line_no++;
    }
 done:
  fclose(table);
  if (get_presets != 0) /* It at least found the CPU */
    {
      *here = tmp;
      return(PAPI_OK);
    }

  PAPIERROR("Failed to find events for CPU %s, type %d in %s",_perfmon2_pfm_pmu_name,_perfmon2_pfm_pmu_type,name);
  return(PAPI_ESBSTR);
}

static int generate_preset_search_map(hwi_search_t **maploc, hwi_dev_notes_t **noteloc, pfm_preset_search_entry_t *strmap, int num_cnt)
{
  int i = 0, j = 0, ret;
  hwi_search_t *psmap;
  hwi_dev_notes_t *notemap;
  pfmlib_event_t event;

  /* Count up the proposed presets */
  while (strmap[i].preset)
    i++;
  SUBDBG("generate_preset_search_map(%p,%p,%p,%d) %d proposed presets\n",maploc,noteloc,strmap,num_cnt,i);
  i++;

  /* Add null entry */
  psmap = (hwi_search_t *)malloc(i*sizeof(hwi_search_t));
  notemap = (hwi_dev_notes_t *)malloc(i*sizeof(hwi_dev_notes_t));
  if ((psmap == NULL) || (notemap == NULL))
    return(PAPI_ENOMEM);
  memset(psmap,0x0,i*sizeof(hwi_search_t));
  memset(notemap,0x0,i*sizeof(hwi_dev_notes_t));

  i = 0;
  while (strmap[i].preset)
    {
      if (strmap[i].preset == PAPI_TOT_CYC) 
	{
	  SUBDBG("pfm_get_cycle_event(%p)\n",&event);
	  if ((ret = pfm_get_cycle_event(&event)) == PFMLIB_SUCCESS)
	    {
	      if (setup_preset(&psmap[i], &notemap[i], &event, strmap[i].preset, strmap[i].derived, num_cnt) == PAPI_OK)
		{
		  j++;
		}
	    }
	  else
	    PAPIERROR("pfm_get_cycle_event(%p): %s",&event, pfm_strerror(ret));	    
	}
      else if (strmap[i].preset == PAPI_TOT_INS) 
	{
	  SUBDBG("pfm_get_inst_retired_event(%p)\n",&event);
	  if ((ret = pfm_get_inst_retired_event(&event)) == PFMLIB_SUCCESS)
	    {
	      if (setup_preset(&psmap[i], &notemap[i], &event, strmap[i].preset, strmap[i].derived, num_cnt) == PAPI_OK)
		{
		  j++;
		}
	    }
	  else
	    PAPIERROR("pfm_get_inst_retired_event(%p): %s",&event, pfm_strerror(ret));	    
	}
      else
	{
	  /* Does not handle derived events yet */
	  SUBDBG("pfm_find_full_event(%s,%p)\n",strmap[i].findme[0],&event);
	  if ((ret = pfm_find_full_event(strmap[i].findme[0],&event)) == PFMLIB_SUCCESS)
	    {
	      if (setup_preset(&psmap[i], &notemap[i], &event, strmap[i].preset, strmap[i].derived, num_cnt) == PAPI_OK)
		{
		  j++;
		}
	    }
	  else
	    PAPIERROR("pfm_find_full_event(%s,%p): %s",strmap[i].findme[0],&event,pfm_strerror(ret));	    
	}
      i++;
    }
       
   if (i != j) 
     {
       PAPIERROR("%d of %d events in %s were not valid",i-j,i,PERFMON_EVENT_FILE);
     }
   SUBDBG("generate_preset_search_map(%p,%p,%p,%d) %d actual presets\n",maploc,noteloc,strmap,num_cnt,j);
   *maploc = psmap;
   *noteloc = notemap;
   return (PAPI_OK);
}

static int generate_native_event_map(hwd_native_event_entry_t **nmap, unsigned int native_cnt, int num_cnt)
{
  int ret, did_something;
  unsigned int n, i, j;
  size_t max_len;
  char *findme = NULL, maskname[PAPI_MAX_STR_LEN],eventname[PAPI_MAX_STR_LEN];
  hwd_native_event_entry_t *ntmp, *orig_ntmp;
  pfmlib_regmask_t impl_cnt;

  if ((ret = pfm_get_max_event_name_len(&max_len)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_max_event_name_len(%p): %s", &max_len, pfm_strerror(ret));
      return(PAPI_ESBSTR);
    }

  if ((ret = pfm_get_impl_counters(&impl_cnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_impl_counters(%p): %s", &impl_cnt, pfm_strerror(ret));
      return(PAPI_ESBSTR);
    }
  orig_ntmp = ntmp = (hwd_native_event_entry_t *)malloc(native_cnt*sizeof(hwd_native_event_entry_t));
  for (i=0;i<native_cnt;i++)
    {
      if ((ret = pfm_get_event_name(i,eventname,sizeof(eventname))) != PFMLIB_SUCCESS)
	{
	  PAPIERROR("pfm_get_event_name(%d,%p,%d): %s", i,ntmp->name,sizeof(ntmp->name),pfm_strerror(ret));
	bail:
	  free(orig_ntmp);
	  return(PAPI_ESBSTR);
	}
      if ((ret = pfm_get_event_description(i,&findme)) != PFMLIB_SUCCESS)
	{
	  PAPIERROR("pfm_get_event_description(%d,%p): %s", i,&findme, pfm_strerror(ret));
	  goto bail;
	}
      if ((ret = pfm_get_event_counters(i,&ntmp->resources.selector)) != PFMLIB_SUCCESS)
	{
	  PAPIERROR("pfm_get_event_counters(%d,%p): %s", i,&ntmp->resources.selector,pfm_strerror(ret));
	  free(orig_ntmp);
	  free(findme);
	  return(PAPI_ESBSTR);
	}
      did_something = 0;
      n = num_cnt;
      for (j=0;n;j++)
	{
	  if (pfm_regmask_isset(&impl_cnt, j) != PFMLIB_SUCCESS)
	      {
		n--;
		if (pfm_regmask_isset(&ntmp->resources.selector,j))
		  {
		    int foo;
		    if ((ret = pfm_get_event_code_counter(i,j,&foo)) != PFMLIB_SUCCESS)
		      {
			PAPIERROR("pfm_get_event_code_counter(%d,%d,%p): %s", i,j,&foo,pfm_strerror(ret));
			free(orig_ntmp);
			free(findme);
			return(PAPI_ESBSTR);
		      }
		    did_something++;
		  }
	      }
	}
      if (did_something)
	{
	  unsigned int num_masks,c;
	  char *desc;

	  if ((ret = pfm_get_num_event_masks(i,&num_masks)) != PFMLIB_SUCCESS)
	    {
	      PAPIERROR("pfm_get_num_event_masks(%d,%p): %s",
			i,&num_masks,pfm_strerror(ret));
	      free(orig_ntmp);
	      free(findme);
	      return(PAPI_ESBSTR);
	    }
	  for (j=0;j<num_masks;j++)
	    {
	      if ((ret = pfm_get_event_mask_description(i, j, &desc)) != PFMLIB_SUCCESS)
		{
		  PAPIERROR("pfm_get_event_mask_description(%d,%d,%p): %s",
			    i,j,&desc,pfm_strerror(ret));
	      free(orig_ntmp);
	      free(findme);
		  return(PAPI_ESBSTR);
		}
	      if ((ret = pfm_get_event_mask_code(i, j, &c)) != PFMLIB_SUCCESS)
		{
		  PAPIERROR("pfm_get_event_mask_code(%d,%d,%p): %s",
			    i,j,&c,pfm_strerror(ret));
	      free(orig_ntmp);
	      free(findme);
	      free(desc);
		  return(PAPI_ESBSTR);
		  }
	      if ((ret = pfm_get_event_mask_name(i, j, maskname, PAPI_MAX_STR_LEN)) != PFMLIB_SUCCESS)
		{
		  PAPIERROR("pfm_get_event_mask_name(%d,%d,%p,%d): %s",
			    i,j,maskname,PAPI_MAX_STR_LEN);
	      free(orig_ntmp);
	      free(findme);
	      free(desc);
		  return(PAPI_ESBSTR);
		}
	      SUBDBG("PFM event %d has umask %s,%d,%s\n",i,maskname,c,desc);
	      free(desc);
	    }
	  strncpy(ntmp->name,eventname,sizeof(ntmp->name));
	  strncpy(ntmp->description,findme,sizeof(ntmp->description));
	  free(findme);
	}
      else if (did_something == 0)
	{
	  PAPIERROR("Could not get an event code for pfm index %d.",i);
	  free(orig_ntmp);
	  return(PAPI_ESBSTR);
	}
      ntmp++;
    }
  *nmap = orig_ntmp;
  return(PAPI_OK);
}

inline static int compute_kernel_args(hwd_control_state_t * ctl)
{
  pfmlib_input_param_t *inp = &ctl->in;
  pfmlib_output_param_t *outp = &ctl->out;
  pfmlib_input_param_t tmpin;
  pfmlib_output_param_t tmpout;
#if 0
  /* This will be used to fixup the overflow and sample args after re-allocation */
  pfarg_pmd_t oldpd;
#endif
  pfarg_pmd_t *pd = ctl->pd;
  pfarg_pmc_t *pc = ctl->pc;
  pfarg_setdesc_t *sets = ctl->set;
  pfarg_setinfo_t *setinfos = ctl->setinfo;
  int *num_sets = &ctl->num_sets;
  int set = 0, donepc = 0, donepd = 0, ret, i, j;
  int togo = inp->pfp_event_count, dispatch_count = inp->pfp_event_count, done = 0;
  
  /* Save old PD array so we can reconstruct certain flags. This can be removed
     when we have higher level code call set_profile,set_overflow etc when there
     is hardware (substrate) support, but this change won't happen for PAPI 3.5 */

  if ((ctl->multiplexed) && (inp->pfp_event_count > _papi_hwi_system_info.sub_info.num_cntrs))
    {
      dispatch_count = _papi_hwi_system_info.sub_info.num_cntrs;
    }

  while (togo)
    {
      memset(&tmpin,0x0,sizeof(tmpin));
      memset(&tmpout,0x0,sizeof(tmpout));
      
      SUBDBG("togo %d, done %d, dispatch_count %d\n",togo,done,dispatch_count);
      tmpin.pfp_event_count = dispatch_count;
      tmpin.pfp_dfl_plm = inp->pfp_dfl_plm;

      /* Make sure we tell dispatch that these PMC's are not available */
      memcpy(&tmpin.pfp_unavail_pmcs,&_perfmon2_pfm_unavailable_pmcs,sizeof(_perfmon2_pfm_unavailable_pmcs));

      for (i=0,j=done;i<dispatch_count;i++,j++)
	{
	  memcpy(tmpin.pfp_events+i,inp->pfp_events+j,sizeof(pfmlib_event_t));
	}

      if ((ret = pfm_dispatch_events(&tmpin, NULL, &tmpout, NULL)) != PFMLIB_SUCCESS)
	{
	  PAPIERROR("pfm_dispatch_events(): %s", pfm_strerror(ret));
	  return(PAPI_ECNFLCT);
	}
  
  /*
    * Now prepare the argument to initialize the PMDs and PMCS.
    * We must pfp_pmc_count to determine the number of PMC to intialize.
    * We must use pfp_event_count to determine the number of PMD to initialize.
    * Some events causes extra PMCs to be used, so  pfp_pmc_count may be >= pfp_event_count.
    *
    * This step is new compared to libpfm-2.x. It is necessary because the library no
    * longer knows about the kernel data structures.
    */

      for (i=0; i < tmpout.pfp_pmc_count; i++, donepc++) 
	{
	  pc[donepc].reg_num   = tmpout.pfp_pmcs[i].reg_num;
	  pc[donepc].reg_value = tmpout.pfp_pmcs[i].reg_value;
	  pc[donepc].reg_set = set;
	}
   
      /* figure out pmd mapping from output pmc */

#if defined(HAVE_PFM_REG_EVT_IDX)
      for (i=0, j=0; i < tmpin.pfp_event_count; i++, donepd++) 
	{
	  pd[donepd].reg_num = tmpout.pfp_pmcs[j].reg_pmd_num;
	  pd[donepd].reg_set = set;

	  /* Skip over entries that map to the same PMD, 
	     PIV has 2 PMCS for every PMD */

	  for(; j < tmpout.pfp_pmc_count; j++)  
	    if (tmpout.pfp_pmcs[j].reg_evt_idx != i) break;
	}
#else
      for (i=0; i < tmpout.pfp_pmd_count; i++, donepd++) 
        {
          pd[donepd].reg_num = tmpout.pfp_pmds[i].reg_num;
        }
#endif
      
      togo -= dispatch_count;
      done += dispatch_count;
      if (togo > _papi_hwi_system_info.sub_info.num_cntrs)
	dispatch_count = _papi_hwi_system_info.sub_info.num_cntrs;
      else
	dispatch_count = togo;

      setinfos[set].set_id = set;
      sets[set].set_id = set;
      sets[set].set_flags = PFM_SETFL_TIME_SWITCH;
      sets[set].set_timeout = _papi_hwi_system_info.sub_info.multiplex_timer_us;
      set++;
    }

  *num_sets = set;
  outp->pfp_pmc_count = donepc;

  return(PAPI_OK);
}

static int attach(hwd_control_state_t *ctl, unsigned long tid)
{
  pfarg_ctx_t *newctx = (pfarg_ctx_t *)malloc(sizeof(pfarg_ctx_t));
  pfarg_load_t *load_args = (pfarg_load_t *)malloc(sizeof(pfarg_load_t));
  int ret;

  if ((newctx == NULL) || (load_args == NULL))
    return(PAPI_ENOMEM);
  memset(newctx,0x0,sizeof(*newctx));
  memset(load_args,0,sizeof(*load_args));

  /* Make sure the process exists and is being ptraced() */

  ret = ptrace(PTRACE_ATTACH, tid, NULL, NULL);
  if (ret == 0)
    {
      ptrace(PTRACE_DETACH, tid, NULL, NULL);
      PAPIERROR("Process/thread %d is not being ptraced",tid);
      free(newctx); free(load_args);
      return(PAPI_EINVAL);
    }
  /* If we get here, then we should hope that the process is being
     ptraced, if not, then we probably can't attach to it. */

  if ((ret == -1) && (errno != EPERM))
    {
      PAPIERROR("Process/thread %d cannot be ptraced: %s",tid,strerror(errno));
      free(newctx); free(load_args);
      return(PAPI_EINVAL);
    }

  SUBDBG("PFM_CREATE_CONTEXT(%p,%p,%p,%d)\n",newctx,NULL,NULL,0);
  if ((ret = pfm_create_context(newctx, NULL, NULL, 0)) == -1)
    {
      PAPIERROR("pfm_create_context(): %s", pfm_strerror(ret));
      free(newctx); free(load_args);
      return(PAPI_ESYS);
    }
  SUBDBG("PFM_CREATE_CONTEXT returns fd %d\n",ret);

  ctl->ctx_fd = ret;
  ctl->ctx = newctx;
  load_args->load_pid = tid;
  ctl->load = load_args;

  return(PAPI_OK);
}

static int detach(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
  close(ctl->ctx_fd);

  /* Restore to main threads context */
  free(ctl->ctx);
  ctl->ctx = &ctx->ctx;
  ctl->ctx_fd = ctx->ctx_fd;
  free(ctl->load);
  ctl->load = &ctx->load;

  return(PAPI_OK);
}

/* This routine effectively does argument checking as the real magic will happen
   in compute_kernel_args. This just gets the value back from the kernel. */

inline static int check_multiplex_timeout(hwd_context_t *ctx, unsigned long *timeout)
{
  int ret, ctx_fd;
  pfarg_setdesc_t set;
  pfarg_ctx_t newctx;

  return(PAPI_OK);

  if (ctx == NULL)
    ctx_fd = 0; /* This happens from inside init_substrate_global, where no context yet exists. */
  else
    ctx_fd = ctx->ctx_fd;

  memset(&set,0,sizeof(set));
  set.set_timeout = *timeout;
  SUBDBG("Requested multiplexing interval is %llu usecs.\n",set.set_timeout);

  /* This may be called before we have a context, so we should build one
     if we need one. */

  if (ctx_fd == 0)
    {
      memset(&newctx, 0, sizeof(newctx));

      SUBDBG("PFM_CREATE_CONTEXT(%p,%p,%p,%d)\n",&newctx,NULL,NULL,0);
      if ((ret = pfm_create_context(&newctx, NULL, NULL, 0)) == -1)
	{
	  PAPIERROR("pfm_create_context(): %s", pfm_strerror(ret));
	  return(PAPI_ESYS);
	}
      SUBDBG("PFM_CREATE_CONTEXT returns fd %d\n",ret);
      ctx_fd = ret;
    }
      
  SUBDBG("PFM_CREATE_EVTSETS(%d,%p,1)\n",ctx_fd,&set);
  if ((ret = pfm_create_evtsets(ctx_fd, &set, 1)) != PFMLIB_SUCCESS)
    {
      DEBUGCALL(DEBUG_SUBSTRATE,dump_sets(&set,1));
      PAPIERROR("pfm_create_evtsets(%d,%p,1): %s", ctx_fd, &set, pfm_strerror(ret));
      return(PAPI_ESYS);
    }      

  SUBDBG("Multiplexing interval is now %llu usecs.\n",set.set_timeout);
  *timeout = set.set_timeout;
  
  /* If we created a context, get rid of it */
  if (ctx_fd) 
    {
      pfm_delete_evtsets(ctx_fd,&set,1);
      pfm_unload_context(ctx_fd);
      close(ctx_fd);
    }

  return(PAPI_OK);
}

inline static int set_domain(hwd_control_state_t * ctl, int domain)
{
  int mode = 0, did = 0;
  pfmlib_input_param_t *inp = &ctl->in;

   if (domain & PAPI_DOM_USER) {
      did = 1;
      mode |= PFM_PLM3;
   }

   if (domain & PAPI_DOM_KERNEL) {
     mode |= PFM_PLM0;
   }

   if (domain & PAPI_DOM_SUPERVISOR) {
      did = 1;
      mode |= PFM_PLM1;
   }

   if (domain & PAPI_DOM_OTHER) {
      did = 1;
      mode |= PFM_PLM2;
   }

   if (!did)
      return (PAPI_EINVAL);

   inp->pfp_dfl_plm = mode;

   return(compute_kernel_args(ctl));
}

inline static int set_granularity(hwd_control_state_t * this_state, int domain)
{
   switch (domain) {
   case PAPI_GRN_PROCG:
   case PAPI_GRN_SYS:
   case PAPI_GRN_SYS_CPU:
   case PAPI_GRN_PROC:
      return(PAPI_ESBSTR);
   case PAPI_GRN_THR:
      break;
   default:
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit(int arg)
{
   return (PAPI_ESBSTR);
}

static int get_string_from_file(char *file, char *str, int len)
{
  FILE *f = fopen(file,"r");
  char buf[PAPI_HUGE_STR_LEN];
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s", file, strerror(errno));
      return(PAPI_ESYS);
    }
  if (fscanf(f,"%s\n",buf) != 1)
    {
      PAPIERROR("fscanf(%s, %%s\\n): Unable to scan 1 token", file);
      fclose(f);
      return(PAPI_ESBSTR);
    }
  strncpy(str,buf,(len > PAPI_HUGE_STR_LEN ? PAPI_HUGE_STR_LEN : len));
  fclose(f);
  return(PAPI_OK);
}

int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
  int i, retval;
  unsigned int ncnt,ncnt_no_avail = 0;
  unsigned int version;
#ifdef DEBUG
  pfmlib_options_t pfmlib_options;
#endif
  char buf[PAPI_HUGE_STR_LEN];
  hwi_dev_notes_t *notemap = NULL;

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _linux_pfm_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

  /* Always initialize globals dynamically to handle forks properly. */

  _perfmon2_pfm_preset_search_map = NULL;
  _perfmon2_pfm_native_map = NULL;
  memset(&_perfmon2_pfm_unavailable_pmcs,0,sizeof(_perfmon2_pfm_unavailable_pmcs));
  _perfmon2_pfm_pmu_type = -1;
  _perfmon2_pfm_pmu_name[0] = '\0';

   /* Opened once for all threads. */
   SUBDBG("pfm_initialize()\n");
   if (pfm_initialize() != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_initialize(): %s", pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }

  /* Find out if any PMC's/PMD's are off limits */

  retval = detect_unavail_pmcs(&_perfmon2_pfm_unavailable_pmcs,
			       &ncnt_no_avail);
  if (retval != PFMLIB_SUCCESS)
    return(retval);

   SUBDBG("pfm_get_pmu_type(%p)\n",&_perfmon2_pfm_pmu_type);
   if (pfm_get_pmu_type(&_perfmon2_pfm_pmu_type) != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_get_pmu_type(%p): %s", _perfmon2_pfm_pmu_type, pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }

   SUBDBG("pfm_get_pmu_name(%p,%d)\n",_perfmon2_pfm_pmu_name,PAPI_MIN_STR_LEN);
   if (pfm_get_pmu_name(_perfmon2_pfm_pmu_name,PAPI_MIN_STR_LEN) != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_get_pmu_name(%p,%d): %s",_perfmon2_pfm_pmu_name,PAPI_MIN_STR_LEN,pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }
   SUBDBG("PMU is a %s, type %d\n",_perfmon2_pfm_pmu_name,_perfmon2_pfm_pmu_type);

   SUBDBG("pfm_get_version(%p)\n",&version);
   if (pfm_get_version(&version) != PFMLIB_SUCCESS)
     {
       PAPIERROR("pfm_get_version(%p): %s", version, pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }

   if (PFM_VERSION_MAJOR(version) != PFM_VERSION_MAJOR(PFMLIB_VERSION)) {
      PAPIERROR("Version mismatch of libpfm: compiled %x vs. installed %x",
              PFM_VERSION_MAJOR(PFMLIB_VERSION), PFM_VERSION_MAJOR(version));
      return (PAPI_ESBSTR);
   }

#ifdef DEBUG
   memset(&pfmlib_options, 0, sizeof(pfmlib_options));
   if (ISLEVEL(DEBUG_SUBSTRATE)) {
      pfmlib_options.pfm_debug = 1;
      pfmlib_options.pfm_verbose = 1;
   }
   SUBDBG("pfm_set_options(%p)\n",&pfmlib_options);
   if (pfm_set_options(&pfmlib_options))
     {
       PAPIERROR("pfm_set_options(%p): %s", &pfmlib_options, pfm_strerror(retval));
       return (PAPI_ESBSTR);
     }
#endif

   /* Fill in sub_info */

   SUBDBG("pfm_get_num_events(%p)\n",&ncnt);
  if ((retval = pfm_get_num_events(&ncnt)) != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_num_events(%p): %s", &ncnt, pfm_strerror(retval));
      return(PAPI_ESBSTR);
    }
  _papi_hwi_system_info.sub_info.num_native_events = ncnt;
  strcpy(_papi_hwi_system_info.sub_info.name, "$Id$");          
  strcpy(_papi_hwi_system_info.sub_info.version, "$Revision$");  
  sprintf(buf,"%08x",version);
  strncpy(_papi_hwi_system_info.sub_info.support_version,buf,sizeof(_papi_hwi_system_info.sub_info.support_version));
  retval = get_string_from_file("/sys/kernel/perfmon/version",_papi_hwi_system_info.sub_info.kernel_version,sizeof(_papi_hwi_system_info.sub_info.kernel_version));
  if (retval != PAPI_OK)
    return(retval);
  pfm_get_num_counters((unsigned int *)&_papi_hwi_system_info.sub_info.num_cntrs);
  _papi_hwi_system_info.sub_info.num_cntrs -= ncnt_no_avail;
  if (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_MIPS)
    _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_KERNEL|PAPI_DOM_SUPERVISOR|PAPI_DOM_OTHER;
  else
    _papi_hwi_system_info.sub_info.available_domains |= PAPI_DOM_KERNEL;    

  if ((_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL) ||
      (_papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_AMD))
    {
      _papi_hwi_system_info.sub_info.fast_counter_read = 1;
      _papi_hwi_system_info.sub_info.fast_real_timer = 1;
    }

  _papi_hwi_system_info.sub_info.hardware_intr = 1;
  _papi_hwi_system_info.sub_info.attach = 1;
  _papi_hwi_system_info.sub_info.attach_must_ptrace = 1;
  _papi_hwi_system_info.sub_info.kernel_multiplex = 1;
  _papi_hwi_system_info.sub_info.kernel_profile = 1;
  _papi_hwi_system_info.sub_info.profile_ear = 1;  
  _papi_hwi_system_info.sub_info.num_mpx_cntrs = PFMLIB_MAX_PMDS;
  //  check_multiplex_timeout(NULL, (unsigned long *)&_papi_hwi_system_info.sub_info.multiplex_timer_us);

  /* FIX: For now, use the pmu_type from Perfmon */

  _papi_hwi_system_info.hw_info.model = _perfmon2_pfm_pmu_type;
   retval = get_system_info(&_papi_hwi_system_info);
   if (retval)
      return (retval);

   /* Setup presets */

   retval = load_preset_table(&_perfmon2_pfm_preset_search_map);
   if (retval)
     return(retval);

   retval = generate_preset_search_map(&preset_search_map,&notemap,_perfmon2_pfm_preset_search_map,_papi_hwi_system_info.sub_info.num_cntrs);
   if (retval)
     {
       return (retval);
     }

   retval = generate_native_event_map(&_perfmon2_pfm_native_map,_papi_hwi_system_info.sub_info.num_native_events,_papi_hwi_system_info.sub_info.num_cntrs);
   if (retval)
     {
       free(preset_search_map);
       return (retval);
     }

   retval = _papi_hwi_setup_all_presets(preset_search_map, notemap);
   if (retval)
     {
       free(preset_search_map);
       free(_perfmon2_pfm_native_map);
       return (retval);
     }

   for (i = 0; i < PAPI_MAX_LOCK; i++)
      _papi_hwd_lock_data[i] = MUTEX_OPEN;
   
   return (PAPI_OK);
}

int _papi_hwd_init(hwd_context_t * thr_ctx)
{
  pfarg_load_t load_args;
  pfarg_ctx_t newctx;
  int ret, ctx_fd;

#if defined(USE_PROC_PTTIMER)
  {
    char buf[LINE_MAX];
    int fd;
    sprintf(buf,"/proc/%d/task/%d/stat",getpid(),mygettid());
    fd = open(buf,O_RDONLY);
    if (fd == -1)
      {
	PAPIERROR("open(%s)",buf);
	return(PAPI_ESYS);
      }
    thr_ctx->stat_fd = fd;
  }
#endif

  memset(&newctx, 0, sizeof(newctx));
  memset(&load_args, 0, sizeof(load_args));

  SUBDBG("PFM_CREATE_CONTEXT(%p,%p,%p,%d)\n",&newctx, NULL, NULL, 0);
  if ((ret = pfm_create_context(&newctx, NULL, NULL, 0)) == -1)
    {
      PAPIERROR("pfm_create_context(%p,%p,%d): %s",&newctx,NULL,0,pfm_strerror(errno));
      return(PAPI_ESYS);
    }
  SUBDBG("PFM_CREATE_CONTEXT returns fd %d\n",ret);
  ctx_fd = ret;

  /* set close-on-exec to ensure we will be getting the PFM_END_MSG, i.e.,
   * fd not visible to child. */

  ret = fcntl(ctx_fd, F_SETFD, FD_CLOEXEC);
  if (ret == -1)
    {
      PAPIERROR("cannot fcntl(FD_CLOEXEC) on %d: %s", ctx_fd,strerror(errno));
    bail:
      pfm_unload_context(ctx_fd);
      close(ctx_fd);
      return(PAPI_ESYS);
    }

  /* setup asynchronous notification on the file descriptor */

  ret = fcntl(ctx_fd, F_SETFL, fcntl(ctx_fd, F_GETFL, 0) | O_ASYNC);
  if (ret == -1)
    {
      PAPIERROR("cannot fcntl(O_ASYNC) on %d: %s", ctx_fd,strerror(errno));
      goto bail;
    }

  /* get ownership of the descriptor */

  ret = fcntl(ctx_fd, F_SETOWN, mygettid());
  if (ret == -1)
    {
      PAPIERROR("cannot fcntl(F_SETOWN) on %d: %s", ctx_fd,strerror(errno));
      goto bail;
    }

  /*
   * when you explicitely declare that you want a particular signal,
   * even with you use the default signal, the kernel will send more
   * information concerning the event to the signal handler.
   *
   * In particular, it will send the file descriptor from which the
   * event is originating which can be quite useful when monitoring
   * multiple tasks from a single thread.
   */

  ret = fcntl(ctx_fd, F_SETSIG, _papi_hwi_system_info.sub_info.hardware_intr_sig);
  if (ret == -1)
    {
      PAPIERROR("cannot fcntl(F_SETSIG,%d) on %d: %s", _papi_hwi_system_info.sub_info.hardware_intr_sig, ctx_fd, strerror(errno));
      goto bail;
    }

  memcpy(&thr_ctx->ctx,&newctx,sizeof(newctx));
  thr_ctx->ctx_fd = ctx_fd;
  load_args.load_pid = mygettid();
  memcpy(&thr_ctx->load,&load_args,sizeof(load_args));

  return(PAPI_OK);
}

long_long _papi_hwd_get_real_usec(void) {
   return((long_long)get_cycles() / (long_long)_papi_hwi_system_info.hw_info.mhz);
}
                                                                                
long_long _papi_hwd_get_real_cycles(void) {
   return((long_long)get_cycles());
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   long_long retval;
#if defined(USE_PROC_PTTIMER)
   {
     char buf[LINE_MAX];
     long_long utime, stime;
     int rv, cnt = 0, i = 0;

     rv = read(zero->stat_fd,buf,LINE_MAX*sizeof(char));
     if (rv == -1)
       {
	 PAPIERROR("read()");
	 return(PAPI_ESYS);
       }
     lseek(zero->stat_fd,0,SEEK_SET);

     buf[rv] = '\0';
     SUBDBG("Thread stat file is:%s\n",buf);
     while ((cnt != 13) && (i < rv))
       {
	 if (buf[i] == ' ')
	   { cnt++; }
	 i++;
       }
     if (cnt != 13)
       {
	 PAPIERROR("utime and stime not in thread stat file?");
	 return(PAPI_ESBSTR);
       }
     
     if (sscanf(buf+i,"%llu %llu",&utime,&stime) != 2)
       {
	 PAPIERROR("Unable to scan two items from thread stat file at 13th space?");
	 return(PAPI_ESBSTR);
       }
     retval = (utime+stime)*(long_long)(1000000/sysconf(_SC_CLK_TCK));
   }
#elif defined(HAVE_CLOCK_GETTIME_THREAD)
   {
     struct timespec foo;
     double bar;
     
     syscall(__NR_clock_gettime,HAVE_CLOCK_GETTIME_THREAD,&foo);
     bar = (double)foo.tv_nsec/1000.0 + (double)foo.tv_sec*1000000.0;
     retval = (long_long) bar;
   }
#elif defined(HAVE_PER_THREAD_TIMES)
   {
     struct tms buffer;
     times(&buffer);
     SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime);
     retval = (long_long)((buffer.tms_utime+buffer.tms_stime)*(1000000/sysconf(_SC_CLK_TCK)));
     /* NOT CLOCKS_PER_SEC as in the headers! */
   }
#else
#error "No working per thread timer"
#endif
   return (retval);
}

long_long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   return (_papi_hwd_get_virt_usec(zero) * (long_long)_papi_hwi_system_info.hw_info.mhz);
}

/* reset the hardware counters */
int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t *ctl)
{
  int i, ret;

  for (i=0; i < ctl->in.pfp_event_count; i++) 
    ctl->pd[i].reg_value = 0ULL;

  SUBDBG("PFM_WRITE_PMDS(%d,%p,%d)\n",ctl->ctx_fd, ctl->pd, ctl->in.pfp_event_count);
  if ((ret = pfm_write_pmds(ctl->ctx_fd, ctl->pd, ctl->in.pfp_event_count)))
    {
      PAPIERROR("pfm_write_pmds(%d,%p,%d): %s",ctl->ctx_fd,ctl->pd,ctl->in.pfp_event_count, pfm_strerror(ret));
      return(PAPI_ESYS);
    }

  return (PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * ctl,
                   long_long ** events, int flags)
{
  int i, ret;
  long_long tot_runs = 0LL;

  SUBDBG("PFM_READ_PMDS(%d,%p,%d)\n",ctl->ctx_fd, ctl->pd, ctl->in.pfp_event_count);
  if ((ret = pfm_read_pmds(ctl->ctx_fd, ctl->pd, ctl->in.pfp_event_count)))
    {
      DEBUGCALL(DEBUG_SUBSTRATE,dump_pmd(ctl));
      PAPIERROR("pfm_read_pmds(%d,%p,%d): %s",ctl->ctx_fd,ctl->pd,ctl->in.pfp_event_count, pfm_strerror(ret));
      *events = NULL;
      return(PAPI_ESYS);
    }
  DEBUGCALL(DEBUG_SUBSTRATE,dump_pmd(ctl));
  
  /* Copy the values over */

  for (i=0; i < ctl->in.pfp_event_count; i++) 
    {
      SUBDBG("PMD[%d] = %lld (LLD),%llu (LLU)\n",i,(unsigned long long)ctl->counts[i],(unsigned long long)ctl->pd[i].reg_value);
      if (ctl->pd[i].reg_flags & PFM_REGFL_OVFL_NOTIFY)
	ctl->counts[i] = ctl->pd[i].reg_value - ctl->pd[i].reg_long_reset;
      else
	ctl->counts[i] = ctl->pd[i].reg_value;
    }
  *events = ctl->counts;

  /* If we're not multiplexing, bail now */

  if (ctl->num_sets == 1)
    return(PAPI_OK);

  /* If we're multiplexing, get the scaling information */

  SUBDBG("PFM_GETINFO_EVTSETS(%d,%p,%d)\n",ctl->ctx_fd, ctl->setinfo, ctl->num_sets);
  if ((ret = pfm_getinfo_evtsets(ctl->ctx_fd, ctl->setinfo, ctl->num_sets)))
    {
      DEBUGCALL(DEBUG_SUBSTRATE,dump_setinfo(ctl));
      PAPIERROR("pfm_getinfo_evtsets(%d,%p,%d): %s",ctl->ctx_fd, ctl->setinfo, ctl->num_sets, pfm_strerror(ret));
      *events = NULL;
      return(PAPI_ESYS);
    }
  DEBUGCALL(DEBUG_SUBSTRATE,dump_setinfo(ctl));

  /* Add up the number of total runs */
  
  for (i=0;i<ctl->num_sets;i++)
    tot_runs += ctl->setinfo[i].set_runs;

  /* Now scale the values */

  for (i=0; i < ctl->in.pfp_event_count; i++) 
    {
      SUBDBG("Counter %d is in set %d ran %llu of %llu times, old count %lld.\n",
	     i,
	     ctl->pd[i].reg_set,
	     (unsigned long long)ctl->setinfo[ctl->pd[i].reg_set].set_runs,
	     (unsigned long long)tot_runs,
	     ctl->counts[i]);
      ctl->counts[i] = (ctl->counts[i]*tot_runs)/ctl->setinfo[ctl->pd[i].reg_set].set_runs;
      SUBDBG("Counter %d, new count %lld.\n",i,ctl->counts[i]);
    }

   return PAPI_OK;
}


int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int i, ret; 

  if (ctl->num_sets > 1)
    {
      SUBDBG("PFM_CREATE_EVTSETS(%d,%p,%d)\n",ctl->ctx_fd,ctl->set,ctl->num_sets);
      if ((ret = pfm_create_evtsets(ctl->ctx_fd,ctl->set,ctl->num_sets)) != PFMLIB_SUCCESS)
	{
	  DEBUGCALL(DEBUG_SUBSTRATE,dump_sets(ctl->set,ctl->num_sets));
	  PAPIERROR("pfm_create_evtsets(%d,%p,%d): %s", ctl->ctx_fd,ctl->set,ctl->num_sets, pfm_strerror(ret));
	  return(PAPI_ESYS);
	}
      DEBUGCALL(DEBUG_SUBSTRATE,dump_sets(ctl->set,ctl->num_sets));
    }      

  /*
   * Now program the registers
   *
   * We don't use the same variable to indicate the number of elements passed to
   * the kernel because, as we said earlier, pc may contain more elements than
   * the number of events (pmd) we specified, i.e., contains more than counting
   * monitors.
   */

  SUBDBG("PFM_WRITE_PMCS(%d,%p,%d)\n",ctl->ctx_fd, ctl->pc, ctl->out.pfp_pmc_count);
  if ((ret = pfm_write_pmcs(ctl->ctx_fd, ctl->pc, ctl->out.pfp_pmc_count)))
    {
      DEBUGCALL(DEBUG_SUBSTRATE,dump_pmc(ctl));
      PAPIERROR("pfm_write_pmcs(%d,%p,%d): %s",ctl->ctx_fd,ctl->pc,ctl->out.pfp_pmc_count, pfm_strerror(ret));
      return(PAPI_ESYS);
    }
  DEBUGCALL(DEBUG_SUBSTRATE,dump_pmc(ctl));
  
  /* Set counters to zero as per PAPI_start man page, unless it is set to overflow */

  for (i=0; i < ctl->in.pfp_event_count; i++) 
    if (!(ctl->pd[i].reg_flags & PFM_REGFL_OVFL_NOTIFY))
      ctl->pd[i].reg_value = 0ULL; 

  /*
   * To be read, each PMD must be either written or declared
   * as being part of a sample (reg_smpl_pmds)
   */

  SUBDBG("PFM_WRITE_PMDS(%d,%p,%d)\n",ctl->ctx_fd, ctl->pd, ctl->in.pfp_event_count);
  if ((ret = pfm_write_pmds(ctl->ctx_fd, ctl->pd, ctl->in.pfp_event_count)))
    {
      DEBUGCALL(DEBUG_SUBSTRATE,dump_pmd(ctl));
      PAPIERROR("pfm_write_pmds(%d,%p,%d): %s",ctl->ctx_fd,ctl->pd,ctl->in.pfp_event_count, pfm_strerror(ret));
      return(PAPI_ESYS);
    }
  DEBUGCALL(DEBUG_SUBSTRATE,dump_pmd(ctl));

  SUBDBG("PFM_LOAD_CONTEXT(%d,%p(%u))\n",ctl->ctx_fd,ctl->load,ctl->load->load_pid);
  if ((ret = pfm_load_context(ctl->ctx_fd,ctl->load)))
    {
      PAPIERROR("pfm_load_context(%d,%p(%u)): %s", ctl->ctx_fd,ctl->load,ctl->load->load_pid, pfm_strerror(ret));
      return PAPI_ESYS;
    }

  SUBDBG("PFM_START(%d,%p)\n",ctl->ctx_fd, NULL);
  if ((ret = pfm_start(ctl->ctx_fd, NULL)))
    {
      PAPIERROR("pfm_start(%d): %s", ctl->ctx_fd, pfm_strerror(ret));
      return(PAPI_ESYS);
    }
   return PAPI_OK;
}

int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * ctl)
{
  int ret;

  SUBDBG("PFM_STOP(%d)\n",ctl->ctx_fd);
  if ((ret = pfm_stop(ctl->ctx_fd)))
    {
      /* If this thread is attached to another thread, and that thread
	 has exited, we can safely discard the error here. */

      if ((ret == PFMLIB_ERR_NOTSUPP) && (ctl->load->load_pid != mygettid()))
	return(PAPI_OK);

      PAPIERROR("pfm_stop(%d): %s", ctl->ctx_fd, pfm_strerror(ret));
      return(PAPI_ESYS);
    }

  SUBDBG("PFM_UNLOAD_CONTEXT(%d) (tid %u)\n",ctl->ctx_fd,ctl->load->load_pid);
  if ((ret = pfm_unload_context(ctl->ctx_fd)))
    {
      PAPIERROR("pfm_unload_context(%d): %s", ctl->ctx_fd, pfm_strerror(ret));
      return PAPI_ESYS;
    }

   return PAPI_OK;
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
  int ret;

  switch (code) {
  case PAPI_DEF_MPX_USEC:
    return(check_multiplex_timeout(ctx,&option->multiplex.us));
  case PAPI_MULTIPLEX:
    ret = check_multiplex_timeout(ctx,&option->multiplex.us);
    if (ret == PAPI_OK)
      option->multiplex.ESI->machdep.multiplexed = 1;
    return(ret);
  case PAPI_ATTACH:
    return(attach(&option->attach.ESI->machdep, option->attach.tid));
  case PAPI_DETACH:
    return(detach(ctx, &option->attach.ESI->machdep));
  case PAPI_DOMAIN:
    return(set_domain(&option->domain.ESI->machdep, option->domain.domain));
  case PAPI_GRANUL:
    return (set_granularity
	    (&option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
  case PAPI_DATA_ADDRESS:
    ret=set_default_domain(&option->address_range.ESI->machdep, option->address_range.domain);
    if(ret != PAPI_OK) return(ret);
    set_drange(ctx, &option->address_range.ESI->machdep, option);
    return (PAPI_OK);
  case PAPI_INSTR_ADDRESS:
    ret=set_default_domain(&option->address_range.ESI->machdep, option->address_range.domain);
    if(ret != PAPI_OK) return(ret);
    set_irange(ctx, &option->address_range.ESI->machdep, option);
    return (PAPI_OK);
#endif
  default:
    return (PAPI_EINVAL);
  }
}

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
#if defined(USR_PROC_PTTIMER)
  close(ctx->stat_fd);
#endif  
  close(ctx->ctx_fd);
  return (PAPI_OK);
}

/* This will need to be modified for the Pentium IV */

static inline int find_profile_index(EventSetInfo_t *ESI, int pmd, int *flags, unsigned int *native_index, int *profile_index)
{
  int pos, esi_index, count;
  hwd_control_state_t *ctl = &ESI->machdep;
  pfarg_pmd_t *pd;
  pfarg_pmc_t *pc;
  int i;

  pd = ctl->pd;
  pc = ctl->pc;
  
  /* Find virtual PMD index, the one we actually read from the physical PMD number that
     overflowed. This index is the one related to the profile buffer. */

  for (i=0;i<ctl->in.pfp_event_count;i++)
    {
      if (pd[i].reg_num == pmd)
	{
	  SUBDBG("Physical PMD %d is Virtual PMD %d\n",pmd,i);
	  pmd = i;
	  break;
	}
    }

  SUBDBG("(%p,%d,%p)\n",ESI,pmd,index);
  for (count = 0; count < ESI->profile.event_counter; count++) 
    {
      /* Find offset of PMD that gets read from the kernel */
      esi_index = ESI->profile.EventIndex[count];
      pos = ESI->EventInfoArray[esi_index].pos[0];
      SUBDBG("Examining event at ESI index %d, PMD position %d\n",esi_index,pos);
      // PMU_FIRST_COUNTER
      if (pos == pmd) 
	{
	  *profile_index = count;
	  *native_index = ESI->NativeInfoArray[pos].ni_event & PAPI_NATIVE_AND_MASK;
	  *flags = ESI->profile.flags;
	  SUBDBG("Native event %d is at profile index %d, flags %d\n",*native_index,*profile_index,*flags);
	  return(PAPI_OK);
	}
    }
  
  PAPIERROR("wrong count: %d vs. ESI->profile.event_counter %d", count, ESI->profile.event_counter);
  return(PAPI_EBUG);
}

#if defined(__ia64__)
static inline int is_montecito_and_dear(unsigned int native_index)
{
  if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
    {
      if (pfm_mont_is_dear(native_index))
	return(1);
    }
  return(0);
}
static inline int is_montecito_and_iear(unsigned int native_index)
{
  if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
    {
      if (pfm_mont_is_iear(native_index))
	return(1);
    }
  return(0);
}
static inline int is_itanium2_and_dear(unsigned int native_index)
{
  if (_perfmon2_pfm_pmu_type == PFMLIB_ITANIUM2_PMU)
    {
      if (pfm_ita2_is_dear(native_index))
	return(1);
    }
  return(0);
}
static inline int is_itanium2_and_iear(unsigned int native_index)
{
  if (_perfmon2_pfm_pmu_type == PFMLIB_ITANIUM2_PMU)
    {
      if (pfm_ita2_is_iear(native_index))
	return(1);
    }
  return(0);
}
#endif

#define BPL (sizeof(uint64_t)<<3)
#define LBPL	6
static inline void pfm_bv_set(uint64_t *bv, uint16_t rnum)
{
	bv[rnum>>LBPL] |= 1UL << (rnum&(BPL-1));
}

static inline int setup_ear_event(unsigned int native_index, pfarg_pmd_t *pd, int flags)
{
#if defined(__ia64__)
  if (_perfmon2_pfm_pmu_type == PFMLIB_MONTECITO_PMU)
    {
      if (pfm_mont_is_dear(native_index)) /* 2,3,17 */
	{
	  pfm_bv_set(pd[0].reg_smpl_pmds, 32);
	  pfm_bv_set(pd[0].reg_smpl_pmds, 33);
	  pfm_bv_set(pd[0].reg_smpl_pmds, 36);
	  pfm_bv_set(pd[0].reg_reset_pmds, 36);
	  return(1);
	}
      else if (pfm_mont_is_iear(native_index))  /* O,1 MK */
	{
	  pfm_bv_set(pd[0].reg_smpl_pmds, 34);
	  pfm_bv_set(pd[0].reg_smpl_pmds, 35);
	  pfm_bv_set(pd[0].reg_reset_pmds, 34);
	  return(1);
	}
      return(0);
    }
  else if (_perfmon2_pfm_pmu_type == PFMLIB_ITANIUM2_PMU)
    {
      if (pfm_mont_is_dear(native_index)) /* 2,3,17 */
	{
	  pfm_bv_set(pd[0].reg_smpl_pmds, 2);
	  pfm_bv_set(pd[0].reg_smpl_pmds, 3);
	  pfm_bv_set(pd[0].reg_smpl_pmds, 17);
	  pfm_bv_set(pd[0].reg_reset_pmds, 17);
	  return(1);
	}
      else if (pfm_mont_is_iear(native_index))  /* O,1 MK */
	{
	  pfm_bv_set(pd[0].reg_smpl_pmds, 0);
	  pfm_bv_set(pd[0].reg_smpl_pmds, 1);
	  pfm_bv_set(pd[0].reg_reset_pmds, 0);
	  return(1);
	}
      return(0);    
    }
#endif
  return(0);
}

static inline int process_smpl_entry(unsigned int native_pfm_index, int flags, pfm_dfl_smpl_entry_t **ent, _papi_hwi_context_t *ctx)
{
  SUBDBG("process_smpl_entry(%d,%d,%p,%p)\n",native_pfm_index,flags,ent,ctx);

#ifdef __ia64__
  /* Fixup EAR stuff here */      
  if (is_montecito_and_dear(native_pfm_index))
    {
      pfm_mont_pmd_reg_t data_addr; 
      pfm_mont_pmd_reg_t latency;
      pfm_mont_pmd_reg_t load_addr;
      unsigned long newent;
      
      if ((flags & (PAPI_PROFIL_DATA_EAR|PAPI_PROFIL_INST_EAR)) == 0)
	goto safety;

      /* Skip the header */
      ++(*ent);
      
      // PMD32 has data address on Montecito
      // PMD33 has latency on Montecito
      // PMD36 has instruction address on Montecito
      data_addr = *(pfm_mont_pmd_reg_t *)*ent;
      latency = *(pfm_mont_pmd_reg_t *)((unsigned long)*ent + sizeof(data_addr));
      load_addr = *(pfm_mont_pmd_reg_t *)((unsigned long)*ent + sizeof(data_addr)+sizeof(latency));
      
      SUBDBG("PMD[32]: 0x%016llx\n",(unsigned long long)data_addr.pmd_val);
      SUBDBG("PMD[33]: 0x%016llx\n",(unsigned long long)latency.pmd_val);
      SUBDBG("PMD[36]: 0x%016llx\n",(unsigned long long)load_addr.pmd_val);

      if ((!load_addr.pmd36_mont_reg.dear_vl) || (!load_addr.pmd33_mont_reg.dear_stat))
	{
	  SUBDBG("Invalid DEAR sample found, dear_vl = %d, dear_stat = 0x%x\n",
		 load_addr.pmd36_mont_reg.dear_vl,load_addr.pmd33_mont_reg.dear_stat);
	bail1:
	  newent = (unsigned long)*ent;
	  newent += 3*sizeof(pfm_mont_pmd_reg_t);
	  *ent = (pfm_dfl_smpl_entry_t *)newent;
	  return 0;
	}

      if (flags & PAPI_PROFIL_DATA_EAR)
	OVERFLOW_ADDRESS((*ctx)) = data_addr.pmd_val;
      else if (flags & PAPI_PROFIL_INST_EAR)
	{
	  unsigned long tmp = ((load_addr.pmd36_mont_reg.dear_iaddr + (unsigned long)load_addr.pmd36_mont_reg.dear_bn) << 4) | (unsigned long)load_addr.pmd36_mont_reg.dear_slot;
	  OVERFLOW_ADDRESS((*ctx)) = tmp;
	}
      else
	{
	  PAPIERROR("BUG!");
	  goto bail1;
	}

      newent = (unsigned long)*ent;
      newent += 3*sizeof(pfm_mont_pmd_reg_t);
      *ent = (pfm_dfl_smpl_entry_t *)newent;
      return 0;
    }
  else if (is_montecito_and_iear(native_pfm_index))
    {
      pfm_mont_pmd_reg_t latency;
      pfm_mont_pmd_reg_t icache_line_addr;
      unsigned long newent;
      
      if ((flags & PAPI_PROFIL_INST_EAR) == 0)
	goto safety;

      /* Skip the header */
      ++(*ent);
      
      // PMD34 has data address on Montecito
      // PMD35 has latency on Montecito
      icache_line_addr = *(pfm_mont_pmd_reg_t *)*ent;
      latency = *(pfm_mont_pmd_reg_t *)((unsigned long)*ent + sizeof(icache_line_addr));
     
      SUBDBG("PMD[34]: 0x%016llx\n",(unsigned long long)icache_line_addr.pmd_val);
      SUBDBG("PMD[35]: 0x%016llx\n",(unsigned long long)latency.pmd_val);

      if ((icache_line_addr.pmd34_mont_reg.iear_stat & 0x1) == 0)
	{
	  SUBDBG("Invalid IEAR sample found, iear_stat = 0x%x\n",icache_line_addr.pmd34_mont_reg.iear_stat);
	bail2:
	  newent = (unsigned long)*ent;
	  newent += 2*sizeof(pfm_mont_pmd_reg_t);
	  *ent = (pfm_dfl_smpl_entry_t *)newent;
	  return(0);
	}

      if (flags & PAPI_PROFIL_INST_EAR)
	{
	  unsigned long tmp = icache_line_addr.pmd34_mont_reg.iear_iaddr << 5;
	  OVERFLOW_ADDRESS((*ctx)) = tmp;
	}
      else 
      	{
	  PAPIERROR("BUG!");
	  goto bail2;
	}

      newent = (unsigned long)*ent;
      newent += 2*sizeof(pfm_mont_pmd_reg_t);
      *ent = (pfm_dfl_smpl_entry_t *)newent;
      return 0;
    }
  else if (is_itanium2_and_dear(native_pfm_index))
    {
      pfm_ita2_pmd_reg_t data_addr; 
      pfm_ita2_pmd_reg_t latency;
      pfm_ita2_pmd_reg_t load_addr;
      unsigned long newent;
      
      if ((flags & (PAPI_PROFIL_DATA_EAR|PAPI_PROFIL_INST_EAR)) == 0)
	goto safety;

      /* Skip the header */
      ++(*ent);
      
      // PMD2 has data address on Itanium 2
      // PMD3 has latency on Itanium 2
      // PMD17 has instruction address on Itanium 2
      data_addr = *(pfm_ita2_pmd_reg_t *)*ent;
      latency = *(pfm_ita2_pmd_reg_t *)((unsigned long)*ent + sizeof(data_addr));
      load_addr = *(pfm_ita2_pmd_reg_t *)((unsigned long)*ent + sizeof(data_addr)+sizeof(latency));
      
      SUBDBG("PMD[2]: 0x%016llx\n",(unsigned long long)data_addr.pmd_val);
      SUBDBG("PMD[3]: 0x%016llx\n",(unsigned long long)latency.pmd_val);
      SUBDBG("PMD[17]: 0x%016llx\n",(unsigned long long)load_addr.pmd_val);

      if ((!load_addr.pmd17_ita2_reg.dear_vl) || (!load_addr.pmd3_ita2_reg.dear_stat))
	{
	  SUBDBG("Invalid DEAR sample found, dear_vl = %d, dear_stat = 0x%x\n",load_addr.pmd17_ita2_reg.dear_vl,load_addr.pmd3_ita2_reg.dear_stat);
	bail3:
	  newent = (unsigned long)*ent;
	  newent += 3*sizeof(pfm_mont_pmd_reg_t);
	  *ent = (pfm_dfl_smpl_entry_t *)newent;
	  return 0;
	}

      if (flags & PAPI_PROFIL_DATA_EAR)
	OVERFLOW_ADDRESS((*ctx)) = data_addr.pmd_val;
      else if (flags & PAPI_PROFIL_INST_EAR)
	{
	  unsigned long tmp = ((load_addr.pmd17_ita2_reg.dear_iaddr + (unsigned long)load_addr.pmd17_ita2_reg.dear_bn) << 4) | (unsigned long)load_addr.pmd17_ita2_reg.dear_slot;
	  OVERFLOW_ADDRESS((*ctx)) = tmp;
	}
      else
	{
	  PAPIERROR("BUG!");
	  goto bail3;
	}
      
      newent = (unsigned long)*ent;
      newent += 3*sizeof(pfm_ita2_pmd_reg_t);
      *ent = (pfm_dfl_smpl_entry_t *)newent;
      return 0;
    }
  else if (is_itanium2_and_iear(native_pfm_index))
    {
      pfm_ita2_pmd_reg_t latency;
      pfm_ita2_pmd_reg_t icache_line_addr;
      unsigned long newent;
      
      if ((flags & PAPI_PROFIL_INST_EAR) == 0)
	goto safety;

      /* Skip the header */
      ++(*ent);
      
      // PMD0 has address on Itanium 2
      // PMD1 has latency on Itanium 2
      icache_line_addr = *(pfm_ita2_pmd_reg_t *)*ent;
      latency = *(pfm_ita2_pmd_reg_t *)((unsigned long)*ent + sizeof(icache_line_addr));
     
      SUBDBG("PMD[0]: 0x%016llx\n",(unsigned long long)icache_line_addr.pmd_val);
      SUBDBG("PMD[1]: 0x%016llx\n",(unsigned long long)latency.pmd_val);

      if ((icache_line_addr.pmd0_ita2_reg.iear_stat & 0x1) == 0) 
	{
	  SUBDBG("Invalid IEAR sample found, iear_stat = 0x%x\n",icache_line_addr.pmd0_ita2_reg.iear_stat);
	bail4:
	  newent = (unsigned long)*ent;
	  newent += 2*sizeof(pfm_mont_pmd_reg_t);
	  *ent = (pfm_dfl_smpl_entry_t *)newent;
	  return(0);
	}

      if (flags & PAPI_PROFIL_INST_EAR)
	{
	  unsigned long tmp = icache_line_addr.pmd0_ita2_reg.iear_iaddr << 5;
	  OVERFLOW_ADDRESS((*ctx)) = tmp;
	}
      else 
	{
	  PAPIERROR("BUG!");
	  goto bail4;
	}

      newent = (unsigned long)*ent;
      newent += 2*sizeof(pfm_ita2_pmd_reg_t);
      *ent = (pfm_dfl_smpl_entry_t *)newent;
      return 0;
    }
#if 0
  (is_btb(native_pfm_index))
    {
      // PMD48-63,39 on Montecito
      // PMD8-15,16 on Itanium 2
    }
#endif
  else
    safety:
#endif
    {
      OVERFLOW_ADDRESS((*ctx)) = (*ent)->ip;
      ++(*ent);
      return(1);
    }
}

static inline int
process_smpl_buf(int num_smpl_pmds, int entry_size, _papi_hwi_context_t *ctx, ThreadInfo_t **thr)
{
  pfm_dfl_smpl_entry_t *ent;
  size_t pos;
  uint64_t entry, count;
  pfm_dfl_smpl_hdr_t *hdr = (*thr)->context.smpl_buf;
  int ret, profile_index, flags;
  unsigned int native_pfm_index;

  DEBUGCALL(DEBUG_SUBSTRATE,dump_smpl_hdr(hdr));
  count = hdr->hdr_count;
  ent   = (pfm_dfl_smpl_entry_t *)(hdr+1);
  pos   = (unsigned long)ent;
  entry = 0;
  
  SUBDBG("This buffer has %llu samples in it.\n",(unsigned long long)count);
  while(count--) 
    {
      SUBDBG("Processing sample entry %llu\n",(unsigned long long)entry);
      DEBUGCALL(DEBUG_SUBSTRATE,dump_smpl(ent));

      /* Find the index of the profile buffers if we are profiling on many events */

      ret = find_profile_index((*thr)->running_eventset,ent->ovfl_pmd,&flags,&native_pfm_index,&profile_index);
      if (ret != PAPI_OK)
	return(ret);

      (*thr)->running_eventset->profile.overflowcount++;

      process_smpl_entry(native_pfm_index,flags,&ent,ctx);

      _papi_hwi_dispatch_profile((*thr)->running_eventset, ctx, (long_long)0,profile_index);
 
      entry++;
    }
  return(PAPI_OK);
}

/* This function only used when hardware overflows ARE working */

void _papi_hwd_dispatch_timer(int n, hwd_siginfo_t * info, void *uc)
{
   _papi_hwi_context_t ctx;
   pfm_msg_t msg;
   int ret, fd = info->si_fd;
   ThreadInfo_t *thread = _papi_hwi_lookup_thread();

#if defined(DEBUG)
   if (thread == NULL)
     {
       PAPIERROR("thread == NULL in _papi_hwd_dispatch_timer!");
       abort();
     }
#endif

   ctx.si = info;
   ctx.ucontext = (hwd_ucontext_t *)uc;

 retry:
   ret = read(fd, &msg, sizeof(msg));
   if (ret == -1)
     {
       if (errno == EINTR) 
	 {
	   SUBDBG("read(%d) interrupted, retrying\n", fd);
	   goto retry;
	 }
       else
	 {
	   PAPIERROR("read(%d): errno %d", fd, errno); 
	 }
     }
   else if (ret != sizeof(msg)) 
     {
       PAPIERROR("read(%d): short %d vs. %d bytes", ret, sizeof(msg)); 
       ret = -1;
     }
   
   if (msg.type != PFM_MSG_OVFL) 
     {
       PAPIERROR("unexpected msg type %d",msg.type);
       ret = -1;
     }

   if (ret != -1)
     {
       if (thread->running_eventset->state & PAPI_PROFILING)
	 process_smpl_buf(0, sizeof(pfm_dfl_smpl_entry_t), &ctx, &thread);
       else
	 {
	   OVERFLOW_ADDRESS(ctx) = msg.pfm_ovfl_msg.msg_ovfl_ip;
	   _papi_hwi_dispatch_overflow_signal((void *) &ctx, NULL, 
					      msg.pfm_ovfl_msg.msg_ovfl_pmds[0], 0, &thread);
	 }
     }

   if ((ret = pfm_restart(info->si_fd)))
     {
       PAPIERROR("pfm_restart(%d): %s", info->si_fd, pfm_strerror(ret));
     }
}

int _papi_hwd_stop_profiling(ThreadInfo_t * thread, EventSetInfo_t * ESI)
{
  _papi_hwi_context_t bogus;
  hwd_ucontext_t uc;
  hwd_siginfo_t si;
  
  memset(&bogus,0,sizeof(bogus));
  memset(&si,0,sizeof(si));
  memset(&uc,0,sizeof(uc));
  bogus.ucontext = &uc;
  bogus.si = &si;
  return(process_smpl_buf(0, sizeof(pfm_dfl_smpl_entry_t), &bogus, &thread));
}

int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
  hwd_control_state_t *ctl = &ESI->machdep;
  hwd_context_t *ctx = &ESI->master->context;
  pfarg_ctx_t newctx;
  void *buf_addr = NULL;
  pfm_dfl_smpl_arg_t buf_arg;
  pfm_dfl_smpl_hdr_t *hdr;
  int ret, ctx_fd;

  memset(&newctx, 0, sizeof(newctx));
  
  if (threshold == 0)
    {
      SUBDBG("PFM_CREATE_CONTEXT(%p,%p,%p,%d)\n",&newctx, NULL, NULL, 0);
      if ((ret = pfm_create_context(&newctx, NULL, NULL, 0)) == -1)
	{
	  PAPIERROR("pfm_create_context(%p,%p,%p,%d): %s",&newctx,NULL,NULL,0,pfm_strerror(ret));
	  return(PAPI_ESYS);
	}
      SUBDBG("PFM_CREATE_CONTEXT returns fd %d\n",ret);
      ctx_fd = ret;

      SUBDBG("MUNMAP(%p,%lld)\n",ctx->smpl_buf,(unsigned long long)ctx->smpl.buf_size);
      munmap(ctx->smpl_buf,ctx->smpl.buf_size);

      SUBDBG("CLOSE(%d)\n",ctx->ctx_fd);
      close(ctx->ctx_fd);

      ctx->ctx_fd = ctx_fd;
      memcpy(&ctx->ctx,&newctx,sizeof(newctx));
      memset(&ctx->smpl,0,sizeof(buf_arg));
      ctx->smpl_buf = NULL;
//      ESI->state &= ~(PAPI_OVERFLOWING);
//      ESI->overflow.flags &= ~(PAPI_OVERFLOW_HARDWARE);
      ESI->profile.overflowcount = 0;  
      return(PAPI_OK);
    }

  memset(&buf_arg, 0, sizeof(buf_arg));
//  newctx.ctx_flags = PFM_FL_NOTIFY_BLOCK;
  buf_arg.buf_size = 4*getpagesize();

  SUBDBG("PFM_CREATE_CONTEXT(%p,%s,%p,%d)\n",&newctx, PFM_DFL_SMPL_NAME, &buf_arg, (int)sizeof(buf_arg));
  if ((ret = pfm_create_context(&newctx, PFM_DFL_SMPL_NAME, &buf_arg, sizeof(buf_arg))) == -1)
    {
      DEBUGCALL(DEBUG_SUBSTRATE,dump_smpl_arg(&buf_arg));
      PAPIERROR("pfm_create_context(%p,%p,%p,%d): %s",&newctx,PFM_DFL_SMPL_NAME,&buf_arg,sizeof(buf_arg),pfm_strerror(ret));
      return(PAPI_ESYS);
    }
  ctx_fd = ret;
  SUBDBG("PFM_CREATE_CONTEXT returns fd %d\n",ctx_fd);

  SUBDBG("MMAP(NULL,%lld,%d,%d,%d,0)\n",(unsigned long long)buf_arg.buf_size,PROT_READ,MAP_PRIVATE,ctx_fd);
  buf_addr = mmap(NULL, (size_t)buf_arg.buf_size, PROT_READ, MAP_PRIVATE, ctx_fd, 0);
  if (buf_addr == MAP_FAILED)
    {
      PAPIERROR("mmap(NULL,%d,%d,%d,%d,0): %s",buf_arg.buf_size,PROT_READ,MAP_PRIVATE,ctx_fd,strerror(errno));
      close(ctx_fd);
      return(PAPI_ESYS);
    }
  SUBDBG("Sample buffer is located at %p\n",buf_addr);

  hdr = (pfm_dfl_smpl_hdr_t *)buf_addr;
  SUBDBG("hdr_cur_offs=%llu version=%u.%u\n",
	 (unsigned long long)hdr->hdr_cur_offs,
	 PFM_VERSION_MAJOR(hdr->hdr_version),
	 PFM_VERSION_MINOR(hdr->hdr_version));
  
  if (PFM_VERSION_MAJOR(hdr->hdr_version) < 1)
    {
      PAPIERROR("invalid buffer format version %d",PFM_VERSION_MAJOR(hdr->hdr_version));
      munmap(buf_addr,buf_arg.buf_size);
      close(ctx_fd);
      return(PAPI_ESBSTR);
    }

  ret = _papi_hwd_set_overflow(ESI,EventIndex,threshold);
  if (ret != PAPI_OK)
    {
      munmap(buf_addr,buf_arg.buf_size);
      close(ctx_fd);
      return(ret);
    }
  
  /* Look up the native event code */

  if (ESI->profile.flags & (PAPI_PROFIL_DATA_EAR|PAPI_PROFIL_INST_EAR))
    {
      pfarg_pmd_t *pd;
      int pos, native_index;
      pd = ctl->pd;
      pos = ESI->EventInfoArray[EventIndex].pos[0];
      native_index = ESI->NativeInfoArray[pos].ni_bits.event;
      setup_ear_event(native_index,&pd[pos],ESI->profile.flags);
    }

  /* Now close our context it is safe */

  close(ctx->ctx_fd);

  /* Copy the new data to the threads context control block */

  ctx->ctx_fd = ctx_fd;
  memcpy(&ctx->ctx,&newctx,sizeof(newctx));
  memcpy(&ctx->smpl,&buf_arg,sizeof(buf_arg));
  ctx->smpl_buf = buf_addr;

  /* We need overflowing because we use the overflow dispatch handler */
  
  ESI->state |= PAPI_OVERFLOWING;
  ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;

  return(PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   int j, retval = PAPI_OK, *pos;

   /* Which counter are we on, this looks suspicious because of the pos[0],
      but this could be because of derived events. We should do more here
      to figure out exactly what the position is, because the event may
      actually have more than one position. */
   
   pos = ESI->EventInfoArray[EventIndex].pos;
   j = pos[0];
   SUBDBG("Hardware counter %d used in overflow, threshold %d\n", j, threshold);

   if (threshold == 0) 
     {
      /* Remove the signal handler */

       retval = _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig);
       if (retval != PAPI_OK)
	 return(retval);

       /* Disable overflow */

       this_state->pd[j].reg_flags ^= PFM_REGFL_OVFL_NOTIFY;

	/*
	 * we may want to reset the other PMDs on
	 * every overflow. If we do not set
	 * this, the non-overflowed counters
	 * will be untouched.

	 if (inp.pfp_event_count > 1)
	 this_state->pd[j].reg_reset_pmds[0] ^= 1UL << counter_to_reset */

       /* Clear the overflow period */

      this_state->pd[j].reg_value = 0;
      this_state->pd[j].reg_long_reset = 0;
      this_state->pd[j].reg_short_reset = 0;

     } 
   else
     {
       /* Enable the signal handler */

       retval = _papi_hwi_start_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig, 1);
       if (retval != PAPI_OK)
	 return(retval);

       /* Set it to overflow */

       this_state->pd[j].reg_flags |= PFM_REGFL_OVFL_NOTIFY;
       
	/*
	 * we may want to reset the other PMDs on
	 * every overflow. If we do not set
	 * this, the non-overflowed counters
	 * will be untouched.

	 if (inp.pfp_event_count > 1)
	 this_state->pd[j].reg_reset_pmds[0] |= 1UL << counter_to_reset */
       
       /* Set the overflow period */

       this_state->pd[j].reg_value = - (unsigned long long) threshold;
       this_state->pd[j].reg_short_reset = - (unsigned long long) threshold;
       this_state->pd[j].reg_long_reset = - (unsigned long long) threshold;
     }
   return (retval);
}

static inline int decode_native_event(unsigned int EventCode, unsigned int *event, unsigned int *umask)
{
  unsigned int tevent, major, minor;

  tevent = EventCode & PAPI_NATIVE_AND_MASK;
  major = (tevent & PAPI_NATIVE_EVENT_AND_MASK) >> PAPI_NATIVE_EVENT_SHIFT;
  if (major >= _papi_hwi_system_info.sub_info.num_native_events)
    return(PAPI_ENOEVNT);

  minor = (tevent & PAPI_NATIVE_UMASK_AND_MASK) >> PAPI_NATIVE_UMASK_SHIFT;
  *event = major;
  *umask = minor;
  SUBDBG("EventCode 0x%08x is event %d, umask 0x%x\n",EventCode,major,minor);
  return(PAPI_OK);
}

static inline int prepare_umask(unsigned int foo,unsigned int *values)
{
  unsigned int tmp = foo, i, j = 0;

  SUBDBG("umask 0x%x\n",tmp);
  while ((i = ffs(tmp)))
    {
      tmp = tmp ^ (1 << (i-1));
      values[j] = i - 1;
      SUBDBG("umask %d is %d\n",j,values[j]);
      j++;
    }
  return(j);
}
      
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
  int ret;
  unsigned int event, umask;
  pfmlib_event_t gete;
  char name[PAPI_MAX_STR_LEN];

  memset(&gete,0,sizeof(gete));
  
  if (decode_native_event(EventCode,&event,&umask) != PAPI_OK)
    return("");
  
  gete.event = event;
  gete.num_masks = prepare_umask(umask,gete.unit_masks);
  
  ret = pfm_get_full_event_name(&gete,name,PAPI_MAX_STR_LEN);
  if (ret != PFMLIB_SUCCESS)
    {
      char tmp[PAPI_MAX_STR_LEN];
      pfm_get_event_name(gete.event,tmp,PAPI_MAX_STR_LEN);
      PAPIERROR("pfm_get_full_event_name(%p(event %d,%s,%d masks),%p,%d): %s",
		&gete,gete.event,tmp,gete.num_masks,name,PAPI_MAX_STR_LEN,pfm_strerror(ret));
      return("");
    }

  return(strdup(name));
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
  unsigned int event, umask;
  char *eventd, **maskd, *tmp;
  int i, ret, total_len = 0;
  pfmlib_event_t gete;

  memset(&gete,0,sizeof(gete));
  
  if (decode_native_event(EventCode,&event,&umask) != PAPI_OK)
    return(NULL);
  
  ret = pfm_get_event_description(event,&eventd);
  if (ret != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_event_description(%d,%p): %s",
		event,&eventd,pfm_strerror(ret));
      return(NULL);
    }

  if ((gete.num_masks = prepare_umask(umask,gete.unit_masks)))
    {
      maskd = (char **)malloc(gete.num_masks*sizeof(char *));
      if (maskd == NULL)
	{
	  free(eventd);
	  return(NULL);
	}
      for (i=0;i<gete.num_masks;i++)
	{
	  ret = pfm_get_event_mask_description(event,gete.unit_masks[i],&maskd[i]);
	  if (ret != PFMLIB_SUCCESS)
	    {
	      PAPIERROR("pfm_get_event_mask_description(%d,%d,%p): %s",
			event,umask,&maskd,pfm_strerror(ret));
	      free(eventd);
	      for (;i>=0;i--)
		free(maskd[i]);
	      free(maskd);
	      return(NULL);
	    }
	  total_len += strlen(maskd[i]);
	}
      tmp = (char *)malloc(strlen(eventd)+strlen(", masks:")+total_len+gete.num_masks+1);
      if (tmp == NULL)
	{
	  for (i=gete.num_masks-1;i>=0;i--)
	    free(maskd[i]);
	  free(maskd);
	  free(eventd);
	}
      tmp[0] = '\0';
      strcat(tmp,eventd);
      strcat(tmp,", masks:");
      for (i=0;i<gete.num_masks;i++)
	{
	  if (i!=0)
	    strcat(tmp,",");
	  strcat(tmp,maskd[i]);
	  free(maskd[i]);
	}
      free(maskd);
    }
  else
    {
      tmp = (char *)malloc(strlen(eventd)+1); 
      if (tmp == NULL)
	{
	  free(eventd);
	  return(NULL);
	}
      tmp[0] = '\0';
      strcat(tmp,eventd);
      free(eventd);
    }

  return(tmp);
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
  unsigned int event, umask, num_masks;
  int ret;

  if (decode_native_event(*EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);

  ret = pfm_get_num_event_masks(event,&num_masks);
  if (ret != PFMLIB_SUCCESS)
    {
      PAPIERROR("pfm_get_num_event_masks(%d,%p): %s",event,&num_masks,pfm_strerror(ret));
      return(PAPI_ENOEVNT);
    }
  SUBDBG("This is umask %d of %d\n",umask,num_masks);

  if (modifier == PAPI_ENUM_EVENTS)
    {
      if (event < _papi_hwi_system_info.sub_info.num_native_events - 1) 
	{
	  *EventCode += 1;
	  return (PAPI_OK);
	}
      return (PAPI_ENOEVNT);
    }
  else if (modifier == PAPI_ENUM_ALL)
    {
      if (umask+1 < (1<<num_masks))
	{
	  *EventCode = encode_native_event_raw(event,umask+1);
	  return (PAPI_OK);
	}
      else if (event < _papi_hwi_system_info.sub_info.num_native_events - 1) 
	{
	  /* Lookup event + 1 and return first umask of group */
	  ret = pfm_get_num_event_masks(event+1,&num_masks);
	  if (ret != PFMLIB_SUCCESS)
	    {
	      PAPIERROR("pfm_get_num_event_masks(%d,%p): %s",event,&num_masks,pfm_strerror(ret));
	      return(PAPI_ENOEVNT);
	    }
	  if (num_masks)
	    *EventCode = encode_native_event_raw(event+1,1);
	  else
	    *EventCode = encode_native_event_raw(event+1,0);
	  return(PAPI_OK);
	}
      return (PAPI_ENOEVNT);
    }
  else if (modifier == PAPI_ENUM_UMASK_COMBOS)
    {
      if (umask+1 < (1<<num_masks))
	{
	  *EventCode = encode_native_event_raw(event,umask+1);
	  return (PAPI_OK);
	}
      return(PAPI_ENOEVNT);
    }
  else if (modifier == PAPI_ENUM_UMASKS)
    {
      int thisbit = ffs(umask);
      
      SUBDBG("First bit is %d in %08x\b",thisbit-1,umask);
      thisbit = 1 << thisbit;

      if (thisbit & ((1<<num_masks)-1))
	{
	  *EventCode = encode_native_event_raw(event,thisbit);
	  return (PAPI_OK);
	}
      return(PAPI_ENOEVNT);
    }
  else
    return(PAPI_EINVAL);
  
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
  unsigned int event, umask;
  pfmlib_event_t gete;

  /* For PFM & Perfmon, native info is just an index into the PFM event table. */
  if (decode_native_event(EventCode,&event,&umask) != PAPI_OK)
    return(PAPI_ENOEVNT);

  memset(&gete,0x0,sizeof(gete));

  gete.event = event;
  gete.num_masks = prepare_umask(umask,gete.unit_masks);
  
  memcpy(bits,&gete,sizeof(*bits));
  return (PAPI_OK);
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
  int ret;
  pfmlib_regmask_t selector = _perfmon2_pfm_native_map[bits->event].resources.selector;
  int j, n = _papi_hwi_system_info.sub_info.num_cntrs;
  int foo, did_something=0;
  for (j=0;n;j++)
    {
      if (pfm_regmask_isset(&selector,j))
	{
	  n--;
	  if ((ret = pfm_get_event_code_counter(bits->event,j,&foo)) != PFMLIB_SUCCESS)
	    {
	      PAPIERROR("pfm_get_event_code_counter(%d,%d,%p): %s",*bits,j,&foo,pfm_strerror(ret));
	      return(PAPI_EBUG);
	    }
	  values[did_something] = foo;
	  strncpy(&names[did_something*name_len],"Event Code",name_len);
	  did_something++;
	}
    }
  return(did_something);
}

int _papi_hwd_init_control_state(hwd_control_state_t *ctl)
{
  pfmlib_input_param_t *inp = &ctl->in;
  pfmlib_output_param_t *outp = &ctl->out;
  pfarg_pmd_t *pd = ctl->pd;
  pfarg_pmc_t *pc = ctl->pc;
  pfarg_setdesc_t *set = ctl->set;
  pfarg_setinfo_t *setinfo = ctl->setinfo;

  memset(inp,0,sizeof(*inp));
  memset(outp,0,sizeof(*inp));
  memset(pc,0,sizeof(ctl->pc));
  memset(pd,0,sizeof(ctl->pd));
  memset(set,0,sizeof(ctl->set));
  memset(setinfo,0,sizeof(ctl->setinfo));
  /* Will be filled by update now...until this gets another arg */
  ctl->ctx = NULL;
  ctl->ctx_fd = -1;
  ctl->load = NULL;
  set_domain(ctl,_papi_hwi_system_info.sub_info.default_domain);
  return(PAPI_OK);
}

int _papi_hwd_allocate_registers(EventSetInfo_t * ESI)
{
  int i, j;
  for (i = 0; i < ESI->NativeCount; i++) 
    {
      if (_papi_hwd_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_event,&ESI->NativeInfoArray[i].ni_bits) != PAPI_OK)
	goto bail;
    }
  return(1);
 bail:
  for (j = 0; j < i; j++) 
    memset(&ESI->NativeInfoArray[j].ni_bits,0x0,sizeof(ESI->NativeInfoArray[j].ni_bits));
  return(0);
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

int _papi_hwd_update_control_state(hwd_control_state_t *ctl,
                                   NativeInfo_t *native, int count, hwd_context_t * ctx) {
  int i = 0, ret;
  int last_reg_set = 0, reg_set_done = 0, offset = 0;
  pfmlib_input_param_t tmpin,*inp = &ctl->in;
  pfmlib_output_param_t tmpout,*outp = &ctl->out;
  pfarg_pmd_t *pd = ctl->pd;

  if (count == 0)
    {
      SUBDBG("Called with count == 0\n");
      inp->pfp_event_count = 0;
      outp->pfp_pmc_count = 0;
      memset(inp->pfp_events,0x0,sizeof(inp->pfp_events));
      return(PAPI_OK);
    }
  
  memcpy(&tmpin,inp,sizeof(tmpin));
  memcpy(&tmpout,outp,sizeof(tmpout));

  for (i=0;i<count;i++)
    {
      SUBDBG("Stuffing native event index %d (code 0x%x) into input structure.\n",
	     i,native[i].ni_bits.event);
      memcpy(inp->pfp_events+i,&(native[i].ni_bits),sizeof(pfmlib_event_t));
    }
  inp->pfp_event_count = count;
      
  /* let the library figure out the values for the PMCS */
  
  ret = compute_kernel_args(ctl);
  if (ret != PAPI_OK)
    {
      /* Restore values */
      memcpy(inp,&tmpin,sizeof(tmpin));
      memcpy(outp,&tmpout,sizeof(tmpout));
      return(ret);
    }
  
  /* Update the native structure, because the allocation is done here. */
  
  last_reg_set = pd[0].reg_set;
  for (i=0;i<count;i++)
    {
      if (pd[i].reg_set != last_reg_set)
	{
	  offset += reg_set_done;
	  reg_set_done = 0;
	}
      reg_set_done++;

      native[i].ni_position = i;
      SUBDBG("native event index %d (code 0x%x) is at PMD offset %d\n", i, native[i].ni_bits.event, native[i].ni_position);
    }
      
   /* If structure has not yet been filled with a context, fill it
      from the thread's context. This should happen in init_control_state
      when we give that a *ctx argument */

  if (ctl->ctx == NULL)
    {
      ctl->ctx = &ctx->ctx;
      ctl->ctx_fd = ctx->ctx_fd;
      ctl->load = &ctx->load;
    }

   return (PAPI_OK);
}
