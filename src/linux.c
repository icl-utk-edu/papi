/* 
* File:    linunx.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

/* Prototypes */
static int mdi_init();
extern int setup_p4_presets(int cputype);
extern int setup_p4_vector_table(papi_vectors_t *);
extern int setup_p3_presets(int cputype);
extern int setup_p3_vector_table(papi_vectors_t *);


/* This should be in a linux.h header file maybe. */
#define FOPEN_ERROR "fopen(%s) returned NULL"

#if defined(PERFCTR26)
#define PERFCTR_CPU_NAME(pi)    perfctr_info_cpu_name(pi)
#define PERFCTR_CPU_NRCTRS(pi)  perfctr_info_nrctrs(pi)
#elif defined(PERFCTR25)
#define PERFCTR_CPU_NAME        perfctr_info_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME        perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_cpu_nrctrs
#endif

inline_static int xlate_cpu_type_to_vendor(unsigned perfctr_cpu_type) {
   switch (perfctr_cpu_type) {
   case PERFCTR_X86_INTEL_P5:
   case PERFCTR_X86_INTEL_P5MMX:
   case PERFCTR_X86_INTEL_P6:
   case PERFCTR_X86_INTEL_PII:
   case PERFCTR_X86_INTEL_PIII:
   case PERFCTR_X86_INTEL_P4:
   case PERFCTR_X86_INTEL_P4M2:
#ifdef PERFCTR_X86_INTEL_P4M3
   case PERFCTR_X86_INTEL_P4M3:
#endif
#ifdef PERFCTR_X86_INTEL_PENTM
   case PERFCTR_X86_INTEL_PENTM:
#endif
      return (PAPI_VENDOR_INTEL);
#ifdef PERFCTR_X86_AMD_K8
   case PERFCTR_X86_AMD_K8:
#endif
#ifdef PERFCTR_X86_AMD_K8C
   case PERFCTR_X86_AMD_K8C:
#endif
   case PERFCTR_X86_AMD_K7:
      return (PAPI_VENDOR_AMD);
   case PERFCTR_X86_CYRIX_MII:
      return (PAPI_VENDOR_CYRIX);
   default:
      return (PAPI_VENDOR_UNKNOWN);
   }
}

/* volatile uint32_t lock; */

#include <inttypes.h>

volatile unsigned int lock[PAPI_MAX_LOCK];


static void lock_init(void) {
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      lock[i] = MUTEX_OPEN;
   }
}

/* 
 * 1 if the processor is a P4, 0 otherwise
 */
int check_p4(int cputype){
  switch(cputype) {
     case PERFCTR_X86_INTEL_P4:
     case PERFCTR_X86_INTEL_P4M2:
#ifdef PERFCTR_X86_INTEL_P4M3
     case PERFCTR_X86_INTEL_P4M3:
#endif
        return(1);
     default:
	return(0);
  }
  return(0);
}

#ifndef PAPI_NO_VECTOR
papi_svector_t _linux_os_table[] = {
 {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 { NULL, VEC_PAPI_END}
};
#endif

int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
  int retval;
  struct perfctr_info info;
  struct vperfctr *dev;
  int is_p4=0;

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _linux_os_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

  
   retval = mdi_init();
   if ( retval ) 
     return(retval);

   /* Opened once for all threads. */
   if ((dev = vperfctr_open()) == NULL)
     { PAPIERROR( VOPEN_ERROR); return(PAPI_ESYS); }
   SUBDBG("_papi_hwd_init_global vperfctr_open = %p\n", dev);

   /* Get info from the kernel */

   if (vperfctr_info(dev, &info) < 0)
     { PAPIERROR( VINFO_ERROR); return(PAPI_ESYS); }

  /* Fill in what we can of the papi_system_info. */
  retval = _papi_hwd_get_system_info();
  if (retval != PAPI_OK)
     return (retval);

   /* Setup memory info */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, (int) info.cpu_type);
   if (retval)
      return (retval);

   is_p4 = check_p4(info.cpu_type);

   /* Setup presets */
   if ( is_p4 ){
     strcpy(_papi_hwi_system_info.substrate, "$Id$");
     retval = setup_p4_vector_table(vtable);
     retval = setup_p4_presets(info.cpu_type);
     if ( retval ) 
       return(retval);
   }
   else{
     strcpy(_papi_hwi_system_info.substrate, "$Id$");
     retval = setup_p3_vector_table(vtable);
     retval = setup_p3_presets(info.cpu_type);
     if ( retval ) 
       return(retval);
   }

   if ( retval ) 
     return(retval);

   /* Fixup stuff from linux.c */

   strcpy(_papi_hwi_system_info.hw_info.model_string, PERFCTR_CPU_NAME(&info));

   _papi_hwi_system_info.supports_hw_overflow =
       (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;
   SUBDBG("Hardware/OS %s support counter generated interrupts\n",
          _papi_hwi_system_info.supports_hw_overflow ? "does" : "does not");

   _papi_hwi_system_info.num_cntrs = PERFCTR_CPU_NRCTRS(&info);
   _papi_hwi_system_info.num_gp_cntrs = PERFCTR_CPU_NRCTRS(&info);
   _papi_hwi_system_info.hw_info.model = info.cpu_type;
   _papi_hwi_system_info.hw_info.vendor = xlate_cpu_type_to_vendor(info.cpu_type);

    SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n", dev);
    vperfctr_close(dev);

   lock_init();

   return (PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
   extern int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain);
   switch (code) {
   case PAPI_DOMAIN:
   case PAPI_DEFDOM:
      return (_papi_hwd_set_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_GRANUL:
   case PAPI_DEFGRN:
      return(PAPI_ESBSTR);
   default:
      return (PAPI_EINVAL);
   }
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *context)
{
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;
   int isHardware=0;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

   _papi_hwi_dispatch_overflow_signal((void *) &ctx, &isHardware,
                                      si->si_pmc_ovf_mask, 0, &master);

   /* We are done, resume interrupting counters */

   if (isHardware) {
      if (vperfctr_iresume(master->context.perfctr) < 0) {
         PAPIERROR("vperfctr_iresume errno %d",errno);
      }
   }
}

int _papi_hwd_init(hwd_context_t * ctx) {
   struct vperfctr_control tmp;

   /* Initialize our thread/process pointer. */
   if ((ctx->perfctr = vperfctr_open()) == NULL)
     { PAPIERROR( VOPEN_ERROR); return(PAPI_ESYS); }
   SUBDBG("_papi_hwd_init vperfctr_open() = %p\n", ctx->perfctr);

   /* Initialize the per thread/process virtualized TSC */
   memset(&tmp, 0x0, sizeof(tmp));
   tmp.cpu_control.tsc_on = 1;

   /* Start the per thread/process virtualized TSC */
   if (vperfctr_control(ctx->perfctr, &tmp) < 0)
     { PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS); }

   return (PAPI_OK);
}


/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
static int mdi_init() {
     /* Name of the substrate we're using */
    strcpy(_papi_hwi_system_info.substrate, "$Id$");

   _papi_hwi_system_info.supports_hw_overflow = 1;
   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_inheritance = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;
   _papi_hwi_system_info.supports_virt_usec = 1;
   _papi_hwi_system_info.supports_virt_cyc = 1;

   return (PAPI_OK);
}

int _papi_hwd_update_shlib_info(void)
{
   char fname[PAPI_HUGE_STR_LEN];
   PAPI_address_map_t *tmp, *tmp2;
   FILE *f;
   char find_data_mapname[PAPI_HUGE_STR_LEN] = "";
   int upper_bound = 0, i, index = 0, find_data_index = 0, count = 0;
   char buf[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], perm[5], dev[6], mapname[PAPI_HUGE_STR_LEN];
   unsigned long begin, end, size, inode, foo;

   sprintf(fname, "/proc/%ld/maps", (long)_papi_hwi_system_info.pid);
   f = fopen(fname, "r");

   if (!f)
     { 
	 PAPIERROR("fopen(%s) returned < 0", fname); 
	 return(PAPI_OK); 
     }

   /* First count up things that look kinda like text segments, this is an upper bound */

   while (1)
     {
      if (fgets(buf, sizeof(buf), f) == NULL)
	{
	  if (ferror(f))
	    {
	      PAPIERROR("fgets(%s, %d) returned < 0", fname, sizeof(buf)); 
	      fclose(f);
	      return(PAPI_OK); 
	    }
	  else
	    break;
	}

      sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);

      if (strlen(mapname) && (perm[0] == 'r') && (perm[1] != 'w') && (perm[2] == 'x') && (inode != 0))
	{
	  upper_bound++;
	}
     }
   if (upper_bound == 0)
     {
       PAPIERROR("No segments found with r-x, inode != 0 and non-NULL mapname"); 
       fclose(f);
       return(PAPI_OK); 
     }

   /* Alloc our temporary space */

   tmp = (PAPI_address_map_t *) papi_calloc(upper_bound, sizeof(PAPI_address_map_t));
   if (tmp == NULL)
     {
       PAPIERROR("calloc(%d) failed", upper_bound*sizeof(PAPI_address_map_t));
       fclose(f);
       return(PAPI_OK);
     }
      
   rewind(f);
   while (1)
     {
      if (fgets(buf, sizeof(buf), f) == NULL)
	{
	  if (ferror(f))
	    {
	      PAPIERROR("fgets(%s, %d) returned < 0", fname, sizeof(buf)); 
	      fclose(f);
	      papi_free(tmp);
	      return(PAPI_OK); 
	    }
	  else
	    break;
	}

      sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);
      size = end - begin;

      if (strlen(mapname) == 0)
	continue;

      if ((strcmp(find_data_mapname,mapname) == 0) && (perm[0] == 'r') && (perm[1] == 'w') && (inode != 0))
	{
	  tmp[find_data_index].data_start = (caddr_t) begin;
	  tmp[find_data_index].data_end = (caddr_t) (begin + size);
	  find_data_mapname[0] = '\0';
	}
      else if ((perm[0] == 'r') && (perm[1] != 'w') && (perm[2] == 'x') && (inode != 0))
	{
	  /* Text segment, check if we've seen it before, if so, ignore it. Some entries
	     have multiple r-xp entires. */

	  for (i=0;i<upper_bound;i++)
	    {
	      if (strlen(tmp[i].name))
		{
		  if (strcmp(mapname,tmp[i].name) == 0)
		    break;
		}
	      else
		{
		  /* Record the text, and indicate that we are to find the data segment, following this map */
		  strcpy(tmp[i].name,mapname);
		  tmp[i].text_start = (caddr_t) begin;
		  tmp[i].text_end = (caddr_t) (begin + size);
		  count++;
		  strcpy(find_data_mapname,mapname);
		  find_data_index = i;
		  break;
		}
	    }
	}
     }
   if (count == 0)
     {
       PAPIERROR("No segments found with r-x, inode != 0 and non-NULL mapname"); 
       fclose(f);
       papi_free(tmp);
       return(PAPI_OK); 
     }
   fclose(f);

   /* Now condense the list and update exe_info */
   tmp2 = (PAPI_address_map_t *) papi_calloc(count, sizeof(PAPI_address_map_t));
   if (tmp2 == NULL)
     {
       PAPIERROR("calloc(%d) failed", count*sizeof(PAPI_address_map_t));
       papi_free(tmp);
       fclose(f);
       return(PAPI_OK);
     }

   for (i=0;i<count;i++)
     {
       if (strcmp(tmp[i].name,_papi_hwi_system_info.exe_info.fullname) == 0)
	 {
	   _papi_hwi_system_info.exe_info.address_info.text_start = tmp[i].text_start;
	   _papi_hwi_system_info.exe_info.address_info.text_end = tmp[i].text_end;
	   _papi_hwi_system_info.exe_info.address_info.data_start = tmp[i].data_start;
	   _papi_hwi_system_info.exe_info.address_info.data_end = tmp[i].data_end;
	 }
       else
	 {
	   strcpy(tmp2[index].name,tmp[i].name);
	   tmp2[index].text_start = tmp[i].text_start;
	   tmp2[index].text_end = tmp[i].text_end;
	   tmp2[index].data_start = tmp[i].data_start;
	   tmp2[index].data_end = tmp[i].data_end;
	   index++;
	 }
     }
   papi_free(tmp);

   if (_papi_hwi_system_info.shlib_info.map)
     papi_free(_papi_hwi_system_info.shlib_info.map);
   _papi_hwi_system_info.shlib_info.map = tmp2;
   _papi_hwi_system_info.shlib_info.count = index;

   return (PAPI_OK);
}

static char *search_cpu_info(FILE * f, char *search_str, char *line)
{
   /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
   /* See the PCL home page for the German version of PAPI. */

   char *s;

   while (fgets(line, 256, f) != NULL) {
      if (strncmp(line, search_str, strlen(search_str)) == 0) {
         /* ignore all characters in line up to : */
         for (s = line; *s && (*s != ':'); ++s);
         if (*s)
            return (s);
      }
   }
   return (NULL);

   /* End stolen code */
}

/* Pentium III
 * processor  : 1
 * vendor     : GenuineIntel
 * arch       : IA-64
 * family     : Itanium 2
 * model      : 0
 * revision   : 7
 * archrev    : 0
 * features   : branchlong
 * cpu number : 0
 * cpu regs   : 4
 * cpu MHz    : 900.000000
 * itc MHz    : 900.000000
 * BogoMIPS   : 1346.37
 * */
/* IA64
 * processor       : 1
 * vendor_id       : GenuineIntel
 * cpu family      : 6
 * model           : 7
 * model name      : Pentium III (Katmai)
 * stepping        : 3
 * cpu MHz         : 547.180
 * cache size      : 512 KB
 * physical id     : 0
 * siblings        : 1
 * fdiv_bug        : no
 * hlt_bug         : no
 * f00f_bug        : no
 * coma_bug        : no
 * fpu             : yes
 * fpu_exception   : yes
 * cpuid level     : 2
 * wp              : yes
 * flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 mmx fxsr sse
 * bogomips        : 1091.17
 * */

int _papi_hwd_get_system_info(void)
{
   int tmp, retval;
   char maxargs[PAPI_HUGE_STR_LEN], *t, *s;
   pid_t pid;
   float mhz = 0.0;
   FILE *f;

   /* Software info */

   /* Path and args */

   pid = getpid();
   if (pid < 0)
     { PAPIERROR("getpid() returned < 0"); return(PAPI_ESYS); }
   _papi_hwi_system_info.pid = pid;

   sprintf(maxargs, "/proc/%d/exe", (int) pid);
   if (readlink(maxargs, _papi_hwi_system_info.exe_info.fullname, PAPI_HUGE_STR_LEN) < 0)
   { 
       PAPIERROR("readlink(%s) returned < 0", maxargs); 
       strcpy(_papi_hwi_system_info.exe_info.fullname,"");
       strcpy(_papi_hwi_system_info.exe_info.address_info.name,"");
   }
   else
   {
   /* basename can modify it's argument */
   strcpy(maxargs,_papi_hwi_system_info.exe_info.fullname);
   strcpy(_papi_hwi_system_info.exe_info.address_info.name, basename(maxargs));
   }

   /* Executable regions, may require reading /proc/pid/maps file */

   retval = _papi_hwd_update_shlib_info();

   /* PAPI_preload_option information */

   strcpy(_papi_hwi_system_info.preload_info.lib_preload_env, "LD_PRELOAD");
   _papi_hwi_system_info.preload_info.lib_preload_sep = ' ';
   strcpy(_papi_hwi_system_info.preload_info.lib_dir_env, "LD_LIBRARY_PATH");
   _papi_hwi_system_info.preload_info.lib_dir_sep = ':';

   SUBDBG("Executable is %s\n", _papi_hwi_system_info.exe_info.address_info.name);
   SUBDBG("Full Executable is %s\n", _papi_hwi_system_info.exe_info.fullname);
   SUBDBG("Text: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.text_start,
          _papi_hwi_system_info.exe_info.address_info.text_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.text_end -
          _papi_hwi_system_info.exe_info.address_info.text_start));
   SUBDBG("Data: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.data_start,
          _papi_hwi_system_info.exe_info.address_info.data_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.data_end -
          _papi_hwi_system_info.exe_info.address_info.data_start));
   SUBDBG("Bss: Start %p, End %p, length %d\n",
          _papi_hwi_system_info.exe_info.address_info.bss_start,
          _papi_hwi_system_info.exe_info.address_info.bss_end,
          (int)(_papi_hwi_system_info.exe_info.address_info.bss_end -
          _papi_hwi_system_info.exe_info.address_info.bss_start));

   /* Hardware info */

   _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
   _papi_hwi_system_info.hw_info.vendor = -1;

   if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
     { PAPIERROR("fopen(/proc/cpuinfo) errno %d",errno); return(PAPI_ESYS); }

   /* All of this information maybe overwritten by the substrate */ 

   /* MHZ */

   rewind(f);
   s = search_cpu_info(f, "cpu MHz", maxargs);
   if (s)
      sscanf(s + 1, "%f", &mhz);
   _papi_hwi_system_info.hw_info.mhz = mhz;

   /* Vendor Name */

   rewind(f);
   s = search_cpu_info(f, "vendor_id", maxargs);
   if (s && (t = strchr(s + 2, '\n'))) 
     {
      *t = '\0';
      strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
     }
   else 
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) {
	 *t = '\0';
	 strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
       }
     }
       
   /* Revision */

   rewind(f);
   s = search_cpu_info(f, "stepping", maxargs);
   if (s)
      {
	sscanf(s + 1, "%d", &tmp);
	_papi_hwi_system_info.hw_info.revision = (float) tmp;
      }
   else
     {
       rewind(f);
       s = search_cpu_info(f, "revision", maxargs);
       if (s)
	 {
	   sscanf(s + 1, "%d", &tmp);
	   _papi_hwi_system_info.hw_info.revision = (float) tmp;
	 }
     }

   /* Model Name */

   rewind(f);
   s = search_cpu_info(f, "family", maxargs);
   if (s && (t = strchr(s + 2, '\n'))) 
     {
       *t = '\0';
       strcpy(_papi_hwi_system_info.hw_info.model_string, s + 2);
     }
   else 
     {
       rewind(f);
       s = search_cpu_info(f, "vendor", maxargs);
       if (s && (t = strchr(s + 2, '\n'))) 
	 {
	   *t = '\0';
	   strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
	 }
     }

   rewind(f);
   s = search_cpu_info(f, "model", maxargs);
   if (s)
      {
	sscanf(s + 1, "%d", &tmp);
	_papi_hwi_system_info.hw_info.model = tmp;
      }

   fclose(f);

   SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
          _papi_hwi_system_info.hw_info.totalcpus,
          _papi_hwi_system_info.hw_info.vendor_string,
          _papi_hwi_system_info.hw_info.vendor,
          _papi_hwi_system_info.hw_info.model_string,
          _papi_hwi_system_info.hw_info.model, _papi_hwi_system_info.hw_info.mhz);

   return (PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

inline_static long_long get_cycles(void) {
   long_long ret;
#ifdef __x86_64__
   do {
      unsigned int a,d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      (ret) = ((unsigned long)a) | (((unsigned long)d)<<32);
   } while(0);
#else
   __asm__ __volatile__("rdtsc"
                       : "=A" (ret)
                       : /* no inputs */);
#endif
   return ret;
}

long_long _papi_hwd_get_real_usec(void) {
   return((long_long)get_cycles() / (long_long)_papi_hwi_system_info.hw_info.mhz);
}

long_long _papi_hwd_get_real_cycles(void) {
   return((long_long)get_cycles());
}

long_long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
   return ((long_long)vperfctr_read_tsc(ctx->perfctr));
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   return ((long_long)vperfctr_read_tsc(ctx->perfctr) /
           (long_long)_papi_hwi_system_info.hw_info.mhz);
}

