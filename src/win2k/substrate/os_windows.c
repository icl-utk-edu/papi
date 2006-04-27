/* 
* File:    os_windows.c
* CVS:     $Id$
* Author:  dan terpstra
*          terpstra@cs.utk.edu
* Mods:    <name>
*		   <email address>
*
*		   This file is heavily derived from linux.c circa 03/2006
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
CRITICAL_SECTION lock[PAPI_MAX_LOCK];

static void lock_init(void)
{
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      InitializeCriticalSection(&lock[i]);
   }
}

static void lock_release(void)
{
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      DeleteCriticalSection(&lock[i]);
   }
}

#ifndef PAPI_NO_VECTOR
papi_svector_t _windows_os_table[] = {
 #ifndef __CATAMOUNT__
   {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 #endif
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
/****WIN32: this callback requires a totally different structure for Windows
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
*/
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
  int cpu_type;

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _windows_os_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

   retval = mdi_init();
   if ( retval ) 
     return(retval);

  /* Fill in what we can of the papi_system_info. */
  retval = _papi_hwd_get_system_info();
  if (retval != PAPI_OK)
     return (retval);
  cpu_type = (int) _papi_hwi_system_info.hw_info.vendor;

   /* Setup memory info */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, cpu_type);
   if (retval)
      return (retval);

   strcpy(_papi_hwi_system_info.substrate, "$Id$");

   /* Setup presets */
   if ( check_p4(cpu_type) ){
     retval = setup_p4_vector_table(vtable);
     if (!retval)
     	retval = setup_p4_presets(cpu_type);
   }
   else{
     retval = setup_p3_vector_table(vtable);
     if (!retval)
     	retval = setup_p3_presets(cpu_type);
   }
   if ( retval ) 
     return(retval);

   /* Fixup stuff from os_windows.c */

/****WIN32: all this stuff comes out of the info structure;
	we should emulate it if we need it.

   strcpy(_papi_hwi_system_info.hw_info.model_string, PERFCTR_CPU_NAME(&info));

   _papi_hwi_system_info.supports_hw_overflow =
       (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;
   SUBDBG("Hardware/OS %s support counter generated interrupts\n",
          _papi_hwi_system_info.supports_hw_overflow ? "does" : "does not");

   _papi_hwi_system_info.num_cntrs = PERFCTR_CPU_NRCTRS(&info);
   _papi_hwi_system_info.num_gp_cntrs = PERFCTR_CPU_NRCTRS(&info);
   _papi_hwi_system_info.hw_info.model = info.cpu_type;
   _papi_hwi_system_info.hw_info.vendor = xlate_cpu_type_to_vendor(cpu_type);
*/

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


void CALLBACK _papi_hwd_timer_callback(UINT wTimerID, UINT msg, 
        DWORD dwUser, DWORD dw1, DWORD dw2) 
{
    _papi_hwi_context_t ctx;
    CONTEXT	context;	// processor specific context structure
    HANDLE	threadHandle;
    BOOL	error;
    ThreadInfo_t *master = NULL;
    int isHardware=0;

#define OVERFLOW_MASK 0
#define GEN_OVERFLOW 1

   ctx.ucontext = &context;

   // dwUser is the threadID passed by timeSetEvent
    // NOTE: This call requires W2000 or later
    threadHandle = OpenThread(THREAD_GET_CONTEXT, FALSE, dwUser);

    // retrieve the contents of the control registers only
    context.ContextFlags = CONTEXT_CONTROL;
    error = GetThreadContext(threadHandle, &context);
    CloseHandle(threadHandle);

    // pass a void pointer to cpu register data here
    _papi_hwi_dispatch_overflow_signal((void *)&ctx, &isHardware, 
										OVERFLOW_MASK, GEN_OVERFLOW, &master); 
}

HANDLE pmc_dev;	// device handle for kernel driver

/* Called once per process. */

int _papi_hwd_shutdown_global(void) {
  pmc_close(pmc_dev);
  lock_release();
   return (PAPI_OK);
}

int _papi_hwd_init(hwd_context_t *ctx)
{
   /* Initialize our thread/process pointer. */
   if ((ctx->self = pmc_dev = pmc_open()) == NULL) {
      PAPIERROR("pmc_open() returned NULL"); 
      return(PAPI_ESYS);
   }
   SUBDBG("_papi_hwd_init pmc_open() = %p\n", ctx->self);

   /* Linux makes sure that each thread has a virtualized TSC here.
      This makes no sense on Windows, since the counters aren't
      saved at context switch.
   */

   return(PAPI_OK);
}

/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
static int mdi_init() {
     /* Name of the substrate we're using */
    strcpy(_papi_hwi_system_info.substrate, "$Id$");
   _papi_hwi_system_info.supports_hw_overflow = 0;
   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_inheritance = 0;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;
   _papi_hwi_system_info.supports_virt_usec = 0;
   _papi_hwi_system_info.supports_virt_cyc = 0;

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

// split the filename from a full path
// roughly equivalent to unix basename()
static void splitpath(const char *path, char *name)
{
        short i = 0, last = 0;

        while (path[i]) {
                if (path[i] == '\\') last = i;
                i++;
        }
        name[0] = 0;
        i = i - last;
        if (last > 0) {
                last++;
                i--;
        }
        strncpy(name, &path[last], i);
        name[i] = 0;
}

int _papi_hwd_get_system_info(void)
{
  struct wininfo win_hwinfo;
  HMODULE hModule;
  DWORD len;
  long i = 0;

  /* Path and args */
  _papi_hwi_system_info.pid = getpid();

  hModule = GetModuleHandle(NULL); // current process
  len = GetModuleFileName(hModule,_papi_hwi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN);
  if (len) splitpath(_papi_hwi_system_info.exe_info.fullname, _papi_hwi_system_info.exe_info.address_info.name);
  else return(PAPI_ESYS);

  SUBDBG("Executable is %s\n",_papi_hwi_system_info.exe_info.address_info.name);
  SUBDBG("Full Executable is %s\n",_papi_hwi_system_info.exe_info.fullname);
  /* Hardware info */
  if (!init_hwinfo(&win_hwinfo))
    return(PAPI_ESYS);

  _papi_hwi_system_info.hw_info.ncpu = win_hwinfo.ncpus;
  _papi_hwi_system_info.hw_info.nnodes = win_hwinfo.nnodes;
  _papi_hwi_system_info.hw_info.totalcpus = win_hwinfo.total_cpus;

  _papi_hwi_system_info.hw_info.vendor = win_hwinfo.vendor;
  _papi_hwi_system_info.hw_info.revision = (float)win_hwinfo.revision;
  strcpy(_papi_hwi_system_info.hw_info.vendor_string,win_hwinfo.vendor_string);

  /* initialize the model to something */
  _papi_hwi_system_info.hw_info.model = PERFCTR_X86_GENERIC;

  if (IS_P3(&win_hwinfo) || IS_P3_XEON(&win_hwinfo) || IS_CELERON(&win_hwinfo))
    _papi_hwi_system_info.hw_info.model = PERFCTR_X86_INTEL_PIII;

  if (IS_MOBILE(&win_hwinfo))
    _papi_hwi_system_info.hw_info.model = PERFCTR_X86_INTEL_PENTM;

  if (IS_P4(&win_hwinfo)) {
    if (win_hwinfo.model >= 2)
      /* this is a guess for Pentium 4 Model 2 */
      _papi_hwi_system_info.hw_info.model = PERFCTR_X86_INTEL_P4M2;
    else
      _papi_hwi_system_info.hw_info.model = PERFCTR_X86_INTEL_P4;
  }

  if (IS_AMDDURON(&win_hwinfo) || IS_AMDATHLON(&win_hwinfo))
    _papi_hwi_system_info.hw_info.model = PERFCTR_X86_AMD_K7;

  strcpy(_papi_hwi_system_info.hw_info.model_string,win_hwinfo.model_string);

  _papi_hwi_system_info.num_cntrs = win_hwinfo.nrctr;
  _papi_hwi_system_info.num_gp_cntrs = _papi_hwi_system_info.num_cntrs;

  _papi_hwi_system_info.hw_info.mhz = (float)win_hwinfo.mhz;

  return(PAPI_OK);
}


/* Low level functions, should not handle errors, just return codes. */

inline_static long_long get_cycles(void) {
   long_long ret = __rdtsc();
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
   return(PAPI_ESBSTR); // Windows can't read virtual cycles...
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   return(PAPI_ESBSTR); // Windows can't read virtual seconds...
}

