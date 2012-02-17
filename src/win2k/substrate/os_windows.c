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

/* This should be in a os_windows.h header file maybe. */
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
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_timer_callback, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
{ NULL, VEC_PAPI_END}
};
#endif

HANDLE pmc_dev = NULL;	// device handle for kernel driver

/* Called once per process. */

int _papi_hwd_shutdown_global(void) {
  pmc_close(pmc_dev);
  pmc_dev = NULL;
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

int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
  int retval;
  int model;
  HANDLE dh;	// device handle for kernel driver


  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _windows_os_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

  /* Fill in what we can of the papi_system_info. */
  retval = _papi_hwd_get_system_info();
  if (retval != PAPI_OK)
     return (retval);
  model = (int) _papi_hwi_system_info.hw_info.model;

   /* Setup memory info */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, model);
   if (retval)
      return (retval);

   /* Setup presets */
   if ( check_p4(model) ){
     retval = setup_p4_vector_table(vtable);
     if (!retval)
     	retval = setup_p4_presets(model);
   }
   else{
     retval = setup_p3_vector_table(vtable);
     if (!retval)
     	retval = setup_p3_presets(model);
   }
   if ( retval ) 
     return(retval);

   strcpy(_papi_hwi_system_info.sub_info.name, "os_windows.c");
   strcpy(_papi_hwi_system_info.sub_info.version, "$Revision$"); // cvs revision of this file
   dh = pmc_open();
   strcpy(_papi_hwi_system_info.sub_info.support_version, pmc_kernel_version(dh));
   pmc_close(dh);

   strcpy(_papi_hwi_system_info.sub_info.kernel_version, pmc_revision()); // cvs revision # of pmclib.c
   _papi_hwi_system_info.sub_info.fast_counter_read = 1; // built into WinPMC driver
   _papi_hwi_system_info.sub_info.fast_real_timer = 1;
   _papi_hwi_system_info.sub_info.fast_virtual_timer = 1;
   _papi_hwi_system_info.sub_info.default_domain = PAPI_DOM_USER;
   _papi_hwi_system_info.sub_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL;
   _papi_hwi_system_info.sub_info.default_granularity = PAPI_GRN_THR;
   _papi_hwi_system_info.sub_info.available_granularities = PAPI_GRN_THR;
   /****WIN32: can we figure out how to get Windows to support hdw interrupts on
    counter overflow? The hardware supports it; does the OS? */
   _papi_hwi_system_info.sub_info.hardware_intr = 0; // no hardware interrupt on Windows

   SUBDBG("Hardware/OS %s support counter generated interrupts\n",
          _papi_hwi_system_info.sub_info.hardware_intr ? "does" : "does not");

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

  if (IS_AMDOPTERON(&win_hwinfo))
    _papi_hwi_system_info.hw_info.model = PERFCTR_X86_AMD_K8;

  strcpy(_papi_hwi_system_info.hw_info.model_string,win_hwinfo.model_string);

  _papi_hwi_system_info.sub_info.num_cntrs = win_hwinfo.nrctr;

  _papi_hwi_system_info.hw_info.mhz = (float)win_hwinfo.mhz;

  return(PAPI_OK);
}


/* Low level functions, should not handle errors, just return codes. */

inline_static long long get_cycles(void) {
   long long ret = __rdtsc();
   return ret;
}

long long _papi_hwd_get_real_usec(void) {
   return((long long)get_cycles() / (long long)_papi_hwi_system_info.hw_info.mhz);
}

long long _papi_hwd_get_real_cycles(void) {
   return((long long)get_cycles());
}

//long long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
//{
//   return(PAPI_ESBSTR); // Windows can't read virtual cycles...
//}

long long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
    HANDLE p;
    BOOL ret;
    FILETIME Creation, Exit, Kernel, User;
    long long virt;

    p = GetCurrentProcess();
    ret = GetProcessTimes(p, &Creation, &Exit, &Kernel, &User);
    if (ret) {
	virt = (((long long)(Kernel.dwHighDateTime + User.dwHighDateTime))<<32)
	     + Kernel.dwLowDateTime + User.dwLowDateTime;
	return(virt/1000);
    }
    else return(PAPI_ESBSTR);
}
