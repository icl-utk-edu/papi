/* 
* File:    perfctr-p3.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    nils smeds
*          smeds@pdc.kth.se
* Mods:    Kevin London
*	   london@cs.utk.edu
* Mods:    Joseph Thomas
*	   jthomas@cs.utk.edu
* Mods:    Haihang You
*	       you@cs.utk.edu
*/

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#ifdef __CATAMOUNT__
#include <asm/cpufunc.h>
#endif

/* PAPI stuff */
#include "papi.h"
#include "papi_internal.h"
#include "perfctr-p3.h"
#include "papi_memory.h"

/* Prototypes for entry points found in linux.c and linux-memory.c */
//int _linux_init_substrate(void);
int _linux_update_shlib_info(void);
int _linux_get_system_info(void);
int _linux_get_memory_info(PAPI_hw_info_t * hw_info, int cpu_type);
int _linux_get_dmem_info(PAPI_dmem_info_t *d);
int _linux_init(hwd_context_t * ctx);
//void _linux_dispatch_timer(int signal, siginfo_t * si, void *context);
//long long _linux_get_real_usec(void);
//long long _linux_get_real_cycles(void);
//long long _linux_get_virt_cycles(const hwd_context_t * ctx);
//long long _linux_get_virt_usec(const hwd_context_t * ctx);

#ifdef PERFCTR_PFM_EVENTS
/* Cleverly remap definitions of ntv routines from p3 to pfm */
#define _p3_ntv_enum_events _papi_pfm_ntv_enum_events
#define _p3_ntv_name_to_code _papi_pfm_ntv_name_to_code
#define _p3_ntv_code_to_name _papi_pfm_ntv_code_to_name
#define _p3_ntv_code_to_descr _papi_pfm_ntv_code_to_descr
#define _p3_ntv_code_to_bits _papi_pfm_ntv_code_to_bits
#define _p3_ntv_bits_to_info _papi_pfm_ntv_bits_to_info
/* Cleverly add an entry that doesn't exist for non-pfm */
int _p3_ntv_name_to_code(char *name, unsigned int *event_code);
#else
/* this routine doesn't exist if not pfm */
#define _p3_ntv_name_to_code NULL
#endif

/* Prototypes for entry points found in either p3_events or papi_pfm_events */
int _p3_ntv_enum_events(unsigned int *EventCode, int modifer);
int _p3_ntv_code_to_name(unsigned int EventCode, char * name, int len);
int _p3_ntv_code_to_descr(unsigned int EventCode, char * name, int len);
int _p3_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits);
int _p3_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values,
                          int name_len, int count);


/*
extern hwi_search_t _papi_hwd_p3_preset_map;
extern hwi_search_t _papi_hwd_pm_preset_map;
extern hwi_search_t _papi_hwd_core_preset_map;
extern hwi_search_t _papi_hwd_p2_preset_map;
extern hwi_search_t _papi_hwd_ath_preset_map;
extern hwi_search_t _papi_hwd_opt_preset_map;
extern hwi_search_t *preset_search_map;

extern int _papi_hwd_p2_native_count;
extern native_event_entry_t _papi_hwd_p2_native_map;
extern int _papi_hwd_p3_native_count;
extern native_event_entry_t _papi_hwd_p3_native_map;
extern int _papi_hwd_pm_native_count;
extern native_event_entry_t _papi_hwd_pm_native_map;
extern int _papi_hwd_core_native_count;
extern native_event_entry_t _papi_hwd_core_native_map;
extern int _papi_hwd_k7_native_count;
extern native_event_entry_t _papi_hwd_k7_native_map;
extern int _papi_hwd_k8_native_count;
extern native_event_entry_t _papi_hwd_k8_native_map;
*/

extern papi_mdi_t _papi_hwi_system_info;


/******************************************************************************
 * The below defines were imported from linux.c and will therefore need to be
 * duplicated in every substrate that relied on them, such as PPC and p4
 ******************************************************************************/

long long tb_scale_factor = (long long)1; /* needed to scale get_cycles on PPC series */

#ifdef PPC64
extern int setup_ppc64_presets(int cputype);
extern int ppc64_setup_vector_table(papi_vector_t *);
#elif defined(PPC32)
extern int setup_ppc32_presets(int cputype);
extern int ppc32_setup_vector_table(papi_vector_t *);
#else
extern int setup_p3_presets(int cputype);
#endif

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

/******************************************************************************
 * The above defines were imported from linux.c and will therefore need to be
 * duplicated in every substrate that relied on them, such as power and p4
 ******************************************************************************/

extern papi_vector_t MY_VECTOR;

#ifdef DEBUG
void print_control(const struct perfctr_cpu_control *control) {
  unsigned int i;

   SUBDBG("Control used:\n");
   SUBDBG("tsc_on\t\t\t%u\n", control->tsc_on);
   SUBDBG("nractrs\t\t\t%u\n", control->nractrs);
   SUBDBG("nrictrs\t\t\t%u\n", control->nrictrs);
   for (i = 0; i < (control->nractrs + control->nrictrs); ++i) {
      if (control->pmc_map[i] >= 18) {
         SUBDBG("pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i]);
      } else {
         SUBDBG("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
      }
      SUBDBG("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
      if (control->ireset[i])
         SUBDBG("ireset[%u]\t%d\n", i, control->ireset[i]);
   }
}
#endif

/******************************************************************************
 * The below routines were imported from linux.c and will therefore need to be
 * duplicated in every substrate that relied on them, such as power and p4
 ******************************************************************************/

#if (!defined(PPC64) && !defined(PPC32))
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
#ifdef PERFCTR_X86_INTEL_CORE
   case PERFCTR_X86_INTEL_CORE:
#endif
#ifdef PERFCTR_X86_INTEL_CORE2
   case PERFCTR_X86_INTEL_CORE2:
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
static int check_p4(int cputype){
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
#endif

/* volatile uint32_t lock; */

volatile unsigned int lock[PAPI_MAX_LOCK];

#if (defined(PPC32))
static void lock_init(void) 
{
   int retval, i;
  	union semun val; 
	val.val=1;
   if ((retval = semget(IPC_PRIVATE,PAPI_MAX_LOCK,0666)) == -1)
     {
       PAPIERROR("semget errno %d",errno);
     }
   sem_set = retval;
   for (i=0;i<PAPI_MAX_LOCK;i++)
     {
       if ((retval = semctl(sem_set,i,SETVAL,val)) == -1)
	 {
	   PAPIERROR("semctl errno %d",errno);
	 }
     }
}
#else
static void lock_init(void) {
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      lock[i] = MUTEX_OPEN;
   }
}
#endif

static int _p3_init_substrate(int cidx)
{
  int retval;
  struct perfctr_info info;
  char abiv[PAPI_MIN_STR_LEN];

#if defined(PERFCTR26)
  int fd;
#else
  struct vperfctr *dev;
#endif

  /* Setup the vector entries that the OS knows about */
  /*retval = _linux_setup_vector_table(vtable);
  if ( retval != PAPI_OK ) return(retval);
  */
 #if defined(PERFCTR26)
  /* Get info from the kernel */
   /* Use lower level calls per Mikael to get the perfctr info
      without actually creating a new kernel-side state.
      Also, close the fd immediately after retrieving the info.
      This is much lighter weight and doesn't reserve the counter
      resources. Also compatible with perfctr 2.6.14.
   */
   fd = _vperfctr_open(0);
   if (fd < 0)
     { PAPIERROR( VOPEN_ERROR); return(PAPI_ESYS); }
   retval = perfctr_info(fd, &info);
 	close(fd);
   if(retval < 0 )
     { PAPIERROR( VINFO_ERROR); return(PAPI_ESYS); }

    /* copy tsc multiplier to local variable        */
    /* this field appears in perfctr 2.6 and higher */
 	tb_scale_factor = (long long)info.tsc_to_cpu_mult;
#else
   /* Opened once for all threads. */
   if ((dev = vperfctr_open()) == NULL)
     { PAPIERROR( VOPEN_ERROR); return(PAPI_ESYS); }
   SUBDBG("_papi_hwd_init_substrate vperfctr_open = %p\n", dev);

   /* Get info from the kernel */
   retval = vperfctr_info(dev, &info);
   if (retval < 0)
     { PAPIERROR( VINFO_ERROR); return(PAPI_ESYS); }
    vperfctr_close(dev);
#endif

  /* Fill in what we can of the papi_system_info. */
  retval = MY_VECTOR.get_system_info();
  if (retval != PAPI_OK)
     return (retval);

   /* Setup memory info */
   retval = MY_VECTOR.get_memory_info(&_papi_hwi_system_info.hw_info, (int) info.cpu_type);
   if (retval)
      return (retval);

   strcpy(MY_VECTOR.cmp_info.name, "$Id$");
   strcpy(MY_VECTOR.cmp_info.version, "$Revision$");
   sprintf(abiv,"0x%08X",info.abi_version);
   strcpy(MY_VECTOR.cmp_info.support_version, abiv);
   strcpy(MY_VECTOR.cmp_info.kernel_version, info.driver_version);
   MY_VECTOR.cmp_info.CmpIdx = cidx;
   MY_VECTOR.cmp_info.num_cntrs = PERFCTR_CPU_NRCTRS(&info);
   MY_VECTOR.cmp_info.fast_counter_read = (info.cpu_features & PERFCTR_FEATURE_RDPMC) ? 1 : 0;
   MY_VECTOR.cmp_info.hardware_intr =
       (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;

   SUBDBG("Hardware/OS %s support counter generated interrupts\n",
          MY_VECTOR.cmp_info.hardware_intr ? "does" : "does not");

   /* Setup presets */
#if (!defined(PPC64) && !defined(PPC32))
   if ( check_p4(info.cpu_type) ){
     	retval = PAPI_ESBSTR;
   }
   else{
     	retval = setup_p3_presets(info.cpu_type);
   }
#elif (defined(PPC64))
	/* Setup native and preset events */
//	retval = ppc64_setup_vector_table(vtable);
//    if (!retval)
    	retval = setup_ppc64_native_table();
    if (!retval)
    	retval = setup_ppc64_presets(info.cpu_type);
#elif (defined(PPC32))
	/* Setup native and preset events */
//	retval = ppc32_setup_vector_table(vtable);
//	if (!retval)
    	retval = setup_ppc32_presets(info.cpu_type);
#endif
   if ( retval ) 
     return(retval);

   strcpy(_papi_hwi_system_info.hw_info.model_string, PERFCTR_CPU_NAME(&info));
   _papi_hwi_system_info.hw_info.model = info.cpu_type;
#if defined(PPC64)
   _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_IBM;
   if (strlen(_papi_hwi_system_info.hw_info.vendor_string) == 0)
     strcpy(_papi_hwi_system_info.hw_info.vendor_string,"IBM");
#elif defined(PPC32)
   _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_FREESCALE;
   if (strlen(_papi_hwi_system_info.hw_info.vendor_string) == 0)
     strcpy(_papi_hwi_system_info.hw_info.vendor_string,"Freescale");
#else
   _papi_hwi_system_info.hw_info.vendor = xlate_cpu_type_to_vendor(info.cpu_type);

#endif

#ifdef __CATAMOUNT__
   if (strstr(info.driver_version,"2.5") != info.driver_version) {
      fprintf(stderr,"Version mismatch of perfctr: compiled 2.5 or higher vs. installed %s\n",info.driver_version);
      return(PAPI_ESBSTR);
    }
   /* I think this was replaced by cmp_info.kernel_profile
   which is initialized to 0 in papi_internal:_papi_hwi_init_global_internal
  _papi_hwi_system_info.supports_hw_profile = 0;
  */
  _papi_hwi_system_info.hw_info.mhz = (float) info.cpu_khz / 1000.0; 
  SUBDBG("Detected MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz);
  _papi_hwi_system_info.hw_info.clock_mhz = mhz;
#endif

   lock_init();

   return (PAPI_OK);
}

void _p3_dispatch_timer(int signal, siginfo_t * si, void *context) {
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;
   int isHardware = 0;
   caddr_t pc;
   int cidx = MY_VECTOR.cmp_info.CmpIdx;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

#ifdef __CATAMOUNT__
#define OVERFLOW_MASK 0
#define GEN_OVERFLOW 1
#else
#define OVERFLOW_MASK si->si_pmc_ovf_mask
#define GEN_OVERFLOW 0
#endif

   pc = GET_OVERFLOW_ADDRESS(ctx);

   _papi_hwi_dispatch_overflow_signal((void *)&ctx, pc, &isHardware,
                                      OVERFLOW_MASK, GEN_OVERFLOW, &master, MY_VECTOR.cmp_info.CmpIdx);

   /* We are done, resume interrupting counters */
   if (isHardware) {
      errno = vperfctr_iresume(((cmp_context_t *)master->context[cidx])->perfctr);
      if (errno < 0) {
	  PAPIERROR("vperfctr_iresume errno %d for perfctr: %p",errno, ((cmp_context_t *)master->context[cidx])->perfctr);
      }
   }
}
static int attach( hwd_control_state_t * ctl, unsigned long tid ) {
	struct vperfctr_control tmp;

	((cmp_control_state_t *)ctl)->rvperfctr = rvperfctr_open( tid );
	if(((cmp_control_state_t *)ctl)->rvperfctr == NULL ) {
		PAPIERROR( VOPEN_ERROR ); return (PAPI_ESYS);
		}
	SUBDBG( "attach rvperfctr_open() = %p\n", ((cmp_control_state_t *)ctl)->rvperfctr );
	
	/* Initialize the per thread/process virtualized TSC */
	memset( &tmp, 0x0, sizeof(tmp) );
	tmp.cpu_control.tsc_on = 1;

	/* Start the per thread/process virtualized TSC */
	if( rvperfctr_control(((cmp_control_state_t *)ctl)->rvperfctr, & tmp ) < 0 ) {
		PAPIERROR(RCNTRL_ERROR); return(PAPI_ESYS);
		}

	return (PAPI_OK);
	} /* end attach() */

static int detach( hwd_control_state_t * ctl, unsigned long tid ) {
	rvperfctr_close(((cmp_control_state_t *)ctl)->rvperfctr );
	return (PAPI_OK);
	} /* end detach() */

/* Low level functions, should not handle errors, just return codes. */

#if (!defined(PPC64) && !defined(PPC32))
inline_static long long get_cycles(void) {
   long long ret = 0;
#ifdef __x86_64__
   do {
      unsigned int a,d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      (ret) = ((long long)a) | (((long long)d)<<32);
   } while(0);
#else
   __asm__ __volatile__("rdtsc"
                       : "=A" (ret)
                       : );
#endif
   return ret;
}
#elif defined(PPC32) || defined(PPC64)
inline_static long long get_cycles(void) {
	unsigned long tbl=0;
	unsigned long tbu=0;
	unsigned long long res=0;
	asm volatile("mftb %0" : "=r" (tbl));
	asm volatile("mftbu %0" : "=r" (tbu));
	res=tbu;
	res = (res << 32) | tbl;
	return (res * tb_scale_factor);
}
#endif //PPC64

static long long _p3_get_real_usec(void) {
   return((long long)get_cycles() / (long long)_papi_hwi_system_info.hw_info.mhz);
}

static long long _p3_get_real_cycles(void) {
   return((long long)get_cycles());
}

static long long _p3_get_virt_cycles(const hwd_context_t * ctx)
{
   return ((long long)vperfctr_read_tsc(((cmp_context_t *)ctx)->perfctr) * tb_scale_factor);
}

static long long _p3_get_virt_usec(const hwd_context_t * ctx)
{
   return (((long long)vperfctr_read_tsc(((cmp_context_t *)ctx)->perfctr) * tb_scale_factor) /
           (long long)_papi_hwi_system_info.hw_info.mhz);
}

/******************************************************************************
 * The above routines were imported from linux.c and will therefore need to be
 * duplicated in every substrate that relied on them, such as power and p4
 ******************************************************************************/

static int _p3_init_control_state(hwd_control_state_t * cntl) {
   int i, def_mode = 0;
   cmp_control_state_t *ptr = (cmp_control_state_t *)cntl;

   if (MY_VECTOR.cmp_info.default_domain & PAPI_DOM_USER)
      def_mode |= PERF_USR;
   if (MY_VECTOR.cmp_info.default_domain & PAPI_DOM_KERNEL)
     def_mode |= PERF_OS;

   ptr->allocated_registers.selector = 0;
   switch (_papi_hwi_system_info.hw_info.model) {
   case PERFCTR_X86_GENERIC:
   case PERFCTR_X86_CYRIX_MII:
   case PERFCTR_X86_WINCHIP_C6:
   case PERFCTR_X86_WINCHIP_2:
   case PERFCTR_X86_VIA_C3:
   case PERFCTR_X86_INTEL_P5:
   case PERFCTR_X86_INTEL_P5MMX:
   case PERFCTR_X86_INTEL_PII:
   case PERFCTR_X86_INTEL_P6:
   case PERFCTR_X86_INTEL_PIII:
#ifdef PERFCTR_X86_INTEL_CORE
   case PERFCTR_X86_INTEL_CORE:
#endif
#ifdef PERFCTR_X86_INTEL_PENTM
   case PERFCTR_X86_INTEL_PENTM:
#endif
      ptr->control.cpu_control.evntsel[0] |= PERF_ENABLE;
      for(i = 0; i < MY_VECTOR.cmp_info.num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= def_mode;
         ptr->control.cpu_control.pmc_map[i] = i;
      }
      break;
#ifdef PERFCTR_X86_INTEL_CORE2
   case PERFCTR_X86_INTEL_CORE2:
#endif
#ifdef PERFCTR_X86_INTEL_ATOM
   case PERFCTR_X86_INTEL_ATOM:
#endif
#ifdef PERFCTR_X86_INTEL_COREI7
   case PERFCTR_X86_INTEL_COREI7:
#endif
#ifdef PERFCTR_X86_AMD_K8
   case PERFCTR_X86_AMD_K8:
#endif
#ifdef PERFCTR_X86_AMD_K8C
   case PERFCTR_X86_AMD_K8C:
#endif
#ifdef PERFCTR_X86_AMD_FAM10H  /* this is defined in perfctr 2.6.29 */
   case PERFCTR_X86_AMD_FAM10H:
#endif
   case PERFCTR_X86_AMD_K7:
      for (i = 0; i < MY_VECTOR.cmp_info.num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= PERF_ENABLE | def_mode;
         ptr->control.cpu_control.pmc_map[i] = i;
      }
      break;
   }

#ifdef VPERFCTR_CONTROL_CLOEXEC
	ptr->control.flags = VPERFCTR_CONTROL_CLOEXEC;
	SUBDBG("close on exec\t\t\t%u\n", ptr->control.flags);
#endif

   /* Make sure the TSC is always on */
   ptr->control.cpu_control.tsc_on = 1;
   return(PAPI_OK);
}

static int _p3_set_domain(hwd_control_state_t * cntrl, int domain) {
   int i, did = 0;
   cmp_control_state_t *ptr = (cmp_control_state_t *)cntrl;
   int num_cntrs = MY_VECTOR.cmp_info.num_cntrs;

     /* Clear the current domain set for this event set */
     /* We don't touch the Enable bit in this code but  */
     /* leave it as it is */
   for(i = 0; i < num_cntrs; i++) {
      ptr->control.cpu_control.evntsel[i] &= ~(PERF_OS|PERF_USR);
   }
   if(domain & PAPI_DOM_USER) {
      did = 1;
      for(i = 0; i < num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= PERF_USR;
      }
   }
   if(domain & PAPI_DOM_KERNEL) {
      did = 1;
      for(i = 0; i < num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= PERF_OS;
      }
   }
   if(!did)
      return(PAPI_EINVAL);
   else
      return(PAPI_OK);
}

/* This function examines the event to determine
    if it can be mapped to counter ctr.
    Returns true if it can, false if it can't. */
static int _p3_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
   return(((cmp_reg_alloc_t *)dst)->ra_selector & (1 << ctr));
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.  */
static void _p3_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
   ((cmp_reg_alloc_t *)dst)->ra_selector = 1 << ctr;
   ((cmp_reg_alloc_t *)dst)->ra_rank = 1;
}

/* This function examines the event to determine
   if it has a single exclusive mapping.
   Returns true if exlusive, false if non-exclusive.  */
static int _p3_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
   return (((cmp_reg_alloc_t *)dst)->ra_rank == 1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.  */
static int _p3_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   return (((cmp_reg_alloc_t *)dst)->ra_selector & ((cmp_reg_alloc_t *)src)->ra_selector);
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.  */
static void _p3_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   int i;
   unsigned shared;

   shared = ((cmp_reg_alloc_t *)dst)->ra_selector & ((cmp_reg_alloc_t *)src)->ra_selector;
   if (shared)
      ((cmp_reg_alloc_t *)dst)->ra_selector ^= shared;
   for (i = 0, ((cmp_reg_alloc_t *)dst)->ra_rank = 0; i < MAX_COUNTERS; i++)
      if (((cmp_reg_alloc_t *)dst)->ra_selector & (1 << i))
         ((cmp_reg_alloc_t *)dst)->ra_rank++;
}

static void _p3_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   ((cmp_reg_alloc_t *)dst)->ra_selector = ((cmp_reg_alloc_t *)src)->ra_selector;
}

/* Register allocation */
static int _p3_allocate_registers(EventSetInfo_t *ESI) {
   int i, j, natNum;
   cmp_reg_alloc_t event_list[MAX_COUNTERS];
   cmp_register_t *ptr;

   /* Initialize the local structure needed
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for(i = 0; i < natNum; i++) {
      /* retrieve the mapping information about this native event */
      _p3_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_event, &event_list[i].ra_bits);

      /* make sure register allocator only looks at legal registers */
      event_list[i].ra_selector = event_list[i].ra_bits.selector & ALLCNTRS;
#ifdef PERFCTR_X86_INTEL_CORE2
      if(_papi_hwi_system_info.hw_info.model == PERFCTR_X86_INTEL_CORE2)
        event_list[i].ra_selector |= ((event_list[i].ra_bits.selector>>16)<<2) & ALLCNTRS;
#endif

      /* calculate native event rank, which is no. of counters it can live on */
      event_list[i].ra_rank = 0;
      for(j = 0; j < MAX_COUNTERS; j++) {
         if(event_list[i].ra_selector & (1 << j)) {
            event_list[i].ra_rank++;
         }
      }
   }
   if(_papi_hwi_bipartite_alloc(event_list, natNum, ESI->CmpIdx)) { /* successfully mapped */
      for(i = 0; i < natNum; i++) {
#ifdef PERFCTR_X86_INTEL_CORE2
         if(_papi_hwi_system_info.hw_info.model == PERFCTR_X86_INTEL_CORE2)
           event_list[i].ra_bits.selector = event_list[i].ra_selector;
#endif
         /* Copy all info about this native event to the NativeInfo struct */
         ptr = ESI->NativeInfoArray[i].ni_bits;
         *ptr = event_list[i].ra_bits;
         /* Array order on perfctr is event ADD order, not counter #... */
         ESI->NativeInfoArray[i].ni_position = i;
      }
      return 1;
   } else
      return 0;
}

static void clear_cs_events(hwd_control_state_t *state) {
   int i,j;
   cmp_control_state_t *this_state = (cmp_control_state_t *)state;

   /* total counters is sum of accumulating (nractrs) and interrupting (nrictrs) */
   j = this_state->control.cpu_control.nractrs + this_state->control.cpu_control.nrictrs;

   /* Remove all counter control command values from eventset. */
   for (i = 0; i < j; i++) {
      SUBDBG("Clearing pmc event entry %d\n", i);
      this_state->control.cpu_control.pmc_map[i] = i;
      this_state->control.cpu_control.evntsel[i] 
         = this_state->control.cpu_control.evntsel[i] & (PERF_ENABLE|PERF_OS|PERF_USR);
      this_state->control.cpu_control.ireset[i] = 0;
   }

   /* clear both a and i counter counts */
   this_state->control.cpu_control.nractrs = 0;
   this_state->control.cpu_control.nrictrs = 0;
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
static int _p3_update_control_state(hwd_control_state_t *state,
                                   NativeInfo_t *native, int count, hwd_context_t * ctx) {
   int i, k;
   cmp_control_state_t *this_state = (cmp_control_state_t *)state;

   /* clear out the events from the control state */
   clear_cs_events(this_state);

   switch (_papi_hwi_system_info.hw_info.model) {
     #ifdef PERFCTR_X86_INTEL_CORE2
     case PERFCTR_X86_INTEL_CORE2:
       /* fill the counters we're using */
       for (i = 0; i < count; i++) {
         for(k=0;k<MAX_COUNTERS;k++)
           if(((cmp_register_t *) native[i].ni_bits)->selector & (1 << k)) {
             break;
           }
         if(k>1)
           this_state->control.cpu_control.pmc_map[i] = (k-2) | 0x40000000;
         else
           this_state->control.cpu_control.pmc_map[i] = k;

         /* Add counter control command values to eventset */
         this_state->control.cpu_control.evntsel[i] |= ((cmp_register_t *) native[i].ni_bits)->counter_cmd;
       }
       break;
     #endif
     default:
	   /* fill the counters we're using */
	   for (i = 0; i < count; i++) {
		  /* Add counter control command values to eventset */
		  this_state->control.cpu_control.evntsel[i] |= ((cmp_register_t *) native[i].ni_bits)->counter_cmd;
	   }
   }
   this_state->control.cpu_control.nractrs = count;
   return (PAPI_OK);
}


static int _p3_start(hwd_context_t * ctx, hwd_control_state_t * state) {
   int error;
cmp_context_t * this_ctx = (cmp_context_t *)ctx;
cmp_control_state_t *this_state = (cmp_control_state_t *)state;

#ifdef DEBUG
   print_control(&this_state->control.cpu_control);
#endif

   if (this_state->rvperfctr != NULL) 
     {
       if((error = rvperfctr_control(this_state->rvperfctr, &this_state->control)) < 0) 
	 {
	   SUBDBG("rvperfctr_control returns: %d\n", error);
	   PAPIERROR(RCNTRL_ERROR); 
	   return(PAPI_ESYS); 
	 }
       return (PAPI_OK);
     }
   
   if((error = vperfctr_control(this_ctx->perfctr, &this_state->control)) < 0) {
      SUBDBG("vperfctr_control returns: %d\n", error);
      PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS);
   }
   return (PAPI_OK);
}

static int _p3_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
	int error;

   if(((cmp_control_state_t *)state)->rvperfctr != NULL ) {
     if(rvperfctr_stop((struct rvperfctr*)((cmp_context_t *)ctx)->perfctr) < 0)
       { PAPIERROR( RCNTRL_ERROR); return(PAPI_ESYS); }
     return (PAPI_OK);
   }

   error = vperfctr_stop(((cmp_context_t *)ctx)->perfctr);
   if(error < 0) {
      SUBDBG("vperfctr_stop returns: %d\n", error);
      PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS);
   }
   return(PAPI_OK);
}

static int _p3_read(hwd_context_t * ctx, hwd_control_state_t * state, long long ** dp, int flags) {
cmp_context_t * this_ctx = (cmp_context_t *)ctx;
cmp_control_state_t *spc = (cmp_control_state_t *)state;

   if ( flags & PAPI_PAUSED ) {
     vperfctr_read_state(this_ctx->perfctr, &spc->state, NULL);
     int i=0;
     for ( i=0;i<spc->control.cpu_control.nractrs+spc->control.cpu_control.nrictrs; i++) {
       SUBDBG("vperfctr_read_state: counter %d =  %lld\n", i, spc->state.pmc[i]);
     }
   }
   else {
      SUBDBG("vperfctr_read_ctrs\n");
      if( spc->rvperfctr != NULL ) {
        rvperfctr_read_ctrs( spc->rvperfctr, &spc->state );
      } else {
        vperfctr_read_ctrs(this_ctx->perfctr, &spc->state);
        }
   }
      *dp = (long long *) spc->state.pmc;
#ifdef DEBUG
   {
      if (ISLEVEL(DEBUG_SUBSTRATE)) {
         int i;
         for ( i=0;i<spc->control.cpu_control.nractrs+spc->control.cpu_control.nrictrs; i++) {
            SUBDBG("raw val hardware index %d is %lld\n", i,
                   (long long) spc->state.pmc[i]);
         }
      }
   }
#endif
   return (PAPI_OK);
}

static int _p3_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_p3_start(ctx, cntrl));
}

/* This routine is for shutting down threads, including the
   master thread. */
static int _p3_shutdown(hwd_context_t * ctx) {
   int retval = vperfctr_unlink(((cmp_context_t *)ctx)->perfctr);
   SUBDBG("_p3_shutdown vperfctr_unlink(%p) = %d\n", ((cmp_context_t *)ctx)->perfctr, retval);
   vperfctr_close(((cmp_context_t *)ctx)->perfctr);
   SUBDBG("_p3_shutdown vperfctr_close(%p)\n", ((cmp_context_t *)ctx)->perfctr);
   memset(ctx, 0x0, sizeof(cmp_context_t));

   if(retval)
      return(PAPI_ESYS);
   return(PAPI_OK);
}

/* Perfctr requires that interrupting counters appear at the end of the pmc list
   In the case a user wants to interrupt on a counter in an evntset that is not
   among the last events, we need to move the perfctr virtual events around to
   make it last. This function swaps two perfctr events, and then adjust the
   position entries in both the NativeInfoArray and the EventInfoArray to keep
   everything consistent.
*/
static void swap_events(EventSetInfo_t * ESI, struct hwd_pmc_control *contr, int cntr1, int cntr2) {
   unsigned int ui;
   int si, i, j;

   for(i = 0; i < ESI->NativeCount; i++) {
      if(ESI->NativeInfoArray[i].ni_position == cntr1)
         ESI->NativeInfoArray[i].ni_position = cntr2;
      else if(ESI->NativeInfoArray[i].ni_position == cntr2)
         ESI->NativeInfoArray[i].ni_position = cntr1;
   }
   for(i = 0; i < ESI->NumberOfEvents; i++) {
      for(j = 0; ESI->EventInfoArray[i].pos[j] >= 0; j++) {
         if(ESI->EventInfoArray[i].pos[j] == cntr1)
            ESI->EventInfoArray[i].pos[j] = cntr2;
         else if(ESI->EventInfoArray[i].pos[j] == cntr2)
            ESI->EventInfoArray[i].pos[j] = cntr1;
      }
   }
   ui = contr->cpu_control.pmc_map[cntr1];
   contr->cpu_control.pmc_map[cntr1] = contr->cpu_control.pmc_map[cntr2];
   contr->cpu_control.pmc_map[cntr2] = ui;

   ui = contr->cpu_control.evntsel[cntr1];
   contr->cpu_control.evntsel[cntr1] = contr->cpu_control.evntsel[cntr2];
   contr->cpu_control.evntsel[cntr2] = ui;

   si = contr->cpu_control.ireset[cntr1];
   contr->cpu_control.ireset[cntr1] = contr->cpu_control.ireset[cntr2];
   contr->cpu_control.ireset[cntr2] = si;
}

static int _p3_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   struct hwd_pmc_control *contr = &((cmp_control_state_t *)ESI->ctl_state)->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

#ifdef __CATAMOUNT__
   if(contr->cpu_control.nrictrs  && (threshold != 0)) { 
      OVFDBG("Catamount can't overflow on more than one event.\n");
      return PAPI_EINVAL;
   }
#endif

   OVFDBG("EventIndex=%d\n", EventIndex);

   /* The correct event to overflow is EventIndex */
   ncntrs = MY_VECTOR.cmp_info.num_cntrs;
   i = ESI->EventInfoArray[EventIndex].pos[0];
   if (i >= ncntrs) {
       PAPIERROR("Selector id %d is larger than ncntrs %d", i, ncntrs);
         return PAPI_EINVAL;
   }
   if (threshold != 0) {        /* Set an overflow threshold */
      retval = _papi_hwi_start_signal(MY_VECTOR.cmp_info.hardware_intr_sig,
	  NEED_CONTEXT, MY_VECTOR.cmp_info.CmpIdx);
      if (retval != PAPI_OK)
         return(retval);

      /* overflow interrupt occurs on the NEXT event after overflow occurs
         thus we subtract 1 from the threshold. */
      contr->cpu_control.ireset[i] = (-threshold + 1);
      contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;
      contr->cpu_control.nrictrs++;
      contr->cpu_control.nractrs--;
      nricntrs = contr->cpu_control.nrictrs;
      nracntrs = contr->cpu_control.nractrs;
      contr->si_signo = MY_VECTOR.cmp_info.hardware_intr_sig;

      /* move this event to the bottom part of the list if needed */
      if (i < nracntrs)
         swap_events(ESI, contr, i, nracntrs);
      OVFDBG("Modified event set\n");
   } else {
      if (contr->cpu_control.evntsel[i] & PERF_INT_ENABLE) {
         contr->cpu_control.ireset[i] = 0;
         contr->cpu_control.evntsel[i] &= (~PERF_INT_ENABLE);
         contr->cpu_control.nrictrs--;
         contr->cpu_control.nractrs++;
      }
      nricntrs = contr->cpu_control.nrictrs;
      nracntrs = contr->cpu_control.nractrs;

      /* move this event to the top part of the list if needed */
      if (i >= nracntrs)
         swap_events(ESI, contr, i, nracntrs - 1);

      if (!nricntrs)
         contr->si_signo = 0;

      OVFDBG("Modified event set\n");

      retval = _papi_hwi_stop_signal(MY_VECTOR.cmp_info.hardware_intr_sig);
   }
   OVFDBG("End of call. Exit code: %d\n", retval);
   return (retval);
}

static int _p3_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
/* For some reason, this warning kills the build */
/* #warning "_stop_profiling isn't implemented" */
	/* How do we turn off overflow? */
/*   ESI->profile.overflowcount = 0; */
   return (PAPI_OK);
}

static int _p3_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
  switch (code) {
   case PAPI_DOMAIN:
   case PAPI_DEFDOM:
      return (_p3_set_domain(option->domain.ESI->ctl_state, option->domain.domain));
   case PAPI_GRANUL:
   case PAPI_DEFGRN:
      return(PAPI_ESBSTR);
   case PAPI_ATTACH:
      return (attach(option->attach.ESI->ctl_state, option->attach.tid));
   case PAPI_DETACH:
      return (detach(option->attach.ESI->ctl_state, option->attach.tid));
   default:
      return (PAPI_EINVAL);
  }
}

/*
papi_svector_t _p3_vector_table[] = {
  {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
  {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
  {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
  {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
  {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
  {(void (*)())_papi_hwd_bpt_map_set, VEC_PAPI_HWD_BPT_MAP_SET },
  {(void (*)())_papi_hwd_bpt_map_avail, VEC_PAPI_HWD_BPT_MAP_AVAIL },
  {(void (*)())_papi_hwd_bpt_map_exclusive, VEC_PAPI_HWD_BPT_MAP_EXCLUSIVE },
  {(void (*)())_papi_hwd_bpt_map_shared, VEC_PAPI_HWD_BPT_MAP_SHARED },
  {(void (*)())_papi_hwd_bpt_map_preempt, VEC_PAPI_HWD_BPT_MAP_PREEMPT },
  {(void (*)())_papi_hwd_bpt_map_update, VEC_PAPI_HWD_BPT_MAP_UPDATE },
  {(void (*)())_papi_hwd_allocate_registers, VEC_PAPI_HWD_ALLOCATE_REGISTERS },
  {(void(*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
  {(void (*))_papi_hwd_set_domain, VEC_PAPI_HWD_SET_DOMAIN},
  {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
  {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
  { NULL, VEC_PAPI_END }
};


int setup_p3_vector_table(papi_vectors_t * vtable){
  int retval=PAPI_OK; 

#ifndef PAPI_NO_VECTOR
#ifdef PERFCTR_PFM_EVENTS
  papi_svector_t *event_vectors = _papi_pfm_event_vectors;
#else
  papi_svector_t *event_vectors = _papi_p3_event_vectors;
#endif

  retval = _papi_hwi_setup_vector_table( vtable, _p3_vector_table);
  if (retval == PAPI_OK) {
    retval = _papi_hwi_setup_vector_table(vtable, event_vectors);
  }
#endif
  return ( retval );
}
*/

///* These should be removed when p3-p4 is merged */
//int setup_p4_vector_table(papi_vectors_t * vtable){
//  return ( PAPI_OK );
//}
//
//int setup_p4_presets(int cputype){
//  return ( PAPI_OK );
//}

papi_vector_t _p3_vector = {
    .cmp_info = {
	/* default component information (unspecified values are initialized to 0) */
	.num_mpx_cntrs =	PAPI_MPX_DEF_DEG,
	.default_domain =	PAPI_DOM_USER,
	.available_domains =	PAPI_DOM_USER|PAPI_DOM_KERNEL,
	.default_granularity =	PAPI_GRN_THR,
	.available_granularities = PAPI_GRN_THR,
	.hardware_intr_sig =	PAPI_INT_SIGNAL,

	/* component specific cmp_info initializations */
	.fast_real_timer =	1,
	.fast_virtual_timer =	1,
	.attach =		1,
	.attach_must_ptrace =	1,
	.cntr_umasks = 1,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
	.context =		sizeof(_p3_perfctr_context_t),
	.control_state =	sizeof(_p3_perfctr_control_t),
	.reg_value =		sizeof(_p3_register_t),
	.reg_alloc =		sizeof(_p3_reg_alloc_t),
    },

    /* function pointers in this component */
    .init_control_state =	_p3_init_control_state,
    .start =			_p3_start,
    .stop =			_p3_stop,
    .read =			_p3_read,
    .shutdown =			_p3_shutdown,
    .ctl =			_p3_ctl,
    .bpt_map_set =		_p3_bpt_map_set,
    .bpt_map_avail =		_p3_bpt_map_avail,
    .bpt_map_exclusive =	_p3_bpt_map_exclusive,
    .bpt_map_shared =		_p3_bpt_map_shared,
    .bpt_map_preempt =		_p3_bpt_map_preempt,
    .bpt_map_update =		_p3_bpt_map_update,
    .allocate_registers =	_p3_allocate_registers,
    .update_control_state =	_p3_update_control_state,
    .set_domain =		_p3_set_domain,
    .reset =			_p3_reset,
    .set_overflow =		_p3_set_overflow,
    .stop_profiling =		_p3_stop_profiling,
    .ntv_enum_events =		_p3_ntv_enum_events,
    .ntv_name_to_code =		_p3_ntv_name_to_code,
    .ntv_code_to_name =		_p3_ntv_code_to_name,
    .ntv_code_to_descr =	_p3_ntv_code_to_descr,
    .ntv_code_to_bits =		_p3_ntv_code_to_bits,
    .ntv_bits_to_info =		_p3_ntv_bits_to_info,
    .init_substrate =		_p3_init_substrate,
    .dispatch_timer =		_p3_dispatch_timer,
    .get_real_usec =		_p3_get_real_usec,
    .get_real_cycles =		_p3_get_real_cycles,
    .get_virt_cycles =		_p3_get_virt_cycles,
    .get_virt_usec =		_p3_get_virt_usec,

    /* from OS */
 #ifndef __CATAMOUNT__
    .update_shlib_info = _linux_update_shlib_info,
 #endif
    .get_memory_info =	_linux_get_memory_info,
    .get_system_info =	_linux_get_system_info,
    .init =		_linux_init,
    .get_dmem_info =	_linux_get_dmem_info
};
