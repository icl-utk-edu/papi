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
*/

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#ifdef PERFCTR26
#define PERFCTR_CPU_NAME   perfctr_info_cpu_name
#define PERFCTR_CPU_NRCTRS perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS perfctr_cpu_nrctrs
#endif

#include "papi.h"
#include "papi_internal.h"

extern hwi_search_t _papi_hwd_p3_preset_map;
extern hwi_search_t _papi_hwd_pm_preset_map;
extern hwi_search_t _papi_hwd_p2_preset_map;
extern hwi_search_t _papi_hwd_ath_preset_map;
extern hwi_search_t _papi_hwd_opt_preset_map;
extern hwi_search_t *preset_search_map;
extern native_event_entry_t _papi_hwd_p3_native_map;
extern native_event_entry_t _papi_hwd_pm_native_map;
extern native_event_entry_t _papi_hwd_p2_native_map;
extern native_event_entry_t _papi_hwd_k7_native_map;
extern native_event_entry_t _papi_hwd_k8_native_map;
extern native_event_entry_t *native_table;
extern papi_mdi_t _papi_hwi_system_info;

#ifdef _WIN32
CRITICAL_SECTION lock[PAPI_MAX_LOCK];
#else
volatile unsigned int lock[PAPI_MAX_LOCK];
#endif

#ifdef DEBUG
#if _WIN32
void print_control(const struct pmc_cpu_control *control) {
#else
void print_control(const struct perfctr_cpu_control *control) {
#endif
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

/* Assign the global native and preset table pointers, find the native
   table's size in memory and then call the preset setup routine. */
inline_static int setup_p3_presets(int cputype) {
   switch (cputype) {
   case PERFCTR_X86_GENERIC:
   case PERFCTR_X86_CYRIX_MII:
   case PERFCTR_X86_WINCHIP_C6:
   case PERFCTR_X86_WINCHIP_2:
   case PERFCTR_X86_VIA_C3:
   case PERFCTR_X86_INTEL_P5:
   case PERFCTR_X86_INTEL_P5MMX:
   case PERFCTR_X86_INTEL_PII:
      native_table = &_papi_hwd_p2_native_map;
      preset_search_map = &_papi_hwd_p2_preset_map;
      break;
   case PERFCTR_X86_INTEL_P6:
   case PERFCTR_X86_INTEL_PIII:
      native_table = &_papi_hwd_p3_native_map;
      preset_search_map = &_papi_hwd_p3_preset_map;
      break;
#ifdef PERFCTR26
   case PERFCTR_X86_INTEL_PENTM:
#endif
      native_table = &_papi_hwd_pm_native_map;
      preset_search_map = &_papi_hwd_pm_preset_map;
      break;
   case PERFCTR_X86_AMD_K7:
      native_table = &_papi_hwd_k7_native_map;
      preset_search_map = &_papi_hwd_ath_preset_map;
      break;
#ifdef PERFCTR26
   case PERFCTR_X86_AMD_K8:
   case PERFCTR_X86_AMD_K8C:
      native_table = &_papi_hwd_k8_native_map;
      preset_search_map = &_papi_hwd_opt_preset_map;
      break;
#endif
   default:
     PAPIERROR(MODEL_ERROR);
     return(PAPI_ESBSTR);
   }
   return (_papi_hwi_setup_all_presets(preset_search_map, NULL));
}

/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
static int mdi_init() 
   {
     /* Name of the substrate we're using */
    strcpy(_papi_hwi_system_info.substrate, "$Id$");       

   _papi_hwi_system_info.supports_hw_overflow = HW_OVERFLOW;
   _papi_hwi_system_info.using_hw_overflow = HW_OVERFLOW;
   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_inheritance = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;
   _papi_hwi_system_info.supports_virt_usec = 1;
   _papi_hwi_system_info.supports_virt_cyc = 1;

   return (PAPI_OK);
}

void _papi_hwd_init_control_state(hwd_control_state_t * ptr) {
   int i, def_mode;

   switch(_papi_hwi_system_info.default_domain) {
   case PAPI_DOM_USER:
      def_mode = PERF_USR;
      break;
   case PAPI_DOM_KERNEL:
      def_mode = PERF_OS;
      break;
   case PAPI_DOM_ALL:
      def_mode = PERF_OS | PERF_USR;
      break;
   default:
      PAPIERROR("BUG! Unknown domain %d, using PAPI_DOM_USER",_papi_hwi_system_info.default_domain);
      def_mode = PERF_USR;
      break;
   }
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
#ifdef PERFCTR26
   case PERFCTR_X86_INTEL_PENTM:
#endif
      ptr->control.cpu_control.evntsel[0] |= PERF_ENABLE;
      for(i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= def_mode;
         ptr->control.cpu_control.pmc_map[i] = i;
      }
      break;
#ifdef PERFCTR26
   case PERFCTR_X86_AMD_K8:
   case PERFCTR_X86_AMD_K8C:
#endif
   case PERFCTR_X86_AMD_K7:
      for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= PERF_ENABLE | def_mode;
         ptr->control.cpu_control.pmc_map[i] = i;
      }
      break;
   }
   /* Make sure the TSC is always on */
   ptr->control.cpu_control.tsc_on = 1;
}

int _papi_hwd_add_prog_event(hwd_control_state_t * state, unsigned int code, void *tmp, EventInfo_t *tmp2) {
   return (PAPI_ESBSTR);
}

int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain) {
   int i, did = 0;
    
     /* Clear the current domain set for this event set */
     /* We don't touch the Enable bit in this code but  */
     /* leave it as it is */
   for(i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
      cntrl->control.cpu_control.evntsel[i] &= ~(PERF_OS|PERF_USR);
   }
   if(domain & PAPI_DOM_USER) {
      did = 1;
      for(i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel[i] |= PERF_USR;
      }
   }
   if(domain & PAPI_DOM_KERNEL) {
      did = 1;
      for(i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel[i] |= PERF_OS;
      }
   }
   if(!did)
      return(PAPI_EINVAL);
   else
      return(PAPI_OK);
}

#ifdef _WIN32

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

HANDLE pmc_dev;	// device handle for kernel driver

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */
int _papi_hwd_init_global(void) {
   int retval;

   /* Initialize outstanding values in machine info structure */
   if (mdi_init() != PAPI_OK) {
      return (PAPI_ESBSTR);
   }

   /* Fill in what we can of the papi_system_info. */
   retval = _papi_hwd_get_system_info();
   if (retval != PAPI_OK)
      return (retval);

   /* Setup presets */
   retval = setup_p3_presets(_papi_hwi_system_info.hw_info.model);
   if (retval)
      return (retval);

   /* Setup memory info */
   retval =
       _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, (int) _papi_hwi_system_info.hw_info.vendor);
   if (retval)
      return (retval);

   lock_init();

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

/* Called once per process. */
int _papi_hwd_shutdown_global(void) {
  pmc_close(pmc_dev);
  lock_release();
   return (PAPI_OK);
}

#else

static void lock_init(void) {
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      lock[i] = MUTEX_OPEN;
   }
}

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

int _papi_hwd_init_global(void) 
{
   int retval;
   struct perfctr_info info;
   struct vperfctr *dev;

   /* Opened once for all threads. */

   if ((dev = vperfctr_open()) == NULL)
     { PAPIERROR( VOPEN_ERROR); return(PAPI_ESYS); }
   SUBDBG("_papi_hwd_init_global vperfctr_open = %p\n", dev);

   /* Get info from the kernel */

   if (vperfctr_info(dev, &info) < 0)
     { PAPIERROR( VINFO_ERROR); return(PAPI_ESYS); }

   /* Initialize outstanding values in machine info structure */

   if (mdi_init() != PAPI_OK) {
      return (PAPI_ESBSTR);
   }

   /* Fill in what we can of the papi_system_info. */
   retval = _papi_hwd_get_system_info();
   if (retval != PAPI_OK)
      return (retval);

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

   /* Setup presets */
   retval = setup_p3_presets(info.cpu_type);
   if (retval)
      return (retval);

   /* Setup memory info */
   retval =
       _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, (int) info.cpu_type);
   if (retval)
      return (retval);

    SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n", dev);
    vperfctr_close(dev);

    lock_init();

    return (PAPI_OK);
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

/* Called once per process. */
int _papi_hwd_shutdown_global(void) {
   return (PAPI_OK);
}

#endif /* _WIN32 */

/* This function examines the event to determine
    if it can be mapped to counter ctr.
    Returns true if it can, false if it can't. */
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
   return(dst->ra_selector & (1 << ctr));
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.  */
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
   dst->ra_selector = 1 << ctr;
   dst->ra_rank = 1;
}

/* This function examines the event to determine
   if it has a single exclusive mapping.
   Returns true if exlusive, false if non-exclusive.  */
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
   return (dst->ra_rank == 1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.  */
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   return (dst->ra_selector & src->ra_selector);
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.  */
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   int i;
   unsigned shared;

   shared = dst->ra_selector & src->ra_selector;
   if (shared)
      dst->ra_selector ^= shared;
   for (i = 0, dst->ra_rank = 0; i < MAX_COUNTERS; i++)
      if (dst->ra_selector & (1 << i))
         dst->ra_rank++;
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   dst->ra_selector = src->ra_selector;
}

/* Register allocation */
int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
   int index, i, j, natNum;
   hwd_reg_alloc_t event_list[MAX_COUNTERS];

   /* Initialize the local structure needed
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for(i = 0; i < natNum; i++) {
      index = ESI->NativeInfoArray[i].ni_event & PAPI_NATIVE_AND_MASK;
      event_list[i].ra_bits = native_table[index].resources;
      event_list[i].ra_selector = event_list[i].ra_bits.selector;
      /* calculate native event rank, which is no. of counters it can live on */
      event_list[i].ra_rank = 0;
      for(j = 0; j < MAX_COUNTERS; j++) {
         if(event_list[i].ra_selector & (1 << j)) {
            event_list[i].ra_rank++;
         }
      }
   }
   if(_papi_hwi_bipartite_alloc(event_list, natNum)) { /* successfully mapped */
      for(i = 0; i < natNum; i++) {
         /* Copy all info about this native event to the NativeInfo struct */
         ESI->NativeInfoArray[i].ni_bits = event_list[i].ra_bits;
         /* Array order on perfctr is event ADD order, not counter #... */
         ESI->NativeInfoArray[i].ni_position = i;
      }
      return 1;
   } else
      return 0;
}

static void clear_control_state(hwd_control_state_t *this_state) {
   unsigned int i;

   /* Remove all counter control command values from eventset. */
   for(i = 0; i < this_state->control.cpu_control.nractrs; i++) {
      SUBDBG("Clearing pmc event entry %d\n", i);
      this_state->control.cpu_control.pmc_map[i] = 0;
      this_state->control.cpu_control.evntsel[i] = this_state->control.cpu_control.evntsel[i] & (PERF_OS|PERF_USR);
      this_state->control.cpu_control.ireset[i] = 0;
   }
   this_state->control.cpu_control.nractrs = 0;
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t *this_state,
                                   NativeInfo_t *native, int count, hwd_context_t * ctx) {
   int i;

   /* clear out everything currently coded */
   clear_control_state(this_state);

   /* fill the counters we're using */
   _papi_hwd_init_control_state(this_state);
   for (i = 0; i < count; i++) {
      /* Add counter control command values to eventset */
      this_state->control.cpu_control.evntsel[i] |= native[i].ni_bits.counter_cmd;
   }
   this_state->control.cpu_control.nractrs = count;
   return (PAPI_OK);
}


#ifdef _WIN32

/* Collected wisdom indicates that each call to pmc_set_control will write 0's
    into the hardware counters, effecting a reset operation.
*/
int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * spc) {
   int error;
   struct pmc_control *ctl = (struct pmc_control *)(spc->control.cpu_control.evntsel);

   /* clear the accumulating counter values */
   memset((void *)spc->state.sum.pmc, 0, _papi_hwi_system_info.num_cntrs * sizeof(long_long) );
   if((error = pmc_set_control(ctx->self, ctl)) < 0) {
      SUBDBG("pmc_set_control returns: %d\n", error);
      { PAPIERROR( "pmc_set_control() returned < 0"); return(PAPI_ESYS); }
   }
#ifdef DEBUG
   print_control(&spc->control.cpu_control);
#endif
   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
  /* Since Windows counts system-wide (no counter saves at context switch)
      and since PAPI 3 no longer merges event sets, this function doesn't
      need to do anything in the Windows version.
  */
   return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * spc, long_long ** dp) {
   pmc_read_state(_papi_hwi_system_info.num_cntrs, &spc->state);
   *dp = (long_long *) spc->state.sum.pmc;
#ifdef DEBUG
   {
      if (ISLEVEL(DEBUG_SUBSTRATE)) {
         unsigned int i;
         for(i = 0; i < spc->control.cpu_control.nractrs; i++) {
            SUBDBG("raw val hardware index %d is %lld\n", i,
                   (long_long) spc->state.sum.pmc[i]);
         }
      }
   }
#endif
   return (PAPI_OK);
}

#else

int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * state) {
   int error;
   if((error = vperfctr_control(ctx->perfctr, &state->control)) < 0) {
      SUBDBG("vperfctr_control returns: %d\n", error);
      { PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS); }
   }
#ifdef DEBUG
   print_control(&state->control.cpu_control);
#endif
   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
   if(vperfctr_stop(ctx->perfctr) < 0)
     { PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS); }
   return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * spc, long_long ** dp) {
   vperfctr_read_ctrs(ctx->perfctr, &spc->state);
   *dp = (long_long *) spc->state.pmc;
#ifdef DEBUG
   {
      if (ISLEVEL(DEBUG_SUBSTRATE)) {
         int i;
         for(i = 0; i < spc->control.cpu_control.nractrs; i++) {
            SUBDBG("raw val hardware index %d is %lld\n", i,
                   (long_long) spc->state.pmc[i]);
         }
      }
   }
#endif
   return (PAPI_OK);
}

#endif /* _WIN32 */

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_papi_hwd_start(ctx, cntrl));
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long * from) {
   return(PAPI_ESBSTR);
}

#ifdef _WIN32

int _papi_hwd_shutdown(hwd_context_t * ctx) {
   int retval = 0;
//   retval = vperfctr_unlink(ctx->self);
   SUBDBG("_papi_hwd_shutdown vperfctr_unlink(%p) = %d\n", ctx->self, retval);
   pmc_close(ctx->self);
   SUBDBG("_papi_hwd_shutdown vperfctr_close(%p)\n", ctx->self);
   memset(ctx, 0x0, sizeof(hwd_context_t));

   if(retval)
      return(PAPI_ESYS);
   return(PAPI_OK);
}


void CALLBACK _papi_hwd_timer_callback(UINT wTimerID, UINT msg, 
        DWORD dwUser, DWORD dw1, DWORD dw2) 
{
    _papi_hwi_context_t ctx;
    CONTEXT	context;	// processor specific context structure
    HANDLE	threadHandle;
    BOOL	error;
    ThreadInfo *t = NULL;

   ctx.ucontext = &context;

   // dwUser is the threadID passed by timeSetEvent
    // NOTE: This call requires W2000 or later
    threadHandle = OpenThread(THREAD_GET_CONTEXT, FALSE, dwUser);

    // retrieve the contents of the control registers only
    context.ContextFlags = CONTEXT_CONTROL;
    error = GetThreadContext(threadHandle, &context);
    CloseHandle(threadHandle);

    // pass a void pointer to cpu register data here
    _papi_hwi_dispatch_overflow_signal((void *)(&ctx), 0, 0, 0, &t); 
}
#else

/* This routine is for shutting down threads, including the
   master thread. */
int _papi_hwd_shutdown(hwd_context_t * ctx) {
   int retval = vperfctr_unlink(ctx->perfctr);
   SUBDBG("_papi_hwd_shutdown vperfctr_unlink(%p) = %d\n", ctx->perfctr, retval);
   vperfctr_close(ctx->perfctr);
   SUBDBG("_papi_hwd_shutdown vperfctr_close(%p)\n", ctx->perfctr);
   memset(ctx, 0x0, sizeof(hwd_context_t));

   if(retval)
      return(PAPI_ESYS);
   return(PAPI_OK);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *context) {
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

   _papi_hwi_dispatch_overflow_signal((void *) &ctx,
                                     _papi_hwi_system_info.supports_hw_overflow,
                                      si->si_pmc_ovf_mask, 0, &master);

   /* We are done, resume interrupting counters */
   if (_papi_hwi_system_info.supports_hw_overflow) {
      if (vperfctr_iresume(master->context.perfctr) < 0) {
         PAPIERROR("vperfctr_iresume errno %d",errno);
      }
   }
}

#endif /* _WIN32 */

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

#ifdef _WIN32

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   hwd_control_state_t *this_state = &ESI->machdep;
   struct hwd_pmc_control *contr = &this_state->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

   OVFDBG("EventIndex=%d\n", EventIndex);

   /* The correct event to overflow is EventIndex */
   ncntrs = _papi_hwi_system_info.num_cntrs;
   i = ESI->EventInfoArray[EventIndex].pos[0];
   if (i >= ncntrs) {
      OVFDBG("Selector id (%d) larger than ncntrs (%d)\n", i, ncntrs);
      return PAPI_EINVAL;
   }
   if (threshold != 0) {        /* Set an overflow threshold */
/*******
      struct sigaction sa;
      int err;
*******/
      if ((ESI->EventInfoArray[EventIndex].derived) &&
          (ESI->EventInfoArray[EventIndex].derived != DERIVED_CMPD)){
         OVFDBG("Can't overflow on a derived event.\n");
         return PAPI_EINVAL;
      }
      /* overflow interrupt occurs on the NEXT event after overflow occurs
         thus we subtract 1 from the threshold. */
      contr->cpu_control.ireset[i] = (-threshold + 1);
/******* can't enable the interrupt bit for windows
      contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;
*******/
      nricntrs = ++contr->cpu_control.nrictrs;
      nracntrs = --contr->cpu_control.nractrs;
/*******
      contr->si_signo = PAPI_SIGNAL;
*******/

      /* move this event to the bottom part of the list if needed */
      if (i < nracntrs)
         swap_events(ESI, contr, i, nracntrs);
/*******
      memset(&sa, 0, sizeof sa);
      sa.sa_sigaction = _papi_hwd_dispatch_timer;
      sa.sa_flags = SA_SIGINFO;
      if ((err = sigaction(PAPI_SIGNAL, &sa, NULL)) < 0) {
         OVFDBG("Setting sigaction failed: SYSERR %d: %s", errno, strerror(errno));
         return (PAPI_ESYS);
      }
*******/
      OVFDBG("Modified event set\n");
   } else {
      if (contr->cpu_control.evntsel[i] & PERF_INT_ENABLE) {
         contr->cpu_control.ireset[i] = 0;
         contr->cpu_control.evntsel[i] &= (~PERF_INT_ENABLE);
         nricntrs = --contr->cpu_control.nrictrs;
         nracntrs = ++contr->cpu_control.nractrs;
      }
      /* move this event to the top part of the list if needed */
      if (i >= nracntrs)
         swap_events(ESI, contr, i, nracntrs - 1);
      if (!nricntrs)
         contr->si_signo = 0;

      OVFDBG("Modified event set\n");

   }
   OVFDBG("%s:%d: Hardware overflow is still experimental.\n", __FILE__, __LINE__);
   OVFDBG("End of call. Exit code: %d\n", retval);
   return (retval);
}

#else

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   hwd_control_state_t *this_state = &ESI->machdep;
   struct hwd_pmc_control *contr = &this_state->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

   OVFDBG("EventIndex=%d\n", EventIndex);

   /* The correct event to overflow is EventIndex */
   ncntrs = _papi_hwi_system_info.num_cntrs;
   i = ESI->EventInfoArray[EventIndex].pos[0];
   if (i >= ncntrs) {
      OVFDBG("Selector id (%d) larger than ncntrs (%d)\n", i, ncntrs);
      return PAPI_EINVAL;
   }
   if (threshold != 0) {        /* Set an overflow threshold */
      if ((ESI->EventInfoArray[EventIndex].derived) &&
          (ESI->EventInfoArray[EventIndex].derived != DERIVED_CMPD)){
         OVFDBG("Can't overflow on a derived event.\n");
         return PAPI_EINVAL;
      }

      if ((retval = _papi_hwi_start_signal(PAPI_SIGNAL,NEED_CONTEXT)) != PAPI_OK)
	return(retval);

      /* overflow interrupt occurs on the NEXT event after overflow occurs
         thus we subtract 1 from the threshold. */
      contr->cpu_control.ireset[i] = (-threshold + 1);
      contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;
      nricntrs = ++contr->cpu_control.nrictrs;
      nracntrs = --contr->cpu_control.nractrs;
      contr->si_signo = PAPI_SIGNAL;

      /* move this event to the bottom part of the list if needed */
      if (i < nracntrs)
         swap_events(ESI, contr, i, nracntrs);
      OVFDBG("Modified event set\n");
   } else {
      if (contr->cpu_control.evntsel[i] & PERF_INT_ENABLE) {
         contr->cpu_control.ireset[i] = 0;
         contr->cpu_control.evntsel[i] &= (~PERF_INT_ENABLE);
         nricntrs = --contr->cpu_control.nrictrs;
         nracntrs = ++contr->cpu_control.nractrs;
      }
      /* move this event to the top part of the list if needed */
      if (i >= nracntrs)
         swap_events(ESI, contr, i, nracntrs - 1);

      if (!nricntrs)
         contr->si_signo = 0;

      OVFDBG("Modified event set\n");

      retval = _papi_hwi_stop_signal(PAPI_SIGNAL);
   }
   OVFDBG("%s:%d: Hardware overflow is still experimental.\n", __FILE__, __LINE__);
   OVFDBG("End of call. Exit code: %d\n", retval);
   return (retval);
}

#endif /* _WIN32 */


int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   /* This function is not used and shouldn't be called. */
   return (PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
   ESI->profile.overflowcount = 0;
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

/* Low level functions, should not handle errors, just return codes. */

#ifdef _WIN32
inline_static long_long get_cycles (void)
{
   __asm rdtsc		// Read Time Stamp Counter
   // This assembly instruction places the 64-bit value in edx:eax
   // Which is exactly where it needs to be for a 64-bit return value...
}
#else
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
#endif

long_long _papi_hwd_get_real_usec(void) {
   return((long_long)get_cycles() / (long_long)_papi_hwi_system_info.hw_info.mhz);
}

long_long _papi_hwd_get_real_cycles(void) {
   return((long_long)get_cycles());
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx) {
#ifdef _WIN32
   return(PAPI_ESBSTR); // Windows can't read virtual cycles...
#else
   return((long_long)vperfctr_read_tsc(ctx->perfctr) /
         (long_long)_papi_hwi_system_info.hw_info.mhz);
#endif /* _WIN32 */
}
 
long_long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx) {
#ifdef _WIN32
   return(PAPI_ESBSTR); // Windows can't read virtual cycles...
#else
   return((long_long)vperfctr_read_tsc(ctx->perfctr));
#endif /* _WIN32 */
}
