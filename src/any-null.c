/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    any-null.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include SUBSTRATE
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"

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

volatile unsigned int _papi_hwd_lock[PAPI_MAX_LOCK] = { 0, };
volatile long_long virt_tsc, cntr[2];

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
   default:
      native_table = &_papi_hwd_p3_native_map;
      preset_search_map = &_papi_hwd_p3_preset_map;
   }
   return (_papi_hwi_setup_all_presets(preset_search_map, NULL));
}

/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
 extern int _papi_hwd_mdi_init() 
   {
     /* Name of the substrate we're using */
    strcpy(_papi_hwi_system_info.substrate, "$Id$");       

   _papi_hwi_system_info.supports_hw_overflow = 0;
   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_inheritance = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;
   _papi_hwi_system_info.supports_virt_usec = 1;
   _papi_hwi_system_info.supports_virt_cyc = 1;
   _papi_hwi_system_info.supports_multiple_threads = 1;

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
      abort();
   }

   ptr->allocated_registers.selector = 0;

   switch (_papi_hwi_system_info.hw_info.model) 
     {
     default:
       for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= def_mode;
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

void _papi_hwd_lock_init(void) 
{
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      _papi_hwd_lock[i] = MUTEX_OPEN;
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
      error_return(PAPI_ESYS, VOPEN_ERROR);
   SUBDBG("_papi_hwd_init_global vperfctr_open = %p\n", dev);

   /* Get info from the kernel */

   if (vperfctr_info(dev, &info) < 0)
      error_return(PAPI_ESYS, VINFO_ERROR);

   /* Initialize outstanding values in machine info structure */

   if (_papi_hwd_mdi_init() != PAPI_OK) {
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

    SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n", dev);
    vperfctr_close(dev);

   virt_tsc = 111LL;
   cntr[0] = 222LL;
   cntr[1] = 333LL;

    return (PAPI_OK);
}

int _papi_hwd_init(hwd_context_t * ctx) 
{
   struct vperfctr_control tmp;

   /* Initialize our thread/process pointer. */
   if ((ctx->perfctr = vperfctr_open()) == NULL)
      error_return(PAPI_ESYS, VOPEN_ERROR);
   SUBDBG("_papi_hwd_init vperfctr_open() = %p\n", ctx->perfctr);

   /* Initialize the per thread/process virtualized TSC */
   memset(&tmp, 0x0, sizeof(tmp));
   tmp.cpu_control.tsc_on = 1;

   /* Start the per thread/process virtualized TSC */
   if (vperfctr_control(ctx->perfctr, &tmp) < 0)
      error_return(PAPI_ESYS, VCNTRL_ERROR);

   return (PAPI_OK);
}

/* Called once per process. */
int _papi_hwd_shutdown_global(void) 
{
   virt_tsc = 0LL;
   cntr[0] = 0LL;
   cntr[1] = 0LL;

   return (PAPI_OK);
}


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
                                   NativeInfo_t *native, int count) {
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

int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * state) {
   int error;
   if((error = vperfctr_control(ctx->perfctr, &state->control)) < 0) {
      SUBDBG("vperfctr_control returns: %d\n", error);
      error_return(PAPI_ESYS, VCNTRL_ERROR);
   }
#ifdef DEBUG
   print_control(&state->control.cpu_control);
#endif
   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
   if(vperfctr_stop(ctx->perfctr) < 0)
      error_return(PAPI_ESYS, VCNTRL_ERROR);
   return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * spc, long_long ** dp) {
   vperfctr_read_ctrs(ctx->perfctr, &spc->state);
   *dp = (long_long *) spc->state.pmc;
#ifdef DEBUG
   {
      if(_papi_hwi_debug & DEBUG_SUBSTRATE) {
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

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_papi_hwd_start(ctx, cntrl));
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long * from) {
   return(PAPI_ESBSTR);
}

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

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *context) 
{
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   /* This function is not used and shouldn't be called. */
   return (PAPI_ESBSTR);
}

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
   default:
      return (PAPI_EINVAL);
   }
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

long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx) {
   return((long_long)vperfctr_read_tsc(ctx->perfctr) /
         (long_long)_papi_hwi_system_info.hw_info.mhz);
}
 
long_long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx) {
   return((long_long)vperfctr_read_tsc(ctx->perfctr));
}
