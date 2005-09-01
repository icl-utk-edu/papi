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

#ifdef __CATAMOUNT__
#include <asm/cpufunc.h>
#endif

/* PAPI stuff */
#define IN_SUBSTRATE

#include "papi.h"
#include "papi_internal.h"
#include "libperfctr.h"
#include "perfctr-p3.h"
#include "papi_protos.h"
#include "papi_vector.h"

int sidx;

/* Prototypes */
#ifdef PPC64
extern int setup_ppc64_presets(int cputype);
extern int ppc64_setup_vector_table(papi_vectors_t *);
#else
extern int setup_p4_presets(int cputype);
extern int setup_p4_vector_table(papi_vectors_t *, int idx);
int setup_p3_presets(int cputype);
int setup_p3_vector_table(papi_vectors_t *);
#endif

extern int p3_papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer);
extern char *p3_papi_hwd_ntv_code_to_name(unsigned int EventCode);
extern char *p3_papi_hwd_ntv_code_to_descr(unsigned int EventCode);
extern int p3_papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits);
extern int p3_papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count);

#ifdef XML
#define BUFFSIZE 8192
#define SPARSE_BEGIN 0
#define SPARSE_EVENT_SEARCH 1
#define SPARSE_EVENT 2
#define SPARSE_DESC 3
#define ARCH_SEARCH 4
#define DENSE_EVENT_SEARCH 5
#define DENSE_NATIVE_SEARCH 6
#define DENSE_NATIVE_DESC 7
#define FINISHED 8
#endif

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
extern native_event_entry_t *p3_native_table;
extern papi_mdi_t _papi_hwi_system_info;

#ifdef DEBUG
static void print_control(const struct perfctr_cpu_control *control) {
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
int setup_p3_presets(int cputype) {
   switch (cputype) {
   case PERFCTR_X86_GENERIC:
   case PERFCTR_X86_CYRIX_MII:
   case PERFCTR_X86_WINCHIP_C6:
   case PERFCTR_X86_WINCHIP_2:
   case PERFCTR_X86_VIA_C3:
   case PERFCTR_X86_INTEL_P5:
   case PERFCTR_X86_INTEL_P5MMX:
   case PERFCTR_X86_INTEL_PII:
      p3_native_table = &_papi_hwd_p2_native_map;
      preset_search_map = &_papi_hwd_p2_preset_map;
      break;
   case PERFCTR_X86_INTEL_P6:
   case PERFCTR_X86_INTEL_PIII:
      p3_native_table = &_papi_hwd_p3_native_map;
#ifdef XML
      return(_xml_papi_hwi_setup_all_presets("Pent III", NULL));
      break;
#endif
      preset_search_map = &_papi_hwd_p3_preset_map;
      break;
#ifdef PERFCTR_X86_INTEL_PENTM
   case PERFCTR_X86_INTEL_PENTM:
      p3_native_table = &_papi_hwd_pm_native_map;
      preset_search_map = &_papi_hwd_pm_preset_map;
      break;
#endif
   case PERFCTR_X86_AMD_K7:
      p3_native_table = &_papi_hwd_k7_native_map;
      preset_search_map = &_papi_hwd_ath_preset_map;
      break;

#ifdef PERFCTR_X86_AMD_K8 /* this is defined in perfctr 2.5.x */
   case PERFCTR_X86_AMD_K8:
      p3_native_table = &_papi_hwd_k8_native_map;
      preset_search_map = &_papi_hwd_opt_preset_map;
      break;
#endif
#ifdef PERFCTR_X86_AMD_K8C  /* this is defined in perfctr 2.6.x */
   case PERFCTR_X86_AMD_K8C:
      p3_native_table = &_papi_hwd_k8_native_map;
      preset_search_map = &_papi_hwd_opt_preset_map;
      break;
#endif

   default:
     PAPIERROR(MODEL_ERROR);
     return(PAPI_ESBSTR);
   }
   return (_papi_hwi_setup_all_presets(preset_search_map, NULL,sidx));
}

static void _papi_hwd_init_control_state(hwd_control_state_t * ptr) {
   int i, def_mode;

   /* XXX Need to change this */
   switch(_papi_hwi_substrate_info[0].default_domain) {
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
   /* XXX Need to change this */
      PAPIERROR("BUG! Unknown domain %d, using PAPI_DOM_USER",_papi_hwi_substrate_info[0].default_domain);
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
#ifdef PERFCTR_X86_INTEL_PENTM
   case PERFCTR_X86_INTEL_PENTM:
#endif
      ptr->control.cpu_control.evntsel[0] |= PERF_ENABLE;
      /* XXX Need to change this */
      for(i = 0; i < _papi_hwi_substrate_info[0].num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= def_mode;
         ptr->control.cpu_control.pmc_map[i] = i;
      }
      break;
#ifdef PERFCTR_X86_AMD_K8
   case PERFCTR_X86_AMD_K8:
#endif
#ifdef PERFCTR_X86_AMD_K8C
   case PERFCTR_X86_AMD_K8C:
#endif
   case PERFCTR_X86_AMD_K7:
      /* XXX Need to change this */
      for (i = 0; i < _papi_hwi_substrate_info[0].num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= PERF_ENABLE | def_mode;
         ptr->control.cpu_control.pmc_map[i] = i;
      }
      break;
   }
   /* Make sure the TSC is always on */
   ptr->control.cpu_control.tsc_on = 1;
}

static int _papi_hwd_set_domain(hwd_control_state_t * cntrl,int domain) {
   int i, did = 0;
    
     /* Clear the current domain set for this event set */
     /* We don't touch the Enable bit in this code but  */
     /* leave it as it is */
   /* XXX need to change this */
   for(i = 0; i < _papi_hwi_substrate_info[0].num_cntrs; i++) {
      cntrl->control.cpu_control.evntsel[i] &= ~(PERF_OS|PERF_USR);
   }
   if(domain & PAPI_DOM_USER) {
      did = 1;
      /* XXX need to change this */
      for(i = 0; i < _papi_hwi_substrate_info[0].num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel[i] |= PERF_USR;
      }
   }
   if(domain & PAPI_DOM_KERNEL) {
      did = 1;
      /* XXX need to change this */
      for(i = 0; i < _papi_hwi_substrate_info[0].num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel[i] |= PERF_OS;
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
static int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
   return(dst->ra_selector & (1 << ctr));
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.  */
static void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
   dst->ra_selector = 1 << ctr;
   dst->ra_rank = 1;
}

/* This function examines the event to determine
   if it has a single exclusive mapping.
   Returns true if exlusive, false if non-exclusive.  */
static int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
   return (dst->ra_rank == 1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.  */
static int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   return (dst->ra_selector & src->ra_selector);
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.  */
static void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   int i;
   unsigned shared;

   shared = dst->ra_selector & src->ra_selector;
   if (shared)
      dst->ra_selector ^= shared;
   for (i = 0, dst->ra_rank = 0; i < MAX_COUNTERS; i++)
      if (dst->ra_selector & (1 << i))
         dst->ra_rank++;
}

static void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   dst->ra_selector = src->ra_selector;
}

/* Register allocation */
static int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
   int i, j, natNum;
   hwd_reg_alloc_t event_list[MAX_COUNTERS];
   hwd_register_t *ptr;

   /* Initialize the local structure needed
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for(i = 0; i < natNum; i++) {
      /* retrieve the mapping information about this native event */
      p3_papi_hwd_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_event, &event_list[i].ra_bits);

      /* make sure register allocator only looks at legal registers */
      event_list[i].ra_selector = event_list[i].ra_bits.selector & ALLCNTRS;

      /* calculate native event rank, which is no. of counters it can live on */
      event_list[i].ra_rank = 0;
      for(j = 0; j < MAX_COUNTERS; j++) {
         if(event_list[i].ra_selector & (1 << j)) {
            event_list[i].ra_rank++;
         }
      }
   }
   if(_papi_hwi_bipartite_alloc(event_list, natNum, ESI->SubstrateIndex)) { /* successfully mapped */
      for(i = 0; i < natNum; i++) {
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

static void clear_cs_events(hwd_control_state_t *this_state) {
   int i,j;

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
static int _papi_hwd_update_control_state(hwd_control_state_t *this_state, NativeInfo_t *native, int count, hwd_context_t * ctx) {
   int i;

   /* clear out the events from the control state */
   clear_cs_events(this_state);

   /* fill the counters we're using */
   for (i = 0; i < count; i++) {
      /* Add counter control command values to eventset */
      this_state->control.cpu_control.evntsel[i] |= ((hwd_register_t *) native[i].ni_bits)->counter_cmd;
   }
   this_state->control.cpu_control.nractrs = count;
   return (PAPI_OK);
}


static int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * state) {
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

static int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
   if(vperfctr_stop(ctx->perfctr) < 0)
     { PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS); }
   return(PAPI_OK);
}

static int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * spc, long_long ** dp, int flags) {
   if ( flags & PAPI_PAUSED ) {
     int i,j=0;
     for ( i=0;i<spc->control.cpu_control.nractrs+spc->control.cpu_control.nrictrs; i++) {
       spc->state.pmc[j] = 0;
       if ( (spc->control.cpu_control.evntsel[i] & PERF_INT_ENABLE) ) continue;
       spc->state.pmc[j] = vperfctr_read_pmc(ctx->perfctr, i);
       j++;
     }
   }
   else {
      vperfctr_read_ctrs(ctx->perfctr, &spc->state);
   }
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

static int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_papi_hwd_start(ctx, cntrl));
}

/* This routine is for shutting down threads, including the
   master thread. */
static int _papi_hwd_shutdown(hwd_context_t * ctx) {
   int retval;// = vperfctr_unlink(ctx->perfctr);
   SUBDBG("_papi_hwd_shutdown vperfctr_unlink(%p) = %d\n", ctx->perfctr, retval);
   vperfctr_close(ctx->perfctr);
   SUBDBG("_papi_hwd_shutdown vperfctr_close(%p)\n", ctx->perfctr);
   memset(ctx, 0x0, sizeof(hwd_context_t));

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

static int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   hwd_control_state_t *this_state=(hwd_control_state_t *)ESI->machdep;
   struct hwd_pmc_control *contr = &this_state->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

#ifdef __CATAMOUNT__
   if(contr->cpu_control.nrictrs) { 
      OVFDBG("Catamount can't overflow on more than one event.\n");
      return PAPI_EINVAL;
   }
#endif

   OVFDBG("EventIndex=%d\n", EventIndex);

   /* The correct event to overflow is EventIndex */
   ncntrs = _papi_hwi_substrate_info[ESI->SubstrateIndex].num_cntrs;
   i = ESI->EventInfoArray[EventIndex].pos[0];
   if (i >= ncntrs) {
       PAPIERROR("Selector id %d is larger than ncntrs %d", i, ncntrs);
       return PAPI_EBUG;
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
      contr->cpu_control.nrictrs++;
      contr->cpu_control.nractrs--;
      nricntrs = contr->cpu_control.nrictrs;
      nracntrs = contr->cpu_control.nractrs;
      contr->si_signo = PAPI_SIGNAL;

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

      retval = _papi_hwi_stop_signal(PAPI_SIGNAL);
   }
   OVFDBG("End of call. Exit code: %d\n", retval);
   return (retval);
}


static int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

papi_svector_t _p3_vector_table[] = {
  {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
  {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
  {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
  {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
  {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
  {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
  {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
  {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
  {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
  {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
  {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
  {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
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
  {(void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
  {(void (*)())p3_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
  {(void (*)())p3_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
  {(void (*)())p3_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
  {(void (*)())p3_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
  {(void (*)())p3_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
  { NULL, VEC_PAPI_END }
};


int setup_p3_vector_table(papi_vectors_t * vtable){
  int retval=PAPI_OK; 
 
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _p3_vector_table);
#endif
  _papi_hwi_substrate_info[0].context_size  = sizeof(hwd_context_t);
  _papi_hwi_substrate_info[0].register_size = sizeof(hwd_register_t);
  _papi_hwi_substrate_info[0].reg_alloc_size = sizeof(hwd_reg_alloc_t);   
  _papi_hwi_substrate_info[0].control_state_size =sizeof(hwd_control_state_t);

  return ( retval );
}

static int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
   extern int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain);
   switch (code) {
   case PAPI_DOMAIN:
   case PAPI_DEFDOM:
      return (_papi_hwd_set_domain(option->domain.ESI->machdep, option->domain.domain));
   case PAPI_GRANUL:
   case PAPI_DEFGRN:
      return(PAPI_ESBSTR);
   default:
      return (PAPI_EINVAL);
   }
}

static void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *context)
{
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;
   int isHardware=0;
   caddr_t pc; 
   ctx.si = si;
   ctx.ucontext = (ucontext_t *) context;

   pc = GET_OVERFLOW_ADDRESS(ctx);

   _papi_hwi_dispatch_overflow_signal((void *)&ctx,&isHardware, si->si_pmc_ovf_mask,0,&master,pc,0);

   /* We are done, resume interrupting counters */

   if (isHardware) {
      if (vperfctr_iresume(((hwd_context_t *)master->context)->perfctr) < 0) {
         PAPIERROR("vperfctr_iresume errno %d",errno);
      }
   }
}


static int _papi_hwd_init(hwd_context_t * ctx) {
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

static long_long _papi_hwd_get_real_usec(void) {
   return((long_long)get_cycles() / (long_long)_papi_hwi_system_info.hw_info.mhz);
}

static long_long _papi_hwd_get_real_cycles(void) {
   return((long_long)get_cycles());
}

static long_long _papi_hwd_get_virt_cycles(hwd_context_t * ctx)
{
   return ((long_long)vperfctr_read_tsc(ctx->perfctr));
}

static long_long _papi_hwd_get_virt_usec(hwd_context_t * ctx)
{
   return ((long_long)vperfctr_read_tsc(ctx->perfctr) /
           (long_long)_papi_hwi_system_info.hw_info.mhz);
}


int _papi_hwd_init_substrate(papi_vectors_t *vtable, int idx)
{
  int retval;
  struct perfctr_info info;
  int is_p4=0;
  int fd;

  sidx = idx;

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = linux_vector_table_setup(vtable);
  if ( retval != PAPI_OK ) return(retval);
#endif

 
   retval = mdi_init();
   if ( retval )
     return(retval);

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

    /* copy tsc multiplier to local variable */
     /*tb_scale_factor = info.tsc_to_cpu_mult;*/

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
   strcpy(_papi_hwi_substrate_info[idx].substrate, "$Id$");
#ifndef PPC64
   if ( is_p4 ){
     retval = setup_p4_vector_table(vtable, idx);
     if (!retval)
        retval = setup_p4_presets(info.cpu_type);
   }
   else{
     retval = setup_p3_vector_table(vtable);
     if (!retval)
        retval = setup_p3_presets(info.cpu_type);
   }
#else
        /* Setup native and preset events */
    retval = ppc64_setup_vector_table(vtable);
    if (!retval)
        retval = setup_ppc64_native_table();
    if (!retval)
        retval = setup_ppc64_presets(info.cpu_type);
#endif

   if ( retval )
     return(retval);

   /* Fixup stuff from linux.c */

   strcpy(_papi_hwi_system_info.hw_info.model_string, PERFCTR_CPU_NAME(&info));

   _papi_hwi_substrate_info[idx].supports_hw_overflow =
       (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;
   SUBDBG("Hardware/OS %s support counter generated interrupts\n",          _papi_hwi_substrate_info[idx].supports_hw_overflow ? "does" : "does not");
   _papi_hwi_substrate_info[idx].num_cntrs = PERFCTR_CPU_NRCTRS(&info);
   _papi_hwi_substrate_info[idx].num_gp_cntrs = PERFCTR_CPU_NRCTRS(&info);
   _papi_hwi_system_info.hw_info.model = info.cpu_type;   
#ifdef PPC64
  _papi_hwi_system_info.hw_info.vendor = PAPI_VENDOR_IBM;
#else
   _papi_hwi_system_info.hw_info.vendor = xlate_cpu_type_to_vendor(info.cpu_type);
#endif

   _papi_hwi_substrate_info[idx].substrate_index = idx;

     /* Name of the substrate we're using */
    strcpy(_papi_hwi_substrate_info[0].substrate, "$Id$");

   _papi_hwi_substrate_info[0].supports_hw_overflow = 1;
   _papi_hwi_substrate_info[0].supports_64bit_counters = 1;
   _papi_hwi_substrate_info[0].supports_inheritance = 1;

#ifdef __CATAMOUNT__
   if (strstr(info.driver_version,"2.5") != info.driver_version) {
      fprintf(stderr,"Version mismatch of perfctr: compiled 2.5 or higher vs. installed %s\n",info.driver_version);
      return(PAPI_ESBSTR);
    }
  _papi_hwi_system_info.supports_hw_profile = 0;
  _papi_hwi_system_info.hw_info.mhz = (float) info.cpu_khz / 1000.0;
  SUBDBG("Detected MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz);
#endif


   lock_init();

   return (PAPI_OK);
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

