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
int _linux_init_substrate(void);
int _linux_update_shlib_info(void);
int _linux_get_system_info(void);
int _linux_get_memory_info(PAPI_hw_info_t * hw_info, int cpu_type);
int _linux_get_dmem_info(PAPI_dmem_info_t *d);
int _linux_init(hwd_context_t * ctx);
void _linux_dispatch_timer(int signal, siginfo_t * si, void *context);
long_long _linux_get_real_usec(void);
long_long _linux_get_real_cycles(void);
long_long _linux_get_virt_cycles(const hwd_context_t * ctx);
long_long _linux_get_virt_usec(const hwd_context_t * ctx);

/* Prototypes for entry points found in p3_events */
int _p3_ntv_enum_events(unsigned int *EventCode, int modifer);
char *_p3_ntv_code_to_name(unsigned int EventCode);
char *_p3_ntv_code_to_descr(unsigned int EventCode);
int _p3_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits);
int _p3_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values,
                          int name_len, int count);

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

extern native_event_entry_t *native_table;
extern papi_mdi_t _papi_hwi_system_info;

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


/* Assign the global native and preset table pointers, find the native
   table's size in memory and then call the preset setup routine. */
int setup_p3_presets(int cputype) {
   int retval;
   hwi_search_t *s = NULL;
   hwi_dev_notes_t *n = NULL;
   extern void _papi_hwd_fixup_fp(hwi_search_t **s, hwi_dev_notes_t **n);

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
      MY_VECTOR.cmp_info.num_native_events =_papi_hwd_p2_native_count;
      preset_search_map = &_papi_hwd_p2_preset_map;
      break;
   case PERFCTR_X86_INTEL_P6:
   case PERFCTR_X86_INTEL_PIII:
      native_table = &_papi_hwd_p3_native_map;
      MY_VECTOR.cmp_info.num_native_events = _papi_hwd_p3_native_count;
#ifdef XML
      return(_xml_papi_hwi_setup_all_presets("Pent III", NULL));
      break;
#endif
      preset_search_map = &_papi_hwd_p3_preset_map;
      break;
#ifdef PERFCTR_X86_INTEL_PENTM
   case PERFCTR_X86_INTEL_PENTM:
      native_table = &_papi_hwd_pm_native_map;
      MY_VECTOR.cmp_info.num_native_events = _papi_hwd_pm_native_count;
      preset_search_map = &_papi_hwd_pm_preset_map;
      break;
#endif
#ifdef PERFCTR_X86_INTEL_CORE
   case PERFCTR_X86_INTEL_CORE:
      native_table = &_papi_hwd_core_native_map;
      MY_VECTOR.cmp_info.num_native_events = _papi_hwd_core_native_count;
      preset_search_map = &_papi_hwd_core_preset_map;
	  break;
#endif
#ifdef PERFCTR_X86_INTEL_CORE2
   case PERFCTR_X86_INTEL_CORE2:
      native_table = &_papi_hwd_core_native_map;
      MY_VECTOR.cmp_info.num_native_events = _papi_hwd_core_native_count;
      preset_search_map = &_papi_hwd_core_preset_map;
	  break;
#endif
   case PERFCTR_X86_AMD_K7:
      native_table = &_papi_hwd_k7_native_map;
      MY_VECTOR.cmp_info.num_native_events = _papi_hwd_k7_native_count;
      preset_search_map = &_papi_hwd_ath_preset_map;
      break;

#ifdef PERFCTR_X86_AMD_K8 /* this is defined in perfctr 2.5.x */
   case PERFCTR_X86_AMD_K8:
      native_table = &_papi_hwd_k8_native_map;
      MY_VECTOR.cmp_info.num_native_events = _papi_hwd_k8_native_count;
      preset_search_map = &_papi_hwd_opt_preset_map;
      _papi_hwd_fixup_fp(&s, &n);
      break;
#endif
#ifdef PERFCTR_X86_AMD_K8C  /* this is defined in perfctr 2.6.x */
   case PERFCTR_X86_AMD_K8C:
      native_table = &_papi_hwd_k8_native_map;
      MY_VECTOR.cmp_info.num_native_events = _papi_hwd_k8_native_count;
      preset_search_map = &_papi_hwd_opt_preset_map;
      _papi_hwd_fixup_fp(&s, &n);
      break;
#endif

   default:
     PAPIERROR(MODEL_ERROR);
     return(PAPI_ESBSTR);
   }
   SUBDBG("Number of native events: %d\n",MY_VECTOR.cmp_info.num_native_events);
   retval = _papi_hwi_setup_all_presets(preset_search_map, NULL);

   /* fix up the floating point ops if needed for opteron */
   if (s) retval =_papi_hwi_setup_all_presets(s,n);

   return(retval);
}

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
#ifdef PERFCTR_X86_AMD_K8
   case PERFCTR_X86_AMD_K8:
#endif
#ifdef PERFCTR_X86_AMD_K8C
   case PERFCTR_X86_AMD_K8C:
#endif
   case PERFCTR_X86_AMD_K7:
      for (i = 0; i < MY_VECTOR.cmp_info.num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= PERF_ENABLE | def_mode;
         ptr->control.cpu_control.pmc_map[i] = i;
      }
      break;
   }
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
   int i;
   cmp_control_state_t *this_state = (cmp_control_state_t *)state;

   /* clear out the events from the control state */
   clear_cs_events(this_state);

   /* fill the counters we're using */
   for (i = 0; i < count; i++) {
      /* Add counter control command values to eventset */
      this_state->control.cpu_control.evntsel[i] |= ((cmp_register_t *) native[i].ni_bits)->counter_cmd;
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
      { PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS); }
   }
   return (PAPI_OK);
}

static int _p3_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
   if(((cmp_control_state_t *)state)->rvperfctr != NULL ) {
     if(rvperfctr_stop((struct rvperfctr*)((cmp_context_t *)ctx)->perfctr) < 0)
       { PAPIERROR( RCNTRL_ERROR); return(PAPI_ESYS); }
     return (PAPI_OK);
   }

   if(vperfctr_stop(((cmp_context_t *)ctx)->perfctr) < 0)
     { PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS); }
   return(PAPI_OK);
}

static int _p3_read(hwd_context_t * ctx, hwd_control_state_t * state, long_long ** dp, int flags) {
cmp_context_t * this_ctx = (cmp_context_t *)ctx;
cmp_control_state_t *spc = (cmp_control_state_t *)state;

   if ( flags & PAPI_PAUSED ) {
     int i,j=0;
     for ( i=0;i<spc->control.cpu_control.nractrs+spc->control.cpu_control.nrictrs; i++) {
       spc->state.pmc[j] = 0;
       if ( (spc->control.cpu_control.evntsel[i] & PERF_INT_ENABLE) ) continue;
       spc->state.pmc[j] = vperfctr_read_pmc(this_ctx->perfctr, i);
       SUBDBG("vperfctr_read_pmc[%d] =  %lld\n", i, spc->state.pmc[j]);
       j++;
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
      *dp = (long_long *) spc->state.pmc;
#ifdef DEBUG
   {
      if (ISLEVEL(DEBUG_SUBSTRATE)) {
         int i;
         for ( i=0;i<spc->control.cpu_control.nractrs+spc->control.cpu_control.nrictrs; i++) {
            SUBDBG("raw val hardware index %d is %lld\n", i,
                   (long_long) spc->state.pmc[i]);
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
       return PAPI_EBUG;
   }
   if (threshold != 0) {        /* Set an overflow threshold */
      if ((ESI->EventInfoArray[EventIndex].derived) &&
          (ESI->EventInfoArray[EventIndex].derived != DERIVED_CMPD)){
         OVFDBG("Can't overflow on a derived event.\n");
         return PAPI_EINVAL;
      }

      if ((retval = _papi_hwi_start_signal(MY_VECTOR.cmp_info.hardware_intr_sig,NEED_CONTEXT)) != PAPI_OK)
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
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

/**********************************************************************************************/
/** This was lifted from linux.c, so might be shared by other substrates                     **/
/**********************************************************************************************/

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

papi_vector_t _p3_vector = {
    /* sizes of component-private structures */
    .context_size =		sizeof(_p3_perfctr_context_t),
    .control_state_size =	sizeof(_p3_perfctr_control_t),
    .register_size =		sizeof(_p3_register_t),
    .reg_alloc_size =		sizeof(_p3_reg_alloc_t),
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
    .ntv_code_to_name =		_p3_ntv_code_to_name,
    .ntv_code_to_descr =	_p3_ntv_code_to_descr,
    .ntv_code_to_bits =		_p3_ntv_code_to_bits,
    .ntv_bits_to_info =		_p3_ntv_bits_to_info,

    /* from OS */
 #ifndef __CATAMOUNT__
    .update_shlib_info = _linux_update_shlib_info,
 #endif
    .get_memory_info =	_linux_get_memory_info,
    .get_system_info =	_linux_get_system_info,
    .init =		_linux_init,
    .init_substrate =	_linux_init_substrate,
    .dispatch_timer =	_linux_dispatch_timer,
    .get_real_usec =	_linux_get_real_usec,
    .get_real_cycles =	_linux_get_real_cycles,
    .get_virt_cycles =	_linux_get_virt_cycles,
    .get_virt_usec =	_linux_get_virt_usec,
    .get_dmem_info =	_linux_get_dmem_info
};

int setup_p3_vector_table(papi_vector_t * vtable){
  /*return _papi_hwi_setup_vector_table( vtable, &_p3_vector);*/
  return ( PAPI_OK );
}

/* These should be removed when p3-p4 is merged */
int setup_p4_vector_table(papi_vector_t * vtable){
  return ( PAPI_OK );
}

int setup_p4_presets(int cputype){
  return ( PAPI_OK );
}

