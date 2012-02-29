/* 
/* 
* File:    winpmc-p3.c
* CVS:     $Id$
* Author:  dan terpstra
*          terpstra@cs.utk.edu
* Mods:    <name>
*		   <email address>
*
*		   This file is heavily derived from perfctr-p3.c circa 03/2006
*/

/* PAPI stuff */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "winpmc-p3.h"
#include "papi_memory.h"

/* Contains source for the Modified Bipartite Allocation scheme */
#include "papi_bipartite.h"

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

/****WIN32: This should be rewritten to be useful for Windows */
/* This structure exists, but I'm not sure if or how it's used. */
#ifdef DEBUG
void print_control(const struct pmc_cpu_control *control) {
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
      _papi_hwi_system_info.sub_info.num_native_events =_papi_hwd_p2_native_count;
      preset_search_map = &_papi_hwd_p2_preset_map;
      break;
   case PERFCTR_X86_INTEL_P6:
   case PERFCTR_X86_INTEL_PIII:
      native_table = &_papi_hwd_p3_native_map;
      _papi_hwi_system_info.sub_info.num_native_events = _papi_hwd_p3_native_count;
#ifdef XML
      return(_xml_papi_hwi_setup_all_presets("Pent III", NULL));
      break;
#endif
      preset_search_map = &_papi_hwd_p3_preset_map;
      break;
#ifdef PERFCTR_X86_INTEL_PENTM
   case PERFCTR_X86_INTEL_PENTM:
      native_table = &_papi_hwd_pm_native_map;
      _papi_hwi_system_info.sub_info.num_native_events = _papi_hwd_pm_native_count;
      preset_search_map = &_papi_hwd_pm_preset_map;
      break;
#endif
#ifdef PERFCTR_X86_INTEL_CORE
   case PERFCTR_X86_INTEL_CORE:
      native_table = &_papi_hwd_core_native_map;
      _papi_hwi_system_info.sub_info.num_native_events = _papi_hwd_core_native_count;
      preset_search_map = &_papi_hwd_core_preset_map;
	  break;
#endif
   case PERFCTR_X86_AMD_K7:
      native_table = &_papi_hwd_k7_native_map;
      _papi_hwi_system_info.sub_info.num_native_events = _papi_hwd_k7_native_count;
      preset_search_map = &_papi_hwd_ath_preset_map;
      break;

#ifdef PERFCTR_X86_AMD_K8 /* this is defined in perfctr 2.5.x */
   case PERFCTR_X86_AMD_K8:
      native_table = &_papi_hwd_k8_native_map;
      _papi_hwi_system_info.sub_info.num_native_events = _papi_hwd_k8_native_count;
      preset_search_map = &_papi_hwd_opt_preset_map;
      _papi_hwd_fixup_fp(&s, &n);
      break;
#endif
#ifdef PERFCTR_X86_AMD_K8C  /* this is defined in perfctr 2.6.x */
   case PERFCTR_X86_AMD_K8C:
      native_table = &_papi_hwd_k8_native_map;
      _papi_hwi_system_info.sub_info.num_native_events = _papi_hwd_k8_native_count;
      preset_search_map = &_papi_hwd_opt_preset_map;
      _papi_hwd_fixup_fp(&s, &n);
      break;
#endif

   default:
     PAPIERROR(MODEL_ERROR);
     return(PAPI_ESBSTR);
   }
   retval = _papi_hwi_setup_all_presets(preset_search_map, NULL);

   /* fix up the floating point ops if needed for opteron */
   if (s) retval =_papi_hwi_setup_all_presets(s,n);

   return(retval);
}

int _papi_hwd_init_control_state(hwd_control_state_t * ptr) {
   int i, def_mode = 0;

   if (_papi_hwi_system_info.sub_info.default_domain & PAPI_DOM_USER)
      def_mode |= PERF_USR;
   if (_papi_hwi_system_info.sub_info.default_domain & PAPI_DOM_KERNEL)
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
#ifdef PERFCTR_X86_INTEL_PENTM
   case PERFCTR_X86_INTEL_PENTM:
#endif
      ptr->control.cpu_control.evntsel[0] |= PERF_ENABLE;
      for(i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
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
      for (i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
         ptr->control.cpu_control.evntsel[i] |= PERF_ENABLE | def_mode;
         ptr->control.cpu_control.pmc_map[i] = i;
      }
      break;
   }
   /* Make sure the TSC is always on */
   ptr->control.cpu_control.tsc_on = 1;

   return(PAPI_OK);
}

int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain) {
   int i, did = 0;
    
     /* Clear the current domain set for this event set */
     /* We don't touch the Enable bit in this code but  */
     /* leave it as it is */
   for(i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
      cntrl->control.cpu_control.evntsel[i] &= ~(PERF_OS|PERF_USR);
   }
   if(domain & PAPI_DOM_USER) {
      did = 1;
      for(i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel[i] |= PERF_USR;
      }
   }
   if(domain & PAPI_DOM_KERNEL) {
      did = 1;
      for(i = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++) {
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
int _bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
   return(dst->ra_selector & (1 << ctr));
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.  */
void _bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
   dst->ra_selector = 1 << ctr;
   dst->ra_rank = 1;
}

/* This function examines the event to determine
   if it has a single exclusive mapping.
   Returns true if exlusive, false if non-exclusive.  */
int _bpt_map_exclusive(hwd_reg_alloc_t * dst) {
   return (dst->ra_rank == 1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.  */
int _bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   return (dst->ra_selector & src->ra_selector);
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.  */
void _bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   int i;
   unsigned shared;

   shared = dst->ra_selector & src->ra_selector;
   if (shared)
      dst->ra_selector ^= shared;
   for (i = 0, dst->ra_rank = 0; i < MAX_COUNTERS; i++)
      if (dst->ra_selector & (1 << i))
         dst->ra_rank++;
}

void _bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   dst->ra_selector = src->ra_selector;
}

/* Register allocation */
int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
   int i, j, natNum;
   hwd_reg_alloc_t event_list[MAX_COUNTERS];

   /* Initialize the local structure needed
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   for(i = 0; i < natNum; i++) {
      /* retrieve the mapping information about this native event */
      _papi_hwd_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_event, &event_list[i].ra_bits);

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
   if(_papi_bipartite_alloc(event_list, natNum)) { /* successfully mapped */
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
int _papi_hwd_update_control_state(hwd_control_state_t *this_state,
                                   NativeInfo_t *native, int count, hwd_context_t * ctx) {
   int i;

   /* clear out the events from the control state */
   clear_cs_events(this_state);

   /* fill the counters we're using */
   for (i = 0; i < count; i++) {
      /* Add counter control command values to eventset */
      this_state->control.cpu_control.evntsel[i] |= native[i].ni_bits.counter_cmd;
   }
   this_state->control.cpu_control.nractrs = count;
   return (PAPI_OK);
}

/* Collected wisdom indicates that each call to pmc_set_control will write 0's
    into the hardware counters, effecting a reset operation.
*/
int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * spc) {
   int error;
   struct pmc_control *ctl = (struct pmc_control *)(spc->control.cpu_control.evntsel);

   /* clear the accumulating counter values */
   memset((void *)spc->state.sum.pmc, 0, _papi_hwi_system_info.sub_info.num_cntrs * sizeof(long long) );
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

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * spc, long long ** dp, int flags) {
   pmc_read_state(_papi_hwi_system_info.sub_info.num_cntrs, &spc->state);
   *dp = (long long *) spc->state.sum.pmc;
#ifdef DEBUG
   {
      if (ISLEVEL(DEBUG_SUBSTRATE)) {
         unsigned int i;
         for(i = 0; i < spc->control.cpu_control.nractrs; i++) {
            SUBDBG("raw val hardware index %d is %lld\n", i,
                   (long long) spc->state.sum.pmc[i]);
         }
      }
   }
#endif
   return (PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_papi_hwd_start(ctx, cntrl));
}

static void lock_release(void)
{
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      DeleteCriticalSection(&lock[i]);
   }
}


/* This routine is for shutting down threads, including the
   master thread. */
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

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   hwd_control_state_t *this_state = &ESI->machdep;
   struct hwd_pmc_control *contr = &this_state->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

   OVFDBG("EventIndex=%d\n", EventIndex);

   /* The correct event to overflow is EventIndex */
   ncntrs = _papi_hwi_system_info.sub_info.num_cntrs;
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


int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

papi_svector_t _p3_vector_table[] = {
  {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
  {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
  {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
  {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
  {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
  {(void (*)())_papi_hwd_allocate_registers, VEC_PAPI_HWD_ALLOCATE_REGISTERS },
  {(void(*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
  {(void (*))_papi_hwd_set_domain, VEC_PAPI_HWD_SET_DOMAIN},
  {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
  {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
  {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
  {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
  {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
  {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
  {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
  { NULL, VEC_PAPI_END }
};


int setup_p3_vector_table(papi_vectors_t * vtable){
  int retval=PAPI_OK; 
 
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _p3_vector_table);
#endif
  return ( retval );
}


/* These should be removed when p3-p4 is merged */
int setup_p4_vector_table(papi_vectors_t * vtable){
  return ( PAPI_OK );
}

int setup_p4_presets(int cputype){
  return ( PAPI_OK );
}

