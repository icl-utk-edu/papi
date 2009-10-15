/* 
* File:    perfctr-ppc32.c
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
* Mods:    Brian Sheely
*	   bsheely@eecs.utk.edu
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
//guanglei
#include "perfctr-ppc32.h"

//guanglei
int sem_set;

extern hwi_search_t _papi_hwd_ppc750_preset_map;
extern native_event_entry_t _papi_hwd_ppc750_native_map;
extern hwi_search_t _papi_hwd_ppc7450_preset_map;
extern native_event_entry_t _papi_hwd_ppc7450_native_map;

extern hwi_search_t *preset_search_map;
extern native_event_entry_t *native_table;
extern papi_mdi_t _papi_hwi_system_info;

volatile unsigned int lock[PAPI_MAX_LOCK];

#ifdef DEBUG
void print_control(const struct perfctr_cpu_control *control) {
  unsigned int i;

   SUBDBG("Control used:\n");
   SUBDBG("tsc_on\t\t%u\n", control->tsc_on);
   SUBDBG("nractrs\t\t%u\n", control->nractrs);
   SUBDBG("nrictrs\t\t%u\n", control->nrictrs);
   SUBDBG("mmcr0\t\t0x%08X\n", control->ppc.mmcr0);
   for (i = 0; i < (control->nractrs + control->nrictrs); ++i) {
     SUBDBG("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
     SUBDBG("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
     if (control->ireset[i])
         SUBDBG("ireset[%u]\t%d\n", i, control->ireset[i]);
   }
}
#endif


/* Assign the global native and preset table pointers, find the native
   table's size in memory and then call the preset setup routine. */

int setup_ppc32_presets(int cputype) {
  int i = 0;
   switch (cputype) {
   case PERFCTR_PPC_750:
      native_table = &_papi_hwd_ppc750_native_map;
      preset_search_map = &_papi_hwd_ppc750_preset_map;
   case PERFCTR_PPC_7450:
      native_table = &_papi_hwd_ppc7450_native_map;
      preset_search_map = &_papi_hwd_ppc7450_preset_map;
   }
   while (native_table[i].resources.selector) i++;
   _papi_hwi_system_info.sub_info.num_native_events = i;
   return (_papi_hwi_setup_all_presets(preset_search_map, NULL));
}

int _papi_hwd_set_domain(hwd_control_state_t * cntrl, int domain)
{
   int did = 0;
   unsigned long tmp = cntrl->control.cpu_control.ppc.mmcr0;
   
   SUBDBG("set domain %d, mmcr0 is %08lX\n",domain,tmp);
   if ((domain == PAPI_DOM_ALL) || (domain == (PAPI_DOM_USER|PAPI_DOM_KERNEL))) {
     did = 1;
     tmp = tmp & PERF_MODE_MASK;
   } else if (domain == PAPI_DOM_KERNEL) {
     did = 1;
     tmp = (tmp & PERF_MODE_MASK) | PERF_OS_ONLY;
   } else if(domain == PAPI_DOM_USER) {
     did = 1;
     tmp = (tmp & PERF_MODE_MASK) | PERF_USR_ONLY;
   }
   SUBDBG("set domain %d, mmcr0 will be %08lX\n",domain,tmp);
   if(!did)
      return(PAPI_EINVAL);
   
   cntrl->control.cpu_control.ppc.mmcr0 = tmp;
   return(PAPI_OK);
}

int _papi_hwd_init_control_state(hwd_control_state_t * ptr) 
{
   _papi_hwd_set_domain(ptr,_papi_hwi_system_info.sub_info.default_domain);
   ptr->allocated_registers.selector = 0;
   ptr->control.cpu_control.tsc_on = 1;
   return(PAPI_OK);
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
   for (i = 0, dst->ra_rank = 0; i < _papi_hwi_system_info.sub_info.num_cntrs; i++)
      if (dst->ra_selector & (1 << i))
         dst->ra_rank++;
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
   dst->ra_selector = src->ra_selector;
}

/* Register allocation */
int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
   hwd_control_state_t *this_state = &ESI->machdep;
   int index, i, j, natNum;
   hwd_reg_alloc_t event_list[MAX_COUNTERS];

   memset(event_list,0x0,sizeof(hwd_reg_alloc_t)*MAX_COUNTERS);

   natNum = ESI->NativeCount;

   /* Initialize the local structure needed
      for counter allocation and optimization. */
   for(i = 0; i < natNum; i++) 
     {
       /* retrieve mapping info */
      index = ESI->NativeInfoArray[i].ni_event & PAPI_NATIVE_AND_MASK;
      SUBDBG("Native event %d index is %d\n",i,index);

      /* Yuck structure assignment */
      memcpy(&event_list[i].ra_bits,&native_table[index].resources,sizeof(event_list[i].ra_bits));
      event_list[i].ra_selector = event_list[i].ra_bits.selector;

      /* calculate native event rank, which is no. of counters it can live on */
      for(j = 0; j < _papi_hwi_system_info.sub_info.num_cntrs; j++) {
         if (event_list[i].ra_selector & (1 << j)) event_list[i].ra_rank++;
      }
      SUBDBG("Can live on %d registers, %08X\n",event_list[i].ra_rank,event_list[i].ra_selector);
   }

   /* Try to find a mapping for all the registers. */

   if(_papi_hwi_bipartite_alloc(event_list, natNum)) 
     { 
       struct hwd_pmc_control *contr = &this_state->control;

       /* for the PPC32, we have funny interrupt bits that control PMC1 and PMC2-n.
	  Thus, due to the PAPI API separating add from overflow, we can only allow
	  1 event with overflow on PMC2-n. Here we see if >=2 events are using PMCS 2-n
	  and >=1 is overflowing. If so, we cannot allow this event to be allocated
          because it too will overflow but the user has not asked it to. */

       if (contr->cpu_control.ppc.mmcr0 & PERF_INT_PMCxEN)
	 {
	   int pmcx_ov_check = 0;
	   for(i = 0; i < natNum; i++) 
	     {
	       if (contr->cpu_control.pmc_map[i] > 0)
		 {
		   if (++pmcx_ov_check > 1)
		     {
		       PAPIERROR("Only 1 event on PMC2-n allowed if overflow is enabled!");
		       return(0);
		     }
		 }
	     }
	 }

      for(i = 0; i < natNum; i++) {
         /* Copy all info about this native event to the NativeInfo struct */
         ESI->NativeInfoArray[i].ni_bits = event_list[i].ra_bits;
         /* Array order on perfctr is event ADD order, not counter #... */
         ESI->NativeInfoArray[i].ni_position = i;
	 this_state->control.cpu_control.pmc_map[i] = ffs(event_list[i].ra_selector) - 1;
	 SUBDBG("bipartite put event_list[%d].ra_selector is %08X, %08X on PMC %d\n",i,event_list[i].ra_selector,event_list[i].ra_bits.selector,ffs(event_list[i].ra_selector)-1);
      }
      return 1;
   } else
      return 0;
}

static void clear_control_state(hwd_control_state_t *this_state) 
{
   this_state->control.cpu_control.nractrs = 0;
   this_state->control.cpu_control.nrictrs = 0;
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */

int _papi_hwd_update_control_state(hwd_control_state_t *this_state,
                                   NativeInfo_t *native, int count, hwd_context_t * ctx) {
   int i;
   /* clear out everything currently coded */
   clear_control_state(this_state);

   SUBDBG("adding %d native events\n",count);
   for (i = 0; i < count; i++) 
     {
       SUBDBG("event %d, counter_cmd 0x%x, selector 0x%x\n",i,native[i].ni_bits.counter_cmd,native[i].ni_bits.selector);
      /* Add counter control command values to eventset */
       this_state->control.cpu_control.evntsel[i] = native[i].ni_bits.counter_cmd;
     }

   this_state->control.cpu_control.nractrs = count;

   print_control(&this_state->control.cpu_control);

   return (PAPI_OK);
}


int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * state) {
   int error;
#ifdef DEBUG
   print_control(&state->control.cpu_control);
#endif
   if (state->rvperfctr != NULL) 
     {
       if((error = rvperfctr_control(state->rvperfctr, &state->control)) < 0) 
	 {
	   SUBDBG("rvperfctr_control returns: %d\n", error);
	   PAPIERROR(RCNTRL_ERROR); 
	   return(PAPI_ESYS); 
	 }
       return (PAPI_OK);
     }
   if((error = vperfctr_control(ctx->perfctr, &state->control)) < 0) {
      SUBDBG("vperfctr_control returns: %d\n", error);
      { PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS); }
   }
   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
   if( state->rvperfctr != NULL ) {
     if(rvperfctr_stop((struct rvperfctr*)ctx->perfctr) < 0)
       { PAPIERROR( RCNTRL_ERROR); return(PAPI_ESYS); }
     return (PAPI_OK);
   }
   if(vperfctr_stop(ctx->perfctr) < 0)
     { PAPIERROR( VCNTRL_ERROR); return(PAPI_ESYS); }
   return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * spc, long long ** dp,  int flags) {
   if ( flags & PAPI_PAUSED ) {
     vperfctr_read_state(ctx->perfctr, &spc->state, NULL);
   }
   else {
      SUBDBG("vperfctr_read_ctrs\n");
      if( spc->rvperfctr != NULL ) {
        rvperfctr_read_ctrs( spc->rvperfctr, &spc->state );
      } else {
        vperfctr_read_ctrs(ctx->perfctr, &spc->state);
        }
   }
      *dp = (long long *) spc->state.pmc;
#ifdef DEBUG
   {
      if (ISLEVEL(DEBUG_SUBSTRATE)) {
         int i;
         for(i = 0; i < spc->control.cpu_control.nractrs; i++) {
            SUBDBG("raw val hardware index %d is %lld\n", i,
                   (long long) spc->state.pmc[i]);
         }
      }
   }
#endif
   return (PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_papi_hwd_start(ctx, cntrl));
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

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) 
{
   hwd_control_state_t *this_state = &ESI->machdep;
   struct hwd_pmc_control *contr = &this_state->control;
   int i, j, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

   OVFDBG("EventIndex=%d\n", EventIndex);
   /* The correct event to overflow is EventIndex */

   /* Set an overflow threshold */
   if (ESI->EventInfoArray[EventIndex].derived) 
     {
       OVFDBG("Can't overflow on a derived event.\n");
       return PAPI_EINVAL;
     }

   ncntrs = _papi_hwi_system_info.sub_info.num_cntrs;
   i = ESI->EventInfoArray[EventIndex].pos[0];
   if (i >= ncntrs) 
     {
       PAPIERROR("Selector id %d is larger than ncntrs %d", i, ncntrs);
       return PAPI_EBUG;
     }

   if (threshold != 0) 
     {
       unsigned long saved_mmcr0 = contr->cpu_control.ppc.mmcr0;
       int pmcx_use_check = 0;
	   
       /* for the PPC32, we have funny interrupt bits that control PMC1 and PMC2-n.
	  Thus, due to the PAPI API separating add from overflow, we can only allow
	  1 event with overflow on PMC2-n. Here we see if >=2 events are using PMCS 2-n
	  and we ask for 1 of those to overflow. If so, we cannot allow this event to 
	  be set to overflow because both of those will overflow and the user has not 
	  specified it to do so. */

       if (contr->cpu_control.pmc_map[i] != 0)
	 {
	   for(j = 0; j < ESI->NativeCount; j++) 
	     {
	       if (contr->cpu_control.pmc_map[j] > 0)
		 {
		   if (++pmcx_use_check > 1)
		     {
		       PAPIERROR("Only 1 event on PMC2-n allowed if overflow is enabled!");
		       return(PAPI_ESBSTR);
		     }
		 }
	     }
	 }

      if (contr->cpu_control.pmc_map[i] == 0)
	contr->cpu_control.ppc.mmcr0 |= PERF_INT_PMC1EN;
      else
	contr->cpu_control.ppc.mmcr0 |= PERF_INT_PMCxEN;
      contr->cpu_control.ppc.mmcr0 |= PERF_INT_ENABLE;

      if ((retval = _papi_hwi_start_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig,NEED_CONTEXT)) != PAPI_OK)
	{
	  contr->cpu_control.ppc.mmcr0 = saved_mmcr0;
	  return(retval);
	}

      /* overflow interrupt occurs on the NEXT event after overflow occurs
         thus we subtract 1 from the threshold. */

      contr->cpu_control.ireset[i] = PMC_OVFL - threshold;
      contr->si_signo = _papi_hwi_system_info.sub_info.hardware_intr_sig;
      nricntrs = ++contr->cpu_control.nrictrs;
      nracntrs = --contr->cpu_control.nractrs;

      /* move this event to the bottom part of the list if needed */
      if (i < nracntrs)
         swap_events(ESI, contr, i, nracntrs);
     }
   else 
     {
       if (contr->cpu_control.ppc.mmcr0 & PERF_INT_ENABLE) 
	 {
	   contr->cpu_control.ireset[i] = 0;
	   nricntrs = --contr->cpu_control.nrictrs;
	   nracntrs = ++contr->cpu_control.nractrs;
	   if (!nricntrs)
	     {
	       contr->cpu_control.ppc.mmcr0 &= (~PERF_INT_ENABLE);
	       contr->si_signo = 0;
	     }

	   if (contr->cpu_control.pmc_map[i] == 0)
	     contr->cpu_control.ppc.mmcr0 &= (~PERF_INT_PMC1EN);
	   else
	     contr->cpu_control.ppc.mmcr0 &= (~PERF_INT_PMCxEN);

	   /* move this event to the top part of the list if needed */
	   if (i >= nracntrs)
	     swap_events(ESI, contr, i, nracntrs - 1);
	 }

      retval = _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.hardware_intr_sig);
   }

#ifdef DEBUG   
   print_control(&contr->cpu_control);
#endif

   return (retval);
}

int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold) {
   /* This function is not used and shouldn't be called. */
   return (PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

  
papi_svector_t _ppc32_vector_table[] = {
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
  {(void (*)())_papi_hwd_set_domain, VEC_PAPI_HWD_SET_DOMAIN},
  {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
  {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
  {(void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
  {(void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
  {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
  {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
  {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
  {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
  {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
  { NULL, VEC_PAPI_END }
};

int ppc32_setup_vector_table(papi_vectors_t * vtable){
  int retval=PAPI_OK; 
 
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _ppc32_vector_table);
#endif
  return ( retval );
}



