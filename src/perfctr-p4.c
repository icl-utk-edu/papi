/* 
* File:    p4.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#include "papi.h"

#ifndef _WIN32
  #include SUBSTRATE
#else
  #include "win32.h"
#endif

#ifdef PAPI3
#include "papi_internal.h"
#include "papi_protos.h"
#endif

/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

/* CPUID model < 2 */
extern P4_search_t _papi_hwd_pentium4_mlt2_preset_map[];

/* CPUID model >= 2 */
extern P4_search_t _papi_hwd_pentium4_mge2_preset_map[];

#ifdef PAPI3
extern papi_mdi_t _papi_hwi_system_info;
#else
#define _papi_hwi_system_info _papi_system_info
papi_mdi_t _papi_system_info = { "$Id$", 
			      1.0, /*  version */
			       -1,  /*  cpunum */
			       { 
				 -1,  /*  ncpu */
				  1,  /*  nnodes */
				 -1,  /*  totalcpus */
				 -1,  /*  vendor */
				 "",  /*  vendor string */
				 -1,  /*  model */
				 "",  /*  model string */
				0.0,  /*  revision */
				0.0  /*  mhz */ 
			       },
			       {
				 "",
				 "",
				 (caddr_t)&_init,
				 (caddr_t)&_etext,
				 (caddr_t)&_etext+1,
				 (caddr_t)&_edata,
				 (caddr_t)NULL,
				 (caddr_t)NULL,
				 "LD_PRELOAD", /* How to preload libs */
			       },
                               { 0,  /*total_tlb_size*/
                                 0,  /*itlb_size */
                                 0,  /*itlb_assoc*/
                                 0,  /*dtlb_size */
                                 0, /*dtlb_assoc*/
                                 0, /*total_L1_size*/
                                 0, /*L1_icache_size*/
                                 0, /*L1_icache_assoc*/
                                 0, /*L1_icache_lines*/
                                 0, /*L1_icache_linesize*/
                                 0, /*L1_dcache_size */
                                 0, /*L1_dcache_assoc*/
                                 0, /*L1_dcache_lines*/
                                 0, /*L1_dcache_linesize*/
                                 0, /*L2_cache_size*/
                                 0, /*L2_cache_assoc*/
                                 0, /*L2_cache_lines*/
                                 0, /*L2_cache_linesize*/
                                 0, /*L3_cache_size*/
                                 0, /*L3_cache_assoc*/
                                 0, /*L3_cache_lines*/
                                 0  /*L3_cache_linesize*/
                               },
			       -1,  /*  num_cntrs */
			       -1,  /*  num_gp_cntrs */
			       -1,  /*  grouped_counters */
			       -1,  /*  num_sp_cntrs */
			       -1,  /*  total_presets */
			       -1,  /*  total_events */
			        PAPI_DOM_USER, /* default domain */
			        PAPI_GRN_THR,  /* default granularity */
			        0,  /* We can use add_prog_event */
			        0,  /* We can write the counters */
			        0,  /* supports HW overflow */
			        0,  /* supports HW profile */
			        1,  /* supports 64 bit virtual counters */
			        1,  /* supports child inheritance */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        1,  /* We can use the virt_usec call */
			        1,  /* We can use the virt_cyc call */
			        0,  /* HW read resets the counters */
			        sizeof(hwd_control_state_t), 
			        { 0, }
};
#endif

/*****************************/
/* END EXTERNAL DECLARATIONS */
/*****************************/

/****************************/
/* BEGIN LOCAL DECLARATIONS */
/****************************/

P4_preset_t _papi_hwd_preset_map[PAPI_MAX_PRESET_EVENTS];

/**************************/
/* END LOCAL DECLARATIONS */
/**************************/

#ifndef PAPI3
int _papi_hwd_query(int preset_index, int *flags, char **note)
{ 
  if (_papi_hwd_preset_map[preset_index].read_selector == 0)
    return(0);
  if (_papi_hwd_preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (_papi_hwd_preset_map[preset_index].note)
    *note = _papi_hwd_preset_map[preset_index].note;
  return(1);
}
#endif

static int setup_presets(P4_search_t *preset_search_map, P4_preset_t *preset_map)
{
  int pnum, unum, preset_index, did_something = 0;
  char *note;

  memset(preset_map,0x0,sizeof(preset_map));
  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      if (preset_search_map[pnum].preset == 0)
	break;
      preset_index = preset_search_map[pnum].preset ^ PRESET_MASK; 

      preset_map[preset_index].control_selector = preset_search_map[pnum].control_selector;
      preset_map[preset_index].read_selector = preset_search_map[pnum].read_selector;
      preset_map[preset_index].derived = NOT_DERIVED;

      /* Number of hardware events in this event */

      preset_map[preset_index].number = preset_search_map[pnum].number;

      /* Fill in the preset's register map of which registers this
	 preset can use. */

#define COUNTER_BITS_FROM_PMC_MAP(a) (1 << (a ^ FAST_RDPMC))
#define ESCR_LOW_BITS_FROM_EVNTSEL(a) (1 << ((a >> 13) & (0x7)))
#define ESCR_HIGH_BITS_FROM_EVNTSEL(a) (0)

      note = preset_map[preset_index].note;
      for (unum = 0; unum < preset_map[preset_index].number; unum++)
	{
	  char tmpnote[PAPI_MAX_STR_LEN];
	  P4_register_t *tmp = &preset_map[preset_index].possible_registers.hardware_event[unum];
	  const P4_perfctr_event_t *tmp2 = &preset_search_map[pnum].info.data[unum];

	  tmp->cccr_bits = COUNTER_BITS_FROM_PMC_MAP(tmp2->pmc_map);
	  SUBDBG("cccr_bits[%d] %#08x\n",unum,tmp->cccr_bits);
	  tmp->escr_low_bits = ESCR_LOW_BITS_FROM_EVNTSEL(tmp2->evntsel);
	  SUBDBG("escr_low_bits[%d] %#08x\n",unum,tmp->escr_low_bits);
	  tmp->uses_pebs = (tmp2->pebs_enable ? 1 : 0);
	  SUBDBG("uses_pebs[%d] %#08x\n",unum,tmp->uses_pebs);
	  tmp->uses_pebs_matrix_vert = (tmp2->pebs_matrix_vert ? 1 : 0);
	  SUBDBG("uses_pebs_matrix_vert[%d] %#08x\n",unum,tmp->uses_pebs_matrix_vert);

	  sprintf(tmpnote,"%s0x%08x/0x%08x@0x%08x",
		  (unum >= 1) ? " " : "",
		  tmp2->evntsel,tmp2->evntsel_aux,tmp2->pmc_map);
	  if ((strlen(note) + strlen(tmpnote)) < (PAPI_MAX_STR_LEN-1))
	    strcat(note,tmpnote);
	}
      preset_map[preset_index].possible_registers.num_hardware_events = preset_map[preset_index].number;
      preset_map[preset_index].info = &preset_search_map[pnum].info;

      if ((preset_search_map[pnum].note) && 
	  ((strlen(note) + 1 + strlen(preset_search_map[pnum].note)) < (PAPI_MAX_STR_LEN-1)))
	{
	  strcat(note,": ");
	  strcat(note,preset_search_map[pnum].note);
	}
      did_something = 1;
    }

  if (did_something == 0)
    error_return(PAPI_ESBSTR,"No presets were defined by the substrate");

  return(PAPI_OK);
}

inline static int setup_all_presets(int cputype)
{
  if (cputype == PERFCTR_X86_INTEL_P4)
    return(setup_presets(_papi_hwd_pentium4_mlt2_preset_map, _papi_hwd_preset_map));
  else if (cputype == PERFCTR_X86_INTEL_P4M2)
    return(setup_presets(_papi_hwd_pentium4_mge2_preset_map, _papi_hwd_preset_map));
  else
    error_return(PAPI_ESBSTR,MODEL_ERROR);
 
  return(PAPI_OK);
}

inline static void init_config(struct vperfctr_control *ptr)
{
  ptr->cpu_control.tsc_on = 1;
  ptr->cpu_control.nractrs = 0;
  ptr->cpu_control.nrictrs = 0;
#if 0
  ptr->interval_usec = sampling_interval;
  ptr->nrcpus = all_cpus;
#endif
}

inline static unsigned long long get_cycles (void)
{
	unsigned long long ret;
        __asm__ __volatile__("rdtsc"
			    : "=A" (ret)
			    : /* no inputs */);
        return ret;
}

inline static int xlate_cpu_type_to_vendor(unsigned perfctr_cpu_type)
{
  switch (perfctr_cpu_type)
    {
    case PERFCTR_X86_INTEL_P5:
    case PERFCTR_X86_INTEL_P5MMX:
    case PERFCTR_X86_INTEL_P6:
    case PERFCTR_X86_INTEL_PII:
    case PERFCTR_X86_INTEL_PIII:
    case PERFCTR_X86_INTEL_P4:
      return(PAPI_VENDOR_INTEL);
    case PERFCTR_X86_AMD_K7:
      return(PAPI_VENDOR_AMD);
    case PERFCTR_X86_CYRIX_MII:
      return(PAPI_VENDOR_CYRIX);
    default:
      return(PAPI_VENDOR_UNKNOWN);
    }
}

/* Dumb hack to make sure I get the cycle time correct. -pjm */

inline static float calc_mhz(void)
{
  unsigned long long ostamp;
  unsigned long long stamp;
  float correction = 4000.0, mhz;

  /* Warm the cache */

  ostamp = get_cycles();
  usleep(1);
  stamp = get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  ostamp = get_cycles();
  sleep(1);
  stamp = get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  return(mhz);
}

/* Called when PAPI/process is initialized */

int _papi_hwd_init_global(void)
{
  int retval;
  struct perfctr_info info;
  float mhz;

  /* Opened once just to get system info */

  struct vperfctr *dev;
  if ((dev = vperfctr_open()) == NULL)
    error_return(PAPI_ESYS,VOPEN_ERROR);
  SUBDBG("_papi_hwd_init_global vperfctr_open = %p\n",dev);

  /* Get info from the kernel */

  if (vperfctr_info(dev, &info) < 0)
    error_return(PAPI_ESYS,VINFO_ERROR);

  strcpy(_papi_hwi_system_info.hw_info.model_string,perfctr_cpu_name(&info));
  _papi_hwi_system_info.supports_hw_overflow = 
    (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;
  SUBDBG("Hardware/OS %s support counter generated interrupts\n",
       _papi_hwi_system_info.supports_hw_overflow ? "does" : "does not");

  _papi_hwi_system_info.num_cntrs = perfctr_cpu_nrctrs(&info);
  _papi_hwi_system_info.num_gp_cntrs = perfctr_cpu_nrctrs(&info);

  _papi_hwi_system_info.hw_info.model = info.cpu_type;
  _papi_hwi_system_info.hw_info.vendor = xlate_cpu_type_to_vendor(info.cpu_type);

  _papi_hwi_system_info.hw_info.mhz = (float)info.cpu_khz / 1000.0; 

  SUBDBG("Detected MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz);
  mhz = calc_mhz();
  SUBDBG("Calculated MHZ is %f\n",mhz);
  /* If difference is larger than 5% (e.g. system info is 0) use 
     calculated value. (If CPU value seems reasonable use it) */
  if (abs(mhz-_papi_hwi_system_info.hw_info.mhz) > 0.95*_papi_hwi_system_info.hw_info.mhz)
    _papi_hwi_system_info.hw_info.mhz = mhz;
  SUBDBG("Actual MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz);

  /* Setup presets */

  retval = setup_all_presets(info.cpu_type);
  if (retval)
    return(retval);

  /* Fill in what we can of the papi_system_info. */
  
  retval = _papi_hwd_get_system_info();
  if (retval != PAPI_OK)
    return(retval);
  
  vperfctr_close(dev);
  SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n",dev);

  return(PAPI_OK);
}

/* Called when thread is initialized */

int _papi3_hwd_init(P4_perfctr_context_t *ctx)
{
  struct vperfctr_control tmp;

  /* Malloc the space for our controls */
  
#if 0
  ctx->start.control = (struct vperfctr_control *)malloc(sizeof(struct vperfctr_control));
  ctx->start.state = (struct perfctr_sum_ctrs *)malloc(sizeof(struct perfctr_sum_ctrs));
  if ((ctx->start.control == NULL) || (ctx->start.state == NULL))
    error_return(PAPI_ENOMEM, STATE_MAL_ERROR);
#endif

  /* Initialize our thread/process pointer. */

  if ((ctx->perfctr = vperfctr_open()) == NULL) 
    error_return(PAPI_ESYS,VOPEN_ERROR);
  SUBDBG("_papi_hwd_init vperfctr_open() = %p\n",ctx->perfctr);
#if 0
  if ((ctx->perfctr = gperfctr_open()) == NULL) 
    error_return(PAPI_ESYS,GOPEN_ERROR);
  SUBDBG("_papi_hwd_init gperfctr_open() = %p\n",ctx->perfctr);
#endif

  /* Initialize the per thread/process virtualized TSC */

  memset(&tmp,0x0,sizeof(tmp));
  init_config(&tmp);
 
  /* Start the per thread/process virtualized TSC */

  if (vperfctr_control(ctx->perfctr, &tmp) < 0)
    error_return(PAPI_ESYS,VCNTRL_ERROR);
#if 0
  if (gperfctr_control(ctx->perfctr, &tmp) < 0)
    error_return(PAPI_ESYS,GCNTRL_ERROR);
#endif

  return(PAPI_OK);
}

#ifndef PAPI3
int _papi_hwd_init(EventSetInfo *zero)
{
  hwd_control_state_t *machdep = zero->machdep;
  int retval;
  if ((retval = _papi3_hwd_init(&machdep->context )) < PAPI_OK)
    return(retval);
  init_config(&machdep->control.control);
  return(retval);
}
#endif

#if 0
int _papi_hwd_allocate_hwcounters(const P4_perfctr_control_t *state, const P4_perfctr_event_t *add)
{
  int i, num = state->control.cpu_control.nractrs + state->control.cpu_control.nrictrs;

  /* Here we should analyze the current state and the preset and choose the best registers */
  /* Instead everything is hard coded. */

  // memcpy(add,preset,sizeof(P4_perfctr_event_t));

  /* if adding a a-mode after adding both an i-mode, then error
     instead of shuffling the whole structure around */

  if ((state->control.cpu_control.nrictrs) && (add->ireset == 0))
    error_return(PAPI_ECNFLCT, AI_ERROR);
     
  for (i=0;i<num;i++)
    {
      if (state->control.cpu_control.pmc_map[i] ==
	  add->pmc_map)
	error_return(PAPI_ECNFLCT,"current %#08x vs. to add %#08x",
		     state->control.cpu_control.pmc_map[i],
		     add->pmc_map);
    }
  return(PAPI_OK);
}
#endif

#ifdef DEBUG
void print_control(const struct perfctr_cpu_control *control)
{
    unsigned int i;

    SUBDBG("Control used:\n");
    SUBDBG("tsc_on\t\t\t%u\n", control->tsc_on);
    SUBDBG("nractrs\t\t\t%u\n", control->nractrs);
    for(i = 0; i < control->nractrs; ++i) {
	if( control->pmc_map[i] >= 18 )
	  {
	    SUBDBG("pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i]);
	  }
	else
	  {
	    SUBDBG("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
	  }
	SUBDBG("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
	if( control->evntsel_aux[i] )
	    SUBDBG("evntsel_aux[%u]\t0x%08X\n", i, control->evntsel_aux[i]);
    }
    if( control->p4.pebs_enable )
      SUBDBG("pebs_enable\t0x%08X\n", 
	     control->p4.pebs_enable);
    if( control->p4.pebs_matrix_vert )
      SUBDBG("pebs_matrix_vert\t0x%08X\n", 
	     control->p4.pebs_matrix_vert);
}
#endif

int _papi_hwd_start(P4_perfctr_context_t *ctx, P4_perfctr_control_t *state)
{
  if (vperfctr_control(ctx->perfctr, &state->control) < 0)
    error_return(PAPI_ESYS,VCNTRL_ERROR);
#if 0
  if (gperfctr_control(ctx->perfctr, &state->control) < 0)
    error_return(PAPI_ESYS,GCNTRL_ERROR);
#endif

#ifdef DEBUG
  print_control(&state->control.cpu_control);
#endif

  return(PAPI_OK);
}

#ifndef PAPI3
int _papi_hwd_merge(EventSetInfo *this_evset, EventSetInfo *context_evset)
{
  hwd_control_state_t *machdep = this_evset->machdep;
  hwd_control_state_t *context_machdep = context_evset->machdep;
  return(_papi_hwd_start(&context_machdep->context, &machdep->control));
}
#endif

int _papi_hwd_stop(P4_perfctr_context_t *ctx, P4_perfctr_control_t *state)
{
  if (vperfctr_stop(ctx->perfctr) < 0)
    error_return(PAPI_ESYS,VCNTRL_ERROR);
#if 0
  if (gperfctr_stop(ctx->perfctr) < 0)
    error_return(PAPI_ESYS,GCNTRL_ERROR);
#endif

  return(PAPI_OK);
}

#ifndef PAPI3
int _papi_hwd_unmerge(EventSetInfo *this_evset, EventSetInfo *context_evset)
{
  hwd_control_state_t *machdep = this_evset->machdep;
  hwd_control_state_t *context_machdep = context_evset->machdep;
  return(_papi_hwd_stop(&context_machdep->context, &machdep->control));
}
#endif

int _papi3_hwd_read(P4_perfctr_context_t *ctx, P4_perfctr_control_t *spc, unsigned long long **dp)
{
  vperfctr_read_ctrs(ctx->perfctr, &spc->state);
  *dp = spc->state.pmc;
#ifdef DEBUG
  {
    int i;
    for (i=0;i<spc->control.cpu_control.nractrs;i++)
      {
	SUBDBG("raw val hardware index %d is %lld\n",i,spc->state.pmc[i]);
      }
  }
#endif
  return(PAPI_OK);
}

#ifndef PAPI3
int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long events[])
{
  unsigned long long *dp;
  int shift_cnt, selector, i, j = 0;
  hwd_control_state_t *machdep = zero->machdep;
  hwd_control_state_t *evset_machdep = ESI->machdep;
  _papi3_hwd_read(&machdep->context,&evset_machdep->control,&dp);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      selector = ESI->EventInfoArray[i].selector;
      if (selector == PAPI_NULL)
	continue;

      DBG((stderr,"Event index %d, selector is 0x%x\n",j,selector));

      /* If this is not a derived event */

      if (ESI->EventInfoArray[i].command == NOT_DERIVED)
	{
	  shift_cnt = ffs(selector) - 1;
	  assert(shift_cnt >= 0);
	  events[j] = dp[shift_cnt];
	}

      /* If this is a derived event */

//      else 
//	events[j] = handle_derived(&ESI->EventInfoArray[i], correct_hw_order);

      /* Early exit! */

      if (++j == ESI->NumberOfEvents)
	{
	  DBG((stderr,"Done\n"));
	  return(PAPI_OK);
	}
    }

  /* Should never get here */

  return(PAPI_EBUG);
}
#endif

/* This routine is for shutting down threads, including the
   master thread. */

int _papi3_hwd_shutdown(P4_perfctr_context_t *ctx)
{
  int retval = vperfctr_unlink(ctx->perfctr);
  SUBDBG("_papi_hwd_init_global vperfctr_unlink(%p) = %d\n",ctx->perfctr,retval);
  vperfctr_close(ctx->perfctr);
  SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n",ctx->perfctr);
  memset(ctx,0x0,sizeof(P4_perfctr_context_t));

  if (retval)
    return(PAPI_ESYS);
  return(PAPI_OK);
}

#ifndef PAPI3
int _papi_hwd_shutdown(EventSetInfo *zero)
{
  hwd_control_state_t *machdep = zero->machdep;
  return(_papi3_hwd_shutdown(&machdep->context));
}
#endif

/* Called once per process. */

int _papi_hwd_shutdown_global(void)
{
   return(PAPI_OK);
}

/* Timers */

u_long_long _papi3_hwd_get_real_usec (void)
 {
  return((u_long_long)get_cycles() / (u_long_long)_papi_hwi_system_info.hw_info.mhz);
}

u_long_long _papi3_hwd_get_real_cycles (void)
{
  return(get_cycles());
}

u_long_long _papi3_hwd_get_virt_cycles (const P4_perfctr_context_t *ctx)
{
  return(vperfctr_read_tsc(ctx->perfctr));
}

u_long_long _papi3_hwd_get_virt_usec (const P4_perfctr_context_t *ctx)
{
  return(_papi3_hwd_get_virt_cycles(ctx) / (u_long_long)_papi_hwi_system_info.hw_info.mhz);
}

#ifndef PAPI3
long_long _papi_hwd_get_real_cycles ()
{
  return(_papi3_hwd_get_real_cycles());
}
long_long _papi_hwd_get_real_usec ()
{
  return(_papi3_hwd_get_real_usec());
}
long_long _papi_hwd_get_virt_cycles (EventSetInfo *zero)
{
  hwd_control_state_t *machdep = zero->machdep;
  return(_papi3_hwd_get_virt_cycles(&machdep->context));
}
long_long _papi_hwd_get_virt_usec (EventSetInfo *zero)
{
  hwd_control_state_t *machdep = zero->machdep;
  return(_papi3_hwd_get_virt_usec(&machdep->context));
}
#endif

/* Register allocation */

int _papi_hwd_allocate_registers(P4_perfctr_control_t *evset_info, P4_preset_t *from, P4_regmap_t *out)
{
  int i, num;
  unsigned tmp_u = 0x0;
//  unsigned tmp_u_high = 0x0;
  P4_register_t *needed_tmp;
  P4_regmap_t *needed;
  P4_register_t *already_taken;

  /* See comments in papi.c under add_preset_event */

  /* get the already allocated counters for this eventset */

  already_taken = &evset_info->allocated_registers;

  /* get the needed counters and number of them from preset */

  needed = &from->possible_registers;
  num = from->number;

  for (i=0;i<num;i++)
    {
      /* First the CCCR's */

      needed_tmp = &needed->hardware_event[i];
      tmp_u = needed_tmp->cccr_bits & already_taken->cccr_bits;
      if (!tmp_u)
	{
	  out->hardware_event[i].cccr_bits = 1 << (ffs(needed_tmp->cccr_bits) - 1);
	}
      else
	error_return(PAPI_ECNFLCT,"needed %#08x vs. already_taken %#08x",
		     needed_tmp->cccr_bits, already_taken->cccr_bits);

      /* Now the ESCR's */

      tmp_u = needed_tmp->escr_low_bits & already_taken->escr_low_bits;
      if (!tmp_u)
	{
	  out->hardware_event[i].escr_low_bits = 1 << (ffs(needed_tmp->escr_low_bits) - 1);
	}
//      tmp_u_high = needed_tmp->escr_high_bits & already_taken->escr_high_bits;
/*      else if (!tmp_u_high)
	{
	  out->hardware_event[i].escr_high_bits = 1 << (ffs(needed_tmp->escr_high_bits) - 1);
	} */
      else
	error_return(PAPI_ECNFLCT,"needed %#08x vs. already_taken %#08x",
		     needed_tmp->escr_low_bits, already_taken->escr_low_bits);

      /* Now the PEBS goodies */

      if (!(needed_tmp->uses_pebs & already_taken->uses_pebs))
	out->hardware_event[i].uses_pebs = 1;
      else
	error_return(PAPI_ECNFLCT,"PEBS already in use");

      if (!(needed_tmp->uses_pebs_matrix_vert & already_taken->uses_pebs_matrix_vert))
	out->hardware_event[i].uses_pebs_matrix_vert = 1;
      else
	error_return(PAPI_ECNFLCT,"PEBS MATRIX VERT already in use");
    }
  return(PAPI_OK);
}

/* After this function is called, ESI->machdep has everything it needs to do a start/read/stop 
   as quickly as possible. This returns the position in the array output by _papi_hwd_read that
   this register lives in. */

int _papi3_hwd_add_event(P4_regmap_t *ev_info, P4_preset_t *preset, 
			 P4_perfctr_control_t *evset_info)
{
  int i, index, mask = 0;
  P4_register_t *bits = &evset_info->allocated_registers;
  
  /* Add allocated register bits to this events bits */

  for (i=0;i<preset->number;i++)
    {
      if (ev_info->hardware_event[i].cccr_bits)
	{
	  bits->cccr_bits |= ev_info->hardware_event[i].cccr_bits;
	  bits->cccr_num += 1;
	}
      if (ev_info->hardware_event[i].escr_low_bits)
	{
	  bits->escr_low_bits |= ev_info->hardware_event[i].escr_low_bits;
	  bits->escr_low_num += 1;
	}
      if (ev_info->hardware_event[i].uses_pebs)
	{
	  bits->uses_pebs = 1;
	}
      if (ev_info->hardware_event[i].uses_pebs_matrix_vert)
	{
	  bits->uses_pebs_matrix_vert = 1;
	}
#if 0      
      if (ev_info->hardware_event[i].escr_high_bits)
	{
	  bits->escr_high_bits |= ev_info->hardware_event[i].escr_high_bits;
	  bits->escr_high_num += 1;
	} 
#endif
    }

  /* Add counter control command values to eventset */

  index = evset_info->control.cpu_control.nractrs;
  for (i=0;i<preset->number;i++)
    {
      evset_info->control.cpu_control.pmc_map[index] = preset->info->data[i].pmc_map;
      evset_info->control.cpu_control.evntsel[index] = preset->info->data[i].evntsel;
      evset_info->control.cpu_control.evntsel_aux[index] = preset->info->data[i].evntsel_aux;
      evset_info->control.cpu_control.ireset[index] = 0;
      if (preset->info->data[i].pebs_enable)
	evset_info->control.cpu_control.p4.pebs_enable = preset->info->data[i].pebs_enable;
      if (preset->info->data[i].pebs_matrix_vert)
	evset_info->control.cpu_control.p4.pebs_matrix_vert = preset->info->data[i].pebs_matrix_vert;
      mask |= 1 << index;
      index++;
    }
  evset_info->control.cpu_control.nractrs = index;
  
  /* Make sure the TSC is always on */

  evset_info->control.cpu_control.tsc_on = 1;

#ifdef DEBUG
//  print_control(&evset_info->control.cpu_control);
#endif

  /* Update this specific events structure containing number of events. For compound events like
     PAPI_FP_INS on the Pentium 4, this is > 1. */

  ev_info->num_hardware_events += preset->number;

/* Events that require tagging should be ordered such that the
   first event is the one that is read. See PAPI_FP_INS for an example. */
  
  /* Here we return the hardware index of the event we are reading. */

#ifndef PAPI3
  return(mask);
#else
  return(index-preset->number);
#endif
}

#ifndef PAPI3
int _papi_hwd_add_event(hwd_control_state_t *this_state, 
			unsigned int EventCode, EventInfo_t *out)
{
  int preset_index = EventCode & PRESET_AND_MASK; 
  /* should be num presets check here */
  int selector = ffs(_papi_hwd_preset_map[preset_index].read_selector) - 1;
  P4_regmap_t *regmap = &_papi_hwd_preset_map[EventCode ^ PRESET_MASK].possible_registers;
  P4_preset_t *preset = &_papi_hwd_preset_map[EventCode ^ PRESET_MASK];
  int hwindex;

  /* For each event in the preset */
  /*   get registers that that event requires */
  /*   if selector = ffs(avail & registers) 
         this_state->selector |= 1 << selector;
	 out->selector = selector;
         stuff next pmc slot with data at preset
  */

  /*if (selector == 0 ) - I believe this was suppose to be -1 -KSL*/
  if (selector == -1 )
    return(PAPI_ENOEVNT);

  hwindex = _papi3_hwd_add_event(regmap,preset,&this_state->control);

  out->code = EventCode;
  out->selector = hwindex;

  return(PAPI_OK);
}
#endif

int _papi_hwd_remove_event(P4_regmap_t *ev_info, int hardware_index, P4_perfctr_control_t *evset_info)
{
  int i, j, index, clear_pebs = 0, clear_pebs_matrix_vert = 0;
  P4_register_t *bits = &evset_info->allocated_registers;

  /* Remove allocation/usage info for this event from bits */

  for (i=0;i<ev_info->num_hardware_events;i++)
    {
      if (ev_info->hardware_event[i].cccr_bits)
	{
	  bits->cccr_bits ^= ev_info->hardware_event[i].cccr_bits;
	  bits->cccr_num -= 1;
	}
      if (ev_info->hardware_event[i].escr_low_bits)
	{
	  bits->escr_low_bits ^= ev_info->hardware_event[i].escr_low_bits;
	  bits->escr_low_num -= 1;
	}
/*      if (ev_info->hardware_event[i].escr_high_bits)
	{
	  bits->escr_high_bits ^= ev_info->hardware_event[i].escr_high_bits;
	  bits->escr_high_num -= 1;
	} */
      if (ev_info->hardware_event[i].uses_pebs)
	{
	  clear_pebs = 1;
	  bits->uses_pebs = 0;
	}
      if (ev_info->hardware_event[i].uses_pebs_matrix_vert)
	{
	  clear_pebs_matrix_vert = 1;
	  bits->uses_pebs_matrix_vert = 0;
	}
    }

  /* Remove counter control command values from eventset. We must do this
     very carefully. As the command structure passed to perfctr must be 
     a densely populated array, we need to shift our entries back to
     the front of the array. */
  
  index = evset_info->control.cpu_control.nractrs;

  /* Zero the control entries that were used */

  for (j=hardware_index;j<ev_info->num_hardware_events+hardware_index;j++)
    {
      SUBDBG("Clearing pmc event entry %d\n",j);
      evset_info->control.cpu_control.pmc_map[j] = 0;
      evset_info->control.cpu_control.evntsel[j] = 0;
      evset_info->control.cpu_control.evntsel_aux[j] = 0;
      evset_info->control.cpu_control.ireset[j] = 0;
    }

  /* Clear pebs stuff if we used it */

  if (clear_pebs)
    evset_info->control.cpu_control.p4.pebs_enable = 0;
  if (clear_pebs_matrix_vert)
    evset_info->control.cpu_control.p4.pebs_matrix_vert = 0;

  /* Shift the entries */

  for (j=hardware_index;j<index;j++)
    {
      evset_info->control.cpu_control.pmc_map[j] = evset_info->control.cpu_control.pmc_map[j+ev_info->num_hardware_events];
      evset_info->control.cpu_control.evntsel[j] = evset_info->control.cpu_control.evntsel[j+ev_info->num_hardware_events];
      evset_info->control.cpu_control.evntsel_aux[j] = evset_info->control.cpu_control.evntsel_aux[j+ev_info->num_hardware_events];
      evset_info->control.cpu_control.ireset[j] = evset_info->control.cpu_control.ireset[j+ev_info->num_hardware_events];
    }
	  
  evset_info->control.cpu_control.nractrs = index-ev_info->num_hardware_events;
  
#ifdef DEBUG
  print_control(&evset_info->control.cpu_control);
#endif

  return(PAPI_OK);
}
#ifndef PAPI3
int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  int j, selector = in->selector;
  P4_register_t *bits = &this_state->control.allocated_registers;
  
  SUBDBG("selector is %x\n",selector);
  while ((j = ffs(selector)))
    {
      j--;
      selector ^= (1 << j);
      SUBDBG("Clearing pmc event entry %d\n",j);
      this_state->control.control.cpu_control.pmc_map[j] = 0;
      this_state->control.control.cpu_control.evntsel[j] = 0;
      this_state->control.control.cpu_control.evntsel_aux[j] = 0;
      this_state->control.control.cpu_control.ireset[j] = 0;
    }
  return(PAPI_OK);
}
#endif

int _papi3_hwd_add_prog_event(P4_perfctr_control_t *state, int code, void *tmp, 
			      EventInfo_t *tmp2)
{
  return(PAPI_ESBSTR);
}

#ifndef PAPI3
int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)
{
  return(_papi3_hwd_add_prog_event(&this_state->control, event, extra, out));
}
#endif

#if 0 
int _papi_hwd_add_event(P4_perfctr_control_t *state, const P4_perfctr_event_t *add)
{ 
  int num = state->control.cpu_control.nractrs + state->control.cpu_control.nrictrs;
  
  state->control.cpu_control.pmc_map[num] = add->pmc_map;
  state->control.cpu_control.evntsel[num] = add->evntsel;
  state->control.cpu_control.evntsel_aux[num] = add->evntsel_aux;
  if (add->ireset)
    {
      state->control.cpu_control.ireset[num] = add->ireset;
      state->control.cpu_control.nrictrs++;
    }
  else
    state->control.cpu_control.nractrs++;
 
  return(PAPI_OK);
}
#endif 

int _papi3_hwd_set_domain(P4_perfctr_control_t *cntrl, int domain)
{
  return(PAPI_ESBSTR);
}

int _papi3_hwd_reset(P4_perfctr_context_t *ctx, P4_perfctr_control_t *cntrl)
{
  return(PAPI_ESBSTR);
}

int _papi3_hwd_write(P4_perfctr_context_t *ctx, P4_perfctr_control_t *cntrl, long long *from)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, EventSetOverflowInfo_t *overflow_option)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

#ifndef PAPI3
int _papi_hwd_reset(EventSetInfo *mine, EventSetInfo *zero) 
{
  hwd_control_state_t *machdep = zero->machdep;
  return(_papi3_hwd_reset(&machdep->context,&machdep->control));
}

int _papi_hwd_write(EventSetInfo *mine, EventSetInfo *zero, long_long events[])
{
  hwd_control_state_t *machdep = zero->machdep;
  return(_papi3_hwd_write(&machdep->context,&machdep->control,events));
}
#endif

void _papi_hwd_dispatch_timer(int signal, siginfo_t *info, void *tmp)
{
  ucontext_t *uc;
  mcontext_t *mc;
  gregset_t *gs;

  uc = (ucontext_t *) tmp;
  mc = &uc->uc_mcontext;
  gs = &mc->gregs;

  DBG((stderr,"Start at 0x%x\n",(*gs)[15]));
  _papi_hwi_dispatch_overflow_signal(mc); 

  /* We are done, resume interrupting counters */
#ifdef PAPI_PERFCTR_INTR_SUPPORT
  if(_papi_hwi_system_info.supports_hw_overflow)
    {
      ThreadInfo_t *master = _papi_hwi_lookup_in_master_list();
      hwd_control_state_t *machdep;
      struct vperfctr* dev;

      if(master==NULL)
	{
	  fprintf(stderr,"%s():%d: master event lookup failure! abort()\n",
		  __FUNCTION__,__LINE__);
	  abort();
	}
      machdep =  master->machdep;
      dev = machdep->self;
      /* This is currently disabled since the restart of the counter */
      /* is made in update_global_counters out of unknown reasons    */
      /* if(vperfctr_isrun(machdep->self))                           */
      /*   if(vperfctr_iresume(machdep->self)<0)                     */
      /*     {                                                       */
      /*       perror("vperfctr_iresume");                           */
      /*       abort();                                              */
      /*     }                                                       */
    }
#endif
  DBG((stderr,"Finished at 0x%x\n",(*gs)[15]));
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->eip;

  return(location);
}

