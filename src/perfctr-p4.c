/* 
* File:    p4.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London 
*	   london@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
  #define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"

#ifdef PERFCTR25
#define PERFCTR_CPU_NAME   perfctr_info_cpu_name
#define PERFCTR_CPU_NRCTRS perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS perfctr_cpu_nrctrs
#endif

/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

#ifdef __i386__
/* CPUID model < 2 */
extern preset_search_t _papi_hwd_pentium4_mlt2_preset_map[];
/* CPUID model >= 2 */
extern preset_search_t _papi_hwd_pentium4_mge2_preset_map[];
#elif defined(__x86_64__)
extern P4_search_t _papi_hwd_x86_64_opteron_map[];
#endif

extern papi_mdi_t _papi_hwi_system_info;

/*****************************/
/* END EXTERNAL DECLARATIONS */
/*****************************/

/****************************/
/* BEGIN LOCAL DECLARATIONS */
/****************************/

/**************************/
/* END LOCAL DECLARATIONS */
/**************************/

inline static int setup_p4_presets(int cputype)
{
#ifdef __i386__
  if (cputype == PERFCTR_X86_INTEL_P4)
    return(_papi_hwi_setup_all_presets(_papi_hwd_pentium4_mlt2_preset_map));
  else if (cputype == PERFCTR_X86_INTEL_P4M2)
    return(_papi_hwi_setup_all_presets(_papi_hwd_pentium4_mge2_preset_map));
  else
    error_return(PAPI_ESBSTR,MODEL_ERROR);
#elif defined(__x86_64__)
  if (PERFCTR_X86_AMD_K8)
    return(_papi_hwi_setup_all_presets(_papi_hwd_x86_64_opteron_map));
  else
    error_return(PAPI_ESBSTR,MODEL_ERROR);
#endif
  return(PAPI_OK);
}

/* This used to be init_config, static to the substrate.
   Now its exposed to the hwi layer and called when an EventSet is allocated.
*/
void _papi_hwd_init_control_state(hwd_control_state_t *ptr)
{
  ptr->control.cpu_control.tsc_on = 1;
  ptr->control.cpu_control.nractrs = 0;
  ptr->control.cpu_control.nrictrs = 0;
#if 0
  ptr->interval_usec = sampling_interval;
  ptr->nrcpus = all_cpus;
#endif
}

inline static u_long_long get_cycles (void)
{
	u_long_long ret;
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
  u_long_long ostamp;
  u_long_long stamp;
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

/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
extern int _papi_hwd_mdi_init() {
   strcpy(_papi_hwi_system_info.substrate, "$Id$");     /* Name of the substrate we're using */
   _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t)&_init;
   _papi_hwi_system_info.exe_info.address_info.text_end   = (caddr_t)&_etext;
   _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t)&_etext+1;
   _papi_hwi_system_info.exe_info.address_info.data_end   = (caddr_t)&_edata;
   _papi_hwi_system_info.exe_info.address_info.bss_start  = (caddr_t)NULL;
   _papi_hwi_system_info.exe_info.address_info.bss_end    = (caddr_t)NULL;

   _papi_hwi_system_info.supports_hw_overflow           = 1;
   _papi_hwi_system_info.supports_64bit_counters        = 1;
   _papi_hwi_system_info.supports_inheritance           = 1;
   _papi_hwi_system_info.supports_real_usec             = 1;
   _papi_hwi_system_info.supports_real_cyc              = 1;
   _papi_hwi_system_info.supports_virt_usec             = 1;
   _papi_hwi_system_info.supports_virt_cyc              = 1;

   _papi_hwi_system_info.shlib_info.map->text_start      = (caddr_t)&_init;
   _papi_hwi_system_info.shlib_info.map->text_end        = (caddr_t)&_etext;
   _papi_hwi_system_info.shlib_info.map->data_start      = (caddr_t)&_etext+1;
   _papi_hwi_system_info.shlib_info.map->data_end        = (caddr_t)&_edata;
   _papi_hwi_system_info.shlib_info.map->bss_start       = (caddr_t)NULL;
   _papi_hwi_system_info.shlib_info.map->bss_end         = (caddr_t)NULL;

   return(PAPI_OK);
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

  /* Initialize outstanding values in machine info structure */
  if(_papi_hwd_mdi_init() != PAPI_OK) {
    return(PAPI_EINVAL);
  }
  strcpy(_papi_hwi_system_info.hw_info.model_string,PERFCTR_CPU_NAME(&info));
  _papi_hwi_system_info.supports_hw_overflow = 
    (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;
  SUBDBG("Hardware/OS %s support counter generated interrupts\n",
       _papi_hwi_system_info.supports_hw_overflow ? "does" : "does not");

  _papi_hwi_system_info.num_cntrs = PERFCTR_CPU_NRCTRS(&info);
  _papi_hwi_system_info.num_gp_cntrs = PERFCTR_CPU_NRCTRS(&info);

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

  retval = setup_p4_presets(info.cpu_type);
  if (retval)
    return(retval);

  /* Fill in what we can of the papi_system_info. */
  
  retval = _papi_hwd_get_system_info();
  if (retval != PAPI_OK)
    return(retval);
  
  /* Setup memory info */

  retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, (int)info.cpu_type);
  if (retval)
    return(retval);

#ifdef PERFCTR25
  SUBDBG("perfctr ABI compile time version: %x\n",PERFCTR_ABI_VERSION);
#endif

  vperfctr_close(dev);
  SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n",dev);

  return(PAPI_OK);
}

/* Called when thread is initialized */

int _papi_hwd_init(P4_perfctr_context_t *ctx)
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
  tmp.cpu_control.tsc_on = 1;

  /* Start the per thread/process virtualized TSC */
  if (vperfctr_control(ctx->perfctr, &tmp) < 0)
    error_return(PAPI_ESYS,VCNTRL_ERROR);
#if 0
  if (gperfctr_control(ctx->perfctr, &tmp) < 0)
    error_return(PAPI_ESYS,GCNTRL_ERROR);
#endif

  return(PAPI_OK);
}

int _papi_hwd_setmaxmem(){
  return(PAPI_OK);
}

#ifdef DEBUG
void print_control(const struct perfctr_cpu_control *control)
{
    unsigned int i;

    SUBDBG("Control used:\n");
    SUBDBG("tsc_on\t\t\t%u\n", control->tsc_on);
    SUBDBG("nractrs\t\t\t%u\n", control->nractrs);
    SUBDBG("nrictrs\t\t\t%u\n", control->nrictrs);
    for(i = 0; i < (control->nractrs+control->nrictrs); ++i) {
//    for(i = 0; i < 4; ++i) {
	if( control->pmc_map[i] >= 18 )
	  {
	    SUBDBG("pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i]);
	  }
	else
	  {
	    SUBDBG("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
	  }
	SUBDBG("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
#ifdef __i386__
	if( control->evntsel_aux[i] )
	    SUBDBG("evntsel_aux[%u]\t0x%08X\n", i, control->evntsel_aux[i]);
#endif
	if (control->ireset[i]) 
	  SUBDBG("ireset[%u]\t%d\n",i,control->ireset[i]);
    }
#ifdef __i386__
    if( control->p4.pebs_enable )
      SUBDBG("pebs_enable\t0x%08X\n", 
	     control->p4.pebs_enable);
    if( control->p4.pebs_matrix_vert )
      SUBDBG("pebs_matrix_vert\t0x%08X\n", 
	     control->p4.pebs_matrix_vert);
#endif
}
#endif

int _papi_hwd_start(P4_perfctr_context_t *ctx, P4_perfctr_control_t *state)
{
  int error;

#ifdef DEBUG
  print_control(&state->control.cpu_control);
#endif

  error = vperfctr_control(ctx->perfctr, &state->control);
  if (error < 0) {
    SUBDBG("vperfctr_control returns: %d\n",error);
    error_return(PAPI_ESYS,VCNTRL_ERROR);
  }
#if 0
  if (gperfctr_control(ctx->perfctr, &state->control) < 0)
    error_return(PAPI_ESYS,GCNTRL_ERROR);
#endif

  return(PAPI_OK);
}

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


int _papi_hwd_read(P4_perfctr_context_t *ctx, P4_perfctr_control_t *spc, long_long **dp)
{
  vperfctr_read_ctrs(ctx->perfctr, &spc->state);
  *dp = (long_long*) spc->state.pmc;
#ifdef DEBUG
 {
   extern int _papi_hwi_debug;
   if (_papi_hwi_debug)
     {
       int i;
       for (i=0;i<spc->control.cpu_control.nractrs;i++)
	 {
	   SUBDBG("raw val hardware index %d is %lld\n",i,(long_long)spc->state.pmc[i]);
	 }
     }
 }
#endif
  return(PAPI_OK);
}


/* This routine is for shutting down threads, including the
   master thread. */

int _papi_hwd_shutdown(P4_perfctr_context_t *ctx)
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

/* Called once per process. */

int _papi_hwd_shutdown_global(void)
{
   return(PAPI_OK);
}

/* Timers */

u_long_long _papi_hwd_get_real_usec (void)
 {
  return((u_long_long)get_cycles() / (u_long_long)_papi_hwi_system_info.hw_info.mhz);
}

u_long_long _papi_hwd_get_real_cycles (void)
{
  return(get_cycles());
}

u_long_long _papi_hwd_get_virt_cycles (const P4_perfctr_context_t *ctx)
{
  return(vperfctr_read_tsc(ctx->perfctr));
}

u_long_long _papi_hwd_get_virt_usec (const P4_perfctr_context_t *ctx)
{
  return(_papi_hwd_get_virt_cycles(ctx) / (u_long_long)_papi_hwi_system_info.hw_info.mhz);
}

#ifdef DEBUG
static void print_bits(P4_register_t *b) {
    SUBDBG("  counter[0,1]: 0x%x, 0x%x\n", b->counter[0], b->counter[1]);
    SUBDBG("  escr[0,1]: 0x%x, 0x%x\n", b->escr[0], b->escr[1]);
    SUBDBG("  cccr: 0x%x,  event: 0x%x\n", b->cccr, b->event);
    SUBDBG("  pebs_enable: 0x%x,  pebs_matrix_vert: 0x%x,  ireset: 0x%x\n", b->pebs_enable, b->pebs_matrix_vert, b->ireset);
}

static void print_alloc(P4_reg_alloc_t *a){
    SUBDBG("P4_reg_alloc:\n");
    print_bits(&(a->ra_bits));
    SUBDBG("  selector: 0x%x\n", a->ra_selector);
    SUBDBG("  rank: 0x%x\n", a->ra_rank);
    SUBDBG("  escr: 0x%x 0x%x\n", a->ra_escr[0], a->ra_escr[1]);
}
#endif

/* This function examines the event to determine
    if it can be mapped to counter ctr. 
    Returns true if it can, false if it can't.
*/
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr)
{
    return(dst->ra_selector  & (1<<ctr));
}

/* This function forces the event to
    be mapped to only counter ctr. 
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr)
{
    dst->ra_selector = (1<<ctr);
    dst->ra_rank = 1;
    /* Pentium 4 requires that both an escr and a counter are selected.
       Find which counter mask contains this counter and set its escr */
    if (dst->ra_bits.counter[0] & dst->ra_selector)
	dst->ra_escr[dst->ra_bits.escr[0] >> 5] = (1 << (dst->ra_bits.escr[0] & 31));
    else
	dst->ra_escr[dst->ra_bits.escr[0] >> 5] = (1 << (dst->ra_bits.escr[1] & 31));
}

/* This function examines the event to determine
    if it has a single exclusive mapping. 
    Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t *dst)
{
    return(dst->ra_rank==1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
  /* Pentium 4 needs to check for conflict of both counters and esc registers */
    return((dst->ra_selector & src->ra_selector) 
      || (dst->ra_escr[0] & src->ra_escr[0]) || (dst->ra_escr[1] & src->ra_escr[1]));
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
    int i, j;
    unsigned shared;
    
    /* On Pentium 4, shared resources include both escrs and counters */
#ifdef DEBUG
	  SUBDBG("src, dst\n");
	  print_alloc(src);
	  print_alloc(dst);
#endif

    /* remove counters referenced by any shared escrs */
    for (i=0;i<2;i++) {
      shared = dst->ra_escr[i] & src->ra_escr[i];
      while (shared) {
	  j = ffs(shared) - 1;
	  shared ^= 1<<j;
	  if (dst->ra_bits.escr[0] == j)
	      dst->ra_selector ^= dst->ra_bits.counter[0];
	  else
	      dst->ra_selector ^= dst->ra_bits.counter[1];
	  dst->ra_escr[i] ^= 1<<j;
      }
    }
#ifdef DEBUG
	  SUBDBG("new dst\n");
	  print_alloc(dst);
#endif
    /* remove any remaining shared counters */
    shared = dst->ra_selector & src->ra_selector;
    if (shared) dst->ra_selector ^= shared;
#ifdef DEBUG
	  SUBDBG("new dst\n");
	  print_alloc(dst);
#endif

    /* recompute rank */
    for (i=0,dst->ra_rank=0;i<MAX_COUNTERS;i++)
      if(dst->ra_selector & (1<<i)) dst->ra_rank++;
#ifdef DEBUG
	  SUBDBG("new dst\n");
	  print_alloc(dst);
#endif
}

/* This function updates the selection status of 
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
    dst->ra_selector = src->ra_selector;
    dst->ra_escr[0] = src->ra_escr[0];
    dst->ra_escr[1] = src->ra_escr[1];
}


/* Register allocation */

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI)
{
  int i, j, natNum;
  P4_reg_alloc_t event_list[MAX_COUNTERS], *e;

  /* not yet successfully mapped, but have enough slots for events */
	
  /* Initialize the local structure needed 
     for counter allocation and optimization. */
  natNum=ESI->NativeCount;
  SUBDBG("native event count: %d\n",natNum);
  for(i=0;i<natNum;i++){
    /* dereference event_list so code is easier to read */
    e = &event_list[i];

    /* retrieve the mapping information about this native event */
    _papi_hwd_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_index, &e->ra_bits);

    /* combine counter bit masks for both esc registers into selector */
    e->ra_selector = e->ra_bits.counter[0] | e->ra_bits.counter[1];
    /* calculate native event rank, which is number of counters it can live on */
    e->ra_rank = 0;
    for(j=0;j<MAX_COUNTERS;j++) {
      if(e->ra_selector & (1<<j)) {
	e->ra_rank++;
      }
    }
    /* set the bits for the two esc registers this event can live on */
    e->ra_escr[0] = e->ra_escr[1] = 0;
    for(j=0;j<2;j++) {
      e->ra_escr[e->ra_bits.escr[j] >> 5] |= (1 << (e->ra_bits.escr[j] & 31));
    }
#ifdef DEBUG
    SUBDBG("i: %d\n",i);
    print_alloc(e);
#endif
  }

  if(_papi_hwi_bipartite_alloc(event_list, natNum)){ /* successfully mapped */
    for(i=0;i<natNum;i++) {
#ifdef DEBUG
	  SUBDBG("i: %d\n",i);
	  print_alloc(&event_list[i]);
#endif
	  /* Copy all the info about this native event to the NativeInfo struct */
	  ESI->NativeInfoArray[i].ni_bits = event_list[i].ra_bits;

	  /* The selector contains the counter bit position. Turn it into a number
	     and store it in the first counter value, zeroing the second. */
	  ESI->NativeInfoArray[i].ni_bits.counter[0] = ffs(event_list[i].ra_selector)-1;
	  ESI->NativeInfoArray[i].ni_bits.counter[1] = 0;

	  /* Array order on perfctr is event ADD order, not counter #... */
	  ESI->NativeInfoArray[i].ni_position = i;
      }
      return 1;
  }

  return(PAPI_OK);
}


static void clear_control_state(hwd_control_state_t *this_state)
{
  int i;

  /* Remove all counter control command values from eventset. */
  
  for (i=0;i<this_state->control.cpu_control.nractrs;i++)
    {
      SUBDBG("Clearing pmc event entry %d\n",i);
      this_state->control.cpu_control.pmc_map[i] = 0;
      this_state->control.cpu_control.evntsel[i] = 0;
#ifdef __i386__
      this_state->control.cpu_control.evntsel_aux[i] = 0;
#endif
      this_state->control.cpu_control.ireset[i] = 0;
    }

  /* Clear pebs stuff */

#ifdef __i386__
    this_state->control.cpu_control.p4.pebs_enable = 0;
    this_state->control.cpu_control.p4.pebs_matrix_vert = 0;
#endif
	  
  this_state->control.cpu_control.nractrs = 0;
  
#ifdef DEBUG
  print_control(&this_state->control.cpu_control);
#endif
}


/* This function clears the current contents of the control structure and updates it 
   with whatever resources are allocated for all the native events 
   in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t *this_state, NativeInfo_t *native, int count)
{
    int i, nractrs;

    P4_register_t *bits;
   
    /* clear out everything currently coded */
    clear_control_state(this_state);

    /* fill the counters we're using */
    nractrs = this_state->control.cpu_control.nractrs;
    for(i=0;i<count;i++){
	/* dereference the mapping information about this native event */
	bits = &native[i].ni_bits;

	/* Add counter control command values to eventset */

	this_state->control.cpu_control.pmc_map[nractrs] = bits->counter[0];
        this_state->control.cpu_control.evntsel[nractrs] = bits->cccr;
	this_state->control.cpu_control.ireset[nractrs] = bits->ireset;
#ifdef __x86_64__
	/* This sets Enable, USR-mode and SYS-mode. The latter two should really 
	   be taken from the current event set scope though */
	this_state->control.cpu_control.evntsel[nractrs] |= (1<<22)|(1<<16)|(1<<17);
#endif
#ifdef __i386__
        this_state->control.cpu_control.pmc_map[nractrs] |= FAST_RDPMC;
	this_state->control.cpu_control.evntsel_aux[nractrs] = bits->event;
	/* What happens if more than one native event has pebs_enable or pebs_matrix_vert?
	   Are these just binary enables or can they actually have conflicting values? */
	if (bits->pebs_enable)
	  this_state->control.cpu_control.p4.pebs_enable = bits->pebs_enable;
	if (bits->pebs_matrix_vert)
	  this_state->control.cpu_control.p4.pebs_matrix_vert = bits->pebs_matrix_vert;
#endif
	nractrs++;
    }
    this_state->control.cpu_control.nractrs = nractrs;
  
    /* Make sure the TSC is always on */
    this_state->control.cpu_control.tsc_on = 1;

#ifdef DEBUG
    print_control(&this_state->control.cpu_control);
#endif
    return(PAPI_OK);
}


int _papi_hwd_add_prog_event(P4_perfctr_control_t *state, unsigned int code, void *tmp, 
			      EventInfo_t *tmp2)
{
  return(PAPI_ESBSTR);
}


int _papi_hwd_set_domain(P4_perfctr_control_t *cntrl, int domain)
{
  return(PAPI_ESBSTR);
}

#ifdef __x86_64__
#include <linux/spinlock.h>
spinlock_t lock[PAPI_MAX_LOCK];
#else
volatile unsigned int lock[PAPI_MAX_LOCK] = {0,};
#endif
/* volatile uint32_t lock; */
                                                                                
#include <inttypes.h>
                                                                                
void _papi_hwd_lock_init()
{
  int lck;
  for (lck=0;lck<PAPI_MAX_LOCK;lck++)
#ifdef __x86_64__
    spin_lock_init(&lock[lck]);
#else
    lock[lck] = MUTEX_OPEN;
#endif
}
                                                                                
                                                                                
int _papi_hwd_reset(P4_perfctr_context_t *ctx, P4_perfctr_control_t *cntrl)
{
  /* this is what I gleaned from PAPI 2.3.4... is it right??? dkt */
  return(_papi_hwd_start(ctx, cntrl));
}


int _papi_hwd_write(P4_perfctr_context_t *ctx, P4_perfctr_control_t *cntrl, long long *from)
{
  return(PAPI_ESBSTR);
}


int _papi_hwd_set_profile(EventSetInfo_t *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}


int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}


static void swap_pmc_map_events(struct vperfctr_control *contr,int cntr1,int cntr2)
{
  unsigned int ui; int si;

  /* In the case a user wants to interrupt on a counter in an evntsel
     that is not among the last events, we need to move the perfctr 
     virtual events around to make it last. This function swaps two
     perfctr events */

  ui=contr->cpu_control.pmc_map[cntr1];
  contr->cpu_control.pmc_map[cntr1]=contr->cpu_control.pmc_map[cntr2];
  contr->cpu_control.pmc_map[cntr2] = ui;

  ui=contr->cpu_control.evntsel[cntr1];
  contr->cpu_control.evntsel[cntr1]=contr->cpu_control.evntsel[cntr2];
  contr->cpu_control.evntsel[cntr2] = ui;
#ifdef __i386__
  ui=contr->cpu_control.evntsel_aux[cntr1];
  contr->cpu_control.evntsel_aux[cntr1]=contr->cpu_control.evntsel_aux[cntr2];
  contr->cpu_control.evntsel_aux[cntr2] = ui;
#endif
  si=contr->cpu_control.ireset[cntr1];
  contr->cpu_control.ireset[cntr1]=contr->cpu_control.ireset[cntr2];
  contr->cpu_control.ireset[cntr2] = si;
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, EventSetOverflowInfo_t *overflow_option)
{
#ifdef __i386__
  const int PERF_INT_ENABLE = CCCR_OVF_PMI_T0;
#elif defined(__x86_64__)
  const int PERF_INT_ENABLE = (1<<20);
#endif
  /* | CCCR_OVF_PMI_T1 (1 << 27) */

  extern int _papi_hwi_using_signal;
  hwd_control_state_t *this_state = &ESI->machdep;
  struct vperfctr_control *contr = &this_state->control;
  int i, ncntrs, nricntrs = 0, nracntrs, retval=0;

  SUBDBG("overflow_option->EventIndex=%d\n",overflow_option->EventIndex);
  if( overflow_option->threshold != 0)  /* Set an overflow threshold */
    {
      struct sigaction sa;
      int err;

      if (ESI->EventInfoArray[overflow_option->EventIndex].derived)
	{
	  fprintf(stderr,"Can't overflow on a derived event.\n");
	  return PAPI_EINVAL;
	}

      /* The correct event to overflow is overflow_option->EventIndex */

      ncntrs = _papi_hwi_system_info.num_cntrs;
      i = ESI->EventInfoArray[overflow_option->EventIndex].pos[0];
      if (i >= ncntrs)
	{
	  fprintf(stderr,"Selector id (%d) larger than ncntrs (%d)\n",i,ncntrs);
	  return PAPI_EINVAL;
	}
/* Neither of these conditions should hold...
      if (contr->cpu_control.nrictrs)
	{
	  fprintf(stderr,"Only one interrupting counter in event set.\n");
	  return PAPI_EINVAL;
	}
      if (contr->cpu_control.nractrs != 1)
	{
	  fprintf(stderr,"Must have only one counter in event set.\n");
	  return PAPI_EINVAL;
	}
*/
      contr->cpu_control.ireset[i] = -overflow_option->threshold;
      contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;
      nricntrs = ++contr->cpu_control.nrictrs;
      nracntrs = --contr->cpu_control.nractrs;
      contr->si_signo = PAPI_SIGNAL;

      /* perfctr 2.x requires the interrupting counters to be placed last
	 in evntsel, swap events that do not fulfill this criterion. This
	 will yield a non-monotonic pmc_map array */

#if 0
      if (ESI->EventInfoArray[i].event_code == PAPI_FP_INS) {
	swap_pmc_map_events(contr,0,1);
	SUBDBG("Swapped events\n");
      }
#endif

      memset(&sa, 0, sizeof sa);
      sa.sa_sigaction = _papi_hwd_dispatch_timer;
      sa.sa_flags = SA_SIGINFO;
      if((err = sigaction(PAPI_SIGNAL, &sa, NULL)) < 0)
	{
	  SUBDBG("Setting sigaction failed: SYSERR %d: %s",errno,strerror(errno));
	  return(PAPI_ESYS);
	}

      _papi_hwd_lock(PAPI_INTERNAL_LOCK);
      _papi_hwi_using_signal++;
      _papi_hwd_unlock(PAPI_INTERNAL_LOCK);

      SUBDBG("Modified event set\n");
    }
  else   
    {
      /* The correct event to overflow is overflow_option->EventIndex */
      ncntrs=_papi_hwi_system_info.num_cntrs;
      for(i=0;i<ncntrs;i++) 
	if(contr->cpu_control.evntsel[i] & PERF_INT_ENABLE)
	  {
	    contr->cpu_control.ireset[i] = 0;
	    contr->cpu_control.evntsel[i] &= (~PERF_INT_ENABLE);
	    nricntrs=--contr->cpu_control.nrictrs;
	    nracntrs=++contr->cpu_control.nractrs;
	    contr->si_signo = 0;
	  }

      /* perfctr 2.x requires the interrupting counters to be placed last
	 in evntsel, swap events that do not fulfill this criterion. This
	 will yield a non-monotonic pmc_map array */

#if 0
      if (ESI->EventInfoArray[i].event_code == PAPI_FP_INS)
	swap_pmc_map_events(contr,1,0);	
#endif

      SUBDBG("Modified event set\n");

      _papi_hwd_lock(PAPI_INTERNAL_LOCK);
      _papi_hwi_using_signal--;
      if (_papi_hwi_using_signal == 0)
	{
	  if (sigaction(PAPI_SIGNAL, NULL, NULL) == -1)
	    retval = PAPI_ESYS;
	}
      _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
    }

  SUBDBG("%s (%s): Hardware overflow is still experimental.\n",
	  __FILE__,__FUNCTION__);
  SUBDBG("End of call. Exit code: %d\n",retval);
  return(retval);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t *info, void *tmp)
{
  ucontext_t *uc;
  mcontext_t *mc;
  gregset_t *gs;

  uc = (ucontext_t *) tmp;
  mc = &uc->uc_mcontext;
  gs = &mc->gregs;

  DBG((stderr,"Start at 0x%lx\n",(unsigned long)(*gs)[15]));
  _papi_hwi_dispatch_overflow_signal(mc); 

  /* We are done, resume interrupting counters */

  if(_papi_hwi_system_info.supports_hw_overflow)
    {
      ThreadInfo_t *master;

      master = _papi_hwi_lookup_in_thread_list();
      if(master==NULL)
	{
	  fprintf(stderr,"%s():%d: master event lookup failure! abort()\n",
		  __FUNCTION__,__LINE__);
	  abort();
	}
      if (vperfctr_iresume(master->context.perfctr) < 0)
	{
	  fprintf(stderr,"%s():%d: vperfctr_iresume %s\n",
		  __FUNCTION__,__LINE__,strerror(errno));
	}
    }
  DBG((stderr,"Finished, returning to address 0x%lx\n",(unsigned long)(*gs)[15]));
}


void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
#ifdef __x86_64__
  location = (void *)info->rip;
#else
  location = (void *)info->eip;
#endif

  return(location);
}

