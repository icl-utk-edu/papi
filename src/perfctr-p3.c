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

extern preset_search_t _papi_hwd_p3_preset_map;
extern preset_search_t _papi_hwd_amd_preset_map;
extern preset_search_t *preset_search_map;
extern native_event_entry_t _papi_hwd_pentium3_native_map;
extern native_event_entry_t _papi_hwd_p2_native_map;
extern native_event_entry_t _papi_hwd_k7_native_map;
extern native_event_entry_t *native_table;
extern hwi_preset_t _papi_hwd_preset_map[];
extern papi_mdi_t _papi_hwi_system_info;
volatile unsigned int lock[PAPI_MAX_LOCK] = {0,};

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
        if( control->evntsel_aux[i] )
            SUBDBG("evntsel_aux[%u]\t0x%08X\n", i, control->evntsel_aux[i]);
        if (control->ireset[i])
          SUBDBG("ireset[%u]\t%d\n",i,control->ireset[i]);
    }
}
#endif

inline static int setup_p3_presets(int cputype) {
   switch(_papi_hwi_system_info.hw_info.model)
    {
    case PERFCTR_X86_GENERIC:
    case PERFCTR_X86_CYRIX_MII:
    case PERFCTR_X86_WINCHIP_C6:
    case PERFCTR_X86_WINCHIP_2:
    case PERFCTR_X86_VIA_C3:
    case PERFCTR_X86_INTEL_P5:
    case PERFCTR_X86_INTEL_P5MMX:
    case PERFCTR_X86_INTEL_PII:
      native_table = &_papi_hwd_p2_native_map;
      preset_search_map = &_papi_hwd_p3_preset_map;
      break;
    case PERFCTR_X86_INTEL_P6:
    case PERFCTR_X86_INTEL_PIII:
      native_table = &_papi_hwd_pentium3_native_map;
      preset_search_map = &_papi_hwd_p3_preset_map;
      break;
    case PERFCTR_X86_AMD_K7:
      native_table = &_papi_hwd_k7_native_map;
      preset_search_map = &_papi_hwd_amd_preset_map;
      break;
    }
   return(_papi_hwi_setup_all_presets(preset_search_map));
}

/* Low level functions, should not handle errors, just return codes. */

static inline u_long_long get_cycles (void)
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

/* Dumb hack to make sure I get the cycle time correct. */

static float calc_mhz(void)
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

void _papi_hwd_init_control_state(hwd_control_state_t *ptr)
{
  int def_mode, i;

  switch (_papi_hwi_system_info.default_domain)
    {
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
  switch(_papi_hwi_system_info.hw_info.model)
    {
    case PERFCTR_X86_GENERIC:
    case PERFCTR_X86_CYRIX_MII:
    case PERFCTR_X86_WINCHIP_C6:
    case PERFCTR_X86_WINCHIP_2:
    case PERFCTR_X86_VIA_C3:
    default:
      ptr->control.cpu_control.tsc_on=1;
      ptr->control.cpu_control.nractrs=0;
      ptr->control.cpu_control.nrictrs=0;
      break;
    case PERFCTR_X86_INTEL_P5:
    case PERFCTR_X86_INTEL_P5MMX:
    case PERFCTR_X86_INTEL_PII:
    case PERFCTR_X86_INTEL_P6:
    case PERFCTR_X86_INTEL_PIII:
      ptr->control.cpu_control.evntsel[0] |= def_mode | PERF_ENABLE;
      ptr->control.cpu_control.evntsel[1] |= def_mode;
      ptr->control.cpu_control.tsc_on=1;
      ptr->control.cpu_control.nractrs=_papi_hwi_system_info.num_cntrs;
      ptr->control.cpu_control.nrictrs=0;
      break;
    case PERFCTR_X86_AMD_K7:
      ptr->control.cpu_control.evntsel[0] |= def_mode | PERF_ENABLE;
      ptr->control.cpu_control.evntsel[1] |= def_mode | PERF_ENABLE;
      ptr->control.cpu_control.evntsel[2] |= def_mode | PERF_ENABLE;
      ptr->control.cpu_control.evntsel[3] |= def_mode | PERF_ENABLE;
      ptr->control.cpu_control.tsc_on=1;
      ptr->control.cpu_control.nractrs=_papi_hwi_system_info.num_cntrs;
      ptr->control.cpu_control.nrictrs=0;
      break;
    }
  /* Identity counter map for starters */
  for(i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    ptr->control.cpu_control.pmc_map[i]=i;
}

#ifdef DEBUG
static void dump_cmd(char *str, struct vperfctr_control *t)
{
  int i,k;

  DBG((stderr,"%s: tsc_on=0x%x  nractrs=0x%x, nrictrs=0x%x\n",str,t->cpu_control.tsc_on,t->cpu_control.nractrs,t->cpu_control.nrictrs));
  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    {
      k=t->cpu_control.pmc_map[i];
      DBG((stderr,"Item %d [map %d]: Evntsel=0x%08x   (ireset=%d)\n",i,k,t->cpu_control.evntsel[i],t->cpu_control.ireset[i]));
    }
}
#endif

int _papi_hwd_add_prog_event(hwd_control_state_t *state, unsigned int code, void *tmp, EventInfo_t *tmp2)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_set_domain(hwd_control_state_t *cntrl, int domain)
{
  return(PAPI_ESBSTR);
}

void _papi_hwd_lock_init(void) {
   int i;
   for(i = 0; i < PAPI_MAX_LOCK; i++) {
      lock[i] = MUTEX_OPEN;
   }
}

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

int _papi_hwd_init_global(void)
{
   int retval;
   struct perfctr_info info;
   float mhz;
   struct vperfctr *dev;

   /* Opened once for all threads. */
   if((dev = vperfctr_open()) == NULL)
      error_return(PAPI_ESYS,VOPEN_ERROR);
   SUBDBG("_papi_hwd_init_global vperfctr_open = %p\n",dev);

   /* Get info from the kernel */
   if(vperfctr_info(dev, &info) < 0)
      error_return(PAPI_ESYS,VINFO_ERROR);

   /* Initialize outstanding values in machine info structure */
   if(_papi_hwd_mdi_init() != PAPI_OK) {
      return(PAPI_EINVAL);
   }
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
   if(abs(mhz-_papi_hwi_system_info.hw_info.mhz) > 0.95*_papi_hwi_system_info.hw_info.mhz)
      _papi_hwi_system_info.hw_info.mhz = mhz;
   SUBDBG("Actual MHZ is %f\n",_papi_hwi_system_info.hw_info.mhz);

   /* Setup presets */
   retval = setup_p3_presets(info.cpu_type);
   if(retval)
      return(retval);

   /* Fill in what we can of the papi_system_info. */
   retval = _papi_hwd_get_system_info();
   if(retval != PAPI_OK)
      return(retval);

   /* Setup memory info */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.mem_info, (int)info.cpu_type);
   if(retval)
      return(retval);

   vperfctr_close(dev);
   SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n",dev);

   return(PAPI_OK);
}

int _papi_hwd_init(hwd_context_t *ctx) {
   struct vperfctr_control tmp;

   /* Initialize our thread/process pointer. */
   if((ctx->perfctr = vperfctr_open()) == NULL)
      error_return(PAPI_ESYS,VOPEN_ERROR);
   SUBDBG("_papi_hwd_init vperfctr_open() = %p\n",ctx->perfctr);

   /* Initialize the per thread/process virtualized TSC */
   memset(&tmp,0x0,sizeof(tmp));
   tmp.cpu_control.tsc_on = 1;

   /* Start the per thread/process virtualized TSC */
   if(vperfctr_control(ctx->perfctr, &tmp) < 0)
      error_return(PAPI_ESYS,VCNTRL_ERROR);

   return(PAPI_OK);
}

u_long_long _papi_hwd_get_real_usec(void)
{
  return((u_long_long)get_cycles() / (u_long_long)_papi_hwi_system_info.hw_info.mhz);
}

u_long_long _papi_hwd_get_real_cycles(void)
{
  return(get_cycles());
}

u_long_long _papi_hwd_get_virt_usec(const hwd_context_t *ctx)
{
  return(_papi_hwd_get_virt_cycles(ctx) / (u_long_long)_papi_hwi_system_info.hw_info.mhz);
}

u_long_long _papi_hwd_get_virt_cycles(const hwd_context_t *ctx)
{
  return(vperfctr_read_tsc(ctx->perfctr));
}

/* This function examines the event to determine
    if it can be mapped to counter ctr.
    Returns true if it can, false if it can't.
*/
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr)
{
   return(dst->ra_selector & (1<<ctr));
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr)
{
   dst->ra_selector = (1<<ctr);
   dst->ra_rank = 1;
}

/* This function examines the event to determine
   if it has a single exclusive mapping.
   Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t *dst)
{
printf("\ndst->rank = %d\n", dst->ra_rank);
   return(dst->ra_rank==1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
   return(dst->ra_selector & src->ra_selector);
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) {
 //  int i;
 //  unsigned shared;

 //  shared = dst->ra_selector & src->ra_selector;
 //  if(shared) dst->ra_selector ^= shared;
 //  for(i = 0, dst->ra_rank = 0; i < MAX_COUNTERS; i++)
 //     if(dst->ra_selector & (1<<i)) dst->ra_rank++;
   dst->ra_selector ^= src->ra_selector;
   dst->ra_rank -= src->ra_rank;
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
   dst->ra_selector = src->ra_selector;
}

/* Register allocation */

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
   int index, i, j, natNum;
   hwd_reg_alloc_t event_list[MAX_COUNTERS];

   /* Initialize the local structure needed
      for counter allocation and optimization. */
   natNum=ESI->NativeCount;
   for(i = 0; i < natNum; i++){
      index=ESI->NativeInfoArray[i].ni_index;
      event_list[i].ra_selector = native_table[index].resources.selector; 
printf("native selector = %d\n", native_table[ESI->NativeInfoArray[i].ni_index].resources.selector);
printf("index = %d\n", ESI->NativeInfoArray[i].ni_index);
printf("selector = %d\n", event_list[i].ra_bits.selector);
      /* calculate native event rank, which is no. of counters it can live on */
      event_list[i].ra_rank = 0;
      for(j=0;j<MAX_COUNTERS;j++) {
         if(event_list[i].ra_selector & (1<<j)) {
            event_list[i].ra_rank++;
         }
      }
   }
   if(_papi_hwi_bipartite_alloc(event_list, natNum)){ /* successfully mapped */ 
      for(i = 0; i < natNum; i++) {
         /* Copy all info about this native event to the NativeInfo struct */
         ESI->NativeInfoArray[i].ni_bits = event_list[i].ra_bits;

         /* The selector contains the counter bit position. */
//         ESI->NativeInfoArray[i].ni_bits.selector = event_list[i].ra_selector;
         /* Array order on perfctr is event ADD order, not counter #... */
         ESI->NativeInfoArray[i].ni_position=ffs(event_list[i].ra_selector)-1;
      }
      return 1;
   }
   else return 0;
}

static void clear_control_state(hwd_control_state_t *this_state) {
   int i;

   /* Remove all counter control command values from eventset. */
   for(i=0;i<this_state->control.cpu_control.nractrs;i++) {
      SUBDBG("Clearing pmc event entry %d\n",i);
      this_state->control.cpu_control.pmc_map[i] = 0;
      this_state->control.cpu_control.evntsel[i] = 0;
#ifdef __i386__
      this_state->control.cpu_control.evntsel_aux[i] = 0;
#endif
      this_state->control.cpu_control.ireset[i] = 0;
   }
   this_state->control.cpu_control.nractrs = 0;
}

/* This function clears the current contents of the control structure and 
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t *this_state, NativeInfo_t *native, int count) {
   int i;
   hwd_register_t *bits;

   /* clear out everything currently coded */
   clear_control_state(this_state);

   /* fill the counters we're using */
print_control(&this_state->control.cpu_control);
   for(i = 0; i < count; i++){
      /* dereference the mapping information about this native event */
      bits = &native[i].ni_bits;
      /* Add counter control command values to eventset */
      this_state->control.cpu_control.pmc_map[bits->selector] = i;
      this_state->control.cpu_control.evntsel[bits->selector] = bits->counter_cmd[0];
   }
   this_state->control.cpu_control.nractrs = count;

   /* Make sure the TSC is always on */
   this_state->control.cpu_control.tsc_on = 1;

print_control(&this_state->control.cpu_control);

   return(PAPI_OK);
}

int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *state)
{
  int error;
  if((error = vperfctr_control(ctx->perfctr, &state->control)) < 0) {
    SUBDBG("vperfctr_control returns: %d\n",error);
    error_return(PAPI_ESYS,VCNTRL_ERROR);
  }
  return(PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state)
{
  if(vperfctr_stop(ctx->perfctr) < 0)
    error_return(PAPI_ESYS,VCNTRL_ERROR);

  return(PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl)
{
  return(_papi_hwd_start(ctx, cntrl));
}

int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *spc, long_long **dp)
{
   vperfctr_read_ctrs(ctx->perfctr, &spc->state);
   *dp = (long_long*) spc->state.pmc;
#ifdef DEBUG
   {
      extern int _papi_hwi_debug;
      if(_papi_hwi_debug) {
         int i;
         for(i=0;i<spc->control.cpu_control.nractrs;i++) {
           SUBDBG("raw val hardware index %d is %lld\n",i,(long_long) spc->state.pmc[i]);
         }
      }
   }
#endif
   return(PAPI_OK);
}

int _papi_hwd_setmaxmem(){
  return(PAPI_OK);
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *cntrl, long_long *from)
{ 
  return(PAPI_ESBSTR);
}

/* Called once per process. */

int _papi_hwd_shutdown_global(void)
{
  return(PAPI_OK);
}

/* This routine is for shutting down threads, including the
   master thread. */

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
  int retval = vperfctr_unlink(ctx->perfctr);
  SUBDBG("_papi_hwd_init_global vperfctr_unlink(%p) = %d\n",ctx->perfctr,retval);
  vperfctr_close(ctx->perfctr);
  SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n",ctx->perfctr);
  memset(ctx,0x0,sizeof(hwd_context_t));

  if(retval)
     return(PAPI_ESYS);
  return(PAPI_OK);
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

  ui=contr->cpu_control.evntsel_aux[cntr1];
  contr->cpu_control.evntsel_aux[cntr1]=contr->cpu_control.evntsel_aux[cntr2];
  contr->cpu_control.evntsel_aux[cntr2] = ui;

  si=contr->cpu_control.ireset[cntr1];
  contr->cpu_control.ireset[cntr1]=contr->cpu_control.ireset[cntr2];
  contr->cpu_control.ireset[cntr2] = si;
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, EventSetOverflowInfo_t *overflow_option)
{
#ifdef PAPI_PERFCTR_INTR_SUPPORT
  extern int _papi_hwi_using_signal;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  struct vperfctr_control *contr = &this_state->counter_cmd;
  int i, ncntrs, nricntrs = 0, nracntrs, cntr, cntr2, retval=0;
  unsigned int selector;

#ifdef DEBUG
  DBG((stderr,"overflow_option->EventIndex=%d\n",
       overflow_option->EventIndex));
  dump_cmd("_papi_hwd_set_overflow",contr);
#endif 
  if( overflow_option->threshold != 0)  /* Set an overflow threshold */
    {
      struct sigaction sa;
      int err;

      /* Return error if installed signal is set earlier (!=SIG_DFL) and
	 it was not set to the PAPI overflow handler */
      /* The following code is commented out because many C libraries
	 replace the signal handler when one links with threads. The
	 name of this signal handler is not exported. So there really
	 is NO WAY to check if the user has installed a signal. */
      /*
      void *tmp;
      tmp = (void *)signal(PAPI_SIGNAL, SIG_IGN);
      if ((tmp != (void *)SIG_DFL) && (tmp != (void *)_papi_hwd_dispatch_timer))
	return(PAPI_EMISC);
      */

      memset(&sa, 0, sizeof sa);
      sa.sa_sigaction = _papi_hwd_dispatch_timer;
      sa.sa_flags = SA_SIGINFO;
      if((err = sigaction(PAPI_SIGNAL, &sa, NULL)) < 0)
	{
	  DBG((stderr,"Setting sigaction failed: SYSERR %d: %s",errno,strerror(errno)));
	  return(PAPI_ESYS);
	}

      /* The correct event to overflow is overflow_option->EventIndex */
      ncntrs=_papi_hwi_system_info.num_cntrs;
      selector = ESI->EventInfoArray[overflow_option->EventIndex].selector;
      DBG((stderr,"selector id is %d.\n",selector));
      i=ffs(selector)-1;
      if(i>=ncntrs)
	{
	  DBG((stderr,"Selector id (0x%x) larger than ncntrs (%d)\n",selector,ncntrs));
	  return PAPI_EINVAL;
	}
      contr->cpu_control.ireset[i] = -overflow_option->threshold;
      contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;
      nricntrs=++contr->cpu_control.nrictrs;
      nracntrs=--contr->cpu_control.nractrs;
      contr->si_signo = PAPI_SIGNAL;

      /* perfctr 2.x requires the interrupting counters to be placed last
	 in evntsel, swap events that do not fulfill this criterion. This
	 will yield a non-monotonic pmc_map array */
      for(i=nricntrs;i>0;i--)
	{
	  cntr = nracntrs + i - 1;
	  if( !(contr->cpu_control.evntsel[cntr] & PERF_INT_ENABLE))
	    { /* A non-interrupting counter was found among the icounters
		 Locate an interrupting counter in the acounters and swap */
	      for(cntr2=0;cntr2<nracntrs;cntr2++)
		{
		  if( (contr->cpu_control.evntsel[cntr2] & PERF_INT_ENABLE))
		    break;
		}
	      if(cntr2==nracntrs)
		{
		  DBG((stderr,"No icounter to swap with!\n"));
		  return(PAPI_EMISC);
		}
	      swap_pmc_map_events(contr,cntr,cntr2);
	    }
	}

      PAPI_lock();
      _papi_hwi_using_signal++;
      PAPI_unlock();

#ifdef DEBUG
      DBG((stderr,"Modified event set\n"));
      dump_cmd("_papi_hwd_set_overflow",contr);
#endif 
    }
  else   /* Disable overflow */
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
      /* The current implementation only supports one interrupting counter */
      if(nricntrs)
	{
	  fprintf(stderr,"%s %s\n","PAPI internal error.",
		  "Only one interrupting counter is supported!");
	  return(PAPI_ESBSTR);
	}

      /* perfctr 2.x requires the interrupting counters to be placed last
	 in evntsel, when the counter is non-interupting, move the order
	 back into the default monotonic pmc_map */
      for(cntr=0;cntr<ncntrs;cntr++)
	if(contr->cpu_control.pmc_map[cntr]!=cntr)
	  { /* This counter is out-of-order. Swap with the correct one*/
	    for(cntr2=cntr+1;cntr2<ncntrs;cntr2++)
	      if(contr->cpu_control.pmc_map[cntr2]==cntr) break;
	    if(cntr2==ncntrs)
	      {
		DBG((stderr,"No icounter to swap with!\n"));
		return(PAPI_EMISC);
	      }
	    swap_pmc_map_events(contr,cntr,cntr2);
	  }

#ifdef DEBUG
      DBG((stderr,"Modified event set\n"));
      dump_cmd(__FUNCTION__,contr);
#endif 

      PAPI_lock();
      _papi_hwi_using_signal--;
      if (_papi_hwi_using_signal == 0)
	{
	  if (sigaction(PAPI_SIGNAL, NULL, NULL) == -1)
	    retval = PAPI_ESYS;
	}
      PAPI_unlock();
    }

  DBG((stderr,"%s (%s): Hardware overflow is still experimental.\n",
	  __FILE__,__FUNCTION__));
  DBG((stderr,"End of call. Exit code: %d\n",retval));
  return(retval);
#else
  /* This function is not used and shouldn't be called. */
  return(PAPI_ESBSTR);
#endif
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
