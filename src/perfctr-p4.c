/* 
* File:    p4.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London 
*          london@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include "papi_internal.h"

#if defined(PERFCTR26)
#define PERFCTR_CPU_NAME(pi)    perfctr_info_cpu_name(pi)
#define PERFCTR_CPU_NRCTRS(pi)  perfctr_info_nrctrs(pi)
#define evntsel_aux             p4.escr
#elif defined(PERFCTR25)
#define PERFCTR_CPU_NAME	perfctr_info_cpu_name
#define PERFCTR_CPU_NRCTRS	perfctr_info_nrctrs
#define evntsel_aux		p4.escr
#else
#define PERFCTR_CPU_NAME	perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS	perfctr_cpu_nrctrs
#endif

/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

extern hwi_search_t _papi_hwd_pentium4_base_preset_map[];
extern hwi_search_t _papi_hwd_pentium4_tot_iis_preset_map[];
extern hwi_search_t _papi_hwd_pentium4_L3_cache_map[];
extern hwi_dev_notes_t _papi_hwd_pentium4_base_dev_notes[];

/*****************************/
/* END EXTERNAL DECLARATIONS */
/*****************************/

/****************************/
/* BEGIN LOCAL DECLARATIONS */
/****************************/

volatile unsigned int lock[PAPI_MAX_LOCK];

/**************************/
/* END LOCAL DECLARATIONS */
/**************************/

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

inline static int setup_p4_presets(int cputype)
{
   hwi_search_t *s;
   hwi_dev_notes_t *n;
   extern void _papi_hwd_fixup_fp(hwi_search_t **s, hwi_dev_notes_t **n);
   extern void _papi_hwd_fixup_vec(hwi_search_t **s, hwi_dev_notes_t **n);

   /* load the baseline event map for all Pentium 4s */
   _papi_hwi_setup_all_presets(_papi_hwd_pentium4_base_preset_map, _papi_hwd_pentium4_base_dev_notes);

   /* fix up the floating point and vector ops */
   _papi_hwd_fixup_fp(&s, &n);
   _papi_hwi_setup_all_presets(s,n);
   _papi_hwd_fixup_vec(&s, &n);
   _papi_hwi_setup_all_presets(s,n);

   /* install L3 cache events iff 3 levels of cache exist */
   if (_papi_hwi_system_info.hw_info.mem_hierarchy.levels == 3)
      _papi_hwi_setup_all_presets(_papi_hwd_pentium4_L3_cache_map, NULL);

   /* overload with any model dependent events */
   if (cputype == PERFCTR_X86_INTEL_P4) {
     /* do nothing besides the base map */
   }
   else if (cputype == PERFCTR_X86_INTEL_P4M2) {
      _papi_hwi_setup_all_presets(_papi_hwd_pentium4_tot_iis_preset_map, NULL);
   }
#ifdef PERFCTR_X86_INTEL_P4M3
   else if (cputype == PERFCTR_X86_INTEL_P4M3) {
      _papi_hwi_setup_all_presets(_papi_hwd_pentium4_tot_iis_preset_map, NULL);
   }
#endif
   else {
      PAPIERROR(MODEL_ERROR);
      return(PAPI_ESBSTR);
   }
   return (PAPI_OK);
}

/* This used to be init_config, static to the substrate.
   Now its exposed to the hwi layer and called when an EventSet is allocated.
*/
void _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   int def_mode, i;
   switch(_papi_hwi_system_info.default_domain) {
   case PAPI_DOM_USER:
      def_mode = ESCR_T0_USR;
      break;
   case PAPI_DOM_KERNEL:
      def_mode = ESCR_T0_OS;
      break;
   case PAPI_DOM_ALL:
      def_mode = ESCR_T0_OS | ESCR_T0_USR;
      break;
   default:
      PAPIERROR("BUG! Unknown domain %d, using PAPI_DOM_USER",_papi_hwi_system_info.default_domain);
      def_mode = ESCR_T0_USR;
      break;
   }
   for(i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
      ptr->control.cpu_control.evntsel_aux[i] |= def_mode;
   }
   ptr->control.cpu_control.tsc_on = 1;
   ptr->control.cpu_control.nractrs = 0;
   ptr->control.cpu_control.nrictrs = 0;
#if 0
   ptr->interval_usec = sampling_interval;
   ptr->nrcpus = all_cpus;
#endif
}

inline static long_long get_cycles(void)
{
   long_long ret;
#ifdef __x86_64__
   do {
      unsigned int a,d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      (ret) = ((unsigned long)a) | (((unsigned long)d)<<32);
   } while(0);
#else
   __asm__ __volatile__("rdtsc":"=A"(ret)
                        : /* no inputs */ );
#endif
   return ret;
}

/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
extern int _papi_hwd_mdi_init()
{
  /* Name of the substrate we're using */
   strcpy(_papi_hwi_system_info.substrate, "$Id$");      

   _papi_hwi_system_info.supports_hw_overflow = 1;
   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_inheritance = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;
   _papi_hwi_system_info.supports_virt_usec = 1;
   _papi_hwi_system_info.supports_virt_cyc = 1;

   return (PAPI_OK);
}

/* volatile uint32_t lock; */

#include <inttypes.h>

static void lock_init()
{
   int lck;
   for (lck = 0; lck < PAPI_MAX_LOCK; lck++)
      lock[lck] = MUTEX_OPEN;
}

/* Called when PAPI/process is initialized */

int _papi_hwd_init_global(void)
{
   int fd, retval;
   struct perfctr_info info;

   /* Use lower level calls per Mikael to get the perfctr info
      without actually creating a new kernel-side state.
      Also, close the fd immediately after retrieving the info.
      This is much lighter weight and doesn't reserve the counter
      resources. Also compatible with perfctr 2.6.14.
   */

   fd = _vperfctr_open(0);
   if (fd < 0)
     { PAPIERROR(VOPEN_ERROR); return(PAPI_ESYS); }

   retval = perfctr_info(fd, &info);
   close(fd);
   if (retval < 0)
     { PAPIERROR(VINFO_ERROR); return(PAPI_ESYS); }

   /* Initialize outstanding values in machine info structure */

   if (_papi_hwd_mdi_init() != PAPI_OK) {
      return (PAPI_ESBSTR);
   }

   /* Fill in what we can of the papi_system_info. */
   retval = _papi_hwd_get_system_info();
   if (retval)
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

   /* Setup memory info */
   retval =
       _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, (int) info.cpu_type);
   if (retval)
      return (retval);

    /* Setup presets */
   retval = setup_p4_presets(info.cpu_type);
   if (retval)
      return (retval);

   SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n", dev);
    vperfctr_close(dev);

    lock_init();

    return (PAPI_OK);
}

/* Called when thread is initialized */

int _papi_hwd_init(P4_perfctr_context_t * ctx)
{
   struct vperfctr_control tmp;

   /* Malloc the space for our controls */

#if 0
   ctx->start.control =
       (struct vperfctr_control *) malloc(sizeof(struct vperfctr_control));
   ctx->start.state = (struct perfctr_sum_ctrs *) malloc(sizeof(struct perfctr_sum_ctrs));
   if ((ctx->start.control == NULL) || (ctx->start.state == NULL))
     { PAPIERROR(STATE_MAL_ERROR); return(PAPI_ENOMEM); }
#endif

   /* Initialize our thread/process pointer. */

   if ((ctx->perfctr = vperfctr_open()) == NULL)
     { PAPIERROR(VOPEN_ERROR); return(PAPI_ESYS); }
   SUBDBG("_papi_hwd_init vperfctr_open() = %p\n", ctx->perfctr);

#if 0
   if ((ctx->perfctr = gperfctr_open()) == NULL)
     { PAPIERROR(GOPEN_ERROR); return(PAPI_ESYS); }
   SUBDBG("_papi_hwd_init gperfctr_open() = %p\n", ctx->perfctr);
#endif

   /* Initialize the per thread/process virtualized TSC */
   memset(&tmp, 0x0, sizeof(tmp));
   tmp.cpu_control.tsc_on = 1;

   /* Start the per thread/process virtualized TSC */
   if (vperfctr_control(ctx->perfctr, &tmp) < 0)
     { PAPIERROR(VCNTRL_ERROR); return(PAPI_ESYS); }
#if 0
   if (gperfctr_control(ctx->perfctr, &tmp) < 0)
     { PAPIERROR(GCNTRL_ERROR); return(PAPI_ESYS); }
#endif

   return (PAPI_OK);
}

#ifdef DEBUG
void print_control(const struct perfctr_cpu_control *control)
{
   unsigned int i;

   SUBDBG("Control used:\n");
   SUBDBG("tsc_on\t\t\t0x%x\n", control->tsc_on);
   SUBDBG("nractrs\t\t\t0x%x\n", control->nractrs);
   SUBDBG("nrictrs\t\t\t0x%x\n", control->nrictrs);
   for (i = 0; i < (control->nractrs + control->nrictrs); ++i) {
      if (control->pmc_map[i] >= 18) {
         SUBDBG("pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i]);
      } else {
         SUBDBG("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
      }
      SUBDBG("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
#if defined(__i386__) || defined(__x86_64__)
      if (control->evntsel_aux[i])
         SUBDBG("evntsel_aux[%u]\t\t0x%08X\n", i, control->evntsel_aux[i]);
#endif
      if (control->ireset[i])
         SUBDBG("ireset[%u]\t\t%d\n", i, control->ireset[i]);
   }
#if defined(__i386__) || defined(__x86_64__)
   if (control->p4.pebs_enable)
      SUBDBG("pebs_enable\t0x%08X\n", control->p4.pebs_enable);
   if (control->p4.pebs_matrix_vert)
      SUBDBG("pebs_matrix_vert\t0x%08X\n", control->p4.pebs_matrix_vert);
#endif
}
#endif

int _papi_hwd_start(P4_perfctr_context_t * ctx, P4_perfctr_control_t * state)
{
   int error;

#ifdef DEBUG
   print_control(&state->control.cpu_control);
#endif

   error = vperfctr_control(ctx->perfctr, &state->control);
   if (error < 0) {
      SUBDBG("vperfctr_control returns: %d\n", error);
      { PAPIERROR(VCNTRL_ERROR); return(PAPI_ESYS); }
   }
#if 0
   if (gperfctr_control(ctx->perfctr, &state->control) < 0)
     { PAPIERROR(GCNTRL_ERROR); return(PAPI_ESYS); }
#endif

   return (PAPI_OK);
}

int _papi_hwd_stop(P4_perfctr_context_t * ctx, P4_perfctr_control_t * state)
{
   if (vperfctr_stop(ctx->perfctr) < 0)
     { PAPIERROR(VCNTRL_ERROR); return(PAPI_ESYS); }
#if 0
   if (gperfctr_stop(ctx->perfctr) < 0)
     { PAPIERROR(GCNTRL_ERROR); return(PAPI_ESYS); }
#endif

   return (PAPI_OK);
}


int _papi_hwd_read(P4_perfctr_context_t * ctx, P4_perfctr_control_t * spc,
                   long_long ** dp, int flags)
{
 
   if ( flags & PAPI_PAUSED ) {
     int i,j=0;
     for ( i=0;i<spc->control.cpu_control.nractrs+spc->control.cpu_control.nrictrs; i++) {
       spc->state.pmc[j] = 0;
       if ( (spc->control.cpu_control.evntsel[i] & CCCR_OVF_PMI_T0) ) continue;
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
         for (i = 0; i < spc->control.cpu_control.nractrs; i++) {
            SUBDBG("raw val hardware index %d is %lld\n", i,
                   (long_long) spc->state.pmc[i]);
         }
      }
   }
#endif
   return (PAPI_OK);
}


/* This routine is for shutting down threads, including the
   master thread. */

int _papi_hwd_shutdown(P4_perfctr_context_t * ctx)
{
   int retval = vperfctr_unlink(ctx->perfctr);
   SUBDBG("_papi_hwd_init_global vperfctr_unlink(%p) = %d\n", ctx->perfctr, retval);
   vperfctr_close(ctx->perfctr);
   SUBDBG("_papi_hwd_init_global vperfctr_close(%p)\n", ctx->perfctr);
   memset(ctx, 0x0, sizeof(P4_perfctr_context_t));

   if (retval)
      return (PAPI_ESYS);
   return (PAPI_OK);
}

/* Called once per process. */

int _papi_hwd_shutdown_global(void)
{
   return (PAPI_OK);
}

/* Timers */

long_long _papi_hwd_get_real_usec(void)
{
   return ((long_long) get_cycles() / (long_long) _papi_hwi_system_info.hw_info.mhz);
}

long_long _papi_hwd_get_real_cycles(void)
{
   return ((long_long)get_cycles());
}

long_long _papi_hwd_get_virt_cycles(const P4_perfctr_context_t * ctx)
{
   return ((long_long)vperfctr_read_tsc(ctx->perfctr));
}

long_long _papi_hwd_get_virt_usec(const P4_perfctr_context_t * ctx)
{
   return ((long_long)vperfctr_read_tsc(ctx->perfctr) /
           (long_long)_papi_hwi_system_info.hw_info.mhz);
}

#ifdef DEBUG

#if 0
static void print_bits(P4_register_t * b)
{
   SUBDBG("  counter[0,1]: 0x%x, 0x%x\n", b->counter[0], b->counter[1]);
   SUBDBG("  escr[0,1]: 0x%x, 0x%x\n", b->escr[0], b->escr[1]);
   SUBDBG("  cccr: 0x%x,  event: 0x%x\n", b->cccr, b->event);
   SUBDBG("  pebs_enable: 0x%x,  pebs_matrix_vert: 0x%x,  ireset: 0x%x\n", b->pebs_enable,
          b->pebs_matrix_vert, b->ireset);
}
#endif

static void print_alloc(P4_reg_alloc_t * a)
{
   SUBDBG("P4_reg_alloc:\n");
//    print_bits(&(a->ra_bits));
   SUBDBG("  selector: 0x%x\n", a->ra_selector);
   SUBDBG("  rank: 0x%x\n", a->ra_rank);
   SUBDBG("  escr: 0x%x 0x%x\n", a->ra_escr[0], a->ra_escr[1]);
}

#endif

/* This function examines the event to determine
    if it can be mapped to counter ctr. 
    Returns true if it can, false if it can't.
*/
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr)
{
   return (dst->ra_selector & (1 << ctr));
}

/* This function forces the event to
    be mapped to only counter ctr. 
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t * dst, int ctr)
{
   dst->ra_selector = (1 << ctr);
   dst->ra_rank = 1;
   /* Pentium 4 requires that both an escr and a counter are selected.
      Find which counter mask contains this counter.
      Set the opposite escr to empty (-1) */
   if (dst->ra_bits.counter[0] & dst->ra_selector)
      dst->ra_escr[1] = -1;
   else
      dst->ra_escr[0] = -1;
}

/* This function examines the event to determine
    if it has a single exclusive mapping. 
    Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst)
{
   return (dst->ra_rank == 1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   int retval1, retval2;
   /* Pentium 4 needs to check for conflict of both counters and esc registers */
             /* selectors must share bits */
   retval1 = ((dst->ra_selector & src->ra_selector) ||
             /* or escrs must equal each other and not be set to -1 */
             ((dst->ra_escr[0] == src->ra_escr[0]) && (dst->ra_escr[0] != -1)) ||
             ((dst->ra_escr[1] == src->ra_escr[1]) && (dst->ra_escr[1] != -1)));
   /* Pentium 4 also needs to check for conflict on pebs registers */
               /* pebs enables must both be non-zero */
   retval2 = (((dst->ra_bits.pebs_enable && src->ra_bits.pebs_enable) &&
               /* and not equal to each other */
               (dst->ra_bits.pebs_enable != src->ra_bits.pebs_enable)) ||
               /* same for pebs_matrix_vert */
              ((dst->ra_bits.pebs_matrix_vert && src->ra_bits.pebs_matrix_vert) &&
               (dst->ra_bits.pebs_matrix_vert != src->ra_bits.pebs_matrix_vert)));
   if (retval2) SUBDBG("pebs conflict!\n");
   return(retval1 | retval2);
}

/* This function removes shared resources available to the src event
    from the resources available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   int i;
   unsigned shared;

   /* On Pentium 4, shared resources include escrs, counters, and pebs registers
      There is only one pair of pebs registers, so if two events use them differently
      there is an immediate conflict, and the dst rank is forced to 0.
   */
#ifdef DEBUG
   SUBDBG("src, dst\n");
   print_alloc(src);
   print_alloc(dst);
#endif

   /* check for a pebs conflict */
       /* pebs enables must both be non-zero */
   i = (((dst->ra_bits.pebs_enable && src->ra_bits.pebs_enable) &&
         /* and not equal to each other */
         (dst->ra_bits.pebs_enable != src->ra_bits.pebs_enable)) ||
         /* same for pebs_matrix_vert */
        ((dst->ra_bits.pebs_matrix_vert && src->ra_bits.pebs_matrix_vert) &&
         (dst->ra_bits.pebs_matrix_vert != src->ra_bits.pebs_matrix_vert)));
   if (i) {
      SUBDBG("pebs conflict! clearing selector\n");
      dst->ra_selector = 0;
      return;
   } else {
      /* remove counters referenced by any shared escrs */
      if ((dst->ra_escr[0] == src->ra_escr[0]) && (dst->ra_escr[0] != -1)) {
         dst->ra_selector &= ~dst->ra_bits.counter[0];
         dst->ra_escr[0] = -1;
      }
      if ((dst->ra_escr[1] == src->ra_escr[1]) && (dst->ra_escr[1] != -1)) {
         dst->ra_selector &= ~dst->ra_bits.counter[1];
         dst->ra_escr[1] = -1;
      }

      /* remove any remaining shared counters */
      shared = (dst->ra_selector & src->ra_selector);
      if (shared)
         dst->ra_selector ^= shared;
   }
   /* recompute rank */
   for (i = 0, dst->ra_rank = 0; i < MAX_COUNTERS; i++)
      if (dst->ra_selector & (1 << i))
         dst->ra_rank++;
#ifdef DEBUG
   SUBDBG("new dst\n");
   print_alloc(dst);
#endif
}

/* This function updates the selection status of 
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   dst->ra_selector = src->ra_selector;
   dst->ra_escr[0] = src->ra_escr[0];
   dst->ra_escr[1] = src->ra_escr[1];
}


/* Register allocation */

int _papi_hwd_allocate_registers(EventSetInfo_t * ESI)
{
   int i, j, natNum;
   P4_reg_alloc_t event_list[MAX_COUNTERS], *e;

   /* not yet successfully mapped, but have enough slots for events */

   /* Initialize the local structure needed 
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   SUBDBG("native event count: %d\n", natNum);
   for (i = 0; i < natNum; i++) {
      /* dereference event_list so code is easier to read */
      e = &event_list[i];

      /* retrieve the mapping information about this native event */
      _papi_hwd_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_event, &e->ra_bits);

      /* combine counter bit masks for both esc registers into selector */
      e->ra_selector = e->ra_bits.counter[0] | e->ra_bits.counter[1];
      /* calculate native event rank, which is number of counters it can live on */
      e->ra_rank = 0;
      for (j = 0; j < MAX_COUNTERS; j++) {
         if (e->ra_selector & (1 << j)) {
            e->ra_rank++;
         }
      }

      /* set the bits for the two esc registers this event can live on */
      e->ra_escr[0] = e->ra_bits.escr[0];
      e->ra_escr[1] = e->ra_bits.escr[1];

#ifdef DEBUG
      SUBDBG("i: %d\n", i);
      print_alloc(e);
#endif
   }

   if (_papi_hwi_bipartite_alloc(event_list, natNum)) { /* successfully mapped */
      for (i = 0; i < natNum; i++) {
#ifdef DEBUG
         SUBDBG("i: %d\n", i);
         print_alloc(&event_list[i]);
#endif
         /* Copy all the info about this native event to the NativeInfo struct */
         ESI->NativeInfoArray[i].ni_bits = event_list[i].ra_bits;

         /* The selector contains the counter bit position. Turn it into a number
            and store it in the first counter value, zeroing the second. */
         ESI->NativeInfoArray[i].ni_bits.counter[0] = ffs(event_list[i].ra_selector) - 1;
         ESI->NativeInfoArray[i].ni_bits.counter[1] = 0;

         /* Array order on perfctr is event ADD order, not counter #... */
         ESI->NativeInfoArray[i].ni_position = i;
      }
      return 1;
   }

   return (PAPI_OK);
}


static void clear_cs_events(hwd_control_state_t * this_state)
{
   int i,j;

   /* total counters is sum of accumulating (nractrs) and interrupting (nrictrs) */
   j = this_state->control.cpu_control.nractrs + this_state->control.cpu_control.nrictrs;

   /* Remove all counter control command values from eventset. */
   for (i = 0; i < j; i++) {
      SUBDBG("Clearing pmc event entry %d\n", i);
      this_state->control.cpu_control.pmc_map[i] = 0;
      this_state->control.cpu_control.evntsel[i] = 0;
      this_state->control.cpu_control.evntsel_aux[i] = 
         this_state->control.cpu_control.evntsel_aux[i] & (ESCR_T0_OS | ESCR_T0_USR);
      this_state->control.cpu_control.ireset[i] = 0;
   }

   /* Clear pebs stuff */
   this_state->control.cpu_control.p4.pebs_enable = 0;
   this_state->control.cpu_control.p4.pebs_matrix_vert = 0;

   /* clear both a and i counter counts */
   this_state->control.cpu_control.nractrs = 0;
   this_state->control.cpu_control.nrictrs = 0;

#ifdef DEBUG
   print_control(&this_state->control.cpu_control);
#endif
}


/* This function clears the current contents of the control structure and updates it 
   with whatever resources are allocated for all the native events 
   in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                                   NativeInfo_t * native, int count, hwd_context_t *ctx)
{
   int i, retval = PAPI_OK;

   P4_register_t *bits;
   struct perfctr_cpu_control *cpu_control = &this_state->control.cpu_control;

   /* clear out the events from the control state */
   clear_cs_events(this_state);

   /* fill the counters we're using */
   for (i = 0; i < count; i++) {
      /* dereference the mapping information about this native event */
      bits = &native[i].ni_bits;

      /* Add counter control command values to eventset */

      cpu_control->pmc_map[i] = bits->counter[0];
      cpu_control->evntsel[i] = bits->cccr;
      cpu_control->ireset[i] = bits->ireset;
      cpu_control->pmc_map[i] |= FAST_RDPMC;
      cpu_control->evntsel_aux[i] |= bits->event;

    /* pebs_enable and pebs_matrix_vert are shared registers used for replay_events.
      Replay_events count L1 and L2 cache events. There is only one of each for 
	   the entire eventset. Therefore, there can be only one unique replay_event 
	   per eventset. This means L1 and L2 can't be counted together. Which stinks.
      This conflict should be trapped in the allocation scheme, but we'll test for it
      here too, just in case.
    */
      if (bits->pebs_enable) {
	 /* if pebs_enable isn't set, just copy */
         if (cpu_control->p4.pebs_enable == 0) {
            cpu_control->p4.pebs_enable = bits->pebs_enable;
	 /* if pebs_enable conflicts, flag an error */
         } else if (cpu_control->p4.pebs_enable != bits->pebs_enable) {
            SUBDBG("WARNING: _papi_hwd_update_control_state -- pebs_enable conflict!");
	         retval = PAPI_ECNFLCT;
         }
	 /* if pebs_enable == bits->pebs_enable, do nothing */
      }
      if (bits->pebs_matrix_vert) {
	 /* if pebs_matrix_vert isn't set, just copy */
         if (cpu_control->p4.pebs_matrix_vert == 0) {
	         cpu_control->p4.pebs_matrix_vert = bits->pebs_matrix_vert;
	 /* if pebs_matrix_vert conflicts, flag an error */
         } else if (cpu_control->p4.pebs_matrix_vert != bits->pebs_matrix_vert) {
            SUBDBG("WARNING: _papi_hwd_update_control_state -- pebs_matrix_vert conflict!");
 	         retval = PAPI_ECNFLCT;
         }
	 /* if pebs_matrix_vert == bits->pebs_matrix_vert, do nothing */
     }
   }
   this_state->control.cpu_control.nractrs = count;

   /* Make sure the TSC is always on */
   this_state->control.cpu_control.tsc_on = 1;

#ifdef DEBUG
   print_control(&this_state->control.cpu_control);
#endif
   return (retval);
}


int _papi_hwd_add_prog_event(P4_perfctr_control_t * state, unsigned int code, void *tmp,
                             EventInfo_t * tmp2)
{
   return (PAPI_ESBSTR);
}


int _papi_hwd_set_domain(P4_perfctr_control_t * cntrl, int domain)
{
   int i, did = 0;
    
     /* Clear the current domain set for this event set */
     /* We don't touch the Enable bit in this code but  */
     /* leave it as it is */
   for(i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
      cntrl->control.cpu_control.evntsel_aux[i] &= ~(ESCR_T0_OS|ESCR_T0_USR);
   }
   if(domain & PAPI_DOM_USER) {
      did = 1;
      for(i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel_aux[i] |= ESCR_T0_USR;
      }
   }
   if(domain & PAPI_DOM_KERNEL) {
      did = 1;
      for(i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel_aux[i] |= ESCR_T0_OS;
      }
   }
   if(!did)
      return(PAPI_EINVAL);
   else
      return(PAPI_OK);
}

int _papi_hwd_reset(P4_perfctr_context_t * ctx, P4_perfctr_control_t * cntrl)
{
   /* this is what I gleaned from PAPI 2.3.4... is it right??? dkt */
   return (_papi_hwd_start(ctx, cntrl));
}


int _papi_hwd_write(P4_perfctr_context_t * ctx, P4_perfctr_control_t * cntrl,
                    long long *from)
{
   return (PAPI_ESBSTR);
}


int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_ESBSTR);
}


int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_ESBSTR);
}


/* Perfctr requires that interrupting counters appear at the end of the pmc list.
   In the case a user wants to interrupt on a counter in an evntset that is not 
   among the last events, we need to move the perfctr virtual events around to 
   make it last. This function swaps two perfctr events, and then adjusts the
   position entries in both the NativeInfoArray and the EventInfoArray to keep
   everything consistent.
*/

static void swap_events(EventSetInfo_t * ESI, struct vperfctr_control *contr, int cntr1,
                        int cntr2)
{
   unsigned int ui;
   int si, i, j;

   for (i = 0; i < ESI->NativeCount; i++) {
      if (ESI->NativeInfoArray[i].ni_position == cntr1)
         ESI->NativeInfoArray[i].ni_position = cntr2;
      else if (ESI->NativeInfoArray[i].ni_position == cntr2)
         ESI->NativeInfoArray[i].ni_position = cntr1;
   }

   for (i = 0; i < ESI->NumberOfEvents; i++) {
      for (j = 0; ESI->EventInfoArray[i].pos[j] >= 0; j++) {
         if (ESI->EventInfoArray[i].pos[j] == cntr1)
            ESI->EventInfoArray[i].pos[j] = cntr2;
         else if (ESI->EventInfoArray[i].pos[j] == cntr2)
            ESI->EventInfoArray[i].pos[j] = cntr1;
      }
   }

   ui = contr->cpu_control.pmc_map[cntr1];
   contr->cpu_control.pmc_map[cntr1] = contr->cpu_control.pmc_map[cntr2];
   contr->cpu_control.pmc_map[cntr2] = ui;

   ui = contr->cpu_control.evntsel[cntr1];
   contr->cpu_control.evntsel[cntr1] = contr->cpu_control.evntsel[cntr2];
   contr->cpu_control.evntsel[cntr2] = ui;

   ui = contr->cpu_control.evntsel_aux[cntr1];
   contr->cpu_control.evntsel_aux[cntr1] = contr->cpu_control.evntsel_aux[cntr2];
   contr->cpu_control.evntsel_aux[cntr2] = ui;

   si = contr->cpu_control.ireset[cntr1];
   contr->cpu_control.ireset[cntr1] = contr->cpu_control.ireset[cntr2];
   contr->cpu_control.ireset[cntr2] = si;
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   struct vperfctr_control *contr = &this_state->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

   OVFDBG("EventIndex=%d\n", EventIndex);

#ifdef DEBUG
   /* The correct event to overflow is EventIndex */
   print_control(&this_state->control.cpu_control);
#endif

   ncntrs = _papi_hwi_system_info.num_cntrs;
   i = ESI->EventInfoArray[EventIndex].pos[0];
   if (i >= ncntrs) 
     {
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
      contr->cpu_control.evntsel[i] |= CCCR_OVF_PMI_T0;
      contr->cpu_control.nrictrs++;
      contr->cpu_control.nractrs--;
      nricntrs = contr->cpu_control.nrictrs;
      nracntrs = contr->cpu_control.nractrs;
      contr->si_signo = PAPI_SIGNAL;

      /* move this event to the bottom part of the list if needed */
      if (i < nracntrs) {
         swap_events(ESI, contr, i, nracntrs);
      }
      OVFDBG("Modified event set\n");
   } else {
      if (contr->cpu_control.evntsel[i] & CCCR_OVF_PMI_T0) {
         contr->cpu_control.ireset[i] = 0;
         contr->cpu_control.evntsel[i] &= (~CCCR_OVF_PMI_T0);
         contr->cpu_control.nrictrs--;
         contr->cpu_control.nractrs++;
      }
      nricntrs = contr->cpu_control.nrictrs;
      nracntrs = contr->cpu_control.nractrs;

      /* move this event to the top part of the list if needed */
      if (i >= nracntrs) {
         swap_events(ESI, contr, i, nracntrs - 1);
      }
      if (!nricntrs)
         contr->si_signo = 0;

      OVFDBG("Modified event set\n");

      retval = _papi_hwi_stop_signal(PAPI_SIGNAL);
   }
#ifdef DEBUG
   print_control(&this_state->control.cpu_control);
#endif
   OVFDBG("End of call. Exit code: %d\n", retval);
   return (retval);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *context)
{
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;
   int isHardware=0;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

   _papi_hwi_dispatch_overflow_signal((void *) &ctx, &isHardware,
                                      si->si_pmc_ovf_mask, 0, &master);

   /* We are done, resume interrupting counters */

   if (isHardware) {
      if (vperfctr_iresume(master->context.perfctr) < 0) {
         PAPIERROR("vperfctr_iresume errno %d",errno);
      }
   }
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
