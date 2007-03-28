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
#include "perfctr-p4.h"

/* Prototypes for entry points found in linux.c and linux-memory.c */
int _linux_update_shlib_info(void);
int _linux_get_system_info(void);
int _linux_get_memory_info(PAPI_hw_info_t * hw_info, int cpu_type);
int _linux_get_dmem_info(PAPI_dmem_info_t *d);
int _linux_init(hwd_context_t * ctx);

/* Prototypes for entry points found in p4_events */
int _p4_ntv_enum_events(unsigned int *EventCode, int modifer);
char *_p4_ntv_code_to_name(unsigned int EventCode);
char *_p4_ntv_code_to_descr(unsigned int EventCode);
int _p4_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits);
int _p4_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values,
                          int name_len, int count);

#if defined(PERFCTR26) || defined (PERFCTR25)
#define evntsel_aux             p4.escr
#endif

/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

extern hwi_search_t _p4_base_preset_map[];
extern hwi_search_t _p4_tot_iis_preset_map[];
extern hwi_search_t _p4_L3_cache_map[];
extern hwi_dev_notes_t _p4_base_dev_notes[];

/*****************************/
/* END EXTERNAL DECLARATIONS */
/*****************************/

/****************************/
/* BEGIN LOCAL DECLARATIONS */
/****************************/

/**************************/
/* END LOCAL DECLARATIONS */
/**************************/

/******************************************************************************
 * The below defines were imported from linux.c and will therefore need to be
 * duplicated in every substrate that relied on them, such as PPC and p4
 ******************************************************************************/

long_long tb_scale_factor = (long_long)1; /* needed to scale get_cycles on PPC series */

extern int setup_p4_presets(int cputype);

#if defined(PERFCTR26)
#define PERFCTR_CPU_NAME(pi)    perfctr_info_cpu_name(pi)
#define PERFCTR_CPU_NRCTRS(pi)  perfctr_info_nrctrs(pi)
#elif defined(PERFCTR25)
#define PERFCTR_CPU_NAME        perfctr_info_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME        perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_cpu_nrctrs
#endif

/******************************************************************************
 * The above defines were imported from linux.c and will therefore need to be
 * duplicated in every substrate that relied on them, such as power and p4
 ******************************************************************************/

int setup_p4_presets(int cputype)
{
   hwi_search_t *s;
   hwi_dev_notes_t *n;
   extern void _p4_fixup_fp(hwi_search_t **s, hwi_dev_notes_t **n);
   extern void _p4_fixup_vec(hwi_search_t **s, hwi_dev_notes_t **n);

   /* load the baseline event map for all Pentium 4s */
   _papi_hwi_setup_all_presets(_p4_base_preset_map, _p4_base_dev_notes);

   /* fix up the floating point and vector ops */
   _p4_fixup_fp(&s, &n);
   _papi_hwi_setup_all_presets(s,n);
   _p4_fixup_vec(&s, &n);
   _papi_hwi_setup_all_presets(s,n);

   /* install L3 cache events iff 3 levels of cache exist */
   if (_papi_hwi_system_info.hw_info.mem_hierarchy.levels == 3)
      _papi_hwi_setup_all_presets(_p4_L3_cache_map, NULL);

   /* overload with any model dependent events */
   if (cputype == PERFCTR_X86_INTEL_P4) {
     /* do nothing besides the base map */
   }
   else if (cputype == PERFCTR_X86_INTEL_P4M2) {
      _papi_hwi_setup_all_presets(_p4_tot_iis_preset_map, NULL);
   }
#ifdef PERFCTR_X86_INTEL_P4M3
   else if (cputype == PERFCTR_X86_INTEL_P4M3) {
      _papi_hwi_setup_all_presets(_p4_tot_iis_preset_map, NULL);
   }
#endif
   else {
      PAPIERROR(MODEL_ERROR);
      return(PAPI_ESBSTR);
   }
   return (PAPI_OK);
}

/* This used to be init_config, static to the substrate.
   Now its exposed through the vector table and called when an EventSet is allocated.
*/
static int _p4_init_control_state(hwd_control_state_t * cntl)
{
   int def_mode = 0, i;
   cmp_control_state_t *ptr = (cmp_control_state_t *)cntl;

   if (MY_VECTOR.cmp_info.default_domain & PAPI_DOM_USER)
      def_mode |= ESCR_T0_USR;
   if (MY_VECTOR.cmp_info.default_domain & PAPI_DOM_KERNEL)
     def_mode |= ESCR_T0_OS;

   for(i = 0; i < MY_VECTOR.cmp_info.num_cntrs; i++) {
      ptr->control.cpu_control.evntsel_aux[i] |= def_mode;
   }
   ptr->control.cpu_control.tsc_on = 1;
   ptr->control.cpu_control.nractrs = 0;
   ptr->control.cpu_control.nrictrs = 0;
#if 0
   ptr->interval_usec = sampling_interval;
   ptr->nrcpus = all_cpus;
#endif
   return(PAPI_OK);
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

/******************************************************************************
 * The below routines were imported from linux.c and will therefore need to be
 * duplicated in every substrate that relied on them, such as power and p4
 ******************************************************************************/

extern int setup_p4_presets(int cputype);
//extern int setup_p4_vector_table(papi_vector_t *);
extern int setup_p3_presets(int cputype);

#if defined(PERFCTR26)
#define PERFCTR_CPU_NAME(pi)    perfctr_info_cpu_name(pi)
#define PERFCTR_CPU_NRCTRS(pi)  perfctr_info_nrctrs(pi)
#elif defined(PERFCTR25)
#define PERFCTR_CPU_NAME        perfctr_info_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_info_nrctrs
#else
#define PERFCTR_CPU_NAME        perfctr_cpu_name
#define PERFCTR_CPU_NRCTRS      perfctr_cpu_nrctrs
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
#ifdef PERFCTR_X86_INTEL_CORE
   case PERFCTR_X86_INTEL_CORE:
#endif
#ifdef PERFCTR_X86_INTEL_CORE2
   case PERFCTR_X86_INTEL_CORE2:
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

/* 
 * 1 if the processor is a P4, 0 otherwise
 */
static int check_p4(int cputype){
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

/* volatile uint32_t lock; */

volatile unsigned int lock[PAPI_MAX_LOCK];

static void lock_init(void) {
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++) {
      lock[i] = MUTEX_OPEN;
   }
}

static int _p4_init_substrate(int cidx)
{
  int retval;
  struct perfctr_info info;
  char abiv[PAPI_MIN_STR_LEN];

#if defined(PERFCTR26)
  int fd;
#else
  struct vperfctr *dev;
#endif

  /* Setup the vector entries that the OS knows about */
  /*retval = _linux_setup_vector_table(vtable);
  if ( retval != PAPI_OK ) return(retval);
  */
 #if defined(PERFCTR26)
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

    /* copy tsc multiplier to local variable        */
    /* this field appears in perfctr 2.6 and higher */
 	tb_scale_factor = (long_long)info.tsc_to_cpu_mult;
#else
   /* Opened once for all threads. */
   if ((dev = vperfctr_open()) == NULL)
     { PAPIERROR( VOPEN_ERROR); return(PAPI_ESYS); }
   SUBDBG("_papi_hwd_init_substrate vperfctr_open = %p\n", dev);

   /* Get info from the kernel */
   retval = vperfctr_info(dev, &info);
   if (retval < 0)
     { PAPIERROR( VINFO_ERROR); return(PAPI_ESYS); }
    vperfctr_close(dev);
#endif

  /* Fill in what we can of the papi_system_info. */
  retval = MY_VECTOR.get_system_info();
  if (retval != PAPI_OK)
     return (retval);

   /* Setup memory info */
   retval = MY_VECTOR.get_memory_info(&_papi_hwi_system_info.hw_info, (int) info.cpu_type);
   if (retval)
      return (retval);

   strcpy(MY_VECTOR.cmp_info.name, "$Id$");
   strcpy(MY_VECTOR.cmp_info.version, "$Revision$");
   sprintf(abiv,"0x%08X",info.abi_version);
   strcpy(MY_VECTOR.cmp_info.support_version, abiv);
   strcpy(MY_VECTOR.cmp_info.kernel_version, info.driver_version);
   MY_VECTOR.cmp_info.CmpIdx = cidx;
   MY_VECTOR.cmp_info.num_cntrs = PERFCTR_CPU_NRCTRS(&info);
   MY_VECTOR.cmp_info.fast_counter_read = (info.cpu_features & PERFCTR_FEATURE_RDPMC) ? 1 : 0;
   MY_VECTOR.cmp_info.hardware_intr =
       (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;

   SUBDBG("Hardware/OS %s support counter generated interrupts\n",
          MY_VECTOR.cmp_info.hardware_intr ? "does" : "does not");

   /* Setup presets */
   if ( check_p4(info.cpu_type) ){
//     retval = setup_p4_vector_table(vtable);
//     if (!retval)
     	retval = setup_p4_presets(info.cpu_type);
   }
   else{
     	retval = PAPI_ESBSTR;
   }
   if ( retval ) 
     return(retval);

   strcpy(_papi_hwi_system_info.hw_info.model_string, PERFCTR_CPU_NAME(&info));
   _papi_hwi_system_info.hw_info.model = info.cpu_type;
   _papi_hwi_system_info.hw_info.vendor = xlate_cpu_type_to_vendor(info.cpu_type);

   lock_init();

   return (PAPI_OK);
}

void _p4_dispatch_timer(int signal, siginfo_t * si, void *context) {
   _papi_hwi_context_t ctx;
   ThreadInfo_t *master = NULL;
   int isHardware = 0;
   caddr_t pc;
   int cidx = MY_VECTOR.cmp_info.CmpIdx;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

#define OVERFLOW_MASK si->si_pmc_ovf_mask
#define GEN_OVERFLOW 0

   pc = GET_OVERFLOW_ADDRESS(ctx);

   _papi_hwi_dispatch_overflow_signal((void *)&ctx,&isHardware,
                                      OVERFLOW_MASK, GEN_OVERFLOW,&master,pc, MY_VECTOR.cmp_info.CmpIdx);

   /* We are done, resume interrupting counters */
   if (isHardware) {
      errno = vperfctr_iresume(((cmp_context_t *)master->context[cidx])->perfctr);
      if (errno < 0) {
	  PAPIERROR("vperfctr_iresume errno %d for perfctr: %p",errno, ((cmp_context_t *)master->context[cidx])->perfctr);
      }
   }
}
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

/* Low level functions, should not handle errors, just return codes. */

inline_static long_long get_cycles(void) {
   long_long ret = 0;
#ifdef __x86_64__
   do {
      unsigned int a,d;
      asm volatile("rdtsc" : "=a" (a), "=d" (d));
      (ret) = ((long_long)a) | (((long_long)d)<<32);
   } while(0);
#else
   __asm__ __volatile__("rdtsc"
                       : "=A" (ret)
                       : );
#endif
   return ret;
}

static long_long _p4_get_real_usec(void) {
   return((long_long)get_cycles() / (long_long)_papi_hwi_system_info.hw_info.mhz);
}

static long_long _p4_get_real_cycles(void) {
   return((long_long)get_cycles());
}

static long_long _p4_get_virt_cycles(const hwd_context_t * ctx)
{
   return ((long_long)vperfctr_read_tsc(((cmp_context_t *)ctx)->perfctr) * tb_scale_factor);
}

static long_long _p4_get_virt_usec(const hwd_context_t * ctx)
{
   return (((long_long)vperfctr_read_tsc(((cmp_context_t *)ctx)->perfctr) * tb_scale_factor) /
           (long_long)_papi_hwi_system_info.hw_info.mhz);
}

/******************************************************************************
 * The above routines were imported from linux.c and will therefore need to be
 * duplicated in every substrate that relied on them, such as power and p4
 ******************************************************************************/


static int _p4_start(hwd_context_t * this_ctx, hwd_control_state_t * this_state)
{
   int error;
   cmp_context_t * ctx = (cmp_context_t *)this_ctx;
   cmp_control_state_t *state = (cmp_control_state_t *)this_state;

#ifdef DEBUG
   SUBDBG("From _p4_start...\n");
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

static int _p4_stop(hwd_context_t * this_ctx, hwd_control_state_t * this_state)
{
   cmp_context_t * ctx = (cmp_context_t *)this_ctx;
   cmp_control_state_t *state = (cmp_control_state_t *)this_state;

   if( state->rvperfctr != NULL ) {
     if(rvperfctr_stop((struct rvperfctr*)ctx->perfctr) < 0)
       { PAPIERROR( RCNTRL_ERROR); return(PAPI_ESYS); }
     return (PAPI_OK);
   }
   if (vperfctr_stop(ctx->perfctr) < 0)
     { PAPIERROR(VCNTRL_ERROR); return(PAPI_ESYS); }
#if 0
   if (gperfctr_stop(ctx->perfctr) < 0)
     { PAPIERROR(GCNTRL_ERROR); return(PAPI_ESYS); }
#endif

   return (PAPI_OK);
}

static int _p4_read(hwd_context_t * this_ctx, hwd_control_state_t * this_state,
                   long_long ** dp, int flags)
{
   cmp_context_t * ctx = (cmp_context_t *)this_ctx;
   cmp_control_state_t *spc = (cmp_control_state_t *)this_state;

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
      SUBDBG("vperfctr_read_ctrs\n");
      if( spc->rvperfctr != NULL ) {
        rvperfctr_read_ctrs( spc->rvperfctr, &spc->state );
      } else {
        vperfctr_read_ctrs(ctx->perfctr, &spc->state);
        }
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

static int _p4_shutdown(hwd_context_t * ctx)
{
   int retval = vperfctr_unlink(((cmp_context_t *)ctx)->perfctr);
   SUBDBG("_p4_shutdown vperfctr_unlink(%p) = %d\n", ((cmp_context_t *)ctx)->perfctr, retval);
   vperfctr_close(((cmp_context_t *)ctx)->perfctr);
   SUBDBG("_p4_shutdown vperfctr_close(%p)\n", ((cmp_context_t *)ctx)->perfctr);
   memset(ctx, 0x0, sizeof(cmp_context_t));

   if (retval)
      return (PAPI_ESYS);
   return (PAPI_OK);
}

#ifdef DEBUG

#if 0
static void print_bits(_p4_register_t * b)
{
   SUBDBG("  counter[0,1]: 0x%x, 0x%x\n", b->counter[0], b->counter[1]);
   SUBDBG("  escr[0,1]: 0x%x, 0x%x\n", b->escr[0], b->escr[1]);
   SUBDBG("  cccr: 0x%x,  event: 0x%x\n", b->cccr, b->event);
   SUBDBG("  pebs_enable: 0x%x,  pebs_matrix_vert: 0x%x,  ireset: 0x%x\n", b->pebs_enable,
          b->pebs_matrix_vert, b->ireset);
}
#endif

static void print_alloc(_p4_reg_alloc_t * a)
{
   SUBDBG("_p4_reg_alloc:\n");
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
static int _p4_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr)
{
   return (((cmp_reg_alloc_t *)dst)->ra_selector & (1 << ctr));
}

/* This function forces the event to
    be mapped to only counter ctr. 
    Returns nothing.
*/
static void _p4_bpt_map_set(hwd_reg_alloc_t * dst, int ctr)
{
   ((cmp_reg_alloc_t *)dst)->ra_selector = (1 << ctr);
   ((cmp_reg_alloc_t *)dst)->ra_rank = 1;
   /* Pentium 4 requires that both an escr and a counter are selected.
      Find which counter mask contains this counter.
      Set the opposite escr to empty (-1) */
   if (((cmp_reg_alloc_t *)dst)->ra_bits.counter[0] & ((cmp_reg_alloc_t *)dst)->ra_selector)
      ((cmp_reg_alloc_t *)dst)->ra_escr[1] = -1;
   else
      ((cmp_reg_alloc_t *)dst)->ra_escr[0] = -1;
}

/* This function examines the event to determine
    if it has a single exclusive mapping. 
    Returns true if exlusive, false if non-exclusive.
*/
static int _p4_bpt_map_exclusive(hwd_reg_alloc_t * dst)
{
   return (((cmp_reg_alloc_t *)dst)->ra_rank == 1);
}

/* This function compares the dst and src events
    to determine if any resources are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
static int _p4_bpt_map_shared(hwd_reg_alloc_t * hdst, hwd_reg_alloc_t * hsrc)
{
   int retval1, retval2;
   cmp_reg_alloc_t * dst = (cmp_reg_alloc_t *)hdst;
   cmp_reg_alloc_t * src = (cmp_reg_alloc_t *)hsrc;

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
static void _p4_bpt_map_preempt(hwd_reg_alloc_t * hdst, hwd_reg_alloc_t * hsrc)
{
   int i;
   unsigned shared;
   cmp_reg_alloc_t * dst = (cmp_reg_alloc_t *)hdst;
   cmp_reg_alloc_t * src = (cmp_reg_alloc_t *)hsrc;


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
static void _p4_bpt_map_update(hwd_reg_alloc_t * hdst, hwd_reg_alloc_t * hsrc)
{
   cmp_reg_alloc_t * dst = (cmp_reg_alloc_t *)hdst;
   cmp_reg_alloc_t * src = (cmp_reg_alloc_t *)hsrc;

   dst->ra_selector = src->ra_selector;
   dst->ra_escr[0] = src->ra_escr[0];
   dst->ra_escr[1] = src->ra_escr[1];
}


/* Register allocation */

static int _p4_allocate_registers(EventSetInfo_t * ESI)
{
   int i, j, natNum;
   cmp_reg_alloc_t event_list[MAX_COUNTERS], *e;
   cmp_register_t *ptr;


   /* not yet successfully mapped, but have enough slots for events */

   /* Initialize the local structure needed 
      for counter allocation and optimization. */
   natNum = ESI->NativeCount;
   SUBDBG("native event count: %d\n", natNum);
   for (i = 0; i < natNum; i++) {
      /* dereference event_list so code is easier to read */
      e = &event_list[i];

      /* retrieve the mapping information about this native event */
      _p4_ntv_code_to_bits(ESI->NativeInfoArray[i].ni_event, &e->ra_bits);

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

   if (_papi_hwi_bipartite_alloc(event_list, natNum, ESI->CmpIdx)) { /* successfully mapped */
      for (i = 0; i < natNum; i++) {
#ifdef DEBUG
         SUBDBG("i: %d\n", i);
         print_alloc(&event_list[i]);
#endif
         /* Copy all info about this native event to the NativeInfo struct */
         ptr = ESI->NativeInfoArray[i].ni_bits;
         *ptr = event_list[i].ra_bits;

         /* The selector contains the counter bit position. Turn it into a number
            and store it in the first counter value, zeroing the second. */
         ptr->counter[0] = ffs(event_list[i].ra_selector) - 1;
         ptr->counter[1] = 0;

         /* Array order on perfctr is event ADD order, not counter #... */
         ESI->NativeInfoArray[i].ni_position = i;
      }
      return 1;
   }

   return (PAPI_OK);
}


static void clear_cs_events(cmp_control_state_t * this_state)
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
   SUBDBG("From clear_cs_events...\n");
   print_control(&this_state->control.cpu_control);
#endif
}


/* This function clears the current contents of the control structure and updates it 
   with whatever resources are allocated for all the native events 
   in the native info structure array. */
static int _p4_update_control_state(hwd_control_state_t * state,
                                   NativeInfo_t * native, int count, hwd_context_t *ctx)
{
   int i, retval = PAPI_OK;

   cmp_register_t *bits;
   cmp_control_state_t *this_state = (cmp_control_state_t *)state;
   struct perfctr_cpu_control *cpu_control = &this_state->control.cpu_control;

   /* clear out the events from the control state */
   clear_cs_events(this_state);

   /* fill the counters we're using */
   for (i = 0; i < count; i++) {
      /* dereference the mapping information about this native event */
      bits = (cmp_register_t *)(native[i].ni_bits);

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
            SUBDBG("WARNING: _p4_update_control_state -- pebs_enable conflict!");
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
            SUBDBG("WARNING: _p4_update_control_state -- pebs_matrix_vert conflict!");
 	         retval = PAPI_ECNFLCT;
         }
	 /* if pebs_matrix_vert == bits->pebs_matrix_vert, do nothing */
     }
   }
   this_state->control.cpu_control.nractrs = count;

   /* Make sure the TSC is always on */
   this_state->control.cpu_control.tsc_on = 1;

#ifdef DEBUG
   SUBDBG("From _p4_update_control_state...\n");
   print_control(&this_state->control.cpu_control);
#endif
   return (retval);
}


static int _p4_set_domain(hwd_control_state_t * ctl, int domain)
{
   int i, did = 0;
   cmp_control_state_t *cntrl = (cmp_control_state_t *)ctl;

    
     /* Clear the current domain set for this event set */
     /* We don't touch the Enable bit in this code but  */
     /* leave it as it is */
   for(i = 0; i < MY_VECTOR.cmp_info.num_cntrs; i++) {
      cntrl->control.cpu_control.evntsel_aux[i] &= ~(ESCR_T0_OS|ESCR_T0_USR);
   }
   if(domain & PAPI_DOM_USER) {
      did = 1;
      for(i = 0; i < MY_VECTOR.cmp_info.num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel_aux[i] |= ESCR_T0_USR;
      }
   }
   if(domain & PAPI_DOM_KERNEL) {
      did = 1;
      for(i = 0; i < MY_VECTOR.cmp_info.num_cntrs; i++) {
         cntrl->control.cpu_control.evntsel_aux[i] |= ESCR_T0_OS;
      }
   }
   if(!did)
      return(PAPI_EINVAL);
   else
      return(PAPI_OK);
}

static int _p4_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl)
{
   /* this is what I gleaned from PAPI 2.3.4... is it right??? dkt */
   return (_p4_start(ctx, cntrl));
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

//static int _p3_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold) {
//   struct hwd_pmc_control *contr = &((cmp_control_state_t *)ESI->ctl_state)->control;

   static int _p4_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   struct hwd_pmc_control *contr = &((cmp_control_state_t *)ESI->ctl_state)->control;
   int i, ncntrs, nricntrs = 0, nracntrs = 0, retval = 0;

   OVFDBG("EventIndex=%d\n", EventIndex);

#ifdef DEBUG
   /* The correct event to overflow is EventIndex */
   OVFDBG("From _p4_set_overflow: top...\n");
   print_control(&(contr->cpu_control));
#endif

   ncntrs = MY_VECTOR.cmp_info.num_cntrs;
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

      retval = _papi_hwi_start_signal(MY_VECTOR.cmp_info.hardware_intr_sig,
	  NEED_CONTEXT, MY_VECTOR.cmp_info.CmpIdx);
      if (retval != PAPI_OK)
	      return(retval);

      /* overflow interrupt occurs on the NEXT event after overflow occurs
         thus we subtract 1 from the threshold. */
      contr->cpu_control.ireset[i] = (-threshold + 1);
      contr->cpu_control.evntsel[i] |= CCCR_OVF_PMI_T0;
      contr->cpu_control.nrictrs++;
      contr->cpu_control.nractrs--;
      nricntrs = contr->cpu_control.nrictrs;
      nracntrs = contr->cpu_control.nractrs;
      contr->si_signo = MY_VECTOR.cmp_info.hardware_intr_sig;

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

      retval = _papi_hwi_stop_signal(MY_VECTOR.cmp_info.hardware_intr_sig);
   }
#ifdef DEBUG
   OVFDBG("From _p4_set_overflow: bottom...\n");
   print_control(&(contr->cpu_control));
#endif
   OVFDBG("End of call. Exit code: %d\n", retval);
   return (retval);
}

   static int _p4_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

static int _p4_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
  switch (code) {
   case PAPI_DOMAIN:
   case PAPI_DEFDOM:
      return (_p4_set_domain(option->domain.ESI->ctl_state, option->domain.domain));
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


papi_vector_t _p4_vector = {
    .cmp_info = {
	/* default component information (unspecified values are initialized to 0) */
	.num_mpx_cntrs =	PAPI_MPX_DEF_DEG,
	.default_domain =	PAPI_DOM_USER,
	.available_domains =	PAPI_DOM_USER|PAPI_DOM_KERNEL,
	.default_granularity =	PAPI_GRN_THR,
	.available_granularities = PAPI_GRN_THR,
	.hardware_intr_sig =	PAPI_SIGNAL,

	/* component specific cmp_info initializations */
	.fast_real_timer =	1,
	.fast_virtual_timer =	1,
	.attach =		1,
	.attach_must_ptrace =	1,
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
	.context =		sizeof(_p4_perfctr_context_t),
	.control_state =	sizeof(_p4_perfctr_control_t),
	.reg_value =		sizeof(_p4_register_t),
	.reg_alloc =		sizeof(_p4_reg_alloc_t),
    },

    /* function pointers in this component */
    .init_control_state =	_p4_init_control_state,
    .start =			_p4_start,
    .stop =			_p4_stop,
    .read =			_p4_read,
    .shutdown =			_p4_shutdown,
    .ctl =			_p4_ctl,
    .bpt_map_set =		_p4_bpt_map_set,
    .bpt_map_avail =		_p4_bpt_map_avail,
    .bpt_map_exclusive =	_p4_bpt_map_exclusive,
    .bpt_map_shared =		_p4_bpt_map_shared,
    .bpt_map_preempt =		_p4_bpt_map_preempt,
    .bpt_map_update =		_p4_bpt_map_update,
    .allocate_registers =	_p4_allocate_registers,
    .update_control_state =	_p4_update_control_state,
    .set_domain =		_p4_set_domain,
    .reset =			_p4_reset,
    .set_overflow =		_p4_set_overflow,
    .stop_profiling =		_p4_stop_profiling,
    .ntv_enum_events =		_p4_ntv_enum_events,
    .ntv_code_to_name =		_p4_ntv_code_to_name,
    .ntv_code_to_descr =	_p4_ntv_code_to_descr,
    .ntv_code_to_bits =		_p4_ntv_code_to_bits,
    .ntv_bits_to_info =		_p4_ntv_bits_to_info,
    .init_substrate =		_p4_init_substrate,
    .dispatch_timer =		_p4_dispatch_timer,
    .get_real_usec =		_p4_get_real_usec,
    .get_real_cycles =		_p4_get_real_cycles,
    .get_virt_cycles =		_p4_get_virt_cycles,
    .get_virt_usec =		_p4_get_virt_usec,

    /* from OS */
    .update_shlib_info =	_linux_update_shlib_info,
    .get_memory_info =		_linux_get_memory_info,
    .get_system_info =		_linux_get_system_info,
    .init =			_linux_init,
    .get_dmem_info =		_linux_get_dmem_info
};
