#ifndef _PAPI_PFMWRAP_H
#define _PAPI_PFMWRAP_H

/*
* File:    linux-ia64.c
* CVS:     $Id$
* Author:  Per Ekman
*          pek@pdc.kth.se
* Mods:	   Zhou Min
*          min@cs.utk.edu
*/

#if defined(__INTEL_COMPILER)

#define hweight64(x)    _m64_popcnt(x)

#elif defined(__GNUC__)

static inline int hweight64 (unsigned long x)
{
    unsigned long result;
    __asm__ ("popcnt %0=%1" : "=r" (result) : "r" (x));
    return (int)result;
}

#else
#error "you need to provide inline assembly from your compiler"
#endif

static inline pid_t mygettid(void)
{
#ifdef SYS_gettid
  return(syscall(SYS_gettid));
#elif defined(__NR_gettid)
  return(syscall(__NR_gettid));
#else
  return(syscall(1105));  
#endif
}

#if defined(ITANIUM2) || defined(ITANIUM3)
char *retired_events[]={
	"IA64_TAGGED_INST_RETIRED_IBRP0_PMC8",
	"IA64_TAGGED_INST_RETIRED_IBRP1_PMC9",
	"IA64_TAGGED_INST_RETIRED_IBRP2_PMC8",
	"IA64_TAGGED_INST_RETIRED_IBRP3_PMC9",
	NULL
};
#endif

#if defined(PFM20)
   #define OVFL_SIGNAL SIGPROF
   #define PFMW_PEVT_EVTCOUNT(evt)            (evt->pfp_event_count)
   #define PFMW_PEVT_EVENT(evt,idx)           (evt->pfp_events[idx].event)
   #define PFMW_PEVT_PLM(evt,idx)             (evt->pfp_events[idx].plm)
   #define PFMW_PEVT_DFLPLM(evt)              (evt->pfp_dfl_plm)
   #define PFMW_PEVT_PFPPC(evt)               (evt->pfp_pc)
   #define PFMW_PEVT_PFPPC_COUNT(evt)         (evt->pfp_pc_count)
   #define PFMW_PEVT_PFPPC_REG_NUM(evt,idx)   (evt->pfp_pc[idx].reg_num)
   #define PFMW_PEVT_PFPPC_REG_VAL(evt,idx)   (evt->pfp_pc[idx].reg_value)
   #define PFMW_PEVT_PFPPC_REG_FLG(evt,idx)   (evt->pfp_pc[idx].reg_flags)
   #define PFMW_ARCH_REG_PMCVAL(reg) (reg.reg_val)
   #define PFMW_ARCH_REG_PMDVAL(reg) (reg.reg_val)
   #ifdef ITANIUM2
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita2_count_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita2_count_reg.pmc_es)
   #else /* Itanium */
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita_count_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita_count_reg.pmc_es)
   #endif

   typedef perfmon_smpl_hdr_t    pfmw_smpl_hdr_t;
   typedef perfmon_smpl_entry_t  pfmw_smpl_entry_t;

   #ifdef ITANIUM2
      typedef pfm_ita2_reg_t pfmw_arch_pmc_reg_t;
      typedef pfm_ita2_reg_t pfmw_arch_pmd_reg_t;
   #else /* Itanium */
      typedef pfm_ita_reg_t pfmw_arch_pmc_reg_t;
      typedef pfm_ita_reg_t pfmw_arch_pmd_reg_t;
   #endif

static inline int pfmw_get_num_counters(int *num) {
  int tmp;
  tmp = pfm_get_num_counters();
  if (tmp <= 0)
    return(PAPI_ESYS);
  *num = tmp;
  return(PAPI_OK);
}

static inline int pfmw_get_num_events(int *num) {
  int tmp = 0;

  while (1)
    {
      int newtmp = pfm_get_next_event(tmp);
      if (newtmp == -1)
	break;
      tmp = newtmp;
    }
  *num = tmp;
  return(PAPI_OK);
}

   static inline void pfmw_start(hwd_context_t *ctx) {
      pfm_start();
   }

   static inline void pfmw_stop(hwd_context_t * ctx) {
      pfm_stop();
   }

   inline int pfmw_dispatch_events(pfmw_param_t *evt) {
      int ret;

      ret=pfm_dispatch_events(evt);   
      if (ret) return PAPI_ESYS;
      else return PAPI_OK;
   }
   /* the parameter fd is useless in libpfm2.0 */
   inline int pfmw_perfmonctl(pid_t tid, int fd, int cmd, void *arg, int narg) {
      return(perfmonctl(tid, cmd, arg, narg));
   }

   inline int pfmw_destroy_context(hwd_context_t *thr_ctx) {
      int ret;

      ret=perfmonctl(thr_ctx->tid, PFM_DESTROY_CONTEXT, NULL, 0);
      if (ret) {   
         PAPIERROR("perfmonctl(PFM_DESTROY_CONTEXT) errno %d", errno);
         return PAPI_ESYS;
      } else return PAPI_OK;
   }

   inline int pfmw_create_context( hwd_context_t *thr_ctx) {
      pfarg_context_t ctx[1];

      memset(ctx, 0, sizeof(ctx));
      ctx[0].ctx_notify_pid = mygettid();
      ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

      SUBDBG("PFM_CREATE_CONTEXT\n");
      if (perfmonctl(mygettid(), PFM_CREATE_CONTEXT, ctx, 1) == -1) 
	{
	  PAPIERROR("perfmonctl(PFM_CREATE_CONTEXT) errno %d", errno);
	  return(PAPI_ESYS);
	}

      /*
       * reset PMU (guarantee not active on return) and unfreeze
       * must be done before writing to any PMC/PMD
       */

      if (perfmonctl(mygettid(), PFM_ENABLE, 0, 0) == -1) {
         if (errno == ENOSYS)
	   PAPIERROR("Your kernel does not have performance monitoring support");
	 else
	   PAPIERROR("perfmonctl(PFM_ENABLE) errno %d", errno);
         return(PAPI_ESYS);
      }

      thr_ctx->tid = mygettid();
      return(PAPI_OK);
   }

   inline int pfmw_recreate_context(EventSetInfo_t *ESI, void **smpl_vaddr, 
                               int EventIndex) 
   {
      pfarg_context_t ctx[1];
      int native_index, pos, EventCode;

      pos= ESI->EventInfoArray[EventIndex].pos[0];
      EventCode= ESI->EventInfoArray[EventIndex].event_code;
      native_index= ESI->NativeInfoArray[pos].ni_event & PAPI_NATIVE_AND_MASK;

      memset(ctx, 0, sizeof(ctx));
      ctx[0].ctx_notify_pid = mygettid();
      ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

      ctx[0].ctx_smpl_entries = SMPL_BUF_NENTRIES;


/* DEAR and BTB events */
#ifdef ITANIUM2
      if (pfm_ita2_is_dear(native_index))
         ctx[0].ctx_smpl_regs[0] = DEAR_REGS_MASK;
      else if (pfm_ita2_is_btb(native_index)
               || EventCode == PAPI_BR_INS)
         ctx[0].ctx_smpl_regs[0] = BTB_REGS_MASK;
#else
      if (pfm_ita_is_dear(native_index))
         ctx[0].ctx_smpl_regs[0] = DEAR_REGS_MASK;
      else if (pfm_ita_is_btb(native_index)
               || EventCode == PAPI_BR_INS)
         ctx[0].ctx_smpl_regs[0] = BTB_REGS_MASK;
#endif

      if (pfmw_perfmonctl(mygettid(), 0, PFM_CREATE_CONTEXT, ctx, 1) == -1) {
         PAPIERROR("perfmonctl(PFM_CREATE_CONTEXT) errno %d", errno);
         return (PAPI_ESYS);
      }
      SUBDBG("Sampling buffer mapped at %p\n", ctx[0].ctx_smpl_vaddr);

      *smpl_vaddr = ctx[0].ctx_smpl_vaddr;

      /*
       * reset PMU (guarantee not active on return) and unfreeze
       * must be done before writing to any PMC/PMD
       */
      if (pfmw_perfmonctl(mygettid(), 0, PFM_ENABLE, 0, 0) == -1) 
	{
	  if (errno == ENOSYS) 
	    PAPIERROR("Your kernel does not have performance monitoring support");
	  else
	    PAPIERROR("perfmonctl(PFM_ENABLE) errno %d", errno);
	  return (PAPI_ESYS);
	}
      return(PAPI_OK);
   }
   
   inline int pfmw_get_event_name(char *name, unsigned int idx)
   {
     char *tmp;
     if (pfm_get_event_name(idx, &tmp) == PFMLIB_SUCCESS)
       {
	 strcpy(name,tmp);
	 return PAPI_OK;
       }
     else
       return PAPI_ENOEVNT;
   }

   inline void pfmw_get_event_description(unsigned int idx, char *dest, int len)
   {
      char *descr;

      if (pfm_get_event_description(idx, &descr) == PFMLIB_SUCCESS)
	strncpy(dest,descr,len);
      else
	*dest = '\0';
   }

   inline int pfmw_is_dear(unsigned int i)
   {
   #ifdef ITANIUM2
      return(pfm_ita2_is_dear((int)i));
   #else
      return(pfm_ita_is_dear((int)i));
   #endif
   }

   inline int pfmw_is_iear(unsigned int i)
   {
   #ifdef ITANIUM2
      return(pfm_ita2_is_iear((int)i));
   #else
      return(pfm_ita_is_iear((int)i));
   #endif
   }

   inline int pfmw_support_darr(unsigned int i)
   {
   #ifdef ITANIUM2
      return(pfm_ita2_support_darr((int)i));
   #else
      return(pfm_ita_support_darr((int)i));
   #endif
   }

   inline int pfmw_support_iarr(unsigned int i)
   {
   #ifdef ITANIUM2
      return(pfm_ita2_support_iarr((int)i));
   #else
      return(pfm_ita_support_iarr((int)i));
   #endif
   }

   inline int pfmw_support_opcm(unsigned int i)
   {
   #ifdef ITANIUM2
      return(pfm_ita2_support_opcm((int)i));
   #else
      return(pfm_ita_support_opcm((int)i));
   #endif
   }

static void check_ibrp_events(pfmlib_param_t *evt)
{
	pfmlib_ita2_param_t *param = ITA2_PARAM(evt);
	unsigned long umasks_retired[4];
	unsigned long umask;
	int j, i, seen_retired, ibrp, code, idx;
	int retired_code, incr;
	
	/*
	 * in fine mode, it is enough to use the event
	 * which only monitors the first debug register
	 * pair. The two pairs making up the range
	 * are guaranteed to be consecutive in rr_br[].
	 */
	incr = pfm_ita2_irange_is_fine(evt) ? 4 : 2;

	for (i=0; retired_events[i]; i++) {
		pfm_find_event(retired_events[i], &idx);
		pfm_ita2_get_event_umask(idx, umasks_retired+i);
	}

	pfm_get_event_code(idx, &retired_code);

	/*
	 * print a warning message when the using IA64_TAGGED_INST_RETIRED_IBRP* which does
	 * not completely cover the all the debug register pairs used to make up the range.
	 * This could otherwise lead to misinterpretation of the results.
	 */
	for (i=0; i < param->pfp_ita2_irange.rr_nbr_used; i+= incr) {

		ibrp = param->pfp_ita2_irange.rr_br[i].dbreg_num >>1;

		seen_retired = 0;
		for(j=0; j < evt->pfp_event_count; j++) {
			pfm_get_event_code(evt->pfp_events[j].event, &code);
			if (code != retired_code) continue;
			seen_retired = 1;
			pfm_ita2_get_event_umask(evt->pfp_events[j].event, &umask);
			if (umask == umasks_retired[ibrp]) break;
		}
		if (seen_retired && j == evt->pfp_event_count)
			printf("warning: code range uses IBR pair %d which is not monitored using %s\n", ibrp, retired_events[ibrp]);
	}
}

inline int install_irange(hwd_context_t *pctx, hwd_control_state_t *current_state)
{
#ifdef ITANIUM2
	int  r;
	int pid=pctx->tid;
    pfmw_ita_param_t *param = &(current_state->ita_lib_param);
    pfmw_param_t *evt=&(current_state->evt);
	
	check_ibrp_events(evt);

	r = perfmonctl(pid, PFM_WRITE_IBRS, param->pfp_ita2_irange.rr_br, param->pfp_ita2_irange.rr_nbr_used);
	if (r == -1){
	   SUBDBG("cannot install code range restriction: %s\n", strerror(errno));
       return (PAPI_ESYS);
	}
    return(PAPI_OK);
#else
    return (PAPI_ESBSTR);
#endif
}

static int install_drange(hwd_context_t *pctx, hwd_control_state_t *current_state)
{
#ifdef ITANIUM2
	int r;
	int pid=pctx->tid;
    pfmw_ita_param_t *param = &(current_state->ita_lib_param);
    
	r = perfmonctl(pid, PFM_WRITE_DBRS, param->pfp_ita2_drange.rr_br, param->pfp_ita2_drange.rr_nbr_used);
	if (r == -1){
	   SUBDBG("cannot install data range restriction: %s\n", strerror(errno));
       return (PAPI_ESYS);
	}
    return(PAPI_OK);
#else
    return (PAPI_ESBSTR);
#endif
}

/* The routines set_{d,i}range() provide places to install the data and / or
   instruction address range restrictions for counting qualified events.
   These routines must set up or clear the appropriate local static data structures.
   The actual work of loading the hardware registers must be done in update_ctl_state().
   Both drange and irange can be set on the same eventset.
   If start=end=0, the feature is disabled. 
*/
inline int set_drange(hwd_context_t *ctx, hwd_control_state_t *current_state, _papi_int_option_t *option)
{
   int ret=PAPI_OK;
#ifdef ITANIUM2
   pfmw_ita_param_t *param = &(current_state->ita_lib_param);
   pfmw_param_t *evt=&(current_state->evt);
   
   if((unsigned long)option->address_range.start==(unsigned long)option->address_range.end || ((unsigned long)option->address_range.start==0 && (unsigned long)option->address_range.end==0))
      return(PAPI_EINVAL);
	  /*
	   * set the privilege mode:
	   * 	PFM_PLM3 : user level only
	   */
	  evt->pfp_dfl_plm   = PFM_PLM3; 
      param->pfp_ita2_drange.rr_used = 1;
	  SUBDBG("++++ before data range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
	       "     start_offset:-0x%lx end_offset:+0x%lx\n", 
			param->pfp_ita2_drange.rr_limits[0].rr_start, 
			param->pfp_ita2_drange.rr_limits[0].rr_end, param->pfp_ita2_drange.rr_limits[0].rr_end-param->pfp_ita2_drange.rr_limits[0].rr_start,
			param->pfp_ita2_drange.rr_nbr_used >> 1, 
			param->pfp_ita2_drange.rr_limits[0].rr_soff, 
			param->pfp_ita2_drange.rr_limits[0].rr_eoff);
      param->pfp_ita2_drange.rr_limits[0].rr_start = (unsigned long)option->address_range.start;
      param->pfp_ita2_drange.rr_limits[0].rr_end   = (unsigned long)option->address_range.end;

	  /*
	   * let the library figure out the values for the PMCS
	   */
	  if ((ret=pfm_dispatch_events(evt)) != PFMLIB_SUCCESS) {
		SUBDBG("cannot configure events: %s\n", pfm_strerror(ret));
	  }

	  SUBDBG("++++ data range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
	       "     start_offset:-0x%lx end_offset:+0x%lx\n", 
			param->pfp_ita2_drange.rr_limits[0].rr_start, 
			param->pfp_ita2_drange.rr_limits[0].rr_end, param->pfp_ita2_drange.rr_limits[0].rr_end - param->pfp_ita2_drange.rr_limits[0].rr_start,
			param->pfp_ita2_drange.rr_nbr_used >> 1, 
			param->pfp_ita2_drange.rr_limits[0].rr_soff, 
			param->pfp_ita2_drange.rr_limits[0].rr_eoff);
			
   if((ret=install_drange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=param->pfp_ita2_drange.rr_limits[0].rr_soff;
	  option->address_range.end_off=param->pfp_ita2_drange.rr_limits[0].rr_eoff;
   }
#else
   ret=PAPI_ESBSTR;
#endif
   return(ret);
}

inline int set_irange(hwd_context_t * ctx, hwd_control_state_t * current_state, _papi_int_option_t *option)
{
   int ret=PAPI_OK;
#ifdef ITANIUM2
   pfmw_ita_param_t *param = &(current_state->ita_lib_param);
   pfmw_param_t *evt=&(current_state->evt);

   if((unsigned long)option->address_range.start==(unsigned long)option->address_range.end || ((unsigned long)option->address_range.start==0 && (unsigned long)option->address_range.end==0))
      return(PAPI_EINVAL);
	  
      /*param->pfp_magic = PFMLIB_ITA2_PARAM_MAGIC;*/
	  /*
	   * set the privilege mode:
	   * 	PFM_PLM3 : user level only
	   */
	  evt->pfp_dfl_plm   = PFM_PLM3; 
	  param->pfp_ita2_irange.rr_used = 1;

	  SUBDBG("---- hwd_control_state[0x%x]  before code range  : [0x%016lx-0x%016lx=%ld]\n"
	       "     start_offset:-0x%lx  end_offset:+0x%lx\n"
		   "     %d pairs of debug registers used\n",
			current_state, param->pfp_ita2_irange.rr_limits[0].rr_start, 
			param->pfp_ita2_irange.rr_limits[0].rr_end, param->pfp_ita2_irange.rr_limits[0].rr_end - param->pfp_ita2_irange.rr_limits[0].rr_start,
			param->pfp_ita2_irange.rr_limits[0].rr_soff, 
			param->pfp_ita2_irange.rr_limits[0].rr_eoff,
			param->pfp_ita2_irange.rr_nbr_used >> 1);
 	  
      param->pfp_ita2_irange.rr_limits[0].rr_start = (unsigned long)option->address_range.start;
      param->pfp_ita2_irange.rr_limits[0].rr_end   = (unsigned long)option->address_range.end;

	  /*
	    * let the library figure out the values for the PMCS
	    */
	  if ((ret=pfm_dispatch_events(evt)) != PFMLIB_SUCCESS) {
		SUBDBG("cannot configure events: %s\n", pfm_strerror(ret));
	  }

	  SUBDBG("---- hwd_control_state[0x%x] code range  : [0x%016lx-0x%016lx=%ld]\n"
	       "     start_offset:-0x%lx  end_offset:+0x%lx\n"
		   "     %d pairs of debug registers used\n",
			current_state, param->pfp_ita2_irange.rr_limits[0].rr_start, 
			param->pfp_ita2_irange.rr_limits[0].rr_end, param->pfp_ita2_irange.rr_limits[0].rr_end - param->pfp_ita2_irange.rr_limits[0].rr_start,
			param->pfp_ita2_irange.rr_limits[0].rr_soff, 
			param->pfp_ita2_irange.rr_limits[0].rr_eoff,
			param->pfp_ita2_irange.rr_nbr_used >> 1);
   if((ret=install_irange(ctx, current_state))==PAPI_OK){
	  option->address_range.start_off=param->pfp_ita2_irange.rr_limits[0].rr_soff;
	  option->address_range.end_off=param->pfp_ita2_irange.rr_limits[0].rr_eoff;
   }   
#else
   ret=PAPI_ESBSTR;
#endif
   return ret;
}

#else
#ifndef PFM30
#warning Maybe you should set -DPFM30 in your Makefile?
#endif

   #define OVFL_SIGNAL SIGPROF
   #define PFMW_PEVT_EVTCOUNT(evt)          (evt->inp.pfp_event_count)
   #define PFMW_PEVT_EVENT(evt,idx)         (evt->inp.pfp_events[idx].event)
   #define PFMW_PEVT_PLM(evt,idx)           (evt->inp.pfp_events[idx].plm)
   #define PFMW_PEVT_DFLPLM(evt)            (evt->inp.pfp_dfl_plm)
   #define PFMW_PEVT_PFPPC(evt)             (evt->pc)
   #define PFMW_PEVT_PFPPD(evt)             (evt->pd)
   #define PFMW_PEVT_PFPPC_COUNT(evt)       (evt->outp.pfp_pmc_count)
   #define PFMW_PEVT_PFPPC_REG_NUM(evt,idx) (evt->outp.pfp_pmcs[idx].reg_num)
   #define PFMW_PEVT_PFPPC_REG_VAL(evt,idx) (evt->pc[idx].reg_value)
   #define PFMW_PEVT_PFPPC_REG_FLG(evt,idx) (evt->pc[idx].reg_flags)
   #define PFMW_ARCH_REG_PMCVAL(reg) (reg.pmc_val)
   #define PFMW_ARCH_REG_PMDVAL(reg) (reg.pmd_val)
   #if defined(ITANIUM3)
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_mont_counter_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_mont_counter_reg.pmc_es)
      #define PFMON_MONT_MAX_IBRS	8
      #define PFMON_MONT_MAX_DBRS	8
   #elif defined(ITANIUM2)
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita2_counter_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita2_counter_reg.pmc_es)
      #define PFMON_ITA2_MAX_IBRS	8
      #define PFMON_ITA2_MAX_DBRS	8
   #else /* Itanium */
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita_count_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita_count_reg.pmc_es)
   #endif

   typedef pfm_default_smpl_hdr_t    pfmw_smpl_hdr_t;
   typedef pfm_default_smpl_entry_t  pfmw_smpl_entry_t;

   #if defined(ITANIUM3)
      typedef pfm_mont_pmc_reg_t pfmw_arch_pmc_reg_t;
      typedef pfm_mont_pmd_reg_t pfmw_arch_pmd_reg_t;
   #elif defined(ITANIUM2)
      typedef pfm_ita2_pmc_reg_t pfmw_arch_pmc_reg_t;
      typedef pfm_ita2_pmd_reg_t pfmw_arch_pmd_reg_t;
   #else /* Itanium */
      typedef pfm_ita_pmc_reg_t pfmw_arch_pmc_reg_t;
      typedef pfm_ita_pmd_reg_t pfmw_arch_pmd_reg_t;
   #endif

   static inline void pfmw_start(hwd_context_t * ctx) {
      pfm_self_start(ctx->fd);
   }

   static inline void pfmw_stop(hwd_context_t * ctx) {
      pfm_self_stop(ctx->fd);
   }

   inline int pfmw_perfmonctl(pid_t tid, int fd , int cmd, void *arg, int narg) {
      return(perfmonctl(fd, cmd, arg, narg));
   }

   inline int pfmw_destroy_context(hwd_context_t *thr_ctx) {
      int ret;
      ret=close(thr_ctx->fd);
      if (ret) return PAPI_ESYS;
      else return PAPI_OK;
   }

   inline int pfmw_dispatch_events(pfmw_param_t *evt) {
      int ret,i;
/*
      PFMW_PEVT_DFLPLM(evt) = PFM_PLM3;
*/
   #if defined(ITANIUM3)
      ret=pfm_dispatch_events(&evt->inp, (pfmlib_mont_input_param_t *)evt->mod_inp, &evt->outp, (pfmlib_mont_output_param_t *)evt->mod_outp);   
   #else
      ret=pfm_dispatch_events(&evt->inp, (pfmlib_ita2_input_param_t *)evt->mod_inp, &evt->outp, (pfmlib_ita2_output_param_t *)evt->mod_outp);   
   #endif
      if (ret) { 
         return PAPI_ESYS;
      } else {
         for (i=0; i < evt->outp.pfp_pmc_count; i++) {
            evt->pc[i].reg_num   = evt->outp.pfp_pmcs[i].reg_num;
            evt->pc[i].reg_value = evt->outp.pfp_pmcs[i].reg_value;
         }
#if defined(HAVE_PFMLIB_OUTPUT_PFP_PMD_COUNT)
         for (i=0; i < evt->outp.pfp_pmd_count; i++) {
	   evt->pd[i].reg_num   = evt->outp.pfp_pmds[i].reg_num;
	 }
#else
	 /* This is really broken */
         for (i=0; i < evt->inp.pfp_event_count; i++) {
	   evt->pd[i].reg_num   = evt->pc[i].reg_num;
	 }
#endif
         return PAPI_OK;
      }
   }

   inline int pfmw_create_ctx_common(hwd_context_t *ctx) 
   {
      pfarg_load_t load_args;
      int ret;

      memset(&load_args, 0, sizeof(load_args));
      /*
       * we want to monitor ourself
       */

      load_args.load_pid = ctx->tid;

      SUBDBG("PFM_LOAD_CONTEXT FD %d, PID %d\n",ctx->fd,ctx->tid);
      if (perfmonctl(ctx->fd, PFM_LOAD_CONTEXT, &load_args, 1) == -1) {
         PAPIERROR("perfmonctl(PFM_LOAD_CONTEXT) errno %d",errno);
         return(PAPI_ESYS);
      }
      /*
       * setup asynchronous notification on the file descriptor
       */
      ret = fcntl(ctx->fd, F_SETFL, fcntl(ctx->fd, F_GETFL, 0) | O_ASYNC);
      if (ret == -1) {
         PAPIERROR("fcntl(%d,F_SETFL,O_ASYNC) errno %d", ctx->fd, errno);
         return(PAPI_ESYS);
      }

      /*
       * get ownership of the descriptor
       */

      ret = fcntl(ctx->fd, F_SETOWN, ctx->tid);
      if (ret == -1) {
         PAPIERROR("fcntl(%d,F_SETOWN) errno %d", ctx->fd, errno);
         return(PAPI_ESYS);
      }

      ret = fcntl(ctx->fd, F_SETSIG, _papi_hwi_system_info.sub_info.hardware_intr_sig);
      if (ret == -1) {
         PAPIERROR("fcntl(%d,F_SETSIG) errno %d", ctx->fd, errno);
        return(PAPI_ESYS);
      }

      /* set close-on-exec to ensure we will be getting the PFM_END_MSG, i.e.,
       * fd not visible to child. */

      ret = fcntl(ctx->fd, F_SETFD, FD_CLOEXEC);
      if (ret == -1) {
	PAPIERROR("fcntl(%d,FD_CLOEXEC) errno %d", ctx->fd, errno);
	return(PAPI_ESYS);
      }

      return(PAPI_OK);

   }

   inline int pfmw_create_context(hwd_context_t *thr_ctx) {
      pfarg_context_t ctx[1];
      memset(ctx, 0, sizeof(ctx));

      SUBDBG("PFM_CREATE_CONTEXT on 0\n");
      if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1) 
	{
	  PAPIERROR("perfmonctl(PFM_CREATE_CONTEXT) errno %d", errno);
	  return(PAPI_ESYS);
      }
      thr_ctx->fd = ctx[0].ctx_fd;
      thr_ctx->tid = mygettid();
      SUBDBG("PFM_CREATE_CONTEXT returns FD %d, TID %d\n",(int)thr_ctx->fd,(int)thr_ctx->tid);

      return(pfmw_create_ctx_common(thr_ctx)); 
   }

   inline int set_pmds_to_write(EventSetInfo_t * ESI, unsigned int index, long value)
   {
      int *pos, count, hwcntr, i;
      hwd_control_state_t *this_state = (hwd_control_state_t *)&ESI->machdep;
      pfmw_param_t *pevt= &(this_state->evt);

      pos = ESI->EventInfoArray[index].pos;
      count = 0;
      while (pos[count] != -1 && count < MAX_COUNTERS) {
         hwcntr = pos[count] + PMU_FIRST_COUNTER;
         for (i = 0; i < MAX_COUNTERS; i++) {
            if ( PFMW_PEVT_PFPPC_REG_NUM(pevt,i) == hwcntr) {
               this_state->evt.pc[i].reg_smpl_pmds[0] = value;
               break;
            }
         }
         count++;
      }
      return (PAPI_OK);
   }
   
extern inline int _pfm_decode_native_event(unsigned int EventCode, unsigned int *event, unsigned int *umask);

   inline int pfmw_recreate_context(EventSetInfo_t * ESI, void **smpl_vaddr, 
                               int EventIndex) 
   {
      pfm_default_smpl_ctx_arg_t ctx[1];
      pfm_uuid_t buf_fmt_id = PFM_DEFAULT_SMPL_UUID;
      int ctx_fd;
      unsigned int native_index, EventCode; 
      int pos;
      hwd_context_t *thr_ctx = (hwd_context_t *) &ESI->master->context;
#if defined(ITANIUM3)
      unsigned int umask;
#endif

      pos= ESI->EventInfoArray[EventIndex].pos[0];
      EventCode= ESI->EventInfoArray[EventIndex].event_code;
#if defined(ITANIUM3)
      if (_pfm_decode_native_event(ESI->NativeInfoArray[pos].ni_event,&native_index,&umask) != PAPI_OK)
         return(PAPI_ENOEVNT);
#else
      native_index= ESI->NativeInfoArray[pos].ni_event & PAPI_NATIVE_AND_MASK;
#endif
      memset(ctx,0,sizeof(ctx[0]));
      /*
       * We initialize the format specific information.
       * The format is identified by its UUID which must be copied
       * into the ctx_buf_fmt_id field.
       */
      memcpy(ctx[0].ctx_arg.ctx_smpl_buf_id, buf_fmt_id, sizeof(pfm_uuid_t));
      /*
       * the size of the buffer is indicated in bytes (not entries).
       * The kernel will record into the buffer up to a certain point.
       * No partial samples are ever recorded.
       */
      ctx[0].buf_arg.buf_size = 4096;
      /*
       * now create the context for self monitoring/per-task
       */
      SUBDBG("PFM_CREATE_CONTEXT on 0\n");
      if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
         if (errno == ENOSYS) 
	   PAPIERROR("Your kernel does not have performance monitoring support");
	 else
	   PAPIERROR("perfmonctl(PFM_CREATE_CONTEXT) errno %d", errno);
         return(PAPI_ESYS);
      }
      /*
       * extract the file descriptor we will use to
       * identify this newly created context
       */
      ctx_fd = ctx[0].ctx_arg.ctx_fd;
      /* save the fd into the thread context struct */
      thr_ctx->fd = ctx_fd;
      thr_ctx->tid = mygettid();
      SUBDBG("PFM_CREATE_CONTEXT returns FD %d, TID %d\n",(int)thr_ctx->fd,(int)thr_ctx->tid);
      /* indicate which PMD to include in the sample */
/* DEAR and BTB events */
#if defined(ITANIUM3)
      if (pfm_mont_is_dear(native_index))
         set_pmds_to_write(ESI, EventIndex, DEAR_REGS_MASK);
      else if (pfm_mont_is_etb(native_index)
               || EventCode == PAPI_BR_INS)
         set_pmds_to_write(ESI, EventIndex, ETB_REGS_MASK);
#elif defined(ITANIUM2)
      if (pfm_ita2_is_dear(native_index))
         set_pmds_to_write(ESI, EventIndex, DEAR_REGS_MASK);
      else if (pfm_ita2_is_btb(native_index)
               || EventCode == PAPI_BR_INS)
         set_pmds_to_write(ESI, EventIndex, BTB_REGS_MASK);
#else
      if (pfm_ita_is_dear(native_index))
         set_pmds_to_write(ESI, EventIndex, DEAR_REGS_MASK);
      else if (pfm_ita_is_btb(native_index)
               || EventCode == PAPI_BR_INS)
         set_pmds_to_write(ESI, EventIndex, BTB_REGS_MASK);
#endif

      *smpl_vaddr = ctx[0].ctx_arg.ctx_smpl_vaddr;

      return(pfmw_create_ctx_common(thr_ctx)); 
   }

   inline int pfmw_get_event_name(char *name, unsigned int idx)
   {
      unsigned int total;

      pfm_get_num_events(&total);
      if(idx >= total)
         return PAPI_ENOEVNT;
      if (pfm_get_event_name(idx, name, PAPI_MAX_STR_LEN) == PFMLIB_SUCCESS)
         return PAPI_OK;
      else
         return PAPI_ENOEVNT;
   }

   inline void pfmw_get_event_description(unsigned int idx, char *dest, int len)
   {
      char *descr;

      if (pfm_get_event_description(idx, &descr) == PFMLIB_SUCCESS)
	{
	  strncpy(dest,descr,len);
	  free(descr);
	}
      else
	*dest = '\0';
   }

   inline int pfmw_is_dear(unsigned int i)
   {
   #if defined(ITANIUM3)
      return(pfm_mont_is_dear(i));
   #elif defined(ITANIUM2)
      return(pfm_ita2_is_dear(i));
   #else
      return(pfm_ita_is_dear(i));
   #endif
   }

   inline int pfmw_is_iear(unsigned int i)
   {
   #if defined(ITANIUM3)
      return(pfm_mont_is_iear(i));
   #elif defined(ITANIUM2)
      return(pfm_ita2_is_iear(i));
   #else
      return(pfm_ita_is_iear(i));
   #endif
   }

   inline int pfmw_support_darr(unsigned int i)
   {
   #if defined(ITANIUM3)
      return(pfm_mont_support_darr(i));
   #elif defined(ITANIUM2)
      return(pfm_ita2_support_darr(i));
   #else
      return(pfm_ita_support_darr(i));
   #endif
   }

   inline int pfmw_support_iarr(unsigned int i)
   {
   #if defined(ITANIUM3)
      return(pfm_mont_support_iarr(i));
   #elif defined(ITANIUM2)
      return(pfm_ita2_support_iarr(i));
   #else
      return(pfm_ita_support_iarr(i));
   #endif
   }

   inline int pfmw_support_opcm(unsigned int i)
   {
   #if defined(ITANIUM3)
      return(pfm_mont_support_opcm(i));
   #elif defined(ITANIUM2)
      return(pfm_ita2_support_opcm(i));
   #else
      return(pfm_ita_support_opcm(i));
   #endif
   }

static void check_ibrp_events(hwd_control_state_t *current_state)
{
#if defined(ITANIUM3)
    pfmw_param_t *evt = &(current_state->evt);
	pfmlib_mont_output_param_t *param = &(current_state->ita_lib_param.mont_output_param);
	unsigned long umasks_retired[4];
	unsigned long umask;
	unsigned int j, i, seen_retired, ibrp, idx;
	int code;
	int retired_code, incr;
	
	/*
	 * in fine mode, it is enough to use the event
	 * which only monitors the first debug register
	 * pair. The two pairs making up the range
	 * are guaranteed to be consecutive in rr_br[].
	 */
	incr = pfm_mont_irange_is_fine(&evt->outp, param) ? 4 : 2;

	for (i=0; retired_events[i]; i++) {
		pfm_find_event(retired_events[i], &idx);
		pfm_mont_get_event_umask(idx, umasks_retired+i);
	}

	pfm_get_event_code(idx, &retired_code);

	/*
	 * print a warning message when the using IA64_TAGGED_INST_RETIRED_IBRP* which does
	 * not completely cover the all the debug register pairs used to make up the range.
	 * This could otherwise lead to misinterpretation of the results.
	 */
	for (i=0; i < param->pfp_mont_irange.rr_nbr_used; i+= incr) {

		ibrp = param->pfp_mont_irange.rr_br[i].reg_num >>1;

		seen_retired = 0;
		for(j=0; j < evt->inp.pfp_event_count; j++) {
			pfm_get_event_code(evt->inp.pfp_events[j].event, &code);
			if (code != retired_code) continue;
			seen_retired = 1;
			pfm_mont_get_event_umask(evt->inp.pfp_events[j].event, &umask);
			if (umask == umasks_retired[ibrp]) break;
		}
		if (seen_retired && j == evt->inp.pfp_event_count)
			printf("warning: code range uses IBR pair %d which is not monitored using %s\n", ibrp, retired_events[ibrp]);
	}
#elif defined(ITANIUM2)
    pfmw_param_t *evt = &(current_state->evt);
	pfmlib_ita2_output_param_t *param = &(current_state->ita_lib_param.ita2_output_param);
	unsigned long umasks_retired[4];
	unsigned long umask;
	unsigned int j, i, seen_retired, ibrp, idx;
	int code;
	int retired_code, incr;
	
	/*
	 * in fine mode, it is enough to use the event
	 * which only monitors the first debug register
	 * pair. The two pairs making up the range
	 * are guaranteed to be consecutive in rr_br[].
	 */
	incr = pfm_ita2_irange_is_fine(&evt->outp, param) ? 4 : 2;

	for (i=0; retired_events[i]; i++) {
		pfm_find_event(retired_events[i], &idx);
		pfm_ita2_get_event_umask(idx, umasks_retired+i);
	}

	pfm_get_event_code(idx, &retired_code);

	/*
	 * print a warning message when the using IA64_TAGGED_INST_RETIRED_IBRP* which does
	 * not completely cover the all the debug register pairs used to make up the range.
	 * This could otherwise lead to misinterpretation of the results.
	 */
	for (i=0; i < param->pfp_ita2_irange.rr_nbr_used; i+= incr) {

		ibrp = param->pfp_ita2_irange.rr_br[i].reg_num >>1;

		seen_retired = 0;
		for(j=0; j < evt->inp.pfp_event_count; j++) {
			pfm_get_event_code(evt->inp.pfp_events[j].event, &code);
			if (code != retired_code) continue;
			seen_retired = 1;
			pfm_ita2_get_event_umask(evt->inp.pfp_events[j].event, &umask);
			if (umask == umasks_retired[ibrp]) break;
		}
		if (seen_retired && j == evt->inp.pfp_event_count)
			printf("warning: code range uses IBR pair %d which is not monitored using %s\n", ibrp, retired_events[ibrp]);
	}
#endif
}

inline int install_irange(hwd_context_t *pctx, hwd_control_state_t *current_state)
{
#if defined(ITANIUM3)
	pfmlib_mont_output_param_t *param = (pfmlib_mont_output_param_t	*)&(current_state->ita_lib_param.mont_output_param);
	pfarg_dbreg_t dbreg[PFMON_MONT_MAX_IBRS];
	unsigned int i, used_dbr;
	int  r;
	int pid=pctx->fd;

	memset(dbreg, 0, sizeof(dbreg));
	check_ibrp_events(current_state);

	used_dbr = param->pfp_mont_irange.rr_nbr_used;

	for(i=0; i < used_dbr; i++) {
		dbreg[i].dbreg_num   = param->pfp_mont_irange.rr_br[i].reg_num; 
		dbreg[i].dbreg_value = param->pfp_mont_irange.rr_br[i].reg_value; 
	}

	r = perfmonctl(pid, PFM_WRITE_IBRS, dbreg, param->pfp_mont_irange.rr_nbr_used);
	if (r == -1){
	   SUBDBG("cannot install code range restriction: %s\n", strerror(errno));
       return (PAPI_ESYS);
	}
	return (PAPI_OK);
#elif defined(ITANIUM2)
	pfmlib_ita2_output_param_t *param = (pfmlib_ita2_output_param_t *)&(current_state->ita_lib_param.ita2_output_param);
	pfarg_dbreg_t dbreg[PFMON_ITA2_MAX_IBRS];
	unsigned int i, used_dbr;
	int  r;
	int pid=pctx->fd;

	memset(dbreg, 0, sizeof(dbreg));
	check_ibrp_events(current_state);

	used_dbr = param->pfp_ita2_irange.rr_nbr_used;

	for(i=0; i < used_dbr; i++) {
		dbreg[i].dbreg_num   = param->pfp_ita2_irange.rr_br[i].reg_num; 
		dbreg[i].dbreg_value = param->pfp_ita2_irange.rr_br[i].reg_value; 
	}

	r = perfmonctl(pid, PFM_WRITE_IBRS, dbreg, param->pfp_ita2_irange.rr_nbr_used);
	if (r == -1){
	   SUBDBG("cannot install code range restriction: %s\n", strerror(errno));
       return (PAPI_ESYS);
	}
	return (PAPI_OK);
#else
    return (PAPI_ESBSTR);
#endif
}

inline int install_drange(hwd_context_t *pctx, hwd_control_state_t *current_state)
{
#if defined(ITANIUM3)
	pfmlib_mont_output_param_t *param = (pfmlib_mont_output_param_t	*)&(current_state->ita_lib_param.mont_output_param);
	pfarg_dbreg_t dbreg[PFMON_MONT_MAX_DBRS];
	unsigned int i, used_dbr;
	int r;
	int pid=pctx->fd;

	memset(dbreg, 0, sizeof(dbreg));
	used_dbr = param->pfp_mont_drange.rr_nbr_used;

	for(i=0; i < used_dbr; i++) {
		dbreg[i].dbreg_num   = param->pfp_mont_drange.rr_br[i].reg_num; 
		dbreg[i].dbreg_value = param->pfp_mont_drange.rr_br[i].reg_value; 
	}

	r = perfmonctl(pid, PFM_WRITE_DBRS, dbreg, param->pfp_mont_drange.rr_nbr_used);
	if (r == -1){
	   SUBDBG("cannot install data range restriction: %s\n", strerror(errno));
       return (PAPI_ESYS);
	}
	return (PAPI_OK);
#elif defined(ITANIUM2)
	pfmlib_ita2_output_param_t *param = (pfmlib_ita2_output_param_t *)&(current_state->ita_lib_param.ita2_output_param);
	pfarg_dbreg_t dbreg[PFMON_ITA2_MAX_DBRS];
	unsigned int i, used_dbr;
	int r;
	int pid=pctx->fd;

	memset(dbreg, 0, sizeof(dbreg));
	used_dbr = param->pfp_ita2_drange.rr_nbr_used;

	for(i=0; i < used_dbr; i++) {
		dbreg[i].dbreg_num   = param->pfp_ita2_drange.rr_br[i].reg_num; 
		dbreg[i].dbreg_value = param->pfp_ita2_drange.rr_br[i].reg_value; 
	}

	r = perfmonctl(pid, PFM_WRITE_DBRS, dbreg, param->pfp_ita2_drange.rr_nbr_used);
	if (r == -1){
	   SUBDBG("cannot install data range restriction: %s\n", strerror(errno));
       return (PAPI_ESYS);
	}
	return (PAPI_OK);
#else
    return (PAPI_ESBSTR);
#endif
}

/* The routines set_{d,i}range() provide places to install the data and / or
   instruction address range restrictions for counting qualified events.
   These routines must set up or clear the appropriate local static data structures.
   The actual work of loading the hardware registers must be done in update_ctl_state().
   Both drange and irange can be set on the same eventset.
   If start=end=0, the feature is disabled. 
*/
inline int set_drange(hwd_context_t *ctx, hwd_control_state_t *current_state, _papi_int_option_t *option)
{
   int ret=PAPI_OK;
#if defined(ITANIUM3)
   pfmw_ita_param_t *param = &(current_state->ita_lib_param);
   pfmw_param_t *evt = &(current_state->evt);
   pfmlib_mont_input_param_t *mont_inp = &(param->mont_input_param);
   pfmlib_mont_output_param_t *mont_outp = &(param->mont_output_param);
   pfmlib_input_param_t *inp = &evt->inp;

   if((unsigned long)option->address_range.start==(unsigned long)option->address_range.end || ((unsigned long)option->address_range.start==0 && (unsigned long)option->address_range.end==0))
      return(PAPI_EINVAL);
	  /*
	   * set the privilege mode:
	   * 	PFM_PLM3 : user level only
	   */
   memset(&mont_inp->pfp_mont_drange, 0, sizeof(pfmlib_mont_input_rr_t));
   memset(mont_outp, 0, sizeof(pfmlib_mont_output_param_t));
   inp->pfp_dfl_plm   = PFM_PLM3; 
   mont_inp->pfp_mont_drange.rr_used = 1;
   mont_inp->pfp_mont_drange.rr_limits[0].rr_start = (unsigned long)option->address_range.start;
   mont_inp->pfp_mont_drange.rr_limits[0].rr_end   = (unsigned long)option->address_range.end;
   SUBDBG("++++ before data range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
          "     start_offset:-0x%lx end_offset:+0x%lx\n", 
          mont_inp->pfp_mont_drange.rr_limits[0].rr_start, 
          mont_inp->pfp_mont_drange.rr_limits[0].rr_end,
          mont_inp->pfp_mont_drange.rr_limits[0].rr_end-mont_inp->pfp_mont_drange.rr_limits[0].rr_start,
          mont_outp->pfp_mont_drange.rr_nbr_used >> 1, 
          mont_outp->pfp_mont_drange.rr_infos[0].rr_soff, 
          mont_outp->pfp_mont_drange.rr_infos[0].rr_eoff);

	  /*
	   * let the library figure out the values for the PMCS
	   */
	  if ((ret=pfmw_dispatch_events(evt)) != PFMLIB_SUCCESS) {
		SUBDBG("cannot configure events: %s\n", pfm_strerror(ret));
	  }

	  SUBDBG("++++ data range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
                 "     start_offset:-0x%lx end_offset:+0x%lx\n", 
                 mont_inp->pfp_mont_drange.rr_limits[0].rr_start, 
                 mont_inp->pfp_mont_drange.rr_limits[0].rr_end,
                 mont_inp->pfp_mont_drange.rr_limits[0].rr_end-mont_inp->pfp_mont_drange.rr_limits[0].rr_start,
                 mont_outp->pfp_mont_drange.rr_nbr_used >> 1, 
                 mont_outp->pfp_mont_drange.rr_infos[0].rr_soff, 
                 mont_outp->pfp_mont_drange.rr_infos[0].rr_eoff);
			
/*   if(	ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start!=0 || 	ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end!=0 )
   if((ret=install_irange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff;
	  option->address_range.end_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff;
   }
*/
   if((ret=install_drange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=mont_outp->pfp_mont_drange.rr_infos[0].rr_soff;
	  option->address_range.end_off=mont_outp->pfp_mont_drange.rr_infos[0].rr_eoff;
   }
   return(ret);
#elif defined(ITANIUM2)
   pfmw_ita_param_t *param = &(current_state->ita_lib_param);
   pfmw_param_t *evt = &(current_state->evt);
   pfmlib_ita2_input_param_t *ita2_inp = &(param->ita2_input_param);
   pfmlib_ita2_output_param_t *ita2_outp = &(param->ita2_output_param);
   pfmlib_input_param_t *inp = &evt->inp;

   if((unsigned long)option->address_range.start==(unsigned long)option->address_range.end || ((unsigned long)option->address_range.start==0 && (unsigned long)option->address_range.end==0))
      return(PAPI_EINVAL);
	  /*
	   * set the privilege mode:
	   * 	PFM_PLM3 : user level only
	   */
	  memset(&ita2_inp->pfp_ita2_drange, 0, sizeof(pfmlib_ita2_input_rr_t));
	  memset(ita2_outp, 0, sizeof(pfmlib_ita2_output_param_t));
	  inp->pfp_dfl_plm   = PFM_PLM3; 
      ita2_inp->pfp_ita2_drange.rr_used = 1;
      ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start = (unsigned long)option->address_range.start;
      ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end   = (unsigned long)option->address_range.end;
	  SUBDBG("++++ before data range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
	       "     start_offset:-0x%lx end_offset:+0x%lx\n", 
			ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start, 
			ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end, ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end-ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start,
			ita2_outp->pfp_ita2_drange.rr_nbr_used >> 1, 
			ita2_outp->pfp_ita2_drange.rr_infos[0].rr_soff, 
			ita2_outp->pfp_ita2_drange.rr_infos[0].rr_eoff);

	  /*
	   * let the library figure out the values for the PMCS
	   */
	  if ((ret=pfmw_dispatch_events(evt)) != PFMLIB_SUCCESS) {
		SUBDBG("cannot configure events: %s\n", pfm_strerror(ret));
	  }

	  SUBDBG("++++ data range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
	       "     start_offset:-0x%lx end_offset:+0x%lx\n", 
			ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start, 
			ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end, ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end-ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start,
			ita2_outp->pfp_ita2_drange.rr_nbr_used >> 1, 
			ita2_outp->pfp_ita2_drange.rr_infos[0].rr_soff, 
			ita2_outp->pfp_ita2_drange.rr_infos[0].rr_eoff);
			
/*   if(	ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start!=0 || 	ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end!=0 )
   if((ret=install_irange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff;
	  option->address_range.end_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff;
   }
*/
   if((ret=install_drange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=ita2_outp->pfp_ita2_drange.rr_infos[0].rr_soff;
	  option->address_range.end_off=ita2_outp->pfp_ita2_drange.rr_infos[0].rr_eoff;
   }
   return(ret);
#else
   ret=PAPI_ESBSTR;
#endif   
}

inline int set_irange(hwd_context_t * ctx, hwd_control_state_t * current_state, _papi_int_option_t *option)
{
   int ret=PAPI_OK;
#if defined(ITANIUM3)
   pfmw_ita_param_t *param = &(current_state->ita_lib_param);
   pfmw_param_t *evt = &(current_state->evt);
   pfmlib_mont_input_param_t *mont_inp = &(param->mont_input_param);
   pfmlib_mont_output_param_t *mont_outp = &(param->mont_output_param);
   pfmlib_input_param_t *inp = &evt->inp;

   if((unsigned long)option->address_range.start==(unsigned long)option->address_range.end || ((unsigned long)option->address_range.start==0 && (unsigned long)option->address_range.end==0))
      return(PAPI_EINVAL);
	  /*
	   * set the privilege mode:
	   * 	PFM_PLM3 : user level only
	   */
	  memset(&mont_inp->pfp_mont_irange, 0, sizeof(pfmlib_mont_input_rr_t));
	  memset(mont_outp, 0, sizeof(pfmlib_mont_output_param_t));
	  inp->pfp_dfl_plm   = PFM_PLM3; 
      mont_inp->pfp_mont_irange.rr_used = 1;
      mont_inp->pfp_mont_irange.rr_limits[0].rr_start = (unsigned long)option->address_range.start;
      mont_inp->pfp_mont_irange.rr_limits[0].rr_end   = (unsigned long)option->address_range.end;
	  SUBDBG("++++ before code range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
	       "     start_offset:-0x%lx end_offset:+0x%lx\n", 
			mont_inp->pfp_mont_irange.rr_limits[0].rr_start, 
			mont_inp->pfp_mont_irange.rr_limits[0].rr_end, mont_inp->pfp_mont_irange.rr_limits[0].rr_end-mont_inp->pfp_mont_irange.rr_limits[0].rr_start,
			mont_outp->pfp_mont_irange.rr_nbr_used >> 1, 
			mont_outp->pfp_mont_irange.rr_infos[0].rr_soff, 
			mont_outp->pfp_mont_irange.rr_infos[0].rr_eoff);

	  /*
	   * let the library figure out the values for the PMCS
	   */
	  if ((ret=pfmw_dispatch_events(evt)) != PFMLIB_SUCCESS) {
		SUBDBG("cannot configure events: %s\n", pfm_strerror(ret));
	  }

	  SUBDBG("++++ code range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
	       "     start_offset:-0x%lx end_offset:+0x%lx\n", 
			mont_inp->pfp_mont_irange.rr_limits[0].rr_start, 
			mont_inp->pfp_mont_irange.rr_limits[0].rr_end, mont_inp->pfp_mont_irange.rr_limits[0].rr_end-mont_inp->pfp_mont_irange.rr_limits[0].rr_start,
			mont_outp->pfp_mont_irange.rr_nbr_used >> 1, 
			mont_outp->pfp_mont_irange.rr_infos[0].rr_soff, 
			mont_outp->pfp_mont_irange.rr_infos[0].rr_eoff);
/*   if(	ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start!=0 || 	ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end!=0 )
   if((ret=install_drange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=ita2_outp->pfp_ita2_drange.rr_infos[0].rr_soff;
	  option->address_range.end_off=ita2_outp->pfp_ita2_drange.rr_infos[0].rr_eoff;
   }
*/   
   if((ret=install_irange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=mont_outp->pfp_mont_irange.rr_infos[0].rr_soff;
	  option->address_range.end_off=mont_outp->pfp_mont_irange.rr_infos[0].rr_eoff;
   }
#elif defined(ITANIUM2)
   pfmw_ita_param_t *param = &(current_state->ita_lib_param);
   pfmw_param_t *evt = &(current_state->evt);
   pfmlib_ita2_input_param_t *ita2_inp = &(param->ita2_input_param);
   pfmlib_ita2_output_param_t *ita2_outp = &(param->ita2_output_param);
   pfmlib_input_param_t *inp = &evt->inp;

   if((unsigned long)option->address_range.start==(unsigned long)option->address_range.end || ((unsigned long)option->address_range.start==0 && (unsigned long)option->address_range.end==0))
      return(PAPI_EINVAL);
	  /*
	   * set the privilege mode:
	   * 	PFM_PLM3 : user level only
	   */
	  memset(&ita2_inp->pfp_ita2_irange, 0, sizeof(pfmlib_ita2_input_rr_t));
	  memset(ita2_outp, 0, sizeof(pfmlib_ita2_output_param_t));
	  inp->pfp_dfl_plm   = PFM_PLM3; 
      ita2_inp->pfp_ita2_irange.rr_used = 1;
      ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start = (unsigned long)option->address_range.start;
      ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end   = (unsigned long)option->address_range.end;
	  SUBDBG("++++ before code range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
	       "     start_offset:-0x%lx end_offset:+0x%lx\n", 
			ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start, 
			ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end, ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end-ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start,
			ita2_outp->pfp_ita2_irange.rr_nbr_used >> 1, 
			ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff, 
			ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff);

	  /*
	   * let the library figure out the values for the PMCS
	   */
	  if ((ret=pfmw_dispatch_events(evt)) != PFMLIB_SUCCESS) {
		SUBDBG("cannot configure events: %s\n", pfm_strerror(ret));
	  }

	  SUBDBG("++++ code range  : [0x%016lx-0x%016lx=%ld]: %d pair of debug registers used\n" 
	       "     start_offset:-0x%lx end_offset:+0x%lx\n", 
			ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start, 
			ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end, ita2_inp->pfp_ita2_irange.rr_limits[0].rr_end-ita2_inp->pfp_ita2_irange.rr_limits[0].rr_start,
			ita2_outp->pfp_ita2_irange.rr_nbr_used >> 1, 
			ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff, 
			ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff);
/*   if(	ita2_inp->pfp_ita2_drange.rr_limits[0].rr_start!=0 || 	ita2_inp->pfp_ita2_drange.rr_limits[0].rr_end!=0 )
   if((ret=install_drange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=ita2_outp->pfp_ita2_drange.rr_infos[0].rr_soff;
	  option->address_range.end_off=ita2_outp->pfp_ita2_drange.rr_infos[0].rr_eoff;
   }
*/   
   if((ret=install_irange(ctx, current_state)) ==PAPI_OK){
	  option->address_range.start_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_soff;
	  option->address_range.end_off=ita2_outp->pfp_ita2_irange.rr_infos[0].rr_eoff;
   }
#else
   ret=PAPI_ESBSTR;
#endif
   return(ret);
}

static inline int pfmw_get_num_counters(int *num) {
  unsigned int tmp;
  if (pfm_get_num_counters(&tmp) != PFMLIB_SUCCESS)
    return(PAPI_ESYS);
  *num = tmp;
  return(PAPI_OK);
}

static inline int pfmw_get_num_events(int *num) {
  unsigned int tmp;
  if (pfm_get_num_events(&tmp) != PFMLIB_SUCCESS)
    return(PAPI_ESYS);
  *num = tmp;
  return(PAPI_OK);
}

#endif

/* These routines are the same in PFM20 and PFM30
   The wrappers hide differences between ITANIUM and ITANIUM2
*/

#endif /* _PFMWRAP_H */



