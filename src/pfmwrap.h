#ifndef _PFMWRAP_H
#define _PFMWRAP_H

#ifdef PFM06A

 #define PFMLIB_SUCCESS 0
 #define PFM_DESTROY_CONTEXT PFM_DISABLE
 typedef pfm_event_config_t pfmw_param_t;
 typedef perfmon_req_t pfmw_reg_t;
 typedef perfmon_reg_t pfmw_arch_reg_t;
 typedef perfmon_req_t pfmw_context_t;
 typedef pme_entry_code_t pfmw_code_t;

 inline int pfmw_find_event(char *v, int r, int *ev) {
     if ( (*ev = pfm_findevent(v, r)) == -1)
	return 1;
     else
  	return PFMLIB_SUCCESS;
 }

 inline int pfmw_set_options(pfmlib_options_t *opt) {
     return pfmlib_config(opt);
 }

 inline int pfmw_perfmonctl(pid_t pid, int cmd, void *arg, int narg) {
     return perfmonctl(pid, cmd, 0, arg, narg);
 }

 inline int pfmw_dispatch_events(pfmw_param_t *p, pfmw_reg_t *pc, int *count) {
     return pfm_dispatch_events(p, pc, count);
 }

 #define PFMW_PEVT_EVTCOUNT(evt)	  (evt->pec_count)
 #define PFMW_PEVT_EVENT(evt,i)    (evt->pec_evt[i])
 #define PFMW_PEVT_PLM(evt,i)      (evt->pec_plm)
 #define PFMW_PEVT_DFLPLM(evt)     (evt->pec_plm)
 #define PFMW_EVT_EVTCOUNT(evt)	  (evt.pec_count)
 #define PFMW_EVT_EVENT(evt,i)     (evt.pec_evt[i])
 #define PFMW_EVT_PLM(evt,i)       (evt.pec_plm)
 #define PFMW_EVT_DFLPLM(evt)      (evt.pec_plm)
 #define PFMW_REG_REGNUM(reg)      (reg.pfr_reg.reg_num)
 #define PFMW_REG_REGVAL(reg)      (reg.pfr_reg.reg_value)
 #define PFMW_REG_SMPLRST(reg)     (reg.pfr_reg.reg_smpl_reset)
 #define PFMW_REG_REGFLAGS(reg)    (reg.pfr_reg.reg_flags)
 #define PFMW_ARCH_REG_REGVAL(reg) (reg.pmu_reg)
 #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_plm)
 #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_es)
 #define PFMW_CTX_NOTIFYPID(ctx)   (ctx.pfr_ctx.notify_pid)
 #define PFMW_CTX_FLAGS(ctx)       (ctx.pfr_ctx.flags)

#elif defined(PFM20)

 typedef pfmlib_param_t pfmw_param_t;
 typedef pfarg_reg_t pfmw_reg_t;
 typedef pfarg_context_t pfmw_context_t;
 typedef pme_ita_code_t pfmw_code_t;

 #ifdef ITANIUM2
  #ifndef PMU_ITA2_MAX_COUNTERS
   #define PMU_ITA2_MAX_COUNTERS PMU_ITA2_NUM_COUNTERS
  #endif
  #ifndef PMU_ITA2_MAX_PMCS
   #define PMU_ITA2_MAX_PMCS PMU_ITA2_NUM_PMCS
  #endif
  #ifndef PMU_ITA2_MAX_PMDS 
   #define PMU_ITA2_MAX_PMDS PMU_ITA2_NUM_PMDS
  #endif
  #ifndef PMU_ITA2_MAX_BTB
   #define PMU_ITA2_MAX_BTB PMU_ITA2_NUM_BTB 
  #endif
  typedef pfm_ita2_reg_t pfmw_arch_reg_t;
 #else /* Itanium */
  #ifndef PMU_ITA_MAX_COUNTERS
   #define PMU_ITA_MAX_COUNTERS PMU_ITA_NUM_COUNTERS
  #endif
  #ifndef PMU_ITA_MAX_PMCS
   #define PMU_ITA_MAX_PMCS PMU_ITA_NUM_PMCS
  #endif
  #ifndef PMU_ITA_MAX_PMDS 
   #define PMU_ITA_MAX_PMDS PMU_ITA_NUM_PMDS
  #endif
  #ifndef PMU_ITA_MAX_BTB
   #define PMU_ITA_MAX_BTB PMU_ITA_NUM_BTB 
  #endif
  typedef pfm_ita_reg_t pfmw_arch_reg_t;
 #endif

 inline int pfmw_find_event(char *v, int r, int *ev) {
     return pfm_find_event(v, ev);
 }

 inline int pfmw_set_options(pfmlib_options_t *opt) {
     return pfm_set_options(opt);
 } 

 inline int pfmw_perfmonctl(pid_t pid, int cmd, void *arg, int narg) {
     return perfmonctl(pid, cmd, arg, narg);
 }

 inline int pfmw_dispatch_events(pfmw_param_t *p, pfmw_reg_t *pc, int *count) {
     int ret;
     memset(p->pfp_pc, 0, sizeof p->pfp_pc);
     p->pfp_pc_count = *count;
     ret = pfm_dispatch_events(p);
     if (ret == PFMLIB_SUCCESS) {
 	memcpy(pc,p->pfp_pc, sizeof p->pfp_pc[0] * p->pfp_pc_count);
 	*count = p->pfp_pc_count;
     }
     return ret;
 }

 #define PFMW_PEVT_EVTCOUNT(evt)   (evt->pfp_event_count)
 #define PFMW_PEVT_EVENT(evt,i)    (evt->pfp_events[i].event)
 #define PFMW_PEVT_PLM(evt,i)      (evt->pfp_events[i].plm) 
 #define PFMW_PEVT_DFLPLM(evt)     (evt->pfp_dfl_plm)
 #define PFMW_EVT_EVTCOUNT(evt)    (evt.pfp_event_count)
 #define PFMW_EVT_EVENT(evt,i)     (evt.pfp_events[i].event)
 #define PFMW_EVT_PLM(evt,i)       (evt.pfp_events[i].plm)
 #define PFMW_EVT_DFLPLM(evt)      (evt.pfp_dfl_plm)
 #define PFMW_REG_REGNUM(reg)      (reg.reg_num)
 #define PFMW_REG_REGVAL(reg)      (reg.reg_value)
 #define PFMW_REG_SMPLRST(reg)     (reg.reg_long_reset)
 #define PFMW_REG_REGFLAGS(reg)    (reg.reg_flags)
 #define PFMW_ARCH_REG_REGVAL(reg) (reg.reg_val)
#ifdef ITANIUM2
 #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita2_count_reg.pmc_plm)
 #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita2_count_reg.pmc_es)
#else
 #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita_count_reg.pmc_plm)
 #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita_count_reg.pmc_es)
#endif
 #define PFMW_CTX_NOTIFYPID(ctx)   (ctx.ctx_notify_pid)
 #define PFMW_CTX_FLAGS(ctx)       (ctx.ctx_flags)


#else
#ifndef PFM11
#warning Maybe you should set -DPFM11 in your Makefile?
#endif
 typedef pfmlib_param_t pfmw_param_t;
 typedef pfarg_reg_t pfmw_reg_t;
 typedef pfarg_context_t pfmw_context_t;
 typedef pme_ita_code_t pfmw_code_t;

 #ifdef ITANIUM2
  typedef pfm_ita2_reg_t pfmw_arch_reg_t;
 #else
  typedef pfm_ita_reg_t pfmw_arch_reg_t;
 #endif  

 inline int pfmw_find_event(char *v, int r, int *ev) {
     return pfm_find_event(v, r, ev);
 }

 inline int pfmw_set_options(pfmlib_options_t *opt) {
     return pfm_set_options(opt);
 }

 inline int pfmw_perfmonctl(pid_t pid, int cmd, void *arg, int narg) {
     return perfmonctl(pid, cmd, arg, narg);
 }

 inline int pfmw_dispatch_events(pfmw_param_t *p, pfmw_reg_t *pc, int *count) {
     return pfm_dispatch_events(p, pc, count);
 }

 #define PFMW_PEVT_EVTCOUNT(evt)   (evt->pfp_count)
 #define PFMW_PEVT_EVENT(evt,i)    (evt->pfp_evt[i])
 #define PFMW_PEVT_PLM(evt,i)      (evt->pfp_plm[i])
 #define PFMW_PEVT_DFLPLM(evt)     (evt->pfp_dfl_plm)
 #define PFMW_EVT_EVTCOUNT(evt)    (evt.pfp_count)
 #define PFMW_EVT_EVENT(evt,i)     (evt.pfp_evt[i])
 #define PFMW_EVT_PLM(evt,i)      (evt.pfp_plm[i])
 #define PFMW_EVT_DFLPLM(evt)      (evt.pfp_dfl_plm)
 #define PFMW_REG_REGNUM(reg)      (reg.reg_num)
 #define PFMW_REG_REGVAL(reg)      (reg.reg_value)
 #define PFMW_REG_SMPLRST(reg)     (reg.reg_long_reset)
 #define PFMW_REG_REGFLAGS(reg)    (reg.reg_flags)
 #define PFMW_ARCH_REG_REGVAL(reg) (reg.reg_val)
 #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_plm)
 #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_es)
 #define PFMW_CTX_NOTIFYPID(ctx)   (ctx.ctx_notify_pid)
 #define PFMW_CTX_FLAGS(ctx)       (ctx.ctx_flags)

#endif

#endif /* _PFMWRAP_H */



