#ifndef _PFMWRAP_H
#define _PFMWRAP_H

/* copy from itanium_events.h */
 #ifdef ITANIUM2
  #define PME_EVENT_COUNT 475
 #else
  #define PME_EVENT_COUNT 230
 #endif

 typedef pfmlib_param_t pfmw_param_t;
 typedef pfarg_reg_t pfmw_reg_t;
 typedef pfarg_context_t pfmw_context_t;
 typedef pme_entry_code_t pfmw_code_t;

 #ifdef ITANIUM2
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
  typedef pfmlib_ita2_param_t pfmw_ita_param_t;
 #else /* Itanium */
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
  typedef pfmlib_ita_param_t pfmw_ita_param_t;
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
/*   memset(p->pfp_pc, 0, sizeof p->pfp_pc);
     p->pfp_pc_count = *count; */
     ret = pfm_dispatch_events(p);
     if (ret == PFMLIB_SUCCESS) {
 	memcpy(pc, p->pfp_pc, sizeof(pfarg_reg_t)*PMU_MAX_PMCS);
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
 #define PFMW_REG_SMPLLRST(reg)     (reg.reg_long_reset)
 #define PFMW_REG_SMPLSRST(reg)     (reg.reg_short_reset)
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

#endif /* _PFMWRAP_H */



