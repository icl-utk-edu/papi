#ifndef _PFMWRAP_H
#define _PFMWRAP_H

#define MAX_COUNTER_TERMS 4

#ifdef ITANIUM2
  #define MAX_NATIVE_EVENT  475  /* the number comes from itanium_events.h */
  #define MAX_COUNTERS PMU_ITA2_NUM_COUNTERS
  #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita2_count_reg.pmc_plm)
  #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita2_count_reg.pmc_es)
  typedef pfm_ita2_reg_t pfmw_arch_reg_t;
  typedef pfmlib_ita2_param_t pfmw_ita_param_t;

#else  /* itanium */
  #define MAX_NATIVE_EVENT  230  /* the number comes from itanium_events.h */
  #define MAX_COUNTERS PMU_ITA_NUM_COUNTERS
  #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita_count_reg.pmc_plm)
  #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita_count_reg.pmc_es)
  typedef pfm_ita_reg_t pfmw_arch_reg_t;
  typedef pfmlib_ita_param_t pfmw_ita_param_t;
#endif

/*
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
 #else 
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
*/

/*
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
 #define PFMW_CTX_NOTIFYPID(ctx)   (ctx.ctx_notify_pid)
 #define PFMW_CTX_FLAGS(ctx)       (ctx.ctx_flags)
*/

#endif /* _PFMWRAP_H */



