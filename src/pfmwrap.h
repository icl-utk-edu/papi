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

inline pid_t mygettid(void)
{
#ifdef SYS_gettid
  return(syscall(SYS_gettid));
#elif defined(__NR_gettid)
  return(syscall(__NR_gettid));
#else
  return(syscall(1105));  
#endif
}

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

      thr_ctx->tid = mygettid();
      ctx[0].ctx_notify_pid = thr_ctx->tid;
      ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

      SUBDBG("PFM_CREATE_CONTEXT");
      if (perfmonctl(thr_ctx->tid, PFM_CREATE_CONTEXT, ctx, 1) == -1) {
	  if (errno == EBUSY)
	    {
	      SUBDBG("Context is busy!\n");
	    }
	  else
	    {
	      PAPIERROR("perfmonctl(PFM_CREATE_CONTEXT) errno %d", errno);
	      return(PAPI_ESYS);
	    }
	}

      /*
       * reset PMU (guarantee not active on return) and unfreeze
       * must be done before writing to any PMC/PMD
       */

      if (perfmonctl(thr_ctx->tid, PFM_ENABLE, 0, 0) == -1) {
         if (errno == ENOSYS)
	   PAPIERROR("Your kernel does not have performance monitoring support");
	 else
	   PAPIERROR("perfmonctl(PFM_ENABLE) errno %d", errno);
         return(PAPI_ESYS);
      }

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
   
   inline char* pfmw_get_event_name(unsigned int idx)
   {
      char *name;

      if (pfm_get_event_name(idx, &name) == PFMLIB_SUCCESS)
         return name;
      else
         return NULL;
   }


#else
#ifndef PFM30
#warning Maybe you should set -DPFM30 in your Makefile?
#endif

#if defined(__INTEL_COMPILER)

#define hweight64(x)    _m64_popcnt(x)

#elif defined(__GNUC__)

static __inline__ int
hweight64 (unsigned long x)
{
    unsigned long result;
    __asm__ ("popcnt %0=%1" : "=r" (result) : "r" (x));
    return (int)result;
}

#else
#error "you need to provide inline assembly from your compiler"
#endif

   #define OVFL_SIGNAL SIGIO
   #define PFMW_PEVT_EVTCOUNT(evt)          (evt->inp.pfp_event_count)
   #define PFMW_PEVT_EVENT(evt,idx)         (evt->inp.pfp_events[idx].event)
   #define PFMW_PEVT_PLM(evt,idx)           (evt->inp.pfp_events[idx].plm)
   #define PFMW_PEVT_DFLPLM(evt)            (evt->inp.pfp_dfl_plm)
   #define PFMW_PEVT_PFPPC(evt)             (evt->pc)
   #define PFMW_PEVT_PFPPC_COUNT(evt)       (evt->outp.pfp_pmc_count)
   #define PFMW_PEVT_PFPPC_REG_NUM(evt,idx) (evt->outp.pfp_pmcs[idx].reg_num)
   #define PFMW_PEVT_PFPPC_REG_VAL(evt,idx) (evt->pc[idx].reg_value)
   #define PFMW_PEVT_PFPPC_REG_FLG(evt,idx) (evt->pc[idx].reg_flags)
   #define PFMW_ARCH_REG_PMCVAL(reg) (reg.pmc_val)
   #define PFMW_ARCH_REG_PMDVAL(reg) (reg.pmd_val)
   #ifdef ITANIUM2
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita2_counter_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita2_counter_reg.pmc_es)
   #else /* Itanium */
      #define PFMW_ARCH_REG_PMCPLM(reg) (reg.pmc_ita_count_reg.pmc_plm)
      #define PFMW_ARCH_REG_PMCES(reg)  (reg.pmc_ita_count_reg.pmc_es)
   #endif

   typedef pfm_default_smpl_hdr_t    pfmw_smpl_hdr_t;
   typedef pfm_default_smpl_entry_t  pfmw_smpl_entry_t;

   #ifdef ITANIUM2
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
      ret=pfm_dispatch_events(&evt->inp, NULL, &evt->outp, NULL);   
      if (ret) { 
         return PAPI_ESYS;
      } else {
         for (i=0; i < evt->outp.pfp_pmc_count; i++) {
            evt->pc[i].reg_num   = evt->outp.pfp_pmcs[i].reg_num;
            evt->pc[i].reg_value = evt->outp.pfp_pmcs[i].reg_value;
         }

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

      if (perfmonctl(ctx->fd, PFM_LOAD_CONTEXT, &load_args, 1) == -1) {
         PAPIERROR("perfmonctl(PFM_WRITE_PMDS) errno %d",errno);
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

      ret = fcntl(ctx->fd, F_SETSIG, SIGIO);
      if (ret == -1) {
         PAPIERROR("fcntl(%d,F_SETSIG) errno %d", ctx->fd, errno);
        return(PAPI_ESYS);
      }


      return(PAPI_OK);

   }

   inline int pfmw_create_context(hwd_context_t *thr_ctx) {
      pfarg_context_t ctx[1];
      pfarg_load_t load_args;
      int ctx_fd;

      memset(ctx, 0, sizeof(ctx));
      memset(&load_args, 0, sizeof(load_args));

      if (perfmonctl(thr_ctx->tid, PFM_CREATE_CONTEXT, ctx, 1) == -1) {
	  if (errno == EBUSY)
	    {
	      SUBDBG("Context is busy!\n");
	    }
	  else
	    {
	      PAPIERROR("perfmonctl(PFM_CREATE_CONTEXT) errno %d", errno);
	      return(PAPI_ESYS);
	    }
      }
      ctx_fd = ctx[0].ctx_fd;
      thr_ctx->fd = ctx_fd;
      thr_ctx->tid = mygettid();

      return(pfmw_create_ctx_common(thr_ctx)); 
   }

   inline int set_pmds_to_write(EventSetInfo_t * ESI, int index, int value)
   {
      int *pos, count, hwcntr, i;
      pfmw_param_t *pevt= &(ESI->machdep.evt);
      hwd_control_state_t *this_state = &ESI->machdep;

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

   inline int pfmw_recreate_context(EventSetInfo_t * ESI, void **smpl_vaddr, 
                               int EventIndex) 
   {
      pfm_default_smpl_ctx_arg_t ctx[1];
      pfm_uuid_t buf_fmt_id = PFM_DEFAULT_SMPL_UUID;
      int ctx_fd;
      int native_index, EventCode, pos;
      hwd_context_t *thr_ctx = &ESI->master->context;
      void *tmp_ptr;

      pos= ESI->EventInfoArray[EventIndex].pos[0];
      EventCode= ESI->EventInfoArray[EventIndex].event_code;
      native_index= ESI->NativeInfoArray[pos].ni_event & PAPI_NATIVE_AND_MASK;

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
      if (perfmonctl(thr_ctx->fd, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
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
      thr_ctx->fd=ctx_fd;
      /* indicate which PMD to include in the sample */
/* DEAR and BTB events */
#ifdef ITANIUM2
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

   inline char* pfmw_get_event_name(unsigned int idx)
   {
      static char name[PAPI_MAX_STR_LEN];

      if (pfm_get_event_name(idx, name, PAPI_MAX_STR_LEN) == PFMLIB_SUCCESS)
         return name;
      else
         return NULL;
   }

#endif

#endif /* _PFMWRAP_H */



