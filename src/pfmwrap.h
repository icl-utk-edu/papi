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



   inline void pfmw_start(hwd_context_t *ctx_fd) {
      pfm_start();
   }

   inline void pfmw_stop(hwd_context_t * ctx_fd) {
      pfm_stop();
   }

   inline int pfmw_dispatch_events(pfmw_param_t *evt) {
      int ret;

      ret=pfm_dispatch_events(evt);   
      if (ret) return PAPI_ESYS;
      else return PAPI_OK;
   }

   inline int pfmw_perfmonctl(pid_t pid, int cmd, void *arg, int narg) {
      return(perfmonctl(pid, cmd, arg, narg));
   }

   inline int pfmw_destroy_context(void) {
      int ret;

      ret=perfmonctl(getpid(), PFM_DESTROY_CONTEXT, NULL, 0);
      if (ret) {   
         fprintf(stderr, "PID %d: perfmonctl error PFM_DESTROY_CONTEXT %d\n",
                 getpid(), errno);
         return PAPI_ESYS;
      } else return PAPI_OK;
   }

   inline int pfmw_create_context( hwd_context_t *thr_ctx) {
      pfarg_context_t ctx[1];
      memset(ctx, 0, sizeof(ctx));

      ctx[0].ctx_notify_pid = getpid();
      ctx[0].ctx_flags = PFM_FL_INHERIT_NONE;

      if (perfmonctl(getpid(), PFM_CREATE_CONTEXT, ctx, 1) == -1) {
         fprintf(stderr, "PID %d: perfmonctl error PFM_CREATE_CONTEXT %d\n", 
              getpid(), errno);
         return(PAPI_ESYS);
      }

      /*
       * reset PMU (guarantee not active on return) and unfreeze
       * must be done before writing to any PMC/PMD
       */

      if (perfmonctl(getpid(), PFM_ENABLE, 0, 0) == -1) {
         if (errno == ENOSYS)
            fprintf(stderr, "Your kernel does not have performance monitoring support !\n");
         fprintf(stderr, "PID %d: perfmonctl error PFM_ENABLE %d\n", getpid(), errno);
         return(PAPI_ESYS);
      }

      return(0);
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
      ctx[0].ctx_notify_pid = getpid();
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

      if (pfmw_perfmonctl(getpid(), PFM_CREATE_CONTEXT, ctx, 1) == -1) {
         fprintf(stderr, "PID %d: perfmonctl error PFM_CREATE_CONTEXT %d\n",
                 getpid(), errno);
         return (PAPI_ESYS);
      }
      SUBDBG("Sampling buffer mapped at %p\n", ctx[0].ctx_smpl_vaddr);

      *smpl_vaddr = ctx[0].ctx_smpl_vaddr;


      /*
       * reset PMU (guarantee not active on return) and unfreeze
       * must be done before writing to any PMC/PMD
       */
      if (pfmw_perfmonctl(getpid(), PFM_ENABLE, 0, 0) == -1) {
         if (errno == ENOSYS) {
            fprintf(stderr,
              "Your kernel does not have performance monitoring support !\n");
            fprintf(stderr, "PID %d: perfmonctl error PFM_ENABLE %d\n", 
                  getpid(), errno);
            return (PAPI_ESYS);
         }
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

#if defined(__ECC) && defined(__INTEL_COMPILER)

/* if you do not have this file, your compiler is too old */
#include <ia64intrin.h>

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

   inline void pfmw_start(hwd_context_t * ctx_fd) {
      int fd;
      fd = *(int *)ctx_fd;
      pfm_self_start(fd);
   }

   inline void pfmw_stop(hwd_context_t * ctx_fd) {
      int fd;
      fd = *(int *)ctx_fd;
      pfm_self_stop(fd);
   }

   inline int pfmw_perfmonctl(pid_t pid, int cmd, void *arg, int narg) {
      int *fd;
      _papi_hwi_get_thr_context((void **)&fd);
      return(perfmonctl(*fd, cmd, arg, narg));
   }

   inline int pfmw_destroy_context(void) {
      int *fd, ret;
      _papi_hwi_get_thr_context((void **)&fd);
/*
      printf("fd of the thread = %d\n", *fd);
*/

      ret=close(*fd);
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

   inline int pfmw_create_ctx_common(int ctx_fd) 
   {
      pfarg_load_t load_args;
      int ret;

      memset(&load_args, 0, sizeof(load_args));
      /*
       * we want to monitor ourself
       */
      load_args.load_pid = getpid();

      if (perfmonctl(ctx_fd, PFM_LOAD_CONTEXT, &load_args, 1) == -1) {
         fprintf(stderr,"perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
         return(PAPI_ESYS);
      }
      /*
       * setup asynchronous notification on the file descriptor
       */
      ret = fcntl(ctx_fd, F_SETFL, fcntl(ctx_fd, F_GETFL, 0) | O_ASYNC);
      if (ret == -1) {
         fprintf(stderr,"cannot set ASYNC: %s\n", strerror(errno));
         return(PAPI_ESYS);
      }

      /*
       * get ownership of the descriptor
       */
      ret = fcntl(ctx_fd, F_SETOWN, getpid());
      if (ret == -1) {
         fprintf(stderr,"cannot setown: %s\n", strerror(errno));
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

      if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1) {
         fprintf(stderr, "PID %d: perfmonctl error PFM_CREATE_CONTEXT %d\n", 
              getpid(), errno);
         return(PAPI_ESYS);
      }
      ctx_fd = ctx[0].ctx_fd;
      *thr_ctx = ctx_fd;
/*
      printf("fd of the thread = %d\n", *thr_ctx);
*/

      return(pfmw_create_ctx_common(ctx_fd)); 
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
      int ctx_fd, *thr_fd;
      int native_index, EventCode, pos;

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
      if (perfmonctl(0, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
         if (errno == ENOSYS) {
            fprintf(stderr, 
              "Your kernel does not have performance monitoring support!\n");
         }
         fprintf(stderr,"Can't create PFM context %s\n", strerror(errno));
         return(PAPI_ESYS);
      }
      /*
       * extract the file descriptor we will use to
       * identify this newly created context
       */
      ctx_fd = ctx[0].ctx_arg.ctx_fd;
      /* save the fd into the thread context struct */
      _papi_hwi_get_thr_context((void **)&thr_fd);
      *thr_fd=ctx_fd;
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

      return(pfmw_create_ctx_common(ctx_fd)); 
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



