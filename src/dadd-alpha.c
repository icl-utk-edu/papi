/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "papi.h"
#include SUBSTRATE
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"


extern EventSetInfo_t *default_master_eventset;

extern papi_mdi_t _papi_hwi_system_info;

static hwi_search_t findem_dadd[] = {
   {PAPI_TOT_CYC, {0, {PAPI_NATIVE_MASK | 0, PAPI_NULL}}},
   {PAPI_L1_ICM, {0, {PAPI_NATIVE_MASK | 3, PAPI_NULL}}},
   {PAPI_L2_TCM, {0, {PAPI_NATIVE_MASK | 1, PAPI_NULL}}},
   {PAPI_TLB_DM, {0, {PAPI_NATIVE_MASK | 2, PAPI_NULL}}},
   {PAPI_BR_UCN, {0, {PAPI_NATIVE_MASK | 21, PAPI_NULL}}},
   {PAPI_BR_CN, {0, {PAPI_NATIVE_MASK | 22, PAPI_NULL}}},
   {PAPI_BR_NTK, {0, {PAPI_NATIVE_MASK | 24, PAPI_NULL}}},
   {PAPI_BR_MSP, {0, {PAPI_NATIVE_MASK | 25, PAPI_NULL}}},
   {PAPI_BR_PRC, {0, {PAPI_NATIVE_MASK | 26, PAPI_NULL}}},
   {PAPI_TOT_INS, {0, {PAPI_NATIVE_MASK | 8, PAPI_NULL}}},
   {PAPI_TOT_IIS, {0, {PAPI_NATIVE_MASK | 7, PAPI_NULL}}},
   {PAPI_FP_INS, {0, {PAPI_NATIVE_MASK | 20, PAPI_NULL}}},
   {PAPI_FP_OPS, {0, {PAPI_NATIVE_MASK | 20, PAPI_NULL}}},
   {PAPI_LD_INS, {0, {PAPI_NATIVE_MASK | 10, PAPI_NULL}}},
   {PAPI_SR_INS, {0, {PAPI_NATIVE_MASK | 11, PAPI_NULL}}},
   {PAPI_LST_INS, {0, {PAPI_NATIVE_MASK | 12, PAPI_NULL}}},
   {PAPI_SYC_INS, {0, {PAPI_NATIVE_MASK | 13, PAPI_NULL}}},
   {PAPI_FML_INS, {0, {PAPI_NATIVE_MASK | 17, PAPI_NULL}}},
   {PAPI_FAD_INS, {0, {PAPI_NATIVE_MASK | 16, PAPI_NULL}}},
   {PAPI_FDV_INS, {0, {PAPI_NATIVE_MASK | 18, PAPI_NULL}}},
   {PAPI_FSQ_INS, {0, {PAPI_NATIVE_MASK | 19, PAPI_NULL}}},
   {PAPI_INT_INS, {0, {PAPI_NATIVE_MASK | 9, PAPI_NULL}}},
   {PAPI_BR_INS, { DERIVED_ADD, {PAPI_NATIVE_MASK|21, PAPI_NATIVE_MASK |22}}},
   {PAPI_TLB_TL, { DERIVED_ADD, {PAPI_NATIVE_MASK|27, PAPI_NATIVE_MASK|2}}},
   {PAPI_TLB_IM, {0, {PAPI_NATIVE_MASK | 27, PAPI_NULL}}},
   {PAPI_BR_TKN, {0, {PAPI_NATIVE_MASK | 23, PAPI_NULL}}},
   {0, {0, {0, 0}}}
};

native_info_t alpha_native_table[] = {
/* 0  */ {"VC_TOTAL_CYCLES", 0},
/* 1  */ {"VC_BCACHE_MISSES", 1},
/* 2  */ {"VC_TOTAL_DTBMISS", 2},
/* 3  */ {"VC_NYP_EVENTS", 3},
/* 4  */ {"VC_TAKEN_EVENTS", 4},
/* 5  */ {"VC_MISPREDICT_EVENTS", 5},
/* 6  */ {"VC_LD_ST_ORDER_TRAPS", 6},
/* 7  */ {"VC_TOTAL_INSTR_ISSUED", 7},
/* 8  */ {"VC_TOTAL_INSTR_EXECUTED", 8},
/* 9  */ {"VC_INT_INSTR_EXECUTED", 9},
/* 10 */ {"VC_LOAD_INSTR_EXECUTED", 10},
/* 11 */ {"VC_STORE_INSTR_EXECUTED", 11},
/* 12 */ {"VC_TOTAL_LOAD_STORE_EXECUTED", 12},
/* 13 */ {"VC_SYNCH_INSTR_EXECUTED", 13},
/* 14 */ {"VC_NOP_INSTR_EXECUTED", 14},
/* 15 */ {"VC_PREFETCH_INSTR_EXECUTED", 15},
/* 16 */ {"VC_FA_INSTR_EXECUTED", 16},
/* 17 */ {"VC_FM_INSTR_EXECUTED", 17},
/* 18 */ {"VC_FD_INSTR_EXECUTED", 18},
/* 19 */ {"VC_FSQ_INSTR_EXECUTED", 19},
/* 20 */ {"VC_FP_INSTR_EXECUTED", 20},
/* 21 */ {"VC_UNCOND_BR_EXECUTED", 21},
/* 22 */ {"VC_COND_BR_EXECUTED", 22},
/* 23 */ {"VC_COND_BR_TAKEN", 23},
/* 24 */ {"VC_COND_BR_NOT_TAKEN", 24},
/* 25 */ {"VC_COND_BR_MISPREDICTED", 25},
/* 26 */ {"VC_COND_BR_PREDICTED", 26},
/* 27 */ {"VC_ITBMISS_TRAPS", 38}
};

static int get_system_info(void)
{
   int fd, retval, family;
   prpsinfo_t info;
   struct cpu_info cpuinfo;
   long proc_type;
   pid_t pid;
   char pname[PATH_MAX], *ptr;
   struct clu_gen_info *clugenptr;

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);
   sprintf(pname, "/proc/%05d", (int) pid);

   fd = open(pname, O_RDONLY);
   if (fd == -1)
      return (PAPI_ESYS);
   if (ioctl(fd, PIOCPSINFO, &info) == -1)
      return (PAPI_ESYS);
   close(fd);

   /* Cut off any arguments to exe */
   {
     char *tmp;
     tmp = strchr(info.pr_psargs, ' ');
     if (tmp != NULL)
       *tmp = '\0';
   }

   if (realpath(info.pr_psargs,pname))
     strncpy(_papi_hwi_system_info.exe_info.fullname, pname, PAPI_HUGE_STR_LEN);
   else
     strncpy(_papi_hwi_system_info.exe_info.fullname, info.pr_psargs, PAPI_HUGE_STR_LEN);

   strncpy(_papi_hwi_system_info.exe_info.address_info.name, info.pr_fname, PAPI_MAX_STR_LEN);

   if (getsysinfo
       (GSI_CPU_INFO, (char *) &cpuinfo, sizeof(cpuinfo), NULL, NULL,
        NULL) == -1)
      return PAPI_ESYS;

   if (getsysinfo
       (GSI_PROC_TYPE, (char *) &proc_type, sizeof(proc_type), 0, 0,
        0) == -1)
      return PAPI_ESYS;
   proc_type &= 0xffffffff;

   clugenptr = NULL;

   retval = clu_get_info(&clugenptr);

   switch (retval) {
   case 0:
      break;
   case CLU_NOT_MEMBER:
   case CLU_NO_CLUSTER_NAME:
   case CLU_NO_MEMBERID:
   case CLU_CNX_ERROR:
      _papi_hwi_system_info.hw_info.nnodes = 1;
   default:
      _papi_hwi_system_info.hw_info.nnodes = 1;
   }

   if (clugenptr == NULL)
      _papi_hwi_system_info.hw_info.nnodes = 1;
   else
      _papi_hwi_system_info.hw_info.nnodes = clugenptr->clu_num_of_members;

   clu_free_info(&clugenptr);

/*
  _papi_hwi_system_info.cpunum = cpuinfo.current_cpu;
*/
   _papi_hwi_system_info.hw_info.mhz = (float) cpuinfo.mhz;
   _papi_hwi_system_info.hw_info.ncpu = cpuinfo.cpus_in_box;
   _papi_hwi_system_info.hw_info.totalcpus =
       _papi_hwi_system_info.hw_info.ncpu *
       _papi_hwi_system_info.hw_info.nnodes;
   _papi_hwi_system_info.hw_info.vendor = -1;
   strcpy(_papi_hwi_system_info.hw_info.vendor_string, "Compaq");
   _papi_hwi_system_info.hw_info.model = proc_type;
   strcpy(_papi_hwi_system_info.hw_info.model_string, "Alpha ");
   family = cpu_implementation_version();

   /* program text segment, data segment  address info */
   _papi_hwi_system_info.exe_info.address_info.text_start =
       (caddr_t) & _ftext;
   _papi_hwi_system_info.exe_info.address_info.text_end =
       (caddr_t) & _etext;
   _papi_hwi_system_info.exe_info.address_info.data_start =
                                            (caddr_t) & _fdata;
   _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) & _edata;




   if (family == 0) {
      strcat(_papi_hwi_system_info.hw_info.model_string, "21064");
   }
   if (family == 2) {
      strcat(_papi_hwi_system_info.hw_info.model_string, "21264");
   } else if (family == 1) {
      strcat(_papi_hwi_system_info.hw_info.model_string, "21164");
   } else if (family == 3) {    /* EV7, thanks Paul */
      strcat(_papi_hwi_system_info.hw_info.model_string, "21364");
   } else
      return (PAPI_ESBSTR);

   _papi_hwi_system_info.num_cntrs = MAX_COUNTERS;
   retval = _papi_hwi_setup_all_presets(findem_dadd);
   if (retval)
      return (retval);

   return (PAPI_OK);
}

long long _papi_hwd_get_real_usec(void)
{
   struct timespec res;

   if ((clock_gettime(CLOCK_REALTIME, &res) == -1))
      return (PAPI_ESYS);
   return (res.tv_sec * 1000000) + (res.tv_nsec / 1000);
}

long long _papi_hwd_get_real_cycles(void)
{
   return ((long long) _papi_hwd_get_real_usec() *
           _papi_hwi_system_info.hw_info.mhz);
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   struct rusage res;

   if ((getrusage(RUSAGE_SELF, &res) == -1))
      return (PAPI_ESYS);
   return ((res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec);
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   return ((long long) _papi_hwd_get_virt_usec(zero) *
           _papi_hwi_system_info.hw_info.mhz);
}

void _papi_hwd_error(int error, char *where)
{
   sprintf(where, "Substrate error");
}

int _papi_hwd_init_global(void)
{
   int retval;

   /* Install termination signal handlers */
   (void) signal(SIGINT, dadd_terminate_cleanup);
   (void) signal(SIGTERM, dadd_terminate_cleanup);

   /* Fill in what we can of the papi_hwi_system_info. */
   retval = get_system_info();
   if (retval)
      return (retval);

   SUBDBG("Found %d %s %s CPU's at %f Mhz.\n",
        _papi_hwi_system_info.hw_info.totalcpus,
        _papi_hwi_system_info.hw_info.vendor_string,
        _papi_hwi_system_info.hw_info.model_string,
        _papi_hwi_system_info.hw_info.mhz);

   return (PAPI_OK);
}

int _papi_hwd_init(hwd_context_t * ctx)
{
   pid_t pid;
   unsigned char *region_address = NULL;

   pid = getpid();
   region_address = dadd_start_monitoring(pid);
   if (region_address == NULL)
      return (PAPI_ESBSTR);
   ctx->ptr_vc = (virtual_counters *) region_address;
   return (PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t * this_state,
                             unsigned int event, void *extra,
                             EventInfo_t * out)
{
   return (PAPI_ESBSTR);
}

int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int i;
   unsigned long count;
   virtual_counters *ptr_vc;

   ptr_vc = ctx->ptr_vc;
   ctrl->latestcycles = (long long) ptr_vc->vc_total_cycles;
   memcpy(ctrl->start_value, (char *) ctrl->ptr_vc + sizeof(unsigned long)
          + sizeof(struct timeval), sizeof(unsigned long) * MAX_COUNTERS);
/*
  for (i=0; i<MAX_COUNTERS; i++) {
    if (ptr_vc) {
      memcpy(&count,
       (char *)ptr_vc+sizeof(unsigned long)+sizeof(struct timeval)+
         sizeof(unsigned long)*i,
       sizeof(unsigned long));
    }
    else
      return(PAPI_ESBSTR);

    ctrl->start_value[i] = (long long)(count);
  }
*/
   return (PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * ctrl,
                   long long **events)
{
   int i;
   unsigned long count;
   virtual_counters *ptr_vc;
   hwd_control_state_t *this_state;

   this_state = ctrl;
   ptr_vc = ctx->ptr_vc;
/*
  for (i=0; i<MAX_COUNTERS; i++) {
    if (ptr_vc) {
        memcpy(&count,
             (char *)ptr_vc+sizeof(unsigned long)+
             sizeof(struct timeval)+sizeof(unsigned long)*i,
             sizeof(unsigned long));
    }
    else
      return(PAPI_ESBSTR);
    this_state->latest[i] = ((long long)(count)) - this_state->start_value[i];
  }
*/
#ifdef DEBUG
   _debug_virtual_counters(ctx);
#endif
   memcpy(ctrl->latest, (char *) ctrl->ptr_vc + sizeof(unsigned long)
          + sizeof(struct timeval), sizeof(unsigned long) * MAX_COUNTERS);
   for (i = 0; i < MAX_COUNTERS; i++) {
      this_state->latest[i] -= this_state->start_value[i];
   }
   *events = this_state->latest;

   return (PAPI_OK);
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * ctrl,
                    long long events[])
{
   return (PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
   return (PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
   hwd_control_state_t *current_state = NULL;
   unsigned char *region_address = NULL;
   int retval;
   pid_t pid;

   pid = getpid();

   if (default_master_eventset) {
      current_state = &default_master_eventset->machdep;
      if (current_state)
         region_address = (unsigned char *) current_state->ptr_vc;
      if (region_address) {
         retval = dadd_stop_monitoring(pid, region_address);
         current_state->ptr_vc = NULL;
         if (retval == -1)
            return (PAPI_ESYS);
      }
   }
   return (PAPI_OK);
}

int _papi_hwd_setmaxmem()
{
   return (PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code,
                  _papi_int_option_t * option)
{
   return (PAPI_ESBSTR);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex,
                           int threshold)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_ESBSTR);
}

int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex,
                          int threshold)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI)
{
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

void _papi_hwd_lock_init(void)
{
}

void _papi_hwd_lock(int lck)
{
}

void _papi_hwd_unlock(int lck)
{
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si,
                              void *info)
{
   _papi_hwi_context_t ctx;
   ctx.si = si;
   ctx.ucontext = info;

   _papi_hwi_dispatch_overflow_signal((void *) &ctx,
                                      _papi_hwi_system_info.
                                      supports_hw_overflow, 0, 0);
}

/* start the hardware counting, in this substrate, we just need
   to reset the latest array in control state  */
int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int i;
   unsigned long count;

#ifdef DEBUG
   _debug_virtual_counters(ctx);
#endif
   ctrl->ptr_vc = ctx->ptr_vc;
   memcpy(ctrl->start_value, (char *) ctrl->ptr_vc + sizeof(unsigned long)
          + sizeof(struct timeval), sizeof(unsigned long) * MAX_COUNTERS);
   return (PAPI_OK);
}

/* stop the counting */
int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   return (PAPI_OK);
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer)
{
   int index = *EventCode & PAPI_NATIVE_AND_MASK;

   if (index < MAX_NATIVE_EVENT - 1) {
      *EventCode = *EventCode + 1;
      return (PAPI_OK);
   } else
      return (PAPI_ENOEVNT);
}

int _papi_hwd_update_shlib_info(void)
{
   return (PAPI_ESBSTR);
}

void _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   return;
}

int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                                   NativeInfo_t * native, int count)
{
   int i, index;

   for (i = 0; i < count; i++) {
      index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
      native[i].ni_position = alpha_native_table[index].encode;
   }
   return (PAPI_OK);
}

int _papi_hwd_allocate_registers(EventSetInfo_t * ESI)
{
   return 1;
}

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   int nidx;

   nidx = EventCode ^ PAPI_NATIVE_MASK;
   if (nidx >= 0 && nidx < PAPI_MAX_NATIVE_EVENTS)
      return (alpha_native_table[nidx].name);
   else
      return NULL;
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (_papi_hwd_ntv_code_to_name(EventCode));
}

int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr)
{
   return (PAPI_OK);
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t * dst, int ctr)
{
}

/* This function examines the event to determine
    if it has a single exclusive mapping.
    Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst)
{
   return (PAPI_OK);
}

/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   return (PAPI_OK);
}

/* This function removes the counters available to the src event
    from the counters available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t * dst,
                               hwd_reg_alloc_t * src)
{
}

/* This function updates the selection status of
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
}

void _debug_virtual_counters(hwd_context_t * ctx)
{
#if 0
   printf("vc_total_cycles=%ld \n", ctx->ptr_vc->vc_total_cycles);
   printf("vc_bcache_misses=%ld \n", ctx->ptr_vc->vc_bcache_misses);
   printf("vc_total_dtbmiss=%ld \n", ctx->ptr_vc->vc_total_dtbmiss);
   printf("vc_nyp_events=%ld \n", ctx->ptr_vc->vc_nyp_events);
   printf("vc_taken_events=%ld \n", ctx->ptr_vc->vc_taken_events);
   printf("vc_mispredict_events=%ld \n",
          ctx->ptr_vc->vc_mispredict_events);
   printf("vc_ld_st_order_traps=%ld \n",
          ctx->ptr_vc->vc_ld_st_order_traps);
   printf("vc_total_instr_issued=%ld \n",
          ctx->ptr_vc->vc_total_instr_issued);
   printf("vc_total_instr_executed=%ld \n",
          ctx->ptr_vc->vc_total_instr_executed);
   printf("vc_int_instr_executed=%ld \n",
          ctx->ptr_vc->vc_int_instr_executed);
   printf("vc_load_instr_executed=%ld \n",
          ctx->ptr_vc->vc_load_instr_executed);
   printf("vc_store_instr_executed=%ld \n",
          ctx->ptr_vc->vc_store_instr_executed);
   printf("vc_total_load_store_executed=%ld \n",
          ctx->ptr_vc->vc_total_load_store_executed);
   printf("vc_synch_instr_executed=%ld \n",
          ctx->ptr_vc->vc_synch_instr_executed);
   printf("vc_nop_instr_executed=%ld \n",
          ctx->ptr_vc->vc_nop_instr_executed);
   printf("vc_prefetch_instr_executed=%ld \n",
          ctx->ptr_vc->vc_prefetch_instr_executed);
   printf("vc_fa_instr_executed=%ld \n",
          ctx->ptr_vc->vc_fa_instr_executed);
   printf("vc_fm_instr_executed=%ld \n",
          ctx->ptr_vc->vc_fm_instr_executed);
   printf("vc_fd_instr_executed=%ld \n",
          ctx->ptr_vc->vc_fd_instr_executed);
   printf("vc_fsq_instr_executed=%ld \n",
          ctx->ptr_vc->vc_fsq_instr_executed);
   printf("vc_fp_instr_executed=%ld \n",
          ctx->ptr_vc->vc_fp_instr_executed);
   printf("vc_uncond_br_executed=%ld \n",
          ctx->ptr_vc->vc_uncond_br_executed);
   printf("vc_cond_br_executed=%ld \n", ctx->ptr_vc->vc_cond_br_executed);
   printf("vc_cond_br_taken=%ld \n", ctx->ptr_vc->vc_cond_br_taken);
   printf("vc_cond_br_not_taken=%ld \n",
          ctx->ptr_vc->vc_cond_br_not_taken);
   printf("vc_cond_br_mispredicted=%ld \n",
          ctx->ptr_vc->vc_cond_br_mispredicted);
   printf("vc_cond_br_predicted=%ld \n",
          ctx->ptr_vc->vc_cond_br_predicted);
   printf("vc_replay_traps=%ld \n", ctx->ptr_vc->vc_replay_traps);
   printf("vc_no_traps=%ld \n", ctx->ptr_vc->vc_no_traps);
   printf("vc_dtb2miss3_traps=%ld \n", ctx->ptr_vc->vc_dtb2miss3_traps);
   printf("vc_dtb2miss4_traps=%ld \n", ctx->ptr_vc->vc_dtb2miss4_traps);
   printf("vc_fpdisabled_traps=%ld \n", ctx->ptr_vc->vc_fpdisabled_traps);
   printf("vc_unalign_traps=%ld \n", ctx->ptr_vc->vc_unalign_traps);
   printf("vc_dtbmiss_traps=%ld \n", ctx->ptr_vc->vc_dtbmiss_traps);
   printf("vc_dfault_traps=%ld \n", ctx->ptr_vc->vc_dfault_traps);
   printf("vc_opcdec_traps=%ld \n", ctx->ptr_vc->vc_opcdec_traps);
   printf("vc_mispredict_traps=%ld \n", ctx->ptr_vc->vc_mispredict_traps);
   printf("vc_mchk_traps=%ld \n", ctx->ptr_vc->vc_mchk_traps);
   printf("vc_itbmiss_traps=%ld \n", ctx->ptr_vc->vc_itbmiss_traps);
   printf("vc_arith_traps=%ld \n", ctx->ptr_vc->vc_arith_traps);
   printf("vc_interrupt_traps=%ld \n", ctx->ptr_vc->vc_interrupt_traps);
   printf("vc_mt_fpcr_traps=%ld \n", ctx->ptr_vc->vc_mt_fpcr_traps);
   printf("vc_iacv_traps=%ld \n", ctx->ptr_vc->vc_iacv_traps);
   printf("vc_did_not_retire=%ld \n", ctx->ptr_vc->vc_did_not_retire);
   printf("vc_early_kill=%ld \n", ctx->ptr_vc->vc_early_kill);
   printf("vc_update_count=%ld \n", ctx->ptr_vc->vc_update_count);
   printf("vc_pdb_sample_count_sum=%ld \n",
          ctx->ptr_vc->vc_pdb_sample_count_sum);
#endif

}
