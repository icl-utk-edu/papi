/* Implementation of IRIX platform is straightfoward.
   First read r10k_counters manual page, then you will get an idea about
   the performance counter in IRIX.
*/

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

char *(r10k_native_events_table[]) = {
   /* 0  */ "Cycles",
       /* 1  */ "Issued_instructions",
       /* 2  */ "Issued_loads",
       /* 3  */ "Issued_stores",
       /* 4  */ "Issued_store_conditionals",
       /* 5  */ "Failed_store_conditionals",
       /* 6  */ "Decoded_branches",
       /* 7  */ "Quadwords_written_back_from_scache",
       /* 8  */ "Correctable_scache_data_array_ECC_errors",
       /* 9  */ "Primary_instruction_cache_misses",
       /* 10  */ "Secondary_instruction_cache_misses",
       /* 11  */ "Instruction_misprediction_from_scache_way_prediction_table",
       /* 12  */ "External_interventions",
       /* 13  */ "External_invalidations",
       /* 14  */ "Virtual_coherency_conditions",
       /* 15  */ "Graduated_instructions",
       /* 16  */ "Cycles",
       /* 17  */ "Graduated_instructions",
       /* 18  */ "Graduated_loads",
       /* 19  */ "Graduated_stores",
       /* 20  */ "Graduated_store_conditionals",
       /* 21  */ "Graduated_floating_point_instructions",
       /* 22  */ "Quadwords_written_back_from_primary_data_cache",
       /* 23  */ "TLB_misses",
       /* 24  */ "Mispredicted_branches",
       /* 25  */ "Primary_data_cache_misses",
       /* 26  */ "Secondary_data_cache_misses",
       /* 27  */ "Data_misprediction_from_scache_way_prediction_table",
       /* 28  */ "External_intervention_hits_in_scache",
       /* 29  */ "External_invalidation_hits_in_scache",
       /* 30  */ "Store/prefetch_exclusive_to_clean_block_in_scache",
       /* 31  */ "Store/prefetch_exclusive_to_shared_block_in_scache",
0};

/* r14k native events are the same as r12k */
char *(r12k_native_events_table[]) = {
/* 0  */ "cycles",
/* 1  */ "decoded_instructions",
/* 2  */ "decoded_loads",
/* 3  */ "decoded_stores",
/* 4  */ "mishandling_table_occupancy",
/* 5  */ "failed_store_conditionals",
/* 6  */ "resolved_conditional_branches",
/* 7  */ "Quadwords_written_back_from_secondary_cache",
/* 8  */ "correctable_secondary_cache_data_array_ECC_errors",
/* 9  */ "primary_instruction_cache_misses",
/* 10 */ "secondary_instruction_cache_misses",
/* 11 */ "instruction_misprediction_from_secondary_cache_way_prediction_table",
/* 12 */ "external_interventions",
/* 13 */ "external_invalidations",
/* 14 */ "ALU/FPU_progress_cycles",
/* 15 */ "graduated_instructions",
/* 16 */ "executed_prefetch_instructions",
/* 17 */ "prefetch_primary_data_cache_misses",
/* 18 */ "graduated_loads",
/* 19 */ "graduated_stores",
/* 20 */ "graduated_store_conditions",
/* 21 */ "graduated_floating-point_instructions",
/* 22 */ "quadwords_written_back_from_primary_data_cache",
/* 23 */ "TLB_misses",
/* 24 */ "mispredicted_branches",
/* 25 */ "primary_data_cache_misses",
/* 26 */ "secondary_data_cache_misses",
/* 27 */ "data_misprediction_from_secondary_cache_way_prediction_table",
/* 28 */ "state_of_external_intervention_hits_in_secondary cache",
/* 29 */ "state_of_invalidation_hits_in_secondary_cache",
/* 30 */ "Miss_Handling_Table_entries_accessing_memory",
/* 31 */ "store/prefetch_exclusive_to_shared_block_in_secondary_cache",
0};

char **native_table;
hwi_search_t *preset_search_map;

/* the number in this preset_search map table is the native event index
   in the native event table, when it ORs the PAPI_NATIVE_MASK, it becomes the
   native event code.
   For example, 25 is the index of native event "Primary_data_cache_misses" 
   in the native event table
*/
hwi_search_t findem_r10k[] = {
   {PAPI_L1_DCM, {0, {PAPI_NATIVE_MASK | 25, PAPI_NULL}}},   /* L1 D-Cache misses */
   {PAPI_L1_ICM, {0, {PAPI_NATIVE_MASK | 9, PAPI_NULL}}},    /* L1 I-Cache misses */
   {PAPI_L2_DCM, {0, {PAPI_NATIVE_MASK | 26, PAPI_NULL}}},   /* L2 D-Cache misses */
   {PAPI_L2_ICM, {0, {PAPI_NATIVE_MASK | 10, PAPI_NULL}}},   /* L2 I-Cache misses */
   {PAPI_L1_TCM, {DERIVED_ADD, {PAPI_NATIVE_MASK | 9, PAPI_NATIVE_MASK | 25, PAPI_NULL}}},
   /* L1 total */
   {PAPI_L2_TCM, {DERIVED_ADD, {PAPI_NATIVE_MASK | 10, PAPI_NATIVE_MASK | 26,PAPI_NULL}}},
   /* L2 total */
   {PAPI_CA_INV, {0, {PAPI_NATIVE_MASK | 13, PAPI_NULL}}},   /* Cache Line Invalidation */
   {PAPI_CA_ITV, {0, {PAPI_NATIVE_MASK | 12, PAPI_NULL}}},   /* Cache Line Intervention */
   {PAPI_TLB_TL, {0, {PAPI_NATIVE_MASK | 23, PAPI_NULL}}},   /* Total TLB misses */
   {PAPI_CSR_FAL, {0, {PAPI_NATIVE_MASK | 5, PAPI_NULL}}},   /* Failed store conditional */
   {PAPI_CSR_SUC, {DERIVED_SUB, {PAPI_NATIVE_MASK | 20, PAPI_NATIVE_MASK | 5, PAPI_NULL}}},
   /* Successful store conditional */
   {PAPI_CSR_TOT, {0, {PAPI_NATIVE_MASK | 20, PAPI_NULL}}},  /* Total store conditional */
   {PAPI_BR_MSP, {0, {PAPI_NATIVE_MASK | 24, PAPI_NULL}}},   /* Cond. branch inst. mispred */
   {PAPI_TOT_IIS, {0, {PAPI_NATIVE_MASK | 1, PAPI_NULL}}},   /* Total inst. issued */
   {PAPI_TOT_INS, {0, {PAPI_NATIVE_MASK | 17, PAPI_NULL}}},  /* Total inst. executed */
   {PAPI_FP_INS, {0, {PAPI_NATIVE_MASK | 21, PAPI_NULL}}},   /* Floating Pt. inst. executed */
   {PAPI_FP_OPS, {0, {PAPI_NATIVE_MASK | 21, PAPI_NULL}}},   /* Floating Pt. inst. executed */
   {PAPI_LD_INS, {0, {PAPI_NATIVE_MASK | 18, PAPI_NULL}}},    /* Loads executed */
   {PAPI_SR_INS, {0, {PAPI_NATIVE_MASK | 19, PAPI_NULL}}},   /* Stores executed */
   {PAPI_BR_INS, {0, {PAPI_NATIVE_MASK | 6, PAPI_NULL}}},    /* Branch inst. executed */
   {PAPI_TOT_CYC, {0, {PAPI_NATIVE_MASK | 0, PAPI_NULL}}},   /* Total cycles */
   {0, {0, {0, 0}}}             /* The END */
};


hwi_search_t findem_r12k[] = {  /* Shared with R14K */
   {PAPI_L1_DCM, {0, {PAPI_NATIVE_MASK | 25, PAPI_NULL}}},   /* L1 D-Cache misses */
   {PAPI_L1_ICM, {0, {PAPI_NATIVE_MASK | 9, PAPI_NULL}}},    /* L1 I-Cache misses */
   {PAPI_L2_DCM, {0, {PAPI_NATIVE_MASK | 26, PAPI_NULL}}},   /* L2 D-Cache misses */
   {PAPI_L2_ICM, {0, {PAPI_NATIVE_MASK | 10, PAPI_NULL}}},   /* L2 I-Cache misses */
   {PAPI_L1_TCM, {DERIVED_ADD, {PAPI_NATIVE_MASK | 9, PAPI_NATIVE_MASK | 25, PAPI_NULL}}},        /* L1 total */
   {PAPI_L2_TCM, {DERIVED_ADD, {PAPI_NATIVE_MASK | 10, PAPI_NATIVE_MASK | 26, PAPI_NULL}}},       /* L2 total */
   {PAPI_CA_INV, {0, {PAPI_NATIVE_MASK | 13, PAPI_NULL}}},   /* Cache Line Invalidation */
   {PAPI_CA_ITV, {0, {PAPI_NATIVE_MASK | 12, PAPI_NULL}}},   /* Cache Line Intervention */
   {PAPI_TLB_TL, {0, {PAPI_NATIVE_MASK | 23, PAPI_NULL}}},   /* Total TLB misses */
   {PAPI_PRF_DM, {0, {PAPI_NATIVE_MASK | 17, PAPI_NULL}}},   /* Prefetch miss */
   {PAPI_CSR_FAL, {0, {PAPI_NATIVE_MASK | 5, PAPI_NULL}}},   /* Failed store conditional */
   {PAPI_CSR_SUC, {DERIVED_SUB, {PAPI_NATIVE_MASK | 20, PAPI_NATIVE_MASK | 5, PAPI_NULL}}},
   /* Successful store conditional */
   {PAPI_CSR_TOT, {0, {PAPI_NATIVE_MASK | 20, PAPI_NULL}}},  /* Total store conditional */
   {PAPI_BR_CN, {0, {PAPI_NATIVE_MASK | 6, PAPI_NULL}}},     /* Cond. branch inst. exe */
   {PAPI_BR_MSP, {0, {PAPI_NATIVE_MASK | 24, PAPI_NULL}}},   /* Cond. branch inst. mispred */
   {PAPI_BR_PRC, {DERIVED_SUB, {PAPI_NATIVE_MASK | 6, PAPI_NATIVE_MASK | 24, PAPI_NULL}}},
   /* Cond. branch inst. correctly pred */
   {PAPI_TOT_IIS, {0, {PAPI_NATIVE_MASK | 1, PAPI_NULL}}},   /* Total inst. issued */
   {PAPI_TOT_INS, {0, {PAPI_NATIVE_MASK | 15, PAPI_NULL}}},  /* Total inst. executed */
   {PAPI_LD_INS, {0, {PAPI_NATIVE_MASK | 18, PAPI_NULL}}},   /* Loads executed */
   {PAPI_SR_INS, {0, {PAPI_NATIVE_MASK | 19, PAPI_NULL}}},   /* Stores executed */
   {PAPI_FP_INS, {0, {PAPI_NATIVE_MASK | 21, PAPI_NULL}}},   /* Floating Pt. inst.executed */
   {PAPI_FP_OPS, {0, {PAPI_NATIVE_MASK | 21, PAPI_NULL}}},   /* Floating Pt. inst.executed */
   {PAPI_TOT_CYC, {0, {PAPI_NATIVE_MASK | 0, PAPI_NULL}}},   /* Total cycles */
   {PAPI_LST_INS, {DERIVED_ADD, {PAPI_NATIVE_MASK | 18, PAPI_NATIVE_MASK | 19, PAPI_NULL}}},
   /* Total load/store inst. exec */
   {0, {0, {0, 0}}}             /* The END */
};

/* Low level functions, should not handle errors, just return codes. */

/* Utility functions */

static int _internal_scan_cpu_info(inventory_t * item, void *foo)
{
#define IPSTRPOS 8
   char *ip_str_pos = &_papi_hwi_system_info.hw_info.model_string[IPSTRPOS];
   char *cptr;
   int i;
   /* The information gathered here will state the information from
    * the last CPU board and the last CPU chip. Previous ones will
    * be overwritten. For systems where each CPU returns an entry, that is.
    *
    * The model string gets two contributions. One is from the 
    *  CPU board (IP-version) and the other is from the chip (R10k/R12k)
    *  Let the first IPSTRPOS characters be the chip part and the 
    *  remaining be the board part 
    *     0....5....0....5....0....
    *     R10000  IP27                                  
    *
    * The model integer will be two parts. The lower 8 bits are used to
    * store the CPU chip type (R10k/R12k) using the codes available in
    * sys/sbd.h. The upper bits store the board type (e.g. P25 using
    * the codes available in sys/cpu.h                                */
#define SETHIGHBITS(var,patt) var = (var & 0x000000ff) | (patt << 8 )
#define SETLOWBITS(var,patt)  var = (var & 0xffffff00) | (patt & 0xff)

   /* If the string is untouched fill the chip part with spaces */
   if ((item->inv_class == INV_PROCESSOR) &&
       (!_papi_hwi_system_info.hw_info.model_string[0])) {
      for (cptr = _papi_hwi_system_info.hw_info.model_string;
           cptr != ip_str_pos; *cptr++ = ' ');
   }

   if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUBOARD)) {
      SUBDBG("scan_system_info(%p,%p) Board: %ld, %d, %ld\n",
             item, foo, item->inv_controller, item->inv_state, item->inv_unit);

      _papi_hwi_system_info.hw_info.mhz = (int) item->inv_controller;

      switch (item->inv_state) {        /* See /usr/include/sys for new models */
      case INV_IPMHSIMBOARD:
         strcpy(ip_str_pos, "IPMHSIM");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, 0);
         break;
      case INV_IP19BOARD:
         strcpy(ip_str_pos, "IP19");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP19);
         break;
      case INV_IP20BOARD:
         strcpy(ip_str_pos, "IP20");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP20);
         break;
      case INV_IP21BOARD:
         strcpy(ip_str_pos, "IP21");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP21);
         break;
      case INV_IP22BOARD:
         strcpy(ip_str_pos, "IP22");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP22);
         break;
      case INV_IP25BOARD:
         strcpy(ip_str_pos, "IP25");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP25);
         break;
      case INV_IP26BOARD:
         strcpy(ip_str_pos, "IP26");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP26);
         break;
      case INV_IP27BOARD:
         strcpy(ip_str_pos, "IP27");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP27);
         break;
      case INV_IP28BOARD:
         strcpy(ip_str_pos, "IP28");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP28);
         break;
/* sys/cpu.h and sys/invent.h varies slightly between systems so
   protect the newer entries in case they are not defined */
#if defined(INV_IP30BOARD) && defined(CPU_IP30)
      case INV_IP30BOARD:
         strcpy(ip_str_pos, "IP30");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP30);
         break;
#endif
#if defined(INV_IP32BOARD) && defined(CPU_IP32)
      case INV_IP32BOARD:
         strcpy(ip_str_pos, "IP32");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP32);
         break;
#endif
#if defined(INV_IP33BOARD) && defined(CPU_IP33)
      case INV_IP33BOARD:
         strcpy(ip_str_pos, "IP33");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP33);
         break;
#endif
#if defined(INV_IP35BOARD) && defined(CPU_IP35)
      case INV_IP35BOARD:
         strcpy(ip_str_pos, "IP35");
         SETHIGHBITS(_papi_hwi_system_info.hw_info.model, CPU_IP35);
         break;
#endif
      default:
         strcpy(ip_str_pos, "Unknown cpu board");
         _papi_hwi_system_info.hw_info.model = PAPI_EINVAL;
      }
      SUBDBG("scan_system_info:       Board: 0x%x, %s\n",
             _papi_hwi_system_info.hw_info.model,
             _papi_hwi_system_info.hw_info.model_string);
   }

   if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUCHIP)) {
      unsigned int imp, majrev, minrev;

      SUBDBG("scan_system_info(%p,%p) CPU: %ld, %d, %ld\n",
             item, foo, item->inv_controller, item->inv_state, item->inv_unit);

      imp = (item->inv_state & C0_IMPMASK) >> C0_IMPSHIFT;
      majrev = (item->inv_state & C0_MAJREVMASK) >> C0_MAJREVSHIFT;
      minrev = (item->inv_state & C0_MINREVMASK) >> C0_MINREVSHIFT;

      _papi_hwi_system_info.hw_info.revision = (float) majrev + ((float) minrev * 0.01);

      SETLOWBITS(_papi_hwi_system_info.hw_info.model, imp);
      switch (imp) {            /* We fill a name here and then remove any \0 characters */
      case C0_IMP_R10000:
         strncpy(_papi_hwi_system_info.hw_info.model_string, "R10000", IPSTRPOS);
         break;
      case C0_IMP_R12000:
         strncpy(_papi_hwi_system_info.hw_info.model_string, "R12000", IPSTRPOS);
         break;
#ifdef C0_IMP_R14000
      case C0_IMP_R14000:
         strncpy(_papi_hwi_system_info.hw_info.model_string, "R14000", IPSTRPOS);
         break;
#endif
      default:
         return (PAPI_ESBSTR);
      }
      /* Remove the \0 inserted above to be able to print the board version */
      for (i = strlen(_papi_hwi_system_info.hw_info.model_string); i < IPSTRPOS; i++)
         _papi_hwi_system_info.hw_info.model_string[i] = ' ';

      SUBDBG("scan_system_info:       CPU: 0x%.2x, 0x%.2x, 0x%.2x, %s\n",
             imp, majrev, minrev, _papi_hwi_system_info.hw_info.model_string);
   }

   /* FPU CHIP is not used now, but who knows about the future? */
   if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_FPUCHIP)) {
      unsigned int imp, majrev, minrev;

      SUBDBG("scan_system_info(%p,%p) FPU: %ld, %d, %ld\n",
             item, foo, item->inv_controller, item->inv_state, item->inv_unit);
      imp = (item->inv_state & C0_IMPMASK) >> C0_IMPSHIFT;
      majrev = (item->inv_state & C0_MAJREVMASK) >> C0_MAJREVSHIFT;
      minrev = (item->inv_state & C0_MINREVMASK) >> C0_MINREVSHIFT;
      SUBDBG("scan_system_info        FPU: 0x%.2x, 0x%.2x, 0x%.2x\n",
             imp, majrev, minrev);
   }
   return (0);
}

static int set_domain(hwd_control_state_t * this_state, int domain)
{
   int i, selector = this_state->selector, mode = 0;
   hwperf_profevctrarg_t *arg = &this_state->counter_cmd;
   int did = 0;

   if (domain & PAPI_DOM_USER) {
      mode |= HWPERF_CNTEN_U;
      did = 1;
   }
   if (domain & PAPI_DOM_KERNEL) {
      mode |= HWPERF_CNTEN_K;
      did = 1;
   }
   if (domain & PAPI_DOM_OTHER) {
      mode |= HWPERF_CNTEN_E;
      did = 1;
   }
   if (!did)
      return (PAPI_EINVAL);

   for (i = 0; i < HWPERF_EVENTMAX; i++) {
      if (selector & (1 << i))
         arg->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode = mode;
   }
   return (PAPI_OK);
}

/* This function now do nothing */
static int set_granularity(hwd_control_state_t * this_state, int domain)
{
   switch (domain) {
   case PAPI_GRN_PROCG:
   case PAPI_GRN_SYS:
   case PAPI_GRN_SYS_CPU:
   case PAPI_GRN_PROC:
      return(PAPI_ESBSTR);
   case PAPI_GRN_THR:
      break;
   default:
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
}

/*
static int set_inherit(EventSetInfo_t *zero, pid_t pid)
{
  int retval;

  hwd_control_state_t *current_state = &zero->machdep;
  if ((pid == PAPI_INHERIT_ALL) || (pid == PAPI_INHERIT_NONE))
    return(PAPI_ESBSTR);

  retval = ioctl(current_state->fd,PIOCSAVECCNTRS,pid);
  if (retval == -1)
    return(PAPI_ESYS);

  return(PAPI_OK);
}
*/

static int set_default_domain(hwd_control_state_t * current_state, int domain)
{
   return (set_domain(current_state, domain));
}

static int set_default_granularity(hwd_control_state_t * current_state, int granularity)
{
   return (set_granularity(current_state, granularity));
}

static int _internal_get_system_info(void)
{
   int fd, retval, nn = 0;
   pid_t pid;
   char pidstr[PAPI_MAX_STR_LEN];
   char pname[PAPI_HUGE_STR_LEN];
   prpsinfo_t psi;

   if (scaninvent(_internal_scan_cpu_info, NULL) == -1)
      return (PAPI_ESBSTR);

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);

   sprintf(pidstr, "/proc/%05d", (int) pid);
   if ((fd = open(pidstr, O_RDONLY)) == -1)
      return (PAPI_ESYS);

   if (ioctl(fd, PIOCPSINFO, (void *) &psi) == -1)
      return (PAPI_ESYS);

   close(fd);

   /* EXEinfo */

   /* Cut off any arguments to exe */
   {
     char *tmp;
     tmp = strchr(psi.pr_psargs, ' ');
     if (tmp != NULL)
       *tmp = '\0';
   }

   if (realpath(psi.pr_psargs,pname))
     strncpy(_papi_hwi_system_info.exe_info.fullname, pname, PAPI_HUGE_STR_LEN);
   else
     strncpy(_papi_hwi_system_info.exe_info.fullname, psi.pr_psargs, PAPI_HUGE_STR_LEN);

   strcpy(_papi_hwi_system_info.exe_info.address_info.name,psi.pr_fname);

   /* Preload info */
   strcpy(_papi_hwi_system_info.preload_info.lib_preload_env, "_RLD_LIST");
   _papi_hwi_system_info.preload_info.lib_preload_sep = ':';
   strcpy(_papi_hwi_system_info.preload_info.lib_dir_env, "LD_LIBRARY_PATH");
   _papi_hwi_system_info.preload_info.lib_dir_sep = ':';

   /* HWinfo */

   _papi_hwi_system_info.hw_info.totalcpus = sysmp(MP_NPROCS);
   if (_papi_hwi_system_info.hw_info.totalcpus > 1) {
      _papi_hwi_system_info.hw_info.ncpu = 2;
      _papi_hwi_system_info.hw_info.nnodes = _papi_hwi_system_info.hw_info.totalcpus /
          _papi_hwi_system_info.hw_info.ncpu;
   } else {
      _papi_hwi_system_info.hw_info.ncpu = 0;
      _papi_hwi_system_info.hw_info.nnodes = 0;
   }

   _papi_hwi_system_info.hw_info.vendor = -1;
   strcpy(_papi_hwi_system_info.hw_info.vendor_string, "MIPS");

   /* Substrate info */

   strcpy(_papi_hwi_system_info.sub_info.name, "$Id$");
   strcpy(_papi_hwi_system_info.sub_info.version, "$Revision$");
   _papi_hwi_system_info.sub_info.num_cntrs = 2;
   _papi_hwi_system_info.sub_info.num_mpx_cntrs = HWPERF_EVENTMAX;
   _papi_hwi_system_info.sub_info.fast_counter_read = 0;
   _papi_hwi_system_info.sub_info.fast_real_timer = 1;
   _papi_hwi_system_info.sub_info.fast_virtual_timer = 0;
   _papi_hwi_system_info.sub_info.default_domain = PAPI_DOM_USER;
   _papi_hwi_system_info.sub_info.available_domains = PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_OTHER|PAPI_DOM_SUPERVISOR;
   _papi_hwi_system_info.sub_info.hardware_intr = 1;
   _papi_hwi_system_info.sub_info.kernel_multiplex = 1;
   _papi_hwi_system_info.sub_info.multiplex_timer_us = 1000000/sysconf(_SC_CLK_TCK);
   
   retval = _papi_hwd_update_shlib_info();
   if (retval != PAPI_OK) 
   {
      printf("retval=%d\n", retval);
   }
/* set text start address and end address, etc */
/*
   _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) & _ftext;
   _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t) & _etext;
   _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) & _fdata;
   _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) & _edata;
   _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) & _fbss;
   _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) & _end;
*/


   if ((_papi_hwi_system_info.hw_info.model & 0xff) == C0_IMP_R10000) {
      preset_search_map = findem_r10k;
      native_table = r10k_native_events_table;
      nn = sizeof(r10k_native_events_table)/sizeof(char *)-1;
   } else if ((_papi_hwi_system_info.hw_info.model & 0xff) == C0_IMP_R12000) {
      preset_search_map = findem_r12k;
      native_table = r12k_native_events_table;
      nn = sizeof(r12k_native_events_table)/sizeof(char *)-1;
   }
#ifdef C0_IMP_R14000
   else if ((_papi_hwi_system_info.hw_info.model & 0xff) == C0_IMP_R14000) {
      preset_search_map = findem_r12k;
      native_table = r12k_native_events_table;
      nn = sizeof(r12k_native_events_table)/sizeof(char *)-1;
   }
#endif

/* setup_all_presets is in papi_preset.c */
   retval = _papi_hwi_setup_all_presets(preset_search_map, NULL);
   if (retval)
      return (retval);
   _papi_hwi_system_info.sub_info.num_native_events = nn;
   return (PAPI_OK);
}

long long _papi_hwd_get_real_usec(void)
{
   timespec_t t;
   long long retval;

   if (clock_gettime(CLOCK_SGI_CYCLE, &t) == -1)
      return (PAPI_ESYS);

   retval = ((long long) t.tv_sec * (long long) 1000000) + (long long) (t.tv_nsec / 1000);
   return (retval);
}

long long _papi_hwd_get_real_cycles(void)
{
   long long retval;

   retval = _papi_hwd_get_real_usec() * (long long) _papi_hwi_system_info.hw_info.mhz;
   return (retval);
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   long long retval;
   struct tms buffer;

   times(&buffer);
   SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime);
   retval = (long long)((buffer.tms_utime+buffer.tms_stime)*
     (1000000/CLK_TCK));
   return (retval);
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
   return (_papi_hwd_get_virt_usec(ctx) * (long long)_papi_hwi_system_info.hw_info.mhz);
}

volatile int lock[PAPI_MAX_LOCK] = { 0, };

static void lock_init(void)
{
   int lck;
   for (lck = 0; lck < PAPI_MAX_LOCK; lck++)
      lock[lck] = 0;
}

/* this function is called by PAPI_library_init */
#ifndef PAPI_NO_VECTOR
papi_svector_t _irix_mips_table[] = {
 {(void (*)())_papi_hwd_get_overflow_address,VEC_PAPI_HWD_GET_OVERFLOW_ADDRESS},
 {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
 {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
 {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
 {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
 {(void (*)())set_domain, VEC_PAPI_HWD_SET_DOMAIN},
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {NULL, VEC_PAPI_END}
};
#endif


int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
   int retval;

   /* Fill in what we can of the papi_system_info. */

#ifndef PAPI_NO_VECTOR
   retval = _papi_hwi_setup_vector_table(vtable, _irix_mips_table);
   if ( retval != PAPI_OK ) return(retval);
#endif


   retval = _internal_get_system_info();
   if (retval)
      return (retval);
   /* the second argument isn't really used in this implementation of the call */
   retval = _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info,
      _papi_hwi_system_info.hw_info.vendor);
   if (retval)
      return (retval);

   lock_init();

   SUBDBG("Found %d %s %s CPU's at %f Mhz.\n",
          _papi_hwi_system_info.hw_info.totalcpus,
          _papi_hwi_system_info.hw_info.vendor_string,
          _papi_hwi_system_info.hw_info.model_string, _papi_hwi_system_info.hw_info.mhz);

   return (PAPI_OK);
}

/* this function is called by _papi_hwi_initialize_thread in papi_internal.c
   this function will generate the file descriptor for hardware counter
   control
*/
int _papi_hwd_init(hwd_context_t * ctx)
{
   char pidstr[PAPI_MAX_STR_LEN];
   hwperf_profevctrarg_t args;
   hwperf_eventctrl_t counter_controls;
   int i, fd, gen;

   memset(&args, 0x0, sizeof(args));

   sprintf(pidstr, "/proc/%05d", (int) getpid());
   if ((fd = open(pidstr, O_RDONLY)) == -1)
      return (PAPI_ESYS);

   if ((gen = ioctl(fd, PIOCGETEVCTRL, (void *) &counter_controls)) == -1) {
      for (i = 0; i < HWPERF_EVENTMAX; i++)
         args.hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode = HWPERF_CNTEN_U;

      if ((gen = ioctl(fd, PIOCENEVCTRS, (void *) &args)) == -1) {
         close(fd);
         return (PAPI_ESYS);
      }
   }

   if (gen <= 0) {
      close(fd);
      return (PAPI_EMISC);
   }
   ctx->fd = fd;

   return (PAPI_OK);
}

/* debug function */
void dump_cmd(hwperf_profevctrarg_t * t)
{
   int i;

   SUBDBG("Command block at %p: Signal %d\n", t, t->hwp_ovflw_sig);
   for (i = 0; i < HWPERF_EVENTMAX; i++) {
      if (t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode)
	SUBDBG(
                 "Event %d: hwp_ev %d hwp_ie %d hwp_mode %d hwp_ovflw_freq %d\n", i,
                       (int) t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ev,
                       (int) t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ie,
                       (int) t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode,
                       (int) t->hwp_ovflw_freq[i]);
   }
}

int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int retval;

   if ((retval = _papi_hwd_stop(ctx, ctrl)) != PAPI_OK)
      return retval;
   if ((retval = _papi_hwd_start(ctx, ctrl)) != PAPI_OK)
      return retval;
   return (PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **events, int flags)
{
   int retval, index, selector;

   /* now read the counter values */
   retval = ioctl(ctx->fd, PIOCGETEVCTRS, (void *) &ctrl->cntrs_read);
   if (retval < 0)
      return (PAPI_ESYS);
#if defined(DEBUG)
if (ISLEVEL(DEBUG_SUBSTRATE)) {
	int i;
	for (i=0;i<HWPERF_EVENTMAX;i++) {
	SUBDBG("Kernel values index %d: Value %lld\n",i,ctrl->cntrs_read.hwp_evctr[i]);
	}
	}
#endif

/* generation number should be the same */
   if (retval != ctrl->generation) {
      PAPIERROR("This process lost access to the event counters");
      return (PAPI_ESBSTR);
   }
   /* adjust the read value by how many events in count 0 and count 1 */
   SUBDBG("(%d,%d)\n",ctrl->num_on_counter[0],ctrl->num_on_counter[1]);
	if (ctrl->num_on_counter[0]>1 || ctrl->num_on_counter[1]>1) {
      selector = ctrl->selector;
      while ( selector ) {
         index = ffs(selector)-1;
         if (index > HWPERF_MAXEVENT) {
            ctrl->cntrs_read.hwp_evctr[index] *= ctrl->num_on_counter[1];
         } else {
            ctrl->cntrs_read.hwp_evctr[index] *= ctrl->num_on_counter[0];
         }
         selector ^= 1<<index;
      }
   }
#if defined(DEBUG)
if (ISLEVEL(DEBUG_SUBSTRATE)) {
        int i;
        for (i=0;i<HWPERF_EVENTMAX;i++) {
        SUBDBG("Scaled values index %d: Value %lld\n",i,ctrl->cntrs_read.hwp_evctr[i]);
        }
        }
#endif

/* set the buffer address */
   *events = (long long *) ctrl->cntrs_read.hwp_evctr;

   return (PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
   switch (code) {
   case PAPI_DEF_MPX_USEC:
   	return(PAPI_ESBSTR);
   case PAPI_MULTIPLEX:
     option->domain.ESI->machdep.multiplexed = 1;
     return(PAPI_OK);
   case PAPI_DEFDOM:
      return (set_default_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DOMAIN:
      return (set_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DEFGRN:
      return (set_default_granularity
              (&option->domain.ESI->machdep, option->granularity.granularity));
   case PAPI_GRANUL:
      return (set_granularity
              (&option->granularity.ESI->machdep, option->granularity.granularity));
   default:
      return (PAPI_EINVAL);
   }
}

/* close the file descriptor */
int _papi_hwd_shutdown(hwd_context_t * ctx)
{
   close(ctx->fd);
   return (PAPI_OK);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *info)
{
   _papi_hwi_context_t ctx;
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread = NULL;
   int overflow_vector=0, generation2,i;
   hwperf_cntr_t cnts;
   hwd_context_t *hwd_ctx;
   hwd_control_state_t *machdep;
   

   thread = _papi_hwi_lookup_thread();
   if (thread == NULL)
     return;

   ESI = (EventSetInfo_t *) thread->running_eventset;
   if ((ESI == NULL) || ((ESI->state & PAPI_OVERFLOWING) == 0))
     {
       OVFDBG("Either no eventset or eventset not set to overflow.\n");
       return;
     }

   if (ESI->master != thread)
     {
       PAPIERROR("eventset->thread 0x%lx vs. current thread 0x%lx mismatch",ESI->master,thread);
       return;
     }

   hwd_ctx = &thread->context;
   machdep = &ESI->machdep;

   ctx.si = si;
   ctx.ucontext = info;

   if ((generation2 = ioctl(hwd_ctx->fd, PIOCGETEVCTRS, (void *)&cnts)) < 0) {
       PAPIERROR("ioctl(PIOCGETEVCTRS) errno %d",errno);
       return;
   }

   for (i=0; i < HWPERF_EVENTMAX; i++) {
      if (machdep->counter_cmd.hwp_ovflw_freq[i] &&
      machdep->counter_cmd.hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ie)
      {
         if (cnts.hwp_evctr[i]/ machdep->counter_cmd.hwp_ovflw_freq[i] >
             machdep->cntrs_last_read.hwp_evctr[i]/ machdep->counter_cmd.hwp_ovflw_freq[i])
            overflow_vector ^= 1<< i;
      }
   }
   machdep->cntrs_last_read = cnts;
/*   if (overflow_vector)*/
      _papi_hwi_dispatch_overflow_signal((void *) &ctx, NULL, overflow_vector, 0, &thread);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   hwperf_profevctrarg_t *arg = &this_state->counter_cmd;
   int hwcntr, retval = PAPI_OK, i;
/*
  if ((this_state->num_on_counter[0] > 1) || 
      (this_state->num_on_counter[1] > 1))
    return(PAPI_ECNFLCT);
   if (ESI->overflow.event_counter >1) return(PAPI_ECNFLCT);
*/
   if (threshold == 0) {
      this_state->overflow_event_count--;
      if (this_state->overflow_event_count==0)
         arg->hwp_ovflw_sig = 0;
      hwcntr = ESI->EventInfoArray[EventIndex].pos[0];
      arg->hwp_evctrargs.hwp_evctrl[hwcntr].hwperf_creg.hwp_ie = 0;
      arg->hwp_ovflw_freq[hwcntr] = 0;

      _papi_hwi_lock(INTERNAL_LOCK);
      _papi_hwi_using_signal--;
      if (_papi_hwi_using_signal == 0) {
         if (sigaction(_papi_hwi_system_info.sub_info.hardware_intr_sig, NULL, NULL) == -1)
            retval = PAPI_ESYS;
      }
      _papi_hwi_unlock(INTERNAL_LOCK);
   } else {
      struct sigaction act;
      void *tmp;

      tmp = (void *) signal(_papi_hwi_system_info.sub_info.hardware_intr_sig, SIG_IGN);
      if ((tmp != (void *) SIG_DFL) && (tmp != (void *) _papi_hwd_dispatch_timer))
         return (PAPI_EMISC);

      memset(&act, 0x0, sizeof(struct sigaction));
      act.sa_handler = _papi_hwd_dispatch_timer;
      act.sa_flags = SA_RESTART;
      if (sigaction(_papi_hwi_system_info.sub_info.hardware_intr_sig, &act, NULL) == -1)
         return (PAPI_ESYS);

      arg->hwp_ovflw_sig = _papi_hwi_system_info.sub_info.hardware_intr_sig;
      hwcntr = ESI->EventInfoArray[EventIndex].pos[0];
      this_state->overflow_event_count++;
      /* when the user set overflow on two or more events and these
         events are counted by different hardware counters, the result
         of one event will be abnormal, so we disable multiple overflow
         on different hardware counters. If these events are counted by
         one hardware counter, we accept it */
     if (this_state->overflow_event_count > 1) {
        if (hwcntr >= HWPERF_CNT1BASE ) { 
           for (i = 0; i < HWPERF_CNT1BASE; i++) {
              if (arg->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ie &&
                  arg->hwp_ovflw_freq[i] )
                 return (PAPI_ESBSTR);
           }
        } else {
           for (i = HWPERF_CNT1BASE; i < HWPERF_EVENTMAX; i++) {
              if (arg->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ie &&
                  arg->hwp_ovflw_freq[i] )
                 return (PAPI_ESBSTR);
           }
           
        }
     }
/*
      this_state->overflow_index = hwcntr;
      this_state->overflow_threshold = threshold;
*/
      /* set the threshold and interrupt flag */
      arg->hwp_evctrargs.hwp_evctrl[hwcntr].hwperf_creg.hwp_ie = 1;
      arg->hwp_ovflw_freq[hwcntr] = (int) threshold;
      if (hwcntr > HWPERF_MAXEVENT) {
         arg->hwp_ovflw_freq[hwcntr] = (int) threshold
                                      /this_state->num_on_counter[1];
      } else {
         arg->hwp_ovflw_freq[hwcntr] = (int) threshold
                                     /this_state->num_on_counter[0];
      }
      _papi_hwi_lock(INTERNAL_LOCK);
      _papi_hwi_using_signal++;
      _papi_hwi_unlock(INTERNAL_LOCK);
   }

   return (retval);
}

void *_papi_hwd_get_overflow_address(void *context)
{
   struct sigcontext *info = (struct sigcontext *) context;

   return ((void *) info->sc_pc);
}

/* start the hardware counting */
int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int retval;

   retval = ioctl(ctx->fd, PIOCSETEVCTRL, &ctrl->counter_cmd);

   if (retval <= 0) {
      if (retval < 0)
         return (PAPI_ESYS);
      else
         return (PAPI_EMISC);
   }
/* save the generation number */
   ctrl->generation = retval;

   return (PAPI_OK);
}

/* stop the counting */
int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{

   if ((ioctl(ctx->fd, PIOCRELEVCTRS)) < 0) {
      perror("prioctl PIOCRELEVCTRS returns error");
      return PAPI_ESYS;
   }
   return PAPI_OK;
}

int _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   return PAPI_OK;
}

/* this function will be called when adding events to the eventset and
   deleting events from the eventset
*/
int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
              NativeInfo_t * native, int count,  hwd_context_t * ctx)
{
  int index, i, selector = 0, mode = 0, threshold, num_on_cntr[2] = { 0, 0 };
   hwperf_eventctrl_t *to = &this_state->counter_cmd.hwp_evctrargs;

   /* Compute conflicts if we are not kernel multiplexing */

   if (this_state->multiplexed == 0)
     {
       for (i = 0; i < count; i++) 
	 {
	   index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
	   if (index > HWPERF_MAXEVENT) 
	     num_on_cntr[1]++;
	   else
	     num_on_cntr[0]++;
	 SUBDBG("Not multiplexed (%d,%d)\n",num_on_cntr[0],num_on_cntr[1]); 
	}
       if ((num_on_cntr[1] > 1) || (num_on_cntr[0] > 1))
	 return(PAPI_ECNFLCT);
     }

   memset(to, 0, sizeof(hwperf_eventctrl_t));

/*
   this_state->counter_cmd.hwp_ovflw_sig=0;
*/

   if (_papi_hwi_system_info.sub_info.default_domain & PAPI_DOM_USER) 
      mode |= HWPERF_CNTEN_U;
   if (_papi_hwi_system_info.sub_info.default_domain & PAPI_DOM_KERNEL)
      mode |= HWPERF_CNTEN_K;
   if (_papi_hwi_system_info.sub_info.default_domain & PAPI_DOM_OTHER)
     mode |= HWPERF_CNTEN_E;
   if (_papi_hwi_system_info.sub_info.default_domain & PAPI_DOM_SUPERVISOR)
     mode |= HWPERF_CNTEN_S;

   this_state->num_on_counter[0]=0;
   this_state->num_on_counter[1]=0;
   for (i = 0; i < count; i++) {
      index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
      selector |= 1 << index;
      if (index > HWPERF_MAXEVENT) {
         to->hwp_evctrl[index].hwperf_creg.hwp_ev = index - HWPERF_CNT1BASE;
         this_state->num_on_counter[1]++;
      } else {
         to->hwp_evctrl[index].hwperf_creg.hwp_ev = index;
         this_state->num_on_counter[0]++;
      }
      native[i].ni_position = index;
      to->hwp_evctrl[index].hwperf_creg.hwp_mode = mode;
      SUBDBG("update_control_state selector=0x%x index=%d mode=0x%x ev=%d (%d,%d)\n", selector, index, mode, to->hwp_evctrl[index].hwperf_creg.hwp_ev, this_state->num_on_counter[0],this_state->num_on_counter[1]);
   }
   this_state->selector = selector;
   if (this_state->overflow_event_count) {
      /* adjust the overflow threshold because of the bug in IRIX */
/*
      hwcntr = this_state->overflow_index;
      threshold = this_state->overflow_threshold;
*/
      for (i=0; i < HWPERF_EVENTMAX; i++) {
         if (this_state->counter_cmd.hwp_ovflw_freq[i] && 
      this_state->counter_cmd.hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ie) 
         {
           threshold=this_state->counter_cmd.hwp_ovflw_freq[i];
           if (i > HWPERF_MAXEVENT) {
              this_state->counter_cmd.hwp_ovflw_freq[i] = (int) threshold
                                      /this_state->num_on_counter[1];
           } else {
              this_state->counter_cmd.hwp_ovflw_freq[i] = (int) threshold
                                     /this_state->num_on_counter[0];
           }
         }
      }
   }


   return (PAPI_OK);
}

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   int nidx;

   nidx = EventCode ^ PAPI_NATIVE_MASK;
   if (nidx >= 0 && nidx < PAPI_MAX_NATIVE_EVENTS)
      return (native_table[nidx]);
   else
      return NULL;
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (_papi_hwd_ntv_code_to_name(EventCode));
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

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
  char buf[128];

  if ( count == 0 ) return(0);
  
  sprintf(buf, "Event: %d", *bits);
  strncpy(names, buf, name_len);

  return(1);
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
  *bits = EventCode;  
  return(PAPI_OK);
}

void * dladdr(void *address, struct Dl_info *dl)
{
   return( _rld_new_interface(_RLD_DLADDR,address,dl));
}

const char * getbasename(const char *fname)
{
    const char *temp;

    temp = strrchr(fname, '/');
    if( temp == NULL) {temp=fname; return temp;}
       else return temp+1;
}

int _papi_hwd_update_shlib_info(void)
{
   char procfile[100];
   prmap_t *p;
   struct Dl_info dlip;
   void * vaddr;
   int i, nmaps, err, fd, nmaps_allocd, count, t_index;
   PAPI_address_map_t *tmp = NULL;

   /* Construct the name of our own "/proc/${PID}" file, then open it. */
   sprintf(procfile, "/proc/%d", getpid());
   fd = open(procfile, O_RDONLY);
   if (fd < 0)
      return(PAPI_ESYS);
   /* Find out (approximately) how many map entries we have. */
   err = ioctl(fd, PIOCNMAP, &nmaps);
   if (err < 0) {
      return(PAPI_ESYS);
   }

   /* create space to hold that many entries, plus a generous buffer,
    * since PIOCNMAP can lie.
    */
   nmaps_allocd = 2 * nmaps + 10;
   p = (prmap_t *) papi_calloc(nmaps_allocd, sizeof(prmap_t));
   if (p == NULL)
      return(PAPI_ENOMEM);
   err = ioctl(fd, PIOCMAP, p);
   if (err < 0) {
      return(PAPI_ESYS);
   }

   /* Basic cross-check between PIOCNMAP & PIOCMAP. Complicated by the
     * fact that PIOCNMAP *always* seems to report one less than PIOCMAP,
     * so we quietly ignore that little detail...

       The PIOCMAP entry on the proc man page says that
       one more is needed, so a minimum  one more than
       is returned by PIOCNMAP is required.
    */
   for (i = 0; p[i].pr_size != 0 && i < nmaps_allocd; ++i)
   ; /*empty*/
   if (i!= nmaps){ 
      printf(" i=%d nmaps=%d \n", i, nmaps);
   }

   count=0;
   t_index=0;
   for (i = 0; p[i].pr_size != 0 && i < nmaps_allocd; ++i)
   {
      vaddr =  (void *)(1+p[i].pr_vaddr); /* map base address */
      if (dladdr(vaddr, &dlip) > 0 ) 
      {
         count++;
         /* count text segments */
         if ((p[i].pr_mflags & MA_EXEC) && (p[i].pr_mflags & MA_READ) ) {
            if ( !(p[i].pr_mflags & MA_WRITE))
               t_index++;
         }
         if (strcmp(_papi_hwi_system_info.exe_info.address_info.name, 
                          getbasename(dlip.dli_fname))== 0 ) 
         {
            if ( (p[i].pr_mflags & MA_EXEC))
            {
                _papi_hwi_system_info.exe_info.address_info.text_start = 
                                   (caddr_t) p[i].pr_vaddr;
                _papi_hwi_system_info.exe_info.address_info.text_end =
                                   (caddr_t) (p[i].pr_vaddr+p[i].pr_size);
            } else {
                _papi_hwi_system_info.exe_info.address_info.data_start = 
                                   (caddr_t) p[i].pr_vaddr;
                _papi_hwi_system_info.exe_info.address_info.data_end =
                                   (caddr_t) (p[i].pr_vaddr+p[i].pr_size);
            }
         }

      };
   }
   tmp = (PAPI_address_map_t *) papi_calloc(t_index-1, sizeof(PAPI_address_map_t));
   if (tmp == NULL)
      return(PAPI_ENOMEM);
   t_index=-1;

   /* assume the records about the same shared object are saved in the
      array contiguously. This may not be right, but it seems work fine.
    */
   for (i = 0; p[i].pr_size != 0 && i < nmaps_allocd; ++i)
   {
      vaddr =  (void *)(1+p[i].pr_vaddr); /* map base address */
      if (dladdr(vaddr, &dlip) > 0 ) 
      {
         if (strcmp(_papi_hwi_system_info.exe_info.address_info.name, 
                          getbasename(dlip.dli_fname))== 0 ) 
            continue;
         if ( (p[i].pr_mflags & MA_EXEC)) {
            t_index++;
            tmp[t_index].text_start = (caddr_t) p[i].pr_vaddr;
            tmp[t_index].text_end =(caddr_t) (p[i].pr_vaddr+p[i].pr_size);
            strncpy(tmp[t_index].name, dlip.dli_fname, PAPI_MAX_STR_LEN);
         } else {
            tmp[t_index].data_start = (caddr_t) p[i].pr_vaddr;
            tmp[t_index].data_end = (caddr_t) (p[i].pr_vaddr+p[i].pr_size);
         }
      }
   }
   if (_papi_hwi_system_info.shlib_info.map)
      papi_free(_papi_hwi_system_info.shlib_info.map);
   _papi_hwi_system_info.shlib_info.map = tmp;
   _papi_hwi_system_info.shlib_info.count = t_index+1;
   papi_free(p);

   return(PAPI_OK);
}
