/* Implementation of IRIX platform is straightfoward.
   First read r10k_counters manual page, then you will get an idea about
   the performance counter in IRIX.
*/

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "irix-mips.h"
#include "papi_protos.h"  
int papi_debug;

extern papi_mdi_t _papi_hwi_system_info;

char * (r10k_native_events_table[])= { 
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
 0
};

/* r14k native events are the same as r12k */
char * (r12k_native_events_table[])= { 
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
 0
};

char ** native_table;
hwi_preset_t *preset_search_map;

/* the number in this preset_search map table is the native event index
   in the native event table, when it ORs the NATIVE_MASK, it becomes the
   native event code.
   For example, 25 is the index of native event "Primary_data_cache_misses" 
   in the native event table
*/
hwi_preset_t findem_r10k[] = {
    { PAPI_L1_DCM,0,{NATIVE_MASK|25,0}},    /* L1 D-Cache misses */
    { PAPI_L1_ICM,0,{NATIVE_MASK| 9,0}},    /* L1 I-Cache misses */
    { PAPI_L2_DCM,0,{NATIVE_MASK|26,0}},    /* L2 D-Cache misses */
    { PAPI_L2_ICM,0,{NATIVE_MASK|10,0}},    /* L2 I-Cache misses */
    { PAPI_L1_TCM,DERIVED_ADD,{NATIVE_MASK| 9,NATIVE_MASK|25,0}},
                                             /* L1 total */
    { PAPI_L2_TCM,DERIVED_ADD,{NATIVE_MASK|10,NATIVE_MASK|26,0}},  
                                             /* L2 total */
    { PAPI_CA_INV,0,{NATIVE_MASK|13,0}},    /* Cache Line Invalidation*/
    { PAPI_CA_ITV,0,{NATIVE_MASK|12,0}},    /* Cache Line Intervention*/
    { PAPI_TLB_TL,0,{NATIVE_MASK|23,0}},    /* Total TLB misses*/
    { PAPI_CSR_FAL,0,{NATIVE_MASK| 5, 0}},  /* Failed store conditional*/
    { PAPI_CSR_SUC,DERIVED_SUB,{NATIVE_MASK|20,NATIVE_MASK|5,0}}, 
                                             /* Successful store conditional*/
    { PAPI_CSR_TOT,0,{NATIVE_MASK|20,0}},   /* Total store conditional*/
    { PAPI_BR_MSP,0,{NATIVE_MASK|24,0}},    /* Cond. branch inst. mispred*/
    { PAPI_TOT_IIS,0,{NATIVE_MASK| 1,0}},   /* Total inst. issued*/
    { PAPI_TOT_INS,0,{NATIVE_MASK|15,0}},   /* Total inst. executed*/
    { PAPI_FP_INS,0,{NATIVE_MASK|21,0}},    /* Floating Pt. inst. executed*/
    { PAPI_LD_INS,0,{NATIVE_MASK|8,0}},     /* Loads executed*/
    { PAPI_SR_INS,0,{NATIVE_MASK|19,0}},    /* Stores executed*/
    { PAPI_BR_INS,0,{NATIVE_MASK| 6,0}},    /* Branch inst. executed*/
    { PAPI_FLOPS,DERIVED_PS,{NATIVE_MASK|0,NATIVE_MASK|21,0}},     /* FLOPS */
    { PAPI_TOT_CYC,0,{ NATIVE_MASK|0,0}},   /* Total cycles */
    { PAPI_IPS,DERIVED_PS,{NATIVE_MASK|0,NATIVE_MASK|15,0}},       /* IPS */
    { 0, 0, {0, 0}}                        /* The END */
};


hwi_preset_t findem_r12k[] = { /* Shared with R14K */
    { PAPI_L1_DCM,0,{NATIVE_MASK|25,0}},       /* L1 D-Cache misses */
    { PAPI_L1_ICM,0,{NATIVE_MASK| 9,0}},       /* L1 I-Cache misses */
    { PAPI_L2_DCM,0,{NATIVE_MASK|26,0}},       /* L2 D-Cache misses */
    { PAPI_L2_ICM,0,{NATIVE_MASK|10,0}},       /* L2 I-Cache misses */
    { PAPI_L1_TCM,DERIVED_ADD,{ NATIVE_MASK|9,NATIVE_MASK|25,0}}, /* L1 total */
    { PAPI_L2_TCM,DERIVED_ADD,{NATIVE_MASK|10,NATIVE_MASK|26,0}}, /* L2 total */
    { PAPI_CA_INV,0,{NATIVE_MASK|13,0}},       /* Cache Line Invalidation*/
    { PAPI_CA_ITV,0,{NATIVE_MASK|12,0}},       /* Cache Line Intervention*/
    { PAPI_TLB_TL,0,{NATIVE_MASK|23,0}},       /* Total TLB misses*/
    { PAPI_PRF_DM,0,{NATIVE_MASK|17,0}},       /* Prefetch miss */
    { PAPI_CSR_FAL,0,{NATIVE_MASK| 5, 0}},     /* Failed store conditional*/
    { PAPI_CSR_SUC,DERIVED_SUB,{NATIVE_MASK|20,NATIVE_MASK|5,0}},        
                                            /* Successful store conditional*/
    { PAPI_CSR_TOT,0,{NATIVE_MASK|20,0}},      /* Total store conditional*/
    { PAPI_BR_CN,0,{NATIVE_MASK| 6,0}},        /* Cond. branch inst. exe*/
    { PAPI_BR_MSP,0,{NATIVE_MASK|24,0}},       /* Cond. branch inst. mispred*/
    { PAPI_BR_PRC,DERIVED_SUB,{NATIVE_MASK|6,NATIVE_MASK|24,0}},     
                                         /* Cond. branch inst. correctly pred*/
    { PAPI_TOT_IIS,0,{ NATIVE_MASK|1, 0}},     /* Total inst. issued*/
    { PAPI_TOT_INS,0,{NATIVE_MASK|15, 0}},     /* Total inst. executed*/
    { PAPI_FP_INS,0,{NATIVE_MASK|21,0}},       /* Floating Pt. inst.executed*/
    { PAPI_LD_INS,0,{NATIVE_MASK|18,0}},       /* Loads executed*/
    { PAPI_SR_INS,0,{NATIVE_MASK|19,0}},       /* Stores executed*/
    { PAPI_FLOPS,DERIVED_PS,{NATIVE_MASK|0,NATIVE_MASK|21,0}},   /* FLOPS */
    { PAPI_TOT_CYC,0,{ NATIVE_MASK|0, 0}},     /* Total cycles */
    { PAPI_IPS,DERIVED_PS,{NATIVE_MASK|0,NATIVE_MASK|15,0}},     /* IPS */
    { PAPI_LST_INS,DERIVED_ADD,{NATIVE_MASK|18,NATIVE_MASK|19,0}},         
                                             /* Total load/store inst. exec */
    { 0, 0, {0, 0}}                           /* The END */
};


/* Low level functions, should not handle errors, just return codes. */

/* Utility functions */

static int _internal_scan_cpu_info(inventory_t *item, void *foo)
{
  #define IPSTRPOS 8
  char *ip_str_pos=&_papi_hwi_system_info.hw_info.model_string[IPSTRPOS];
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
      (!_papi_hwi_system_info.hw_info.model_string[0]))
    {
      for(cptr=_papi_hwi_system_info.hw_info.model_string; 
	  cptr!=ip_str_pos;*cptr++=' ');
    }

  if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUBOARD)) 
    {
      DBG((stderr,"scan_system_info(%p,%p) Board: %ld, %d, %ld\n",
	   item,foo,item->inv_controller,item->inv_state,item->inv_unit));

      _papi_hwi_system_info.hw_info.mhz = (int)item->inv_controller;

      switch(item->inv_state) {   /* See /usr/include/sys for new models */
      case INV_IPMHSIMBOARD:
	strcpy(ip_str_pos,"IPMHSIM");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,0);
	break;
      case INV_IP19BOARD:
	strcpy(ip_str_pos,"IP19");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP19);
	break;
      case INV_IP20BOARD:
	strcpy(ip_str_pos,"IP20");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP20);
	break;
      case INV_IP21BOARD:
	strcpy(ip_str_pos,"IP21");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP21);
	break;
      case INV_IP22BOARD:
	strcpy(ip_str_pos,"IP22");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP22);
	break;
      case INV_IP25BOARD:
	strcpy(ip_str_pos,"IP25");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP25);
	break;
      case INV_IP26BOARD:
	strcpy(ip_str_pos,"IP26");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP26);
	break;
      case INV_IP27BOARD:
	strcpy(ip_str_pos,"IP27");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP27);
	break;
      case INV_IP28BOARD:
	strcpy(ip_str_pos,"IP28");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP28);
	break;
/* sys/cpu.h and sys/invent.h varies slightly between systems so
   protect the newer entries in case they are not defined */
#if defined(INV_IP30BOARD) && defined(CPU_IP30)
      case INV_IP30BOARD:
	strcpy(ip_str_pos,"IP30");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP30);
	break;
#endif
#if defined(INV_IP32BOARD) && defined(CPU_IP32)
      case INV_IP32BOARD:
	strcpy(ip_str_pos,"IP32");
	SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP32);
	break;
#endif
#if defined(INV_IP33BOARD) && defined(CPU_IP33)
      case INV_IP33BOARD:
        strcpy(ip_str_pos,"IP33");
        SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP33);
        break;
#endif
#if defined(INV_IP35BOARD) && defined(CPU_IP35)
      case INV_IP35BOARD:
        strcpy(ip_str_pos,"IP35");
        SETHIGHBITS(_papi_hwi_system_info.hw_info.model,CPU_IP35);
        break;
#endif
      default:
	strcpy(ip_str_pos,"Unknown cpu board");
	_papi_hwi_system_info.hw_info.model = PAPI_EINVAL;
      }
      DBG((stderr,"scan_system_info:       Board: 0x%x, %s\n",
	   _papi_hwi_system_info.hw_info.model,
	   _papi_hwi_system_info.hw_info.model_string));
    }

  if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUCHIP))
    {
      unsigned int imp,majrev,minrev;

      DBG((stderr,"scan_system_info(%p,%p) CPU: %ld, %d, %ld\n",
	   item,foo,item->inv_controller,item->inv_state,item->inv_unit)); 

      imp=(item->inv_state & C0_IMPMASK ) >> C0_IMPSHIFT;
      majrev=(item->inv_state & C0_MAJREVMASK ) >> C0_MAJREVSHIFT;
      minrev=(item->inv_state & C0_MINREVMASK ) >> C0_MINREVSHIFT;

      _papi_hwi_system_info.hw_info.revision = (float) majrev + 
	((float)minrev*0.01);

      SETLOWBITS(_papi_hwi_system_info.hw_info.model,imp);
      switch (imp)
	 {  /* We fill a name here and then remove any \0 characters */
	 case C0_IMP_R10000:
	   strncpy(_papi_hwi_system_info.hw_info.model_string,"R10000",IPSTRPOS);
	   _papi_hwi_system_info.num_gp_cntrs = 2;
	   break;
	 case C0_IMP_R12000:
	   strncpy(_papi_hwi_system_info.hw_info.model_string,"R12000",IPSTRPOS);
	   _papi_hwi_system_info.num_gp_cntrs = 2;
	   break;
#ifdef C0_IMP_R14000
       case C0_IMP_R14000:
         strncpy(_papi_hwi_system_info.hw_info.model_string,"R14000",IPSTRPOS);
         _papi_hwi_system_info.num_gp_cntrs = 2;
         break;
#endif
	 default:
	   return(PAPI_ESBSTR);
	 }
      /* Remove the \0 inserted above to be able to print the board version */
      for(i=strlen(_papi_hwi_system_info.hw_info.model_string);i<IPSTRPOS;i++)
	_papi_hwi_system_info.hw_info.model_string[i]=' ';

    DBG((stderr,"scan_system_info:       CPU: 0x%.2x, 0x%.2x, 0x%.2x, %s\n",
	 imp,majrev,minrev,_papi_hwi_system_info.hw_info.model_string));
    }	  

  /* FPU CHIP is not used now, but who knows about the future? */
  if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_FPUCHIP))
    {
      unsigned int imp,majrev,minrev;

      DBG((stderr,"scan_system_info(%p,%p) FPU: %ld, %d, %ld\n",
	   item,foo,item->inv_controller,item->inv_state,item->inv_unit)); 
      imp=(item->inv_state & C0_IMPMASK ) >> C0_IMPSHIFT;
      majrev=(item->inv_state & C0_MAJREVMASK ) >> C0_MAJREVSHIFT;
      minrev=(item->inv_state & C0_MINREVMASK ) >> C0_MINREVSHIFT;
      DBG((stderr,"scan_system_info        FPU: 0x%.2x, 0x%.2x, 0x%.2x\n",
	   imp,majrev,minrev))
    }
  return(0);
}

static int set_domain(hwd_control_state_t *this_state, int domain)
{
  int i, selector = this_state->selector, mode = 0;
  hwperf_profevctrarg_t *arg = &this_state->counter_cmd;

  if (domain & PAPI_DOM_USER)
    mode |= HWPERF_CNTEN_U;
  if (domain & PAPI_DOM_KERNEL)
    mode |= HWPERF_CNTEN_K;
  if (domain & PAPI_DOM_OTHER)
    mode |= HWPERF_CNTEN_E;
  
  for (i=0;i<HWPERF_EVENTMAX;i++)
  {
    if (selector & (1 << i))
	  arg->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode = mode;
  }
  return(PAPI_OK);
}

/* This function now do nothing */
static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  switch (domain)
  {
    case PAPI_GRN_THR:
      return(PAPI_OK);
    default:
      return(PAPI_EINVAL);
  }
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

static int set_default_domain(hwd_control_state_t *current_state, int domain)
{
  return(set_domain(current_state,domain));
}

static int set_default_granularity(hwd_control_state_t *current_state, int granularity)
{
  return(set_granularity(current_state,granularity));
}

static int _internal_get_system_info(void)
{
  int fd, retval;
  pid_t pid;
  char pidstr[PAPI_MAX_STR_LEN];
  prpsinfo_t psi;

  if (scaninvent(_internal_scan_cpu_info, NULL) == -1)
    return(PAPI_ESBSTR);

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);

  sprintf(pidstr,"/proc/%05d",(int)pid);
  if ((fd = open(pidstr,O_RDONLY)) == -1)
    return(PAPI_ESYS);

  if (ioctl(fd, PIOCPSINFO, (void *)&psi) == -1)
    return(PAPI_ESYS);
  
  close(fd);
  
  /* EXEinfo */

  if (getcwd(_papi_hwi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == NULL)
    return(PAPI_ESYS);

/*
  _papi_hwi_system_info.hw_info.ncpu = psi.pr_sonproc;
*/
  strcat(_papi_hwi_system_info.exe_info.fullname,"/");
  strcat(_papi_hwi_system_info.exe_info.fullname,psi.pr_fname);
  strncpy(_papi_hwi_system_info.exe_info.name,psi.pr_fname,PAPI_MAX_STR_LEN);

  /* HWinfo */

  _papi_hwi_system_info.hw_info.totalcpus = sysmp(MP_NPROCS);
  if (_papi_hwi_system_info.hw_info.totalcpus > 1)
    {
      _papi_hwi_system_info.hw_info.ncpu = 2;
      _papi_hwi_system_info.hw_info.nnodes = _papi_hwi_system_info.hw_info.totalcpus /
	_papi_hwi_system_info.hw_info.ncpu;
    }
  else
    {
      _papi_hwi_system_info.hw_info.ncpu = 0;
      _papi_hwi_system_info.hw_info.nnodes = 0;
    }

  _papi_hwi_system_info.hw_info.vendor = -1;
  strcpy(_papi_hwi_system_info.hw_info.vendor_string,"MIPS");

  /* Generic info */

  _papi_hwi_system_info.num_cntrs = HWPERF_EVENTMAX;
/*
  _papi_hwi_system_info.hw_info.ncpu = get_cpu();
*/
  _papi_hwi_system_info.supports_hw_overflow = 1;

/* set text start address and end address, etc */
  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t)&_ftext;
  _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t)&_etext;
  _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t)&_fdata;
  _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t)&_edata;
  _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t)&_fbss;
  _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t)&_end;


  if ((_papi_hwi_system_info.hw_info.model & 0xff)  == C0_IMP_R10000)
  {
    preset_search_map = findem_r10k;
    native_table=r10k_native_events_table;
  }
  else if ((_papi_hwi_system_info.hw_info.model & 0xff) == C0_IMP_R12000)
    {
    preset_search_map = findem_r12k;
    native_table=r12k_native_events_table;
    }
#ifdef C0_IMP_R14000
  else if ((_papi_hwi_system_info.hw_info.model & 0xff) == C0_IMP_R14000)
   {
    preset_search_map = findem_r12k;
    native_table=r12k_native_events_table;
   }
#endif

/* setup_all_presets is in papi_preset.c */
  retval = _papi_hwi_setup_all_presets(preset_search_map);
  if (retval)
    return(retval);

  return(PAPI_OK);
} 

u_long_long _papi_hwd_get_real_usec (void)
{
  timespec_t t;
  long long retval;

  if (clock_gettime(CLOCK_SGI_CYCLE, &t) == -1)
    return(PAPI_ESYS);

  retval = ((long long)t.tv_sec * (long long)1000000) + (long long)(t.tv_nsec / 1000);
  return(retval);
}

u_long_long _papi_hwd_get_real_cycles (void)
{
  long long retval;

  retval = _papi_hwd_get_real_usec() * (long long)_papi_hwi_system_info.hw_info.mhz;
  return(retval);
}

u_long_long _papi_hwd_get_virt_usec (const hwd_context_t *ctx)
{
  long long retval;
  struct tms buffer;

  times(&buffer);
  retval = (long long)buffer.tms_utime*(long long)(1000000/CLK_TCK);
  return(retval);
}

u_long_long _papi_hwd_get_virt_cycles (const hwd_context_t *ctx)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_virt_usec(ctx);
  cyc = usec * _papi_hwi_system_info.hw_info.mhz;
  return((long long)cyc);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error: %s",strerror(error));
}

/* this function is called by PAPI_library_init */
int _papi_hwd_init_global(void)
{
  int retval;

  /* Fill in what we can of the papi_system_info. */
  
  retval = _internal_get_system_info();
  if (retval)
    return(retval);

  retval = get_memory_info(&_papi_hwi_system_info.hw_info);
  if (retval)
    return(retval);

  
  DBG((stderr,"Found %d %s %s CPU's at %f Mhz.\n",
       _papi_hwi_system_info.hw_info.totalcpus,
       _papi_hwi_system_info.hw_info.vendor_string,
       _papi_hwi_system_info.hw_info.model_string,
       _papi_hwi_system_info.hw_info.mhz));

  return(PAPI_OK);
}

/*
static int translate_domain(int domain)
{
  int mode = 0;

  if (domain & PAPI_DOM_USER)
    mode |= HWPERF_CNTEN_U;
  if (domain & PAPI_DOM_KERNEL)
    mode |= HWPERF_CNTEN_K;
  if (domain & PAPI_DOM_OTHER)
    mode |= HWPERF_CNTEN_E | HWPERF_CNTEN_S;
  assert(mode);
  return(mode);
}
*/

/* this function is called by _papi_hwi_initialize_thread in papi_internal.c
   this function will generate the file descriptor for hardware counter
   control
*/
int _papi_hwd_init(hwd_context_t *ctx)
{
  char pidstr[PAPI_MAX_STR_LEN];
  hwperf_profevctrarg_t args;
  hwperf_eventctrl_t counter_controls;
  int i, fd, gen;

  memset(&args,0x0,sizeof(args));

  sprintf(pidstr,"/proc/%05d",(int)getpid());
  if ((fd = open(pidstr,O_RDONLY)) == -1)
    return(PAPI_ESYS);

  if ((gen = ioctl(fd, PIOCGETEVCTRL, (void *)&counter_controls)) == -1)
  {
    for (i=0;i<HWPERF_EVENTMAX;i++)
      args.hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode = HWPERF_CNTEN_U;

    if ((gen = ioctl(fd, PIOCENEVCTRS, (void *)&args)) == -1)
    {
      close(fd);
      return(PAPI_ESYS);
    }
  }

  if (gen <= 0)
  {
    close(fd);
    return(PAPI_EMISC);
  }
  ctx->fd = fd;

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, unsigned int event, void *extra, EventInfo_t *out)
{
  return(PAPI_ESBSTR);
}

/* debug function */
void dump_cmd(hwperf_profevctrarg_t *t)
{
  int i;

  fprintf(stderr,"Command block at %p: Signal %d\n",t,t->hwp_ovflw_sig);
  for (i=0;i<HWPERF_EVENTMAX;i++)
  {
    if (t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode)
	fprintf(stderr,
        "Event %d: hwp_ev %d hwp_ie %d hwp_mode %d hwp_ovflw_freq %d\n",i,
		t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ev,
		t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ie,
		t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode,
		t->hwp_ovflw_freq[i]);
  }
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
  int retval;

  if ( (retval=_papi_hwd_stop(ctx, ctrl))!= PAPI_OK)
    return retval;
  if ( (retval=_papi_hwd_start(ctx, ctrl))!= PAPI_OK)
    return retval;
  return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long **events)
{
  int retval;

  /* now read the counter values */
  retval = ioctl(ctx->fd, PIOCGETEVCTRS, (void *)&ctrl->cntrs_read);
  if (retval < 0)
    return(PAPI_ESYS);

/* generation number should be the same */
  if (retval != ctrl->generation) {
    fprintf(stderr,"program lost event counters\n");
    return(PAPI_ESYS);
  }
/* set the buffer address */
  *events = (long long *)ctrl->cntrs_read.hwp_evctr;

  return(PAPI_OK);
}

int _papi_hwd_setmaxmem(){
  return(PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
  switch (code)
  {
    case PAPI_DEFDOM:
      return(set_default_domain(&option->domain.ESI->machdep, option->domain.domain));
    case PAPI_DOMAIN:
      return(set_domain(&option->domain.ESI->machdep, option->domain.domain));
    case PAPI_DEFGRN:
      return(set_default_granularity(&option->domain.ESI->machdep, option->granularity.granularity));
    case PAPI_GRANUL:
      return(set_granularity(&option->granularity.ESI->machdep, option->granularity.granularity));
    default:
      return(PAPI_EINVAL);
  }
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long events[])
{ 
  return(PAPI_ESBSTR);
}

/* close the file descriptor */
int _papi_hwd_shutdown(hwd_context_t *ctx)
{
  close(ctx->fd);
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
  return(PAPI_OK);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, void *info)
{
  _papi_hwi_context_t ctx;

  ctx.si=si;
  ctx.ucontext=info;
/*
  _papi_hwi_dispatch_overflow_signal((void *)info); 
*/
  _papi_hwi_dispatch_overflow_signal((void *)&ctx,_papi_hwi_system_info.supports_hw_overflow, 0, 1);
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  extern int _papi_hwi_using_signal;
  hwd_control_state_t *this_state = &ESI->machdep;
  hwperf_profevctrarg_t *arg = &this_state->counter_cmd;
  int hwcntr, retval = PAPI_OK;

/*
  if ((this_state->num_on_counter[0] > 1) || 
      (this_state->num_on_counter[1] > 1))
    return(PAPI_ECNFLCT);
*/
/*
  if (ESI->overflow.event_counter >1) return(PAPI_ECNFLCT);
*/
  if (threshold == 0)
  {
    arg->hwp_ovflw_sig = 0;
    hwcntr = ESI->EventInfoArray[EventIndex].pos[0];
	arg->hwp_evctrargs.hwp_evctrl[hwcntr].hwperf_creg.hwp_ie = 0;
	arg->hwp_ovflw_freq[hwcntr] = 0;

    _papi_hwd_lock(PAPI_INTERNAL_LOCK);
    _papi_hwi_using_signal--;
    if (_papi_hwi_using_signal == 0)
	{
	  if (sigaction(PAPI_SIGNAL, NULL, NULL) == -1)
	    retval = PAPI_ESYS;
	}
    _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
  }
  else
  {
    struct sigaction act;
    void *tmp;

    tmp = (void *)signal(PAPI_SIGNAL, SIG_IGN);
    if ((tmp != (void *)SIG_DFL) && (tmp != (void *)_papi_hwd_dispatch_timer))
	  return(PAPI_EMISC);

    memset(&act,0x0,sizeof(struct sigaction));
    act.sa_handler = _papi_hwd_dispatch_timer;
    act.sa_flags = SA_RESTART;
    if (sigaction(PAPI_SIGNAL, &act, NULL) == -1)
	  return(PAPI_ESYS);

    arg->hwp_ovflw_sig = PAPI_SIGNAL;
    hwcntr = ESI->EventInfoArray[EventIndex].pos[0];
    /* set the threshold and interrupt flag */
	arg->hwp_evctrargs.hwp_evctrl[hwcntr].hwperf_creg.hwp_ie = 1;
	arg->hwp_ovflw_freq[hwcntr] = (int)threshold;
    _papi_hwd_lock(PAPI_INTERNAL_LOCK);
    _papi_hwi_using_signal++;
    _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
  }

  return(retval);
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  return(PAPI_OK);
}


void *_papi_hwd_get_overflow_address(void *context)
{
  struct sigcontext *info = (struct sigcontext *)context;

  return((void *)info->sc_pc);
}

volatile int lock[PAPI_MAX_LOCK] = {0,};

void _papi_hwd_lock_init(void)
{
}

/* start the hardware counting */
int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
  int retval;

  retval = ioctl(ctx->fd,PIOCSETEVCTRL,&ctrl->counter_cmd);

  if (retval <= 0)
  {
    if (retval < 0)
      return(PAPI_ESYS);
    else
      return(PAPI_EMISC);
  }
/* save the generation number */
  ctrl->generation=retval;

  return(PAPI_OK);
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

int _papi_hwd_update_shlib_info(void)
{
  return PAPI_OK;
}


void _papi_hwd_init_control_state(hwd_control_state_t *ptr)
{
  return;
}

/* this function will be called when adding events to the eventset and
   deleting events from the eventset
*/
int _papi_hwd_update_control_state(hwd_control_state_t *this_state, NativeInfo_t *native, int count)
{
  int index, i, selector=0, mode = 0;
  hwperf_eventctrl_t *to= &this_state->counter_cmd.hwp_evctrargs;

  memset(to, 0, sizeof(hwperf_eventctrl_t));

  if (_papi_hwi_system_info.default_domain & PAPI_DOM_USER)
    mode |= HWPERF_CNTEN_U;
  if (_papi_hwi_system_info.default_domain & PAPI_DOM_KERNEL)
    mode |= HWPERF_CNTEN_K;
  if (_papi_hwi_system_info.default_domain & PAPI_DOM_OTHER)
    mode |= HWPERF_CNTEN_E | HWPERF_CNTEN_S;
 
  for (i=0; i< count; i++) 
  {
    index = native[i].ni_event & NATIVE_AND_MASK;
    selector |= 1<<index;
    DBG((stderr,"update_control_state index = %d mode=0x%x\n",index, mode));
    if (index > HWPERF_MAXEVENT)
      to->hwp_evctrl[index].hwperf_creg.hwp_ev = index-HWPERF_CNT1BASE;
    else
      to->hwp_evctrl[index].hwperf_creg.hwp_ev = index;
    native[i].ni_position = index;

    to->hwp_evctrl[index].hwperf_creg.hwp_mode = mode;
  }
  this_state->selector = selector;
  

  return(PAPI_OK);
}

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI )
{
  return 1;
}

char * _papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
  int nidx;

  nidx= EventCode^NATIVE_MASK;
  if (nidx>=0 && nidx<PAPI_MAX_NATIVE_EVENTS)
    return(native_table[nidx]);
  else return NULL;
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
  return(_papi_hwd_ntv_code_to_name(EventCode));
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer)
{
  int index=*EventCode & NATIVE_AND_MASK;
    
  if(index < MAX_NATIVE_EVENT-1 ) {
    *EventCode=*EventCode+1;
    return(PAPI_OK);
  } else  return(PAPI_ENOEVNT);
}

int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr)
{
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr)
{
}

/* This function examines the event to determine
    if it has a single exclusive mapping.
    Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t *dst)
{
}

/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
}

/* This function removes the counters available to the src event
    from the counters available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
}

/* This function updates the selection status of
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
}

