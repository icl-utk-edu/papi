#include "dadd-alpha.h"

extern EventSetInfo *default_master_eventset;

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };

static hwd_search_t findem_dadd[] = {
  { PAPI_TOT_CYC, VC_TOTAL_CYCLES},
  { PAPI_L1_ICM, VC_NYP_EVENTS },
  { PAPI_L2_TCM, VC_BCACHE_MISSES },
  { PAPI_TLB_DM, VC_TOTAL_DTBMISS },
  { PAPI_BR_UCN, VC_UNCOND_BR_EXECUTED },
  { PAPI_BR_CN, VC_COND_BR_EXECUTED },
  { PAPI_BR_NTK, VC_COND_BR_NOT_TAKEN },
  { PAPI_BR_MSP, VC_COND_BR_MISPREDICTED },
  { PAPI_BR_PRC, VC_COND_BR_PREDICTED },
  { PAPI_TOT_INS, VC_TOTAL_INSTR_EXECUTED },
  { PAPI_TOT_IIS, VC_TOTAL_INSTR_ISSUED },
  { PAPI_FP_INS, VC_FP_INSTR_EXECUTED },
  { PAPI_LD_INS, VC_LOAD_INSTR_EXECUTED },
  { PAPI_SR_INS, VC_STORE_INSTR_EXECUTED },
  { PAPI_LST_INS, VC_TOTAL_LOAD_STORE_EXECUTED },
  { PAPI_SYC_INS, VC_SYNCH_INSTR_EXECUTED },
  { PAPI_FML_INS, VC_FM_INSTR_EXECUTED },
  { PAPI_FAD_INS, VC_FA_INSTR_EXECUTED },
  { PAPI_FDV_INS, VC_FD_INSTR_EXECUTED },
  { PAPI_FSQ_INS, VC_FSQ_INSTR_EXECUTED },
  { PAPI_INT_INS, VC_INT_INSTR_EXECUTED },
  { -1, -1}};

static int setup_all_presets(void)
{
  int first, event, derived, hwnum;
  hwd_search_t *findem;
  char str[PAPI_MAX_STR_LEN];
  int code;

  findem = findem_dadd;
  while ((code = findem->papi_code) != -1)
    {
      int index;

      index = code ^ PRESET_MASK;
      preset_map[index].selector = 1;
      preset_map[index].derived = NOT_DERIVED;
      preset_map[index].operand_index = 0;
      preset_map[index].counter_cmd = findem->dadd_code;
      sprintf(str,"0x%x",findem->dadd_code);
      if (strlen(preset_map[index].note))
        strcat(preset_map[index].note,",");
      strcat(preset_map[index].note,str);
      findem++;
    }
  return(PAPI_OK);
}

static int get_system_info(void)
{
  int fd, retval, family;
  prpsinfo_t info;
  struct cpu_info cpuinfo;
  long proc_type;
  pid_t pid;
  char pname[PAPI_MAX_STR_LEN], *ptr;

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);
  sprintf(pname,"/proc/%05d",(int)pid);

  fd = open(pname,O_RDONLY);
  if (fd == -1)
    return(PAPI_ESYS);
  if (ioctl(fd,PIOCPSINFO,&info) == -1)
    return(PAPI_ESYS);
  close(fd);

  if (getcwd(_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == NULL)
    return(PAPI_ESYS);
  strcat(_papi_system_info.exe_info.fullname,"/");
  strcat(_papi_system_info.exe_info.fullname,info.pr_fname);
  strncpy(_papi_system_info.exe_info.name,info.pr_fname,PAPI_MAX_STR_LEN);

  if (getsysinfo(GSI_CPU_INFO, (char *)&cpuinfo, sizeof(cpuinfo), NULL, NULL, NULL) == -1)
    return PAPI_ESYS;

  if (getsysinfo(GSI_PROC_TYPE, (char *)&proc_type, sizeof(proc_type), 0, 0,0) == -1)
    return PAPI_ESYS;
  proc_type &= 0xffffffff;

  _papi_system_info.cpunum = cpuinfo.current_cpu;
  _papi_system_info.hw_info.mhz = (float)cpuinfo.mhz;
  _papi_system_info.hw_info.ncpu = cpuinfo.cpus_in_box;
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus =
    _papi_system_info.hw_info.ncpu * _papi_system_info.hw_info.nnodes;
  _papi_system_info.hw_info.vendor = -1;
  strcpy(_papi_system_info.hw_info.vendor_string,"Compaq");
  _papi_system_info.hw_info.model = proc_type;
  strcpy(_papi_system_info.hw_info.model_string,"Alpha ");
  family = cpu_implementation_version();

  if (family == 0)
    {
      strcat(_papi_system_info.hw_info.model_string,"21064");
      _papi_system_info.num_cntrs = 2;
      _papi_system_info.num_gp_cntrs = 2;
    }
  if (family == 2)
    {
      strcat(_papi_system_info.hw_info.model_string,"21264");
      _papi_system_info.num_cntrs = 2;
      _papi_system_info.num_gp_cntrs = 2;
    }
  else if (family == 1)
    {
      strcat(_papi_system_info.hw_info.model_string,"21164");
      _papi_system_info.num_cntrs = 3;
      _papi_system_info.num_gp_cntrs = 3;
    }
  else
    return(PAPI_ESBSTR);

  _papi_system_info.num_cntrs = 27;
  retval = setup_all_presets();
  if (retval)
    return(retval);

  return(PAPI_OK);
}


long long _papi_hwd_get_real_usec (void)
{
  struct timespec res;

  if ( (clock_gettime( CLOCK_REALTIME,  &res ) == -1 ) )
        return (PAPI_ESYS);
  return (res.tv_sec * 1000000) + (res.tv_nsec/1000);
}

long long _papi_hwd_get_real_cycles (void)
{
 return((long long) _papi_hwd_get_real_usec() * _papi_system_info.hw_info.mhz);
}

long long _papi_hwd_get_virt_usec (EventSetInfo *zero)
{
  struct rusage res;

  if ( (getrusage ( RUSAGE_SELF, &res )== -1 ) )
        return (PAPI_ESYS);
  return ( (res.ru_utime.tv_sec*1000000)+res.ru_utime.tv_usec);
}

long long _papi_hwd_get_virt_cycles (EventSetInfo *zero)
{
 return((long long) _papi_hwd_get_virt_usec(zero) * _papi_system_info.hw_info.mhz);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error");
}

int _papi_hwd_init_global(void) {
  int retval;

  /* Install termination signal handlers */
  (void) signal(SIGINT,  dadd_terminate_cleanup) ;
  (void) signal(SIGTERM, dadd_terminate_cleanup) ;

  /* Fill in what we can of the papi_system_info. */
  retval = get_system_info();
  if (retval)
    return(retval);

  DBG((stderr,"Found %d %s %s CPU's at %f Mhz.\n",
       _papi_system_info.hw_info.totalcpus,
       _papi_system_info.hw_info.vendor_string,
       _papi_system_info.hw_info.model_string,
       _papi_system_info.hw_info.mhz));

  return(PAPI_OK);
}

int _papi_hwd_init(EventSetInfo *zero)
{
  pid_t pid;
  unsigned char *region_address = NULL;
  hwd_control_state_t *this_state = zero->machdep;

  pid = getpid();
  region_address = dadd_start_monitoring(pid);
  if (region_address == NULL)
    return(PAPI_ESBSTR);
  this_state->ptr_vc = (virtual_counters *)region_address;
  return(PAPI_OK);
}

int _papi_hwd_add_event(hwd_control_state_t *this_state, unsigned int EventCode,
 EventInfo_t *out)
{
  long code;

  if (EventCode & PRESET_MASK)
    {
      int preset_index;

      preset_index = EventCode ^ PRESET_MASK;
      out->selector = preset_map[preset_index].selector;
      if (out->selector == 0)
        return(PAPI_ENOEVNT);
      out->code = EventCode;
      out->command = preset_map[preset_index].counter_cmd;
    }
  else {
    out->code = EventCode;
    out->selector = 1;
  }

  return(PAPI_OK);
}

int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state,
                             unsigned int event, void *extra, EventInfo_t *out)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{
  return(_papi_hwd_reset(ESI, zero));
}


int _papi_hwd_reset(EventSetInfo *ESI, EventSetInfo *zero)
{
  int i;
  unsigned long count;
  virtual_counters *ptr_vc;
  int dadd_code;
  hwd_control_state_t *this_state;

  this_state = (hwd_control_state_t *) (ESI->master)->machdep;
  ptr_vc = this_state->ptr_vc;
  for (i=0; i<ESI->NumberOfEvents; i++) {
    if (ESI->EventInfoArray[i].code != PAPI_NULL) {
      dadd_code = (ESI->EventInfoArray[i]).command;
      if (ptr_vc) {
        memcpy(&count,
       (char *)ptr_vc+sizeof(struct timeval)+sizeof(unsigned long)*dadd_code,
       sizeof(unsigned long)); 
      
/*        switch (dadd_code) {
          case VC_TOTAL_CYCLES:
            count = ptr_vc->vc_total_cycles;
            break;
          case VC_TOTAL_INSTR_EXECUTED:
            count = ptr_vc->vc_total_instr_executed;
            break;
          case VC_TOTAL_INSTR_ISSUED:
            count = ptr_vc->vc_total_instr_issued;
            break;
          case VC_FP_INSTR_EXECUTED:
            count = ptr_vc->vc_fp_instr_executed;
            break;
          default:
            return(PAPI_ENOEVNT);
          }
*/
        }
      else
        return(PAPI_ESBSTR);
      ESI->latest[i] = (long long)(count);
    }
  }
  return(PAPI_OK);
}

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long *events)
{
  int i;
  unsigned long count;
  virtual_counters *ptr_vc;
  int dadd_code;
  hwd_control_state_t *this_state;

  this_state = (ESI->master)->machdep;
  ptr_vc = this_state->ptr_vc;
  for (i=0; i<ESI->NumberOfEvents; i++) {
    if (ESI->EventInfoArray[i].code != PAPI_NULL) {

    dadd_code = (ESI->EventInfoArray[i]).command;
    if (ptr_vc) 
        memcpy(&count,
       (char *)ptr_vc+sizeof(struct timeval)+sizeof(unsigned long)*dadd_code,
       sizeof(unsigned long)); 

/*        switch (dadd_code) {
          case VC_TOTAL_CYCLES:
            count = ptr_vc->vc_total_cycles;
            break;
          case VC_TOTAL_INSTR_EXECUTED:
            count = ptr_vc->vc_total_instr_executed;
            break;
          case VC_TOTAL_INSTR_ISSUED:
            count = ptr_vc->vc_total_instr_issued;
            break;
          case VC_FP_INSTR_EXECUTED:
            count = ptr_vc->vc_fp_instr_executed;
            break;
          default:
            return(PAPI_ENOEVNT);
          }
*/

    else
      return(PAPI_ESBSTR);
    events[i] = ((long long)(count)) - ESI->latest[i];
    }
  }
  return(PAPI_OK);
}

int _papi_hwd_write(EventSetInfo *master, EventSetInfo *ESI, long long events[])
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(EventSetInfo *zero)
{
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
  hwd_control_state_t *current_state=NULL;
  unsigned char *region_address=NULL;
  int retval;
  pid_t pid;

  pid = getpid();

  if (default_master_eventset) {
    current_state = (hwd_control_state_t *)default_master_eventset->machdep;
    if (current_state)
      region_address = (unsigned char *)current_state->ptr_vc;
    if (region_address) {
      retval = dadd_stop_monitoring(pid, region_address);
      current_state->ptr_vc = NULL;
      if (retval == -1)
        return(PAPI_ESYS);
      }
  }
  return(PAPI_OK);
}

int _papi_hwd_query(int preset_index, int *flags, char **note)
{
  if (preset_map[preset_index].selector == 0)
    return(0);
  if (preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (preset_map[preset_index].note)
    *note = preset_map[preset_index].note;
  return(1);
}

int _papi_hwd_ctl(EventSetInfo *zero, int code, _papi_int_option_t *option)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_EMISC);
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_EMISC);
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_pc;

  return(location);
}

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{
  return(PAPI_OK);
}

void _papi_hwd_lock_init(void)
{
}

void _papi_hwd_lock(void)
{
}

void _papi_hwd_unlock(void)
{
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, ucontext_t *info)
{
  _papi_hwi_dispatch_overflow_signal((void *)info);
}

papi_mdi _papi_system_info = { "dadd-alpha.c 2002/05/28 shirley",
                              1.0, /*  version */
                               -1,  /*  cpunum */
                               {
                                 -1,  /*  ncpu */
                                  1,  /*  nnodes */
                                 -1,  /*  totalcpus */
                                 -1,  /*  vendor */
                                 "",  /*  vendor string */
                                 -1,  /*  model */
                                 "",  /*  model string */
                                0.0,  /*  revision */
                                0.0  /*  mhz */
                               },
                               {
                                 "",
                                 "",
                                 (caddr_t)&_ftext,
                                 (caddr_t)&_etext,
                                 (caddr_t)NULL,
                                 (caddr_t)NULL,
                                 (caddr_t)NULL,
                                 (caddr_t)NULL,
                                 "_RLD_LIST", /* How to preload libs */
                               },
                               { 0,  /*total_tlb_size*/
                                 0,  /*itlb_size */
                                 0,  /*itlb_assoc*/
                                 0,  /*dtlb_size */
                                 0, /*dtlb_assoc*/
                                 0, /*total_L1_size*/
                                 0, /*L1_icache_size*/
                                 0, /*L1_icache_assoc*/
                                 0, /*L1_icache_lines*/
                                 0, /*L1_icache_linesize*/
                                 0, /*L1_dcache_size */
                                 0, /*L1_dcache_assoc*/
                                 0, /*L1_dcache_lines*/
                                 0, /*L1_dcache_linesize*/
                                 0, /*L2_cache_size*/
                                 0, /*L2_cache_assoc*/
                                 0, /*L2_cache_lines*/
                                 0, /*L2_cache_linesize*/
                                 0, /*L3_cache_size*/
                                 0, /*L3_cache_assoc*/
                                 0, /*L3_cache_lines*/
                                 0  /*L3_cache_linesize*/
                               },
                               -1,  /*  num_cntrs */
                               -1,  /*  num_gp_cntrs */
                               -1,  /*  grouped_counters */
                               -1,  /*  num_sp_cntrs */
                               -1,  /*  total_presets */
                               -1,  /*  total_events */
                                PAPI_DOM_USER, /* default domain */
                                PAPI_GRN_PROC,  /* default granularity */
                                0,  /* We can use add_prog_event */
                                0,  /* We can write the counters */
                                0,  /* supports HW overflow */
                                0,  /* supports HW profile */
                                1,  /* supports 64 bit virtual counters */
                                0,  /* supports child inheritance */
                                0,  /* supports attaching to another process */
                                1,  /* We can use the real_usec call */
                                1,  /* We can use the real_cyc call */
                                1,  /* We can use the virt_usec call */
                                1,  /* We can use the virt_cyc call */
                                0,  /* HW read resets the counters */
                                sizeof(hwd_control_state_t),
                                { 0} };

