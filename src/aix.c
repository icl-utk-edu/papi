/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* This file handles the OS dependent part of the POWER3 and POWER4 architectures.
  It supports both AIX 4 and AIX 5. The switch between AIX 4 and 5 is driven by the 
  system defined value _AIX_VERSION_510.
  Other routines also include minor conditionally compiled differences.
*/

#include "papi.h"
#include SUBSTRATE
/*
#include "papi_internal.h"
#include "papi_protos.h"
*/

/* 
 some heap information, start_of_text, start_of_data .....
 ref: http://publibn.boulder.ibm.com/doc_link/en_US/a_doc_lib/aixprggd/genprogc/sys_mem_alloc.htm#HDRA9E4A4C9921SYLV 
*/
#ifndef _P64
  #define START_OF_TEXT 0x10000000
  #define END_OF_TEXT   &_etext
  #define START_OF_DATA 0x20000000
  #define END_OF_DATA   &_end
#else
  #define START_OF_TEXT 0x100000000
  #define END_OF_TEXT   &_etext
  #define START_OF_DATA 0x110000000
  #define END_OF_DATA   &_end
#endif

static int maxgroups = 0;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/* The following is for any POWER hardware */

/**********************************************************************************/
/* The next four functions implement the native event interface */
/* On POWER, native events take the form:
    0x4000EECC, where 4 is NATIVE_MASK, EE is the event code, and CC is the counter.
    The same event can often be found on multiple counters, so there can be a
    one-to-many relationship between event names and native codes.
    In these cases the first instance found will be the one reported.
*/

#ifdef HAS_NATIVE_MAP

/* Reverse lookup of event code to index */
unsigned int _papi_hwd_native_code_to_idx(unsigned int event_code)
{
  unsigned int idx, counter, pmc;

  idx = (event_code >> 8) & 0xff;
  counter = event_code & 0xff;
  if ((counter < pminfo.maxpmcs) && (idx < pminfo.maxevents[counter])) {
    for (pmc = 1; pmc <= counter; pmc++) {
      idx += pminfo.maxevents[pmc-1];
    }
    return (idx);
  }
  return (PAPI_ENOEVNT);
}

/* Returns event code based on index. NATIVE_MASK bit must be set if not predefined */
unsigned int _papi_hwd_native_idx_to_code(unsigned int idx)
{
  unsigned int pmc;

  for (pmc = 0; pmc < pminfo.maxpmcs; pmc++) {
    if (idx < pminfo.maxevents[pmc]) break;
    idx -= pminfo.maxevents[pmc];
  }
  if (pmc < pminfo.maxpmcs) {
    return (NATIVE_MASK | (idx << 8) | pmc);
  }
  return(PAPI_ENOEVNT);
}

/* Returns event name based on index. */
char *_papi_hwd_native_idx_to_name(unsigned int idx)
{
  unsigned int pmc;

  for (pmc = 0; pmc < pminfo.maxpmcs; pmc++) {
    if (idx < pminfo.maxevents[pmc]) break;
    idx -= pminfo.maxevents[pmc];
  }
  if (pmc < pminfo.maxpmcs) {
    return (pminfo.list_events[pmc][idx].short_name);
  }
  return(NULL);
}

/* Returns event description based on index. */
char *_papi_hwd_native_idx_to_descr(unsigned int idx)
{
  unsigned int pmc;

  for (pmc = 0; pmc < pminfo.maxpmcs; pmc++) {
    if (idx < pminfo.maxevents[pmc]) break;
    idx -= pminfo.maxevents[pmc];
  }
  if (pmc < pminfo.maxpmcs) {
    return (pminfo.list_events[pmc][idx].description);
  }
  return(NULL);
}

#endif /* HAS_NATIVE_MAP */
/**********************************************************************************/


static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  ptr->counter_cmd.events[arg1] = arg2;
}

static void unset_config(hwd_control_state_t *ptr, int arg1)
{
  ptr->counter_cmd.events[arg1] = 0;
}

int update_global_hwcounters(EventSetInfo *global)
{
  int i, retval;
  pm_data_t data;

  retval = pm_get_data_mythread(&data);
  if (retval > 0)
    return(retval);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
#if 0
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+data.accu[i],global->hw_start[i],data.accu[i]));
#endif
      global->hw_start[i] = global->hw_start[i] + data.accu[i];
    }

  retval = pm_reset_data_mythread();
  if (retval > 0)
    return(retval);
   
  return(0);
}

static int correct_local_hwcounters(EventSetInfo *global, EventSetInfo *local, long long *correct)
{
  int i;

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
#if 0
      DBG((stderr,"correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n",i,
	   global->hw_start[i]-local->hw_start[i],global->hw_start[i],local->hw_start[i]));
#endif
      correct[i] = global->hw_start[i] - local->hw_start[i];
    }

  return(0);
}

int set_domain(hwd_control_state_t *this_state, int domain)
{
  pm_mode_t *mode = &(this_state->counter_cmd.mode);

  switch (domain)
    {
    case PAPI_DOM_USER:
      mode->b.user = 1;
      mode->b.kernel = 0;
      break;
    case PAPI_DOM_KERNEL:
      mode->b.user = 0;
      mode->b.kernel = 1;
      break;
    case PAPI_DOM_ALL:
      mode->b.user = 1;
      mode->b.kernel = 1;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

int set_granularity(hwd_control_state_t *this_state, int domain)
{
  pm_mode_t *mode = &(this_state->counter_cmd.mode);

  switch (domain)
    {
    case PAPI_GRN_THR:
      mode->b.process = 0;
      mode->b.proctree = 0;
      break;
    /* case PAPI_GRN_PROC:
      mode->b.process = 1;
      mode->b.proctree = 0;
      break;
    case PAPI_GRN_PROCG:
      mode->b.process = 0;
      mode->b.proctree = 1;
      break; */
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int set_default_domain(EventSetInfo *zero, int domain)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_domain(current_state,domain));
}

static int set_default_granularity(EventSetInfo *zero, int granularity)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_granularity(current_state,granularity));
}

static int set_inherit(int arg)
{
  return(PAPI_ESBSTR);
}


static int get_system_info(void)
{
  int retval;
 /* pm_info_t pminfo;*/
  struct procsinfo psi = { 0 };
  pid_t pid;
  char maxargs[PAPI_MAX_STR_LEN];

#ifdef _AIXVERSION_510
  pm_groups_info_t pmgroups;
#endif

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);
  psi.pi_pid = pid;
  retval = getargs(&psi,sizeof(psi),maxargs,PAPI_MAX_STR_LEN);
  if (retval == -1)
    return(PAPI_ESYS);
  if (getcwd(_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == NULL)
    return(PAPI_ESYS);
  strcat(_papi_system_info.exe_info.fullname,"/");
  strcat(_papi_system_info.exe_info.fullname,maxargs);
  strncpy(_papi_system_info.exe_info.name,basename(maxargs),PAPI_MAX_STR_LEN);

#ifdef _AIXVERSION_510
  DBG((stderr,"Calling AIX 5 version of pm_init...\n"));
  retval = pm_init(PM_INIT_FLAGS, &pminfo, &pmgroups);
#else
  DBG((stderr,"Calling AIX 4 version of pm_init...\n"));
  retval = pm_init(PM_INIT_FLAGS,&pminfo);
#endif
  DBG((stderr,"...Back from pm_init\n"));

  if (retval > 0)
    return(retval);

  _papi_system_info.hw_info.ncpu = _system_configuration.ncpus;
  _papi_system_info.hw_info.totalcpus = 
    _papi_system_info.hw_info.ncpu * _papi_system_info.hw_info.nnodes;
  _papi_system_info.hw_info.vendor = -1;
  strcpy(_papi_system_info.hw_info.vendor_string,"IBM");
  _papi_system_info.hw_info.model = _system_configuration.implementation;
  strcpy(_papi_system_info.hw_info.model_string,pminfo.proc_name);
  _papi_system_info.hw_info.revision = (float)_system_configuration.version;
  _papi_system_info.hw_info.mhz = (float)(pm_cycles() / 1000000.0);
  _papi_system_info.num_gp_cntrs = pminfo.maxpmcs;
  _papi_system_info.num_cntrs = pminfo.maxpmcs;
  _papi_system_info.cpunum = mycpu();
/*  _papi_system_info.exe_info.text_end = (caddr_t)&_etext;*/

#ifdef _POWER4
  retval = setup_p4_presets(&pminfo, &pmgroups);
#else
  retval = setup_all_presets(&pminfo);
#endif

  if (retval)
    return(retval);

  return(PAPI_OK);
} 

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

long_long _papi_hwd_get_real_usec (void)
{
  timebasestruct_t t;
  long_long retval;

  read_real_time(&t,TIMEBASE_SZ);
  time_base_to_time(&t,TIMEBASE_SZ);
  retval = (t.tb_high * 1000000) + t.tb_low / 1000;
  return(retval);
}

long_long _papi_hwd_get_real_cycles (void)
{
  long_long usec, cyc;

  usec = _papi_hwd_get_real_usec();
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long_long)cyc);
}

long long _papi_hwd_get_virt_usec (EventSetInfo *zero)
{
  long long retval;
  struct tms buffer;

  times(&buffer);
  retval = (long long)buffer.tms_utime*(long long)(1000000/CLK_TCK);
  return(retval);
}

long long _papi_hwd_get_virt_cycles (EventSetInfo *zero)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_virt_usec(zero);
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error");
  pm_error(where,error);
}

int _papi_hwd_init_global(void)
{
  int retval;

  /* Fill in what we can of the papi_system_info. */
  
  retval = get_system_info();
  if (retval)
    return(retval);
  
  retval = get_memory_info(&_papi_system_info.mem_info);
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
  /* Initialize our global machdep. */

  init_config(zero->machdep);

  return(PAPI_OK);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (POWER_MAX_COUNTERS-1);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, int *to)
{
  int useme, i;
  
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  to[i] = from[i];
	}
    }
}

#if 1
static void dump_state(hwd_control_state_t *s)
{
  fprintf(stderr,"master_selector %x\n",s->master_selector);
  fprintf(stderr,"event_codes %x %x %x %x %x %x %x %x\n",s->preset[0],s->preset[1],
    s->preset[2],s->preset[3],s->preset[4],s->preset[5],s->preset[6],s->preset[7]);
  fprintf(stderr,"event_selectors %x %x %x %x %x %x %x %x\n",s->selector[0],s->selector[1],
    s->selector[2],s->selector[3],s->selector[4],s->selector[5],s->selector[6],s->selector[7]);
  fprintf(stderr,"counters %x %x %x %x %x %x %x %x\n",s->counter_cmd.events[0],
    s->counter_cmd.events[1],s->counter_cmd.events[2],s->counter_cmd.events[3],
    s->counter_cmd.events[4],s->counter_cmd.events[5],s->counter_cmd.events[6],
    s->counter_cmd.events[7]);
}
#endif
  

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)

{
  return(PAPI_ESBSTR);
}

void dump_cmd(pm_prog_t *t)
{
  fprintf(stderr,"mode.b.threshold %d\n",t->mode.b.threshold);
  fprintf(stderr,"mode.b.spare %d\n",t->mode.b.spare);
  fprintf(stderr,"mode.b.process %d\n",t->mode.b.process);
  fprintf(stderr,"mode.b.kernel %d\n",t->mode.b.kernel);
  fprintf(stderr,"mode.b.user %d\n",t->mode.b.user);
  fprintf(stderr,"mode.b.count %d\n",t->mode.b.count);
  fprintf(stderr,"mode.b.proctree %d\n",t->mode.b.proctree);
  fprintf(stderr,"events[0] %d\n",t->events[0]);
  fprintf(stderr,"events[1] %d\n",t->events[1]);
  fprintf(stderr,"events[2] %d\n",t->events[2]);
  fprintf(stderr,"events[3] %d\n",t->events[3]);
  fprintf(stderr,"events[4] %d\n",t->events[4]);
  fprintf(stderr,"events[5] %d\n",t->events[5]);
  fprintf(stderr,"events[6] %d\n",t->events[6]);
  fprintf(stderr,"events[7] %d\n",t->events[7]);
  fprintf(stderr,"reserved %d\n",t->reserved);
}


int _papi_hwd_reset(EventSetInfo *ESI, EventSetInfo *zero)
{
  int i, retval;

  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    ESI->hw_start[i] = zero->hw_start[i];

  return(PAPI_OK);
}

/****************************************************************************/
static long long handle_derived_add(int selector, long long *from)
{
  int pos;
  long long retval = 0;

  while ((pos = ffs(selector)))
    {
      DBG((stderr,"Compound event, adding %lld to %lld\n",from[pos-1],retval));
      retval += from[pos-1];
      selector ^= 1 << pos-1;
    }
  return(retval);
}

static long long handle_derived_subtract(int operand_index, int selector, long long *from)
{
  int pos;
  long long retval = from[operand_index];

  selector = selector ^ (1 << operand_index);
  while (pos = ffs(selector))
    {
      DBG((stderr,"Compound event, subtracting %lld to %lld\n",from[pos-1],retval));
      retval -= from[pos-1];
      selector ^= 1 << pos-1;
    }
  return(retval);
}

static long long units_per_second(long long units, long long cycles)
{
  return((long long)((float)units * _papi_system_info.hw_info.mhz * 1000000.0 / (float)cycles));
}

static long long handle_derived_ps(int operand_index, int selector, long long *from)
{
  int pos;

  pos = ffs(selector ^ (1 << operand_index)) - 1;
  assert(pos >= 0);

  return(units_per_second(from[pos],from[operand_index]));
}

static long long handle_derived_add_ps(int operand_index, int selector, long long *from)
{
  int add_selector = selector ^ (1 << operand_index);
  long long tmp = handle_derived_add(add_selector, from);
  return(units_per_second(tmp, from[operand_index]));
}

static long long handle_derived(EventInfo_t *cmd, long long *from)
{
  switch (cmd->command)
    {
    case DERIVED_ADD: 
      return(handle_derived_add(cmd->selector, from));
    case DERIVED_ADD_PS:
      return(handle_derived_add_ps(cmd->operand_index, cmd->selector, from));
    case DERIVED_SUB:
      return(handle_derived_subtract(cmd->operand_index, cmd->selector, from));
    case DERIVED_PS:
      return(handle_derived_ps(cmd->operand_index, cmd->selector, from));
    default:
      abort();
    }
}
/****************************************************************************/

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long *events)
{
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  long long correct[POWER_MAX_COUNTERS];
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  retval = correct_local_hwcounters(zero, ESI, correct);
  if (retval)
    return(retval);

  /* This routine distributes hardware counters to software counters in the
     order that they were added. Note that the higher level 
     EventInfoArray[i] entries may not be contiguous because the user
     has the right to remove an event. */

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      selector = ESI->EventInfoArray[i].selector;
      if (selector == 0)
	continue;

     DBG((stderr,"Event index %d, selector is 0x%x\n",j,selector));
#ifdef _POWER4
     DBG((stderr,"Group is %d\n",this_state->counter_cmd.events[0]));
#endif
     assert(selector != 0);

      /* If this is not a derived event */

      DBG((stderr,"Derived: %d\n", ESI->EventInfoArray[i].command));
      if (ESI->EventInfoArray[i].command == NOT_DERIVED)
	{
	  shift_cnt = ffs(selector) - 1;
	  assert(shift_cnt >= 0);
	  events[j] = correct[shift_cnt];
	}
      
      /* If this is a derived event */

      else 
	events[j] = handle_derived(&ESI->EventInfoArray[i], correct);
	
      /* Early exit! */

      if (++j == ESI->NumberOfEvents)
	return(PAPI_OK);
    }

  /* Should never get here */

  return(PAPI_EBUG);
}

int _papi_hwd_setmaxmem(){
  return(PAPI_OK);
}

int _papi_hwd_ctl(EventSetInfo *zero, int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_DEFDOM:
      return(set_default_domain(zero, option->domain.domain));
    case PAPI_SET_DOMAIN:
      return(set_domain(option->domain.ESI->machdep, option->domain.domain));
    case PAPI_SET_DEFGRN:
      return(set_default_granularity(zero, option->granularity.granularity));
    case PAPI_SET_GRANUL:
      return(set_granularity(option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
    case PAPI_SET_INHERIT:
      return(set_inherit(option->inherit.inherit));
#endif
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_write(EventSetInfo *master, EventSetInfo *ESI, long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(EventSetInfo *zero)
{
  pm_delete_program_mythread();
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
  return(PAPI_OK);
}


void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, void *i)
{
#ifdef DEBUG
  ucontext_t *info;
  info = (ucontext_t *)i;
  DBG((stderr,"_papi_hwd_dispatch_timer() at 0x%lx\n",info->uc_mcontext.jmp_context.iar));
#endif

  _papi_hwi_dispatch_overflow_signal(i); 
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  if (overflow_option->threshold == 0)
    {
      this_state->timer_ms = 0;
      overflow_option->timer_ms = 0;
    }
  else
    {
      this_state->timer_ms = 1; /* Millisecond intervals are the only way to go */
      overflow_option->timer_ms = 1;
    }

  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(EventSetInfo *ESI, EventSetInfo *master)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_jmpbuf.jmp_context.iar;

  return(location);
}

static volatile int lock_var = 0;
static atomic_p lock;

void _papi_hwd_lock_init(void)
{
  lock = (int *)&lock_var;
}

void _papi_hwd_lock(void)
{
  while (_check_lock(lock,0,1) == TRUE)
    {
      DBG((stderr,"Waiting..."));
      usleep(1000);
    }
}

void _papi_hwd_unlock(void)
{
  _clear_lock(lock, 0);
}



/* Machine info structure. -1 is initialized by _papi_hwd_init. */

papi_mdi _papi_system_info = { "$Id$",
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
				 -1  /*  mhz */ 
			       },
			       {
				 "",
				 "",
				 (caddr_t)START_OF_TEXT,
				 (caddr_t)END_OF_TEXT,
				 (caddr_t)START_OF_DATA,
				 (caddr_t)END_OF_DATA,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 ""
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
			        PAPI_GRN_THR,  /* default granularity */
			        0,  /* We can use add_prog_event */
			        0,  /* We can write the counters */
			        0,  /* supports HW overflow */
			        0,  /* supports HW profile */
			        1,  /* supports 64 bit virtual counters */
			        0,  /* supports child inheritance */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        0,  /* We can use the virt_usec call */
			        0,  /* We can use the virt_cyc call */
			        0,  /* HW Read also resets the counters */
			        sizeof(hwd_control_state_t), 
			        { 0, } };

