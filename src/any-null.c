/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "any-null.h"

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS];

static hwd_search_t findem_foobar[] = {
  { PAPI_TOT_CYC, NOT_DERIVED, -1, { 0x1, 0xa1 }},
  { PAPI_FP_INS , NOT_DERIVED, -1, {  -1, 0x2 }},
  { PAPI_TOT_INS, NOT_DERIVED, -1, { 0x3, 0xa3 }},
  { PAPI_L1_TCM,  DERIVED_ADD, -1, { 0x4, 0xa4 }},
  { PAPI_L1_DCM,  NOT_DERIVED, -1, { 0x5, -1  }},
  { PAPI_L1_ICM,  NOT_DERIVED, -1, {  -1, 0x6 }},
  { PAPI_FLOPS,   DERIVED_PS,   0, { 0x1, 0x2 }},
  { -1, NOT_DERIVED, -1, { -1, -1 }}};

/* Low level functions, should not handle errors, just return codes. */

inline static char *search_cpu_info(FILE *f, char *search_str, char *line)
{
  /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
  /* See the PCL home page for the German version of PAPI. */

  char *s;

  while (fgets(line, 256, f) != NULL)
    {
      if (strstr(line, search_str) != NULL)
	{
	  /* ignore all characters in line up to : */
	  for (s = line; *s && (*s != ':'); ++s)
	    ;
	  if (*s)
	    return(s);
	}
    }
  return(NULL);

  /* End stolen code */
}

/* Dumb hack to make sure I get the cycle time correct. */

static float estimate_mhz(void)
{
  return(1.0);
}

/* Utility functions */

static int setup_all_presets(void)
{
  hwd_search_t *findem = findem_foobar;
  char str[PAPI_MAX_STR_LEN];
  int num = _papi_system_info.num_gp_cntrs;
  int code;

  while ((code = findem->papi_code) != -1)
    {
      int i, index;

      index = code ^ PRESET_MASK; 
      preset_map[index].derived = NOT_DERIVED;
      preset_map[index].operand_index = 0;
      for (i=0;i<num;i++)
	{
	  if (findem->findme[i] != -1)
	    {
	      preset_map[index].selector |= 1 << i;
	      preset_map[index].counter_cmd.cmd[i] = findem->findme[i];
	      sprintf(str,"0x%x",findem->findme[i]);
	      if (strlen(preset_map[index].note))
		strcat(preset_map[index].note,",");
	      strcat(preset_map[index].note,str);
	    }
	}
      if (preset_map[index].selector != 0)
	{
	  DBG((stderr,"Preset %d found, selector 0x%x\n",
	       index,preset_map[index].selector));
	  preset_map[index].operand_index = findem->operand_index;
	  preset_map[index].derived = findem->derived_op;
	}
      findem++;
    }
  return(PAPI_OK);
}

/* Utility functions */

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

inline static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (MAX_COUNTERS-1);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
}

inline static void set_hwcntr_codes(int selector, unsigned int *from, int *to)
{
  int useme, i;
  
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  to[i] = to[i] & ~PERF_EVNT_MASK;
	  to[i] = to[i] | (int)from[i];
	}
    }
}

inline static void init_config(hwd_control_state_t *ptr)
{
  int def_mode;

  switch (_papi_system_info.default_domain)
    {
    case PAPI_DOM_USER:
      def_mode = PERF_USR;
      break;
    case PAPI_DOM_KERNEL:
      def_mode = PERF_OS;
      break;
    case PAPI_DOM_ALL:
      def_mode = PERF_OS | PERF_USR;
      break;
    default:
      abort();
    }

  ptr->selector = 0;
  ptr->counter_cmd.cmd[0] |= def_mode | PERF_ENABLE;
  ptr->counter_cmd.cmd[1] |= def_mode;
  ptr->counter_cmd.cmd[2] = 0;
}

static int get_system_info(void)
{
  int tmp;
  float mhz;

  strcpy(_papi_system_info.exe_info.fullname,"/home/you/foo");
  strcpy(_papi_system_info.exe_info.name,"foo");

  DBG((stderr,"Executable is %s\n",_papi_system_info.exe_info.name));
  DBG((stderr,"Full Executable is %s\n",_papi_system_info.exe_info.fullname));

  /* Hardware info */

  _papi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
  _papi_system_info.hw_info.vendor = -1;

  strcpy(_papi_system_info.hw_info.model_string,"Foo");
  strcpy(_papi_system_info.hw_info.vendor_string,"Bar");
  _papi_system_info.hw_info.model = 1;
  _papi_system_info.hw_info.vendor = 2;
  _papi_system_info.hw_info.revision = 0.1;

  /* Check mhz calc. */

  mhz = estimate_mhz();
  DBG((stderr,"Calculated MHZ is %f\n",mhz));
  DBG((stderr,"Detected MHZ is %f\n",_papi_system_info.hw_info.mhz));
  if (_papi_system_info.hw_info.mhz < mhz)
    _papi_system_info.hw_info.mhz = mhz;
  {
    int tmp = (int)_papi_system_info.hw_info.mhz;
    _papi_system_info.hw_info.mhz = (float)tmp;
  }
  DBG((stderr,"Actual MHZ is %f\n",_papi_system_info.hw_info.mhz));

  /* Setup counters */

  _papi_system_info.num_gp_cntrs = 2;
  _papi_system_info.num_cntrs = 2;

  /* Setup presets */

  tmp = setup_all_presets();
  if (tmp)
    return(tmp);

  return(PAPI_OK);
} 

#ifdef DEBUG
static void dump_cmd(any_command_t *t)
{
  int i;

  for (i=0;i<MAX_COUNTERS;i++)
    DBG((stderr,"Event %d: 0x%x\n",i,t->cmd[i]));
}
#endif

inline static int counter_event_shared(const int *a, const int *b, int cntr)
{
  if (a[cntr] == b[cntr])
    return(1);

  return(0);
}

inline static int counter_event_compat(const int *a, const int *b, int cntr)
{
  unsigned int priv_mask = ~PERF_EVNT_MASK;

  if ((a[cntr] & priv_mask) == (b[cntr] & priv_mask))
    return(1);

  return(0);
}

inline static void counter_event_copy(const int *a, int *b, int cntr)
{
  b[cntr] = a[cntr];
}

inline static int update_global_hwcounters(EventSetInfo *global)
{
  /* hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep; */
  unsigned long long events[MAX_COUNTERS];
  static unsigned long long sample = 1;
  int i;

  /* ret = perf(PERF_FASTREAD, (int)events, 0);
  if (ret)
    return(PAPI_ESYS); */
  
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    events[i] = sample;

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+events[i],
	   global->hw_start[i],events[i]));
      global->hw_start[i] = global->hw_start[i] + events[i];
    }

  /* ret = perf(PERF_RESET_COUNTERS, 0, 0);
  if (ret)
    return(PAPI_ESYS); */

  sample++;
  return(PAPI_OK);
}

inline static int correct_local_hwcounters(EventSetInfo *global, EventSetInfo *local, long long *correct)
{
  int i;

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      DBG((stderr,"correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n",i,
	   global->hw_start[i]-local->hw_start[i],global->hw_start[i],local->hw_start[i]));
      correct[i] = global->hw_start[i] - local->hw_start[i];
    }

  return(0);
}

inline static int set_domain(hwd_control_state_t *this_state, int domain)
{
  int mode0 = 0, mode1 = 0, did = 0;
  
  if (domain & PAPI_DOM_USER)
    {
      did = 1;
      mode0 |= PERF_USR | PERF_ENABLE;
      mode1 |= PERF_USR;
    }
  if (domain & PAPI_DOM_KERNEL)
    {
      did = 1;
      mode0 |= PERF_OS | PERF_ENABLE;
      mode1 |= PERF_OS;
    }

  if (!did)
    return(PAPI_EINVAL);

  this_state->counter_cmd.cmd[0] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd.cmd[0] |= mode0;
  this_state->counter_cmd.cmd[1] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd.cmd[1] |= mode1;

  return(PAPI_OK);
}

inline static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  switch (domain)
    {
    case PAPI_GRN_THR:
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit(int arg)
{
  /* int r;

  if (arg)
    arg = 1;

  r = perf(PERF_SET_OPT, PERF_DO_CHILDREN, arg);
  if (r != 0)
    return(PAPI_ESYS); */

  return(PAPI_ESBSTR);
}

inline static int set_default_domain(EventSetInfo *zero, int domain)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_domain(current_state,domain));
}

inline static int set_default_granularity(EventSetInfo *zero, int granularity)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_granularity(current_state,granularity));
}

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

int _papi_hwd_init_global(void)
{
  int retval;

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
  /* Initialize our global machdep. */

  init_config(zero->machdep);

  return(PAPI_OK);
}

long long _papi_hwd_get_real_usec (void)
{
  struct timeval tv;

  gettimeofday(&tv,NULL);
  return((tv.tv_sec * 1000000) + tv.tv_usec);
}

long long _papi_hwd_get_real_cycles (void)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_real_usec();
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
}

long long _papi_hwd_get_virt_cycles (EventSetInfo *zero)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_real_usec();
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
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

  usec = (float)_papi_hwd_get_real_usec();
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error: %s",strerror(error));
}

/* Do not ever use ESI->NumberOfCounters in here. */

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector = 0;
  int avail = 0;
  any_command_t tmp_cmd;
  any_command_t *codes;

  if (EventCode & PRESET_MASK)
    { 
      int preset_index;
      int derived;

      preset_index = EventCode ^ PRESET_MASK; 

      selector = preset_map[preset_index].selector;
      if (selector == 0)
	return(PAPI_ENOEVNT);
      derived = preset_map[preset_index].derived;

      /* Find out which counters are available. */

      avail = selector & ~this_state->selector;

      /* If not derived */

      if (preset_map[preset_index].derived == 0) 
	{
	  /* Pick any counter available */

	  selector = get_avail_hwcntr_bits(avail);
	  if (selector == 0)
	    return(PAPI_ECNFLCT);
	}    
      else
	{
	  /* Check the case that if not all the counters 
	     required for the derived event are available */

	  if ((avail & selector) != selector)
	    return(PAPI_ECNFLCT);	    
	}

      /* Get the codes used for this event */

      codes = &preset_map[preset_index].counter_cmd;
      ESI->EventInfoArray[index].command = derived;
      ESI->EventInfoArray[index].operand_index = preset_map[preset_index].operand_index;
    }
  else
    {
      int hwcntr_num;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  
      if ((hwcntr_num > _papi_system_info.num_gp_cntrs) ||
	  (hwcntr_num < 0))
	return(PAPI_EINVAL);

      tmp_cmd.cmd[hwcntr_num] = EventCode >> 8; 
      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->selector & selector)
	return(PAPI_ECNFLCT);	    

      codes = &tmp_cmd;
    }

  /* Lower two bits tell us what counters we need
     since we only have 2 counters 

  assert((this_state->selector | 0x3) == 0x3); */
  
  /* Perform any initialization of the control bits */

  if (this_state->selector == 0)
    init_config(this_state);
  
  /* Turn on the bits for this counter */

  set_hwcntr_codes(selector,codes->cmd,this_state->counter_cmd.cmd);

  /* Update the new counter select field */

  this_state->selector |= selector;

  /* Inform the upper level that the software event 'index' 
     consists of the following information. */

  ESI->EventInfoArray[index].code = EventCode;
  ESI->EventInfoArray[index].selector = selector;

  return(PAPI_OK);
}

int _papi_hwd_rem_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector, used, preset_index;

  /* Find out which counters used. */
  
  used = ESI->EventInfoArray[index].selector;

  if (EventCode & PRESET_MASK)
    { 
      preset_index = EventCode ^ PRESET_MASK; 

      selector = preset_map[preset_index].selector;
      if (selector == 0)
	return(PAPI_ENOEVNT);
    }
  else
    {
      int hwcntr_num, code, old_code;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0x3; 
      if ((hwcntr_num > _papi_system_info.num_gp_cntrs) || 
	  (hwcntr_num < 0))
	return(PAPI_EINVAL);

      old_code = ESI->EventInfoArray[index].command;
      code = EventCode >> 8; 
      if (old_code != code)
	return(PAPI_EINVAL);

      selector = 1 << hwcntr_num;
    }

  /* Check if these counters aren't used. */

  if ((used & selector) != used)
    return(PAPI_EINVAL);

  /* Clear out counters that are part of this event. */

  this_state->selector = this_state->selector ^ selector;

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, int index, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  
  /* If we ARE NOT nested, 
     just copy the global counter structure to the current eventset */

  if (current_state->selector == 0x0)
    {
      current_state->selector = this_state->selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,MAX_COUNTERS*sizeof(int));

      /* Stop the current context */

      /* retval = perf(PERF_RESET_COUNTERS, 0, 0);
      if (retval) 
	return(PAPI_ESYS); */
      
      /* (Re)start the counters */
      
#ifdef DEBUG
      dump_cmd(&current_state->counter_cmd);
#endif
      /* retval = perf(PERF_FASTCONFIG, (int)current_state->counter_cmd, (int)NULL);
      if (retval) 
	return(PAPI_ESYS); */
      
      return(PAPI_OK);
    }

  /* If we ARE nested, 
     carefully merge the global counter structure with the current eventset */
  else
    {
      int tmp, hwcntrs_in_both, hwcntrs_in_all, hwcntr;

      /* Stop the current context */

      /* retval = perf(PERF_STOP, 0, 0);
      if (retval) 
	return(PAPI_ESYS); */
  
      /* Update the global values */

      retval = update_global_hwcounters(zero);
      if (retval)
	return(retval);

      /* Delete the current context */

      hwcntrs_in_both = this_state->selector & current_state->selector;
      hwcntrs_in_all  = this_state->selector | current_state->selector;

      /* Check for events that are shared between eventsets and 
	 therefore require no modification to the control state. */

      /* First time through, error check */

      tmp = hwcntrs_in_all;
      while ((i = ffs(tmp)))
	{
	  hwcntr = 1 << (i-1);
	  tmp = tmp ^ hwcntr;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      if (!(counter_event_shared(this_state->counter_cmd.cmd, current_state->counter_cmd.cmd, i-1)))
		return(PAPI_ECNFLCT);
	    }
	  else if (!(counter_event_compat(this_state->counter_cmd.cmd, current_state->counter_cmd.cmd, i-1)))
	    return(PAPI_ECNFLCT);
	}

      /* Now everything is good, so actually do the merge */

      tmp = hwcntrs_in_all;
      while ((i = ffs(tmp)))
	{
	  hwcntr = 1 << (i-1);
	  tmp = tmp ^ hwcntr;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      ESI->hw_start[i-1] = zero->hw_start[i-1];
	      zero->multistart.SharedDepth[i-1]++; 
	    }
	  else if (hwcntr & this_state->selector)
	    {
	      current_state->selector |= hwcntr;
	      counter_event_copy(this_state->counter_cmd.cmd, current_state->counter_cmd.cmd, i-1);
	      ESI->hw_start[i-1] = 0;
	      zero->hw_start[i-1] = 0;
	    }
	}
    }

  /* Set up the new merged control structure */
  
#ifdef DEBUG
  dump_cmd(&current_state->counter_cmd);
#endif
      
  /* Stop the current context */

  /* retval = perf(PERF_RESET_COUNTERS, 0, 0);
  if (retval) 
    return(PAPI_ESYS); */

  /* (Re)start the counters */
  
  /* retval = perf(PERF_FASTCONFIG, (int)current_state->counter_cmd, (int)NULL);
  if (retval) 
    return(PAPI_ESYS); */

  return(PAPI_OK);
} 

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, hwcntr, tmp;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  /* Check for events that are NOT shared between eventsets and 
     therefore require modification to the selection mask. */

  if ((zero->multistart.num_runners - 1) == 0)
    {
      current_state->selector = 0;
      return(PAPI_OK);
    }
  else
    {
      tmp = this_state->selector;
      while ((i = ffs(tmp)))
	{
	  hwcntr = 1 << (i-1);
	  if (zero->multistart.SharedDepth[i-1] - 1 < 0)
	    current_state->selector ^= hwcntr;
	  else
	    zero->multistart.SharedDepth[i-1]--;
	  tmp ^= hwcntr;
	}
      return(PAPI_OK);
    }
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

static long long handle_derived_add(int selector, long long *from)
{
  int pos;
  long long retval = 0;

  while ((pos = ffs(selector)))
    {
      DBG((stderr,"Compound event, adding %lld to %lld\n",from[pos-1],retval));
      retval += from[pos-1];
      selector ^= 1 << (pos-1);
    }
  return(retval);
}

static long long handle_derived_subtract(int operand_index, int selector, long long *from)
{
  int pos;
  long long retval = from[operand_index];

  selector = selector ^ (1 << operand_index);
  while ((pos = ffs(selector)))
    {
      DBG((stderr,"Compound event, subtracting %lld to %lld\n",from[pos-1],retval));
      retval -= from[pos-1];
      selector ^= 1 << (pos-1);
    }
  return(retval);
}

static long long units_per_second(long long units, long long cycles)
{
  float tmp;

  tmp = (float)units * _papi_system_info.hw_info.mhz * 1000000.0;
  tmp = tmp / (float) cycles;
  return((long long)tmp);
}

static long long handle_derived_ps(int operand_index, int selector, long long *from)
{
  int pos;

  pos = ffs(selector ^ (1 << operand_index)) - 1;
  /* assert(pos >= 0); */

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

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long events[])
{
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  long long correct[MAX_COUNTERS];

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
      if (selector == PAPI_NULL)
	continue;

      DBG((stderr,"Event index %d, selector is 0x%x\n",j,selector));

      /* If this is not a derived event */

      if (ESI->EventInfoArray[i].command == NOT_DERIVED)
	{
	  shift_cnt = ffs(selector) - 1;
	  /* assert(shift_cnt >= 0); */
	  events[j] = correct[shift_cnt];
	}

      /* If this is a derived event */

      else 
	events[j] = handle_derived(&ESI->EventInfoArray[i], correct);

      /* Early exit! */

      if (++j == ESI->NumberOfCounters)
	return(PAPI_OK);
    }

  /* Should never get here */

  return(PAPI_EBUG);
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
    case PAPI_SET_INHERIT:
      return(set_inherit(option->inherit.inherit));
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
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
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

void _papi_hwd_dispatch_timer(int signal, struct sigcontext info)
{
  DBG((stderr,"_papi_hwd_dispatch_timer() at 0x%lx\n",info.eip));
  _papi_hwi_dispatch_overflow_signal((void *)&info); 
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->eip;

  return(location);
}

#define __SMP__
#include <asm/atomic.h>
static atomic_t lock;

void _papi_hwd_lock_init(void)
{
  atomic_set(&lock,0);
}

void _papi_hwd_lock(void)
{
  atomic_inc(&lock);
  while (atomic_read(&lock) > 1)
    {
      DBG((stderr,"Waiting..."));
      usleep(1000);
    }
}

void _papi_hwd_unlock(void)
{
  atomic_dec(&lock);
}

/* Machine info structure. -1 is unused. */

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
				0.0  /*  mhz */ 
			       },
			       {
				 "",
				 "",
				 (caddr_t)NULL,
				 (caddr_t)NULL,
				 (caddr_t)NULL,
				 (caddr_t)NULL,
				 (caddr_t)NULL,
				 (caddr_t)NULL,
				 "LD_PRELOAD", /* How to preload libs */
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
			        1,  /* supports child inheritance */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        0,  /* We can use the virt_usec call */
			        0,  /* We can use the virt_cyc call */
			        0,  /* HW read resets the counters */
			        sizeof(hwd_control_state_t), 
			        { 0, } };

