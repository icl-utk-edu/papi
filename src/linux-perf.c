/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#include "linux-x86.h"

_syscall3(int, perf, int, op, int, counter, int, event);

/* First entry is mask, counter code 1, counter code 2, and TSC. 
A high bit in the mask entry means it is an OR mask, not an
and mask. This means that the same even is available on either
counter. */

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                {CNTR2|CNTR1,0,0,{0x45,0x45,0},""},	// L1 Cache Dmisses 
                {CNTR2|CNTR1,0,0,{0x81,0x81,0},""},	// L1 Cache Imisses 
		{0,0,0,{0,0,0},""}, 			// L2 Cache Dmisses
		{0,0,0,{0,0,0},""}, 			// L2 Cache Imisses
		{0,0,0,{0,0,0},""}, 			// L3 Cache Dmisses
		{0,0,0,{0,0,0},""}, 			// L3 Cache Imisses
                {CNTR2|CNTR1,DERIVED_ADD,0,{0x45,0x81,0},""},	// L1 Total Cache misses 
		{CNTR2|CNTR1,0,0,{0x24,0x24,0},""}, 	// L2 Total Cache misses
		{0,0,0,{0,0,0},""}, 			// L3 Total Cache misses
		{0,0,0,{0,0,0},""},			// Snoops
		{0,0,0,{0,0,0},""},		 	// Req. access to shared cache line
		{0,0,0,{0,0,0},""},		 	// Req. access to clean cache line
		{0,0,0,{0,0,0},""},		 	// Cache Line Invalidation
                {0,0,0,{0,0,0},""},			// Cache Line Intervention
                {0,0,0,{0,0,0},""},			// 14
                {0,0,0,{0,0,0},""},			// 15
                {0,0,0,{0,0,0},""},			// cycles branch idle
                {0,0,0,{0,0,0},""},			// cycles int idle
                {0,0,0,{0,0,0},""},			// cycles fpu idle
                {0,0,0,{0,0,0},""},			// cycles load/store idle
		{0,0,0,{0,0,0},""},		 	// D-TLB misses
		{CNTR2|CNTR1,0,0,{0x85,0x85,0},""},	// I-TLB misses
                {0,0,0,{0,0,0},""},			// Total TLB misses
                {0,0,0,{0,0,0},""},			// L1 load M
                {0,0,0,{0,0,0},""},			// L1 store M
                {0,0,0,{0,0,0},""},			// L2 load M
                {0,0,0,{0,0,0},""},			// L2 store M
                {0,0,0,{0,0,0},""},			// BTAC misses
                {0,0,0,{0,0,0},""},			// 28
                {0,0,0,{0,0,0},""},			// 29
		{0,0,0,{0,0,0},""},			// TLB shootdowns
                {0,0,0,{0,0,0},""},			// Failed Store cond.
                {0,0,0,{0,0,0},""},			// Suc. store cond.
                {0,0,0,{0,0,0},""},			// total. store cond.
                {CNTR2|CNTR1,0,0,{0xA2,0xA2,0},""},	/* Cycles stalled waiting for memory */
                {0,0,0,{0,0,0},""},		   	/* Cycles stalled waiting for memory read */
                {0,0,0,{0,0,0},""},		   	/* Cycles stalled waiting for memory write */
                {0,0,0,{0,0,0},""},			/* Cycles no instructions issued */
                {0,0,0,{0,0,0},""},			/* Cycles max instructions issued */
                {0,0,0,{0,0,0},""},			/* Cycles max instructions comleted */
                {0,0,0,{0,0,0},""},			/* Cycles max instructions completed */
                {0,0,0,{0,0,0},""},			// hardware interrupts
		{0,0,0,{0,0,0},""},	// Uncond. branches executed
		{0,0,0,{0,0,0},""},	// Cond. Branch inst. executed
		{CNTR2|CNTR1,0,0,{0xC9,0xC9,0},""},	// Cond. Branch inst. taken
		{0,0,0,{0,0,0},""},	// Cond. Branch inst. not taken
                {CNTR2|CNTR1,0,0,{0xC5,0xC5,0},""},	// Cond. branch inst. mispred.
                {0,0,0,{0,0,0},""},     // Cond. branch inst. corr. pred.
                {0,0,0,{0,0,0},""},			// FMA
                {CNTR2|CNTR1,0,0,{0xD0,0xD0,0},""},	// Total inst. issued
		{CNTR2|CNTR1,0,0,{0xC0,0xC0,0},""},	// Total inst. executed
		{0,0,0,{0,0,0},""},			// Integer inst. executed
		{CNTR1,0,0,{0xC1,0,0},""},		// Floating Pt. inst. executed
		{0,0,0,{0,0,0},""},			// Loads executed
		{0,0,0,{0,0,0},""},			// Stores executed
		{CNTR2|CNTR1,0,0,{0xC4,0xC4,0},""},	// Branch inst. executed
		{CNTR2|CNTR1,0,0,{0xB0,0xB0,0},""},	// Vector/SIMD inst. executed 
		{CNTR2|CNTR1,DERIVED_PS,1,{0xC1,0x79,0},""},	// FLOPS
                {0,0,0,{0,0,0},""},			// 58
                {0,0,0,{0,0,0},""},			// FPU stalled
		{CNTR2|CNTR1,0,0,{0x79,0x79,0},""},	// Total cycles
		{CNTR2|CNTR1,DERIVED_PS,1,{0xC0,0x79,0},""},	// IPS
                {CNTR2|CNTR1,0,0,{0x43,0x43,0},""},	// Total load/store inst. exec
                {0,0,0,{0,0,0},""},			// SYnc exec.
             };

/* Low level functions, should not handle errors, just return codes. */

/* Utility functions */

static int setup_all_presets(PAPI_hw_info_t *info)
{
  int pnum;

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      if (preset_map[pnum].selector)
	{
	  if (preset_map[pnum].derived == 0)
	    sprintf(preset_map[pnum].note,"0x%x",preset_map[pnum].counter_cmd[0]);
	  else
	    {
	      int j = preset_map[pnum].operand_index;
	      
	      if (j == 0)
		sprintf(preset_map[pnum].note,"0x%x,0x%x",preset_map[pnum].counter_cmd[0],
		       preset_map[pnum].counter_cmd[1]);
	      else 
		sprintf(preset_map[pnum].note,"0x%x,0x%x",preset_map[pnum].counter_cmd[1],
		       preset_map[pnum].counter_cmd[0]);
	    }
	}
    }
  return(PAPI_OK);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

inline static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (PERF_COUNTERS-1);
  
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
	  to[i] = to[i] & ~PERF_EVNT_MASK;
	  to[i] = to[i] | (int)from[i];
	}
    }
}

inline static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  ptr->counter_cmd[arg1] |= arg2;
}

inline static void unset_config(hwd_control_state_t *ptr, int arg1)
{
  int arg2 = 0;

  if (arg1 == 0)
    arg2 = ptr->domain | PERF_ENABLE;
  else if (arg2 == 1)
    arg2 = ptr->domain;

  ptr->counter_cmd[arg1] = arg2;
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
  ptr->counter_cmd[0] |= def_mode | PERF_ENABLE;
  ptr->counter_cmd[1] |= def_mode;
  ptr->counter_cmd[2] = 0;
  ptr->domain = def_mode;
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

  this_state->counter_cmd[0] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd[0] |= mode0;
  this_state->counter_cmd[1] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd[1] |= mode1;

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

inline static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  if (a->counter_cmd[cntr] == b->counter_cmd[cntr])
    return(1);

  return(0);
}

inline static int update_global_hwcounters(EventSetInfo *global)
{
  /* hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep; */
  unsigned long long events[PERF_COUNTERS];
  int i, ret;

  ret = perf(PERF_FASTREAD, (int)events, 0);
  if (ret)
    return(PAPI_ESYS);
  
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+events[i],
	   global->hw_start[i],events[i]));
      global->hw_start[i] = global->hw_start[i] + events[i];
    }

  ret = perf(PERF_RESET_COUNTERS, 0, 0);
  if (ret)
    return(PAPI_ESYS);

  return(PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

inline static int set_inherit(int arg)
{
  int r;

  if (arg)
    arg = 1;

  r = perf(PERF_SET_OPT, PERF_DO_CHILDREN, arg);
  if (r != 0)
    return(PAPI_ESYS);

  return(PAPI_OK);
}

inline static int get_cpu_num(void)
{
  return smp_processor_id();
}

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

static float calc_mhz(void)
{
  unsigned long long ostamp;
  unsigned long long stamp;
  float correction = 3000.0, mhz;

  /* Warm the cache */

  usleep(1);
  perf_get_cycles();

  ostamp = perf_get_cycles();
  sleep(1);
  stamp = perf_get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  return(mhz);
}

static int get_system_info(void)
{
  int tmp;
  char line[PAPI_MAX_STR_LEN], *t, *s;
  FILE *f;
  float mhz;

  if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
    return -1;
 
  _papi_system_info.cpunum = get_cpu_num();
  _papi_system_info.hw_info.ncpu = NR_CPUS;
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus = NR_CPUS;
  _papi_system_info.hw_info.vendor = -1;

  rewind(f);
  s = search_cpu_info(f,"vendor_id",line);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_system_info.hw_info.vendor_string,s+2);
    }

  rewind(f);
  t = s = search_cpu_info(f,"model name",line);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_system_info.hw_info.model_string,s+2);
    }

  rewind(f);
  if (t) /* 2.2 or greater */
    {
      s = search_cpu_info(f,"model",line);
      if (s)
	sscanf(s+1, "%d", &_papi_system_info.hw_info.model);
    }
  else
    {
      s = search_cpu_info(f,"model",line);
      if (s && (t = strchr(s+2,'\n')))
	{
	  *t = '\0';
	  strcpy(_papi_system_info.hw_info.model_string,s+2);
	}
      _papi_system_info.hw_info.model = 0;
    }

  rewind(f);
  s = search_cpu_info(f,"stepping",line);
  if (s)
    sscanf(s+1, "%d", &tmp);
  _papi_system_info.hw_info.revision = (float)tmp;

  rewind(f);
  s = search_cpu_info(f,"cpu MHz",line);
  if (s)
    sscanf(s+1, "%f", &_papi_system_info.hw_info.mhz);

  mhz = calc_mhz();
  if (_papi_system_info.hw_info.mhz < mhz)
    _papi_system_info.hw_info.mhz = mhz;

  fclose(f);

  sprintf(line,"/proc/%d/cmdline",(int)getpid());
  if ((f = fopen(line, "r")) == NULL)
    return PAPI_ESYS;
  fscanf(f,"%s",line);
  sprintf(_papi_system_info.exe_info.name,"%s",basename(line));
  DBG((stderr,"Executable is %s\n",_papi_system_info.exe_info.name));
  getcwd(_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN);
  strcat(_papi_system_info.exe_info.fullname,"/");
  strcat(_papi_system_info.exe_info.fullname,line);
  DBG((stderr,"Full Executable is %s\n",_papi_system_info.exe_info.fullname));

  tmp = setup_all_presets(&_papi_system_info.hw_info);
  if (tmp)
    return(tmp);

  return(PAPI_OK);
} 

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

long long _papi_hwd_get_real_usec (void)
{
  return((long long)((float)perf_get_cycles()/_papi_system_info.hw_info.mhz));
}
long long _papi_hwd_get_real_cycles (void)
{
  return(perf_get_cycles());
}

void _papi_hwd_error(int error, char *where)
{
  abort();
}

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

/* Do not ever use ESI->NumberOfCounters in here. */

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector = 0;
  int avail = 0;
  unsigned char tmp_cmd[PERF_COUNTERS];
  unsigned char *codes;

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

      codes = preset_map[preset_index].counter_cmd;
      ESI->EventInfoArray[index].command = derived;
      ESI->EventInfoArray[index].operand_index = preset_map[preset_index].operand_index;
    }
  else
    {
      int hwcntr_num;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  /* 0 through 2 */ 
      if (hwcntr_num > _papi_system_info.num_gp_cntrs)
	return(PAPI_EINVAL);

      tmp_cmd[hwcntr_num] = EventCode >> 8; 
      /* if (tmp_cmd[hwcntr_num] > 50)
	return(PAPI_EINVAL); */

      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->selector & selector)
	return(PAPI_ECNFLCT);	    

      codes = tmp_cmd;
    }

  /* Lower three bits tell us what counters we need */

  assert((this_state->selector | 0x7) == 0x7);
  
  /* Perform any initialization of the control bits */

  if (this_state->selector == 0)
    init_config(this_state);
  
  /* Turn on the bits for this counter */

  set_hwcntr_codes(selector,codes,this_state->counter_cmd);

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
      selector = EventCode >> 24 & 0x7;

      /* There must be only one event for custom encodings */

      if ((selector != 0x1) && (selector != 0x2) && (selector != 0x4))
	return(PAPI_EINVAL);
    }

  /* Check if these counters aren't used. */

  if ((used & selector) != used)
    return(PAPI_EINVAL);

  /* Lower three bits tell us what counters we need */

  assert((this_state->selector | 0x7) == 0x7);
  
  /* Clear out counters that are part of this event. */

  if (selector & 0x1)
    {
      unset_config(this_state,0);
      this_state->selector ^= 0x1;
    }
  if (selector & 0x2)
    {
      unset_config(this_state,1);
      this_state->selector ^= 0x2;
    }
  if (selector & 0x4)
    {
      unset_config(this_state,2);
      this_state->selector ^= 0x4;
    }

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, int index, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

void dump_cmd(int *t)
{
  int i;

  for (i=0;i<PERF_COUNTERS;i++)
    fprintf(stderr,"Event %d: 0x%x\n",i,t[i]);
}

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  
  /* If we are nested, merge the global counter structure
     with the current eventset */

  if (current_state->selector)
    {
      int hwcntrs_in_both, hwcntr;

      /* Stop the current context */

      retval = perf(PERF_STOP, 0, 0);
      if (retval) 
	return(PAPI_ESYS); 
  
      /* Update the global values */

      retval = update_global_hwcounters(zero);
      if (retval)
	return(retval);

      /* Delete the current context */

      hwcntrs_in_both = this_state->selector & current_state->selector;

      for (i = 0; i < _papi_system_info.num_cntrs; i++)
	{
	  /* Check for events that are shared between eventsets and 
	     therefore require no modification to the control state. */
	  
	  hwcntr = 1 << i;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      if (counter_shared(this_state, current_state, i))
		zero->multistart.SharedDepth[i]++;
	      else
		return(PAPI_ECNFLCT);
	      ESI->hw_start[i] = zero->hw_start[i];
	    }

	  /* Merge the unshared configuration registers. */
	  
	  else if (this_state->selector & hwcntr)
	    {
	      current_state->selector |= hwcntr;
	      current_state->counter_cmd[i] = this_state->counter_cmd[i];
	      ESI->hw_start[i] = 0;
	    }
	}
    }
  else
    {
      /* If we are NOT nested, just copy the global counter 
	 structure to the current eventset */

      current_state->selector = this_state->selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,PERF_COUNTERS*sizeof(int));

    }

  /* Set up the new merged control structure */
  
#ifdef DEBUG
  dump_cmd(current_state->counter_cmd);
#endif
      
  /* Stop the current context */

  retval = perf(PERF_RESET, 0, 0);
  if (retval) 
    return(PAPI_ESYS); 

  /* (Re)start the counters */
  
  retval = perf(PERF_FASTCONFIG, (int)current_state->counter_cmd, (int)NULL);
  if (retval) 
    return(PAPI_ESYS);

  retval = perf(PERF_START, 0, 0);
  if (retval) 
    return(PAPI_ESYS); 
  
  return(PAPI_OK);
} 

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, hwcntr, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  retval = perf(PERF_STOP, 0, 0);
  if (retval) 
    return(PAPI_ESYS); 
  
  for (i = 0; i < _papi_system_info.num_cntrs; i++)
    {
      /* Check for events that are NOT shared between eventsets and 
	 therefore require modification to the control state. */
      
      hwcntr = 1 << i;
      if (hwcntr & this_state->selector)
	{
	  if (zero->multistart.SharedDepth[i] - 1 < 0)
	    current_state->selector ^= hwcntr;
	  else
	    zero->multistart.SharedDepth[i]--;
	}
    }

  /* If we're not the outermost EventSet, then we need to start again 
     because someone is still running. */

  if (zero->multistart.num_runners - 1)
    {
      retval = perf(PERF_START, 0, 0);
      if (retval) 
	return(PAPI_ESYS); 
    }
  else
    {
      /* retval = pm_delete_program_mythread();
      if (retval > 0) 
	return(retval); */
    }

  return(PAPI_OK);
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
  return((long long)((float)units * _papi_system_info.hw_info.mhz * 1000000.0 / (float)cycles));
}

static long long handle_derived_ps(int operand_index, int selector, long long *from)
{
  int pos;

  pos = ffs(selector ^ (1 << operand_index)) - 1;
  assert(pos != -1);

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
  long long correct[PERF_COUNTERS];

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

      assert(selector != 0);
      DBG((stderr,"Event index %d, selector is 0x%x\n",j,selector));

      /* If this is not a derived event */

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

int _papi_hwd_write(EventSetInfo *ESI, long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(EventSetInfo *zero)
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

  abort();
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->eip;

  return(location);
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
				 -1  /*  mhz */ 
			       },
			       {
				 "",
				 "",
				 (caddr_t)&_start,
				 (caddr_t)&_etext,
				 (caddr_t)&data_start,
				 (caddr_t)&_edata,
				 (caddr_t)&_edata+1,
				 (caddr_t)&_fini,
				 "LD_PRELOAD", /* How to preload libs */
			       },
			        2,  /*  num_cntrs */
			        2,  /*  num_gp_cntrs */
			        0,  /*  grouped_counters */
			        0,  /*  num_sp_cntrs */
			       -1,  /*  total_presets */
			       -1,  /*  total_events */
			        1,  /*  needs overflow emulation */
			        1,  /*  needs profile emulation */
			        0,  /*  needs 64 bit virtual counters */
			        1,  /*  supports child inheritance option */
			        0,  /*  can attach to another process */
			        0,  /*  read resets the counters */
			        PAPI_DOM_USER, /* default domain */
			        PAPI_GRN_THR,  /* default granularity */
			        sizeof(hwd_control_state_t), 
			        NULL };

