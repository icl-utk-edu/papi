/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "tru64-alpha.h"

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };

static hwd_search_t findem_ev4[PAPI_MAX_PRESET_EVENTS] = { 
                { 0, { -1, PF_DCACHE, -1 } },  /* L1 D-Cache misses  */
                { 0, { -1, PF_ICACHE, -1 } },	/* L1 I-Cache misses  */
		{ 0, { -1, -1, -1 } },	/* L2 D-Cache misses */
		{ 0, { -1, -1, -1 } },	/* L2 I-ditto */
		{ 0, { -1, -1, -1 } },	/* L3 D-misses */
		{ 0, { -1, -1, -1 } },	/* L3 I-misses */
		{ 0, { -1, -1, -1 } },	/* L1 total */
		{ DERIVED_ADD, { PF_EXTPIN0, PF_EXTPIN1, -1 } }, /* L2 total */
		{ 0, { -1, -1, -1 } },	/* L3 total */
		{ 0, { -1, -1, -1 } },	/* snoop */
		{ 0, { -1, -1, -1 } }, 	/* Req. access to shared cache line */
		{ 0, { -1, -1, -1 } }, 	/* Req. access to clean cache line */
		{ 0, { -1, -1, -1 } }, 	/* Cache Line Invalidation */
                { 0, { -1, -1, -1 } },	/* Cache line intervention */
                { 0, { -1, -1, -1 } },	/* L3 LM */
                { 0, { -1, -1, -1 } },	/* L3 SM */
                { 0, { -1, -1, -1 } },	/* BRU idle cyc */
                { 0, { -1, -1, -1 } },	/* ALU idle cyc */
                { 0, { -1, -1, -1 } },	/* FPU idle cyc */
                { 0, { -1, -1, -1 } },	/* LD/ST idle cyc */
		{ 0, { -1, -1, -1 } },  /* D-TLB misses */
		{ 0, { -1, -1, -1 } },	/* I-TLB misses */
                { 0, { -1, -1, -1 } },	/* Total TLB misses */
                { 0, { -1, -1, -1 } },	/* L1 LM */
                { 0, { -1, -1, -1 } },	/* L1 SM */ 
                { 0, { -1, -1, -1 } },	/* L2 LM */
                { 0, { -1, -1, -1 } },	/* L2 SM */
                { 0, { -1, -1, -1 } },	/* BTAC miss */
                { 0, { -1, -1, -1 } },	/* Prefetch data caused a miss */
                { 0, { -1, -1, -1 } },	
		{ 0, { -1, -1, -1 } },	/* TLB shootdowns */
                { 0, { -1, -1, -1 } },	/* Failed store cond. */
                { 0, { -1, -1, -1 } },	/* Suc. store cond. */
                { 0, { -1, -1, -1 } },	/* Tot. store cond. */
                { 0, { -1, -1, -1 } },	/* Cycles stalled waiting for memory */
                { 0, { -1, -1, -1 } },	/* Cycles stalled waiting for memory read */
                { 0, { -1, -1, -1 } },	/* Cycles stalled waiting for memory write */
                { 0, { PF_NONISSUES, -1, -1 } }, /* Cycles no instructions issued */
                { 0, { -1, PF_DUAL, -1 } },	/* Cycles max instructions issued */
                { 0, { PF_NONISSUES, -1, -1 } },/* Cycles no instructions comlpeted */
		{ 0, { -1, PF_DUAL, -1 } },	/* Cycles max instructions completed */
                { 0, { -1, -1, -1 } },	/* hardware interrupts */
		{ 0, { -1, -1, -1 } },	/* Uncond. branches executed */
		{ 0, { -1, -1, -1 } },	/* Cond. branch inst. executed*/
		{ 0, { -1, -1, -1 } },	/* Cond. branch inst. taken*/
		{ 0, { -1, -1, -1 } },	/* Cond. branch inst. not taken*/
		{ 0, { -1, PF_BRANCHMISS, -1 } },	/* Cond. branch inst. mispred.*/
                { 0, { -1, -1, -1 } },	/* Cond. branch inst. corr. pred */
                { 0, { -1, -1, -1 } },	/* FMA instructions */
                { 0, { PF_ISSUES, -1, -1 } },	/* Total instructions issued */
		{ 0, { PF_ISSUES, -1, -1 } },	/* Total inst. executed */
		{ 0, { -1, PF_INTOPS, -1 } },	/* Integer inst. executed */
		{ 0, { -1, PF_FPINST, -1 } },	/* Floating Pt. inst. executed */
		{ 0, { PF_LOADI, -1, -1 } },	/* Loads executed */
		{ 0, { -1, PF_STOREI, -1 } },	/* Stores executed */
		{ 0, { PF_BRANCHI, -1, -1 } },	/* Branch inst. executed */
		{ 0, { -1, -1, -1 } },	/* Vector/SIMD inst. executed  */
		{ DERIVED_PS, { PF_CYCLES, PF_FPINST, -1 } },	/* FLOPS */
                { 0, { PF_PIPEFROZEN, -1, -1 } },	/* Cycles stalled */
                { 0, { -1, -1, -1 } },	/* Cycles FP stalled */
		{ 0, { PF_CYCLES, -1, -1 } },	/* Total cycles */
		{ DERIVED_PS, { PF_CYCLES, PF_ISSUES, -1 } },	/* IPS */
                { DERIVED_ADD, { PF_LOADI, PF_STOREI, -1 } },	/* Total load/stores */
                { 0, { -1, -1, -1 } }	/* Syncs */ };

/* Utility functions */

static int setup_all_presets(int model)
{
  int first, event, pnum, derived, hwnum;
  hwd_search_t *findem;
  char str[PAPI_MAX_STR_LEN];
  
  if (model == 0)
    findem = findem_ev4;
  else
    return(PAPI_ESBSTR);
  
  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      first = -1;
      derived = findem[pnum].derived;
      if (derived == -1)
	continue;
      for (hwnum = 0; hwnum < _papi_system_info.num_cntrs; hwnum++)
	{
	  event = findem[pnum].findme[hwnum];
	  if (event == -1)
	    continue;
	  if (first == -1)
	    first = event;

	  preset_map[pnum].selector |= 1 << event;
	  preset_map[pnum].counter_cmd[hwnum] = event;

	  sprintf(str,"%d",event);
	  if ((strlen(preset_map[pnum].note) != (size_t)0))
	    strcat(preset_map[pnum].note,",");
	  strcat(preset_map[pnum].note,str);
	  
	  DBG((stderr,"setup_all_presets(%d): Preset %d, event %d found, selector 0x%x\n",
	       model,pnum,event,preset_map[pnum].selector));
	}				
      if (preset_map[pnum].selector)
	{
	  preset_map[pnum].derived = derived;
	  if (derived)
	    preset_map[pnum].operand_index = first;
	  else
	    preset_map[pnum].operand_index = -1;
	}
    }

  return(PAPI_OK);
}

static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  /* ptr->counter_cmd.events[arg1] = arg2; */
}

static void unset_config(hwd_control_state_t *ptr, int arg1)
{
  /* ptr->counter_cmd.events[arg1] = 0; */
}

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  /* if (a->counter_cmd.events[cntr] == b->counter_cmd.events[cntr])
    return(1); */

  return(0);
}

static int update_global_hwcounters(EventSetInfo *global)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)global->machdep;
  int retval;
  struct pfcntrs hwcntrs;

  retval = ioctl(current_state->fd, PCNTGETCNT, &hwcntrs);
  if (retval == -1)
    return(PAPI_ESYS);

  if (current_state->selector & 0x1)
    {
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",0,
	   global->hw_start[0]+hwcntrs.pf_cntr0,global->hw_start[0],hwcntrs.pf_cntr0));
      global->hw_start[0] = global->hw_start[0] + hwcntrs.pf_cntr0;
    }

  if (current_state->selector & 0x2)
    {
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",1,
	   global->hw_start[1]+hwcntrs.pf_cntr1,global->hw_start[1],hwcntrs.pf_cntr1));
      global->hw_start[1] = global->hw_start[1] + hwcntrs.pf_cntr1;
    }

  retval = ioctl(current_state->fd, PCNTCLEARCNT);
  if (retval == -1)
    return(PAPI_ESYS);

  return(0);
}

static int correct_local_hwcounters(EventSetInfo *global, EventSetInfo *local, long long *correct)
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

static int set_domain(hwd_control_state_t *this_state, int domain)
{
  return(PAPI_ESBSTR);
}

static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  return(PAPI_ESBSTR);
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

static int set_inherit(EventSetInfo *zero, int arg)
{
  return(PAPI_ESBSTR);
}

static void init_config(hwd_control_state_t *ptr)
{
  ptr->counter_cmd.items = PFM_COUNTERS;
  memset(&ptr->counter_cmd.mux,0x0,sizeof(struct iccsr));
  /* set_domain(ptr,_papi_system_info.default_domain);
  set_granularity(ptr,_papi_system_info.default_granularity); */
}

static int get_system_info(void)
{
  int fd, retval;
  prpsinfo_t info;
  struct cpu_info cpuinfo;
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

  /* retval = pm_init(0,&tmp);
  if (retval > 0)
    return(retval); */

  if (getsysinfo(GSI_CPU_INFO, (char *)&cpuinfo, sizeof(cpuinfo), NULL, NULL, NULL) == -1)
    return PAPI_ESYS;

  _papi_system_info.cpunum = cpuinfo.current_cpu;
  _papi_system_info.hw_info.revision = cpu_implementation_version();
  _papi_system_info.hw_info.mhz = (float)cpuinfo.mhz;
  _papi_system_info.hw_info.ncpu = cpuinfo.cpus_in_box;
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus = 
    _papi_system_info.hw_info.ncpu * _papi_system_info.hw_info.nnodes;
  _papi_system_info.hw_info.vendor = -1;
  strcpy(_papi_system_info.hw_info.vendor_string,"Compaq");
  _papi_system_info.hw_info.model = cpuinfo.cpu_type;

  _papi_system_info.num_sp_cntrs = 1;
  if (_papi_system_info.hw_info.revision == 0)
    {
      _papi_system_info.num_cntrs = 2;
      _papi_system_info.num_gp_cntrs = 2;
      strcpy(_papi_system_info.hw_info.model_string,"Alpha 21064"); 
    }
  else if (_papi_system_info.hw_info.revision == 2)
    {
      _papi_system_info.num_cntrs = 2;
      _papi_system_info.num_gp_cntrs = 2;
      strcpy(_papi_system_info.hw_info.model_string,"Alpha 21264"); 
    }
  else if (_papi_system_info.hw_info.revision == 1)
    {
      _papi_system_info.num_cntrs = 3;
      _papi_system_info.num_gp_cntrs = 3;
      strcpy(_papi_system_info.hw_info.model_string,"Alpha 21164");
    }
  else
    return(PAPI_ESBSTR);

  retval = setup_all_presets(_papi_system_info.hw_info.revision);
  if (retval)
    return(retval);

  return(PAPI_OK);
} 

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

long long _papi_hwd_get_real_usec (void)
{
  struct timespec t;
  long long retval;

  if (getclock(CLOCK_REALTIME,&t) == -1)
    return PAPI_ESYS;

  retval = t.tv_sec * 1000000 + t.tv_nsec / 1000;
  return(retval);
}

long long _papi_hwd_get_real_cycles (void)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_real_usec();
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
}

long long _papi_hwd_get_virt_usec (void)
{
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != -1)
    return((long long)usage.ru_utime.tv_usec + ((long long)usage.ru_utime.tv_sec * (long long)1000000));
  else
    return(-1);
}

long long _papi_hwd_get_virt_cycles (void)
{
  return(_papi_hwd_get_virt_usec() * (long long)_papi_system_info.hw_info.mhz);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error");
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
  int fd;
  hwd_control_state_t *machdep = (hwd_control_state_t *)zero->machdep;

  /* Initialize our global machdep. */

  fd = open("/dev/pfcntr",O_RDONLY | PCNTOPENALL);
  if (fd == -1)
    return(PAPI_ESYS);

  machdep->fd = fd;

  ioctl(machdep->fd,PCNTRENABLE);

  ioctl(machdep->fd,PCNTLOGSELECT);
      
  init_config(zero->machdep); 

  return(PAPI_OK);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (_papi_system_info.num_cntrs);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
}

static int get_avail_hwcntr_num(int cntr_avail_bits)
{
  int tmp = 0, i = _papi_system_info.num_cntrs - 1;
  
  while (i)
    {
      tmp = (1 << i) & cntr_avail_bits;
      if (tmp)
	return(i);
      i--;
    }
  return(0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, ev4_command_t *to)
{
  int useme, i;
  
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  /* to[i] = from[i]; */
	}
    }
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector = 0;
  int avail = 0;
  unsigned char tmp_cmd[EV_MAX_COUNTERS];
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

      hwcntr_num = EventCode & 0xff;  /* 0 through 7 */ 
      if (hwcntr_num > _papi_system_info.num_gp_cntrs)
	return(PAPI_EINVAL);

      tmp_cmd[hwcntr_num] = EventCode >> 8; /* 0 through 50 */
      if (tmp_cmd[hwcntr_num] > 50)
	return(PAPI_EINVAL); 

      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->selector & selector)
	return(PAPI_ECNFLCT);	    

      codes = tmp_cmd;
    }

  /* Lower eight bits tell us what counters we need */

  assert((this_state->selector | 0xff) == 0xff);

  /* Perform any initialization of the control bits */

  if (this_state->selector == 0)
    init_config(this_state);
  
  /* Turn on the bits for this counter */

  set_hwcntr_codes(selector,codes,&this_state->counter_cmd);

  /* Update the new counter select field. */

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
      int hwcntr_num, code;
      
      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  /* 0 through 7 */ 
      if (hwcntr_num > _papi_system_info.num_gp_cntrs)
	return(PAPI_EINVAL);

      code = EventCode >> 8; /* 0 through 50 */
      if (code > 50)
	return(PAPI_EINVAL); 

      selector = 1 << hwcntr_num;
    }

  /* Check if these counters aren't used. */

  if ((used & selector) != used)
    return(PAPI_EINVAL);

  /* Clear out counters that are part of this event. */
  /* Remember, that selector might encode duplicate events
     so we need to know only the ones that are used. */
  
  this_state->selector = this_state->selector ^ (selector & used);

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, int index, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

void dump_cmd(ev4_command_t *t)
{
  fprintf(stderr,"Command block at %p: items 0x%x\n",t,t->items);
  fprintf(stderr,"iccsr_pc1 %d\n",t->mux.iccsr_pc1);
  fprintf(stderr,"iccsr_pc0 %d\n",t->mux.iccsr_pc0);
  fprintf(stderr,"iccsr_mux0 0x%x\n",t->mux.iccsr_mux0);
  fprintf(stderr,"iccsr_mux1 0x%x\n",t->mux.iccsr_mux1);
  fprintf(stderr,"iccsr_disable 0x%x\n",t->mux.iccsr_disable);
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
	      /* current_state->counter_cmd.mode.w = this_state->counter_cmd.mode.w;
	      current_state->counter_cmd.events[i] = this_state->counter_cmd.events[i]; */
	      ESI->hw_start[i] = 0;
	    }
	}
    }
  else
    {
      /* If we are NOT nested, just copy the global counter 
	 structure to the current eventset */

      current_state->selector = this_state->selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(ev_command_t));

    }

  /* If overflowing is enabled, turn it on */
  
  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_start_overflow_timer(ESI, zero);
      if (retval < PAPI_OK)
	return(PAPI_EBUG);
    }

  /* Set up the new merged control structure */
  
#ifdef DEBUG
  dump_cmd(&current_state->counter_cmd);
#endif
      
  retval = ioctl(current_state->fd,PCNTSETITEMS,current_state->counter_cmd.items);
  if (retval == -1)
    return(PAPI_ESYS);

  /* (Re)start the counters */
  
  retval = ioctl(current_state->fd,PCNTSETMUX,&current_state->counter_cmd.mux);
  if (retval == -1)
    return(PAPI_ESYS);

  return(PAPI_OK);
} 

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, hwcntr, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  /* retval = pm_stop_mythread();
  if (retval > 0) 
    return(retval); */
  
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

  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_stop_overflow_timer(ESI, zero);
      if (retval < PAPI_OK)
	return(PAPI_EBUG);
    }
  
  /* If we're not the outermost EventSet, then we need to start again 
     because someone is still running. */

  if (zero->multistart.num_runners - 1)
    {
      /* retval = pm_start_mythread();
      if (retval > 0) 
	return(retval); */
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
  assert(pos != 0);

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

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long *events)
{
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  long long correct[EV_MAX_COUNTERS];

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
      return(set_inherit(zero, option->inherit.inherit));
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
  location = (void *)info->sc_pc;

  return(location);
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
				 "_RLD_LIST", /* How to preload libs */
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
			        0,  /* HW read resets the counters */
			        sizeof(hwd_control_state_t), 
			        { 0} };

