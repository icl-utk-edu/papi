/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* PAPI stuff */

#include "unicos-ev5.h"

/* First entry is counter code 1, counter code 2 and counter code 3.
   Then is the mask. There are no derived metrics for the T3E.
   Notes:
     Resource stalls only count long(>15 cycle) stalls and not MB stall cycles
      */

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                { CNTR3, NOT_DERIVED, -1, {-1,-1,0x6}, "" },  /* L1 D-Cache misses */
                { CNTR3, NOT_DERIVED, -1, {-1,-1,0x4}, "" },  /* L1 I-Cache misses */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L2 D-Cache misses */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L2 I-ditto */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L3 D-misses */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L3 I-misses */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L1 total */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2 total */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L3 total */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* snoop */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, 	/* Req. acc. to shared cache line */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, 	/* Req. acc. to clean cache line */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, 	/* Cache Line Invalidation */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Cache line intervention */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L3 LM */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L3 SM */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* BRU idle cyc */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* ALU idle cyc */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* FPU idle cyc */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* LD/ST idle cyc */
		{ CNTR3, NOT_DERIVED, -1, {-1,-1,0x7}, "" },  /* D-TLB misses */
		{ CNTR3, NOT_DERIVED, -1, {-1,-1,0x5}, "" },  /* I-TLB misses */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Total TLB misses */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L1 LM */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L1 SM */ 
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L2 LM */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* L2 SM */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* BTAC miss */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Prefetch data caused a miss */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* 29 */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* TLB shootdowns */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Failed store cond. */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Suc. store cond. */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Tot. store cond. */
                { CNTR3, NOT_DERIVED, -1, {-1,-1,0xd}, "" }, /* Cycles stl. wait for memory */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },  /* Cycles stl. wait for memory read */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },  /* Cycles stl. wait memory write */
                { CNTR2, NOT_DERIVED, -1, {-1,0x0,-1}, "" }, /* Cycles no instructions issued */
                { CNTR2, NOT_DERIVED, -1, {-1,0x7,-1}, "" }, /* Cycles max instructions issued */
                { CNTR2, NOT_DERIVED, -1, {-1,0x0,-1}, "" }, /* Cycles no instrs completed */
		{ CNTR2, NOT_DERIVED, -1, {-1,0x7,-1}, "" }, /* Cycles max instrs completed */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* hardware interrupts */
		{ CNTR2|CNTR3, DERIVED_ADD, 1, {-1,0x8,0x2}, "" },/* Uncond. branches executed */
		{ CNTR2|CNTR3, DERIVED_ADD, 1, {-1,0x8,0x3}, "" },/* Cond. branch inst. executed*/
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Cond. branch inst. taken*/
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Cond. branch inst. not taken*/
		{ CNTR3, NOT_DERIVED, -1, {-1,-1,0x3}, "" },  /* Cond. branch inst. mispred.*/
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Cond. branch inst. corr. pred */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* FMA instructions */
                { CNTR2, NOT_DERIVED, -1, {-1,0xd,-1}, "" },	/* Total instructions issued */
		{ CNTR1, NOT_DERIVED, -1, {0x1,-1,-1}, "" },	/* Total inst. executed */
		{ CNTR2, NOT_DERIVED, -1, {-1,0x9,-1}, "" },	/* Integer inst. executed */
		{ CNTR2, NOT_DERIVED, -1, {-1,0xa,-1}, "" },  /* Floating Pt. inst. executed */
		{ CNTR2, NOT_DERIVED, -1, {-1,0xb,-1}, "" },	/* Loads executed */
		{ CNTR2, NOT_DERIVED, -1, {-1,0xc,-1}, "" },	/* Stores executed */
		{ CNTR2, NOT_DERIVED, -1, {-1,0x8,-1}, "" },	/* Branch inst. executed */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Vector/SIMD inst. executed  */
		{ CNTR2|CNTR1, DERIVED_PS, 0, {0x0,0xa,-1}, "" },	/* FLOPS */
                { CNTR3, NOT_DERIVED, -1, {-1,-1,0x0}, "Counts only long (>15cyc) stalls, not MB stalls" },	/* Resource stalls */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* FPU stalled */
		{ CNTR1, NOT_DERIVED, -1, {0x0,-1,-1}, ""},   /* Total cycles */
		{ CNTR1|CNTR3, DERIVED_PS, 2, {0x1,-1,0xc}, "" },	/* IPS */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Total load/stores */
                { 0, NOT_DERIVED, -1, {-1,-1,-1}, "" },	/* Syncs */ 
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_DCH */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_DCH */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_DCA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_DCA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_DCA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_DCR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_DCR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_DCR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_DCW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_DCW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_DCW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_ICH */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_ICH */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_ICH */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_ICA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_ICA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_ICA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_ICR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_ICR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_ICR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_ICW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_ICW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_ICW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_TCH */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_TCH */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_TCH */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_TCA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_TCA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_TCA */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_TCR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_TCR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_TCR */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L1_TCW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L2_TCW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* L3_TCW */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* FPM */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* FPA */
   		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* FPD */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* FPSQ */
		{ 0, NOT_DERIVED, -1, {-1,-1,-1}, "" }, /* FPI */
};

/* Utility functions */

static int setup_all_presets(PAPI_hw_info_t *info)
{
  int pnum;

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      if (preset_map[pnum].selector)
	{
	  char str[PAPI_MAX_STR_LEN];
	  sprintf(str,"0x%x,0x%x,0x%x",
		  preset_map[pnum].counter_cmd[0],
		  preset_map[pnum].counter_cmd[1],
		  preset_map[pnum].counter_cmd[2]);
	  if (strlen(preset_map[pnum].note) != 0)
	    {
	      strcat(preset_map[pnum].note,str);
	      strcat(preset_map[pnum].note,":");
	    }
	  else
	    strcpy(preset_map[pnum].note,str);
	}
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

static int set_inherit(EventSetInfo *zero, pid_t pid)
{
  return(PAPI_ESBSTR);
}

static void init_config(hwd_control_state_t *ptr)
{
  int kill_pal = 0;
  int kill_user = 0;
  int kill_kernel = 0;

  memset(&ptr->counter_cmd,0x0,sizeof(pmctr_t));

  switch (_papi_system_info.default_domain)
    {
    case PAPI_DOM_USER:
      {
	kill_kernel = 0;
	kill_pal = 0;
	kill_user = 1;
      }
      break;
    case PAPI_DOM_KERNEL:
      {
	kill_kernel = 1;
	kill_pal = 0;
	kill_user = 0;
      }
      break;
    case PAPI_DOM_OTHER:
      {
	kill_kernel = 0;
	kill_pal = 1;
	kill_user = 0;
      }
      break;
    case PAPI_DOM_ALL:
      {
	kill_kernel = 1;
	kill_pal = 1;
	kill_user = 1;
      }
    default:
      abort();
    }

  ptr->counter_cmd.Kp = kill_pal;
  ptr->counter_cmd.Ku = kill_user;
  ptr->counter_cmd.Kk = kill_kernel;
  ptr->counter_cmd.CTL0 = CTL_OFF;
  ptr->counter_cmd.CTL1 = CTL_OFF;
  ptr->counter_cmd.CTL2 = CTL_OFF;
}

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  if ((cntr == 0) && (a->counter_cmd.CTL0 == b->counter_cmd.CTL0))
    return(1);
  else if ((cntr == 1) && (a->counter_cmd.CTL1 == b->counter_cmd.CTL1))
    return(1);
  else if ((cntr == 2) && (a->counter_cmd.CTL2 == b->counter_cmd.CTL2))
    return(1);

  return(0);
}

static int update_global_hwcounters(EventSetInfo *global)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)global->machdep;
  int retval, selector;
  pmctr_t *pmctr;
  long pc_data[4];

  retval = _rdperf(pc_data);
  if (retval != 0)
    return(PAPI_ESBSTR);

  pmctr = (pmctr_t *)&pc_data[0];

  selector = this_state->selector;
  if (selector & 0x1)
    {
      DBG((stderr,"update_global_counters() %d: G%lld = G%lld + C%lld\n",0,
	   global->hw_start[0]+((pc_data[1] << 16) + pmctr->CTR0),
	   global->hw_start[0],
	   ((pc_data[1] << 16) + pmctr->CTR0)));	   
      global->hw_start[0] = global->hw_start[0] + (pc_data[1] << 16) + pmctr->CTR0;
    }
  if (selector & 0x2)
    {
      DBG((stderr,"update_global_counters() %d: G%lld = G%lld + C%lld\n",1,
	   global->hw_start[1]+((pc_data[2] << 16) + pmctr->CTR1),
	   global->hw_start[1],
	   ((pc_data[2] << 16) + pmctr->CTR1)));	   
      global->hw_start[1] = global->hw_start[1] + (pc_data[2] << 16) + pmctr->CTR1;
    }
  if (selector & 0x4)
    {
      DBG((stderr,"update_global_counters() %d: G%lld = G%lld + C%lld\n",2,
	   global->hw_start[2]+((pc_data[3] << 14) + pmctr->CTR2),
	   global->hw_start[2],
	   ((pc_data[3] << 14) + pmctr->CTR2)));	   
      global->hw_start[2] = global->hw_start[2] + (pc_data[3] << 14) + pmctr->CTR2;
    }
  
  retval = _wrperf(this_state->counter_cmd,0,0,0);
  if (retval)
    return(PAPI_ESBSTR);

  return(PAPI_OK);
}

static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  return(PAPI_ESBSTR);
}

static int correct_local_hwcounters(EventSetInfo *global, EventSetInfo *local, long long *correct)
{
  int i;
  hwd_control_state_t *machdep = (hwd_control_state_t *)local->machdep;
  int selector = machdep->selector;

  while (i = ffs(selector))
    {
      i = i - 1;
      selector ^= 1 << i;
      DBG((stderr,"correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n",i,
	   global->hw_start[i]-local->hw_start[i],global->hw_start[i],local->hw_start[i]));
      correct[i] = global->hw_start[i] - local->hw_start[i];
    }

  return(0);
}

static int set_domain(hwd_control_state_t *this_state, int domain)
{
  if (domain & PAPI_DOM_USER)
    this_state->counter_cmd.Ku = 1;
  else
    this_state->counter_cmd.Ku = 0;
  if (domain & PAPI_DOM_KERNEL)
    this_state->counter_cmd.Kk = 1;
  else
    this_state->counter_cmd.Kk = 0;
  if (domain & PAPI_DOM_OTHER)
    this_state->counter_cmd.Kp = 1;
  else
    this_state->counter_cmd.Kp = 0;

  return(PAPI_OK);
}

static float getmhz(void)
{
  long sysconf(int request);
  float p;
  
  p = (float) sysconf(_SC_CRAY_CPCYCLE); /* Picoseconds */
  p = p * 1.0e-12; /* Convert to seconds */
  p = (int)(1.0 / (p * 1000000.0)); /* Convert to MHz */
  return(p);
}

static int get_system_info(void)
{
  pid_t pid;
  int tmp;

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);

  /* _papi_system_info.exe_info.fullname; */
  /* _papi_system_info.exe_info.name; */
  _papi_system_info.exe_info.text_start = (caddr_t)0x800000000;
  _papi_system_info.exe_info.data_start = (caddr_t)0x100000000;
  _papi_system_info.exe_info.bss_start =  (caddr_t)0x200000000;
  _papi_system_info.exe_info.text_end = (caddr_t)_infoblk.i_segs[0].size;
  _papi_system_info.exe_info.data_end = (caddr_t)_infoblk.i_segs[1].size;
  _papi_system_info.exe_info.bss_end = (caddr_t)_infoblk.i_segs[2].size;

  _papi_system_info.hw_info.ncpu = sysconf(_SC_CRAY_NCPU);
  _papi_system_info.hw_info.totalcpus = sysconf(_SC_CRAY_NCPU);
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.mhz = getmhz();
  strcpy(_papi_system_info.hw_info.vendor_string,"Cray");
  _papi_system_info.hw_info.vendor = -1;
  _papi_system_info.hw_info.revision = 0.0;
  _papi_system_info.hw_info.model = -1;
  strcpy(_papi_system_info.hw_info.model_string,"Alpha 21164");

  _papi_system_info.cpunum = sysconf(_SC_CRAY_PPE);

  tmp = setup_all_presets(&_papi_system_info.hw_info);
  if (tmp)
    return(tmp);

  return(PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

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

int _papi_hwd_init(EventSetInfo *global)
{
  return(PAPI_OK);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (3-1);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, pmctr_t *to)
{
  if (selector & 0x1)
    {
      to->CTL0 = CTL_ON;
      to->SEL0 = from[0];
    }
  if (selector & 0x2)
    {
      to->CTL1 = CTL_ON;
      to->SEL1 = from[1];
    }
  if (selector & 0x4)
    {
      to->CTL2 = CTL_ON;
      to->SEL2 = from[2];
    }
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector = 0;
  int avail = 0;
  unsigned char tmp_cmd[3];
  unsigned char *codes;


  if (EventCode & PRESET_MASK)
    { 
      int preset_index;
      int derived;

      preset_index = EventCode ^ PRESET_MASK; 

      selector = preset_map[preset_index].selector;
      if (selector == 0x0)
	return(PAPI_ENOEVNT);
      derived = preset_map[preset_index].derived;

      /* Find out which counters are available. */

      avail = selector & ~this_state->selector;

      /* If not derived */

      if (derived == 0)
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
      if ((hwcntr_num > _papi_system_info.num_gp_cntrs) ||
	  (hwcntr_num < 0))
	return(PAPI_EINVAL);

      tmp_cmd[hwcntr_num] = EventCode >> 8; /* 0 through 50 */
      if (tmp_cmd[hwcntr_num] > 0xff)
	return(PAPI_EINVAL); 

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

      hwcntr_num = EventCode & 0xff;  /* 0 through 2 */ 
      if ((hwcntr_num > _papi_system_info.num_gp_cntrs) ||
	  (hwcntr_num < 0))
	return(PAPI_EINVAL);

      code = EventCode >> 8; /* 0 through 50 */
      if (code > 0xff)
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

void dump_cmd(pmctr_t *t)
{
  fprintf(stderr,"Command block at %p\n",t);
  fprintf(stderr,"SEL0: %d\n",t->SEL0);
  fprintf(stderr,"SEL1: %d\n",t->SEL1);
  fprintf(stderr,"SEL2: %d\n",t->SEL2);
  fprintf(stderr,"CTL0: %d\n",t->CTL0);
  fprintf(stderr,"CTL1: %d\n",t->CTL1);
  fprintf(stderr,"CTL2: %d\n",t->CTL2);
  fprintf(stderr,"Ku: %d\n",t->Ku);
  fprintf(stderr,"Kp: %d\n",t->Kp);
  fprintf(stderr,"Kk: %d\n",t->Kk);
}

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

      /* Stop the current context 

      retval = pm_stop_mythread();
      if (retval > 0) 
	return(retval); */
  
      /* Update the global values */

      retval = update_global_hwcounters(zero);
      if (retval)
	return(retval);

      /* Delete the current context 

      retval = pm_delete_program_mythread();
      if (retval > 0)
	return(retval); */

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
	      ESI->hw_start[i] = 0;
	      zero->hw_start[i] = 0;
	      if (hwcntr == 0x1)
		{
		  current_state->counter_cmd.CTL0 = this_state->counter_cmd.CTL0;
		  current_state->counter_cmd.SEL0 = this_state->counter_cmd.SEL0;
		}
	      else if (hwcntr == 0x2)
		{
		  current_state->counter_cmd.CTL1 = this_state->counter_cmd.CTL1;
		  current_state->counter_cmd.SEL1 = this_state->counter_cmd.SEL1;
		}
	      else if (hwcntr == 0x4)
		{
		  current_state->counter_cmd.CTL2 = this_state->counter_cmd.CTL2;
		  current_state->counter_cmd.SEL2 = this_state->counter_cmd.SEL2;
		}
	      else
		abort();
	    }
	}
    }
  else
    {
      /* If we are NOT nested, just copy the global counter 
	 structure to the current eventset */

      current_state->selector = this_state->selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(pmctr_t));

    }

  /* Set up the new merged control structure */
  
#if 0
  dump_cmd(&current_state->counter_cmd);
#endif
      
  /* Start the counters with the new merged event set machdep structure */

  retval = _wrperf(this_state->counter_cmd,0,0,0);
  if (retval)
    return(PAPI_ESBSTR);

  /* (Re)start the counters 
  
  retval = pm_start_mythread();
  if (retval > 0) 
    return(retval); */

  return(PAPI_OK);

} 

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, hwcntr;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

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

  while (pos = ffs(selector))
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

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long events[])
{
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  long long correct[3];

  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  retval = correct_local_hwcounters(zero, ESI, correct);
  if (retval)
    return(retval);

  /* This routine distributes hardware counters to software counters in the
     order that they were added. Note that the higher level 
     EventSelectArray[i] entries may not be contiguous because the user
     has the right to remove an event. */

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      selector = ESI->EventInfoArray[i].selector;
      if (selector == PAPI_NULL)
	continue;

      DBG((stderr,"Event %d, mask is 0x%x\n",j,selector));

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

int _papi_hwd_shutdown_global(void)
{
  return(PAPI_OK);
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

void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, ucontext_t *info)
{
  DBG((stderr,"_papi_hwd_dispatch_timer() at 0x%lx\n",info->uc_mcontext.gregs[31]));
  _papi_hwi_dispatch_overflow_signal((void *)info); 
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

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *overflow_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  ucontext_t *info = (ucontext_t *)context;

  location = (void *)info->uc_mcontext.gregs[31];

  return(location);
}

/* 75 Mhz sys. clock */

long long _papi_hwd_get_real_cycles (void)
{
  return(_rtc()*(long long)(_papi_system_info.hw_info.mhz/75.0));
}

long long _papi_hwd_get_real_usec (void)
{
  return(_rtc()/75);
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

void _papi_hwd_lock_init(void)
{
}

#define PIF_LOCK        (010)
#pragma _CRI soft $MULTION

extern $MULTION(void);

void _papi_hwd_lock(void)
{
    if ($MULTION == 0) _semts(PIF_LOCK);
    return;
}
void _papi_hwd_unlock(void)
{ 
    if ($MULTION == 0) _semclr(PIF_LOCK);
    return;
}

/* Machine info structure. -1 is unused. */

papi_mdi _papi_system_info = { "$Id$",
			       1.0, /*  version */
			       -1,  /*  cpunum */
			       {
				 -1,  /*  ncpu */
				 -1,  /*  nnodes */
				 -1,  /*  totalcpus */
				 -1,  /*  vendor */
				 "",  /*  vendor string */
				 -1,  /*  model */
				 "",  /*  model string */
				 0.0, /* revision */
				 -1   /* mhz */
			       },
			       {
				 "",
				 "",
				 NULL,
				 NULL,
				 NULL,
				 NULL,
				 NULL,
				 NULL,
				 NULL
			       },
			        3,   /*  num_cntrs */
			        3,   /*  num_gp_cntrs */
			       -1,   /*  grouped_counters */
			       -1,   /*  num_sp_cntrs */
			       -1,   /*  total_presets */
			       -1,   /*  total_events */
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
				{ 0, } };

/* Thread hooks 

void __pdf_th_create(void)
{
  extern PAPI_notify_handler_t thread_notifier;
}

void __pdf_th_destroy(void)
{
  extern PAPI_notify_handler_t thread_notifier;
}

*/
