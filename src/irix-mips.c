/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "irix-mips.h"

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };

static hwd_search_t findem_r10k[PAPI_MAX_PRESET_EVENTS] = {
                {  0,{-1,25}},			/* L1 D-Cache misses */
                {  0,{ 9,-1}},		        /* L1 I-Cache misses */
		{  0,{-1,26}},		        /* L2 D-Cache misses */
		{  0,{10,-1}},		        /* L2 I-Cache misses */
		{  0,{-1,-1}},			/* L3 D-Cache misses */
		{  0,{-1,-1}},			/* L3 I-Cache misses */
		{ DERIVED_ADD,{ 9,25}},		/* L1 total */
		{ DERIVED_ADD,{10,26}},		/* L2 total */
		{ -1,{-1,-1}},			/* L3 total */
		{ -1,{-1,-1}},			/* Snoops*/
		{  0,{-1,31}}, 			/* Req. access to shared cache line*/
		{  0,{-1,30}}, 			/* Req. access to clean cache line*/
		{  0,{-1,29}}, 			/* Cache Line Invalidation*/
                {  0,{-1,28}},			/* Cache Line Intervention*/
                { -1,{-1,-1}},			/* 14*/
                { -1,{-1,-1}},			/* 15*/
                { -1,{-1,-1}},			/* Cycles branch units are idle*/
                { -1,{-1,-1}},			/* Cycles integer units are idle*/
                { -1,{-1,-1}},			/* Cycles floating point units are idle*/
                { -1,{-1,-1}},			/* Cycles load/store units are idle*/
		{ -1,{-1,-1}}, 			/* D-TLB misses*/
		{ -1,{-1,-1}},		        /* I-TLB misses*/
                {  0,{-1,23}},			/* Total TLB misses*/
                { -1,{-1,-1}},			/* L1 load misses*/
                { -1,{-1,-1}},			/* L1 store misses*/
                { -1,{-1,-1}},			/* L2 load misses*/
                { -1,{-1,-1}},			/* L2 store misses*/
                { -1,{-1,-1}},			/* 27*/
                { -1,{-1,-1}},			/* 28*/
                { -1,{-1,-1}},			/* 29*/
		{ -1,{-1,-1}},			/* TLB shootdowns*/
                {  0,{ 5,-1}},			/* Failed store conditional*/
                { DERIVED_SUB,{20,5}},		/* Successful store conditional*/
                {  0,{-1,20}},			/* Total store conditional*/
                { -1,{-1,-1}},			/* cycles stalled for memory*/
                { -1,{-1,-1}},			/* cycles stalled for memory read*/
                { -1,{-1,-1}},			/* cycles stalled for memory write*/
                { -1,{-1,-1}},			/* cycles no instructions issued*/
                { -1,{-1,-1}},			/* cycles max instructions issued*/
                { -1,{-1,-1}},			/* cycles no instructions exe*/
		{ -1,{-1,-1}},			/* cycles max instructions exe*/
		{ -1,{-1,-1}},			/* 41*/
		{ -1,{-1,-1}},			/* Uncond. branches executed */
		{ -1,{-1,-1}},			/* Cond. branch inst. exe*/
		{ -1,{-1,-1}},			/* Cond. branch inst. taken*/
		{ -1,{-1,-1}},			/* Cond. branch inst. not taken*/
                {  0,{-1,24}},			/* Cond. branch inst. mispred*/
                { -1,{-1,-1}},			/* Cond. branch inst. correctly pred*/
                { -1,{-1,-1}},			/* FMA*/
                {  0,{ 1,-1}},			/* Total inst. issued*/
		{  0,{15,17}},			/* Total inst. executed*/
		{ -1,{-1,-1}},		        /* Integer inst. executed*/
		{  0,{-1,21}},			/* Floating Pt. inst. executed*/
		{  0,{-1,18}},			/* Loads executed*/
		{  0,{-1,19}},			/* Stores executed*/
		{  0,{ 6,-1}},			/* Branch inst. executed*/
		{ -1,{-1,-1}},			/* Vector/SIMD inst. executed */
		{ DERIVED_PS,{0,21}},		/* FLOPS */
                { -1,{-1,-1}},			/* Any stalls */
                { -1,{-1,-1}},			/* FP units are stalled */
		{  0,{ 0,16}},			/* Total cycles */
		{ DERIVED_PS,{0,15}},		/* IPS */
                { -1,{-1,-1}},			/* Total load/store inst. exec */
                { -1,{-1,-1}}			/* Sync exec. */
             };

static hwd_search_t findem_r12k[PAPI_MAX_PRESET_EVENTS] = {
                {  0,{-1,25}},			/* L1 D-Cache misses */
                {  0,{ 9,-1}},		        /* L1 I-Cache misses */
		{  0,{-1,26}},		        /* L2 D-Cache misses */
		{  0,{10,-1}},		        /* L2 I-Cache misses */
		{  0,{-1,-1}},			/* L3 D-Cache misses */
		{  0,{-1,-1}},			/* L3 I-Cache misses */
		{ DERIVED_ADD,{ 9,25}},		/* L1 total */
		{ DERIVED_ADD,{10,26}},		/* L2 total */
		{ -1,{-1,-1}},			/* L3 total */
		{ -1,{-1,-1}},			/* Snoops*/
		{  0,{-1,31}}, 			/* Req. access to shared cache line*/
		{  0,{-1,30}}, 			/* Req. access to clean cache line*/
		{  0,{-1,29}}, 			/* Cache Line Invalidation*/
                {  0,{-1,28}},			/* Cache Line Intervention*/
                { -1,{-1,-1}},			/* 14*/
                { -1,{-1,-1}},			/* 15*/
                { -1,{-1,-1}},			/* Cycles branch units are idle*/
                { -1,{-1,-1}},			/* Cycles integer units are idle*/
                { -1,{-1,-1}},			/* Cycles floating point units are idle*/
                { -1,{-1,-1}},			/* Cycles load/store units are idle*/
		{ -1,{-1,-1}}, 			/* D-TLB misses*/
		{ -1,{-1,-1}},		        /* I-TLB misses*/
                {  0,{-1,23}},			/* Total TLB misses*/
                { -1,{-1,-1}},			/* L1 load misses*/
                { -1,{-1,-1}},			/* L1 store misses*/
                { -1,{-1,-1}},			/* L2 load misses*/
                { -1,{-1,-1}},			/* L2 store misses*/
                { -1,{-1,-1}},			/* BTAC miss */
                {  0,{17,-1}},			/* Prefetch miss */
                { -1,{-1,-1}},			/* 29*/
		{ -1,{-1,-1}},			/* TLB shootdowns*/
                {  0,{ 5,-1}},			/* Failed store conditional*/
                { DERIVED_SUB,{20,5}},		/* Successful store conditional*/
                {  0,{-1,20}},			/* Total store conditional*/
                { DERIVED_SUB,{0,14}},		/* cycles stalled for memory*/
                { -1,{-1,-1}},			/* cycles stalled for memory read*/
                { -1,{-1,-1}},			/* cycles stalled for memory write*/
                { -1,{-1,-1}},			/* cycles no instructions issued*/
                { -1,{-1,-1}},			/* cycles max instructions issued*/
                { -1,{-1,-1}},			/* cycles no instructions exe*/
		{ -1,{-1,-1}},			/* cycles max instructions exe*/
		{ -1,{-1,-1}},			/* hardware interrupts*/
		{ -1,{-1,-1}},			/* Uncond. branches executed */
		{ -1,{-1,-1}},			/* Cond. branch inst. exe*/
		{ -1,{-1,-1}},			/* Cond. branch inst. taken*/
		{ -1,{-1,-1}},			/* Cond. branch inst. not taken*/
                {  0,{-1,24}},			/* Cond. branch inst. mispred*/
                { -1,{-1,-1}},			/* Cond. branch inst. correctly pred*/
                { -1,{-1,-1}},			/* FMA*/
                {  0,{ 1,-1}},			/* Total inst. issued*/
		{  0,{15,-1}},			/* Total inst. executed*/
		{ -1,{-1,-1}},		        /* Integer inst. executed*/
		{  0,{-1,21}},			/* Floating Pt. inst. executed*/
		{  0,{-1,18}},			/* Loads executed*/
		{  0,{-1,19}},			/* Stores executed*/
		{  0,{ 6,-1}},			/* Branch inst. executed*/
		{ -1,{-1,-1}},			/* Vector/SIMD inst. executed */
		{ DERIVED_PS,{0,21}},		/* FLOPS */
                { -1,{-1,-1}},			/* Any res stalled */
                { -1,{-1,-1}},			/* FP units are stalled */
		{  0,{0,-1}},			/* Total cycles */
		{ DERIVED_PS,{0,15}},		/* IPS */
                { DERIVED_ADD,{18,19}},		/* Total load/store inst. exec */
                { -1,{-1,-1}},			/* Sync exec. */
		/* L1 data cache hits */
		{ -1,{-1,-1}},
		/* L2 data cache hits */
		{ -1,{-1,-1}},
		/* L1 data cache accesses */
		{ -1,{-1,-1}},
		/* L2 data cache accesses */
		{ -1,{-1,-1}},
		/* L3 data cache accesses */
		{ -1,{-1,-1}},
		/* L1 data cache reads */
		{ -1,{-1,-1}},
		/* L2 data cache reads */
		{ -1,{-1,-1}},
		/* L3 data cache reads */
		{ -1,{-1,-1}},
		/* L1 data cache writes */
		{ -1,{-1,-1}},
		/* L2 data cache writes */
		{ -1,{-1,-1}},
		/* L3 data cache writes */
		{ -1,{-1,-1}},
		/* L1 instruction cache hits */
		{ -1,{-1,-1}},
		/* L2 instruction cache hits */
		{ -1,{-1,-1}},
		/* L3 instruction cache hits */
		{ -1,{-1,-1}},
		/* L1 instruction cache accesses */
		{ -1,{-1,-1}},
		/* L2 instruction cache accesses */
		{ -1,{-1,-1}},
		/* L3 instruction cache accesses */
		{ -1,{-1,-1}},
		/* L1 instruction cache reads */
		{ -1,{-1,-1}},
		/* L2 instruction cache reads */
		{ -1,{-1,-1}},
		/* L3 instruction cache reads */
		{ -1,{-1,-1}},
		/* L1 instruction cache writes */
		{ -1,{-1,-1}},
		/* L2 instruction cache writes */
		{ -1,{-1,-1}},
		/* L3 instruction cache writes */
		{ -1,{-1,-1}},
		/* L1 total cache hits */
		{ -1,{-1,-1}},
		/* L2 total cache hits */
		{ -1,{-1,-1}},
		/* L3 total cache hits */
		{ -1,{-1,-1}},
		/* L1 total cache accesses */
		{ -1,{-1,-1}},
		/* L2 total cache accesses */
		{ -1,{-1,-1}},
		/* L3 total cache accesses */
		{ -1,{-1,-1}},
		/* L1 total cache reads */
		{ -1,{-1,-1}},
		/* L2 total cache reads */
		{ -1,{-1,-1}},
		/* L3 total cache reads */
		{ -1,{-1,-1}},
		/* L1 total cache writes */
		{ -1,{-1,-1}},
		/* L2 total cache writes */
		{ -1,{-1,-1}},
		/* L3 total cache writes */
		{ -1,{-1,-1}},
		/* FP mult */
		{ -1,{-1,-1}},
		/* FP add */
		{ -1,{-1,-1}},
		/* FP Div */
		{ -1,{-1,-1}},
		/* FP Sqrt */
		{ -1,{-1,-1}},
		/* FP inv */
		{ -1,{-1,-1}},
};

/* Low level functions, should not handle errors, just return codes. */

/* Utility functions */

static int scan_cpu_info(inventory_t *item, void *foo)
{
  papi_rev_id_t *ri;

  if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUBOARD)) 
    _papi_system_info.hw_info.mhz = (int)item->inv_controller;

  if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUCHIP))
    {
      DBG((stderr,"scan_system_info(%p,%p) %d, %d, %d\n",
	   item,foo,item->inv_controller,item->inv_state,item->inv_unit));
      ri = (papi_rev_id_t *)&item->inv_state;
      _papi_system_info.hw_info.revision = (float)ri->ri_majrev + 
	((float)ri->ri_minrev*0.1);
      switch (ri->ri_imp)
	 {
	 case C0_IMP_R10000:
	   _papi_system_info.hw_info.model = CPU_IP27;	      
	   strcpy(_papi_system_info.hw_info.model_string,"R10000");
	   _papi_system_info.num_gp_cntrs = 2;
	   break;
	 case C0_IMP_R12000:
	   _papi_system_info.hw_info.model = CPU_IP30;	      
	   strcpy(_papi_system_info.hw_info.model_string,"R12000");
	   _papi_system_info.num_gp_cntrs = 4;
	   break;
	 default:
	   return(PAPI_ESBSTR);
	 }
    }	  
  return(0);
}

/* Utility functions */

static int setup_all_presets(PAPI_hw_info_t *info)
{
  int pnum, hwnum, event, derived, first;
  char str[PAPI_MAX_STR_LEN];
  hwd_search_t *findem;  

  if (info->model == CPU_IP27)
    findem = findem_r10k;
  else if (info->model == CPU_IP30)
    findem = findem_r12k;
  else
    return(PAPI_ESBSTR);

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      first = -1;
      derived = findem[pnum].derived;
      if (derived == -1)
	continue;
      for (hwnum = 0; hwnum < 2; hwnum++)
	{
	  event = findem[pnum].findme[hwnum];
	  if (event == -1)
	    continue;
	  assert(event < HWPERF_EVENTMAX);
	  if (first == -1)
	    first = event;

	  preset_map[pnum].selector |= 1 << event;
	  if (event > HWPERF_MAXEVENT)
	    {
	      preset_map[pnum].counter_cmd[event] = event - HWPERF_CNT1BASE;
	      preset_map[pnum].num_on_counter[1]++;
	    }
	  else
	    {
	      preset_map[pnum].counter_cmd[event] = event;
	      preset_map[pnum].num_on_counter[0]++;   
	    }
	  sprintf(str,"%d",event);
	  if (strlen(preset_map[pnum].note))
	    strcat(preset_map[pnum].note,",");
	  strcat(preset_map[pnum].note,str);
	  
	  DBG((stderr,"setup_all_presets(%p): Preset %d, event %d found, selector 0x%x, num %d,%d\n",
	       info,pnum,event,preset_map[pnum].selector,
	       preset_map[pnum].num_on_counter[0],   	
	       preset_map[pnum].num_on_counter[1]));
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

/* static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  assert(arg2 >= HWPERF_MINEVENT && arg2 <= HWPERF_MAXEVENT);
  assert(arg1 >= 0 && arg1 <= HWPERF_EVENTMAX);
  ptr->counter_cmd.hwp_evctrargs.hwp_evctrl[arg1].hwperf_creg.hwp_ev = arg2;
}

static void unset_config(hwd_control_state_t *ptr, int arg1)
{
  assert(arg1 >= 0 && arg1 <= HWPERF_EVENTMAX);
  ptr->counter_cmd.hwp_evctrargs.hwp_evctrl[arg1].hwperf_spec = 0;
} */

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int arg1)
{
  assert(arg1 >= 0 && arg1 <= HWPERF_EVENTMAX);
  if (a->counter_cmd.hwp_evctrargs.hwp_evctrl[arg1].hwperf_creg.hwp_ev == 
      b->counter_cmd.hwp_evctrargs.hwp_evctrl[arg1].hwperf_creg.hwp_ev)
    return(1);

  return(0);
}

static int update_global_hwcounters(EventSetInfo *global)
{
  int i, retval, selector;
  hwperf_cntr_t readem;
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  hwperf_profevctrarg_t *arg = &machdep->counter_cmd;

  retval = ioctl(machdep->fd, PIOCGETEVCTRS, (void *)&readem);
  if (retval <= 0)
    return(PAPI_ESYS);

  DBG((stderr,"update_global_hwcounters() num on counters: %d %d\n",
       machdep->num_on_counter[0],machdep->num_on_counter[1]));
  selector = machdep->selector;
  while (i = ffs(selector))
    {
      i = i - 1;
      selector ^= 1 << i;
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld * %d\n",i,
	   global->hw_start[i]+readem.hwp_evctr[i],
	   global->hw_start[i],readem.hwp_evctr[i],
	   machdep->num_on_counter[(i < HWPERF_CNT1BASE ? 0 : 1)]));
      global->hw_start[i] = global->hw_start[i] + readem.hwp_evctr[i] * 
	machdep->num_on_counter[(i < HWPERF_CNT1BASE ? 0 : 1)];
    }

  retval = ioctl(machdep->fd, PIOCSETEVCTRL, arg);
  if (retval <= 0)
    return(PAPI_ESYS);
   
  return(PAPI_OK);
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
	{
	  arg->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode &= ~HWPERF_MODEMASK;
	  arg->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode = mode;
	}
    }
  return(0);
}

static int set_granularity(hwd_control_state_t *this_state, int domain)
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

static int set_inherit(EventSetInfo *zero, pid_t pid)
{
  int retval;

  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  if ((pid == PAPI_INHERIT_ALL) || (pid == PAPI_INHERIT_NONE))
    return(PAPI_ESBSTR);

  retval = ioctl(current_state->fd,PIOCSAVECCNTRS,pid);
  if (retval == -1)
    return(PAPI_ESYS);

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

/* static void init_config(hwd_control_state_t *ptr)
{
  int i;

  for (i=0;i<HWPERF_EVENTMAX;i++)
    {
      ptr->counter_cmd.hwp_evctrargs.hwp_evctrl[i].hwperf_spec = 0;
      ptr->counter_cmd.hwp_ovflw_freq[i] = 0;
    }
  ptr->counter_cmd.hwp_ovflw_sig = 0;
} */

static int get_system_info(void)
{
  int fd, retval;
  pid_t pid;
  char pidstr[PAPI_MAX_STR_LEN];
  prpsinfo_t psi;

  if (scaninvent(scan_cpu_info, NULL) == -1)
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

  if (getcwd(_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == NULL)
    return(PAPI_ESYS);

  _papi_system_info.cpunum = psi.pr_sonproc;
  strcat(_papi_system_info.exe_info.fullname,"/");
  strcat(_papi_system_info.exe_info.fullname,psi.pr_fname);
  strncpy(_papi_system_info.exe_info.name,psi.pr_fname,PAPI_MAX_STR_LEN);

  /* HWinfo */

  _papi_system_info.hw_info.totalcpus = sysmp(MP_NPROCS);
  if (_papi_system_info.hw_info.totalcpus > 1)
    {
      _papi_system_info.hw_info.ncpu = 2;
      _papi_system_info.hw_info.nnodes = _papi_system_info.hw_info.totalcpus /
	_papi_system_info.hw_info.ncpu;
    }
  else
    {
      _papi_system_info.hw_info.ncpu = 0;
      _papi_system_info.hw_info.nnodes = 0;
    }

  _papi_system_info.hw_info.vendor = -1;
  strcpy(_papi_system_info.hw_info.vendor_string,"MIPS");

  /* Generic info */

  _papi_system_info.num_cntrs = HWPERF_EVENTMAX;
  _papi_system_info.cpunum = get_cpu();

  retval = setup_all_presets(&_papi_system_info.hw_info);
  if (retval)
    return(retval);

  return(PAPI_OK);
} 

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

long long _papi_hwd_get_real_usec (void)
{
  timespec_t t;
  long long retval;

  if (clock_gettime(CLOCK_SGI_CYCLE, &t) == -1)
    return(PAPI_ESYS);

  retval = (t.tv_sec * 1000000) + t.tv_nsec / 1000;
  return(retval);
}

long long _papi_hwd_get_real_cycles (void)
{
  long long retval;

  retval = _papi_hwd_get_real_usec() * (long long)_papi_system_info.hw_info.mhz;
  return(retval);
}

long long _papi_hwd_get_virt_usec (void)
{
  return(-1);
}

long long _papi_hwd_get_virt_cycles (void)
{
  return(-1);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error: %s",strerror(error));
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

int _papi_hwd_init(EventSetInfo *global)
{
  char pidstr[PAPI_MAX_STR_LEN];
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  hwperf_profevctrarg_t args;
  int fd, gen;

  memset(&args,0x0,sizeof(args));

  sprintf(pidstr,"/proc/%05d",(int)getpid());
  if ((fd = open(pidstr,O_RDONLY)) == -1)
    return(PAPI_ESYS);

  if ((gen = ioctl(fd, PIOCENEVCTRS, (void *)&args)) == -1)
    {
      close(fd);
      return(PAPI_ESYS);
    }

  if (gen <= 0)
    {
      close(fd);
      return(PAPI_EMISC);
    }

  machdep->fd = fd;
  machdep->generation = gen;
  /* machdep->default_mode = translate_domain(_papi_system_info.default_domain); */

  return(PAPI_OK);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (HWPERF_EVENTMAX-1);
  
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
  int tmp = 0, i = HWPERF_EVENTMAX - 1;
  
  while (i)
    {
      tmp = (1 << i) & cntr_avail_bits;
      if (tmp)
	return(i);
      i--;
    }
  return(0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, hwperf_eventctrl_t *to)
{
  int index, mode = 0;
  
  if (_papi_system_info.default_domain & PAPI_DOM_USER) 
    mode |= HWPERF_CNTEN_U;
  if (_papi_system_info.default_domain & PAPI_DOM_KERNEL)
    mode |= HWPERF_CNTEN_K;
  if (_papi_system_info.default_domain & PAPI_DOM_OTHER)
    mode |= HWPERF_CNTEN_E | HWPERF_CNTEN_S; 
  assert(mode);
  
  while (index = ffs(selector))
    {
      index = index - 1;
      DBG((stderr,"set_hwcntr_codes(%x,%p,%p) index = %d, from %d, mode 0x%x\n",selector,from,to,index,from[index],mode));
      selector ^= 1 << index;
      assert(from[index] < HWPERF_MAXEVENT);
      to->hwp_evctrl[index].hwperf_creg.hwp_ev = from[index];
      to->hwp_evctrl[index].hwperf_creg.hwp_mode = mode;
    }
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  unsigned int hwcntr, selector = 0;
  unsigned int avail = 0;
  unsigned char tmp_cmd[HWPERF_EVENTMAX];
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

	  /* If it's available on both */

	  if (preset_map[preset_index].num_on_counter[0] &&
	      preset_map[preset_index].num_on_counter[1])
	    {
	      /* Pick the one most available */

	      if (this_state->num_on_counter[0] < 
		  this_state->num_on_counter[1])
		avail = avail & 0xffff;
	      else
		avail = avail & 0xffff0000;
	    }
	  
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

      hwcntr_num = EventCode & 0xff;  /* 0 through 31 */ 
      if (hwcntr_num > _papi_system_info.num_cntrs)
	return(PAPI_EINVAL);

      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->selector & selector)
	return(PAPI_ECNFLCT);	    

      /* Set up the native encoding */

      if (hwcntr_num < HWPERF_CNT1BASE)
	tmp_cmd[hwcntr_num] = hwcntr_num;
      else
	tmp_cmd[hwcntr_num] = hwcntr_num - HWPERF_CNT1BASE;

      codes = tmp_cmd;
    }

  /* Turn on the bits for this counter */

  set_hwcntr_codes(selector,codes,&this_state->counter_cmd.hwp_evctrargs);

  /* Update the new counter select field. */

  this_state->selector |= selector;

  /* Inform the upper level that the software event 'index' 
     consists of the following information. */

  ESI->EventInfoArray[index].code = EventCode;
  ESI->EventInfoArray[index].selector = selector;

  /* Update the counts on the 'physical' registers. */

  while ((hwcntr = ffs(selector)))
    {
      hwcntr = hwcntr - 1;
      if (hwcntr < HWPERF_CNT1BASE)
	this_state->num_on_counter[0]++;
      else
	this_state->num_on_counter[1]++;
      selector ^= 1 << hwcntr;
    }

  /* Don't use selector after here */

  return(PAPI_OK);
}

int _papi_hwd_rem_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector, hwcntr, used, preset_index;

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
      int hwcntr_num;
      
      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  /* 0 through 31 */ 
      if (hwcntr_num > _papi_system_info.num_cntrs)
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

  /* Update the counts on the 'physical' registers. */

  while ((hwcntr = ffs(selector)))
    {
      hwcntr = hwcntr - 1;
      if (hwcntr < HWPERF_CNT1BASE)
	this_state->num_on_counter[0]++;
      else
	this_state->num_on_counter[1]++;
      selector ^= 1 << hwcntr;
    }

  /* Don't use selector after here */

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, int index, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

void dump_cmd(hwperf_profevctrarg_t *t)
{
  int i;

  fprintf(stderr,"Command block at %p: Signal %d\n",t,t->hwp_ovflw_sig);
  for (i=0;i<HWPERF_EVENTMAX;i++)
    {
      if (t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode)
	fprintf(stderr,"Event %d: hwp_ev %d hwp_ie %d hwp_mode %d hwp_ovflw_freq %d\n",
		i,
		t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ev,
		t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_ie,
		t->hwp_evctrargs.hwp_evctrl[i].hwperf_creg.hwp_mode,
		t->hwp_ovflw_freq[i]);
    }
}

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  hwd_control_state_t previous_state;

  /* If we are nested, merge the global counter structure
     with the current eventset */

  /* Save the context in case of error */

  memcpy(&previous_state,current_state,sizeof(hwd_control_state_t));

  if (current_state->selector)
    {
      int hwcntrs_in_both, hwcntrs_to_add, hwcntr;

      /* Stop the current context */

      /* Update the global values */

      retval = update_global_hwcounters(zero);
      if (retval)
	return(retval);

      /* Delete the current context */

      /* Check for events that are shared between eventsets and 
	 therefore require no modification to the control state. */

      hwcntrs_in_both = this_state->selector & current_state->selector;
      if (hwcntrs_in_both)
	{
	  while (hwcntr = ffs(hwcntrs_in_both))
	    {
	      hwcntr = hwcntr - 1;
	      hwcntrs_in_both ^= 1 << hwcntr;
	      if (counter_shared(this_state, current_state, hwcntr))
		zero->multistart.SharedDepth[hwcntr]++;
	      else
		return(PAPI_ECNFLCT);
	      ESI->hw_start[hwcntr] = zero->hw_start[hwcntr];
	    }
	}

      /* Merge the unshared configuration registers. */
	  
      hwcntrs_to_add = this_state->selector ^ (this_state->selector & current_state->selector);
      while (hwcntr = ffs(hwcntrs_to_add))
	{	
	  hwcntr = hwcntr - 1;
	  hwcntrs_to_add ^= 1 << hwcntr;
	  current_state->selector |= 1 << hwcntr;
	  current_state->counter_cmd.hwp_evctrargs.hwp_evctrl[hwcntr].hwperf_spec = 
	    this_state->counter_cmd.hwp_evctrargs.hwp_evctrl[hwcntr].hwperf_spec;
	  ESI->hw_start[i] = 0;
	  if (hwcntr >= HWPERF_CNT1BASE)
	    current_state->num_on_counter[1]++;
	  else
	    current_state->num_on_counter[0]++;
	}
    }
  else
    {
      /* If we are NOT nested, just copy the global counter 
	 structure to the current eventset */

      current_state->selector = this_state->selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(hwperf_profevctrarg_t));
      memcpy(&current_state->num_on_counter,&this_state->num_on_counter,2*sizeof(int));
    }

  /* Set up the new merged control structure */
  
#ifdef DEBUG
  dump_cmd(&current_state->counter_cmd);
#endif
      
  retval = ioctl(current_state->fd,PIOCSETEVCTRL,&current_state->counter_cmd);
  if (retval <= 0) 
    {
      memcpy(current_state,&previous_state,sizeof(hwd_control_state_t));
      if (retval < 0)
	return(PAPI_ESYS);
      else
	return(PAPI_EMISC);
    }	

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
  return(-1);
}

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long *events)
{
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  long long correct[HWPERF_EVENTMAX];

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

int _papi_hwd_write(EventSetInfo *ESI, long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(EventSetInfo *zero)
{
  hwd_control_state_t *machdep = (hwd_control_state_t *)zero->machdep;

  close(machdep->fd);
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
  hwperf_profevctrarg_t *arg = &this_state->counter_cmd;
  int selector, hwcntr;

  if (overflow_option->threshold == 0)
    {
      arg->hwp_ovflw_sig = 0;
      selector = ESI->EventInfoArray[overflow_option->EventIndex].selector;
      while (hwcntr = ffs(selector))
	{
	  hwcntr = hwcntr - 1;
	  arg->hwp_evctrargs.hwp_evctrl[hwcntr].hwperf_creg.hwp_ie = 0;
	  arg->hwp_ovflw_freq[hwcntr] = 0;
	  selector ^= 1 << hwcntr;
	}
      /* this_state->timer_ms = 0;
      overflow_option->timer_ms = 0; */
    }
  else
    {
      arg->hwp_ovflw_sig = PAPI_SIGNAL;
      selector = ESI->EventInfoArray[overflow_option->EventIndex].selector;
      while (hwcntr = ffs(selector))
	{
	  hwcntr = hwcntr - 1;
	  arg->hwp_evctrargs.hwp_evctrl[hwcntr].hwperf_creg.hwp_ie = 1;
	  arg->hwp_ovflw_freq[hwcntr] = (int)overflow_option->threshold;
	  selector ^= 1 << hwcntr;
	}
      /* this_state->timer_ms = 1; 
      overflow_option->timer_ms = 1; */
    }

  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
  return(-1);
}

void *_papi_hwd_get_overflow_address(void *context)
{
  struct sigcontext *info = (struct sigcontext *)context;

  return((void *)info->sc_pc);
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
				 (caddr_t)&_ftext,
				 (caddr_t)&_etext,
				 (caddr_t)&_fdata,
				 (caddr_t)&_edata,
				 (caddr_t)&_fbss,
				 (caddr_t)&_end,
			        "_RLD_LIST", /* environment variable */
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
			        1,  /* supports HW overflow overflow emulation */
			        0,  /* supports HW profile emulation */
			        1,  /* supports 64 bit virtual counters */
			        1,  /* supports child inheritance option */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        0,  /* We can use the virt_usec call */
			        0,  /* We can use the virt_cyc call */
			        0,  /* HW read resets the counters */
			        sizeof(hwd_control_state_t), 
			        NULL };

