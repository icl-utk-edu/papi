#if defined(sun) && defined(__SVR4) && defined(sparc)

#include "solaris-ultra.h"

const static int pcr_shift[2] = { CPC_ULTRA_PCR_PIC0_SHIFT, CPC_ULTRA_PCR_PIC1_SHIFT };

static int cpuver = -1;

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = {
  /* L1 Cache Dmisses */
  {0,0,0,{0,0,0},""},		
  /* L1 Cache Imisses */
  {0x3,DERIVED_SUB,0,{0x8,0x8,0},"IC_ref,IC_hit"},		
  /* L2 Cache Dmisses*/
  {0,0,0,{0,0,0},""}, 			
  /* L2 Cache Imisses*/
  {0x3,DERIVED_SUB,0,{0xc,0xc,0},""},			
  /* L3 Cache Dmisses*/
  {0,0,0,{0,0,0},""}, 			
  /* L3 Cache Imisses*/
  {0,0,0,{0,0,0},""}, 			
  /* L1 Total Cache misses */
  {0,0,0,{0,0,0},""}, 			
  /* L2 Total Cache misses*/
  {0x2,0,0,{0,0xd,0},""},
  /* L3 Total Cache misses*/
  {0,0,0,{0,0,0},""}, 			
  /* Req. for snoop*/
  {0x2,0,0,{0,0xe,0},""},	
  /* Req. shared cache line*/
  {0,0,0,{0,0,0},""},		 	
  /* Req. clean cache line*/
  {0,0,0,{0,0,0},""},		 	
  /* Req. invalidate cache line*/
  {0x1,0,0,{0xe,0,0},""},		 	
  /* Req. intervention cache line*/
  {0,0,0,{0,0,0},""},		
  /* L3 Load misses*/
  {0,0,0,{0,0,0},""},			
  /* L3 Store misses*/
  {0,0,0,{0,0,0},""},			
  /* BRU idle cycles*/
  {0,0,0,{0,0,0},""},		
  /* FXU idle cycles*/
  {0,0,0,{0,0,0},""},		
  /* FPU idle cycles*/
  {0,0,0,{0,0,0},""},          
  /* LSU idle cycles*/
  {0,0,0,{0,0,0},""},          
  /* D-TLB misses*/
  {0,0,0,{0,0,0},""},		
  /* I-TLB misses*/
  {0,0,0,{0,0,0},""},		
  /* Total TLB misses*/
  {0,0,0,{0,0,0},""},		
  /* L1LM */
  {0x3,DERIVED_SUB,0,{0x9,0x9,0},""},			
  /* L1SM */
  {0x3,DERIVED_SUB,0,{0xa,0xa,0},""},			
  /* L2LM*/
  {0,0,0,{0,0,0},""},			
  /* L2SM*/
  {0,0,0,{0,0,0},""},			
  /* L1DCH */
  {0,0,0,{0,0,0},""},			
  /* L2DCH */
  {0,0,0,{0,0,0},""},			
  /* L3DCH*/
  {0,0,0,{0,0,0},""},			
  /* TLB shootdowns*/
  {0,0,0,{0,0,0},""},			
  /* Suc. store conditional instructions*/
  {0,0,0,{0,0,0},""},	
  /* Failed store conditional instructions*/
  {0,0,0,{0,0,0},""},	
  /* Total store conditional instructions*/
  {0,0,0,{0,0,0},""},			
  /* Cycles stalled waiting for memory */
  {0,0,0,{0,0,0},""},			
  /* Cycles stalled waiting for memory read */
  {0,0,0,{0,0,0},""},   	
  /* Cycles stalled waiting for memory write */
  {0,0,0,{0,0,0},""},
  /* Cycles no instructions issued */
  {0,0,0,{0,0,0},""},	
  /* Cycles max instructions issued */
  {0,0,0,{0,0,0},""},			
  /* Cycles no instructions completed */
  {0,0,0,{0,0,0},""},	
  /* Cycles max instructions completed */
  {0,0,0,{0,0,0},""},		
  /* Hardware interrupts */
  {0,0,0,{0,0,0},""},		
  /* Uncond. branches executed*/
  {0,0,0,{0,0,0},""},			
  /* Cond. Branch inst. executed*/
  {0,0,0,{0,0,0},""},		
  /* Cond. Branch inst. taken*/
  {0,0,0,{0,0,0},""},			
  /* Cond. Branch inst. not taken*/
  {0,0,0,{0,0,0},""},			
  /* Cond. branch inst. mispred.*/
  {0x2,0,0,{0,0x2,0},"Dispatch0_mispred"},          
  /* Cond. branch inst. pred. */
  {0,0,0,{0,0,0},""},		
  /* FMA's */
  {0,0,0,{0,0,0},""},		
  /* Total inst. issued*/
  {0x3,0,0,{0x1,0x1,0},"Instr_cnt"},		
  /* Total inst. executed*/
  {0x3,0,0,{0x1,0x1,0},"Instr_cnt"},		
  /* Integer inst. executed*/
  {0,0,0,{0,0,0},""},		
  /* Floating Pt. inst. executed*/
  {0x2,0,0,{0,0x3,0},"Dispatch0_FPU_use"},	        
  /* Loads executed*/
  {0x1,0,0,{0x9,0,0},"DC_rd"},		
  /* Stores executed*/
  {0x1,0,0,{0xa,0,0},"DC_wr"},		
  /* Branch inst. executed*/
  {0,0,0,{0,0,0},""},	
  /* Vector/SIMD inst. executed */
  {0,0,0,{0,0,0},""},			
  /* FLOPS */
  {0x3,DERIVED_PS,1,{0,0x3,0},"Cycle_cnt,Dispatch_FP_use"},			
  /* Any stalls */
  {0,0,0,{0,0,0},""},			
  /* Cycles FP units are stalled*/
  {0,0,0,{0,0,0},""},			
  /* Total cycles */
  {0x3,0,0,{0,0,0},"Cycle_cnt"},		
  /* IPS */
  {0x3,DERIVED_PS,1,{0,0,0},"Cycle_cnt,Instr_cnt"},			
  /* load/store*/
  {0,0,0,{0,0,0},""},		
  /* Synchronization inst. executed*/
  {0,0,0,{0,0,0},""}		
};

/* Utility functions */

/* Find all the hwcntrs that name lives on */

static int getmhz(void)
{
  /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
  /* See the PCL home page for the German version of PAPI. */

  int mhz;
  char line[256], cmd[80];
  FILE *f;
  char cmd_line[80], fname[L_tmpnam];
  
  /*??? system call takes very long */
  /* get system configuration and put output into file */
  sprintf(cmd_line, "/usr/sbin/prtconf -vp >%s", tmpnam(fname));
  if(system(cmd_line) == -1)
    {
      remove(fname);
      return -1;
    }
  
  /* open output file */
  if((f = fopen(fname, "r")) == NULL)
    {
      remove(fname);
      return -1;
    }
  
  /* ignore all lines until we reach something with a sparc line */
  while(fgets(line, 256, f) != NULL)
    {
      if((sscanf(line, "%s", cmd) == 1)
	 && !strcmp(cmd, "sparc-version:"))
	break;
    }
  
  /* then read until we find clock frequency */
  while(fgets(line, 256, f) != NULL)
    {
      if((sscanf(line, "%s %x", cmd, &mhz) == 2)
	 && !strcmp(cmd, "clock-frequency:"))
	break;
    }
  
  /* remove temporary file */
  remove(fname);
  
  /* if everything wqent ok, return mhz */
  if(strcmp(cmd, "clock-frequency:"))
    return -1;
  else
    return mhz / 1000000;

  /* End stolen code */
}

static int get_cpu_num(void)
{
  int cpu;

  processor_bind(P_LWPID, P_MYID, PBIND_QUERY, &cpu);
  return cpu;
}

static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  uint64_t old;

  old = PCR_CODE_INV_MASK(arg1);
  old = old & ptr->counter_cmd.ce_pcr;
  ptr->counter_cmd.ce_pcr = old | (arg2 << pcr_shift[arg1]);
}

static void unset_config(hwd_control_state_t *ptr, int arg1)
{
  uint64_t old;

  old = PCR_CODE_INV_MASK(arg1);
  ptr->counter_cmd.ce_pcr = old & ptr->counter_cmd.ce_pcr;
}

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  uint64_t t;

  t = PCR_CODE_MASK(cntr);
  
  if ((a->counter_cmd.ce_pcr & t) == (b->counter_cmd.ce_pcr & t))
    return(1);

  return(0);
}

static int update_global_hwcounters(EventSetInfo *global)
{
  int i, retval;
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  cpc_event_t *readem = &machdep->counter_cmd;

  retval = cpc_take_sample(readem);
  if (retval == -1)
    return(PAPI_ESYS);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+readem->ce_pic[i],global->hw_start[i],readem->ce_pic[i]));
      global->hw_start[i] = global->hw_start[i] + readem->ce_pic[i];
      readem->ce_pic[i] = 0;
    }

  retval = cpc_bind_event(readem,machdep->flags);
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
  cpc_event_t *event = &(this_state->counter_cmd);
  uint64_t pcr = event->ce_pcr;

  pcr = pcr | 0x7;
  pcr = pcr ^ 0x7;
  if (domain & PAPI_DOM_USER)
    pcr = pcr | 1 << CPC_ULTRA_PCR_USR;
  if (domain & PAPI_DOM_KERNEL)
    pcr = pcr | 1 << CPC_ULTRA_PCR_SYS;

  event->ce_pcr = pcr;

  return(PAPI_OK);
}

static int set_granularity(hwd_control_state_t *this_state, int domain)
{
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

static int set_inherit(EventSetInfo *global, int arg)
{
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;

  if (arg == 0)
    {
      if (machdep->flags & CPC_BIND_LWP_INHERIT)
	machdep->flags = machdep->flags ^ CPC_BIND_LWP_INHERIT;
    }
  else if (arg == 1)
    {
      machdep->flags = machdep->flags | CPC_BIND_LWP_INHERIT;
    }
  else
    return(PAPI_EINVAL);

  return(PAPI_OK);
}

static void init_config(hwd_control_state_t *ptr)
{
  ptr->flags = 0x0;
  ptr->counter_cmd.ce_cpuver = cpuver;
  ptr->counter_cmd.ce_pcr = 0x0;
  set_domain(ptr,_papi_system_info.default_domain);
  set_granularity(ptr,_papi_system_info.default_granularity);
}

static int get_system_info(void)
{
  int retval;
  pid_t pid;
  char maxargs[PAPI_MAX_STR_LEN] = "<none>";
  psinfo_t psi;
  int fd;

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);

  sprintf(maxargs,"/proc/%d/psinfo",(int)pid);
  assert((fd = open(maxargs,O_RDONLY)) != -1);
  read(fd,&psi,sizeof(psi));
  close(fd);
  strcpy(maxargs,psi.pr_fname);

  if (cpc_version(CPC_VER_CURRENT) != CPC_VER_CURRENT)
    return(PAPI_ESBSTR);

  if (cpc_access() == -1)
    return(PAPI_ESBSTR);

  cpuver = cpc_getcpuver();
  if (cpuver == -1)
    return(PAPI_ESBSTR);

  /* Path and args */

  if (getcwd(_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == NULL)
    return(PAPI_ESYS);
  strcat(_papi_system_info.exe_info.fullname,"/");
  strcat(_papi_system_info.exe_info.fullname,maxargs);
  strncpy(_papi_system_info.exe_info.name,basename(maxargs),PAPI_MAX_STR_LEN);

  /* Hardware info */

  retval = get_cpu_num();
  if (retval < 1)
    return(PAPI_ESBSTR);

  _papi_system_info.cpunum = retval;
  _papi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
  _papi_system_info.hw_info.model = cpuver;

  strcpy(_papi_system_info.hw_info.model_string,"UltraSPARC");
  sprintf(maxargs," %d",cpuver - 1000);
  strcat(_papi_system_info.hw_info.model_string,maxargs);
    
  _papi_system_info.hw_info.vendor = -1;
  strcpy(_papi_system_info.hw_info.vendor_string,"SUN");
  _papi_system_info.hw_info.revision = 0;

  retval = getmhz();
  if (retval == -1)
    return(PAPI_ESBSTR);
  _papi_system_info.hw_info.mhz = (float)retval;

  retval = cpc_getnpic(cpuver);
  if (retval < 1)
    return(PAPI_ESBSTR);
  _papi_system_info.num_gp_cntrs = retval;
  _papi_system_info.num_cntrs = retval + 1;

  /* Software info */

  /* Setup presets */

  return(PAPI_OK);
} 

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

long long _papi_hwd_get_real_usec (void)
{
  long long retval;
  struct timeval tp;

  gettimeofday(&tp,NULL);
  retval = tp.tv_sec * 1000000 + tp.tv_usec;
  return(retval);
}

long long _papi_hwd_get_real_cycles (void)
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

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
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

static int get_avail_hwcntr_num(int cntr_avail_bits)
{
  int tmp = 0, i = MAX_COUNTERS - 1;
  
  while (i)
    {
      tmp = (1 << i) & cntr_avail_bits;
      if (tmp)
	return(i);
      i--;
    }
  return(0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, uint64_t *to)
{
  int useme, i;
  
  for (i=0;i<_papi_system_info.num_gp_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  to[i] = from[i];
	}
    }
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector = 0;
  int avail = 0;
  unsigned char tmp_cmd[MAX_COUNTERS];
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
      uint64_t valid,code;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0x7;  /* 0, 1 or 2. 2 is virtualized tick register */ 
      if (hwcntr_num > _papi_system_info.num_cntrs)
	return(PAPI_EINVAL);

      code = EventCode >> 8; /* 0 through 0xf */
      if (hwcntr_num == 0)
	{
	  if (cpuver < CPC_ULTRA3)
	    valid = CPC_ULTRA2_PCR_PIC0_MASK;
	  else
	    valid = CPC_ULTRA3_PCR_PIC0_MASK;
	  if ((code | valid) != valid)
	    return(PAPI_EINVAL); 
	}
      else if (hwcntr_num == 1)
	{
	  if (cpuver < CPC_ULTRA3)
	    valid = CPC_ULTRA2_PCR_PIC1_MASK;
	  else
	    valid = CPC_ULTRA3_PCR_PIC1_MASK;
	  if ((code | valid) != valid)
	    return(PAPI_EINVAL); 
	}
      else if (hwcntr_num == 2)
	{
	  if (code != 1)
	    return(PAPI_EINVAL); 
	}

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

  set_hwcntr_codes(selector,codes,&this_state->counter_cmd.ce_pcr);

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
      int hwcntr_num;
      uint64_t code, valid;
      
      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0x7;  /* 0, 1 or 2. 2 is virtualized tick register */ 
      if (hwcntr_num > _papi_system_info.num_cntrs)
	return(PAPI_EINVAL);

      code = EventCode >> 8; /* 0 through 50 */
      if (hwcntr_num == 0)
	{
	  if (cpuver < CPC_ULTRA3)
	    valid = CPC_ULTRA2_PCR_PIC0_MASK;
	  else
	    valid = CPC_ULTRA3_PCR_PIC0_MASK;
	  if ((code | valid) != valid)
	    return(PAPI_EINVAL); 
	}
      else if (hwcntr_num == 1)
	{
	  if (cpuver < CPC_ULTRA3)
	    valid = CPC_ULTRA2_PCR_PIC1_MASK;
	  else
	    valid = CPC_ULTRA3_PCR_PIC1_MASK;
	  if ((code | valid) != valid)
	    return(PAPI_EINVAL); 
	}
      else if (hwcntr_num == 2)
	{
	  if (code != 1)
	    return(PAPI_EINVAL); 
	}

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

void dump_cmd(cpc_event_t *t, int flags)
{
  fprintf(stderr,"cpc_event_t.ce_cpuver %d\n",t->ce_cpuver);
  fprintf(stderr,"ce_tick %llu\n",t->ce_tick);
  fprintf(stderr,"ce_pic[0] %llu ce_pic[1] %llu\n",t->ce_pic[0],t->ce_pic[1]);
  fprintf(stderr,"ce_pcr 0x%llx\n",t->ce_pcr);
  fprintf(stderr,"flags %x\n",flags);
}

static void merge_flags(int from, int *to)
{
  *to = from | *to;
}

static void merge_pcr(int hwcntr_num, uint64_t from, uint64_t *to)
{
  uint64_t mask;
  uint64_t tmp;

  if (hwcntr_num == 0)
    {
      if (cpuver < CPC_ULTRA3)
	mask = CPC_ULTRA2_PCR_PIC0_MASK;
      else
	mask = CPC_ULTRA3_PCR_PIC0_MASK;
    }
  else if (hwcntr_num == 1)
    {
      if (cpuver < CPC_ULTRA3)
	mask = CPC_ULTRA2_PCR_PIC1_MASK;
      else
	mask = CPC_ULTRA3_PCR_PIC1_MASK;
    }
  else
    return;

  tmp = *to; /* copy it */
  tmp = tmp & ~(mask); /* turn bits off */
  *to = tmp | (mask & from); /* copy bits in */
}

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  
  /* If we are nested, merge the global counter structure with the current eventset */

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

      for (i = 0; i < _papi_system_info.num_gp_cntrs; i++)
	{
	  /* Check for events that are shared between eventsets and 
	     therefore require no modification to the control state. */

	  hwcntr = 1 << i;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      if (!counter_shared(this_state, current_state, i))
		return(PAPI_ECNFLCT);
	      ESI->hw_start[i] = zero->hw_start[i];
	      zero->multistart.SharedDepth[i]++;
	    }

	  /* Merge the unshared configuration registers. */
	  
	  else if (this_state->selector & hwcntr)
	    {
	      current_state->selector |= hwcntr;
	      merge_pcr(i, this_state->counter_cmd.ce_pcr, &current_state->counter_cmd.ce_pcr);
	      merge_flags(this_state->flags,&current_state->flags);
	      ESI->hw_start[i] = 0;
	    }
	}
    }
  else
    {
      /* If we are NOT nested, just copy the global counter structure to the current eventset */

      current_state->selector = this_state->selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(cpc_event_t));
      /* current_state->flags = this_state->flags; */
    }

  /* Set up the new merged control structure */

  /* (Re)start the counters */  

#ifdef DEBUG
  dump_cmd(&current_state->counter_cmd, current_state->flags);
#endif
      
  retval = cpc_bind_event(&current_state->counter_cmd, current_state->flags);
  if (retval == -1) 
    return(PAPI_ESYS);

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

  /* If we're not the outermost EventSet, then we need to start again 
     because someone is still running. 

  if ((zero->multistart.num_runners - 1) == 0)
    {
      retval = cpc_bind_event(NULL, 0);
      if (retval == -1) 
	return(PAPI_ESYS);
    } */

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
      selector ^= (1 << (pos-1));
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
      selector ^= (1 << (pos-1));
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
      return(set_inherit(zero,option->inherit.inherit));
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
  (void)cpc_rele();
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
  /* This function is not used and shouldn't be called. */

  abort();
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
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 ""
			       },
			       -1,  /*  num_cntrs */
			       -1,  /*  num_gp_cntrs */
			       -1,  /*  grouped_counters */
			       -1,  /*  num_sp_cntrs */
			       -1,  /*  total_presets */
			       -1,  /*  total_events */
			        0,  /*  needs overflow emulation */
			        1,  /*  needs profile emulation */
			        0,  /*  needs 64 bit virtual counters */
			        1,  /*  supports child inheritance option */
			        0,  /*  can attach to another process */
			        0,  /*  read resets the counters */
			        PAPI_DOM_USER, /* default domain */
			        PAPI_GRN_THR,  /* default granularity */
			        sizeof(hwd_control_state_t), 
			        NULL };
#else
#error "Mismatch of substrate and architecture"
#endif

