
#include "solaris-ultra.h"

const static int pcr_shift[2] = { CPC_ULTRA_PCR_PIC0_SHIFT, CPC_ULTRA_PCR_PIC1_SHIFT };

const static uint64_t pcr_event_mask[2] = { (CPC_ULTRA2_PCR_PIC0_MASK<<CPC_ULTRA_PCR_PIC0_SHIFT), 
					    (CPC_ULTRA2_PCR_PIC1_MASK<<CPC_ULTRA_PCR_PIC1_SHIFT) };

const static uint64_t pcr_inv_mask[2] = { ~(CPC_ULTRA2_PCR_PIC0_MASK<<CPC_ULTRA_PCR_PIC0_SHIFT), ~(CPC_ULTRA2_PCR_PIC1_MASK<<CPC_ULTRA_PCR_PIC1_SHIFT) };

static int cpuver = -1;

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = {
  /* L1 Cache Dmisses */
  {0,0,0,{0,0},""},		
  /* L1 Cache Imisses */
  {0x3,DERIVED_SUB,0,{0x8,0x8},""},		
  /* L2 Cache Dmisses*/
  {0,0,0,{0,0},""}, 			
  /* L2 Cache Imisses*/
  {0,0,0,{0,0},""}, 			
  /* L3 Cache Dmisses*/
  {0,0,0,{0,0},""}, 			
  /* L3 Cache Imisses*/
  {0,0,0,{0,0},""}, 			
  /* L1 Total Cache misses */
  {0,0,0,{0,0},""}, 			
  /* L2 Total Cache misses*/
  {0x3,DERIVED_SUB,0,{0xc,0xc},""},			
  /* L3 Total Cache misses*/
  {0,0,0,{0,0},""}, 			
  /* Req. for snoop*/
  {0x2,0,0,{0,0xe},""},	
  /* Req. shared cache line*/
  {0,0,0,{0,0},""},		 	
  /* Req. clean cache line*/
  {0,0,0,{0,0},""},		 	
  /* Req. invalidate cache line*/
  {0x1,0,0,{0xe,0},""},		 	
  /* Req. intervention cache line*/
  {0,0,0,{0,0},""},		
  /* L3 Load misses*/
  {0,0,0,{0,0},""},			
  /* L3 Store misses*/
  {0,0,0,{0,0},""},			
  /* BRU idle cycles*/
  {0,0,0,{0,0},""},		
  /* FXU idle cycles*/
  {0,0,0,{0,0},""},		
  /* FPU idle cycles*/
  {0,0,0,{0,0},""},          
  /* LSU idle cycles*/
  {0,0,0,{0,0},""},          
  /* D-TLB misses*/
  {0,0,0,{0,0},""},		
  /* I-TLB misses*/
  {0,0,0,{0,0},""},		
  /* Total TLB misses*/
  {0,0,0,{0,0},""},		
  /* L1LM */
  {0x3,DERIVED_SUB,0,{0x9,0x9},""},			
  /* L1SM */
  {0x3,DERIVED_SUB,0,{0xa,0xa},""},			
  /* L2LM*/
  {0,0,0,{0,0},""},			
  /* L2SM*/
  {0,0,0,{0,0},""},			
  /* BTAC */
  {0,0,0,{0,0},""},			
  /* PRF_DM */
  {0,0,0,{0,0},""},			
  /* L3DCH*/
  {0,0,0,{0,0},""},			
  /* TLB shootdowns*/
  {0,0,0,{0,0},""},			
  /* Suc. store conditional instructions*/
  {0,0,0,{0,0},""},	
  /* Failed store conditional instructions*/
  {0,0,0,{0,0},""},	
  /* Total store conditional instructions*/
  {0,0,0,{0,0},""},			
  /* Cycles stalled waiting for memory */
  {0,0,0,{0,0},""},			
  /* Cycles stalled waiting for memory read */
  {0,0,0,{0,0},""},   	
  /* Cycles stalled waiting for memory write */
  {0,0,0,{0,0},""},
  /* Cycles no instructions issued */
  {0,0,0,{0,0},""},	
  /* Cycles max instructions issued */
  {0,0,0,{0,0},""},			
  /* Cycles no instructions completed */
  {0,0,0,{0,0},""},	
  /* Cycles max instructions completed */
  {0,0,0,{0,0},""},		
  /* Hardware interrupts */
  {0,0,0,{0,0},""},		
  /* Uncond. branches executed*/
  {0,0,0,{0,0},""},			
  /* Cond. Branch inst. executed*/
  {0,0,0,{0,0},""},		
  /* Cond. Branch inst. taken*/
  {0,0,0,{0,0},""},			
  /* Cond. Branch inst. not taken*/
  {0,0,0,{0,0},""},			
  /* Cond. branch inst. mispred.*/
  {0x2,0,0,{0,0x2},""},          
  /* Cond. branch inst. pred. */
  {0,0,0,{0,0},""},		
  /* FMA's */
  {0,0,0,{0,0},""},		
  /* Total inst. issued*/
  {0x3,0,0,{0x1,0x1},""},		
  /* Total inst. executed*/
  {0x3,0,0,{0x1,0x1},""},		
  /* Integer inst. executed*/
  {0,0,0,{0,0},""},		
  /* Floating Pt. inst. executed*/
  {0x2,0,0,{0,0x3},""},	        
  /* Loads executed*/
  {0x1,0,0,{0x9,0},""},		
  /* Stores executed*/
  {0x1,0,0,{0xa,0},""},		
  /* Branch inst. executed*/
  {0,0,0,{0,0},""},	
  /* Vector/SIMD inst. executed */
  {0,0,0,{0,0},""},			
  /* FLOPS */
  {0x3,DERIVED_PS,0,{0,0x3},""},
  /* Any stalls */
  {0,0,0,{0,0},""},			
  /* Cycles FP units are stalled*/
  {0,0,0,{0,0},""},			
  /* Total cycles */
  {0x3,0,0,{0,0},""},		
  /* IPS */
  {0x3,DERIVED_PS,0,{0,0},""},			
  /* load/store*/
  {0,0,0,{0,0},""},		
  /* Synchronization inst. executed*/
  {0,0,0,{0,0},""},		
  /* L1 data cache hits */
  {0,0,0,{0,0},""},		
  /* L2 data cache hits */
  {0,0,0,{0,0},""},		
  /* L1 data cache accesses */
  {0,0,0,{0,0},""},		
  /* L2 data cache accesses */
  {0,0,0,{0,0},""},		
  /* L3 data cache accesses */
  {0,0,0,{0,0},""},		
  /* L1 data cache reads */
  {0x1,0,0,{0x9,0},""},		
  /* L2 data cache reads */
  {0,0,0,{0,0},""},		
  /* L3 data cache reads */
  {0,0,0,{0,0},""},		
  /* L1 data cache writes */
  {0x1,0,0,{0xa,0},""},		
  /* L2 data cache writes */
  {0,0,0,{0,0},""},		
  /* L3 data cache writes */
  {0,0,0,{0,0},""},
  /* L1 instruction cache hits */
  {0x2,0,0,{0,0x8},""},
  /* L2 instruction cache hits */
  {0x2,0,0,{0,0xf},""},		
  /* L3 instruction cache hits */
  {0,0,0,{0,0},""},		
  /* L1 instruction cache accesses */
  {0x1,0,0,{0x8,0},""},		
  /* L2 instruction cache accesses */
  {0,0,0,{0,0},""},		
  /* L3 instruction cache accesses */
  {0,0,0,{0,0},""},		
  /* L1 instruction cache reads */
  {0,0,0,{0,0},""},		
  /* L2 instruction cache reads */
  {0,0,0,{0,0},""},		
  /* L3 instruction cache reads */
  {0,0,0,{0,0},""},		
  /* L1 instruction cache writes */
  {0,0,0,{0,0},""},		
  /* L2 instruction cache writes */
  {0,0,0,{0,0},""},		
  /* L3 instruction cache writes */
  {0,0,0,{0,0},""},
  /* L1 total cache hits */
  {0,0,0,{0,0},""},
  /* L2 total cache hits */
  {0x2,0,0,{0,0xc},""},		
  /* L3 total cache hits */
  {0,0,0,{0,0},""},		
  /* L1 total cache accesses */
  {0,0,0,{0,0},""},		
  /* L2 total cache accesses */
  {0x1,0,0,{0xc,0},""},		
  /* L3 total cache accesses */
  {0,0,0,{0,0},""},		
  /* L1 total cache reads */
  {0,0,0,{0,0},""},		
  /* L2 total cache reads */
  {0,0,0,{0,0},""},		
  /* L3 total cache reads */
  {0,0,0,{0,0},""},		
  /* L1 total cache writes */
  {0,0,0,{0,0},""},		
  /* L2 total cache writes */
  {0,0,0,{0,0},""},		
  /* L3 total cache writes */
  {0,0,0,{0,0},""},
  /* FP mult */
  {0,0,0,{0,0},""},
  /* FP add */
  {0,0,0,{0,0},""},
  /* FP Div */
  {0,0,0,{0,0},""},
  /* FP Sqrt */
  {0,0,0,{0,0},""},
  /* FP inv */
  {0,0,0,{0,0},""},
};

static void merge_flags(int from, int *to)
{
  *to = from | *to;
}

static void merge_pcr(int hwcntr_num, uint64_t from, uint64_t *to)
{
  uint64_t mask;
  uint64_t tmp;

  if (hwcntr_num == 0)
    mask = CPC_ULTRA2_PCR_PIC0_MASK;
  else if (hwcntr_num == 1)
    mask = CPC_ULTRA2_PCR_PIC1_MASK;
  else
    abort();

  tmp = *to; /* copy it */
  tmp = tmp & ~(mask); /* turn bits off */
  *to = tmp | (mask & from); /* copy bits in */
}

static void dump_cmd(papi_cpc_event_t *t)
{
  fprintf(stderr,"cpc_event_t.ce_cpuver %d\n",t->cmd.ce_cpuver);
  fprintf(stderr,"ce_tick %llu\n",t->cmd.ce_tick);
  fprintf(stderr,"ce_pic[0] %llu ce_pic[1] %llu\n",t->cmd.ce_pic[0],t->cmd.ce_pic[1]);
  fprintf(stderr,"ce_pcr 0x%llx\n",t->cmd.ce_pcr);
  fprintf(stderr,"flags %x\n",t->flags);
}

#if 0
static void dispatch_emt(int signal, siginfo_t *sip, void *arg)
{
#ifdef DEBUG
  psignal(signal, "dispatch_emt");
  psiginfo(sip, "dispatch_emt");
#endif

  if (sip->si_code == EMT_CPCOVF)
    {
      papi_cpc_event_t *sample;
      EventSetInfo *ESI;
      int t;
      INIT_MAP_VOID;
      sample = &((hwd_control_state_t *)master_event_set->machdep)->counter_cmd;
      ESI = event_set_overflowing;

      /* GROSS! Hack to push the correct values back into the hardware */
      
      /* Find which HW counter is overflowing */
      
      if (ESI->EventInfoArray[ESI->overflow.EventIndex].selector = 0x1)
	t = 0;
      else
	t = 1;
      
      /* Push the correct value */
      
      sample->cmd.ce_pic[t] = ESI->overflow.threshold;
      dump_cmd(sample);
      if (cpc_bind_event(&sample->cmd,0) == -1)
	return;
      
      /* Call the regular overflow function in extras.c */

      _papi_hwi_dispatch_overflow_signal(ESI,master_event_set,arg);
      
      /* Reset the threshold */
      
      sample->cmd.ce_pic[t] = UINT64_MAX - ESI->overflow.threshold;
      
      dump_cmd(sample);
      if (cpc_bind_event(&sample->cmd,sample->flags) == -1)
	return;
    }
  else
    {
      DBG((stderr,"dispatch_emt() dropped, si_code = %d\n",sip->si_code));
      return;
    }
}
#endif

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
  
  /* if everything went ok, return mhz */
  if(strcmp(cmd, "clock-frequency:"))
    return -1;
  else
    return mhz / 1000000;

  /* End stolen code */
}

static int setup_all_presets(PAPI_hw_info_t *info)
{
  int pnum, s;

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      if ((s = preset_map[pnum].selector))
	{
	  if (preset_map[pnum].derived == 0)
	    {
	      if (s == 0x1)
		sprintf(preset_map[pnum].note,"0x%x,-1",preset_map[pnum].counter_cmd[0]);
	      else
		sprintf(preset_map[pnum].note,"-1,0x%x",preset_map[pnum].counter_cmd[1]);
	    }
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

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  if (1 & cntr_avail_bits)
    return(1);
  else if (2 & cntr_avail_bits)
    return(2);
  else
  return(0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, uint64_t *to)
{
  uint64_t tmp = *to;
  if (selector & 0x1)
    {
      tmp = tmp & pcr_inv_mask[0];
      tmp = tmp | ((uint64_t)from[0] << pcr_shift[0]);
    }
  if (selector & 0x2)
    {
      tmp = tmp & pcr_inv_mask[1];
      tmp = tmp | ((uint64_t)from[1] << pcr_shift[1]);
    }
  *to = tmp;
}

static void init_config(hwd_control_state_t *ptr)
{
  ptr->counter_cmd.flags = 0x0;
  ptr->counter_cmd.cmd.ce_cpuver = cpuver;
  ptr->counter_cmd.cmd.ce_pcr = 0x0;
  set_domain(ptr,_papi_system_info.default_domain);
  set_granularity(ptr,_papi_system_info.default_granularity);
}

/* Utility functions */

static int get_system_info(void)
{
  int retval;
  pid_t pid;
  char maxargs[PAPI_MAX_STR_LEN] = "<none>";
  psinfo_t psi;
  int fd;

  /* Check counter access */

  if (cpc_version(CPC_VER_CURRENT) != CPC_VER_CURRENT)
    return(PAPI_ESBSTR);

  if (cpc_access() == -1)
    return(PAPI_ESBSTR);

  cpuver = cpc_getcpuver();
  if (cpuver == -1)
    return(PAPI_ESBSTR);

  /* Path and args */

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);

  /* Turn on microstate accounting for this process and any LWPs. */
  
  sprintf(maxargs,"/proc/%d/ctl",(int) pid);
  if ((fd = open(maxargs,O_WRONLY)) == -1)
    return(PAPI_ESYS);
  {
    int retval;
    struct { long cmd; long flags; } cmd;
    cmd.cmd = PCSET;
    cmd.flags = PR_MSACCT | PR_MSFORK;
    retval = write(fd,&cmd,sizeof(cmd));
    close(fd);
    DBG((stderr,"Write PCSET returned %d\n",retval));
    if (retval != sizeof(cmd))
      return(PAPI_ESYS);
  }

  /* Get executable info */

  sprintf(maxargs,"/proc/%d/psinfo",(int)pid);
  if ((fd = open(maxargs,O_RDONLY)) == -1)
    return(PAPI_ESYS);
  read(fd,&psi,sizeof(psi));
  close(fd);
  {
    char *tmp;

    tmp = strchr(psi.pr_psargs,' ');
    if (tmp != NULL)
      *tmp = '\0';
  }
  strncpy(_papi_system_info.exe_info.fullname,psi.pr_psargs,PAPI_MAX_STR_LEN);
  strncpy(_papi_system_info.exe_info.name,psi.pr_fname,PAPI_MAX_STR_LEN);
  DBG((stderr,"Executable is %s\n",_papi_system_info.exe_info.name));
  DBG((stderr,"Full Executable is %s\n",_papi_system_info.exe_info.fullname));

  /* Hardware info */

  _papi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
  _papi_system_info.hw_info.vendor = -1;

  _papi_system_info.hw_info.model = cpuver;
  strcpy(_papi_system_info.hw_info.model_string,"UltraSPARC");
  sprintf(maxargs," %d",cpuver - 999);
  strcat(_papi_system_info.hw_info.model_string,maxargs);
    
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
  _papi_system_info.num_cntrs = retval;

  /* Software info */

  /* Setup presets */

  retval = setup_all_presets(&_papi_system_info.hw_info);
  if (retval)
    return(retval);

  return(PAPI_OK);
} 

static int counter_event_shared(const papi_cpc_event_t *a, const papi_cpc_event_t *b, int cntr)
{
  uint64_t t;

  if (a->flags == b->flags)
    {
      t = pcr_event_mask[cntr];
      if ((a->cmd.ce_pcr & t) == (b->cmd.ce_pcr & t))
	return(1);
    }

  return(0);
}

static int counter_event_compat(const papi_cpc_event_t *a, const papi_cpc_event_t *b, int cntr)
{
  uint64_t pcr_priv_mask = 0x6;

  if (a->flags == b->flags)
    {
      if ((a->cmd.ce_pcr & pcr_priv_mask) == (b->cmd.ce_pcr & pcr_priv_mask))
	return(1);
    }

  return(0);
}

static void counter_event_copy(const papi_cpc_event_t *a, papi_cpc_event_t *b, int cntr)
{
  uint64_t t1, t2, t3;
  uint64_t pcr_priv_mask = 0x6;

  t1 = a->cmd.ce_pcr & pcr_event_mask[cntr];  /* Bits to be turned on */
  t2 = b->cmd.ce_pcr & pcr_event_mask[cntr];  /* Bits to be turned on */
  t3 = b->cmd.ce_pcr & pcr_priv_mask;         /* Bits to be turned on */
  b->cmd.ce_pcr = t1 | t2 | t3;
}

static int update_global_hwcounters(EventSetInfo *global)
{
  int i, retval;
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  papi_cpc_event_t *command= &machdep->counter_cmd;
  cpc_event_t *readem = &command->cmd;

  retval = cpc_take_sample(readem);
  if (retval == -1)
    return(PAPI_ESYS);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      if (machdep->selector & (1 << i))
	{
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+readem->ce_pic[i],global->hw_start[i],readem->ce_pic[i]));
      global->hw_start[i] = global->hw_start[i] + readem->ce_pic[i];
	}
      else
	{
	  DBG((stderr,"update_global_hwcounters() %d: G%lld\n",i,
	       global->hw_start[i]));
	}
      readem->ce_pic[i] = 0;
    }

#ifdef DEBUG
  dump_cmd(command);
#endif

  retval = cpc_bind_event(readem,command->flags);
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
  papi_cpc_event_t *command= &this_state->counter_cmd;
  cpc_event_t *event = &command->cmd;
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

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static int set_inherit(EventSetInfo *global, int arg)
{
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  papi_cpc_event_t *command= &machdep->counter_cmd;

  if (arg == 0)
    {
      if (command->flags & CPC_BIND_LWP_INHERIT)
	command->flags = command->flags ^ CPC_BIND_LWP_INHERIT;
    }
  else if (arg == 1)
    {
      command->flags = command->flags | CPC_BIND_LWP_INHERIT;
    }
  else
    return(PAPI_EINVAL);

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

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

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
  return((long long)gethrtime()/(long long)1000);
}

long long _papi_hwd_get_real_cycles (void)
{
  return(get_tick());
}

long long _papi_hwd_get_virt_usec (void)
{
  return((long long)gethrvtime()/(long long)1000);
}

long long _papi_hwd_get_virt_cycles (void)
{
  return(-1);
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
      uint64_t code;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  
      if (hwcntr_num > _papi_system_info.num_gp_cntrs) /* 0 or 1 */
	return(PAPI_EINVAL);

      tmp_cmd[hwcntr_num] = EventCode >> 8; /* 0 through 0xf */
      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->selector & selector)
	return(PAPI_ECNFLCT);	    

      codes = tmp_cmd;
    }

  /* Lower two bits tell us what counters we need */

  assert((this_state->selector | 0x3) == 0x3);

  /* Perform any initialization of the control bits */

  if (this_state->selector == 0)
    init_config(this_state);
  
  /* Turn on the bits for this counter */

  set_hwcntr_codes(selector,codes,&this_state->counter_cmd.cmd.ce_pcr);

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
      int hwcntr_num, code, old_code;
      
      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0x3;  
      if (hwcntr_num > _papi_system_info.num_gp_cntrs) /* counter 0 or 1 */ 
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
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(papi_cpc_event_t));
    }

  /* If we ARE nested, 
     carefully merge the global counter structure with the current eventset */
  else   
    {
      int tmp, hwcntrs_in_both, hwcntrs_in_all, hwcntr;

      /* Stop the current context */

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
      while (i = ffs(tmp))
	{
	  hwcntr = 1 << (i-1);
	  tmp = tmp ^ hwcntr;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      if (!counter_event_shared(&this_state->counter_cmd, &current_state->counter_cmd, i-1))
		return(PAPI_ECNFLCT);
	    }
	  else if (!counter_event_compat(&this_state->counter_cmd, &current_state->counter_cmd, i-1))
	    return(PAPI_ECNFLCT);
	}

      /* Now everything is good, so actually do the merge */

      tmp = hwcntrs_in_all;
      while (i = ffs(tmp))
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
	      counter_event_copy(&this_state->counter_cmd, &current_state->counter_cmd, i-1);
	      ESI->hw_start[i-1] = 0;
	      zero->hw_start[i-1] = 0;
	    }
	}
    }

  /* (Re)start the counters */  

#ifdef DEBUG
  dump_cmd(&current_state->counter_cmd);
#endif
      
  retval = cpc_bind_event(&current_state->counter_cmd.cmd, 
			  current_state->counter_cmd.flags);
  if (retval == -1) 
    return(PAPI_ESYS);

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
  float tmp;

  tmp = (float)units * _papi_system_info.hw_info.mhz * 1000000.0;
  tmp = tmp / (float) cycles;
  return((long long)tmp);
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

int _papi_hwd_write(EventSetInfo *master, EventSetInfo *ESI, long long events[])
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
#if 0
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  papi_cpc_event_t *arg = &this_state->counter_cmd;
  int selector, hwcntr;

  if (overflow_option->threshold == 0)
    {
      arg->flags ^= CPC_BIND_EMT_OVF;
      if (sigaction(SIGEMT, NULL, NULL) == -1)
	return(PAPI_ESYS);
    }
  else
    {
      struct sigaction act;

      act.sa_sigaction = dispatch_emt;
      memset(&act.sa_mask,0x0,sizeof(act.sa_mask));
      act.sa_flags = SA_RESTART|SA_SIGINFO;
      if (sigaction(SIGEMT, &act, NULL) == -1)
	return(PAPI_ESYS);

      arg->flags |= CPC_BIND_EMT_OVF;
      selector = ESI->EventInfoArray[overflow_option->EventIndex].selector;
      if (selector == 0x1)
	arg->cmd.ce_pic[0] = UINT64_MAX	- (uint64_t)overflow_option->threshold;
      else if (selector == 0x2)
	arg->cmd.ce_pic[1] = UINT64_MAX	- (uint64_t)overflow_option->threshold;
    }

  return(PAPI_OK);
#else
  /* This function is not used and shouldn't be called. */

  abort();
#endif
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  ucontext_t *info = (ucontext_t *)context;
  location = (void *)info->uc_mcontext.gregs[REG_PC];

  return(location);
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
				 (caddr_t)&_start,
				 (caddr_t)&_etext,
				 (caddr_t)&_etext+1,
				 (caddr_t)&_edata,
				 (caddr_t)&_edata+1,
				 (caddr_t)&_end,
				 "LD_PRELOAD",
			       },
			       -1,  /* num_cntrs */
			       -1,  /* num_gp_cntrs */
			       -1,  /* grouped_counters */
			       -1,  /* num_sp_cntrs */
			       -1,  /* total_presets */
			       -1,  /* total_events */
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
			        1,  /* We can use the virt_usec call */
			        0,  /* We can use the virt_cyc call */
			        0,  /* HW Read also resets the counters */
			        sizeof(hwd_control_state_t), 
			        NULL };

