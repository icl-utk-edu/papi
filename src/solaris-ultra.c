/* 
* File:    solaris-ultra.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
*/  

#include SUBSTRATE

/* Globals used to access the counter registers. */

static int cpuver;
static int pcr_shift[2]; 
static uint64_t pcr_event_mask[2];
static uint64_t pcr_inv_mask[2];

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS];

static hwd_search_t usii_preset_search_map[] = {
  /* L1 Cache Imisses */
  {PAPI_L1_ICM,DERIVED_SUB,{0x8,0x8}},		
  /* L2 Total Cache misses*/
  {PAPI_L2_TCM,DERIVED_SUB,{0xc,0xc}},			
  /* Req. for snoop*/
  {PAPI_CA_SNP,0,{-1,0xe}},	
  /* Req. invalidate cache line*/
  {PAPI_CA_INV,0,{0xe,-1}},		 	
  /* L1LM */
  {PAPI_L1_LDM,DERIVED_SUB,{0x9,0x9}},			
  /* L1SM */
  {PAPI_L1_STM,DERIVED_SUB,{0xa,0xa}},			
  /* Cond. branch inst. mispred.*/
  {PAPI_BR_MSP,0,{-1,0x2}},          
  /* Total inst. issued*/
  {PAPI_TOT_IIS,0,{-1,0x1}},	/*	can be counted on both counters
  {PAPI_TOT_IIS,0,{0x1,0x1}},		but causes conflicts with TOT_CYC */	
  /* Total inst. executed*/
  {PAPI_TOT_INS,0,{-1,0x1}},	/*	can be counted on both counters
  {PAPI_TOT_INS,0,{0x1,0x1}},		but causes conflicts with TOT_CYC */	
  /* Loads executed*/
  {PAPI_LD_INS,0,{0x9,-1}},		
  /* Stores executed*/
  {PAPI_SR_INS,0,{0xa,-1}},		
  /* Total cycles */
  {PAPI_TOT_CYC,0,{0,0}},		
  /* IPS */
  {PAPI_IPS,DERIVED_PS,{0,0x1}},			
  /* L1 data cache reads */
  {PAPI_L1_DCR,0,{0x9,-1}},		
  /* L1 data cache writes */
  {PAPI_L1_DCW,0,{0xa,-1}},		
  /* L1 instruction cache hits */
  {PAPI_L1_ICH,0,{-1,0x8}},
  /* L2 instruction cache hits */
  {PAPI_L2_ICH,0,{-1,0xf}},		
  /* L1 instruction cache accesses */
  {PAPI_L1_ICA,0,{0x8,-1}},		
  /* L2 total cache hits */
  {PAPI_L2_TCH,0,{-1,0xc}},		
  /* L2 total cache accesses */
  {PAPI_L2_TCA,0,{0xc,-1}},
  /* Terminator */
  {0,0,{0,0}}};

static hwd_search_t usiii_preset_search_map[] = {
  /* Floating point instructions */
  {PAPI_FP_INS,DERIVED_ADD,{0x18,0x27}}, /* pic0 FA_pipe_completion and pic1 FM_pipe_completion */
  /* Floating point add instructions */
  {PAPI_FAD_INS,0,{0x18,-1}},          /* pic0 FA_pipe_completion */
  /* Floating point multiply instructions */
  {PAPI_FML_INS,0,{-1,0x27}},          /* pic1 FM_pipe_completion */
  /* ITLB */
  {PAPI_TLB_IM,0,{-1,0x11}},           /* pic1 ITLB_miss */
  /* DITLB */
  {PAPI_TLB_DM,0,{-1,0x12}},           /* pic1 DTLB_miss */
  /* Total cycles */
  {PAPI_TOT_CYC,0,{0,0}},              /* pic0 and pic1 Cycle_cnt */				
  /* Total inst. issued*/
  {PAPI_TOT_IIS,0,{0x1,0x1}},          /* pic0 and pic1 Instr_cnt */				
  /* Total inst. executed*/
  {PAPI_TOT_INS,0,{0x1,0x1}},          /* pic0 and pic1 Instr_cnt */		
  /* L2 Total Cache misses*/
  {PAPI_L2_TCM,0,{-1,0xc}},            /* pic1 EC_misses */			
  /* L2 Total ICache misses*/
  {PAPI_L2_ICM,0,{-1,0xf}},            /* pic1 EC_ic_miss */			
  /* L1 Total ICache misses */
  {PAPI_L1_ICM,0,{-1,0x8}},            /* pic1 IC_miss (actually hits) */      		
  /* L1 Load Misses */
  {PAPI_L1_LDM,0,{-1,0x9}},            /* pic1 DC_rd_miss */			
  /* L1 Store Misses */
  {PAPI_L1_STM,0,{-1,0xa}},            /* pic1 DC_wr_miss */			
  /* Cond. branch inst. mispred.*/
  {PAPI_BR_MSP,0,{-1,0x2}},            /* pic1 Dispatch0_mispred */
  /* IPS */
  {PAPI_IPS,DERIVED_PS,{0x0,0x1}},   /* pic0 Cycle_cnt, pic1 Instr_cnt */		
  /* L1 data cache reads */
  {PAPI_L1_DCR,0,{0x9,-1}},	       /* pic0 DC_rd */	
  /* L1 data cache writes */
  {PAPI_L1_DCW,0,{0xa,-1}},	       /* pic0 DC_wr */	
  /* L1 instruction cache hits */
  {PAPI_L1_ICH,0,{0x8,-1}},            /* pic0 IC_ref (actually hits only)
  /* L1 instruction cache accesses */
  {PAPI_L1_ICA,DERIVED_ADD,{0x8,0x8}}, /* pic0 IC_ref (actually hits only) + pic1 IC_miss */
  /* L2 total cache hits */
  {PAPI_L2_TCH,DERIVED_SUB,{0xc,0xc}}, /* pic0 EC_ref - pic1 EC_misses */
  /* L2 total cache accesses */
  {PAPI_L2_TCA,0,{0xc,-1}},            /* pic0 EC_ref */
  /* Terminator */
  {0,0,{0,0}}};

#ifdef DEBUG
static void dump_cmd(papi_cpc_event_t *t)
{
  DBG((stderr,"cpc_event_t.ce_cpuver %d\n",t->cmd.ce_cpuver));
  DBG((stderr,"ce_tick %llu\n",t->cmd.ce_tick));
  DBG((stderr,"ce_pic[0] %llu ce_pic[1] %llu\n",t->cmd.ce_pic[0],t->cmd.ce_pic[1]));
  DBG((stderr,"ce_pcr 0x%llx\n",t->cmd.ce_pcr));
  DBG((stderr,"flags %x\n",t->flags));
}
#endif DEBUG

static void dispatch_emt(int signal, siginfo_t *sip, void *arg)
{
#ifdef DEBUG
  psignal(signal, "dispatch_emt");
  psiginfo(sip, "dispatch_emt");
#endif

  if (sip->si_code == EMT_CPCOVF)
    {
      papi_cpc_event_t *sample;
      EventSetInfo *ESI, *master_event_set;
      int t;

      master_event_set = _papi_hwi_lookup_in_master_list();
      ESI = master_event_set->event_set_overflowing;
      sample = &((hwd_control_state_t *)master_event_set->machdep)->counter_cmd;

      /* GROSS! This is a hack to 'push' the correct values 
	 back into the hardware, such that when PAPI handles
         the overflow and reads the values, it gets the correct
         ones. */
      
      /* Find which HW counter is overflowing */
      
      if (ESI->EventInfoArray[ESI->overflow.EventIndex].selector = 0x1)
	t = 0;
      else
	t = 1;
      
      /* Push the correct value */
      
      sample->cmd.ce_pic[t] = ESI->overflow.threshold;
#if DEBUG
      dump_cmd(sample);
#endif
      if (cpc_bind_event(&sample->cmd,0) == -1)
	return;
      
      /* Call the regular overflow function in extras.c */

      _papi_hwi_dispatch_overflow_signal(arg);
      
      /* Reset the threshold */
      
      sample->cmd.ce_pic[t] = UINT64_MAX - ESI->overflow.threshold;
      
#if DEBUG
      dump_cmd(sample);
#endif
      if (cpc_bind_event(&sample->cmd,sample->flags) == -1)
	return;
    }
  else
    {
      DBG((stderr,"dispatch_emt() dropped, si_code = %d\n",sip->si_code));
      return;
    }
}

static int scan_prtconf(char *cpuname,int len_cpuname,int *hz, int *ver)
{
  /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
  /* See the PCL home page for the German version of PAPI. */
  /* Modified by Nils Smeds, all new bugs are my fault */
  /*    The routine now looks for the first "Node" with the following: */
  /*           "device_type"     = 'cpu'                    */
  /*           "name"            = (Any value)              */
  /*           "sparc-version"   = (Any value)              */
  /*           "clock-frequency" = (Any value)              */
  int ihz, version;
  char line[256], cmd[80], name[256];
  FILE *f;
  char cmd_line[80], fname[L_tmpnam];
  unsigned int matched;
  
  /*??? system call takes very long */
  /* get system configuration and put output into file */
  sprintf(cmd_line, "/usr/sbin/prtconf -vp >%s", tmpnam(fname));
  DBG((stderr,"/usr/sbin/prtconf -vp > %s \n", fname));
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
  
  DBG((stderr,"Parsing %s...\n", fname));
  /* ignore all lines until we reach something with a sparc line */
  matched = 0x0; ihz = -1;
  while(fgets(line, 256, f) != NULL)
    {
      /* DBG((stderr,">>> %s",line)); */
      if((sscanf(line, "%s", cmd) == 1)
	 && strstr(line, "Node 0x")) {
	matched = 0x0;
        /* DBG((stderr,"Found 'Node' -- search reset. (0x%2.2x)\n",matched)); */
        }
      else {
	 if (strstr(cmd, "device_type:") &&
             strstr(line, "'cpu'" )) {
            matched |= 0x1;
            /* DBG((stderr,"Found 'cpu'. (0x%2.2x)\n",matched)); */
            }
	 else if (!strcmp(cmd, "sparc-version:") &&
               (sscanf(line, "%s %x", cmd, &version) == 2) ) {
            matched |= 0x2;
            /* DBG((stderr,"Found version=%d. (0x%2.2x)\n", version, matched)); */
            }
	 else if (!strcmp(cmd, "clock-frequency:") &&
               (sscanf(line, "%s %x", cmd, &ihz) == 2) ) {
            matched |= 0x4;
            /* DBG((stderr,"Found ihz=%d. (0x%2.2x)\n", ihz,matched)); */
            }
	 else if (!strcmp(cmd, "name:") &&
               (sscanf(line, "%s %s", cmd, name) == 2) ) {
            matched |= 0x8;
            /* DBG((stderr,"Found name: %s. (0x%2.2x)\n", name,matched)); */
            }
      }
     if((matched & 0xF) == 0xF ) break;
    }
  DBG((stderr,"Parsing found name=%s, speed=%dHz, version=%d\n", name, ihz,version));
  
  if(matched ^ 0x0F)
    ihz = -1;
  else {
    *hz = (float) ihz;
    *ver = version;
    strncpy(cpuname,name,len_cpuname);
    }

  return ihz;

  /* End stolen code */
}

static int setup_all_presets(PAPI_hw_info_t *info)
{
  int snum, preset_index, selector;
  hwd_search_t *findem;

  if (info->model <= CPC_ULTRA2)
    findem = usii_preset_search_map;
  else if (info->model == CPC_ULTRA3)
    findem = usiii_preset_search_map;
  else
    abort();

  memset(preset_map,0x0,sizeof(preset_map));

  for (snum = 0; snum < PAPI_MAX_PRESET_EVENTS; snum++)
    {
      selector = 0;

      /* Get index */

      preset_index = findem[snum].preset ^ PRESET_MASK; 

      /* Check for premature end of list */

      if (findem[snum].preset == 0)
	break;

      /* Set Derived flag */

      preset_map[preset_index].derived = findem[snum].derived_op;

      /* This might need to be changed */

      preset_map[preset_index].operand_index = 0;

      /* Set Selector bits and control values */

      if (findem[snum].findme[0] != -1)
	{
	  preset_map[preset_index].counter_cmd[0] = findem[snum].findme[0];
	  selector |= 0x1;
	}
      if (findem[snum].findme[1] != -1)
	{
	  preset_map[preset_index].counter_cmd[1] = findem[snum].findme[1];
	  selector |= 0x2;
	}
      preset_map[preset_index].selector = selector;

      /* Put in the note */
     
      if (selector == 0x1)
	sprintf(preset_map[preset_index].note,"0x%x,-1",preset_map[preset_index].counter_cmd[0]);
      else if (selector == 0x2)
	sprintf(preset_map[preset_index].note,"-1,0x%x",preset_map[preset_index].counter_cmd[1]);
      else if (selector == 0x3)
	sprintf(preset_map[preset_index].note,"0x%x,0x%x",
		preset_map[preset_index].counter_cmd[0],preset_map[preset_index].counter_cmd[1]);
      else
	abort();
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

/* This is a wrapper arount fprintf(stderr,...) for cpc_walk_events() */
void print_walk_names(void  *arg,  int regno,  const char *name, uint8_t bits)
  {
    fprintf(stderr,arg,regno,name,bits);
  }

static int get_system_info(void)
{
  int retval;
  pid_t pid;
  char maxargs[PAPI_MAX_STR_LEN] = "<none>";
  psinfo_t psi;
  int fd;
  int i,hz,version;
  char cpuname[PAPI_MAX_STR_LEN];
  const char *name;

  /* Check counter access */

  if (cpc_version(CPC_VER_CURRENT) != CPC_VER_CURRENT)
    return(PAPI_ESBSTR);
  DBG((stderr,"CPC version %d successfully opened\n",CPC_VER_CURRENT));

  if (cpc_access() == -1)
    return(PAPI_ESBSTR);

  /* Global variable cpuver */

  cpuver = cpc_getcpuver();
  DBG((stderr,"Got %d from cpc_getcpuver()\n",cpuver))
  if (cpuver == -1)
    return(PAPI_ESBSTR);
  name = cpc_getcciname(cpuver);
  if (name)
    DBG((stderr,"Got %s from cpc_getcciname\n",name))
  else
    DBG((stderr,"Got no name from cpc_getcciname\n"));

#ifdef DEBUG
  {
  extern int papi_debug;
  if (papi_debug) {
    name = cpc_getcpuref(cpuver);
    if(name)
      fprintf(stderr,"CPC CPU reference: %s\n",name);
    else
      fprintf(stderr,"Could not get a CPC CPU reference.\n");
  
    for(i=0;i<cpc_getnpic(cpuver);i++) {
      fprintf(stderr,"\n%6s %-40s %8s\n","Reg","Symbolic name","Code");
      cpc_walk_names(cpuver, i, "%6d %-40s %02x\n",print_walk_names);
      }
      fprintf(stderr,"\n");
    }
  }
#endif


  /* Initialize other globals */

  if (cpuver <= CPC_ULTRA2)
    {
      DBG((stderr,"cpuver (==%d) <= CPC_ULTRA2 (==%d)\n",cpuver,CPC_ULTRA2));
      pcr_shift[0] = CPC_ULTRA_PCR_PIC0_SHIFT; 
      pcr_shift[1] = CPC_ULTRA_PCR_PIC1_SHIFT; 
      pcr_event_mask[0] = (CPC_ULTRA2_PCR_PIC0_MASK<<CPC_ULTRA_PCR_PIC0_SHIFT);
      pcr_event_mask[1] = (CPC_ULTRA2_PCR_PIC1_MASK<<CPC_ULTRA_PCR_PIC1_SHIFT);
      pcr_inv_mask[0] = ~(pcr_event_mask[0]);
      pcr_inv_mask[1] = ~(pcr_event_mask[1]);
    }
  else if (cpuver == CPC_ULTRA3)
    {
      DBG((stderr,"cpuver (==%d) == CPC_ULTRA3 (==%d)\n",cpuver,CPC_ULTRA3));
      pcr_shift[0] = CPC_ULTRA_PCR_PIC0_SHIFT; 
      pcr_shift[1] = CPC_ULTRA_PCR_PIC1_SHIFT; 
      pcr_event_mask[0] = (CPC_ULTRA3_PCR_PIC0_MASK<<CPC_ULTRA_PCR_PIC0_SHIFT);
      pcr_event_mask[1] = (CPC_ULTRA3_PCR_PIC1_MASK<<CPC_ULTRA_PCR_PIC1_SHIFT);
      pcr_inv_mask[0] = ~(pcr_event_mask[0]);
      pcr_inv_mask[1] = ~(pcr_event_mask[1]);
      _papi_system_info.supports_hw_overflow = 1;
    }
  else
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
  strncpy(_papi_system_info.exe_info.name,basename(psi.pr_psargs),PAPI_MAX_STR_LEN);
  DBG((stderr,"Executable is %s\n",_papi_system_info.exe_info.name));
  DBG((stderr,"Full Executable is %s\n",_papi_system_info.exe_info.fullname));

  /* Hardware info */

  _papi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);

  /* Default strings until we know better... */
  _papi_system_info.hw_info.model = -1;
  strcpy(_papi_system_info.hw_info.model_string,"UltraSPARC???");
  sprintf(maxargs," %d",cpuver - 999);
  strcat(_papi_system_info.hw_info.model_string,maxargs);
  _papi_system_info.hw_info.vendor = cpuver;
  strcpy(_papi_system_info.hw_info.vendor_string,"SUN unknown");
  _papi_system_info.hw_info.revision = 0;

  retval = scan_prtconf(cpuname,PAPI_MAX_STR_LEN,&hz,&version);
  if (retval == -1)
    return(PAPI_ESBSTR);
  _papi_system_info.hw_info.mhz = ( (float) hz / 1.0e6 );
  DBG((stderr,"hw_info.mhz = %f\n",_papi_system_info.hw_info.mhz));
  /* Fill in the strings we got */
  strcpy(_papi_system_info.hw_info.model_string,cpuname);
  _papi_system_info.hw_info.model=version;

  strcpy(_papi_system_info.hw_info.vendor_string,cpc_getcciname(cpuver));
  _papi_system_info.hw_info.vendor=cpuver;

  retval = cpc_getnpic(cpuver);
  if (retval < 1)
    return(PAPI_ESBSTR);
  _papi_system_info.num_gp_cntrs = retval;
  _papi_system_info.num_cntrs = retval;
  DBG((stderr,"num_cntrs = %d\n",_papi_system_info.num_cntrs));

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

#if 0
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

  /* This doesn't exist on this platform */

  if (domain == PAPI_DOM_OTHER)
    return(PAPI_EINVAL);

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
  return(PAPI_ESBSTR);

/*
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  papi_cpc_event_t *command= &machdep->counter_cmd;

  return(PAPI_EINVAL);
*/

#if 0
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
#endif
}

static int set_default_domain(EventSetInfo *zero, int domain)
{
  /* This doesn't exist on this platform */

  if (domain == PAPI_DOM_OTHER)
    return(PAPI_EINVAL);

  return(PAPI_OK);

/*  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_domain(current_state,domain)); */
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

long long _papi_hwd_get_virt_usec (EventSetInfo *zero)
{
  return((long long)gethrvtime()/(long long)1000);
}

long long _papi_hwd_get_virt_cycles (EventSetInfo *zero)
{
  long long retval;
  float usec, cyc;
  struct tms buffer;

  times(&buffer);
  retval = (long long)buffer.tms_utime*(long long)(1000000/CLK_TCK);

  usec = (float)retval;
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error: %s",strerror(error));
}

int _papi_hwd_add_event(hwd_control_state_t *this_state, unsigned int EventCode, EventInfo_t *out)
{
  int selector = 0;
  int avail = 0;
  unsigned char tmp_cmd[US_MAX_COUNTERS], *codes;

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
      out->command = derived;
      out->operand_index = preset_map[preset_index].operand_index;
    }
  else
    {
      int hwcntr_num;
      uint64_t code;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  
      if ((hwcntr_num > _papi_system_info.num_gp_cntrs) ||
	  (hwcntr_num < 0))
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

  out->code = EventCode;
  out->selector = selector;

  return(PAPI_OK);
}

int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  int selector, used, preset_index, EventCode;

  /* Find out which counters used. */
  
  used = in->selector;
  EventCode = in->code;
 
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

      old_code = in->command;
      code = EventCode >> 8; 
      if (old_code != code)
	return(PAPI_EINVAL);

      selector = 1 << hwcntr_num;
    }

  /* Check if these counters aren't used. */

  if ((used & selector) != used)
    return(PAPI_EINVAL);

  /* Clear out counters that are part of this event. */

  this_state->selector = this_state->selector ^ used;

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)
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
#if DEBUG
      {
        extern int papi_debug;
        if (papi_debug) {
	  fprintf(stderr,"this selector:    %x\n", this_state->selector);
	  fprintf(stderr,"current selector: %x\n", current_state->selector);
	  fprintf(stderr,"both:             %x\n", hwcntrs_in_both);
	  fprintf(stderr,"all:              %x\n", hwcntrs_in_all);
        }
      }
#endif
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

#if 0
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
  long long correct[US_MAX_COUNTERS];

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
#if 0
    case PAPI_SET_INHERIT:
      return(set_inherit(zero,option->inherit.inherit));
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
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
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

void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, ucontext_t *info)
{
  DBG((stderr,"_papi_hwd_dispatch_timer() at 0x%lx\n",info->uc_mcontext.gregs[REG_PC]));
  _papi_hwi_dispatch_overflow_signal((void *)info); 
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
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

static rwlock_t lock;

void _papi_hwd_lock_init(void)
{
}

void _papi_hwd_lock(void)
{
  rw_wrlock(&lock);
}

void _papi_hwd_unlock(void)
{
  rw_unlock(&lock);
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

