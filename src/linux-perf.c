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
                {CNTR2|CNTR1,0,0,{0x45,0x45},""},	// L1 Cache Dmisses 
                {CNTR2|CNTR1,0,0,{0x81,0x81},""},	// L1 Cache Imisses 
		{0,0,0,{0,0},""}, 			// L2 Cache Dmisses
		{0,0,0,{0,0},""}, 			// L2 Cache Imisses
		{0,0,0,{0,0},""}, 			// L3 Cache Dmisses
		{0,0,0,{0,0},""}, 			// L3 Cache Imisses
                {CNTR2|CNTR1,DERIVED_ADD,0,{0x45,0x81},""},	// L1 Total Cache misses 
		{CNTR2|CNTR1,0,0,{0x24,0x24},""}, 	// L2 Total Cache misses
		{0,0,0,{0,0},""}, 			// L3 Total Cache misses
		{0,0,0,{0,0},""},			// Snoops
		{0,0,0,{0,0},""},		 	// Req. access to shared cache line
		{0,0,0,{0,0},""},		 	// Req. access to clean cache line
		{CNTR2|CNTR1,0,0,{0x69,0x69},""},	// Cache Line Invalidation
                {0,0,0,{0,0},""},			// Cache Line Intervention
                {0,0,0,{0,0},""},			// L3 LDM
                {0,0,0,{0,0},""},			// L3 STM
                {0,0,0,{0,0},""},			// cycles branch idle
                {0,0,0,{0,0},""},			// cycles int idle
                {0,0,0,{0,0},""},			// cycles fpu idle
                {0,0,0,{0,0},""},			// cycles load/store idle
		{0,0,0,{0,0},""},		 	// D-TLB misses
		{CNTR2|CNTR1,0,0,{0x85,0x85},""},	// I-TLB misses
                {0,0,0,{0,0},""},			// Total TLB misses
                {0,0,0,{0,0},""},			// L1 load M
                {0,0,0,{0,0},""},			// L1 store M
                {0,0,0,{0,0},""},			// L2 load M
                {0,0,0,{0,0},""},			// L2 store M
                {CNTR2|CNTR1,0,0,{0xe2,0xe2},""},	// BTAC misses
                {0,0,0,{0,0},""},			// Prefmiss
                {0,0,0,{0,0},""},			// L3DCH
		{0,0,0,{0,0},""},			// TLB shootdowns
                {0,0,0,{0,0},""},			// Failed Store cond.
                {0,0,0,{0,0},""},			// Suc. store cond.
                {0,0,0,{0,0},""},			// total. store cond.
                {0,0,0,{0,0},""},	                /* Cycles stalled waiting for memory */
                {0,0,0,{0,0},""},		   	/* Cycles stalled waiting for memory read */
                {0,0,0,{0,0},""},		   	/* Cycles stalled waiting for memory write */
                {0,0,0,{0,0},""},			/* Cycles no instructions issued */
                {0,0,0,{0,0},""},			/* Cycles max instructions issued */
                {0,0,0,{0,0},""},			/* Cycles no instructions comleted */
                {0,0,0,{0,0},""},			/* Cycles max instructions completed */
                {CNTR2|CNTR1,0,0,{0xC8,0xC8},""},	// hardware interrupts
		{0,0,0,{0,0},""},	                // Uncond. branches executed
		{CNTR2|CNTR1,0,0,{0xC4,0xC4},""},	// Cond. Branch inst. executed
		{CNTR2|CNTR1,0,0,{0xC9,0xC9},""},	// Cond. Branch inst. taken
		{CNTR2|CNTR1,DERIVED_SUB,0,{0xC4,0xC9},""}, // Cond. Branch inst. not taken
                {CNTR2|CNTR1,0,0,{0xC5,0xC5},""},	// Cond. branch inst. mispred.
                {CNTR2|CNTR1,DERIVED_SUB,0,{0xC4,0xC5},""}, // Cond. branch inst. corr. pred.
                {0,0,0,{0,0},""},			// FMA
                {CNTR2|CNTR1,0,0,{0xD0,0xD0},""},	// Total inst. issued
		{CNTR2|CNTR1,0,0,{0xC0,0xC0},""},	// Total inst. executed
		{0,0,0,{0,0},""},			// Integer inst. executed
		{CNTR1,0,0,{0xC1,0},""},	// Floating Pt. inst. executed
		{0,0,0,{0,0},""},			// Loads executed
		{0,0,0,{0,0},""},			// Stores executed
		{CNTR2|CNTR1,0,0,{0xC4,0xC4},""},	// Branch inst. executed
		{CNTR2|CNTR1,0,0,{0xB0,0xB0},""},	// Vector/SIMD inst. executed 
		{CNTR2|CNTR1,DERIVED_PS,1,{0xC1,0x79},""},	// FLOPS
                {CNTR2|CNTR1,0,0,{0xA2,0xA2},""},		// Cycles any resource stalls
                {0,0,0,{0,0},""},			// Cycles FPU stalled
		{CNTR2|CNTR1,0,0,{0x79,0x79},""},	// Total cycles
		{CNTR2|CNTR1,DERIVED_PS,1,{0xC0,0x79},""},	// IPS
                {CNTR2|CNTR1,0,0,{0x43,0x43},""},	// Total load/store inst. exec
                {0,0,0,{0,0},""}, // SYnc exec.
		{0,0,0,{0,0},""}, // L1_DCH
		{0,0,0,{0,0},""}, // L2_DCH
		{0,0,0,{0,0},""}, // L1_DCA
		{CNTR2|CNTR1,DERIVED_ADD,0,{0x29,0x2a},""}, // L2_DCA
		{0,0,0,{0,0},""}, // L3_DCA
		{0,0,0,{0,0},""}, // L1_DCR
		{CNTR2|CNTR1,0,0,{0x29,0x29},""}, // L2_DCR
		{0,0,0,{0,0},""}, // L3_DCR
		{0,0,0,{0,0},""}, // L1_DCW
		{CNTR2|CNTR1,0,0,{0x2a,0x2a},""}, // L2_DCW
		{0,0,0,{0,0},""}, // L3_DCW
		{0,0,0,{0,0},""}, // L1_ICH
		{0,0,0,{0,0},""}, // L2_ICH
		{0,0,0,{0,0},""}, // L3_ICH
		{0,0,0,{0,0},""}, // L1_ICA
		{0,0,0,{0,0},""}, // L2_ICA
		{0,0,0,{0,0},""}, // L3_ICA
		{CNTR2|CNTR1,0,0,{0x80,0x80},""}, // L1_ICR
		{0,0,0,{0,0},""}, // L2_ICR
		{0,0,0,{0,0},""}, // L3_ICR
		{0,0,0,{0,0},""}, // L1_ICW
		{0,0,0,{0,0},""}, // L2_ICW
		{0,0,0,{0,0},""}, // L3_ICW
		{0,0,0,{0,0},""}, // L1_TCH
		{0,0,0,{0,0},""}, // L2_TCH
		{0,0,0,{0,0},""}, // L3_TCH
		{0,0,0,{0,0},""}, // L1_TCA
		{CNTR2|CNTR1,0,0,{0x2e,0x2e},""}, // L2_TCA
		{0,0,0,{0,0},""}, // L3_TCA
		{0,0,0,{0,0},""}, // L1_TCR
		{0,0,0,{0,0},""}, // L2_TCR
		{0,0,0,{0,0},""}, // L3_TCR
		{0,0,0,{0,0},""}, // L1_TCW
		{0,0,0,{0,0},""}, // L2_TCW
		{0,0,0,{0,0},""}, // L3_TCW
		{CNTR2,0,0,{0,0x12},""}, // FPM
		{0,0,0,{0,0},""}, // FPA
   		{CNTR2,0,0,{0,0x13},""}, // FPD
		{0,0,0,{0,0},""}, // FPSQ
		{0,0,0,{0,0},""}, // FPI

             };

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

static float calc_mhz(void)
{
  unsigned long long ostamp;
  unsigned long long stamp;
  float correction = 4000.0, mhz;

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

static int setup_all_presets(PAPI_hw_info_t *info)
{
  int pnum, s;

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      if ((s = preset_map[pnum].selector))
	{
	  if (preset_map[pnum].derived == 0)
	    {
	      if (s == CNTR1)
		sprintf(preset_map[pnum].note,"0x%x",preset_map[pnum].counter_cmd[0]);
	      else
		sprintf(preset_map[pnum].note,"0x%x",preset_map[pnum].counter_cmd[1]);
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
  ptr->counter_cmd[0] |= def_mode | PERF_ENABLE;
  ptr->counter_cmd[1] |= def_mode;
  ptr->counter_cmd[2] = 0;
}

static int get_system_info(void)
{
  pid_t pid;
  int tmp;
  char maxargs[PAPI_MAX_STR_LEN], *t, *s;
  FILE *f;
  float mhz;

  /* Path and args */

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);

  sprintf(maxargs,"/proc/%d/exe",(int)getpid());
  if (readlink(maxargs,_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == -1)
    return(PAPI_ESYS);
  sprintf(_papi_system_info.exe_info.name,"%s",basename(_papi_system_info.exe_info.fullname));

  DBG((stderr,"Executable is %s\n",_papi_system_info.exe_info.name));
  DBG((stderr,"Full Executable is %s\n",_papi_system_info.exe_info.fullname));

  if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
    return -1;
 
  /* Hardware info */

  _papi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
  _papi_system_info.hw_info.vendor = -1;

  /* The following statement courtesy of PCL. */

  {
    unsigned int signature, feature_flags;
    
    asm("cpuid"
	: "=a"(signature), "=d"(feature_flags)
	: "a"(1)
	: "%eax", "%ebx", "%ecx", "%edx");

    _papi_system_info.hw_info.model = ((signature >> 4) & 0xf); /* Bits 4 through 7 */
    if ((((signature >> 8) & 0xf) != 0x6) ||
	(((feature_flags >> 4) & 0x1) != 1) ||
	(((feature_flags >> 5) & 0x1) != 1))
      {
	fprintf(stderr,"This processor is not supported.\n");
	return(PAPI_ESBSTR);
      }
  }

  t = s = search_cpu_info(f,"model name",maxargs);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_system_info.hw_info.model_string,s+2);
    }
  rewind(f);
  if (t == NULL) /* < Linux 2.2 */
    {
      s = search_cpu_info(f,"model",maxargs);
      if (s && (t = strchr(s+2,'\n')))
	{
	  *t = '\0';
	  strcpy(_papi_system_info.hw_info.model_string,s+2);
	}
    }

  rewind(f);
  s = search_cpu_info(f,"vendor_id",maxargs);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_system_info.hw_info.vendor_string,s+2);
    }

  rewind(f);
  s = search_cpu_info(f,"stepping",maxargs);
  if (s)
    sscanf(s+1, "%d", &tmp);
  _papi_system_info.hw_info.revision = (float)tmp;

  rewind(f);
  s = search_cpu_info(f,"cpu MHz",maxargs);
  if (s)
    sscanf(s+1, "%f", &_papi_system_info.hw_info.mhz);
  fclose(f);

  _papi_system_info.num_gp_cntrs = 2;
  _papi_system_info.num_cntrs = 2;

  /* Check mhz calc. */

  mhz = calc_mhz();
  DBG((stderr,"Calculated MHZ is %f\n",mhz));
  DBG((stderr,"Detected MHZ is %f\n",_papi_system_info.hw_info.mhz));
  if (_papi_system_info.hw_info.mhz < mhz)
    _papi_system_info.hw_info.mhz = mhz;
  {
    int tmp = (int)_papi_system_info.hw_info.mhz;
    _papi_system_info.hw_info.mhz = (float)tmp;
  }
  DBG((stderr,"Actual MHZ is %f\n",_papi_system_info.hw_info.mhz));

  /* Setup presets */

  tmp = setup_all_presets(&_papi_system_info.hw_info);
  if (tmp)
    return(tmp);

  return(PAPI_OK);
} 

#ifdef DEBUG
static void dump_cmd(int *t)
{
  int i;

  for (i=0;i<MAX_COUNTERS;i++)
    fprintf(stderr,"Event %d: 0x%x\n",i,t[i]);
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
  long long cyc;

  cyc = perf_get_cycles()*(unsigned long long)1000;
  cyc = cyc / (long long)_papi_system_info.hw_info.mhz;
  return(cyc / (long long)1000);
}

long long _papi_hwd_get_real_cycles (void)
{
  return(perf_get_cycles());
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

/* Do not ever use ESI->NumberOfCounters in here. */

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector = 0;
  int avail = 0;
  unsigned int tmp_cmd[MAX_COUNTERS];
  unsigned int *codes;

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

      hwcntr_num = EventCode & 0xff;  
      if (hwcntr_num > _papi_system_info.num_gp_cntrs) /* 0 or 1 */ 
	return(PAPI_EINVAL);

      tmp_cmd[hwcntr_num] = EventCode >> 8; 
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
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,MAX_COUNTERS*sizeof(int));

      /* Stop the current context */

      retval = perf(PERF_RESET, 0, 0);
      if (retval) 
	return(PAPI_ESYS); 
      
      /* (Re)start the counters */
      
#ifdef DEBUG
      dump_cmd(current_state->counter_cmd);
#endif
      retval = perf(PERF_FASTCONFIG, (int)current_state->counter_cmd, (int)NULL);
      if (retval) 
	return(PAPI_ESYS);
      
      return(PAPI_OK);
    }

  /* If we ARE nested, 
     carefully merge the global counter structure with the current eventset */
  else
    {
      int tmp, hwcntrs_in_both, hwcntrs_in_all, hwcntr;

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
	      if (!(counter_event_shared(this_state->counter_cmd, current_state->counter_cmd, i-1)))
		return(PAPI_ECNFLCT);
	    }
	  else if (!(counter_event_compat(this_state->counter_cmd, current_state->counter_cmd, i-1)))
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
	      counter_event_copy(this_state->counter_cmd, current_state->counter_cmd, i-1);
	      ESI->hw_start[i-1] = 0;
	    }
	}
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
			        0,  /*  num_cntrs */
			        0,  /*  num_gp_cntrs */
			        0,  /*  grouped_counters */
			        0,  /*  num_sp_cntrs */
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
			        NULL };

