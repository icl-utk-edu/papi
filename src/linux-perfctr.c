/* 
* File:    linux-perfctr.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    nils smeds
*          smeds@pdc.kth.se
* Mods:    Kevin London
*	   london@cs.utk.edu
*/  

#ifdef PERFCTR20
#define PERFCTR18
#endif
/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#include SUBSTRATE

#include "ia32_presets.h"

/* First entry is mask, counter code 1, counter code 2, and TSC. 
A high bit in the mask entry means it is an OR mask, not an
and mask. This means that the same even is available on either
counter. */

static hwd_preset_t *preset_map = NULL;

/* Since the preset maps are identical for all ia32 substrates
  (at least Linux and Windows) the preset maps are in a common
  file to minimimze redundant maintenance.
  NOTE: The obsolete linux-perf substrate is not supported by
  this scheme, although it could be.
*/

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

static inline unsigned long long get_cycles (void)
{
	unsigned long long ret;
        __asm__ __volatile__("rdtsc"
			    : "=A" (ret)
			    : /* no inputs */);
        return ret;
}

/* Dumb hack to make sure I get the cycle time correct. */

static float calc_mhz(void)
{
  unsigned long long ostamp;
  unsigned long long stamp;
  float correction = 4000.0, mhz;

  /* Warm the cache */

  ostamp = get_cycles();
  usleep(1);
  stamp = get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  ostamp = get_cycles();
  sleep(1);
  stamp = get_cycles();
  stamp = stamp - ostamp;
  mhz = (float)stamp/(float)(1000000.0 + correction);

  return(mhz);
}

inline static int setup_all_presets(int cpu_type)
{
  int pnum, s;
  char note[100];

  preset_map = NULL; 

  switch(cpu_type)
    {

    case PERFCTR_X86_GENERIC:
      fprintf(stderr,"This processor is not properly identified by the substrate.");
      preset_map = calloc(1, sizeof p6_preset_map);
      break;

    case PERFCTR_X86_INTEL_P5:
    case PERFCTR_X86_INTEL_P5MMX:
    case PERFCTR_X86_INTEL_P6:
    case PERFCTR_X86_INTEL_PII:
    case PERFCTR_X86_INTEL_PIII:
    case PERFCTR_X86_CYRIX_MII:
      preset_map = p6_preset_map;
      break;

    case PERFCTR_X86_WINCHIP_C6:
    case PERFCTR_X86_WINCHIP_2:
      fprintf(stderr,"Ask yourself, why am I tuning code on a WinChip?\n");
      preset_map = calloc(1, sizeof p6_preset_map);
      break;
      
    case PERFCTR_X86_AMD_K7:
      preset_map = k7_preset_map;
      break;

    case PERFCTR_X86_VIA_C3:
      fprintf(stderr,"This platform is not supported by PAPI\n");
      /* This is most probably wrong, but it is backwards compatible to 
	 the behaviour of earlier versions of linux-perfctr.c */
      preset_map = p6_preset_map; 
      break;

    case PERFCTR_X86_INTEL_P4:
    case PERFCTR_X86_INTEL_P4M2:
      fprintf(stderr,"Intel Pentium 4 is not supported by this substrate.\n");
      break;

    default:
      fprintf(stderr,"%s, %s:%d:: %s (%d)\n",
	      __FILE__, __FUNCTION__, __LINE__,
	      "Unexpected PERFCTR processor type",cpu_type);
    }

  /* We are running on an unsupported CPU and this substrate can not
     handle a NULL preset_map. So we'll return PAPI_ESBSTR to prevent
     the rest of the library init to try to do things with it */
  if (!preset_map)
    return PAPI_ESBSTR;

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      if ((s = preset_map[pnum].selector))
	{
	  if (_papi_system_info.num_cntrs == 2)
	    snprintf(note,sizeof note,"0x%x,0x%x",
		     preset_map[pnum].counter_cmd.evntsel[0],
		     preset_map[pnum].counter_cmd.evntsel[1]);
	  else if (_papi_system_info.num_cntrs == 4)
	    snprintf(note,sizeof note,"0x%x,0x%x,0x%x,0x%x",
		     preset_map[pnum].counter_cmd.evntsel[0],
		     preset_map[pnum].counter_cmd.evntsel[1],
		     preset_map[pnum].counter_cmd.evntsel[2],
		     preset_map[pnum].counter_cmd.evntsel[3]);
	  else
	    {
	      fprintf(stderr,"%s, %s:%d:: %s\n",
		      __FILE__, __FUNCTION__, __LINE__,
		      "Unexpected internal error.");
	      return PAPI_ESBSTR;
	    }

	  /* If there is a string, add a space before the information here */
	  if(preset_map[pnum].note[0] && 
	     ((sizeof preset_map[pnum].note) - strlen(preset_map[pnum].note) > 2))
	    strcat(preset_map[pnum].note," ");
	  /* Be careful with string sizes... */
	  strncat(preset_map[pnum].note,note,
		  (sizeof preset_map[pnum].note)-(strlen(preset_map[pnum].note)+1));
	}
    }
  return(PAPI_OK);
}

/* Utility functions */

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

inline static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (_papi_system_info.num_cntrs-1);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
}

#ifdef PERFCTR20
inline static void set_hwcntr_codes(int selector, struct perfctr_cpu_control *from, struct perfctr_cpu_control *to)
#else
inline static void set_hwcntr_codes(int selector, struct perfctr_control *from, struct perfctr_control *to)
#endif
{
  int useme, i;
  
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  to->evntsel[i] &= ~(PERF_UNIT_MASK | PERF_EVNT_MASK);
	  to->evntsel[i] |= from->evntsel[i];
#ifdef PERFCTR20
	  to->pmc_map[i] = i;
#endif
	}
    }
}

inline static void init_config(hwd_control_state_t *ptr)
{
  int def_mode, i;

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
#ifdef PERFCTR20
  switch(_papi_system_info.hw_info.model)
    {
    case PERFCTR_X86_GENERIC:
    case PERFCTR_X86_CYRIX_MII:
    case PERFCTR_X86_WINCHIP_C6: 
    case PERFCTR_X86_WINCHIP_2:
    case PERFCTR_X86_VIA_C3:    
      ptr->counter_cmd.cpu_control.tsc_on=1;
      ptr->counter_cmd.cpu_control.nractrs=0;
      ptr->counter_cmd.cpu_control.nrictrs=0;
      break;
    case PERFCTR_X86_INTEL_P5:
    case PERFCTR_X86_INTEL_P5MMX:
    case PERFCTR_X86_INTEL_P6:
    case PERFCTR_X86_INTEL_PII:  
    case PERFCTR_X86_INTEL_PIII: 
      ptr->counter_cmd.cpu_control.evntsel[0] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.cpu_control.evntsel[1] |= def_mode;
      ptr->counter_cmd.cpu_control.tsc_on=1;
      ptr->counter_cmd.cpu_control.nractrs=_papi_system_info.num_cntrs;
      ptr->counter_cmd.cpu_control.nrictrs=0;
      break;
    case PERFCTR_X86_AMD_K7:
      ptr->counter_cmd.cpu_control.evntsel[0] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.cpu_control.evntsel[1] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.cpu_control.evntsel[2] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.cpu_control.evntsel[3] |= def_mode | PERF_ENABLE;
      ptr->counter_cmd.cpu_control.tsc_on=1;
      ptr->counter_cmd.cpu_control.nractrs=_papi_system_info.num_cntrs;
      ptr->counter_cmd.cpu_control.nrictrs=0;
      break;
    default:
      abort();
    }
  /* Identity counter map for starters */
  for(i=0;i<_papi_system_info.num_cntrs;i++) 
    ptr->counter_cmd.cpu_control.pmc_map[i]=i;
#else
  ptr->counter_cmd.evntsel[0] |= def_mode | PERF_ENABLE;
  ptr->counter_cmd.evntsel[1] |= def_mode;
#endif
}

#ifdef PERFCTR18
static int get_system_info(const struct vperfctr *dev)
#else
static int get_system_info(struct perfctr_dev *dev)
#endif
{
  struct perfctr_info info;
  pid_t pid;
  int tmp;
  float mhz;
  char maxargs[PAPI_MAX_STR_LEN], *t, *s;
  FILE *cpuinfo;

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

  if ((cpuinfo = fopen("/proc/cpuinfo", "r")) == NULL)
    return PAPI_ESYS;
 
  /* Hardware info */

  _papi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_system_info.hw_info.nnodes = 1;
  _papi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
  _papi_system_info.hw_info.vendor = -1;

  rewind(cpuinfo);
  s = search_cpu_info(cpuinfo,"vendor_id",maxargs);
  if (s && (t = strchr(s+2,'\n')))
    {
      *t = '\0';
      strcpy(_papi_system_info.hw_info.vendor_string,s+2);
    }

  rewind(cpuinfo);
  s = search_cpu_info(cpuinfo,"stepping",maxargs);
  if (s)
    sscanf(s+1, "%d", &tmp);
  _papi_system_info.hw_info.revision = (float)tmp;

  fclose(cpuinfo);

#if defined(PERFCTR18) || defined(PERFCTR20) 
  if (vperfctr_info(dev, &info) < 0)
    return(PAPI_ESYS);
  if (strstr(info.version,"2.4") != info.version)
    {
      fprintf(stderr,"Version mismatch of perfctr: compiled 2.4 or higher vs. installed %s\n",info.version);
      return(PAPI_ESBSTR);
    }

  strcpy(_papi_system_info.hw_info.model_string,perfctr_cpu_name(&info));
  _papi_system_info.supports_hw_overflow = 
    (info.cpu_features & PERFCTR_FEATURE_PCINT) ? 1 : 0;
  DBG((stderr,"Hardware/OS %s support counter generated interrupts\n",
       _papi_system_info.supports_hw_overflow ? "does" : "does not"));
#ifndef PAPI_PERFCTR_INTR_SUPPORT
  if(_papi_system_info.supports_hw_overflow)
    DBG((stderr,"PAPI_PERFCTR_INTR_SUPPORT disabled at compile time.\n"));
  _papi_system_info.supports_hw_overflow = 0;
#endif
  _papi_system_info.supports_hw_profile = 0; /* != _papi_system_info.supports_hw_overflow? */
#ifdef PERFCTR20
  _papi_system_info.num_cntrs = perfctr_cpu_nrctrs(&info);
  _papi_system_info.num_gp_cntrs = perfctr_cpu_nrctrs(&info);
#else /* PERFCTR20 */
  _papi_system_info.num_cntrs = perfctr_cpu_nrctrs(&info) - 1;
  _papi_system_info.num_gp_cntrs = perfctr_cpu_nrctrs(&info) - 1;
#endif /* PERFCTR20 */
#elif defined(PERFCTR16)  /* Neither PERFCTR18 nor PERFCTR20 */
  if (perfctr_info(dev, &info) < 0)
    return(PAPI_ESYS);
  strcpy(_papi_system_info.hw_info.model_string,perfctr_cpu_name(&info));
  _papi_system_info.num_cntrs = perfctr_cpu_nrctrs(&info) - 1;
  _papi_system_info.num_gp_cntrs = perfctr_cpu_nrctrs(&info) - 1;
#endif /* PERFCTR18 */

  _papi_system_info.hw_info.model = (int)info.cpu_type;
  _papi_system_info.hw_info.mhz = (float) info.cpu_khz / 1000.0; 
  DBG((stderr,"Detected MHZ is %f\n",_papi_system_info.hw_info.mhz));
  mhz = calc_mhz();
  DBG((stderr,"Calculated MHZ is %f\n",mhz));

  /* If difference is larger than 5% (e.g. system info is 0) use 
     calculated value. (If CPU value seems reasonable use it) */
  if (abs(mhz-_papi_system_info.hw_info.mhz) > 0.95*_papi_system_info.hw_info.mhz)
    _papi_system_info.hw_info.mhz = mhz;

#ifndef PERFCTR20 /* Skip truncating MHz value */
  {
    int tmp = (int)_papi_system_info.hw_info.mhz;
    _papi_system_info.hw_info.mhz = (float)tmp;
  }
#endif
  DBG((stderr,"Actual MHZ is %f\n",_papi_system_info.hw_info.mhz));

  /* Setup memory info */

  tmp = get_memory_info(&_papi_system_info.mem_info, (int)info.cpu_type);
  if (tmp)
    return(tmp);

  /* Setup presets */

  tmp = setup_all_presets((int)info.cpu_type);
  if (tmp)
    return(tmp);

  return(PAPI_OK);
} 

#ifdef DEBUG
#ifdef PERFCTR20
static void dump_cmd(char *str, struct vperfctr_control *t)
#else
static void dump_cmd(char *str, struct perfctr_control *t)
#endif
{
  int i,k;

#ifdef PERFCTR20
  DBG((stderr,"%s: tsc_on=0x%x  nractrs=0x%x, nrictrs=0x%x\n",str,t->cpu_control.tsc_on,t->cpu_control.nractrs,t->cpu_control.nrictrs));
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      k=t->cpu_control.pmc_map[i];
      DBG((stderr,"Item %d [map %d]: Evntsel=0x%08x   (ireset=%d)\n",i,k,t->cpu_control.evntsel[i],t->cpu_control.ireset[i]));
    }
#else
  DBG((stderr,"%s:",str));
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    DBG((stderr,"Event %d: 0x%08x\n",i,t->evntsel[i]));
#endif
}
#endif

#ifdef PERFCTR20
inline static int counter_event_shared(const struct vperfctr_control *a, const struct vperfctr_control *b, int cntr)
#else
inline static int counter_event_shared(const struct perfctr_control *a, const struct perfctr_control *b, int cntr)
#endif
{
#ifdef PERFCTR20
  if (a->cpu_control.evntsel[cntr] == b->cpu_control.evntsel[cntr])
    return(1);
#else
  if (a->evntsel[cntr] == b->evntsel[cntr])
    return(1);
#endif

  return(0);
}

#ifdef PERFCTR20
inline static int counter_event_compat(const struct vperfctr_control *a, const struct vperfctr_control *b, int cntr)
#else
inline static int counter_event_compat(const struct perfctr_control *a, const struct perfctr_control *b, int cntr)
#endif
{
  unsigned int priv_mask = ~(PERF_EVNT_MASK|PERF_UNIT_MASK);

#ifdef PERFCTR20
  if ((a->cpu_control.evntsel[cntr] & priv_mask) ==
      (b->cpu_control.evntsel[cntr] & priv_mask))
#else
  if ((a->evntsel[cntr] & priv_mask) == (b->evntsel[cntr] & priv_mask))
#endif
    return(1);

  return(0);
}

#ifdef PERFCTR20
inline static void counter_event_copy(const struct vperfctr_control *a, struct vperfctr_control *b, int cntr)
#else
inline static void counter_event_copy(const struct perfctr_control *a, struct perfctr_control *b, int cntr)
#endif
{
#ifdef PERFCTR20
  b->cpu_control.evntsel[cntr] = a->cpu_control.evntsel[cntr];
#else
  b->evntsel[cntr] = a->evntsel[cntr];
#endif
}

inline static int update_global_hwcounters(EventSetInfo *global)
{
  hwd_control_state_t *machdep = global->machdep;
#ifdef PERFCTR20
  struct perfctr_sum_ctrs sum;
  int *pmc_map=machdep->counter_cmd.cpu_control.pmc_map;
  struct vperfctr_control control;
#else
  struct vperfctr_state state;
#endif
  int cntr,i,nrictrs,nractrs;

#ifdef PERFCTR20
  /* read_state seems necessary to get the sum right here */
  vperfctr_read_state(machdep->self, &sum, &control);
  DBG((stderr,"sum tsc=%lld   [0]%lld   [1]%lld\n",
       sum.tsc,sum.pmc[0],sum.pmc[1]));
  nractrs = machdep->counter_cmd.cpu_control.nractrs;
  nrictrs = machdep->counter_cmd.cpu_control.nrictrs;
  /*  We don't map the perfcntr order back to the actual
      hardware counter map order here. This mapping is left 
      for _papi_hwd_read to do
  sum.tsc = unmapped_sum.tsc;
  for(i=0;i<nractrs+nrictrs;i++)
    sum.pmc[i]=unmapped_sum.pmc[pmc_map[i]];
  DBG((stderr,"mapped_sum   tsc=%lld   [0]%lld   [1]%lld\n",
       sum.tsc,sum.pmc[0],sum.pmc[1]));
  */
#else
  if (vperfctr_read_state(machdep->self, &state) < 0) 
    return(PAPI_ESYS);
  nractrs = _papi_system_info.num_cntrs;
  nrictrs = 0;
#endif
#ifdef DEBUG
  DBG((stderr,"nractrs=%d nrictrs=%d\n",nractrs,nrictrs));
  dump_cmd("global->machdep",&machdep->counter_cmd);
#endif

  for (i=0;i<nractrs;i++)
    {  
      unsigned long long ull_count;
#ifdef PERFCTR20
      /*      ull_count=sum.pmc[pmc_map[i]]; */
      ull_count=sum.pmc[i];
#else
      ull_count=state.sum.ctr[i+1];
#endif
      DBG((stderr,"[%d]: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+ull_count,
	   global->hw_start[i],ull_count));
      global->hw_start[i] = global->hw_start[i] + ull_count;
    }

  for (i=0;i<nrictrs;i++)
    {
      unsigned long long ull_count;
      int now;

      cntr = nractrs+i;
#ifdef PERFCTR20
#define rdpmcl(ctr,low) \
        __asm__ __volatile__("rdpmc" : "=a"(low) : "c"(ctr) : "edx")
      rdpmcl(pmc_map[cntr],now);
      ull_count = now - machdep->counter_cmd.cpu_control.ireset[cntr];
      DBG((stderr,"[intr(%d)]: C%lld = rdpmc(%d) - ireset(%d)\n",cntr,
	   ull_count,now,machdep->counter_cmd.cpu_control.ireset[cntr]));
#else
      /* This shouldn't happen */
      abort();
#endif
      DBG((stderr,"[intr(%d)]: G%lld = G%lld + C%lld\n",cntr,
	   global->hw_start[cntr]+ull_count,
	   global->hw_start[cntr],ull_count));
      global->hw_start[cntr] = global->hw_start[cntr] + ull_count;
    }

  /* This restarts the interrupting counters and is the reason why
     vperfctr_iresume can not be used in _papi_hwd_dispatch_timer() 
     Exactly why is it needed here? */
  if (vperfctr_control(machdep->self, &machdep->counter_cmd) < 0) 
    return(PAPI_ESYS);

  return(PAPI_OK);
}

inline static int correct_local_hwcounters(EventSetInfo *global, EventSetInfo *local, long long *correct)
{
  int i;

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      correct[i] = global->hw_start[i] - local->hw_start[i];
      DBG((stderr,"correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n",i,
	   correct[i],global->hw_start[i],local->hw_start[i]));
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

#ifdef PERFCTR20
  this_state->counter_cmd.cpu_control.evntsel[0] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd.cpu_control.evntsel[0] |= mode0;
  this_state->counter_cmd.cpu_control.evntsel[1] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd.cpu_control.evntsel[1] |= mode1;
#else
  this_state->counter_cmd.evntsel[0] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd.evntsel[0] |= mode0;
  this_state->counter_cmd.evntsel[1] &= ~(PERF_OS|PERF_USR);
  this_state->counter_cmd.evntsel[1] |= mode1;
#endif

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

#ifndef PERFCTR18
struct perfctr_dev *dev;
#endif
int _papi_hwd_init_global(void)
{
  int retval;

  /* Opened once for all threads. */

#ifdef PERFCTR18
  struct vperfctr *dev;
  dev = vperfctr_open();
  DBG((stderr,"_papi_hwd_init_global dev=%p\n",dev));
#else
  dev = perfctr_dev_open();
#endif
  if (!dev)
    return(PAPI_ESYS);

  /* Fill in what we can of the papi_system_info. */
  
  retval = get_system_info(dev);
  if (retval)
    return(retval);
  
  DBG((stderr,"Found %d %s %s CPU's at %f Mhz.\n",
       _papi_system_info.hw_info.totalcpus,
       _papi_system_info.hw_info.vendor_string,
       _papi_system_info.hw_info.model_string,
       _papi_system_info.hw_info.mhz));

#ifdef PERFCTR18
  vperfctr_unlink(dev);
  vperfctr_close(dev);
#endif
  return(PAPI_OK);
}

int _papi_hwd_init(EventSetInfo *zero)
{
  hwd_control_state_t *machdep = zero->machdep;
  
  /* Initialize our global machdep. */

#ifdef PERFCTR18
  if ((machdep->self = vperfctr_open()) == NULL) 
    return(PAPI_ESYS);
  DBG((stderr,"_papi_hwd_init dev=%p\n",machdep->self));
#else
  if ((machdep->self = vperfctr_attach(dev)) == NULL) 
    return(PAPI_ESYS);
#endif

  /* Initialize the event fields */

  init_config(zero->machdep);

  /* Start the TSC counter */
  if(vperfctr_control(machdep->self, &machdep->counter_cmd) < 0)
    return(PAPI_ESYS);

  return(PAPI_OK);
}

long long _papi_hwd_get_real_usec (void)
{
  long long cyc;

  cyc = get_cycles()*(unsigned long long)1000;
  cyc = cyc / (long long)_papi_system_info.hw_info.mhz;
  return(cyc / (long long)1000);
}

long long _papi_hwd_get_real_cycles (void)
{
  return(get_cycles());
}

long long _papi_hwd_get_virt_usec (EventSetInfo *zero)
{
  long long retval;
#ifdef PERFCTR16
  struct tms buffer;

  times(&buffer);
  retval = (long long)buffer.tms_utime*(long long)(1000000/CLK_TCK);
  return(retval);
#else
  retval = _papi_hwd_get_virt_cycles(zero);
  retval = retval / _papi_system_info.hw_info.mhz;
  return(retval);
#endif
}

long long _papi_hwd_get_virt_cycles (EventSetInfo *zero)
{
#ifdef PERFCTR16
/* (NCSA change)
   Reverted back to prior version 2/2/02
*/
  float usec, cyc;

  usec = (float)_papi_hwd_get_virt_usec(zero);
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);

#else
  unsigned long long lcyc;
  hwd_control_state_t *machdep = zero->machdep;

  lcyc = vperfctr_read_tsc(machdep->self);
  DBG((stderr,"Read virt. cycles is %llu (%p -> %p)\n",lcyc,machdep,machdep->self));
  return(lcyc);
#endif
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error: %s",strerror(error));
}

int _papi_hwd_add_event(hwd_control_state_t *this_state, unsigned int EventCode, EventInfo_t *out)
{
  int selector = 0;
  int avail = 0;
#ifdef PERFCTR20
  struct perfctr_cpu_control tmp_cmd, *codes;
#else
  struct perfctr_control tmp_cmd, *codes;
#endif

  if (EventCode & PRESET_MASK)
    { 
      int preset_index;
      int derived;

      preset_index = EventCode & PRESET_AND_MASK; 

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

#ifdef PERFCTR20
      codes=&tmp_cmd;
      memcpy(&tmp_cmd.evntsel,&preset_map[preset_index].counter_cmd,
	     sizeof(preset_map[preset_index].counter_cmd));
#else
      codes = &preset_map[preset_index].counter_cmd;
#endif
      out->command = derived;
      out->operand_index = preset_map[preset_index].operand_index;
    }
  else
    {
      int hwcntr_num;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  
      if ((hwcntr_num > _papi_system_info.num_gp_cntrs) ||
	  (hwcntr_num < 0))
	return(PAPI_EINVAL);

      tmp_cmd.evntsel[hwcntr_num] = EventCode >> 8; 
      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->selector & selector)
	return(PAPI_ECNFLCT);	    

      codes = &tmp_cmd;
    }

  /* Lower bits tell us what counters we need */

  assert((this_state->selector | ((1<<_papi_system_info.num_cntrs)-1)) == ((1<<_papi_system_info.num_cntrs)-1));
  
  /* Perform any initialization of the control bits */

  if (this_state->selector == 0)
    init_config(this_state);
  
  /* Turn on the bits for this counter */
#ifdef PERFCTR20
  set_hwcntr_codes(selector,codes,&this_state->counter_cmd.cpu_control);
#else
  set_hwcntr_codes(selector,codes,&this_state->counter_cmd);
#endif

  /* Update the new counter select field */

  this_state->selector |= selector;

  /* Inform the upper level that the software event 'index' 
     consists of the following information. */

  out->code = EventCode;
  out->selector = selector;

  return(PAPI_OK);
}

int _papi_hwd_rem_event(hwd_control_state_t *this_state, EventInfo_t *in)
{
  int used;

  /* Find out which counters used. */
  
  used = in->selector;

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
#ifdef PERFCTR20
      memcpy(&current_state->counter_cmd, &this_state->counter_cmd,
	     sizeof this_state->counter_cmd);
#else
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(struct perfctr_control));
#endif

      /* Stop the current context 

      retval = perf(PERF_RESET_COUNTERS, 0, 0);
      if (retval) 
	return(PAPI_ESYS);  */
      
      /* (Re)start the counters */

#ifdef DEBUG
      dump_cmd("_papi_hwd_merge (this)",&this_state->counter_cmd);
      dump_cmd("_papi_hwd_merge (current)",&current_state->counter_cmd);
#endif

      /* Should anything of the below be added here ??? - NS */
      /* zero->hw_start[i-1] = 
	 !(this_state->counter_cmd.cpu_control.evntsel[i-1] & PERF_INT_ENABLE)
	 ? 0 : this_state->counter_cmd.cpu_control.ireset[i-1]; */
      /* ESI->hw_start[i-1] = zero->hw_start[i-1]; */
      /* zero->multistart.SharedDepth[i-1] = 0;    */

      if (vperfctr_control(current_state->self, &current_state->counter_cmd) < 0)
	{
	  DBG((stderr,"Setting counters failed: SYSERR %d: %s",errno,strerror(errno)));
	  return(PAPI_ESYS);
	}

    }

  /* If we ARE nested, 
     carefully merge the global counter structure with the current eventset */
  else
    {
      int tmp, hwcntrs_in_both, hwcntrs_in_all, hwcntr;

      DBG((stderr,"Nested event set\n"));
      /* Stop the current context 

      retval = perf(PERF_STOP, 0, 0);
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
	      if (!(counter_event_shared(&this_state->counter_cmd, &current_state->counter_cmd, i-1)))
		return(PAPI_ECNFLCT);
	    }
	  else if (!(counter_event_compat(&this_state->counter_cmd, &current_state->counter_cmd, i-1)))
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
	      counter_event_copy(&this_state->counter_cmd, &current_state->counter_cmd, i-1);
	      ESI->hw_start[i-1] = zero->hw_start[i-1] = 
		!(this_state->counter_cmd.cpu_control.evntsel[i-1] & PERF_INT_ENABLE) ?
		  0 : this_state->counter_cmd.cpu_control.ireset[i-1];
	      zero->multistart.SharedDepth[i-1] = 0; 
	    }
	}
    }

  /* Set up the new merged control structure */
  
#ifdef DEBUG
  dump_cmd(__FUNCTION__,&current_state->counter_cmd);
#endif
      
  /* Stop the current context 

  retval = perf(PERF_RESET_COUNTERS, 0, 0);
  if (retval) 
    return(PAPI_ESYS); */

  /* (Re)start the counters */
  
  if (vperfctr_control(current_state->self, &current_state->counter_cmd) < 0) 
    {
      DBG((stderr,"Calling vperfctr_control: SYSERR %d: %s",errno,strerror(errno)));
      return(PAPI_ESYS);
    }
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
#ifdef PAPI_PERFCTR_INTR_SUPPORT
  hwd_control_state_t *machdep = zero->machdep;
  int *pmc_map = machdep->counter_cmd.cpu_control.pmc_map;
  long long correct_pmc_order[PERF_MAX_COUNTERS];
#endif
  long long correct_hw_order[PERF_MAX_COUNTERS];

  DBG((stderr,"Start\n"));
  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

#ifdef PAPI_PERFCTR_INTR_SUPPORT
  retval = correct_local_hwcounters(zero, ESI, correct_pmc_order);
  if (retval)
    return(retval);
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    correct_hw_order[pmc_map[i]] = correct_pmc_order[i];
#else
  retval = correct_local_hwcounters(zero, ESI, correct_hw_order);
  if (retval)
    return(retval);
#endif

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
	  events[j] = correct_hw_order[shift_cnt];
	}

      /* If this is a derived event */

      else 
	events[j] = handle_derived(&ESI->EventInfoArray[i], correct_hw_order);

      /* Early exit! */

      if (++j == ESI->NumberOfEvents)
	{
	  DBG((stderr,"Done\n"));
	  return(PAPI_OK);
	}
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

/* Called once per process. */

int _papi_hwd_shutdown_global(void)
{
#ifndef PERFCTR18
  perfctr_dev_close(dev);
#endif
  preset_map = NULL;
  return(PAPI_OK);
}

/* This routine is for shutting down threads, including the
   master thread. */

int _papi_hwd_shutdown(EventSetInfo *zero)
{
  hwd_control_state_t *machdep = zero->machdep;
  vperfctr_unlink(machdep->self);
  vperfctr_close(machdep->self);
  memset(machdep,0x0,sizeof(hwd_control_state_t));
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

void _papi_hwd_dispatch_timer(int signal, siginfo_t* info, void * tmp)
{
  struct ucontext *uc;
  struct sigcontext *mc;
  struct ucontext realc;

  uc = (struct ucontext *) tmp;
  realc = *uc;
  mc = &uc->uc_mcontext;
  DBG((stderr,"Start at 0x%lx\n",mc->eip));
  _papi_hwi_dispatch_overflow_signal(mc); 

  /* We are done, resume interrupting counters */
#ifdef PAPI_PERFCTR_INTR_SUPPORT
  if(_papi_system_info.supports_hw_overflow)
    {
      EventSetInfo *master;
      hwd_control_state_t *machdep;
      struct vperfctr* dev;

      master = _papi_hwi_lookup_in_master_list();
      if(master==NULL)
	{
	  fprintf(stderr,"%s():%d: master event lookup failure! abort()\n",
		  __FUNCTION__,__LINE__);
	  abort();
	}
      machdep =  master->machdep;
      dev = machdep->self;
      /* This is currently disabled since the restart of the counter */
      /* is made in update_global_counters out of unknown reasons    */
      /* if(vperfctr_isrun(machdep->self))                           */
      /*   if(vperfctr_iresume(machdep->self)<0)                     */
      /*     {                                                       */
      /*       perror("vperfctr_iresume");                           */
      /*       abort();                                              */
      /*     }                                                       */
    }
#endif
  DBG((stderr,"Finished at 0x%lx\n",mc->eip));
}

static void swap_pmc_map_events(struct vperfctr_control *contr,int cntr1,int cntr2)
{
  unsigned int ui; int si;

  /* In the case a user wants to interrupt on a counter in an evntsel
     that is not among the last events, we need to move the perfctr 
     virtual events around to make it last. This function swaps two
     perfctr events */

  ui=contr->cpu_control.pmc_map[cntr1];
  contr->cpu_control.pmc_map[cntr1]=contr->cpu_control.pmc_map[cntr2];
  contr->cpu_control.pmc_map[cntr2] = ui;

  ui=contr->cpu_control.evntsel[cntr1];
  contr->cpu_control.evntsel[cntr1]=contr->cpu_control.evntsel[cntr2];
  contr->cpu_control.evntsel[cntr2] = ui;

  ui=contr->cpu_control.evntsel_aux[cntr1];
  contr->cpu_control.evntsel_aux[cntr1]=contr->cpu_control.evntsel_aux[cntr2];
  contr->cpu_control.evntsel_aux[cntr2] = ui;

  si=contr->cpu_control.ireset[cntr1];
  contr->cpu_control.ireset[cntr1]=contr->cpu_control.ireset[cntr2];
  contr->cpu_control.ireset[cntr2] = si;
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
#ifdef PAPI_PERFCTR_INTR_SUPPORT
  extern int _papi_hwi_using_signal;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  struct vperfctr_control *contr = &this_state->counter_cmd;
  int i, ncntrs, nricntrs = 0, nracntrs, cntr, cntr2, retval=0;
  unsigned int selector;

#ifdef DEBUG
  DBG((stderr,"overflow_option->EventIndex=%d\n",
       overflow_option->EventIndex));
  dump_cmd("_papi_hwd_set_overflow",contr);
#endif 
  if( overflow_option->threshold != 0)  /* Set an overflow threshold */
    {
      struct sigaction sa;
      int err;

      /* Return error if installed signal is set earlier (!=SIG_DFL) and
	 it was not set to the PAPI overflow handler */
      /* The following code is commented out because many C libraries
	 replace the signal handler when one links with threads. The
	 name of this signal handler is not exported. So there really
	 is NO WAY to check if the user has installed a signal. */
      /*
      void *tmp;
      tmp = (void *)signal(PAPI_SIGNAL, SIG_IGN);
      if ((tmp != (void *)SIG_DFL) && (tmp != (void *)_papi_hwd_dispatch_timer))
	return(PAPI_EMISC);
      */

      memset(&sa, 0, sizeof sa);
      sa.sa_sigaction = _papi_hwd_dispatch_timer;
      sa.sa_flags = SA_SIGINFO;
      if((err = sigaction(PAPI_SIGNAL, &sa, NULL)) < 0)
	{
	  DBG((stderr,"Setting sigaction failed: SYSERR %d: %s",errno,strerror(errno)));
	  return(PAPI_ESYS);
	}

      /* The correct event to overflow is overflow_option->EventIndex */
      ncntrs=_papi_system_info.num_cntrs;
      selector = ESI->EventInfoArray[overflow_option->EventIndex].selector;
      DBG((stderr,"selector id is %d.\n",selector));
      i=ffs(selector)-1;
      if(i>=ncntrs)
	{
	  DBG((stderr,"Selector id (0x%x) larger than ncntrs (%d)\n",selector,ncntrs));
	  return PAPI_EINVAL;
	}
      contr->cpu_control.ireset[i] = -overflow_option->threshold;
      contr->cpu_control.evntsel[i] |= PERF_INT_ENABLE;
      nricntrs=++contr->cpu_control.nrictrs;
      nracntrs=--contr->cpu_control.nractrs;
      contr->si_signo = PAPI_SIGNAL;

      /* perfctr 2.x requires the interrupting counters to be placed last
	 in evntsel, swap events that do not fulfill this criterion. This
	 will yield a non-monotonic pmc_map array */
      for(i=nricntrs;i>0;i--)
	{
	  cntr = nracntrs + i - 1;
	  if( !(contr->cpu_control.evntsel[cntr] & PERF_INT_ENABLE))
	    { /* A non-interrupting counter was found among the icounters
		 Locate an interrupting counter in the acounters and swap */
	      for(cntr2=0;cntr2<nracntrs;cntr2++)
		{
		  if( (contr->cpu_control.evntsel[cntr2] & PERF_INT_ENABLE))
		    break;
		}
	      if(cntr2==nracntrs)
		{
		  DBG((stderr,"No icounter to swap with!\n"));
		  return(PAPI_EMISC);
		}
	      swap_pmc_map_events(contr,cntr,cntr2);
	    }
	}

      PAPI_lock();
      _papi_hwi_using_signal++;
      PAPI_unlock();

#ifdef DEBUG
      DBG((stderr,"Modified event set\n"));
      dump_cmd("_papi_hwd_set_overflow",contr);
#endif 
    }
  else   /* Disable overflow */
    {
      /* The correct event to overflow is overflow_option->EventIndex */
      ncntrs=_papi_system_info.num_cntrs;
      for(i=0;i<ncntrs;i++) 
	if(contr->cpu_control.evntsel[i] & PERF_INT_ENABLE)
	  {
	    contr->cpu_control.ireset[i] = 0;
	    contr->cpu_control.evntsel[i] &= (~PERF_INT_ENABLE);
	    nricntrs=--contr->cpu_control.nrictrs;
	    nracntrs=++contr->cpu_control.nractrs;
	    contr->si_signo = 0;
	  }
      /* The current implementation only supports one interrupting counter */
      if(nricntrs)
	{
	  fprintf(stderr,"%s %s\n","PAPI internal error.",
		  "Only one interrupting counter is supported!");
	  return(PAPI_ESBSTR);
	}

      /* perfctr 2.x requires the interrupting counters to be placed last
	 in evntsel, when the counter is non-interupting, move the order
	 back into the default monotonic pmc_map */
      for(cntr=0;cntr<ncntrs;cntr++)
	if(contr->cpu_control.pmc_map[cntr]!=cntr)
	  { /* This counter is out-of-order. Swap with the correct one*/
	    for(cntr2=cntr+1;cntr2<ncntrs;cntr2++)
	      if(contr->cpu_control.pmc_map[cntr2]==cntr) break;
	    if(cntr2==ncntrs)
	      {
		DBG((stderr,"No icounter to swap with!\n"));
		return(PAPI_EMISC);
	      }
	    swap_pmc_map_events(contr,cntr,cntr2);
	  }

#ifdef DEBUG
      DBG((stderr,"Modified event set\n"));
      dump_cmd(__FUNCTION__,contr);
#endif 

      PAPI_lock();
      _papi_hwi_using_signal--;
      if (_papi_hwi_using_signal == 0)
	{
	  if (sigaction(PAPI_SIGNAL, NULL, NULL) == -1)
	    retval = PAPI_ESYS;
	}
      PAPI_unlock();
    }

  DBG((stderr,"%s (%s): Hardware overflow is still experimental.\n",
	  __FILE__,__FUNCTION__));
  DBG((stderr,"End of call. Exit code: %d\n",retval));
  return(retval);
#else
  /* This function is not used and shouldn't be called. */
  abort();
  return(PAPI_ESBSTR);
#endif
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
  return(PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(EventSetInfo *ESI, EventSetInfo *master)
{
  /* This function is not used and shouldn't be called. */

  abort();
  return(PAPI_ESBSTR);
}


void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->eip;

  return(location);
}

static volatile unsigned int lock = 0;
static volatile unsigned int *lock_addr = &lock;

void _papi_hwd_lock_init(void)
{
}

void _papi_hwd_lock(void)
{
  while (1)
    {
      if (test_and_set_bit(0,lock_addr)) /* from asm/bitops.h */
	{
	  mb(); /* memory barrier/flush from asm/bitops.h */
	  return;
	}
    }
}

void _papi_hwd_unlock(void)
{
  clear_bit(0, lock_addr); /* from asm/bitops.h */
  mb(); /* memory barrier/flush from asm/bitops.h */
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
				 (caddr_t)&_init,
				 (caddr_t)&_etext,
				 (caddr_t)&_etext+1,
				 (caddr_t)&_edata,
				 (caddr_t)NULL,
				 (caddr_t)NULL,
				 "LD_PRELOAD", /* How to preload libs */
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
			        1,  /* supports child inheritance */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        1,  /* We can use the virt_usec call */
			        1,  /* We can use the virt_cyc call */
			        0,  /* HW read resets the counters */
			        sizeof(hwd_control_state_t), 
			        { 0, } };

