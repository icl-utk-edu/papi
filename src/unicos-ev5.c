/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#include "unicos-ev5.h"

/* First entry is counter code 1, counter code 2 and counter code 3.
   Then is the mask. There are no derived metrics for the T3E. */

static hwd_control_state_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                { -1, -1, 0x4, 0x4 },   /* L1 D-Cache misses  */
                { -1, -1, 0x6, 0x4 },	/* L1 I-Cache misses  */
		{ -1, 0xf, 0xf, 0x6 },	/* L2 Cache misses */
		{ -1, -1, -1, 0x6 },	/* ditto */
		{ -1, -1, -1, 0 },	/* L3 misses */
		{ -1, -1, -1, 0 },	/* ditto */
		{ -1, -1, -1, 0 },	/* 6	**unused preset map elements** */
		{ -1, -1, -1, 0 },	/* 7 */
		{ -1, -1, -1, 0 },	/* 8 */
		{ -1, -1, -1, 0 },	/* 9 */
		{ -1, -1, -1, 0 }, 	/* Req. access to shared cache line */
		{ -1, -1, -1, 0 }, 	/* Req. access to clean cache line */
		{ -1, -1, -1, 0 }, 	/* Cache Line Invalidation */
                { -1, -1, -1, 0 },	/* 13 */
                { -1, -1, -1, 0 },	/* 14 */
                { -1, -1, -1, 0 },	/* 15 */
                { -1, -1, -1, 0 },	/* 16 */
                { -1, -1, -1, 0 },	/* 17 */
                { -1, -1, -1, 0 },	/* 18 */
                { -1, -1, -1, 0 },	/* 19 */
		{ -1, -1, 0x7, 0x4 },   /* D-TLB misses */
		{ -1, -1, 0x5, 0x4 },	/* I-TLB misses */
                { -1, -1, -1, 0 },	/* Total TLB misses */
                { -1, -1, -1, 0 },	/* 23 */
                { -1, -1, -1, 0 },	/* 24 */
                { -1, -1, -1, 0 },	/* 25 */
                { -1, -1, -1, 0 },	/* 26 */
                { -1, -1, -1, 0 },	/* 27 */
                { -1, -1, -1, 0 },	/* 28 */
                { -1, -1, -1, 0 },	/* 29 */
		{ -1, -1, -1, 0 },	/* TLB shootdowns */
                { -1, -1, -1, 0 },	/* 31 */
                { -1, -1, -1, 0 },	/* 32 */
                { -1, -1, -1, 0 },	/* 33 */
                { -1, -1, 0xa, 0x4 },	/* Cycles stalled waiting for memory */
                { -1, -1, -1, 0 },	/* Cycles stalled waiting for memory read */
                { -1, -1, -1, 0 },	/* Cycles stalled waiting for memory write */
                { -1, 0x0, 0xd, 0x2 },	/* Cycles no instructions issued */
                { -1, 0x7, -1, 0x2 },	/* Cycles max instructions issued */
                { -1, -1, -1, 0 },	/* 39 */
		{ -1, -1, -1, 0 },	/* 40 */
                { -1, -1, -1, 0 },	/* 41 */
		{ -1, -1, -1, 0 },	/* Uncond. branches executed */
		{ -1, 0x8, 0x3, 0x2 },	/* Cond. branch inst. executed*/
		{ -1, -1, -1, 0 },	/* Cond. branch inst. taken*/
		{ -1, -1, -1, 0 },	/* Cond. branch inst. not taken*/
		{ -1, -1, 0x3, 0x4 },	/* Cond. branch inst. mispred.*/
                { -1, -1, -1, 0 },	/* 47 */
                { -1, -1, -1, 0 },	/* 48 */
                { -1, -1, -1, 0 },	/* 49 */
		{ 0x1, -1, -1, 0x1 },	/* Total inst. executed */
		{ -1, 0x9, -1, 0x2 },	/* Integer inst. executed */
		{ -1, 0xa, -1, 0x2 },	/* Floating Pt. inst. executed */
		{ -1, 0xb, -1, 0x2 },	/* Loads executed */
		{ -1, 0xc, -1, 0x2 },	/* Stores executed */
		{ -1, 0x8, -1, 0x2 },	/* Branch inst. executed */
		{ -1, -1, -1, 0 },	/* Vector/SIMD inst. executed  */
		{ -1, -1, -1, 0 },	/* FLOPS */
                { -1, -1, -1, 0 },	/* 58 */
                { -1, -1, -1, 0 },	/* 59 */
		{ 0x0, -1, -1, 0x1 },	/* Total cycles */
		{ -1, -1, -1, 0 },	/* MIPS */
                { -1, -1, -1, 0 },	/* 62 */
                { -1, -1, -1, 0 },	/* 63 */ };

/* Global's ugh. */

/* Privilege bits */

static int kill_pal = 1, kill_user = 1, kill_kernel = 1;

/* Utility functions */

static int set_def_domain(int domain)
{
  switch (domain)
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
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int set_domain(EventSetInfo *ESI, int domain)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int ret;

  switch (domain)
    {
    case PAPI_DOM_USER:
      {
	this_state->pmctr.Kk = 0;
	this_state->pmctr.Kp = 0;
	this_state->pmctr.Ku = 1;
      }
      break;
    case PAPI_DOM_KERNEL:
      {
	this_state->pmctr.Kk = 1;
	this_state->pmctr.Kp = 0;
	this_state->pmctr.Ku = 0;
      }
      break;
    case PAPI_DOM_OTHER:
      {
	this_state->pmctr.Kk = 0;
	this_state->pmctr.Kp = 1;
	this_state->pmctr.Ku = 0;
      }
      break;
    case PAPI_DOM_ALL:
      {
	this_state->pmctr.Kk = 1;
	this_state->pmctr.Kp = 1;
	this_state->pmctr.Ku = 1;
      }
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int get_def_domain(void)
{
  int retval = 0;

  if (kill_kernel == 1)
    retval |= PAPI_DOM_KERNEL;
  if (kill_pal == 1)
    retval |= PAPI_DOM_OTHER;
  if (kill_user == 1)
    retval |= PAPI_DOM_USER;

  return(retval);
}

static int get_domain(EventSetInfo *ESI)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  int retval = 0;

  if (this_state->pmctr.Kk == 1)
    retval |= PAPI_DOM_KERNEL;
  if (this_state->pmctr.Kp == 1)
    retval |= PAPI_DOM_OTHER;
  if (this_state->pmctr.Ku == 1)
    retval |= PAPI_DOM_USER;

  return(retval);
}

static int getmhz(void)
{
  long sysconf(int request);
  float p;
  
  p = (float) sysconf(_SC_CRAY_CPCYCLE); /* Picoseconds */
  p = p * 1.0e-12; /* Convert to seconds */
  p = 1.0 / (p * 1000000.0); /* Convert to MHz */
  return((int)p);
}

static void init_pmctr(pmctr_t *pmctr)
{
  *(long *)pmctr = 0;

  pmctr.Kp = kill_pal;
  pmctr.Ku = kill_user;
  pmctr.Kk = kill_kernel;
  pmctr.CTL0 = CTL_OFF;
  pmctr.CTL1 = CTL_OFF;
  pmctr.CTL2 = CTL_OFF;
}

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  if ((cntr == 0) && (a->pmctr.CTL0 == b->pmctr.CTL0))
    return(1);
  else if ((cntr == 1) && (a->pmctr.CTL1 == b->pmctr.CTL1))
    return(1);
  else if ((cntr == 2) && (a->pmctr.CTL2 == b->pmctr.CTL2))
    return(1);

  return(0);
}

static int update_counters(int events[])
{
  pmctr_t pmctr;
  long pc_data[4];
  int ret;

  ret = _rdperf(pc_data);
  if (ret != 0)
    return(PAPI_ESBSTR);

  *(long *) &pmctr = pc_data[0];

  events[0] = (pc_data[1] << 16) + pmctr.CTR0;
  DBG((stderr,"update_counters() events[0] = %lld\n",events[0]));

  events[1] = (pc_data[2] << 16) + pmctr.CTR1;
  DBG((stderr,"update_counters() events[1] = %lld\n",events[1]));

  events[2] = (pc_data[3] << 14) + pmctr.CTR2;
  DBG((stderr,"update_counters() events[2] = %lld\n",events[2]));
  
  return(PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Fill in papi_mdi */
  
  /* At init time, the higher level library should always allocate and 
     reserve EventSet zero. */

  _papi_system_info.ncpu = 1;
  _papi_system_info.type = 0;
  _papi_system_info.cpu = 0;
  _papi_system_info.mhz = getmhz();

  /* As long as MHZ is close, we are fine. We only need it for brain-dead
     kernel modules. */

  DBG((stderr,"Found %d CPU's at %d Mhz.\n",_papi_system_info.ncpu,
       ,_papi_system_info.mhz));

  return(PAPI_OK);
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int sel0, sel1, sel2, mask;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 

      mask = preset_map[event].mask;
      if (mask == 0x0)
	return(PAPI_ENOEVNT);

      sel0 = preset_map[event].sel0;
      sel1 = preset_map[event].sel1;
      sel2 = preset_map[event].sel2;
    }
  else
    {
      sel0 = event & 0x1;
      sel1 = event >> 1 & 0xf;
      sel2 = event >> 5 & 0xf;
      mask = event >> 9 & 0x7;

      /* There must be at only one event for custom encodings */

      if ((mask != 0x1) && (mask != 0x2) && (mask != 0x4))
	return(PAPI_EINVAL);
    }

  /* Lower three bits tell us what counters we need */

  assert(this_state->mask <= 0x7);
  
  if (this_state->mask & mask)
    return(PAPI_ECNFLCT);

  if (this_state->mask == 0)
    init_pmctr(&this_state->pmctr);
 
  if (mask & 0x1)
    {
      this_state->pmctr.CTL0 = CTL_ON;
      this_state->pmctr.SEL0 = *sel0;
      this_state->mask |= 0x1;
    }
  if (mask & 0x2)
    {
      this_state->pmctr.CTL1 = CTL_ON;
      this_state->pmctr.SEL1 = *sel1;
      this_state->mask |= 0x2;
    }
  if (mask & 0x4)
    {
      this_state->pmctr.CTL2 = CTL_ON;
      this_state->pmctr.SEL2 = *sel2;
      this_state->mask |= 0x4;
    }
  
  /* Inform the upper level that software event 'index' consists of the
     following hardware counters. Mask is 0x0 through 0x7. */

  ESI->EventSelectArray[index] = mask;

  return(PAPI_OK);
}

int _papi_hwd_rem_event(EventSetInfo *ESI, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int sel0, sel1, sel2, mask;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 

      mask = preset_map[event].mask;
      if (mask == 0x0)
	return(PAPI_ENOEVNT);

      sel0 = preset_map[event].sel0;
      sel1 = preset_map[event].sel1;
      sel2 = preset_map[event].sel2;
    }
  else
    {
      sel0 = event & 0x1;
      sel1 = event >> 1 & 0xf;
      sel2 = event >> 5 & 0xf;
      mask = event >> 9 & 0x7;

      /* There must be at least one event */

      if (mask & 0x7 == 0)
	return(PAPI_EINVAL);
    }

  assert(this_state->mask & 0x7);
  
  /* Check if we are removing something that's not preset */

  if (this_state->mask & mask != mask)
    return(PAPI_EINVAL);
  
  if (mask & 0x1)
    {
      this_state->pmctr.CTL0 = CTL_OFF;
      this_state->mask ^= 0x1;
    }
  if (mask & 0x2)
    {
      this_state->pmctr.CTL1 = CTL_OFF;
      this_state->mask ^= 0x2;
    }
  if (mask & 0x4)
    {
      this_state->pmctr.CTL2 = CTL_OFF;
      this_state->mask ^= 0x4;
    }
  
  /* The following step is not necessary, as it is done by the higher level remove_event() */
  /* ESI->EventSelectArray[index] = PAPI_NULL */

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_start(EventSetInfo *ESI, EventSetInfo *zero)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  int retval;

  /* If we're the outermost start, we fill the global machdep.
     Good idea George. */

  if (EventSet->EventSetIndex != 0)
    {
      current_state->mask = this_state->mask;
      current_state->start_conf[0] = this_state->start_conf[0];
      current_state->start_conf[1] = this_state->start_conf[1];
      current_state->start_conf[2] = this_state->start_conf[2];
    }

  retval = _wrperf(this_state->pmctr,0,0,0);
  if (retval == 0)
    return(PAPI_OK);
  else
    return(PAPI_ESBSTR);
}

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{
  int retval, one_shared = 0, two_shared = 0, three_shared = 0;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  if (this_state->mask & current_state->mask & 0x1)
    {
      if (counter_shared(this_state, current_state, 0))
	{
	  zero->multistart.SharedDepth[0] ++; 
	  one_shared = 1;
	}
    }
  if (this_state->mask & current_state->mask & 0x2)
    {
      if (counter_shared(this_state, current_state, 1))
	{
	  zero->multistart.SharedDepth[1] ++; 
	  two_shared = 1;
	}
    }
  if (this_state->mask & current_state->mask & 0x4)
    {
      if (counter_shared(this_state, current_state, 1))
	{
	  zero->multistart.SharedDepth[2] ++; 
	  three_shared = 1;
	}
    }

  /* Merge the unshared configuration registers. */

  if ((this_state->mask & 0x1) && (!one_shared))
    {
      zero->multistart.SharedDepth[0] ++; 
      current_state->mask ^= 0x1;
      current_state->pmctr.CTL0 = this_state->pmctr.CTL0;
      current_state->pmctr.SEL0 = this_state->pmctr.SEL0;
    }
  if ((this_state->mask & 0x2) && (!two_shared))
    {
      zero->multistart.SharedDepth[1] ++; 
      current_state->mask ^= 0x2;
      current_state->pmctr.CTL1 = this_state->pmctr.CTL1;
      current_state->pmctr.SEL1 = this_state->pmctr.SEL1;
    }
  if ((this_state->mask & 0x4) && (!three_shared))
    {
      zero->multistart.SharedDepth[2] ++; 
      current_state->mask ^= 0x4;
      current_state->pmctr.CTL2 = this_state->pmctr.CTL2;
      current_state->pmctr.SEL2 = this_state->pmctr.SEL2;
    }

  return(_papi_hwd_start(zero));
} 

int _papi_hwd_stop(EventSetInfo *ESI, unsigned long long events[])
{ 
  int retval;

  retval = _papi_hwd_read(ESI,events);
  if (retval < PAPI_OK)
    return(retval);

  retval = perfmonctl(PERFCNT_EV5, PERFCNT_OFF);
  if (retval != 0)
    return(PAPI_ESBSTR);

  return(PAPI_OK);
}

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  retval=_papi_hwd_stop(ESI, ESI->stop);
  if (retval < PAPI_OK)
    return(retval);

  if (this_state->mask & 0x1)
    {
      zero->multistart.SharedDepth[0] --;
      if (zero->multistart.SharedDepth[0] == 0)
	{ 
	  current_state->mask ^= 0x1;
	  current_state->pmctr.CTL0 = this_state->pmctr.CTL0;
	  current_state->pmctr.SEL0 = this_state->pmctr.SEL0;
	}
    }

  if (this_state->mask & 0x2)
    {
      zero->multistart.SharedDepth[1] --;
      if (zero->multistart.SharedDepth[1] == 0)
	{ 
	  current_state->mask ^= 0x2;
	  current_state->pmctr.CTL1 = this_state->pmctr.CTL1;
	  current_state->pmctr.SEL1 = this_state->pmctr.SEL1;
	}
    }

  if (this_state->mask & 0x4)
    {
      zero->multistart.SharedDepth[2] --;
      if (zero->multistart.SharedDepth[2] == 0)
	{ 
	  current_state->mask ^= 0x4;
	  current_state->pmctr.CTL2 = this_state->pmctr.CTL2;
	  current_state->pmctr.SEL2 = this_state->pmctr.SEL2;
	}
    }

  return(_papi_hwd_start(zero));
}

int _papi_hwd_reset(EventSetInfo *ESI)
{
  int ret;

  ret = _wrperf(this_state->pmctr,0,0,0);
  if (ret == 0)
    return(PAPI_OK);

  return(PAPI_ESBSTR);
}

int _papi_hwd_read(EventSetInfo *ESI, unsigned long long events[])
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int i, j=0, k=0;
  int selector;
  int last_read[3];

  retval = update_counters(last_read);
  if (retval < PAPI_OK)
    return(retval);

  /* This routine distributes hardware counters to software counters in the
     order that they were added. Note that the higher level 
     EventSelectArray[i] entries may not be contiguous because the user
     has the right to remove an event. */

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      selector = ESI->EventSelectArray[i];
      
      assert(selector != 0);
      if (selector == PAPI_NULL)
	continue;

      DBG((stderr,"Event %d, mask is 0x%x\n",j,selector));

      switch (selector)
	{
	  case 0x1:
	    events[j] = last_read[0];
	    if (zero->multistart.SharedDepth[0] > 1)
	      events[j] -= ESI->start[j]; 
	  break;
	case 0x2:
	  events[j] = last_read[1];
	  if (zero->multistart.SharedDepth[1] > 1)
	    events[j] -= ESI->start[j]; 
	  break;
	case 0x4:
	  events[j] = last_read[2];
	  if (zero->multistart.SharedDepth[2] > 1)
	    events[j] -= ESI->start[j]; 
	  break;
	case 0x13:
	case 0x15:
	case 0x16:
	  /* Here we could calculate derived metrics based on
	     ESI->EventCodeArray[i]; But I'm lazy, so someone else
	     can do it. */
	default:
	  return(PAPI_EBUG);
	}

      /* Early exit! */

      if (++j == ESI->NumberOfCounters)
	return(PAPI_OK);
    }      

  /* Should never get here */
  
  return(PAPI_EBUG);
}

int _papi_hwd_ctl(int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_DEFDOM:
      return(set_def_domain(option->domain.domain));
    case PAPI_SET_DOMAIN:
      return(set_domain(option->domain.ESI,option->domain.domain));
    default:
      return(PAPI_ESBSTR);
    }
}

int _papi_hwd_write(EventSetInfo *ESI, unsigned long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(EventSetInfo *zero)
{
  return(PAPI_OK);
}

int _papi_hwd_query(int preset)
{
  if (preset & PRESET_MASK)
    { 
      preset ^= PRESET_MASK; 

      if (preset_map[preset].mask == 0)
	return(PAPI_EINVAL);
      else
	return(PAPI_OK);
    }
  return(PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  if ((overflow_option->handler == NULL) || (overflow_option->threshold == 0))
    this_state->timer_ms = 0;
  else
    this_state->timer_ms = 1; /* Millisecond intervals are the only way to go */
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *overflow_option)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  if ((overflow_option->handler == NULL) || (overflow_option->threshold == 0))
    this_state->timer_ms = 0;
  else
    this_state->timer_ms = 1; /* Millisecond intervals are the only way to go */
}

/* Machine info structure. -1 is unused. */

papi_mdi _papi_system_info = { "$Id$",
			       1.0, /*  version */
			       -1,  /*  ncpu */
			       -1,  /*  nnodes */
			       -1,  /*  type */
			       -1,  /*  cpu */
			       -1,  /*  mhz */
			       3,   /*  num_cntrs */
			       3,   /*  num_gp_cntrs */
			       0,   /*  grouped_counters */
			       0,   /*  num_sp_cntrs */
			       -1,  /*  total_presets */
			       -1,  /*  total_events */
			       1,   /*  needs overflow emulation */
			       1,   /*  needs profile emulation */
			       0,   /*  needs 64 bit virtual counters */
			       0,   /*  supports child inheritance option */
			       0,   /*  can attach to another process */
			       0,   /*  read resets the counters */
			       PAPI_DOM_USER, /* default domain */
			       PAPI_GRN_THR,  /* default granularity */
			       sizeof(hwd_control_state_t), 
			       NULL };
