/* $Id$ */

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#include "linux-pentium.h"

_syscall3(int, perf, int, op, int, counter, int, event); 

/* First entry is mask, counter code 1, counter code 2, and TSC. 
A high bit in the mask entry means it is an OR mask, not an
and mask. This means that the same even is available on either
counter. */

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                {0,-1,-1,-1},		// L1 D-Cache misses 
                {0x3,0x28,0xC0,-1},	// L1 I-Cache misses 
		{0x3,0x24,0x2E,-1},	// L2 Cache misses
		{0x3,0x24,0x2E,-1},	// ditto
		{0,-1,-1,-1},		// L3 misses
		{0,-1,-1,-1},		// ditto
		{0,-1,-1,-1},		// 6	**unused preset map elements**
		{0,-1,-1,-1},		// 7
		{0,-1,-1,-1},		// 8
		{0,-1,-1,-1},		// 9
		{0,-1,-1,-1}, 		// Req. access to shared cache line
		{0,-1,-1,-1}, 		// Req. access to clean cache line
		{0,-1,-1,-1}, 		// Cache Line Invalidation
                {0,-1,-1,-1},		// 13
                {0,-1,-1,-1},		// 14
                {0,-1,-1,-1},		// 15
                {0,-1,-1,-1},		// 16
                {0,-1,-1,-1},		// 17
                {0,-1,-1,-1},		// 18
                {0,-1,-1,-1},		// 19
		{0,-1,-1,-1}, 		// D-TLB misses
		{0x13, 0x81,0x81,-1},	// I-TLB misses
                {0,-1,-1,-1},	   	// Total TLB misses
                {0,-1,-1,-1},		// 23
                {0,-1,-1,-1},			// 24
                {0,-1,-1,-1},			// 25
                {0,-1,-1,-1},			// 26
                {0,-1,-1,-1},			// 27
                {0,-1,-1,-1},			// 28
                {0,-1,-1,-1},			// 29
		{0,-1,-1,-1 },			// TLB shootdowns
                {0,-1,-1,-1},			// 31
                {0,-1,-1,-1},			// 32
                {0,-1,-1,-1},			// 33
                {0,-1,-1,-1},	/* Cycles stalled waiting for memory */
                {0,-1,-1,-1},   /* Cycles stalled waiting for memory read */
                {0,-1,-1,-1},   /* Cycles stalled waiting for memory write */
                {0,-1,-1,-1},	/* Cycles no instructions issued */
                {0,-1,-1,-1},	/* Cycles max instructions issued */
                {0,-1,-1,-1},			// 39
                {0,-1,-1,-1},			// 40
                {0,-1,-1,-1},			// 41
		{0x13,0xC9,0xC9,-1},	// Uncond. branches executed
		{0x13,0xC5,0xC5,-1},	// Cond. Branch inst. executed
		{0x13,0xC9,0xC9,-1},	// Cond. Branch inst. taken
		{0x13,0xE4,0xE4,-1},	// Cond. Branch inst. not taken
                {0,-1,-1,-1},	// Cond. branch inst. mispred.
                {0,-1,-1,-1},			// 47
                {0,-1,-1,-1},			// 48
                {0,-1,-1,-1},			// 49
		{0x13,0xC0,0xC0,-1},		// Total inst. executed
		{0x13,0xC0,0x10,-1},		// Integer inst. executed
		{0x1,0x10,-1,-1},		// Floating Pt. inst. executed
		{0,-1,-1,-1},			// Loads executed
		{0,-1,-1,-1},			// Stores executed
		{0x13,0xC4,0xC4,-1},		// Branch inst. executed
		{0,-1,-1,-1},			// Vector/SIMD inst. executed 
		{0x5,0x10,-1,1},		// FLOPS
                {0,-1,-1,-1},			// 58
                {0,-1,-1,-1},			// 59
		{0x4,-1,-1,1},			// Total cycles
		{0x17,0xC0,0xC0,1},		// MIPS
                {0,-1,-1,-1},			// 62
                {0,-1,-1,-1},			// 63
             };

/* Globals are BAD */

static EventSetInfo *_papi_global = NULL;
static hwd_control_state_t *_papi_global_machdep = NULL;

/* Utility functions */

static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  ptr->start_conf[arg1] = arg2;
}

static void init_config(hwd_control_state_t *ptr)
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

  ptr->mask = 0;
  ptr->start_conf[0] |= def_mode | PERF_ENABLE;
  ptr->start_conf[1] |= def_mode;
  ptr->start_conf[2] = 0;
  ptr->domain = def_mode;
}

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  if (a->start_conf[cntr] == b->start_conf[cntr])
    return(1);

  return(0);
}

static int update_counters(unsigned long long events[])
{
  int ret;

  ret = perf(PERF_FASTREAD, (int)events, 0);
  if (ret != 0)
    return(PAPI_ESBSTR);
  
  DBG((stderr,"update_counters() events[0] = %llu\n",events[0]));
  DBG((stderr,"update_counters() events[1] = %llu\n",events[1]));
  DBG((stderr,"update_counters() events[2] = %llu\n",events[2])); 

  return(PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static int set_inherit(int arg)
{
  int r;

  if (arg)
    arg = 1;

  r = perf(PERF_SET_OPT, PERF_DO_CHILDREN, arg);
  if (r != 0)
    return(PAPI_ESYS);

  return(PAPI_OK);
}

static int set_default_domain(int domain)
{
  int def_mode;

  switch (domain)
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
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int set_eventset_domain(EventSetInfo *ESI, int domain)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int mask = PERF_USR|PERF_OS;

  switch (domain)
    {
    case PAPI_DOM_USER:
      this_state->start_conf[0] |= mask;
      this_state->start_conf[0] ^= mask;
      this_state->start_conf[1] |= mask;
      this_state->start_conf[1] ^= mask;
      this_state->start_conf[0] |= PERF_USR | PERF_ENABLE;
      this_state->start_conf[1] |= PERF_USR;
      this_state->domain = PERF_USR;
      break;
    case PAPI_DOM_KERNEL:
      this_state->start_conf[0] |= mask;
      this_state->start_conf[0] ^= mask;
      this_state->start_conf[1] |= mask;
      this_state->start_conf[1] ^= mask;
      this_state->start_conf[0] |= PERF_OS | PERF_ENABLE;
      this_state->start_conf[1] |= PERF_OS;
      this_state->domain = PERF_OS;
      break;
    case PAPI_DOM_ALL:
      this_state->start_conf[0] |= mask;
      this_state->start_conf[0] ^= mask;
      this_state->start_conf[1] |= mask;
      this_state->start_conf[1] ^= mask;
      this_state->start_conf[0] |= PERF_USR | PERF_OS | PERF_ENABLE;
      this_state->start_conf[1] |= PERF_USR | PERF_OS;
      this_state->domain = PERF_USR | PERF_OS;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Fill in what we can of the papi_system_info. */
  
  unsigned long stamp;

  stamp = get_cycles();
  sleep(1);
  stamp = (get_cycles() - stamp)/1000000;

  _papi_system_info.mhz = stamp;
  _papi_system_info.ncpu = NR_CPUS;

  /* As long as MHZ is close, we are fine. We only need it for brain-dead
     kernel modules. Which mine most definitely is not! */

  DBG((stderr,"Found %d CPUs at %d Mhz.\n",_papi_system_info.ncpu,_papi_system_info.mhz));

  /* Hook up our static variables. */

  _papi_global = zero;
  _papi_global_machdep = zero->machdep;

  /* Initialize our global machdep. */

  init_config(_papi_global_machdep);
  return(PAPI_OK);
}

/* Do not ever use ESI->NumberOfCounters in here. */

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int mask;
  int counter_code1;
  int counter_code2;
  int tsc_code;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 

      mask = preset_map[event].mask;
      if (mask == 0)
	return(PAPI_ENOEVNT);

      counter_code1 = preset_map[event].counter_code1;
      counter_code2 = preset_map[event].counter_code2;
      tsc_code = preset_map[event].sp_code;

      /* Now figure out the mask based on George's 
	 number encoding. Ok, I changed it. The rule is
         a high bit indicates events that is events 
         that live on mutliple counters. If a compound event, like
         MIPS has the high bit set, obviously it needs the special
         purpose counter. Low bits only mean it is a compound event. */
      
      if (mask & 0x10) /* Same events live on multiple counters */
	{
	  int avail = 0;

	  /* Which counters are available? */

	  mask &= 0x7;
	  avail = mask & ~this_state->mask;

	  /* Pick which counter based on what's available */

	  if (mask & 0x2 & avail) 
	    mask = 0x2;
	  else if (mask & 0x1 & avail) /* Counter 1 is needed and available */
	    mask = 0x1;          
	  else if (mask & 0x4 & avail)
	    mask = 0x4;
	  else
	    return(PAPI_ECNFLCT);
	}    
    }
  else
    {
      /* Support for native events here. */

      counter_code1 = event & 0xff;
      counter_code2 = event >> 8 & 0xff;
      tsc_code = event >> 16 & 0xff;
      mask = event >> 24 & 0x7;

      /* There must be only one event for custom encodings */

      if ((mask != 0x1) && (mask != 0x2) && (mask != 0x4))
	return(PAPI_EINVAL);
    }

  /* Lower three bits tell us what counters we need */

  assert(this_state->mask <= 0x7);
  
  if (this_state->mask & mask)
    return(PAPI_ECNFLCT);
  
  if (this_state->mask == 0)
    init_config(this_state);
  
  if (mask & 0x1)
    {
      set_config(this_state,0,this_state->domain | counter_code1 | PERF_ENABLE);
      this_state->mask |= 0x1;
    }
  if (mask & 0x2)
    {
      set_config(this_state,1,this_state->domain | counter_code2);
      this_state->mask |= 0x2;
    }
  if (mask & 0x4)
    {
      set_config(this_state,2,tsc_code);
      this_state->mask |= 0x4;
    }
  
  /* Inform the upper level that software event 'index' consists of the
     following hardware counters. Mask is 0x0 through 0x7. */

  ESI->EventSelectArray[index] = mask;

  return(PAPI_OK);
}

int _papi_hwd_rem_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int mask;
  int counter_code1;
  int counter_code2;
  int tsc_code;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 

      mask = preset_map[event].mask;
      if (mask == 0)
	return(PAPI_ENOEVNT);

      /* If it is an event that can live on multiple
	 counters, then we must find out exacly which
	 one we selected. */

      if (mask & 0x10)
	mask = ESI->EventSelectArray[index];

      counter_code1 = preset_map[event].counter_code1;
      counter_code2 = preset_map[event].counter_code2;
      tsc_code = preset_map[event].sp_code;
    }
  else
    {
      counter_code1 = event & 0xff;
      counter_code2 = event >> 8 & 0xff;
      tsc_code = event >> 16 & 0xff;
      mask = event >> 24 & 0x7;

      /* There must be only one event for custom encodings */

      if ((mask != 0x1) && (mask != 0x2) && (mask != 0x4))
	return(PAPI_EINVAL);
    }

  /* Lower three bits tell us what counters we need */

  assert(this_state->mask <= 0x7);
  
  /* Check if we are removing something that's not preset */

  if ((this_state->mask & mask) != mask)
    return(PAPI_EINVAL);

  /* Clear out counters that are part of this event. */

  if (mask & 0x1)
    {
      set_config(this_state,0,PERF_ENABLE | this_state->domain);
      this_state->mask ^= 0x1;
    }
  if (mask & 0x2)
    {
      set_config(this_state,1,this_state->domain);
      this_state->mask ^= 0x2;
    }
  if (mask & 0x4)
    {
      set_config(this_state,2,0x0);
      this_state->mask ^= 0x4;
    }
  
  /* Inform the upper level that software event 'index' consists of 
     no hardware counters. Mask is 0x0 through 0x7. */

  ESI->EventSelectArray[index] = 0;

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int retval, one_shared = 0, two_shared = 0, three_shared = 0;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  
  /* Short circuit this stuff if it's not necessary */

  if (zero->multistart.num_runners)
    {
      /* We need the latest values of the shared counters so we can
	 handle nested, shared, starts. If nothing is shared, 
         we'll notice later on and this won't hurt us. */

      retval = _papi_hwd_read(ESI, zero, ESI->start);
      if (retval < PAPI_OK)
	return(retval);

      /* Check for shared events that require no machdep modification */
      
      if (this_state->mask & current_state->mask & 0x1)
	{
	  if (counter_shared(this_state, current_state, 0))
	    {
	      zero->multistart.SharedDepth[0] ++; 
	      one_shared = 1;
	    }
	  else
	    return(PAPI_ECNFLCT);
	}
      
      if (this_state->mask & current_state->mask & 0x2)
	{
	  if (counter_shared(this_state, current_state, 1))
	    {
	      zero->multistart.SharedDepth[1] ++; 
	      two_shared = 1;
	    }
	  else
	    return(PAPI_ECNFLCT);
	}
      
      if (this_state->mask & current_state->mask & 0x4)
	{
	  if (counter_shared(this_state, current_state, 2))
	    {
	      zero->multistart.SharedDepth[2] ++; 
	      three_shared = 1;
	    }
	  else
	    return(PAPI_ECNFLCT);
	}
    }

  /* Merge the unshared configuration registers. */

  if ((this_state->mask & 0x1) && (!one_shared))
    {
      zero->multistart.SharedDepth[0] ++; 
      current_state->start_conf[0] = this_state->start_conf[0];
      current_state->mask ^= 0x1;
    }
  if ((this_state->mask & 0x2) && (!two_shared))
    {
      zero->multistart.SharedDepth[1] ++; 
      current_state->start_conf[1] = this_state->start_conf[1];
      current_state->mask ^= 0x2;
    }
  if ((this_state->mask & 0x4) && (!three_shared))
    {
      zero->multistart.SharedDepth[2] ++; 
      current_state->start_conf[2] = this_state->start_conf[2];
      current_state->mask ^= 0x4;
    }

  /* Start the counters with the new merged event set machdep structure */

  retval = perf(PERF_FASTCONFIG, (int)current_state->start_conf, (int)NULL);
  if (retval) 
    return(PAPI_EBUG);

  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = start_overflow_timer(ESI);
      if (retval < PAPI_OK)
	return(PAPI_EBUG);
    }

  return(PAPI_OK);
} 

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  retval = _papi_hwd_read(ESI, zero, ESI->stop);
  if (retval < PAPI_OK)
    return(retval);

  retval = perf(PERF_STOP, 0, 0);
  if (retval) 
    return(PAPI_EBUG); 
  
  /* With x86, the enable bit is in counter control 0. */

  if (this_state->mask & 0x1)
    {
      zero->multistart.SharedDepth[0] --;
      if (zero->multistart.SharedDepth[0] == 0)
	{ 
	  current_state->mask ^= 0x1;
	  current_state->start_conf[0] = this_state->domain | PERF_ENABLE;

	  /* This is necessary to zero out the appropriate counter value in the hardware
	     here. */

	  retval = perf(PERF_SET_CONFIG, 0, (int)current_state->start_conf[0]);
	  if (retval) 
	    return(PAPI_EBUG);
	}
    }

  /* With x86, mode bits are in both counter control 0 and 1. */

  if (this_state->mask & 0x2)
    {
      zero->multistart.SharedDepth[1] --;
      if (zero->multistart.SharedDepth[1] == 0)
	{ 
	  current_state->mask ^= 0x2;
	  current_state->start_conf[1] = this_state->domain;

	  /* This is necessary to zero out the appropriate counter value in the hardware
	     here. */

	  retval = perf(PERF_SET_CONFIG, 1, (int)current_state->start_conf[1]);
	  if (retval) 
	    return(PAPI_EBUG);
	}
    }

  /* With my kernel extension, the last counter is just a per processor
     cycle count. */

  if (this_state->mask & 0x4)
    {
      zero->multistart.SharedDepth[2] --;
      if (zero->multistart.SharedDepth[2] == 0)
	{ 
	  current_state->mask ^= 0x4;
	  current_state->start_conf[2] = 0;
	  
	  /* This is necessary to zero out the appropriate counter value in the hardware
	     here. */

	  retval = perf(PERF_SET_CONFIG, 2, (int)current_state->start_conf[2]);
	  if (retval) 
	    return(PAPI_EBUG);
	}
    }

  /* Start the counters with the new unmerged event set machdep structure */

  retval = perf(PERF_FASTCONFIG, (int)current_state->start_conf, (int)NULL);
  if (retval) 
    return(PAPI_EBUG);

  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = stop_overflow_timer(ESI);
      if (retval < PAPI_OK)
	return(PAPI_EBUG);
    }

  return(PAPI_OK);
}

int _papi_hwd_reset(EventSetInfo *ESI)
{
  int ret;
  
  ret = perf(PERF_RESET_COUNTERS, 0, 0); 
  if (ret == 0)
    return(PAPI_OK);

  return(PAPI_ESBSTR);
}

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, unsigned long long events[])
{
  int retval, selector, j = 0, i;
  unsigned long long last_read[PERF_COUNTERS];

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
    case PAPI_SET_INHERIT:
      return(set_inherit(option->inherit.inherit));
    case PAPI_SET_DEFDOM:
      return(set_default_domain(option->domain.domain));
    case PAPI_SET_DOMAIN:
      return(set_eventset_domain(option->domain.ESI,option->domain.domain));
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

  if (overflow_option->threshold == 0)
    this_state->timer_ms = 0;
  else
    this_state->timer_ms = 1; /* Millisecond intervals are the only way to go */

  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  return(PAPI_OK);
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
			       2,   /*  num_gp_cntrs */
			       0,   /*  grouped_counters */
			       1,   /*  num_sp_cntrs */
			       -1,  /*  total_presets */
			       -1,  /*  total_events */
			       1,   /*  needs overflow emulation */
			       1,   /*  needs profile emulation */
			       0,   /*  needs 64 bit virtual counters */
			       1,   /*  supports child inheritance option */
			       0,   /*  can attach to another process */
			       0,   /*  read resets the counters */
			       PAPI_DOM_USER, /* default domain */
			       PAPI_GRN_THR,  /* default granularity */
			       sizeof(hwd_control_state_t), 
			       NULL };

