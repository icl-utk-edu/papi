/* Null substrate. This file is intended as an example for 
   substrate designers. This file assumes the following
   hardware. 2 general purpose counters supporting ALL
   PAPI preset events on either counter. This substrate
   only supports overflow and profiling through emulation.
   Those hooks are included. The event values increase
   randomly by 0-255 any time update_counters is called. */

#include "any-null.h"

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                { 0x1,0,-1,},		/* L1 D-Cache misses  */
                { 0x1,1,-1,},		/* L1 I-Cache misses  */
		{ 0x1,2,-1,},		/* L2 Cache misses */
		{ 0x1,3,-1,},		/* ditto */
		{ 0x1,4,-1,},		/* L3 misses */
		{ 0x1,5,-1,},		/* ditto */
		{ 0x0,-1,-1,},		/* 6**unused preset map elements** */
		{ 0x0,-1,-1,},		/* 7 */
		{ 0x0,-1,-1,},		/* 8 */
		{ 0x0,-1,-1,},		/* 9 */
		{ 0x3,10,10,},	/* Req. access to shared cache line */
		{ 0x3,11,11,},	/* Req. access to clean cache line */
		{ 0x3,12,12,},	/* Cache Line Invalidation */
                { 0x0,-1,-1,},		/* 13 */
                { 0x0,-1,-1,},		/* 14 */
                { 0x0,-1,-1,},		/* 15 */
                { 0x0,-1,-1,},		/* 16 */
                { 0x0,-1,-1,},		/* 17 */
                { 0x0,-1,-1,},		/* 18 */
                { 0x0,-1,-1,},		/* 19 */
		{ 0x1,20,-1,},	/* D-TLB misses */
		{ 0x1,21,-1,},		/* I-TLB misses */
                { 0x0,-1,-1,},	 	/* 22 */
                { 0x0,-1,-1,},	        /* 23 */
                { 0x0,-1,-1,},		/* 24 */
                { 0x0,-1,-1,},		/* 25 */
                { 0x0,-1,-1,},		/* 26 */
                { 0x0,-1,-1,},		/* 27 */
                { 0x0,-1,-1,},		/* 28 */
                { 0x0,-1,-1,},		/* 29 */
		{ 0x1,30,-1,},		/* TLB shootdowns */
                { 0x0,-1,-1,},		/* 31 */
                { 0x0,-1,-1,},		/* 32 */
                { 0x0,-1,-1,},		/* 33 */
                { 0x0,-1,-1,},		/* 34 */
                { 0x0,-1,-1,},		/* 35 */
                { 0x0,-1,-1,},		/* 36 */
                { 0x0,-1,-1,},		/* 37 */
                { 0x0,-1,-1,},		/* 38 */
                { 0x0,-1,-1,},		/* 39 */
		{ 0x0,-1,-1,},		/* 40 */
                { 0x0,-1,-1,},		/* 41 */
		{ 0x2,-1,42,},		/* Uncond. branches executed */
		{ 0x2,-1,43,},		/* Cond. branch inst. executed.*/
		{ 0x2,-1,44,},		/* Cond. branch inst. taken*/
		{ 0x2,-1,45,},		/* Cond. branch inst. not taken*/
		{ 0x2,-1,46,},		/* Cond. branch inst. mispred.*/
                { 0x0,-1,-1,},		/* 47 */
                { 0x0,-1,-1,},		/* 48 */
                { 0x0,-1,-1,},		/* 49 */
		{ 0x1,50,-1,},		/* Total inst. executed */
		{ 0x1,51,-1,},		/* Integer inst. executed */
		{ 0x1,52,-1,},		/* Floating Pt. inst. executed */
		{ 0x1,53,-1,},		/* Loads executed */
		{ 0x2,-1,54,},		/* Stores executed */
		{ 0x2,-1,55,},		/* Branch inst. executed */
		{ 0x2,-1,56,},		/* Vector/SIMD inst. executed  */
		{ 0x3,57,57,},		/* FLOPS */
                { 0x0,-1,-1,},		/* 58 */
                { 0x0,-1,-1,},		/* 59 */
		{ 0x1,60,-1,},		/* Total cycles */
		{ 0x3,61,61,},		/* MIPS */
                { 0x0,-1,-1,},		/* 62 */
                { 0x0,-1,-1,},		/* 63 */
             };

/* Globals are BAD. */

static EventSetInfo *_papi_global = NULL;
static hwd_control_state_t *_papi_global_machdep = NULL;

/* Utility functions */

static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  if (arg1 == 1)
    ptr->cntrl.ev1 = arg2;
  else if (arg1 == 0)
    ptr->cntrl.ev0 = arg2;
  else
    abort();
  return;
}

static void init_config(hwd_control_state_t *ptr)
{
  ptr->cntrl.ev0 = 0;
  ptr->cntrl.ev1 = 0;
  ptr->cntrl.enable = 0;

  /* Copy the counting domain from the global machdep. The global 
     eventset and associated machdep hold the global presets
     among other things. Usually, this needs to be processed a bit
     to convert to the native setting. */

  ptr->cntrl.domain = _papi_system_info.default_domain;
}

/* This routine is used by merge and unmerge. */

static int counter_shared(hwd_control_state_t *a, 
			  hwd_control_state_t *b, int cntr)
{
  if (cntr == 0)
    {
      if (a->cntrl.ev0 == b->cntrl.ev0)
	return(1);
      else
	return(0);
    }
  else if (cntr == 1)
    {
      if (a->cntrl.ev1 == b->cntrl.ev1)
	return(1);
      else
	return(0);
    }
  else
    abort();
}

/* This routine is always called to get the counter values. This may be
   useful for some neat features too. This might need other arguments, 
   feel free to adjust to your needs. */

static int update_counters(unsigned long long events[])
{
  /* Fill it with some random numbers. */

  events[0] = time(NULL) & 0xff;
  events[1] = time(NULL) & 0xff;

  DBG((stderr,"update_counters() events[0] = %llu\n",events[0]));
  DBG((stderr,"update_counters() events[1] = %llu\n",events[1]));

  return(PAPI_OK);
}

static int set_default_domain(int domain)
{
  switch (domain)
    {
    case PAPI_DOM_USER:
      _papi_global_machdep->cntrl.domain = ANY_DOM_USER;
      break;
    case PAPI_DOM_KERNEL:
      _papi_global_machdep->cntrl.domain = ANY_DOM_KERNEL;
      break;
    case PAPI_DOM_OTHER:
      _papi_global_machdep->cntrl.domain = ANY_DOM_INTERRUPT;
      break;
    case PAPI_DOM_ALL:
      _papi_global_machdep->cntrl.domain = ANY_DOM_ALL;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

/* This function changes the domain of the eventset. Not always
   easy, of course for this substrate, everything's easy. */

static int set_eventset_domain(EventSetInfo *ESI, int domain)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  switch (domain)
    {
    case PAPI_DOM_USER:
      this_state->cntrl.domain = ANY_DOM_USER;
      break;
    case PAPI_DOM_KERNEL:
      this_state->cntrl.domain = ANY_DOM_KERNEL;
      break;
    case PAPI_DOM_OTHER:
      this_state->cntrl.domain = ANY_DOM_INTERRUPT;
      break;
    case PAPI_DOM_ALL:
      this_state->cntrl.domain = ANY_DOM_ALL;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static int set_inherit(int arg)
{
  return(PAPI_ESBSTR);
}

static int set_eventset_granularity(EventSetInfo *ESI, int arg)
{
  return(PAPI_ESBSTR);
}

static int set_default_granularity(int arg)
{
  return(PAPI_ESBSTR);
}

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Fill in what we can of the papi_system_info. */
  
  _papi_system_info.mhz = 100;
  _papi_system_info.ncpu = 1;

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
  int counter_code1;
  int counter_code2;
  int mask;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 

      mask = preset_map[event].mask;
      if (mask == 0)
	return(PAPI_ENOEVNT);

      counter_code1 = preset_map[event].code1;
      counter_code2 = preset_map[event].code2;
    }
  else
    {
      /* Support for native events here. */

      return(PAPI_ESBSTR);
    }

  /* Lower two bits tell us what counters we need */

  assert(this_state->mask <= 0x3);
  
  if (this_state->mask & mask)
    return(PAPI_ECNFLCT);
  
  if (this_state->mask == 0)
    init_config(this_state);
  
  if (mask & 0x1)
    {
      set_config(this_state,0,counter_code1);
      this_state->mask |= 0x1;
    }
  if (mask & 0x2)
    {
      set_config(this_state,1,counter_code2);
      this_state->mask |= 0x2;
    }
  
  /* Inform the upper level that software event 'index' consists of the
     following hardware counters. Mask is 0x0 through 0x3. */

  ESI->EventSelectArray[index] = mask;

  return(PAPI_OK);
}

/* Do not ever use ESI->NumberOfCounters in here. */

int _papi_hwd_rem_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int mask;
  int counter_code1;
  int counter_code2;

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

      counter_code1 = preset_map[event].code1;
      counter_code2 = preset_map[event].code2;
    }
  else
    {
      /* Support for native events here. */

      return(PAPI_ESBSTR);
    }

  /* Lower three bits tell us what counters we need */

  assert(this_state->mask <= 0x3);
  
  /* Check if we are removing something that's not preset */

  if ((this_state->mask & mask) != mask)
    return(PAPI_EINVAL);

  /* Clear out counters that are part of this event. */

  if (mask & 0x1)
    {
      set_config(this_state,0,0);
      this_state->mask ^= 0x1;
    }
  if (mask & 0x2)
    {
      set_config(this_state,1,0);
      this_state->mask ^= 0x2;
    }

  
  /* Inform the upper level that software event 'index' consists of 
     no hardware counters. Mask is 0x0 through 0x3. */

  ESI->EventSelectArray[index] = 0;

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, int index, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{
  return(PAPI_OK);
}
int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{
  return(PAPI_OK);
}

int _papi_hwd_stop(EventSetInfo *ESI, unsigned long long events[])
{ 
  return(PAPI_OK);
}

int _papi_hwd_reset(EventSetInfo *ESI)
{
  return(PAPI_OK);
}

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, unsigned long long events[])
{
  int retval, selector, j = 0, i;
  unsigned long long last_read[2];

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
	  break;
	case 0x2:
	  events[j] = last_read[1];
	  break;
	  /* case 0x3: This should never happen */
	  /* case 0x13: Here we could calculate derived metrics based on
	     ESI->EventCodeArray[i]; But this is an example, 
	     so it should never happen here. */
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
    case PAPI_SET_DEFGRN:
      return(set_default_granularity(option->granularity.granularity));
    case PAPI_SET_GRANUL:
      return(set_eventset_granularity(option->granularity.ESI,option->granularity.granularity));
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
  int code;

  if (preset & PRESET_MASK)
    { 
      preset ^= PRESET_MASK; 

      code = preset_map[preset].mask;
      if (code == 0x0)
	return(PAPI_ENOEVNT);
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
			       2,   /*  num_cntrs */
			       2,   /*  num_gp_cntrs */
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
