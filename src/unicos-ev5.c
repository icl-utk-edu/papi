/* $Id$ */

#include <mpp/globals.h>
#include <stdio.h>
#include <unistd.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"
#include "unicos-ev5.h"

static hwd_control_state_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                { -1, -1, 0x4, 0x4 },   /* L1 D-Cache misses  */
                { -1, -1, 0x6, 0x4 },	/* L1 I-Cache misses  */
		{ -1, 0xf, 0xf, 0x6 },	/* L2 Cache misses */
		{ -1, -1, -1, 0x6 },	/* ditto */
		{ -1, -1, -1, 0 },	/* L3 misses */
		{ -1, -1, -1, 0 },	/* ditto */
		{ -1, -1, -1, 0 },		/* 6	**unused preset map elements** */
		{ -1, -1, -1, 0 },		/* 7 */
		{ -1, -1, -1, 0 },		/* 8 */
		{ -1, -1, -1, 0 },		/* 9 */
		{ -1, -1, -1, 0 }, 	/* Req. access to shared cache line */
		{ -1, -1, -1, 0 }, 	/* Req. access to clean cache line */
		{ -1, -1, -1, 0 }, 	/* Cache Line Invalidation */
                { -1, -1, -1, 0 },		/* 13 */
                { -1, -1, -1, 0 },		/* 14 */
                { -1, -1, -1, 0 },		/* 15 */
                { -1, -1, -1, 0 },		/* 16 */
                { -1, -1, -1, 0 },		/* 17 */
                { -1, -1, -1, 0 },		/* 18 */
                { -1, -1, -1, 0 },		/* 19 */
		{ -1, -1, 0x7, 0x4 },   /* D-TLB misses */
		{ -1, -1, 0x5, 0x4 },	/* I-TLB misses */
                { -1, -1, -1, 0 },	/* Total TLB misses */
                { -1, -1, -1, 0 },	        /* 23 */
                { -1, -1, -1, 0 },		/* 24 */
                { -1, -1, -1, 0 },		/* 25 */
                { -1, -1, -1, 0 },		/* 26 */
                { -1, -1, -1, 0 },		/* 27 */
                { -1, -1, -1, 0 },		/* 28 */
                { -1, -1, -1, 0 },		/* 29 */
		{ -1, -1, -1, 0 },	/* TLB shootdowns */
                { -1, -1, -1, 0 },		/* 31 */
                { -1, -1, -1, 0 },		/* 32 */
                { -1, -1, -1, 0 },		/* 33 */
                { -1, -1, 0xa, 0x4 },	/* Cycles stalled waiting for memory */
                { -1, -1, -1, 0 },	/* Cycles stalled waiting for memory read */
                { -1, -1, -1, 0 },	/* Cycles stalled waiting for memory write */
                { -1, 0x0, 0xd, 0x2 },	/* Cycles no instructions issued */
                { -1, 0x7, -1, 0x2 },	/* Cycles max instructions issued */
                { -1, -1, -1, 0 },		/* 39 */
		{ -1, -1, -1, 0 },		/* 40 */
                { -1, -1, -1, 0 },		/* 41 */
		{ -1, -1, -1, 0 },	/* Uncond. branches executed */
		{ -1, 0x8, 0x3, 0x2 },	/* Cond. branch inst. executed.*/
		{ -1, -1, -1, 0 },	/* Cond. branch inst. taken*/
		{ -1, -1, -1, 0 },	/* Cond. branch inst. not taken*/
		{ -1, -1, 0x3, 0x4 },	/* Cond. branch inst. mispred.*/
                { -1, -1, -1, 0 },		/* 47 */
                { -1, -1, -1, 0 },		/* 48 */
                { -1, -1, -1, 0 },		/* 49 */
		{ 0x1, -1, -1, 0x1 },	/* Total inst. executed */
		{ -1, 0x9, -1, 0x2 },	/* Integer inst. executed */
		{ -1, 0xa, -1, 0x2 },	/* Floating Pt. inst. executed */
		{ -1, 0xb, -1, 0x2 },	/* Loads executed */
		{ -1, 0xc, -1, 0x2 },	/* Stores executed */
		{ -1, 0x8, -1, 0x2 },	/* Branch inst. executed */
		{ -1, -1, -1, 0 },	/* Vector/SIMD inst. executed  */
		{ -1, -1, -1, 0 },	/* FLOPS */
                { -1, -1, -1, 0 },		/* 58 */
                { -1, -1, -1, 0 },		/* 59 */
		{ 0x0, -1, -1, 0x1 },	/* Total cycles */
		{ -1, -1, -1, 0 },	/* MIPS */
                { -1, -1, -1, 0 },		/* 62 */
                { -1, -1, -1, 0 },		/* 63 */ };

/* Globals are BAD */

static hwd_control_state_t current; /* not yet used. */

/* Privilege bits */

static int kill_pal = 1, kill_user = 1, kill_kernel = 1;

/* Utility functions */

static int getmhz(void)
{
  long sysconf(int request);
  float p;
  
  p = (float) sysconf(_SC_CRAY_CPCYCLE); /* Picoseconds */
  p = p * 1.0e-12; /* Convert to seconds */
  p = 1.0 / (p * 1000000.0); /* Convert to MHz */
  return((int)p);
}

/* Low level functions, should not handle errors, just return codes. */

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Fill in papi_mdi */
  
  /* At init time, the higher level library should always allocate and 
     reserve EventSet zero. */

  _papi_system_info.ncpu = 0;
  _papi_system_info.type = 0;
  _papi_system_info.cpu = 0;
  _papi_system_info.mhz = getmhz();
  DBG((stderr,"CPU number %d at %d MHZ found\n",1,_papi_system_info.mhz));
  zero->machdep = (void *)&current;

  return(PAPI_OK);
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

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int sel0, sel1, sel2, mask;

  if (event & PRESET_MASK)
    { 
      event ^= PRESET_MASK; 

      sel0 = preset_map[event].sel0;
      sel1 = preset_map[event].sel1;
      sel2 = preset_map[event].sel2;
      mask = preset_map[event].mask;

      if (mask == 0x0)
	return(PAPI_ENOEVNT);
    }
  else
    {
      sel0 = event & 0x1;
      sel1 = event >> 1 & 0xf;
      sel2 = event >> 5 & 0xf;
      mask = event >> 9 & 0x7;

      /* There must be at least one event */

      if ((mask & 0x7) == 0)
	return(PAPI_EINVAL);
    }

  assert(this_state->mask <= 0x7);
  
  if (this_state->mask == 0)
    init_pmctr(&this_state->pmctr);
  
  if (this_state->mask & mask)
    return(PAPI_ECNFLCT);
  
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

      sel0 = preset_map[event].sel0;
      sel1 = preset_map[event].sel1;
      sel2 = preset_map[event].sel2;
      mask = preset_map[event].mask;

      if (mask == 0x0)
	return(PAPI_ENOEVNT);
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

int _papi_hwd_start(EventSetInfo *ESI)
{
  int ret;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  ret = _wrperf(this_state->pmctr,0,0,0);
  if (ret == 0)
    return(PAPI_OK);
  else
    return(PAPI_ESBSTR);
}

int _papi_hwd_stop(EventSetInfo *ESI, unsigned long long events[])
{ 
  if (events)
    {
      retval = _papi_hwd_read(ESI,events);
      if (retval < PAPI_OK)
	return(retval);
    }

  ret = perfmonctl(PERFCNT_EV5, PERFCNT_OFF);
  if (ret != 0)
    return(PAPI_ESBSTR);

  return(retval);
}

int _papi_hwd_reset(EventSetInfo *ESI)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int ret;

  ret = _wrperf(this_state->pmctr,0,0,0);
  if (ret == 0)
    return(PAPI_OK);

  return(PAPI_ESBSTR);
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

int _papi_hwd_read(EventSetInfo *ESI, unsigned long long events[])
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int i, j=0, k=0;
  int selector;

  retval = update_counters(this_state->last_read);
  if (retval < PAPI_OK)
    return(retval);


  /* This routine distributes hardware counters to software counters */
  /* Note that the higher level selector[i] entries may not be contiguous */

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      selector = ESI->EventSelectArray[i];
      
      assert(selector != 0);
      if (selector == PAPI_NULL)
	continue;
      while (selector)
	{
	  if (selector & 0x1)
	    {
	      DBG((stderr,"Event %d, mask is 0x%x, including counter %d\n",j,selector,k + 1));
	      events[j] += this_state->last_read[k + 1];
	      selector = selector >> 1;
	      k++;
	    }
	}
      j++;

      /* Early exit! */

      if (j == ESI->NumberOfCounters)
	return(retval);
    }

  /* Should never get here */
  
  return(PAPI_EBUG);
}

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

int _papi_hwd_ctl(int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_DEFDOM:
      return(set_def_domain(option->defdomain.defdomain.domain));
      break;
    case PAPI_SET_DOMAIN:
      return(set_domain(option->domain.ESI,option->domain.domain.domain));
    case PAPI_GET_DEFDOM:
      option->defdomain.defdomain.domain = get_def_domain();
      return(PAPI_OK);
    case PAPI_GET_DOMAIN:
      option->domain.domain.domain = get_domain(option->domain.ESI);
      return(PAPI_OK);      
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
  zero->machdep = NULL;
  memset(&_papi_system_info,0x00,sizeof(_papi_system_info));

  return(PAPI_OK);
}

int _papi_hwd_query(int preset)
{
  int code;

  if (preset & PRESET_MASK)
    { 
      preset ^= PRESET_MASK; 

      code = preset_map[preset].mask;
      if (mask == 0)
	return(PAPI_EINVAL);
      else
	return(PAPI_OK);
    }
  return(PAPI_OK);
}

/* Machine info structure. -1 is unused. */

papi_mdi _papi_system_info = { "$Id$",
			       1.0,
			        -1,
			        -1, 
			        -1,
			        -1,
			        -1,
			         3,
			         3,
			         0,
			         0,
			         -1, 
			         -1,
			         sizeof(hwd_control_state_t),
			         NULL };
