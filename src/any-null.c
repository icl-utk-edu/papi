/* $Id$ */

/* Null substrate that does nothing */

#include <stdio.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"
#include "any-null.h"

static hwd_control_state_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                { 0 },		/* L1 D-Cache misses  */
                { 1 },		/* L1 I-Cache misses  */
		{ 2 },		/* L2 Cache misses */
		{ 3 },		/* ditto */
		{ 4 },		/* L3 misses */
		{ 5 },		/* ditto */
		{ -1 },		/* 6	**unused preset map elements** */
		{ -1 },		/* 7 */
		{ -1 },		/* 8 */
		{ -1 },		/* 9 */
		{ 10 }, 	/* Req. access to shared cache line */
		{ 11 }, 	/* Req. access to clean cache line */
		{ 12 }, 	/* Cache Line Invalidation */
                { -1 },		/* 13 */
                { -1 },		/* 14 */
                { -1 },		/* 15 */
                { -1 },		/* 16 */
                { -1 },		/* 17 */
                { -1 },		/* 18 */
                { -1 },		/* 19 */
		{ 20 }, 	/* D-TLB misses */
		{ 21 },		/* I-TLB misses */
                { -1 },	 	/* 22 */
                { -1 },	        /* 23 */
                { -1 },		/* 24 */
                { -1 },		/* 25 */
                { -1 },		/* 26 */
                { -1 },		/* 27 */
                { -1 },		/* 28 */
                { -1 },		/* 29 */
		{ 30 },		/* TLB shootdowns */
                { -1 },		/* 31 */
                { -1 },		/* 32 */
                { -1 },		/* 33 */
                { -1 },		/* 34 */
                { -1 },		/* 35 */
                { -1 },		/* 36 */
                { -1 },		/* 37 */
                { -1 },		/* 38 */
                { -1 },		/* 39 */
		{ -1 },		/* 40 */
                { -1 },		/* 41 */
		{ 42 },		/* Uncond. branches executed */
		{ 43 },		/* Cond. branch inst. executed.*/
		{ 44 },		/* Cond. branch inst. taken*/
		{ 45 },		/* Cond. branch inst. not taken*/
		{ 46 },		/* Cond. branch inst. mispred.*/
                { -1 },		/* 47 */
                { -1 },		/* 48 */
                { -1 },		/* 49 */
		{ 50 },		/* Total inst. executed */
		{ 51 },		/* Integer inst. executed */
		{ 52 },		/* Floating Pt. inst. executed */
		{ 53 },		/* Loads executed */
		{ 54 },		/* Stores executed */
		{ 55 },		/* Branch inst. executed */
		{ 56 },		/* Vector/SIMD inst. executed  */
		{ 57 },		/* FLOPS */
                { -1 },		/* 58 */
                { -1 },		/* 59 */
		{ 60 },		/* Total cycles */
		{ 61 },		/* MIPS */
                { -1 },		/* 62 */
                { -1 },		/* 63 */
             };

/* Globals are BAD */

static unsigned long long reads = 0;

static hwd_control_state_t current; /* not yet used. */

/* Low level functions, should not handle errors, just return codes. */

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Fill in papi_mdi */
  
  /* At init time, the higher level library should always allocate and 
     reserve EventSet zero. */

  _papi_system_info.mhz = 999;

  zero->machdep = (void *)&current;

  return(PAPI_OK);
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  unsigned int preset;

  /* We only support 1 preset */

  if (event & PRESET_MASK)
    { 
      preset = event ^= PRESET_MASK; 
      switch (preset_map[preset].code)
	{ 
	case -1:
	  return(PAPI_ENOEVNT);
	default:
	  this_state->code = preset_map[preset].code;	
	}
      return(PAPI_OK);
    }
  else
    return(PAPI_ENOEVNT);
}

int _papi_hwd_rem_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  unsigned int preset;

  if (event & PRESET_MASK)
    { 
      preset = event ^= PRESET_MASK; 
      switch (preset_map[preset].code)
	{ 
	case -1:
	  return(PAPI_ENOEVNT);
	default:
	  this_state->code = -1;
	}
      return(PAPI_OK);
    }
  else
    return(PAPI_ENOEVNT);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_start(EventSetInfo *ESI)
{
  return(PAPI_OK);
}

int _papi_hwd_stop(EventSetInfo *ESI, unsigned long long events[])
{ 
  if (events)
    {
      events[0] = reads;
      reads++;
    }

  return(PAPI_OK);
}

int _papi_hwd_reset(EventSetInfo *ESI)
{
  reads = 0;
  return(PAPI_OK);
}

int _papi_hwd_read(EventSetInfo *ESI, unsigned long long events[])
{
  events[0] = reads;
  reads++;

  return(PAPI_OK);
}

int _papi_hwd_write(EventSetInfo *ESI, unsigned long long events[])
{ 
  reads = events[0];

  return(PAPI_OK);
}

int _papi_hwd_ctl(int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_MPXRES:
    case PAPI_SET_OVRFLO:
    case PAPI_SET_DEFDOM:
    case PAPI_SET_DEFGRN:
    case PAPI_SET_DOMAIN:
    case PAPI_SET_GRANUL:
    case PAPI_GET_MPXRES:
    case PAPI_GET_OVRFLO:
    case PAPI_GET_DEFDOM:
    case PAPI_GET_DEFGRN:
    case PAPI_GET_DOMAIN:
    case PAPI_GET_GRANUL:
    default:
      return(PAPI_EINVAL);
    }
}

/* Machine info structure. -1 is unused. */

papi_mdi _papi_system_info = { "$Id$",
			       1.0,
			        -1,
			        -1, 
			        -1,
			        -1,
			        -1,
			         1,
			         1,
			         0,
			         0,
			         -1, 
			         -1,
			         sizeof(hwd_control_state_t),
			         NULL };
