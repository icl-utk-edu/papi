/* $Id$ */

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#include "papi.h"
#include "papi_internal.h"
#include "perf.h" /* substrate */

/* The following structure holds preset values for that defined in the 
   standard. Some presets may require the use
   of more than 1 hardware event in order to calculate that metric. 
   For example, if there is no miss counter, but there are counters for
   hits and total accesses, then we would need to measure the two and 
   take the difference. IN convention is the following:

if number == 0, then not supported
if number == 1, then only 1 counter is needed and it is only in counter 1.
if number == 2, then only 1 counter is needed and it is only in counter 2.
if number == 3, then either counter 1 or 2 may be used.
if number == 4, then both counters are needed. 
if number == 5, then counter 1 is needed in conjunction with the special 
                purpose counter.
if number == 6, then counter 2 is needed in conjunction with the special 
                purpose counter.
if number == 7, then both counters are needed in conjunction with the 
                special purpose counter.
if number == 8, then either counter 1 or 2 may be used with the special 
		purpose counter.
if number == 9, then only the special purpose counter is needed.
*/

#ifdef LINUX_PENTIUM

/* For PII/PPRO Linux, we need to compute a couple of presets (Like L2 
   cache misses) */

typedef struct _hwd_preset {
  int number;                
  int counter_code1;
  int counter_code2;
  int sp_sode;   /* possibly needed for setting certain registers to 
                    enable special purpose counters */
  int pad;
} hwd_control_state;


/*example values for now */
static hwd_control_state preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                { , , , , },			// L1 D-Cache misses 
                { 4, 0x28, 0xC0, , },		// L1 I-Cache misses 
		{ 4, 0x24, 0x2E, , },		// L2 Cache misses
		{ 4, 0x24, 0x2E, , },		// ditto
		{ 0, , , , },			// L3 misses
		{ 0, , , , },			// ditto
		{,,,,}				// 6	**unused preset map elements**
		{,,,,}				// 7
		{,,,,}				// 8
		{,,,,}				// 9
		{ , , , , }, 			// Req. access to shared cache line
		{ , , , , }, 			// Req. access to clean cache line
		{ , , , , }, 			// Cache Line Invalidation
                {,,,,}				// 13
                {,,,,}				// 14
                {,,,,}				// 15
                {,,,,}				// 16
                {,,,,}				// 17
                {,,,,}				// 18
                {,,,,}				// 19
		{ , , , , }, 			// D-TLB misses
		{ 3, 0x81, , , },		// I-TLB misses
                {,,,,}				// 22
                {,,,,}				// 23
                {,,,,}				// 24
                {,,,,}				// 25
                {,,,,}				// 26
                {,,,,}				// 27
                {,,,,}				// 28
                {,,,,}				// 29
		{ , , , , },			// TLB shootdowns
                {,,,,}				// 31
                {,,,,}				// 32
                {,,,,}				// 33
                {,,,,}				// 34
                {,,,,}				// 35
                {,,,,}				// 36
                {,,,,}				// 37
                {,,,,}				// 38
                {,,,,}				// 39
		{ 3, 0xC5, , , },		// Branch inst. mispred.
		{ 3, 0xC9, , , },		// Branch inst. taken
		{ 3, 0xE4, , , },		// Branch inst. not taken
                {,,,,}				// 43
                {,,,,}				// 44
                {,,,,}				// 45
                {,,,,}				// 46
                {,,,,}				// 47
                {,,,,}				// 48
                {,,,,}				// 49
		{ 3, 0xC0, , , },		// Total inst. executed
		{ 3, 0xC0, 0x10, , },		// Integer inst. executed
		{ 1, 0x10, , , },		// Floating Pt. inst. executed
		{ , , , , },			// Loads executed
		{ , , , , },			// Stores executed
		{ 3, 0xC4, , , },		// Branch inst. executed
		{ 0, , , , },			// Vector/SIMD inst. executed 
		{ 5, 0x10, , , },		// FLOPS
                {,,,,}				// 58
                {,,,,}				// 59
		{ 9, , , , },			// Total cycles
		{ 8, 0xC0, , , },		// MIPS
                {,,,,}				// 62
                {,,,,}				// 63
                };

static hwd_control_state current; /* not yet used. */

/* Low level functions, should not handle errors, just return codes. */

int _papi_hwd_init(EventSet *zero)
{
  /* Fill in papi_mdi */
  
  /* At init time, the higher level library should always allocate and 
     reserve EventSet zero. */

  zero.machdep = (void *)&current;

}

int _papi_hwd_add_event(void *machdep, int event)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  unsigned int foo = event;
  unsigned int preset;

  if (foo & PRESET_MASK)
    {
      preset = foo ^= PRESET_MASK; 
      /* lookup preset_map[preset] */
      /* check if it can be integrated into this state */
      /* do it */
    }
  else
    {
      /* check if it can be integrated into this state */
      /* do it */
    }
}

int _papi_hwd_rem_event(void *machdep, int event)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  unsigned int foo = event;
  unsigned int preset;

  if (foo & PRESET_MASK)
    {
      preset = foo ^= PRESET_MASK; 
      /* lookup preset_map[preset] */
      /* check if it exists in this state */
      /* remove it */
    }
  else
    {
      /* check if it exists in this state */
      /* remove it */
    }
}

int _papi_hwd_add_prog_event(void *machdep, int event, void *extra)
{
  return(PAPI_NOT_IMPLEM);
}

int _papi_hwd_start(void *machdep)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  int retval;

/* to explain the value for machdep->number briefly:
	I used the same values for counter 1, counter 2, and TSC 
	being used as for UNIX permissions read, write, execute
	(4, 2, 1).
*/

  switch (machdep->number)
  { case 6 :
      retval = perf(PERF_SET_CONFIG, 0, machdep->counter_code1);
      if(retval) = return(PAPI_EBUG); 
      retval = perf(PERF_SET_CONFIG, 1, machdep->counter_code2);  
      if(retval) = return(PAPI_EBUG);
      retval = perf(PERF_START, 0, 0);
      if(retval) = return(PAPI_EBUG);
      break;

    case 4 :
      retval = perf(PERF_SET_CONFIG, 0, machdep->counter_code1);
      if(retval) = return(PAPI_EBUG);
      retval = perf(PERF_START, 0, 0);
      if(retval) = return(PAPI_EBUG);
      break;

    case 2 :
      retval = perf(PERF_SET_CONFIG, 1, machdep->counter_code2);
      if(retval) = return(PAPI_EBUG);
      retval = perf(PERF_START, 0, 0);
      if(retval) = return(PAPI_EBUG);
      break;

    case 5 :
      retval = perf(PERF_SET_CONFIG, 0, machdep->counter_code1);
      if(retval) = return(PAPI_EBUG);
      // start TSC;
      retval = perf(PERF_START, 0, 0);
      if(retval) = return(PAPI_EBUG);
      break;

    case 3 :
      retval = perf(PERF_SET_CONFIG, 1, machdep->counter_code2);
      if(retval) = return(PAPI_EBUG);
      // start TSC;
      retval = perf(PERF_START, 0, 0);
      if(retval) = return(PAPI_EBUG);
      break;

    case 7 :
      retval = perf(PERF_SET_CONFIG, 0, machdep->counter_code1);
      if(retval) = return(PAPI_EBUG);
      retval = perf(PERF_SET_CONFIG, 1, machdep->counter_code2);
      if(retval) = return(PAPI_EBUG);
      // start TSC;
      retval = perf(PERF_START, 0, 0);
      if(retval) = return(PAPI_EBUG);
      break;

    case 1 :
      // start TSC;
      break;

    case 0 :
      return(PAPI_ENOTRUN);

    default:
      return(PAPI_EBUG);
  }
  return(retval);
}

  return perf(PERF_START, 0, 0);  /* from perf.c */

  /* set control registers and start counting */
}

int _papi_hwd_stop(void *machdep, long long *events)
 {
  hwd_control_state *this_state = (hwd_control_state *)machdep;

  /* leave control registers and stop counting */

  if (events)
    /* copy appropriate events from machdep into *events */
    /* i.e. if machdep->number == 2, then you only copy counter2 into */
    /* events[1]... */

  return perf(PERF_STOP, 0, 0); /* from perf.c */

}

int _papi_hwd_reset(void *machdep)
{
  /* reset the hardware counters if necessary */

  hwd_control_state *this_state = (hwd_control_state *)machdep;

  return perf(PERF_RESET, 0, 0); /*from perf.c */
}

int _papi_hwd_read(void *machdep, long long *events)
{
  /* copy appropriate events from machdep into *events */

  hwd_control_state *this_state = (hwd_control_state *)machdep;

  /*figure out which counter to read from this_state->number */

  return perf(PERF_READ, appropriate_counter, (int) events);
    /* from perf.c */
}

int _papi_hwd_write(void *machdep, long long *events)
{
  /* copy appropriate events from *events into kernel */

  hwd_control_state *this_state = (hwd_control_state *)machdep;

  return perf(PERF_WRITE, appropriate_counter, (int) events);
    /* from perf.c */
}

int _papi_hwd_setopt(int code, void *option)
{
  /* used for native options like counting level, etc...*/

  /* probably from User Low Level API functions 
     int PAPI_set_granularity(int granularity) 
     and int PAPI_set_context(int context) 

     we can probably use code=1 for granularity, 
                         code=2 for context, 
                         code=3 for overflow threshold,
                         code=4 multiplexing,
     and void *option for either (PAPI_PER_THR, PAPI_PER_PROC,
     PAPI_PER_CPU or PAPI_PER_NODE), or (PAPI_USER, PAPI_KERNEL, 
     PAPI_SYSTEM), for granularity and context, respectively, and
     user defined values for overflow and multiplexing.
  */
}

int _papi_hwd_getopt(int code, void *option)
{
  /* may require use of a global static structure that records calls
     to setopt iff substrate doesn't support it. */

  /* probably the same info as above */
}

int papi_err_level(int level)
{
  /* if the level = 0, return current value of error level, make no changes
            level = 1, set error level to PAPI_QUIET, return 1
            level = 2, set error level to PAPI_VERB_ECONT, return 2
            level = 3, set error level to PAPI_VERB_ESTOP, return 3
  */
}

/* For linux/x86 sample fields */

papi_mdi _papi_system_info = { "Curtis Jansens Perf library + GLUE",
			        1.0,
			        2, /* 2 way SMP */
			        PAPI_CHIP_PENTIUM_II,
			        PAPI_VENDOR_INTEL,
			        450,
			        2,
			        0,
			        1,
			        16, /* Approx */
			        68, /* Approx */
			        sizeof(hwd_control_state)  };

