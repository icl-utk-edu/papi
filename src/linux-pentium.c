/* $Id$ */

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#include <stdio.h>
#include <sys/types.h>
#include <linux/unistd.h>	/* For syscall numbers */
#include <perf.h>

_syscall3(int, perf, int, op, int, counter, int, event);

#include "papi.h"
#include "papi_internal.h"
//#include "extras.c"
#include "papiStdEventDefs.h"

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

/* For PII/PPRO Linux, we need to compute a couple of presets (Like L2 
   cache misses) */

typedef struct _hwd_preset {
  int number;                
  int counter_code1;
  int counter_code2;
  int sp_code;   /* possibly needed for setting certain registers to 
                    enable special purpose counters */
} hwd_control_state;


/*example values for now */
static hwd_control_state preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                { },				// L1 D-Cache misses 
                { 4, 0x28, 0xC0,},		// L1 I-Cache misses 
		{ 4, 0x24, 0x2E,},		// L2 Cache misses
		{ 4, 0x24, 0x2E,},		// ditto
		{ 0,-1,-1,-1},			// L3 misses
		{ 0,-1,-1,-1},			// ditto
		{},				// 6	**unused preset map elements**
		{},				// 7
		{},				// 8
		{},				// 9
		{ }, 				// Req. access to shared cache line
		{ }, 				// Req. access to clean cache line
		{ }, 				// Cache Line Invalidation
                {},				// 13
                {},				// 14
                {},				// 15
                {},				// 16
                {},				// 17
                {},				// 18
                {},				// 19
		{ }, 				// D-TLB misses
		{ 3, 0x81, },			// I-TLB misses
                {},				// 22
                {},				// 23
                {},				// 24
                {},				// 25
                {},				// 26
                {},				// 27
                {},				// 28
                {},				// 29
		{ },				// TLB shootdowns
                {},				// 31
                {},				// 32
                {},				// 33
                {},				// 34
                {},				// 35
                {},				// 36
                {},				// 37
                {},				// 38
                {},				// 39
		{ 3, 0xC5, },			// Branch inst. mispred.
		{ 3, 0xC9, },			// Branch inst. taken
		{ 3, 0xE4, },			// Branch inst. not taken
                {},				// 43
                {},				// 44
                {},				// 45
                {},				// 46
                {},				// 47
                {},				// 48
                {},				// 49
		{ 3, 0xC0, },			// Total inst. executed
		{ 3, 0xC0, 0x10, },		// Integer inst. executed
		{ 1, 0x10, },			// Floating Pt. inst. executed
		{ },				// Loads executed
		{ },				// Stores executed
		{ 3, 0xC4, },			// Branch inst. executed
		{ 0,-1,-1,-1 },			// Vector/SIMD inst. executed 
		{ 5, 0x10, },			// FLOPS
                {},				// 58
                {},				// 59
		{ 9, },				// Total cycles
		{ 8, 0xC0, },			// MIPS
                {},				// 62
                {},				// 63
             };


static hwd_control_state current; /* not yet used. */

/* Low level functions, should not handle errors, just return codes. */

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Fill in papi_mdi */
  
  /* At init time, the higher level library should always allocate and 
     reserve EventSet zero. */

  zero->machdep = (void *)&current;

  return(PAPI_OK);
}

int _papi_hwd_add_event(void *machdep, int event)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  unsigned int foo = event;
  unsigned int preset;

  if (foo & PRESET_MASK)
  { preset = foo ^= PRESET_MASK; 
    switch (preset_map[preset].number)
    { case 0 : return(PAPI_ENOEVNT);
      case 1 :  
        if(this_state->number >= 4) return(PAPI_ECNFLCT);
        this_state->counter_code1 = preset_map[preset].counter_code1;
        this_state->number += 4;
        return 0;
      case 2 :
        if((this_state->number == 2) ||
           (this_state->number == 3) ||
           (this_state->number == 6) ||
           (this_state->number == 7)) return(PAPI_ECNFLCT);
        this_state->counter_code2 = preset_map[preset].counter_code2;
        this_state->number += 2;
        return 0;
      case 3 :
        if(this_state->number <= 3) 
        { this_state->counter_code1 = preset_map[preset].counter_code1;
          this_state->number += 4;
          return 0;
        }
        if(this_state->number >= 6) return(PAPI_ECNFLCT);
        else
        { this_state->counter_code2 = preset_map[preset].counter_code2;
          this_state->number += 2;
          return 0;
        }
      case 4 :
        if(this_state->number > 1) return(PAPI_ECNFLCT);
        this_state->counter_code1 = preset_map[preset].counter_code1;
        this_state->counter_code2 = preset_map[preset].counter_code2;
        this_state->number += 6;
        return 0;
      case 5 :
        if((this_state->number == 0) ||
           (this_state->number == 2)) 
        { this_state->counter_code1 = preset_map[preset].counter_code1;
          this_state->sp_code = 1;
          this_state->number += 5;
          return 0;
        }
        else return(PAPI_ECNFLCT);
      case 6 :
        if((this_state->number == 0) ||
           (this_state->number == 4))
        { this_state->counter_code2 = preset_map[preset].counter_code2;
          this_state->sp_code = 1;
          this_state->number += 3;
          return 0;
        }
        else return(PAPI_ECNFLCT);
      case 7 :
        if(this_state->number == 0) 
        { this_state->counter_code1 = preset_map[preset].counter_code1;
          this_state->counter_code2 = preset_map[preset].counter_code2;
          this_state->sp_code = 1;
          this_state->number += 7;
          return 0;
        }
        else return(PAPI_ECNFLCT);
      case 8 :
        if((this_state->number == 0) ||
           (this_state->number == 2))
        { this_state->counter_code1 = preset_map[preset].counter_code1;
          this_state->sp_code = 1;
          this_state->number += 5;
          return 0;
        }
        if(this_state->number == 4) 
        { this_state->counter_code2 = preset_map[preset].counter_code2;
          this_state->sp_code = 1;
          this_state->number += 3;
          return 0;
        }
        else return(PAPI_ECNFLCT);
      case 9 :
        if(this_state->number % 2) return(PAPI_ECNFLCT);
        this_state->sp_code = 1;
        this_state->number += 1;
        return 0; 
     }
  }
  else
  { if(this_state->number >= 6) return(PAPI_ECNFLCT);
    if(this_state->number >= 4) 
    { this_state->counter_code2 = foo;
      this_state->number += 2;
      return 0;
    }
    this_state->counter_code1 = foo;
    this_state->number += 4;
    return 0;
  }

  return(PAPI_OK);
}



int _papi_hwd_rem_event(void *machdep, int event)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  unsigned int foo = event;
  unsigned int preset;

  if (foo & PRESET_MASK)
  { preset = foo ^= PRESET_MASK; 
    switch (preset_map[preset].number)
    { case 0 : return(PAPI_ENOEVNT);

      case 1 :
        if(this_state->counter_code1 == preset_map[preset].counter_code1)
        { this_state->counter_code1 = -1; 	//because value 0 may be an event 
          this_state->number -= 4;
          return 0;
        }
        return(PAPI_ENOTRUN); 		//user tried to remove wrong event

      case 2 :
        if(this_state->counter_code2 == preset_map[preset].counter_code2) 
        { this_state->counter_code2 = -1; 
          this_state->number -= 2;
          return 0;
        }
        return(PAPI_ENOTRUN);

      case 3 :
        if(this_state->counter_code1 == preset_map[preset].counter_code1);
        { this_state->counter_code1 = -1;
          this_state->number -= 4;
          return 0;
        }
        if(this_state->counter_code2 == preset_map[preset].counter_code2)
        { this_state->counter_code2 = -1;
          this_state->number -= 2;
          return 0;
        }
        return(PAPI_ENOTRUN);

      case 4 :
        if(this_state->counter_code1 == preset_map[preset].counter_code1);
        { if(this_state->counter_code2 == preset_map[preset].counter_code2)
          { this_state->counter_code1 = -1;
            this_state->counter_code2 = -1;
            this_state->number -= 6;
            return 0;
          }
          return(PAPI_ENOTRUN);
        }
        return(PAPI_ENOTRUN);

      case 5 :
        if(this_state->counter_code1 == preset_map[preset].counter_code1)
        { this_state->counter_code1 = -1;
          this_state->sp_code = -1;
          this_state->number -= 5;
          return 0;
        }
        return(PAPI_ENOTRUN);

      case 6 :
        if(this_state->counter_code2 == preset_map[preset].counter_code2)
        { this_state->counter_code2 = -1;
          this_state->sp_code = -1;
          this_state->number -= 3;
          return 0;
        }
        return(PAPI_ENOTRUN);

      case 7 :
        if(this_state->counter_code1 == preset_map[preset].counter_code1);
        { if(this_state->counter_code2 == preset_map[preset].counter_code2)
          { this_state->counter_code1 = -1;
            this_state->counter_code2 = -1;
            this_state->sp_code = -1;
            this_state->number -= 7;
            return 0;
          }
          return(PAPI_ENOTRUN);
        }
        return(PAPI_ENOTRUN);

      case 8 :
        if(this_state->counter_code1 == preset_map[preset].counter_code1);
        { this_state->counter_code1 = -1;
          this_state->sp_code = -1;
          this_state->number -= 5;
          return 0;
        }
        if(this_state->counter_code2 == preset_map[preset].counter_code2)
        { this_state->counter_code2 = -1;
          this_state->sp_code = -1;
          this_state->number -= 3;
          return 0;
        }
        return(PAPI_ENOTRUN);

      case 9 :
        if(this_state->number % 2) 
        { this_state->number -= 1;
          this_state->sp_code = -1;
          return 0;
        }
        return(PAPI_ENOTRUN); 
    }
  } 
  if(this_state->counter_code1 == foo)
  { this_state->counter_code1 = -1;
    this_state->number -= 4;
    return 0;
  }
  if(this_state->counter_code2 == foo)
  { this_state->counter_code2 = -1;
    this_state->number -= 2;
    return 0;
  }
  return(PAPI_ENOTRUN);
}

int _papi_hwd_add_prog_event(void *machdep, int event, void *extra)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_start(void *machdep)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  int retval;

  if(this_state->counter_code1 >= 0)
  { retval = perf(PERF_SET_CONFIG, 0, this_state->counter_code1);
    if(retval) return(PAPI_EBUG);
  }

  if(this_state->counter_code1 >= 0)
  { retval = perf(PERF_SET_CONFIG, 1, this_state->counter_code2);
    if(retval) return(PAPI_EBUG);
  }

  if(this_state->sp_code >= 0)
  { retval = perf(PERF_SET_CONFIG, 2, 1);
    if(retval) return(PAPI_EBUG);
  }

  return (retval);
}


int _papi_hwd_stop(void *machdep, long long events[])
{ hwd_control_state *this_state = (hwd_control_state *)machdep;
  int retval, machnum;

  retval = perf(PERF_STOP, 0, 0);
  if(retval) return(PAPI_EBUG); 
  return (_papi_hwd_read(this_state, events));
}


int _papi_hwd_reset(void *machdep)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;

  if(this_state->number == 0) return(PAPI_ENOTRUN);

  return perf(PERF_RESET, 0, 0); /*from perf.c */
}

int _papi_hwd_read(void *machdep, long long events[])
{
  int retval, machnum, i;
  hwd_control_state *this_state = (hwd_control_state *)machdep;

  for(i=0; i<3; i++) events[i] = -1;

  machnum = this_state->number;
  if(machnum == 0) return(PAPI_ENOTRUN);
  if(machnum >= 4)
  { retval = perf(PERF_READ, 0, events[0]);
    if(retval) return(PAPI_EBUG);
  }
  if((machnum == 3) || (machnum == 7))
  { retval = perf(PERF_READ, 1, events[1]);
    if(retval) return(PAPI_EBUG);
    retval = perf(PERF_READ, 2, events[2]);
    if(retval) return(PAPI_EBUG);
  }
  if((machnum == 2) || (machnum == 6))
  { retval = perf(PERF_READ, 1, events[1]);
    if(retval) return(PAPI_EBUG);
  }
  else 
  { retval = perf(PERF_READ, 2, events[2]);
    if(retval) return(PAPI_EBUG);
  }
  return 0;
}

int _papi_hwd_write(void *machdep, long long events[])
{ hwd_control_state *this_state = (hwd_control_state *)machdep;
  int retval;

  switch (this_state->number)
  { case 6 :
      retval = perf(PERF_WRITE, 0, events[0]);
      if(retval) return(PAPI_EBUG);
      retval = perf(PERF_WRITE, 1, events[1]);
      if(retval) return(PAPI_EBUG);
      break;

    case 4 :
      retval = perf(PERF_WRITE, 0, events[0]);
      if(retval) return(PAPI_EBUG);
      break;

    case 2 :
      retval = perf(PERF_WRITE, 1, events[1]);
      if(retval) return(PAPI_EBUG);
      break;

    case 5 :
      retval = perf(PERF_WRITE, 0, events[0]);
      if(retval) return(PAPI_EBUG);
      retval = perf(PERF_WRITE, 2, events[2]);
      if(retval) return(PAPI_EBUG);
      break;

    case 3 :
      retval = perf(PERF_WRITE, 1, events[1]);
      if(retval) return(PAPI_EBUG);
      retval = perf(PERF_WRITE, 2, events[2]);
      if(retval) return(PAPI_EBUG);
      break;

    case 7 :
      retval = perf(PERF_WRITE, 0, events[0]);
      if(retval) return(PAPI_EBUG);
      retval = perf(PERF_WRITE, 1, events[1]);
      if(retval) return(PAPI_EBUG);
      retval = perf(PERF_WRITE, 2, events[2]);
      if(retval) return(PAPI_EBUG);
      break;

    case 1 :
      retval = perf(PERF_WRITE, 2, events[2]);
      if(retval) return(PAPI_EBUG);
      break;

    case 0 :
      return(PAPI_ENOTRUN);

    default:
      return(PAPI_EBUG);
  }
  return(retval);
}

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

int _papi_hwd_setopt(int code, EventSetInfo *value, PAPI_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_MPXRES:
      return(_papi_portable_set_multiplex((EventSetInfo *)value,&option->multiplex));
    case PAPI_SET_OVRFLO:
      return(_papi_portable_set_overflow((EventSetInfo *)value,&option->overflow));
    default:
      return(PAPI_EINVAL);	//changed NULL to 0
    }
}

int _papi_hwd_getopt(int code, EventSetInfo *value, PAPI_option_t *option)
{
  switch (code)
    {
    case PAPI_GET_MPXRES:
      return(_papi_portable_get_multiplex((EventSetInfo *)value,&option->multiplex));
    case PAPI_GET_OVRFLO:
      return(_papi_portable_get_overflow((EventSetInfo *)value,&option->overflow));
    default:
      return(PAPI_EINVAL);
    }
}

/* Machine info structure. -1 is unused. */

papi_mdi _papi_system_info = { "$Id$",
			        -1,
			        -1, 
			        -1,
			        -1,
			        -1,
			        2,
			        0,
			        1,
			        -1, 
			        -1,
			       sizeof(hwd_control_state), 
			       NULL };
