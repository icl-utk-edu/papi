#if defined(i386) || defined(i486) || defined(i586) || defined(i686)
#if defined(linux)

/* $Id$ */

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* The values defined in this file may be X86-specific (2 general 
   purpose counters, 1 special purpose counter, etc.*/

/* PAPI stuff */

#include <stdio.h>
#include <unistd.h>
#include "linux-pentium.h"

_syscall3(int, perf, int, op, int, counter, int, event); 

#include "papi.h"
#include "papi_internal.h"
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

/*example values for now */

static hwd_control_state_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 
                { 0,-1,-1,-1},			// L1 D-Cache misses 
                { 4, 0x28, 0xC0,},		// L1 I-Cache misses 
		{ 4, 0x24, 0x2E,},		// L2 Cache misses
		{ 4, 0x24, 0x2E,},		// ditto
		{ 0,-1,-1,-1},			// L3 misses
		{ 0,-1,-1,-1},			// ditto
		{0,-1,-1,-1},			// 6	**unused preset map elements**
		{0,-1,-1,-1},			// 7
		{0,-1,-1,-1},			// 8
		{0,-1,-1,-1},			// 9
		{0,-1,-1,-1 }, 			// Req. access to shared cache line
		{0,-1,-1,-1 }, 			// Req. access to clean cache line
		{ 0,-1,-1,-1}, 			// Cache Line Invalidation
                {0,-1,-1,-1},			// 13
                {0,-1,-1,-1},			// 14
                {0,-1,-1,-1},			// 15
                {0,-1,-1,-1},			// 16
                {0,-1,-1,-1},			// 17
                {0,-1,-1,-1},			// 18
                {0,-1,-1,-1},			// 19
		{0,-1,-1,-1 }, 			// D-TLB misses
		{ 3, 0x81, },			// I-TLB misses
                {0,-1,-1,-1},			// 22
                {0,-1,-1,-1},			// 23
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
                {0,-1,-1,-1},			// 34
                {0,-1,-1,-1},			// 35
                {0,-1,-1,-1},			// 36
                {0,-1,-1,-1},			// 37
                {0,-1,-1,-1},			// 38
                {0,-1,-1,-1},			// 39
                {0,-1,-1,-1},			// 40
                {0,-1,-1,-1},			// 41
		{ 3, 0xC9, },			// Uncond. branches executed
		{ 3, 0xC5, },			// Cond. Branch inst. mispred.
		{ 3, 0xC9, },			// Cond. Branch inst. taken
		{ 3, 0xE4, },			// Cond. Branch inst. not taken
                {0,-1,-1,-1},			// 46
                {0,-1,-1,-1},			// 47
                {0,-1,-1,-1},			// 48
                {0,-1,-1,-1},			// 49
		{ 3, 0xC0, },			// Total inst. executed
		{ 3, 0xC0, 0x10, },		// Integer inst. executed
		{ 1, 0x10, },			// Floating Pt. inst. executed
		{0,-1,-1,-1 },			// Loads executed
		{0,-1,-1,-1 },			// Stores executed
		{ 3, 0xC4, },			// Branch inst. executed
		{ 0,-1,-1,-1 },			// Vector/SIMD inst. executed 
		{ 5, 0x10, },			// FLOPS
                {0,-1,-1,-1},			// 58
                {0,-1,-1,-1},			// 59
		{ 9, },				// Total cycles
		{ 8, 0xC0, },			// MIPS
                {0,-1,-1,-1},			// 62
                {0,-1,-1,-1},			// 63
             };


hwd_control_state_t *array[PAPI_INIT_SLOTS];

/* Low level functions, should not handle errors, just return codes. */

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Fill in papi_mdi */
  
  /* At init time, the higher level library should always allocate and 
     reserve EventSet zero. */

  unsigned long long stamp;

  stamp = rdtsc();
  sleep (1);
  stamp = (rdtsc() - stamp)/1000000;
  _papi_system_info.mhz = stamp;

  DBG((stderr,"CPU number %d at %d MHZ found\n",1,_papi_system_info.mhz));

  return(PAPI_OK);
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  unsigned int foo = event;
  unsigned int preset;
  unsigned int tmp_event, tmp_code;

  if(foo & SHARED_MASK) foo = foo ^= SHARED_MASK;

  if (foo & PRESET_MASK)
  { preset = foo ^= PRESET_MASK; 
    switch (preset_map[preset].number)
    { case 0 : return(PAPI_ENOEVNT);
      case 1 :  
        if(this_state->number >= 6) return(PAPI_ECNFLCT);
        if(this_state->number >= 4)
        { tmp_event=preset_map[(ESI->EventCodeArray[0]^=PRESET_MASK)].counter_code1;
          tmp_code=ESI->EventCodeArray[0]^=PRESET_MASK;
          if(tmp_event != this_state->counter_code1)
          { tmp_event=preset_map[(ESI->EventCodeArray[1]^=PRESET_MASK)].counter_code1;
            tmp_code=ESI->EventCodeArray[1]^=PRESET_MASK;
          }
          if((preset_map[tmp_code].number == 3) ||
             (preset_map[tmp_code].number == 6) ||
             (preset_map[tmp_code].number == 8))
          { this_state->counter_code2 = tmp_event;
            this_state->counter_code1 = preset_map[preset].counter_code1;
            this_state->number += 2;
            return(PAPI_OK); 
          }
          return(PAPI_ECNFLCT);
        }
        this_state->counter_code1 = preset_map[preset].counter_code1;
        this_state->number += 4;
        return(PAPI_OK);
      case 2 :
        if((this_state->number == 2) ||
           (this_state->number == 3) ||
           (this_state->number == 6) ||
           (this_state->number == 7)) return(PAPI_ECNFLCT);
        this_state->counter_code2 = preset_map[preset].counter_code2;
        this_state->number += 2;
        return(PAPI_OK);
      case 3 :
        if(this_state->number <= 3) 
        { this_state->counter_code1 = preset_map[preset].counter_code1;
          this_state->number += 4;
          return 0;
        }
        if(this_state->number >= 6) return(PAPI_ECNFLCT);
        else
        { this_state->counter_code2 = preset_map[preset].counter_code1;
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



int _papi_hwd_rem_event(EventSetInfo *ESI, int index, unsigned int event)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
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

int _papi_hwd_add_prog_event(EventSetInfo *ESI, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_merge(EventSetInfo *ESI)
{ 
  int i, retval, code[_papi_system_info.num_cntrs];
  hwd_control_state_t *this_state = ESI->machdep;

  retval=_papi_hwd_stop(ESI, ESI->start);
  if(retval) return(PAPI_EBUG);

  for(i=0; i<_papi_system_info.num_cntrs; i++)
  { retval=perf(PERF_GET_CONFIG, i, code[i]);
    if(retval) return(PAPI_EBUG);
  }
  for(i=0; i<_papi_system_info.num_cntrs; i++)
  { if(code[i] == this_state->counter_code1)  
    { ESI->EventCodeArray[i] = ESI->EventCodeArray[i] & SHARED_MASK;
    }
    else if(code[i] != 0) return(PAPI_ECNFLCT);
  }
  return (_papi_hwd_start(ESI));
} 

int _papi_hwd_start(EventSetInfo *EventSet)
{
  hwd_control_state_t *this_state = EventSet->machdep;
  int retval;

  retval=_papi_set_domain(EventSet, &EventSet->all_options.domain);
  if(retval) return(PAPI_EBUG);

  if(this_state->counter_code1 >= 0)
  { retval = perf(PERF_SET_CONFIG, 0, this_state->counter_code1);
    if(retval) return(PAPI_EBUG);
  }

  if(this_state->counter_code2 >= 0)
  { retval = perf(PERF_SET_CONFIG, 1, this_state->counter_code2);
    if(retval) return(PAPI_EBUG);
  }

  if(this_state->sp_code >= 0)
  { retval = perf(PERF_SET_CONFIG, 2, this_state->sp_code);
    if(retval) return(PAPI_EBUG);
  }
  retval = perf(PERF_START, 0, 0);
  return (retval);
}


int _papi_hwd_stop(EventSetInfo *ESI, unsigned long long events[])
{ 
  int retval;

  retval = perf(PERF_STOP, 0, 0);
  if (retval) 
    return(PAPI_EBUG); 

  if (events)
    return (_papi_hwd_read(ESI, events));
  else
    return (PAPI_OK);
}

int _papi_hwd_unmerge(EventSetInfo *ESI)
{ int i, retval, code[_papi_system_info.num_cntrs];
  hwd_control_state_t *this_state = ESI->machdep;

  retval=_papi_hwd_stop(ESI, ESI->stop);
  if(retval) return(PAPI_EBUG);

  for(i=0; i<_papi_system_info.num_cntrs; i++)
  { retval=perf(PERF_GET_CONFIG, i, code[i]);
    if(retval) return(PAPI_EBUG);
  }
  for(i=0; i<_papi_system_info.num_cntrs; i++)
  { if(code[i] == this_state->counter_code1)  
    { if(ESI->EventCodeArray[i] & SHARED_MASK)  /*find out if still needed by others*/
      { ESI->EventCodeArray[i] ^= SHARED_MASK;
      }
      this_state->counter_code1 = 0;
    }
  }

  return(_papi_hwd_start(ESI));
}


int _papi_hwd_reset(EventSetInfo *ESI)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

  if(this_state->number == 0) return(PAPI_ENOTRUN);

  if(ESI->state==PAPI_RUNNING)
  { perf(PERF_RESET, 0, 0); /*from perf.c */
    return _papi_hwd_start(ESI);
  }
  return perf(PERF_RESET, 0, 0); /*from perf.c */
}

int _papi_hwd_read(EventSetInfo *ESI, unsigned long long events[])
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int retval, machnum, i;

  for(i=0; i<3; i++) events[i] = -1;

  machnum = this_state->number;
  if(machnum == 0) return(PAPI_ENOTRUN);
  if(machnum >= 4)
  { retval = perf(PERF_READ, 0, (int)&events[0]);
    if(retval) return(PAPI_EBUG);
  }
  if(machnum == 3)
  { retval = perf(PERF_READ, 1, (int)&events[0]);
    if(retval) return(PAPI_EBUG);
    retval = perf(PERF_READ, 2, (int)&events[1]);
    if(retval) return(PAPI_EBUG);
  }
  if(machnum == 7)
  { retval = perf(PERF_READ, 1, (int)&events[1]);
    if(retval) return(PAPI_EBUG);
    retval = perf(PERF_READ, 2, (int)&events[2]);
    if(retval) return(PAPI_EBUG);
  }
  if((machnum == 2) || (machnum == 6))
  { retval = perf(PERF_READ, 1, (int)&events[1]);
    if(retval) return(PAPI_EBUG);
  }
  if(machnum == 5) 
  { retval = perf(PERF_READ, 2, (int)&events[1]);
    if(retval) return(PAPI_EBUG);
  }

  return PAPI_OK;
}

int _papi_hwd_write(EventSetInfo *ESI, unsigned long long events[])
{ 
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
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

int _papi_set_domain(EventSetInfo *ESI, _papi_int_domain_t *domain)
{ hwd_control_state_t *this_state = ESI->machdep;
  if (domain->domain.domain == 2)
  { this_state->counter_code1 = this_state->counter_code1 | 0x00020000;
    this_state->counter_code2 = this_state->counter_code2 | 0x00020000;
    this_state->sp_code = this_state->sp_code | 0x00020000;
  }
  if(domain->domain.domain == 3)
  { this_state->counter_code1 = this_state->counter_code1 | 0x00020000 | 0x00010000;
    this_state->counter_code2 = this_state->counter_code2 | 0x00020000 | 0x00010000;
    this_state->sp_code = this_state->sp_code | 0x00020000 | 0x00010000;
  }
  return(PAPI_OK);
}


int _papi_hwd_ctl(int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_MPXRES:
      /* return(_papi_portable_set_multiplex((EventSetInfo *)option->
           multiplex.ESI,&option->multiplex)); */
    case PAPI_SET_OVRFLO:
      /* return(_papi_portable_set_overflow((EventSetInfo *)option->
           multiplex.ESI,&option->overflow)); */
    case PAPI_GET_MPXRES:
      /* return(_papi_portable_get_multiplex((EventSetInfo *)option->
           multiplex.ESI,&option->multiplex)); */
    case PAPI_GET_OVRFLO:
      /* return(_papi_portable_get_overflow((EventSetInfo *)option->
           multiplex.ESI,&option->overflow)); */
    case PAPI_SET_DEFDOM:
    case PAPI_SET_DEFGRN:
    case PAPI_SET_DOMAIN:
    { option->domain.ESI->all_options.domain.domain.domain=option->domain.domain.domain;
      return(PAPI_OK);
    }
    case PAPI_SET_GRANUL:
      return(PAPI_EINVAL);
    case PAPI_GET_DEFDOM:
    case PAPI_GET_DEFGRN:
      return(PAPI_EINVAL);
    case PAPI_GET_DOMAIN:
    case PAPI_GET_GRANUL:
      return(PAPI_EINVAL);
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_query(int preset)
{ if (preset & PRESET_MASK)
  { preset ^= PRESET_MASK;
    if (preset_map[preset].number == 0)
      return(PAPI_EINVAL);
    else
      return(PAPI_OK);
    }
  return(PAPI_OK);
}


/* Machine info structure. -1 is unused. */

papi_mdi _papi_system_info = { "$Id$",
			       1.0,   // version
			        -1,   // ncpu
			        -1,   // nnodes
			        -1,   // type
			        -1,   // cpu
			        -1,   // mhz
			         3,   // num_cntrs
			         2,   // num_gp_cntrs
			         0,   // grouped_counters
			         1,   // num_sp_cntrs
 				-1,   // total_presets
 				-1,   // total_events
			       sizeof(hwd_control_state_t), 
			       NULL,
			       PAPI_DOM_USER,
			       PAPI_GRN_THR };
#endif
#endif
