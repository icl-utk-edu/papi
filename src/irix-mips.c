#if defined(sgi) && defined(mips)

#include <sys/hwperftypes.h>
#include <invent.h>
#include <time.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

/* Globals to get rid of */

static int fd = -1, generation = -1;

static hwperf_profevctrarg_t none;

/* Globals */ 

typedef hwperf_profevctrarg_t hwd_control_state;

static hwd_control_state preset_map[PAPI_MAX_PRESET_EVENTS] = { { -1, -1 }, };

static hwd_control_state current; /* not yet used. */

/* Low level functions, should not handle errors, just return codes. */

static int cpu(inventory_t *item, void *bar)
{
  if ((item->inv_class == INV_PROCESSOR) && (item->inv_type == INV_CPUBOARD))
    {
      _papi_system_info.ncpu++;
      _papi_system_info.mhz = item->inv_controller;
    }
  return(0);
}

int _papi_hwd_init(EventSetInfo *zero)
{
  char pfile[80];
  void *foo = NULL;
  int retval;
  pid_t pid;

  /* Get machine config */

  retval = scaninvent(cpu, foo);
  if (retval == -1)
    return(PAPI_ESBSTR);
    
  /* Acquire counters */

  pid = getpid();
  sprintf(pfile, "/proc/%05d", pid);
  fd = open(pfile, O_RDWR);
  if (fd == -1)
    return(PAPI_ESBSTR);

  memset(&none,0x00,sizeof(none));
  generation = ioctl(fd,PIOCENEVCTRS, (void *)&none);
  if (generation < 0)
    return(PAPI_ESBSTR);

  zero->machdep = (void *)&current;

  return(PAPI_OK);
}

int _papi_hwd_add_event(void *machdep, int event)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  unsigned int foo = event;
  unsigned int preset;

  if (foo & PRESET_MASK)
    { 
      preset = foo ^= PRESET_MASK; 
      this_state->counter_code1 = preset_map[preset].counter_code1;
      this_state->counter_code2 = preset_map[preset].counter_code2;
      return(PAPI_OK);
    }
  else
    {
      if ((event >= 0) && (event <= 15))
	{
	  if (this_state->counter_code1 == PAPI_NULL)
	    {
	      this_state->counter_code1 = event;
	      return(PAPI_OK);
	    }
	  return(PAPI_ECNFLCT);
	}
      else if ((event >= 16) && (event <= 31))
	{
	  if (this_state->counter_code2 == PAPI_NULL)
	    {
	      this_state->counter_code2 = event;
	      return(PAPI_OK);
	    }
	  return(PAPI_ECNFLCT);
	}
      else
	return(PAPI_ENOEVNT);
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
      if (preset_map[preset].counter_code1 != PAPI_NULL)
	this_state->counter_code1 = PAPI_NULL;
      if (preset_map[preset].counter_code2 != PAPI_NULL)
	this_state->counter_code2 = PAPI_NULL;
      return(PAPI_OK);
    }
  else
    {
      if ((event >= 0) && (event <= 15))
	{
	  if (this_state->counter_code1 != PAPI_NULL)
	    {
	      this_state->counter_code1 = PAPI_NULL;
	      return(PAPI_OK);
	    }
	  return(PAPI_ECNFLCT);
	}
      else if ((event >= 16) && (event <= 31))
	{
	  if (this_state->counter_code2 != PAPI_NULL)
	    {
	      this_state->counter_code2 = PAPI_NULL;
	      return(PAPI_OK);
	    }
	  return(PAPI_ECNFLCT);
	}
      else
	return(PAPI_ENOEVNT);
    }
}

int _papi_hwd_add_prog_event(void *machdep, int event, void *extra)
{
  return(PAPI_ESBSTR);
}

int _papi_hwd_start(void *machdep)
{
  int retval = PAPI_OK;

  return(retval);
}


int _papi_hwd_stop(void *machdep, long long events[])
{ 
  int retval = PAPI_OK;

  return(retval);
}


int _papi_hwd_reset(void *machdep)
{
  int retval = PAPI_OK;

  return(retval);
}

int _papi_hwd_read(void *machdep, long long events[])
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  int retval = PAPI_OK;

  return(retval);
}

int _papi_hwd_write(void *machdep, long long events[])
{ 
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  int retval = PAPI_OK;

  return(retval);
}

int _papi_hwd_setopt(int code, EventSetInfo *value, PAPI_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_MPXRES:
    case PAPI_SET_OVRFLO:
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_getopt(int code, EventSetInfo *value, PAPI_option_t *option)
{
  switch (code)
    {
    case PAPI_GET_MPXRES:
    case PAPI_GET_OVRFLO:
    default:
      return(PAPI_EINVAL);
    }
}

/* Machine info structure. */

papi_mdi _papi_system_info = { "$Id$",
			        0,
			        0, 
			        0,
			        0,
			        0,
			        2,
			        2,
			        0,
			        0,
			        0, 
			        0,
			       sizeof(hwd_control_state), 
			       NULL };
#endif
