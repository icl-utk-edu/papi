#if defined(_POWER) && defined(_AIX)

#include <time.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

typedef struct _hwd_preset {
  /* Fill in lots of stuff here */
  int group_code; } hwd_control_state;

static hwd_control_state preset_map[PAPI_MAX_PRESET_EVENTS] = { { -1 }, };

static hwd_control_state current; /* not yet used. */

/* Low level functions, should not handle errors, just return codes. */

int _papi_hwd_init(EventSetInfo *zero)
{
  zero->machdep = (void *)&current;

  /* Call sysconf and syssgi to get machine config */

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
      return(PAPI_OK);
    }
  else
    {
      return(PAPI_ECNFLCT);
    }
}

int _papi_hwd_rem_event(void *machdep, int event)
{
  hwd_control_state *this_state = (hwd_control_state *)machdep;
  unsigned int foo = event;
  unsigned int preset;

  if (foo & PRESET_MASK)
    { 
      return(PAPI_OK);
    }
  else
    {
      return(PAPI_ECNFLCT);
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

/* Machine info structure. -1 is PAPI_NULL. */

papi_mdi _papi_system_info = { "$Id$",
			        PAPI_NULL,
			        PAPI_NULL, 
			        PAPI_NULL,
			        PAPI_NULL,
			        PAPI_NULL,
			        2,
			        2,
			        0,
			        0,
			        PAPI_NULL, 
			        PAPI_NULL,
			       sizeof(hwd_control_state), 
			       NULL };
#endif
