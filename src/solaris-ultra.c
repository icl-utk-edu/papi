#if defined(sun) && defined(__SVR4) && defined(sparc)

#include <time.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

typedef struct _hwd_preset {
  /* Fill in lots of stuff here */
  int counter_code; } hwd_control_state;

static hwd_control_state preset_map[PAPI_MAX_PRESET_EVENTS] = { { -1 }, };

static hwd_control_state current; /* not yet used. */

static int getmhz(void)
{
  /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
  /* See the PCL home page for the German version of PAPI. */

  int mhz;
  char line[256], cmd[80];
  FILE *f;
  char cmd_line[80], fname[L_tmpnam];
  
  /*??? system call takes very long */
  /* get system configuration and put output into file */
  sprintf(cmd_line, "/usr/sbin/prtconf -vp >%s", tmpnam(fname));
  if(system(cmd_line) == -1)
    {
      remove(fname);
      return -1;
    }
  
  /* open output file */
  if((f = fopen(fname, "r")) == NULL)
    {
      remove(fname);
      return -1;
    }
  
  /* ignore all lines until we reach something with a sparc line */
  while(fgets(line, 256, f) != NULL)
    {
      if((sscanf(line, "%s", cmd) == 1)
	 && !strcmp(cmd, "sparc-version:"))
	break;
    }
  
  /* then read until we find clock frequency */
  while(fgets(line, 256, f) != NULL)
    {
      if((sscanf(line, "%s %x", cmd, &mhz) == 2)
	 && !strcmp(cmd, "clock-frequency:"))
	break;
    }
  
  /* remove temporary file */
  remove(fname);
  
  /* if everything wqent ok, return mhz */
  if(strcmp(cmd, "clock-frequency:"))
    return -1;
  else
    return mhz / 1000000;

  /* End stolen code */
}

static int get_cpu_num(void)
{
  int cpu;

  processor_bind(P_LWPID, P_MYID, PBIND_QUERY, &cpu);
  return cpu;
}

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
