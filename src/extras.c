/* This file contains portable routines to do things that we wish the
vendors did in the kernel extensions or performance libraries. This includes
the following:

1) Handle overflow in < 64 bit hardware registers
2) Software overflow callbacks to user functions (for prof-like functions)
3) Software multiplexing (like perfex -a) */

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <string.h>
#include <strings.h>

#include "papi.h"
#include "papi_internal.h"

#define PAPI_ITIMER ITIMER_PROF
#define PAPI_SIGNAL SIGPROF

extern EventSetInfo *event_set_overflowing;
extern EventSetInfo *event_set_zero;

static void dispatch_profile(EventSetInfo *ESI, caddr_t eip)
{
  unsigned long address;
  EventSetProfileInfo_t *profile = &ESI->profile;

  address = (unsigned long)(eip - profile->offset);
  address *= profile->scale;
  address /= profile->divisor;
  
  switch (profile->flags)
    {
    case PAPI_PROFIL_POSIX:
    default:
      {
	unsigned short *buf = (unsigned short *)profile->buf;
	if (address < profile->bufsiz)
	  {
	    DBG((stderr,"dispatch_profile() handled at eip %p, bucket %lu\n",eip,address));
	    buf[address]++;
	  }
	else
	  DBG((stderr,"dispatch_profile() ignored at eip %p, bucket %lu\n",eip,address));	  
      }
    }
}

static void dispatch_overflow_signal(EventSetInfo *ESI, void *context)
{
  int retval;
  unsigned long long latest;

  retval = _papi_hwd_read(ESI, event_set_zero, ESI->latest); 
  if (retval < PAPI_OK)
    return;
  _papi_hwi_correct_counters(ESI, ESI->latest);

  /* Get the latest counter value */
  latest = ESI->latest[ESI->overflow.EventIndex];

  DBG((stderr,"dispatch_overflow() latest %llu, deadline %llu, threshold %d\n",
       latest,ESI->overflow.deadline,ESI->overflow.threshold));

  /* Is it bigger than the deadline? */
  if (latest > ESI->overflow.deadline)
    {
      ESI->overflow.count++;
      if (ESI->state & PAPI_PROFILING)
	dispatch_profile(ESI, (caddr_t)context); 
      else
	ESI->overflow.handler(ESI->EventSetIndex, ESI->overflow.count, ESI->overflow.EventCode, 
			      latest, &ESI->overflow.threshold, context);
      ESI->overflow.deadline = latest + ESI->overflow.threshold;
    }
}

static void dispatch_timer(int signal, struct sigcontext info)
{
  DBG((stderr,"dispatch_timer() at 0x%lx\n",info.eip));

  if (event_set_overflowing->state & PAPI_OVERFLOWING)
    {
      dispatch_overflow_signal(event_set_overflowing, (void *)info.eip); 
      return;
    }
  abort();
}

static int start_timer(int milliseconds)
{
  int retval;
  struct itimerval value;
  struct sigaction action;

  /* If the user has installed a SIGPROF, don't do anything */

  if (signal(PAPI_SIGNAL, SIG_IGN) != SIG_DFL)
    return(PAPI_ESYS);

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = milliseconds * 1000;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = milliseconds * 1000;

  memset(&action,0x00,sizeof(struct sigaction));
  action.sa_handler = (void *)dispatch_timer;
  
  if (sigaction(PAPI_SIGNAL, &action, NULL) < 0)
    return(PAPI_ESYS);

  retval = setitimer(PAPI_ITIMER, &value, NULL);
  if (retval == -1)
    {
      signal(PAPI_SIGNAL, SIG_DFL);
      return(PAPI_ESYS);
    }
  
  return(PAPI_OK);
}

int _papi_hwi_start_overflow_timer(EventSetInfo *ESI)
{
  event_set_overflowing = ESI;
  return(start_timer(ESI->overflow.timer_ms));
}

static int stop_timer(void)
{
  int retval;
  struct itimerval value;

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = 0;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = 0;
  
  retval = setitimer(PAPI_ITIMER, &value, NULL);

  if (retval == -1)
    return(PAPI_ESYS);

  if (signal(PAPI_SIGNAL,SIG_DFL) == SIG_ERR)
    return(PAPI_ESYS);

  return(PAPI_OK);
}

int _papi_hwi_stop_overflow_timer(EventSetInfo *ESI)
{
  event_set_overflowing = NULL;
  return(stop_timer());
}

/* int _papi_portable_set_multiplex(EventSetInfo *ESI, papi_multiplex_option_t *ptr)
{
  return(PAPI_ESBSTR);
}

int _papi_portable_set_overflow(EventSetInfo *ESI, papi_overflow_option_t *ptr)
{
  return(PAPI_ESBSTR);
}

int _papi_portable_get_overflow(EventSetInfo *ESI, papi_overflow_option_t *ptr)
{
  memcpy(ptr,&ESI->overflow.option,sizeof(*ptr));
  return(PAPI_OK);
}

int _papi_portable_get_multiplex(EventSetInfo *ESI, papi_multiplex_option_t *ptr)
{
  memcpy(ptr,&ESI->multiplex.option,sizeof(*ptr));
  return(PAPI_ESBSTR);
}
*/
