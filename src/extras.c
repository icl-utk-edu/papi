#ifdef LINUX_PENTIUM

/* $Id$ */

/* This file contains portable routines to do things that we wish the
vendors did in the kernel extensions or performance libraries. This includes
the following:

1) Handle overflow in < 64 bit hardware registers
2) Software overflow callbacks to user functions (for prof-like functions)
3) Software multiplexing (like perfex -a) */

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <string.h>
#include <strings.h>

#include "papi.h"
#include "papi_internal.h"

extern DynamicArray PAPI_EVENTSET_MAP;    

static void dispatch_eventset(EventSetInfo *ESI, void *context)
{
  long long latest;
  int retval;

  if (ESI->state & PAPI_ACCUMULATING) /* Not implemented */
    return;
  if (ESI->state & PAPI_MULTIPLEXING) /* Not implemented */
    return;
  if (ESI->state & PAPI_OVERFLOWING)
    ;

  /* First get the latest counters */

  /* This doesn't work until George fixes the substrate */

  retval = _papi_hwd_read(ESI->machdep, ESI->latest); 

  if (retval < PAPI_OK)
    return;

  /* Get the latest counter value */
  latest = ESI->latest[ESI->overflow.eventindex];

  /* Is it bigger than the deadline? */
  if (latest > ESI->overflow.deadline)
    {
      ESI->overflow.option.handler(ESI,context);
      ESI->overflow.deadline = latest + ESI->overflow.option.threshold;
    }
}

static void dispatch_timer(int signal, struct sigcontext_struct info)
{
  int i;
  int total = 0;
  EventSetInfo *t;

  /* for each running eventset */
  for (i=1;i<=PAPI_EVENTSET_MAP.totalSlots-PAPI_EVENTSET_MAP.availSlots;i++)
    {
      /* If we exist */
      if ((t = PAPI_EVENTSET_MAP.dataSlotArray[i])) {
	  /* and if we're running and doing something funny */
	if ((t->state & PAPI_RUNNING) &&
	    ((t->state & PAPI_ACCUMULATING) ||
	     (t->state & PAPI_OVERFLOWING) || 
	     (t->state & PAPI_MULTIPLEXING)))
	  dispatch_eventset(t,&info); 
	/* Short circuit */
	if ((++total) == PAPI_EVENTSET_MAP.fullSlots)
	  return;
      }
    }
}

static int start_timer(int milliseconds)
{
  int retval;
  struct itimerval value;
  struct sigaction action;

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = milliseconds * 1000;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = milliseconds * 1000;
  action.sa_handler = (void *)dispatch_timer;
  sigemptyset(&action.sa_mask);
#ifdef SA_ONSTACK
  action.sa_flags |= SA_ONSTACK;
#endif
#ifdef SA_SIGINFO
  action.sa_flags |= SA_SIGINFO;
#endif
  action.sa_flags |= SA_RESTART;
  
#ifdef ITIMER_REALPROF
  retval = setitimer(ITIMER_REALPROF, &value, NULL);
#else
  retval = setitimer(ITIMER_PROF, &value, NULL);
#endif

  if (retval == -1)
    return(PAPI_ESYS);

  if (sigaction(SIGPROF, &action, NULL) < 0)
    return(PAPI_ESYS);
  
  return(PAPI_OK);
}

static int stop_timer(void)
{
  int retval;
  struct itimerval value;

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = 0;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = 0;
  
#ifdef ITIMER_REALPROF
  retval = setitimer(ITIMER_REALPROF, &value, NULL);
#else
  retval = setitimer(ITIMER_PROF, &value, NULL);
#endif

  if (retval == -1)
    return(PAPI_ESYS);

  if (signal(SIGPROF,SIG_DFL) == SIG_ERR)
    return(PAPI_ESYS);

  return(PAPI_OK);
}

int _papi_portable_set_multiplex(EventSetInfo *ESI, papi_multiplex_option_t *ptr)
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

#endif
