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
#include <assert.h>
#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <string.h>
#include <strings.h>

#include "papi.h"
#include "papi_internal.h"

#define PAPI_ITIMER ITIMER_REAL
#define PAPI_SIGNAL SIGALRM

extern EventSetInfo *event_set_overflowing;
extern EventSetInfo *event_set_zero;

static void dispatch_overflow_signal(EventSetInfo *ESI, void *context)
{
  int retval;
  unsigned long long latest;

  assert(ESI->state & PAPI_OVERFLOWING);

  retval = _papi_hwd_read(ESI, event_set_zero, ESI->latest); 
  if (retval < PAPI_OK)
    return;

  /* Get the latest counter value */
  latest = ESI->latest[ESI->overflow.EventIndex];

  /* Is it bigger than the deadline? */
  if (latest > ESI->overflow.deadline)
    {
      ESI->overflow.count++;
      ESI->overflow.handler(ESI->overflow.count, ESI->overflow.EventCode, ESI->overflow.EventIndex, ESI->latest, ESI->NumberOfCounters, context);
      ESI->overflow.deadline = latest + ESI->overflow.threshold;
    }
}

#ifdef __linux__
static void dispatch_timer(int signal, struct sigcontext info)
{
  DBG((stderr,"dispatch_timer() at 0x%lx\n",info.eip));
  dispatch_overflow_signal(event_set_overflowing, (void *)&info); 
}
#else
#include <ucontext.h>
static void dispatch_timer(int signal, siginfo_t *info, ucontext_t *context)
{
  DBG((stderr,"dispatch_timer() at %p\n",info->_data._prof._faddr));
  dispatch_overflow_signal(event_set_overflowing, (void *)&info); 
}
#endif

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

int start_overflow_timer(EventSetInfo *ESI)
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

int stop_overflow_timer(EventSetInfo *ESI)
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
