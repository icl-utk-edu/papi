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

#if 0

int _papi_portable_set_multiplex(int value, PAPI_option_t *ptr)
{
  return(PAPI_ESUBSTR);
}

int _papi_portable_get_multiplex(int *value, PAPI_option_t *ptr)
{
  return(PAPI_ESUBSTR);
}

int _papi_portable_set_overflow(int value, PAPI_option_t *ptr)
{
  return(PAPI_OK);
}

int _papi_portable_get_overflow(int *value, PAPI_option_t *ptr)
{
  return(PAPI_OK);
}

static int time_for_accumulate_64()
{
  return(PAPI_OK);
}

static int time_for_multiplex()
{
  return(PAPI_OK);
}
static int time_for_overflow()
{
  return(PAPI_OK);
}

int _papi_dispatch_timer(int signal, siginfo_t *sip, ucontext_t *uap)
{
  /* for each running eventset */
  if (time_for_accumulate_64())
    ;
  if (time_for_overflow())
    /* call ESI->sigaction */
    ;
  if (time_for_multiplex())
    ;
  return(PAPI_OK);
}

static int start_timer(int milliseconds, struct sigaction *old)
{
  struct itimerval value;
  struct sigaction action;

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = milliseconds * 1000;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = milliseconds * 1000;
  action.sa_sigaction = dispatch_timer;
  action.sa_mask = 0;
  action.sa_flags = SA_RESTART | SA_ONSTACK | SA_SIGINFO;
  
#ifdef ITIMER_REALPROF
  retval = setitimer(ITIMER_REALPROF, &value, old);
#else
  retval = setitimer(ITIMER_PROF, &value, old);
#endif

  if (retval == -1)
    return(PAPI_error(PAPI_ESYS,NULL));

  if (sigaction(SIGPROF, action, NULL) < 0)
    return(PAPI_error(PAPI_ESYS,NULL));
  
  return(PAPI_OK);
}

static int stop_timer(struct sigaction *old)
{
  struct itimerval value;

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = 0;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = 0;
  
#ifdef ITIMER_REALPROF
  retval = setitimer(ITIMER_REALPROF, old, NULL);
#else
  retval = setitimer(ITIMER_PROF, old, NULL);
#endif

  if (retval == -1)
    return(PAPI_error(PAPI_ESYS,NULL));

  if (signal(SIGPROF,SIG_DFL) == SIG_ERR)
    return(PAPI_error(PAPI_ESYS,NULL));

  return(PAPI_OK);
}

int _papi_portable_enable_overflow(int hardware_event, papi_overflow_option_t *ptr)
{
  return(start_timer(ptr->milliseconds));
}

int _papi_portable_disable_overflow(int hardware_event, papi_overflow_option_t *ptr)
{
  return(stop_timer());
}

#endif
