/* This file contains portable routines to do things that we wish the
vendors did in the kernel extensions or performance libraries. This includes
the following:

1) Handle overflow in < 64 bit hardware registers (not yet)
2) Software overflow callbacks to user functions (for prof-like functions)
3) Software multiplexing (like perfex -a) (not yet) */

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <string.h>
#include <strings.h>
#include <limits.h>

#include "papi.h"
#include "papi_internal.h"

typedef union {
  PAPI_timer_handler_t timer_handler;
  PAPI_sample_handler_t sample_handler;
  PAPI_overflow_handler_t overflow_handler;
  PAPI_notify_handler_t notify_handler; } PAPI_handler_t;

typedef struct {
  int deadline;
  int current_ms;
  int emulated_interval_ms;
  PAPI_handler_t fn;
} PAPI_timer_t;

static unsigned int rnum = 0xdeadbeef;

unsigned short random_ushort(void)
{
  return (unsigned short)(rnum = 1664525 * rnum + 1013904223);
}

unsigned char random_uchar(void)
{
  return (unsigned char)(rnum = 1664525 * rnum + 1013904223);
}

void posix_profil(int flags, long long excess, long long threshold, 
		  unsigned short *buf, unsigned long address)
{
  int increment = 1;

  if (flags & PAPI_PROFIL_RANDOM)
    {
      if (random_ushort() <= (USHRT_MAX/4))
	return;
    }

  if (flags & PAPI_PROFIL_COMPRESS)
    {
      /* We're likely to ignore the sample if buf[address] gets big. */

       if (random_ushort() < buf[address]) 
	 {
	   return;
	 }
    }

  if (flags & PAPI_PROFIL_WEIGHTED)     /* Increment is between 1 and 255 */
    {
      if (excess <= 1LL)
	increment = 1;
      else if (excess > threshold)
	increment = 255;
      else
	{
	  threshold = threshold / 255LL;
	  increment = (int)(excess / threshold);
	}	
    }

  if (buf[address] + (unsigned short)1) /* Guard against overflow */
    {
      buf[address] += increment;
      DBG((stderr,"posix_profile() bucket %lu = %u\n",address,buf[address]));
      return;
    }
  DBG((stderr,"posix_profile() bucket overflow %lu = %u\n",address,buf[address]));
}

static void dispatch_profile(EventSetInfo *ESI, void *context,
			     long long over, long long threshold)
{
  EventSetProfileInfo_t *profile = &ESI->profile;
  unsigned long pc;

  pc = (unsigned long)_papi_hwd_get_overflow_address(context);
  DBG((stderr,"dispatch_profile() handled at 0x%lx\n",pc));

  pc = (pc - (unsigned long)profile->offset)/2;
  pc = pc * profile->scale;
  pc = pc >> 16;
  if (pc < profile->bufsiz)
    {
      posix_profil(profile->flags, over, threshold, profile->buf, pc);
      return;
    }
  DBG((stderr,"dispatch_profile() bucket %lu out of range\n",pc));
}

void _papi_hwi_dispatch_overflow_signal(EventSetInfo *ESI, EventSetInfo *master_event_set, void *context)
{
  int retval;
  long long latest;

  /* Get the latest counter value */

  retval = _papi_hwd_read(ESI, master_event_set, ESI->sw_stop); 
  if (retval < PAPI_OK)
    return;

  latest = ESI->sw_stop[ESI->overflow.EventIndex];

  DBG((stderr,"dispatch_overflow() latest %llu, deadline %llu, threshold %d\n",
       latest,ESI->overflow.deadline,ESI->overflow.threshold));

  /* Is it bigger than the deadline? */

  if ((_papi_system_info.supports_hw_overflow) || (latest > ESI->overflow.deadline))
    {
      ESI->overflow.count++;
      if (ESI->state & PAPI_PROFILING)
	dispatch_profile(ESI, (caddr_t)context, latest - ESI->overflow.deadline, ESI->overflow.threshold); 
      else
	ESI->overflow.handler(ESI->EventSetIndex, ESI->overflow.EventCode, ESI->overflow.EventIndex,
			      ESI->sw_stop, &ESI->overflow.threshold, context);
      ESI->overflow.deadline = latest + ESI->overflow.threshold;
    }
}

#if defined(_CRAYT3E)
#include <sys/ucontext.h>
static void dispatch_timer(int signal, siginfo_t *si, ucontext_t *info)
{
  extern EventSetInfo *default_master_eventset;
  EventSetInfo *eventset_overflowing = default_master_eventset->event_set_overflowing;
  DBG((stderr,"dispatch_timer() at 0x%lx\n",info->uc_mcontext.gregs[31]));
  
  if (event_set_overflowing->state & PAPI_OVERFLOWING)
    _papi_hwi_dispatch_overflow_signal(event_set_overflowing, master_event_set, (void *)info); 
  return;
}
#elif defined(sun) && defined(sparc)
#include <sys/ucontext.h>
static void dispatch_timer(int signal, siginfo_t *si, ucontext_t *info)
{
  extern EventSetInfo *default_master_eventset;
  EventSetInfo *eventset_overflowing = default_master_eventset->event_set_overflowing;
  DBG((stderr,"dispatch_timer() at 0x%lx\n",info->uc_mcontext.gregs[31]));
  
  if (event_set_overflowing->state & PAPI_OVERFLOWING)
    _papi_hwi_dispatch_overflow_signal(event_set_overflowing, master_event_set, (void *)info); 
  return;
}
#elif defined(linux)
static void dispatch_timer(int signal, struct sigcontext info)
{
  extern EventSetInfo *default_master_eventset;
  EventSetInfo *eventset_overflowing = default_master_eventset->event_set_overflowing;
  DBG((stderr,"dispatch_timer() at 0x%lx\n",info.eip));

  if (eventset_overflowing->state & PAPI_OVERFLOWING)
    _papi_hwi_dispatch_overflow_signal(eventset_overflowing, default_master_eventset, (void *)&info); 
  return;
}
#elif defined(_AIX)
static void dispatch_timer(int signal, siginfo_t *si, void *i)
{
  extern EventSetInfo *default_master_eventset;
  EventSetInfo *eventset_overflowing = default_master_eventset->event_set_overflowing;
#ifdef DEBUG
  ucontext_t *info;
#endif
#ifdef DEBUG
  info = (ucontext_t *)i;
  DBG((stderr,"dispatch_timer() at 0x%lx\n",info->uc_mcontext.jmp_context.iar));
#endif

  if (event_set_overflowing->state & PAPI_OVERFLOWING)
    _papi_hwi_dispatch_overflow_signal(event_set_overflowing, master_event_set, i); 
}
#elif defined(sgi) && defined(mips)
static void dispatch_timer(int signal, int code, struct sigcontext *info)
{
  extern EventSetInfo *default_master_eventset;
  EventSetInfo *eventset_overflowing = default_master_eventset->event_set_overflowing;
#ifdef DEBUG
  DBG((stderr,"dispatch_timer() at %p\n",(void *)info->sc_pc));
#endif

  if (event_set_overflowing->state & PAPI_OVERFLOWING)
    _papi_hwi_dispatch_overflow_signal(event_set_overflowing, master_event_set, (void *)info); 
}
#endif

static int start_timer(int milliseconds)
{
  int retval;
  struct sigaction action;
  struct itimerval value;

  /* If the user has installed a SIGPROF, don't do anything */

  if (signal(PAPI_SIGNAL, SIG_IGN) != SIG_DFL)
    return(PAPI_ESYS);

  memset(&action,0x00,sizeof(struct sigaction));
  action.sa_flags = SA_RESTART;
#if defined(_AIX) 
  action.sa_sigaction = (void (*)(int, siginfo_t *, void *))dispatch_timer;
  action.sa_flags |= SA_SIGINFO;
#elif defined(sgi) && defined(mips)
  action.sa_sigaction = (void (*)(int, siginfo_t *, void *))dispatch_timer;
#elif defined(_CRAYT3E)
  action.sa_sigaction = (void (*)(int, siginfo_t *, void *))dispatch_timer;
  action.sa_flags |= SA_SIGINFO;
#elif defined(sun) && defined(sparc)
  action.sa_sigaction = (void (*)(int, siginfo_t *, void *))dispatch_timer;
  action.sa_flags |= SA_SIGINFO;
#elif defined(linux)
  action.sa_handler = (void (*)(int))dispatch_timer;
#endif

  if (sigaction(PAPI_SIGNAL, &action, NULL) < 0)
    return(PAPI_ESYS);

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = milliseconds * 1000;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = milliseconds * 1000;
  
  retval = setitimer(PAPI_ITIMER, &value, NULL);
  if (retval == -1)
    {
      signal(PAPI_SIGNAL, SIG_DFL);
      return(PAPI_ESYS);
    }

  return(PAPI_OK);
}

static int stop_timer(void)
{
  int retval = PAPI_OK;
  struct itimerval value;

  if (signal(PAPI_SIGNAL,SIG_DFL) == SIG_ERR)
    retval = PAPI_ESYS;
  
  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = 0;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = 0;
  
  if (retval != PAPI_OK)
    setitimer(PAPI_ITIMER, &value, NULL);
  else
    if (setitimer(PAPI_ITIMER, &value, NULL) == -1)
      retval = PAPI_ESYS;

  return(retval);
}

int _papi_hwi_start_overflow_timer(EventSetInfo *ESI, EventSetInfo *master)
{
  int retval = PAPI_OK;

  master->event_set_overflowing = ESI;
  if (_papi_system_info.supports_hw_overflow == 0)
    retval = start_timer(ESI->overflow.timer_ms);
  return(retval);
}

int _papi_hwi_stop_overflow_timer(EventSetInfo *ESI, EventSetInfo *master)
{
  int retval = PAPI_OK;

  if (_papi_system_info.supports_hw_overflow == 0)
    retval = stop_timer();
  master->event_set_overflowing = NULL;
  return(retval);
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
