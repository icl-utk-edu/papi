/* 
* File:    extras.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* This file contains portable routines to do things that we wish the
vendors did in the kernel extensions or performance libraries. */

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

static unsigned int rnum = 0xdeadbeef;

static unsigned short random_ushort(void)
{
  return (unsigned short)(rnum = 1664525 * rnum + 1013904223);
}

static void posix_profil(unsigned long address, PAPI_sprofil_t *prof, unsigned short *outside_bin, int flags, long long excess, long long threshold)
{
  int increment = 1;
  unsigned short *buf = prof->pr_base;

  address = (address - prof->pr_off)/2;
  address = address * prof->pr_scale;
  address = address >> 16;

  if (address >= prof->pr_size)
    {
      *outside_bin = *outside_bin + 1;
      DBG((stderr,"outside bucket at %p = %u\n",outside_bin,*outside_bin));
      return;
    }

  if (flags == PAPI_PROFIL_POSIX)
    {
      buf[address]++;
      DBG((stderr,"bucket %lu = %u\n",address,buf[address]));
      return;
    }
    
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

  buf[address] += increment;
  DBG((stderr,"posix_profile() bucket %lu = %u\n",address,buf[address]));
}

static void dispatch_profile(EventSetInfo *ESI, void *context,
			     long long over, long long threshold)
{
  EventSetProfileInfo_t *profile = &ESI->profile;
  unsigned long pc;
  unsigned offset = 0;
  int count;
  unsigned best_offset = 0;
  int best_index = -1;
  unsigned short overflow_dummy;
  unsigned short *overflow_bin = NULL;
  int i;

  pc = (unsigned long)_papi_hwd_get_overflow_address(context);
  DBG((stderr,"handled at 0x%lx\n",pc));

  count = profile->count;
  if ((profile->prof[count-1].pr_off == 0) &&
      (profile->prof[count-1].pr_scale == 0x2))
    {
      overflow_bin = profile->prof[count-1].pr_base;
      count--;
    }
  else
    {
      overflow_bin = &overflow_dummy;
    }
    
  for (i = 0; i < count; i++)
    {
      offset = profile->prof[i].pr_off;
      if ((offset < pc) && (offset > best_offset))
	{
	  best_index = i;
	  best_offset = offset;
	}
    }

  if (best_index == -1)
    best_index = 0;

  posix_profil(pc, &profile->prof[best_index], overflow_bin, profile->flags, over, threshold);
}

typedef struct _thread_list {
  EventSetInfo *master;
  struct _thread_list *next; 
} EventSetInfoList;

static EventSetInfoList *head = NULL;
extern unsigned long int (*thread_id_fn)(void);

int _papi_hwi_insert_in_master_list(EventSetInfo *ptr)
{
  EventSetInfoList *entry = (EventSetInfoList *)malloc(sizeof(EventSetInfoList));
  if (entry == NULL)
    return(PAPI_ENOMEM);
  DBG((stderr,"(%p): New entry is at %p\n",ptr,entry));
  entry->master = ptr;
  PAPI_lock();
  entry->next = head;
  DBG((stderr,"(%p): Old head is at %p\n",ptr,entry->next));
  head = entry;
  DBG((stderr,"(%p): New head is at %p\n",ptr,head));
  PAPI_unlock();
  return(PAPI_OK);
}

EventSetInfo *_papi_hwi_lookup_in_master_list(void)
{
  extern EventSetInfo *default_master_eventset;
  if (thread_id_fn == NULL)
    return(default_master_eventset);
  else
    {
      unsigned long int id_to_find = (*thread_id_fn)();
      EventSetInfoList *tmp = head;
      while (tmp != NULL)
	{
	  if (tmp->master->tid == id_to_find)
	    return(tmp->master);
	  tmp = tmp->next;
	}
      return(NULL);
    }
}

void _papi_hwi_dispatch_overflow_signal(void *context)
{
  int retval;
  long long latest;
  EventSetInfo *master_event_set;
  EventSetInfo *ESI;

  master_event_set = _papi_hwi_lookup_in_master_list();
  ESI = master_event_set->event_set_overflowing;

  if ((ESI->state & PAPI_OVERFLOWING) == 0)
    abort();

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

static int start_timer(int milliseconds)
{
  int retval;
  struct sigaction action;
  struct itimerval value;

  /* If the user has installed a signal, don't do anything */

  if (signal(PAPI_SIGNAL, SIG_IGN) != SIG_DFL)
    return(PAPI_ESYS);

  memset(&action,0x00,sizeof(struct sigaction));
  action.sa_flags = SA_RESTART;
#if defined(_AIX) || defined(_CRAYT3E) || defined(sun)
  action.sa_sigaction = (void (*)(int, siginfo_t *, void *))_papi_hwd_dispatch_timer;
  action.sa_flags |= SA_SIGINFO;
#elif defined(linux)
  action.sa_handler = (void (*)(int))_papi_hwd_dispatch_timer;
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
