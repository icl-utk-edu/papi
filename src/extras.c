/* 
* File:    extras.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
*/  

/* This file contains portable routines to do things that we wish the
vendors did in the kernel extensions or performance libraries. */

/* It also contains a new section at the end with Windows routines
 to emulate standard stuff found in Unix/Linux, but not Windows. */

#ifndef _WIN32
  #include SUBSTRATE
#else
  #include "win32.h"
#endif


static unsigned int rnum = 0xdeadbeef;

static unsigned short random_ushort(void)
{
  return (unsigned short)(rnum = 1664525 * rnum + 1013904223);
}

static void posix_profil(unsigned long address, PAPI_sprofil_t *prof, unsigned short *outside_bin, int flags, long_long excess, long_long threshold)
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
      if (excess <= (long_long)1)
	increment = 1;
      else if (excess > threshold)
	increment = 255;
      else
	{
	  threshold = threshold / (long_long)255;
	  increment = (int)(excess / threshold);
	}	
    }

  buf[address] += increment;
  DBG((stderr,"posix_profile() bucket %lu = %u\n",address,buf[address]));
}

static void dispatch_profile(EventSetInfo *ESI, void *context,
			     long_long over, long_long threshold)
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

void _papi_hwi_cleanup_master_list(void)
{
  EventSetInfoList *tmp;

  while (head)
    {
      tmp = head;
      head = head->next;
      DBG((stderr,"Freeing master EventSet for thread %ld at %p\n",tmp->master->tid,tmp));
      free(tmp);
    }
}

int _papi_hwi_insert_in_master_list(EventSetInfo *ptr)
{
  EventSetInfoList *entry = (EventSetInfoList *)malloc(sizeof(EventSetInfoList));
  if (entry == NULL)
    return(PAPI_ENOMEM);
  DBG((stderr,"(%p): New entry is at %p\n",ptr,entry));
  entry->master = ptr;

  _papi_hwd_lock();
  entry->next = head;
  DBG((stderr,"(%p): Old head is at %p\n",ptr,entry->next));
  head = entry;
  DBG((stderr,"(%p): New head is at %p\n",ptr,head));
  _papi_hwd_unlock();
  
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
      EventSetInfoList *tmp;

      _papi_hwd_lock();
      tmp = head;
      while (tmp != NULL)
	{
	  if (tmp->master->tid == id_to_find)
	    {
	      _papi_hwd_unlock();
	      return(tmp->master);
	    }
	  tmp = tmp->next;
	}
      DBG((stderr,"New thread %lu found, but not initialized.\n",id_to_find));
      _papi_hwd_unlock();
      return(NULL);
    }
}

void _papi_hwi_dispatch_overflow_signal(void *context)
{
  int retval;
  long_long latest;
  EventSetInfo *master_event_set;
  EventSetInfo *ESI;

  master_event_set = _papi_hwi_lookup_in_master_list();
  if (master_event_set == NULL)
    return;
  ESI = master_event_set->event_set_overflowing;
  if (ESI == NULL)
    {
      DBG((stderr,"New thread %lu initialized, but not overflowing.\n",(*thread_id_fn)()));
      return;
    }

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

#ifdef _WIN32

static MMRESULT	wTimerID;	// unique ID for referencing this timer

static int start_timer(int milliseconds)
{
  int retval = PAPI_OK;

  TIMECAPS	tc;
  UINT		wTimerRes;

  // get the timer resolution capability on this system
  if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) return(PAPI_ESYS);
  
  wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
  timeBeginPeriod(wTimerRes);
  
  // initialize a periodic timer
  //	triggering every (milliseconds) 
  //	and calling (_papi_hwd_timer_callback())
  //	with no data
  wTimerID = timeSetEvent(milliseconds, wTimerRes, 
		_papi_hwd_timer_callback, (DWORD)NULL, TIME_PERIODIC);
  if(!wTimerID) return PAPI_ESYS;

  return(retval);
}

static int stop_timer(void)
{
  if (timeKillEvent(wTimerID) != TIMERR_NOERROR) return(PAPI_ESYS);
  return(PAPI_OK);
}

#else

int _papi_hwi_using_signal = 0;

static int start_timer(int milliseconds)
{
  int retval;
  struct sigaction action, oaction;
  struct itimerval value;

  /* If the user has installed a signal, don't do anything

     The following code is commented out because many C libraries
     replace the signal handler when one links with threads. The
     name of this signal handler is not exported. So there really
     is NO WAY to check if the user has installed a signal. */

  /* tmp = (void *)signal(PAPI_SIGNAL, SIG_IGN);
  if ((tmp != (void *)SIG_DFL) && (tmp != (void *)_papi_hwd_dispatch_timer))
    {
      fprintf(stderr,"%p %p %p\n",SIG_DFL,_papi_hwd_dispatch_timer,tmp);
      return(PAPI_EMISC);
    }
  */

  memset(&action,0x00,sizeof(struct sigaction));
  action.sa_flags = SA_RESTART;
#if defined(_AIX) || defined(_CRAYT3E) || defined(sun)
  action.sa_sigaction = (void (*)(int, siginfo_t *, void *))_papi_hwd_dispatch_timer;
  action.sa_flags |= SA_SIGINFO;
#elif defined(linux) && !defined(__ia64)
  action.sa_handler = (void (*)(int))_papi_hwd_dispatch_timer;
#elif defined(__ALPHA) && defined(__osf__)
  action.sa_handler = (void (*)(int))_papi_hwd_dispatch_timer;
#endif


  if (sigaction(PAPI_SIGNAL, &action, &oaction) < 0)
    return(PAPI_ESYS);

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = milliseconds * 1000;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = milliseconds * 1000;
  
  retval = setitimer(PAPI_ITIMER, &value, NULL);
  if (retval == -1)
    {
      sigaction(PAPI_SIGNAL, &oaction, NULL);
      return(PAPI_ESYS);
    }

  _papi_hwd_lock();
  _papi_hwi_using_signal++;
  _papi_hwd_unlock();

  return(PAPI_OK);
}

static int stop_timer(void)
{
  int retval = PAPI_OK;
  struct itimerval value;

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = 0;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = 0;
  
  if (setitimer(PAPI_ITIMER, &value, NULL) == -1)
    retval = PAPI_ESYS;

  _papi_hwd_lock();
  _papi_hwi_using_signal--;
  if (_papi_hwi_using_signal == 0)
    {
      if (signal(PAPI_SIGNAL,SIG_DFL) == SIG_ERR)
	retval = PAPI_ESYS;
    }
  _papi_hwd_unlock();
  
  return(retval);
}

#endif /* _WIN32 */

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


/**********************************************************************
	Windows Compatability stuff
	Delimited by the _WIN32 define
**********************************************************************/
#ifdef _WIN32

/*
 This routine normally lives in <strings> on Unix.
 Microsoft Visual C++ doesn't have this file.
*/
extern int ffs(int i)
{
	int c = 1;
	
	do {
		if (i & 1) return(c);
		i = i >> 1;
		c++;
	}while (i);
	return(0);
}

/*
 Another Unix routine that I can't find in Windows
 This one should return a pseudo-random integer
 given an unsigned int seed. For now it just returns 1
*/
/*
 This routine is called only once and only by MPX_start.
 If it isn't implemented, stuff will still work, but the
 starting event won't be randomized, reducing the validity
 of the statistics...
*/
extern int rand_r (unsigned int *Seed)
{
	return (1);
}

#endif /* _WIN32 */
