/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

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

#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
  #define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"


#ifdef __LINUX__
#include <limits.h>
#endif

/*******************/
/* BEGIN EXTERNALS */
/*******************/

extern papi_mdi_t _papi_hwi_system_info;

#ifdef ANY_THREAD_GETS_SIGNAL
extern void _papi_hwi_lookup_thread_symbols(void);
extern int (*_papi_hwi_thread_kill_fn)(int, int);
#endif

/*****************/
/* END EXTERNALS */
/*****************/

/****************/
/* BEGIN LOCALS */
/****************/

static unsigned int rnum = DEADBEEF;

/**************/
/* END LOCALS */
/**************/

extern unsigned long int (*_papi_hwi_thread_id_fn)(void);

static unsigned short random_ushort(void)
{
  return (unsigned short)(rnum = 1664525 * rnum + 1013904223);
}

static void posix_profil(caddr_t address, PAPI_sprofil_t *prof, unsigned short *outside_bin, int flags, long_long excess, long_long threshold)
{
  int increment = 1;
  unsigned short *buf = prof->pr_base;
  unsigned long addr;
  u_long_long laddr;

  addr = (unsigned long)(address - prof->pr_off);
  laddr = addr / 2;
  laddr = laddr * prof->pr_scale;
  addr = (unsigned long)(laddr >> 16);

  if (addr >= prof->pr_size)
    {
      *outside_bin = *outside_bin + 1;
      DBG((stderr,"outside bucket at %p = %u\n",outside_bin,*outside_bin));
      return;
    }

  if (flags == PAPI_PROFIL_POSIX)
    {
      buf[addr]++;
      DBG((stderr,"bucket %lu = %u\n",addr,buf[addr]));
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

       if (random_ushort() < buf[addr]) 
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

  buf[addr] += increment;
  DBG((stderr,"posix_profile() bucket %lu = %u\n",addr,buf[addr]));
}

/*
static void dispatch_profile(EventSetInfo_t *ESI, void *context,
			     long_long over, long_long threshold)
*/
void dispatch_profile(EventSetInfo_t *ESI, void *context,
			     long_long over, long_long threshold)
{
  EventSetProfileInfo_t *profile = &ESI->profile;
  caddr_t pc = (caddr_t)_papi_hwd_get_overflow_address(context);
  caddr_t offset = (caddr_t)0;
  caddr_t best_offset = (caddr_t)0;
  int count;
  int best_index = -1;
  unsigned short overflow_dummy;
  unsigned short *overflow_bin = NULL;
  int i;

#ifdef PROFILE_DEBUG
  fprintf(stderr,"%lld:%s:0x%x:handled at 0x%lx\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),pc);
#endif

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
      offset = (caddr_t)profile->prof[i].pr_off;
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

void _papi_hwi_dispatch_overflow_signal(void *context)
{
  int retval;
  u_long_long latest;
  ThreadInfo_t *thread;
  EventSetInfo_t *ESI;

#ifdef OVERFLOW_DEBUG_TIMER
  if (_papi_hwi_thread_id_fn)
    fprintf(stderr,"%lld:%s:0x%x:Thread %x in _papi_hwi_dispatch_overflow_signal\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),(*_papi_hwi_thread_id_fn)());
#endif

  thread = _papi_hwi_lookup_in_thread_list();
  if (thread != NULL)
  {
    ESI = thread->event_set_overflowing;
    if (ESI == NULL)
	{
#ifdef OVERFLOW_DEBUG_TIMER
	  fprintf(stderr,"%lld:%s:0x%x:I'm using PAPI, but not overflowing.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)());
#endif
#ifdef ANY_THREAD_GETS_SIGNAL
	  _papi_hwi_broadcast_overflow_signal(thread->tid);
#endif
	  return;
	}

    if ((ESI->state & PAPI_OVERFLOWING) == 0)
	  abort();
      
    /* Get the latest counter value */
      
    retval = _papi_hwi_read(&thread->context, ESI, ESI->sw_stop); 
    if (retval < PAPI_OK)
	  return;
      
    latest = ESI->sw_stop[ESI->overflow.EventIndex];
      
    DBG((stderr,"dispatch_overflow() latest %llu, deadline %llu, threshold %d\n",latest,ESI->overflow.deadline,ESI->overflow.threshold));
  
  /* Is it bigger than the deadline? */
  
    if ((_papi_hwi_system_info.supports_hw_overflow) || 
             (latest > ESI->overflow.deadline))
    {
      ESI->overflow.count++;
      if (ESI->state & PAPI_PROFILING)
	    dispatch_profile(ESI, (caddr_t)context, 
             latest - ESI->overflow.deadline, ESI->overflow.threshold); 
      else
	    ESI->overflow.handler(ESI->EventSetIndex, ESI->overflow.EventCode, 
                ESI->overflow.EventIndex, ESI->sw_stop, 
                &ESI->overflow.threshold, context);
      ESI->overflow.deadline = latest + ESI->overflow.threshold;
    }
  }
#ifdef ANY_THREAD_GETS_SIGNAL
  else
  {
#ifdef OVERFLOW_DEBUG
    fprintf(stderr,"%lld:%s:0x%x:I haven't been noticed by PAPI before\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)());
#endif
    _papi_hwi_broadcast_overflow_signal((*_papi_hwi_thread_id_fn)());
  }
#endif
}

#ifdef _WIN32

static MMRESULT	wTimerID;	// unique ID for referencing this timer
static UINT		wTimerRes;	// resolution for this timer

static int start_timer(int milliseconds)
{
  int retval = PAPI_OK;

  TIMECAPS	tc;
  DWORD		threadID;

  // get the timer resolution capability on this system
  if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) return(PAPI_ESYS);
  
  // get the ID of the current thread to read the context later
  // NOTE: Use of this code is restricted to W2000 and later...
  threadID = GetCurrentThreadId();

  // set the minimum usable resolution of the timer
  wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
  timeBeginPeriod(wTimerRes);
  
  // initialize a periodic timer
  //	triggering every (milliseconds) 
  //	and calling (_papi_hwd_timer_callback())
  //	with no data
  wTimerID = timeSetEvent(milliseconds, wTimerRes, 
		_papi_hwd_timer_callback, threadID, TIME_PERIODIC);
  if(!wTimerID) return PAPI_ESYS;

  return(retval);
}

static int stop_timer(void)
{
  int retval = PAPI_OK;

  if (timeKillEvent(wTimerID) != TIMERR_NOERROR) retval = PAPI_ESYS;
  timeEndPeriod(wTimerRes);
  return(retval);
}

#else

int _papi_hwi_using_signal = 0;

static int start_timer(int milliseconds)
{
  struct sigaction action;
  struct itimerval value;

  DBG((stderr,"Timer start...\n"));
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
#if defined(_AIX) || defined(_CRAYT3E) || defined(sun) || defined(linux)
  action.sa_sigaction = (void (*)(int, siginfo_t *, void *))_papi_hwd_dispatch_timer;
  action.sa_flags |= SA_SIGINFO;
#elif defined(__ALPHA) && defined(__osf__)
  action.sa_handler = (void (*)(int))_papi_hwd_dispatch_timer;
#endif

#if defined(ANY_THREAD_GETS_SIGNAL)
  _papi_hwi_lookup_thread_symbols();

  _papi_hwd_lock();
  if (++_papi_hwi_using_signal > 1)
    {
      _papi_hwd_unlock();
#ifdef OVERFLOW_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:_papi_hwi_using_signal is now %d.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),_papi_hwi_using_signal);
#endif
      return(PAPI_OK);
    }
#endif

  if (sigaction(PAPI_SIGNAL, &action, NULL) < 0)
    return(PAPI_ESYS);

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = milliseconds * 1000;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = milliseconds * 1000;
  
  if (setitimer(PAPI_ITIMER, &value, NULL) < 0)
    return(PAPI_ESYS);
  
#ifndef ANY_THREAD_GETS_SIGNAL
  _papi_hwd_lock();
  _papi_hwi_using_signal++;
#endif
 _papi_hwd_unlock();

  DBG((stderr,"Timer started.\n"));
#ifdef OVERFLOW_DEBUG
  fprintf(stderr,"%lld:%s:0x%x:_papi_hwi_using_signal is now %d.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),_papi_hwi_using_signal);
#endif
  return(PAPI_OK);
}

static int stop_timer(void)
{
  int retval = PAPI_OK;
  struct itimerval value;

  DBG((stderr,"Stopping timer...\n"));
  value.it_interval.tv_sec = 0;
  value.it_interval.tv_usec = 0;
  value.it_value.tv_sec = 0;
  value.it_value.tv_usec = 0;
  
#ifdef ANY_THREAD_GETS_SIGNAL
  _papi_hwd_lock();
  if (--_papi_hwi_using_signal == 0)
    {
#ifdef OVERFLOW_DEBUG
      if (_papi_hwi_thread_id_fn)
	fprintf(stderr,"%lld:%s:0x%x:Thread 0x%x, turning off timer and signal.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),(*_papi_hwi_thread_id_fn)());
#endif
      retval = setitimer(PAPI_ITIMER, &value, NULL);
      assert(retval == 0);
      signal(PAPI_SIGNAL,SIG_DFL);
    }
#else
  if (setitimer(PAPI_ITIMER, &value, NULL) == -1)
    retval = PAPI_ESYS;

  _papi_hwd_lock();
  _papi_hwi_using_signal--;
  if (_papi_hwi_using_signal == 0)
    {
#ifdef OVERFLOW_DEBUG
      fprintf(stderr,"Turning off signal.\n");
#endif
      if (signal(PAPI_SIGNAL,SIG_DFL) == SIG_ERR)
	retval = PAPI_ESYS;
    }
#endif
  _papi_hwd_unlock();
#ifdef OVERFLOW_DEBUG
    fprintf(stderr,"%lld:%s:0x%x:_papi_hwi_using_signal is now %d.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),_papi_hwi_using_signal);
#endif
  DBG((stderr,"Timer stopped\n"));
  return(retval);
}

#endif /* _WIN32 */

int _papi_hwi_start_overflow_timer(ThreadInfo_t *thread, EventSetInfo_t *ESI)
{
  int retval = PAPI_OK;

  thread->event_set_overflowing = ESI;
  if (_papi_hwi_system_info.supports_hw_overflow == 0)
#ifdef OVERFLOW_DEBUG
    retval = start_timer(500);
#else
    retval = start_timer(ESI->overflow.timer_ms);
#endif
  return(retval);
}

int _papi_hwi_stop_overflow_timer(ThreadInfo_t *thread, EventSetInfo_t *ESI)
{
  int retval = PAPI_OK;

  if (_papi_hwi_system_info.supports_hw_overflow == 0)
    retval = stop_timer();
  thread->event_set_overflowing = NULL;
  return(retval);
}

/* int _papi_portable_set_multiplex(EventSetInfo_t *ESI, papi_multiplex_option_t *ptr)
{
  return(PAPI_ESBSTR);
}

int _papi_portable_set_overflow(EventSetInfo_t *ESI, papi_overflow_option_t *ptr)
{
  return(PAPI_ESBSTR);
}

int _papi_portable_get_overflow(EventSetInfo_t *ESI, papi_overflow_option_t *ptr)
{
  memcpy(ptr,&ESI->overflow.option,sizeof(*ptr));
  return(PAPI_OK);
}

int _papi_portable_get_multiplex(EventSetInfo_t *ESI, papi_multiplex_option_t *ptr)
{
  memcpy(ptr,&ESI->multiplex.option,sizeof(*ptr));
  return(PAPI_ESBSTR);
}
*/

/*
  Hardware independent routines to support an opaque native event table.
  #define HAS_NATIVE_MAP for these functions to be active.
  Otherwise they return benign values (PAPI_ENOEVNT)
  These routines assume the existence of two hardware dependent routines:
    _papi_hwd_native_code_to_name()
    _papi_hwd_native_code_to_descr()
  A third routine is required to extract hardware dependent mapping info from the structure:
    _papi_hwd_native_code_to_bits()
  In addition, two more optional hardware dependent routines provide for the creation
  of new native events that may not be included in the distribution:
    _papi_hwd_encode_native()
    _papi_hwd_decode_native()
  These five routines provide the mapping from the opaque hardware dependent
  native event structure array to the higher level hardware independent interface.
*/

/* Returns PAPI_OK if native EventCode found, or PAPI_ENOEVNT if not;
   Used to enumerate the entire array, e.g. for native_avail.c */
int _papi_hwi_query_native_event(unsigned int EventCode)
{
#ifdef HAS_NATIVE_MAP
  char *name;

  if ((EventCode & NATIVE_MASK) &&((EventCode ^ NATIVE_MASK)<PAPI_MAX_NATIVE_EVENTS)){
    name = _papi_hwi_native_code_to_name(EventCode);
    if (name) return(PAPI_OK);
  }
#endif
  return(PAPI_ENOEVNT);
}

/* Converts an ASCII name into a native event code usable by other routines
   Returns code = 0 and PAPI_OK if name not found.
   This allows for sparse native event arrays */
int _papi_hwi_native_name_to_code(char *in, int *out)
{
#ifdef HAS_NATIVE_MAP
  char *name;
  int i;
  for(i=0;i<PAPI_MAX_NATIVE_EVENTS;i++){
    name = _papi_hwd_native_code_to_name(i | NATIVE_MASK);
    if (name != NULL) {
      if (strcasecmp(name,in) == 0) {
	*out = i | NATIVE_MASK;
	return(PAPI_OK);
      }
    }
    else {
      *out = 0;
      return(PAPI_OK);
    }
  }
#endif
  return(PAPI_ENOEVNT);
}


/* Returns event name based on native event code. 
   Returns NULL if name not found */
char *_papi_hwi_native_code_to_name(unsigned int EventCode)
{
#ifdef HAS_NATIVE_MAP
  if ((EventCode & NATIVE_MASK) &&((EventCode ^ NATIVE_MASK)<PAPI_MAX_NATIVE_EVENTS)){
    return(_papi_hwd_native_code_to_name(EventCode));
  }
#endif
  return(NULL);
}


/* Returns event description based on native event code.
   Returns NULL if description not found */
char *_papi_hwi_native_code_to_descr(unsigned int EventCode)
{
#ifdef HAS_NATIVE_MAP
  
  if ((EventCode & NATIVE_MASK) &&((EventCode ^ NATIVE_MASK)<PAPI_MAX_NATIVE_EVENTS)){
    return(_papi_hwd_native_code_to_descr(EventCode));
  }
#endif
  return(NULL);
}


/* The native event equivalent of PAPI_query_event_verbose */
int _papi_hwi_query_native_event_verbose(unsigned int EventCode, PAPI_preset_info_t *info)
{
#ifdef HAS_NATIVE_MAP

  if ((EventCode & NATIVE_MASK) &&((EventCode ^ NATIVE_MASK)<PAPI_MAX_NATIVE_EVENTS)){
    info->event_name = _papi_hwi_native_code_to_name(EventCode);
    if (info->event_name != NULL) {
      /* Fill in the info structure */
      info->event_code = EventCode;
      info->event_descr = _papi_hwi_native_code_to_descr(EventCode);
      info->event_label = NULL;
      info->event_note = NULL;
      info->avail = 1;	/* if we found it, it's available */
      info->flags = 0; /* not derived */
      return(PAPI_OK);
    }
  }
#endif
  return(PAPI_ENOEVNT);
}


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
 More Unix routines that I can't find in Windows
 This one returns a pseudo-random integer
 given an unsigned int seed.
*/
extern int rand_r (unsigned int *Seed)
{
	srand(*Seed);
	return (rand());
}

/*
  Another Unix routine that doesn't exist in Windows.
  Kevin uses it in the memory stuff, specifically in PAPI_get_dmem_info().
*/
extern int getpagesize(void)
{
  SYSTEM_INFO SystemInfo; 	// system information structure  

  GetSystemInfo(&SystemInfo);	
	return((int)SystemInfo.dwPageSize);
}

#endif /* _WIN32 */











