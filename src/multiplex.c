/* 
* File:    multiplex.c
* CVS:     $Id$
* Author:  John May
           ?
* Mods:    Philip Mucci
*          mucci@cs.utk.edu
*	   John May
*	   johnmay@llnl.gov
*/  

/* John May */
/* Lawrence Livermore National Laboratory */
/* MPX Version 1.0
 * UCRL-CODE-2001-007
 *
 * This work was produced at the University of California, Lawrence
 * Livermore National Laboratory (UC LLNL) under contract no.
 * W-7405-ENG-48 (Contract 48) between the U.S. Department of Energy 
 * (DOE) and The Regents of the University of California (University) 
 * for the operation of UC LLNL. The rights of the Federal Government
 * are reserved under Contract 48 subject to the restrictions agreed
 * upon by the DOE and University as allowed under DOE Acquisition
 * Letter 97-1. 
 *  
 *  
 *  
 * DISCLAIMER
 *  
 * This work was prepared as an account of work sponsored by an agency
 * of the United States Government. Neither the United States Government
 * nor the University of California nor any of their employees, makes
 * any warranty, express or implied, or assumes any liability or
 * responsibility for the accuracy, completeness, or usefulness of any
 * information, apparatus, product, or process disclosed, or represents
 * that its use would not infringe privately-owned rights.  Reference
 * herein to any specific commercial products, process, or service by
 * trade name, trademark, manufacturer or otherwise does not necessarily
 * constitute or imply its endorsement, recommendation, or favoring by
 * the United States Government or the University of California. The
 * views and opinions of authors expressed herein do not necessarily
 * state or reflect those of the United States Government or the
 * University of California, and shall not be used for advertising or
 * product endorsement purposes.
 *
 *  
 * NOTIFICATION OF COMMERCIAL USE
 *  
 * Commercialization of this product is prohibited without notifying the
 * Department of Energy (DOE) or Lawrence Livermore National Laboratory
 * (LLNL).
 */

#ifdef PTHREADS
#include <pthread.h>
#endif

#ifndef _WIN32
  #include SUBSTRATE
#else
  #include "win32.h"
#endif

#define MPX_SIGNAL PAPI_SIGNAL
#define MPX_ITIMER PAPI_ITIMER

/* Globals for this file. */

/* List of threads that are multiplexing. */

static Threadlist * tlist = NULL;

/* Number of threads that have been signaled */

static int threads_responding = 0;

static unsigned int randomseed;

#ifdef PTHREADS
static pthread_once_t mpx_once_control = PTHREAD_ONCE_INIT;
static pthread_mutex_t tlistlock;
static pthread_key_t master_events_key;
static pthread_key_t thread_record_key;
static MasterEvent *global_master_events;
static void *global_process_record;
#endif

/* Forward prototypes */

static void mpx_delete_events(MPX_EventSet *);
static int mpx_insert_events(MPX_EventSet *, int * event_list, int num_events, int domain, int granularity);
static void mpx_handler(int signal);

#if defined(ANY_THREAD_GETS_SIGNAL)
extern int (*thread_kill_fn)(int, int);
#endif

#ifdef _WIN32

static MMRESULT	mpxTimerID;	// unique ID for referencing this timer
static int mpx_time;

static void mpx_init_timers(int interval)
{
	/* Fill in the interval timer values now to save a
	 * little time later.
	 */
#ifdef OUTSIDE_PAPI
	interval = MPX_DEFAULT_INTERVAL;
#endif
	// interval is in usec & Windows needs msec resolution
	mpx_time = interval/1000;
}

void CALLBACK mpx_timer_callback(UINT wTimerID, UINT msg, 
    DWORD dwUser, DWORD dw1, DWORD dw2) 
{
	mpx_handler(0); 
} 


static int mpx_startup_itimer(void)
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
  mpxTimerID = timeSetEvent(mpx_time, wTimerRes, 
		mpx_timer_callback, (DWORD)NULL, TIME_PERIODIC);
  if(!mpxTimerID) return PAPI_ESYS;

  return(retval);
}

#define mpx_restore_signal()	// NOP on Windows

static void mpx_shutdown_itimer(void)
{
	if (timeKillEvent(mpxTimerID) != TIMERR_NOERROR)
		perror("setitimer(MPX_ITIMER)");
}

static void mpx_release(void)
{
	mpx_startup_itimer();
}

static void mpx_hold(void)
{
	mpx_shutdown_itimer();
}

#else

/* Timer stuff */

static struct itimerval itime;
static struct itimerval itimestop;
struct sigaction oaction;

static void mpx_init_timers(int interval)
{
	/* Fill in the interval timer values now to save a
	 * little time later.
	 */
#ifdef OUTSIDE_PAPI
	interval = MPX_DEFAULT_INTERVAL;
#endif

#ifdef REGENERATE
	/* Signal handler restarts the timer every time it runs */
	itime.it_interval.tv_sec = 0;
	itime.it_interval.tv_usec = 0;
	itime.it_value.tv_sec = 0;
	itime.it_value.tv_usec = interval;
#else
	/* Timer resets itself automatically */
	itime.it_interval.tv_sec = 0;
	itime.it_interval.tv_usec = interval;
	itime.it_value.tv_sec = 0;
	itime.it_value.tv_usec = interval;
#endif

	itimestop.it_interval.tv_sec = 0;
	itimestop.it_interval.tv_usec = 0;
	itimestop.it_value.tv_sec = 0;
	itimestop.it_value.tv_usec = 0;
}

static int mpx_startup_itimer(void)
{
	int retval;
	struct sigaction sigact;

	/* Set up the signal handler and the timer that triggers it */
	memset(&sigact, 0, sizeof(sigact));
	sigact.sa_handler = mpx_handler;
	retval = sigaction(MPX_SIGNAL, &sigact, &oaction);
	assert(retval == 0);

	retval = setitimer(MPX_ITIMER, &itime, NULL);
	if (retval != 0)
	  perror("setitimer(MPX_ITIMER)");
	assert(retval == 0);
	return(retval);
}

static void mpx_restore_signal(void)
{
	int retval;

	retval = sigaction(MPX_SIGNAL, &oaction, NULL);
	assert(retval == 0);
}

static void mpx_shutdown_itimer(void)
{
  if (setitimer(MPX_ITIMER, &itimestop, NULL) == -1)
    perror("setitimer(MPX_ITIMER)");
}

static void mpx_hold(void)
{
  sighold(MPX_SIGNAL);
}

static void mpx_release(void)
{
  sigrelse(MPX_SIGNAL);
}

#endif /* _WIN32 */

static MasterEvent *get_my_threads_master_event_list(void)
{
  Threadlist *t = tlist;
  unsigned long tid;

  if (thread_id_fn == NULL)
    return(tlist->head);

  tid = thread_id_fn();

  while (t)
    {
      if (t->pid == tid)
	return(t->head);
      t = t->next;
    }
  return(NULL);
}

static MPX_EventSet *mpx_malloc(Threadlist *t)
{
  MPX_EventSet *newset = (MPX_EventSet *)malloc(sizeof(MPX_EventSet));
  if (newset == NULL)
    return(NULL);

  newset->status = MPX_STOPPED;
  newset->mythr = t;
  newset->num_events = 0;
  newset->start_c = newset->stop_c = 0;
  memset(newset->start_hc, 0, PAPI_MPX_DEF_DEG * sizeof(long_long));
  memset(newset->start_values, 0, PAPI_MPX_DEF_DEG * sizeof(long_long));
  memset(newset->start_cycles, 0, PAPI_MPX_DEF_DEG * sizeof(long_long));
  memset(newset->stop_values, 0, PAPI_MPX_DEF_DEG * sizeof(long_long));
  memset(newset->stop_cycles, 0, PAPI_MPX_DEF_DEG * sizeof(long_long));
  return(newset);
}

int mpx_add_event(MPX_EventSet **mpx_events, int EventCode)
{
  MPX_EventSet *newset = *mpx_events;
  int retval, def_dom, def_grn, alloced_thread = 0, alloced_newset = 0;
  Threadlist *t;

  /* Get the global list of threads */

  _papi_hwd_lock();
  t = tlist;

  /* If there are no threads in the list at all, then allocate the new Threadlist */

  if (t == NULL)
    {
    new_thread:
      t = (Threadlist *)malloc(sizeof(Threadlist));
      if (t == NULL)
	{
	  _papi_hwd_unlock();
	  return(PAPI_ENOMEM);
	}

      /* If we're actually threaded, fill the 
       * field with the thread_id otherwise
       * use getpid() as a placeholder. */

      if (thread_id_fn)
	{
#ifdef MPX_DEBUG
	  fprintf(stderr,"New thread %x at %p\n",thread_id_fn(),t);
#endif
	  t->pid = thread_id_fn();
	}
      else 
	{
#ifdef MPX_DEBUG
	  fprintf(stderr,"New process %x at %p\n",getpid(),t);
#endif
	  t->pid = getpid();
	}

      /* Fill in the fields */

      t->head = NULL;
      t->cur_event = NULL;
      t->next = tlist;
      tlist = t;
#ifdef MPX_DEBUG
      fprintf(stderr,"New head is at %p(%lu).\n",tlist,tlist->pid);
#endif
      alloced_thread = 1;
    }
  else if (thread_id_fn)
    {

      /* If we are threaded, AND there exists threads in the list, 
       *  then try to find our thread in the list. */

      unsigned long tid = thread_id_fn();

      while (t)
	{
	  if (t->pid == tid)
	    {
#ifdef MPX_DEBUG
	      fprintf(stderr,"Found thread %x\n",t->pid);
#endif
	      break;
	    }
	  t = t->next;
	}

      /* Our thread is not in the list, so make a new
       * thread entry. */

      if (t == NULL)
	{
#ifdef MPX_DEBUG
	  fprintf(stderr,"New thread %lx\n",tid);
#endif
	  goto new_thread;
	}
    }

  /* Now t & tlist points to our thread, also at the head of the list */

  /* Allocate a the MPX_EventSet if necessary */

  if (newset == NULL)
    {
      newset = mpx_malloc(t);
      if (newset == NULL)
	{
	  _papi_hwd_unlock();
	  return(PAPI_ENOMEM);
	}
      alloced_newset = 1;
    }

  /* Now we're finished playing with the thread list */

  _papi_hwd_unlock();

  /* Removed newset->num_events++, moved to mpx_insert_events() */

  def_dom = PAPI_get_opt(PAPI_GET_DEFDOM, NULL);
  def_grn = PAPI_get_opt(PAPI_GET_DEFGRN, NULL);

  mpx_hold();

  /* Create PAPI events (if they don't already exist) and link
   * the new event set to them, add them to the master list for
     the thread, reset master event list for this thread */

  retval = mpx_insert_events(newset, &EventCode, 1, def_dom, def_grn);
  if (retval != PAPI_OK)
    {
      if (alloced_newset)
	{
	  free(newset);
	  newset = NULL;
	}
    }

  mpx_release();
  
  /* Output the new or existing EventSet */

  *mpx_events = newset;

  return(retval);
}

int mpx_remove_event(MPX_EventSet **mpx_events, int EventCode)
{
  return(PAPI_OK);
}

#ifdef MPX_DEBUG_TIMER
static long long lastcall;
#endif
#define MINCYCLES 100000
static void mpx_handler(int signal)
{
	int retval;
	MasterEvent * mev, * head;
	Threadlist *me = NULL;
#ifdef REGENERATE
	int lastthread;
#endif
#ifdef MPX_DEBUG_OVERHEAD
	long long usec;
	int didwork = 0;
	usec = PAPI_get_real_usec();
#endif
#ifdef MPX_DEBUG_TIMER
	long long thiscall;
#endif

	signal = signal;	/* unused */

#ifdef MPX_DEBUG
	if (thread_id_fn)
	  fprintf(stderr,"Handler in thread %x\n",thread_id_fn());
#endif

	/* This handler can be invoked either when a timer expires
	 * or when another thread in this handler responding to the
	 * timer signals other threads.  We have to distinguish
	 * these two cases so that we don't get infinite loop of 
	 * handler calls.  To do that, we look at the value of
	 * threads_responding.  We assume that only one thread can
	 * be active in this signal handler at a time, since the
	 * invoking signal is blocked while the handler is active.
	 * If threads_responding == 0, the current thread caught
	 * the original timer signal.  (This thread may not have
	 * any active event lists itself, though.)  This first
	 * thread sends a signal to each of the other threads in
	 * our list of threads that have master events lists.  If
	 * threads_responding != 0, then this thread was signaled
	 * by another thread.  We decrement that value and look
	 * for an active events.  threads_responding should
	 * reach zero when all active threads have handled their
	 * signal.  It's probably possible for a thread to die
	 * before it responds to a signal; if that happens,
	 * threads_responding won't reach zero until the next
	 * timer signal happens.  Then the signalled thread won't
	 * signal any other threads.  If that happens only
	 * occasionally, there should be no harm.  Likewise if
	 * a new thread is added that fails to get signalled.
	 * As for locking, we have to lock this list to prevent
	 * another thread from modifying it, but if *this* thread
	 * is trying to update the list (from another function) and
	 * is signaled while it holds the lock, we will have deadlock.
	 * Therefore, noninterrupt functions that update *this* list
	 * must disable the signal that invokes this handler.
	 */

#ifdef PTHREADS
	_papi_hwd_lock();

	if( threads_responding == 0 ) {	/* this thread caught the timer sig */
		/* Signal the other threads with event lists */
#ifdef MPX_DEBUG_TIMER
		thiscall = PAPI_get_real_usec();
		fprintf( stderr, "last signal was %lld usec ago\n",
			thiscall - lastcall);
		lastcall = thiscall;
#endif
#ifdef MPX_DEBUG_SIGNALS
		fprintf(stderr,"%x caught it\n", self);
#endif
		for( t = tlist; t != NULL; t = t->next ) {
			if( pthread_equal(t->thr, self) == 0 ) {
				++threads_responding;
				retval = pthread_kill(t->thr, MPX_SIGNAL);
				assert(retval == 0);
#ifdef MPX_DEBUG_SIGNALS
				fprintf(stderr,"%x signaling %x\n",
						self, t->thr);
#endif
			}
		}
	} else {
#ifdef MPX_DEBUG_SIGNALS
		fprintf(stderr, "%x was tapped, tr = %d\n",
				self, threads_responding);
#endif
		--threads_responding;
	}
#ifdef REGENERATE
	lastthread = (threads_responding == 0);
#endif
	_papi_hwd_unlock();
#endif

	/* See if this thread has an active event list */
	head = get_my_threads_master_event_list();
	if( head != NULL ) {

		/* Get the thread header for this master event set.  It's
		 * always in the first record of the set (and maybe in others)
		 * if any record in the set is active.
		 */
		me = head->mythr;

		/* Find the event that's currently active, stop and read
		 * it, then start the next event in the list.
		 * No need to lock the list because other functions
		 * disable the timer interrupt before they update the list.
		 */
		if( me != NULL && me->cur_event != NULL ) {
			long_long counts[2];
			MasterEvent * cur_event = me->cur_event;
			long_long cycles;

			retval = PAPI_stop(cur_event->papi_event, counts);
			assert(retval == PAPI_OK);
#ifdef MPX_DEBUG
			fprintf(stderr, "retval %d cure %p I'm %x\n",
				retval, cur_event, me->pid);
			fprintf(stderr, "counts[0] = %lld counts[1] = %lld\n",
						counts[0], counts[1]);
#endif

			cur_event->count += counts[0];
			cycles = (cur_event->pi.event_type == PAPI_TOT_CYC)
				? counts[0] : counts[1];

			if( retval == PAPI_OK ) {
			/* If it's a rate, count occurrences & average later */
				if ( cur_event->is_a_rate ) {
					/* Make sure we ran long enough to
					 * get a useful measurement (otherwise
					 * potentially inaccurate rate
					 * measurements get averaged in with
					 * the same weight as longer, more
					 * accurate ones.)
					 */
					if( cycles >= MINCYCLES ) {
						cur_event->cycles += 1;
					} else {
						cur_event->count -= counts[0];
					}
				} else {
					cur_event->cycles += cycles;
				}
				me->total_c += cycles;
				cur_event->handler_count++;
			} else {
				fprintf(stderr, "%x retval = %d, skipping\n",
						me->pid, retval );
				fprintf(stderr,
					"%x value = %lld cycles = %lld\n\n",
					me->pid, cur_event->count,
					cur_event->cycles);
			}

#ifdef MPX_DEBUG
			fprintf(stderr, "%x value = %lld cycles = %lld\n\n",
				me->pid, cur_event->count, cur_event->cycles);
#endif
			/* Start running the next event; look for the
			 * next one in the list that's marked active.
			 * It's possible that this event is the only
			 * one active; if so, we should restart it,
			 * but only after considerating all the other
			 * possible events.
			 */
			if( cycles < MINCYCLES || retval != PAPI_OK ) {
				mev = cur_event;
			} else {
			  
			for(mev = ((cur_event->next == NULL) ? head : cur_event->next);
			    mev != cur_event;
			    mev = (mev->next == NULL) ? head : mev->next )
			  {

				/* Found the next one to start */
				if( mev->active ) {
					me->cur_event = mev;
					retval = PAPI_start(
						me->cur_event->papi_event);
					assert(retval == PAPI_OK);

					/* A hack that makes the if statement
					 * after this loop evaluate false;
					 * otherwise, it would try to
					 * start this event a second time
					 * if it's the only event in the set.
					 */
					cur_event = NULL;
					break;
				}
			}
			}

			if( mev == cur_event ) {	/* wrapped around */
				retval = PAPI_start(me->cur_event->papi_event);
				assert(retval == PAPI_OK);
			}
#ifdef MPX_DEBUG_OVERHEAD
			didwork = 1;
#endif
		}
	}
#ifdef ANY_THREAD_GETS_SIGNAL
	else {
	  Threadlist *t;
#ifdef MPX_DEBUG_TIMER
	  fprintf(stderr,"nothing to do in thread %x\n", (*thread_id_fn)());
#endif
	  for( t = tlist; t != NULL; t = t->next ) {
#ifdef MPX_DEBUG_TIMER
	    fprintf(stderr,"forwarding signal to thread %x\n", t->pid);
#endif
	    retval = (*thread_kill_fn)(t->pid, MPX_SIGNAL);
	    assert(retval == 0);
	  }
	}
#endif

#ifdef REGENERATE
	/* Regenerating the signal each time through has the
	 * disadvantage that if any thread ever drops a signal,
	 * the whole time slicing system will stop.  Using
	 * an automatically regenerated signal may have the
	 * disadvantage that a new signal can arrive very
	 * soon after all the threads have finished handling
	 * the last one, so the interval may be too small for
	 * accurate data collection.  However, using the
	 * MIN_CYCLES check above should alleviate this.
	 */
	/* Reset the timer once all threads have responded */
	if( lastthread ) {
		retval = setitimer(MPX_ITIMER, &itime, NULL);
		assert(retval == 0);
#ifdef MPX_DEBUG_TIMER
		fprintf(stderr, "timer restarted by %x\n", me->pid);
#endif
	}
#endif

#ifdef MPX_DEBUG_OVERHEAD
	usec = PAPI_get_real_usec() - usec;
	printf("handler %x did %swork in %lld usec\n",
			self, (didwork ? "" : "no "), usec);
#endif
}
	
int MPX_add_events(MPX_EventSet ** mpx_events, int * event_list, int num_events)
{
  int i, retval = PAPI_OK;

  for (i=0;i<num_events;i++)
    {
      retval = mpx_add_event(mpx_events,event_list[i]);
      if (retval != PAPI_OK)
	return(retval);
    }
  return(retval);
}

int MPX_start(MPX_EventSet * mpx_events)
{
	int retval;
	int i;
	long_long cycles_this_slice = 0;
	Threadlist * t;
	long_long prev_total_c;
	
	t = mpx_events->mythr;

	mpx_hold();

	/* Make all events in this set active, and for those
	 * already active, get the current count and cycles.
	 */
	for( i = 0; i < mpx_events->num_events; i++ ) {
		MasterEvent * mev = mpx_events->mev[i];

		long_long prev_count, prev_cycles;
		prev_count = mpx_events->stop_values[i]
			- mpx_events->start_values[i];
		prev_cycles = mpx_events->stop_cycles[i]
			- mpx_events->start_cycles[i];

		if( mev->active++ ) {
			mpx_events->start_values[i] = mev->count;
			mpx_events->start_cycles[i] = mev->cycles;

			mpx_events->start_hc[i] = mev->handler_count;

			/* If this happens to be the currently-running
			 * event, add in the current amounts from this
			 * time slice.  If it's a rate, though, don't
			 * bother since the event might not have been
			 * running long enough to get an accurate count.
			 */
			if( mev == t->cur_event
					&& !(t->cur_event->is_a_rate)) {
				long_long values[2];
				retval = PAPI_read(mev->papi_event, values);
				assert(retval == PAPI_OK);
				mpx_events->start_values[i] += values[0];
				cycles_this_slice =
					((mev->pi.event_type == PAPI_TOT_CYC)
					? values[0] : values[1]);

				mpx_events->start_cycles[i] +=
					cycles_this_slice;
			}
		} else {
			/* The = 0 isn't actually necessary; we only need
			 * to sync up the mpx event to the master event,
			 * but it seems safe to set the mev to 0 here, and
			 * that gives us a change to avoid (very unlikely)
			 * rollover problems for events used repeatedly over
			 * a long time.
			 */
			mpx_events->start_values[i] = mev->count = 0;
			mpx_events->start_cycles[i] = mev->cycles = 0;

			mpx_events->start_hc[i] = mev->handler_count = 0;
		}
		/* Adjust start value to include events and cycles
		 * counted previously for this event set.
		 */
		mpx_events->start_values[i] -= prev_count;
		mpx_events->start_cycles[i] -= prev_cycles;
	}

	mpx_events->status = MPX_RUNNING;

	prev_total_c = mpx_events->stop_c - mpx_events->start_c;

	/* Start first counter if one isn't already running */
	if( t->cur_event == NULL ) {
		/* Pick an events at random to start. */
		int index = (rand_r(&randomseed) % mpx_events->num_events);
		t->cur_event = mpx_events->mev[index];
		t->total_c = 0;
		mpx_events->start_c = 0;
		retval = PAPI_start(mpx_events->mev[index]->papi_event);
		/* if( retval ) pm_error("papi start", retval); */
		assert(retval == PAPI_OK);
	} else {
		/* If an event is already running, record the starting cycle
		 * count for mpx_events, which is the accumlated cycle count
		 * for the master event set plus the cycles for this time
		 * slice.
		 */
		mpx_events->start_c = t->total_c + cycles_this_slice;
	}

	/* Adjust the total cycle count for this event set to include
	 * cycles counted in previous instantiations.
	 */
	mpx_events->start_c -= prev_total_c;

	mpx_release();

	retval = mpx_startup_itimer();

	assert(retval == 0);

	return PAPI_OK;
}

int MPX_read(MPX_EventSet * mpx_events, long_long * values)
{
	int i;
	int retval;
	long_long last_value[2];
	long_long cycles_this_slice = 0;
	long_long time_interval;
	MasterEvent * cur_event;
	Threadlist * thread_data;

	if( mpx_events->status == MPX_RUNNING ) {

		/* Hold timer interrupts while we read values */
		mpx_hold();

		thread_data = mpx_events->mythr;
		cur_event = thread_data->cur_event;

		/* Save the current counter values and get
		 * the lastest data for the current event
		 */
		for( i = 0; i < mpx_events->num_events; i++ ) {
			MasterEvent * mev = mpx_events->mev[i];

			mpx_events->stop_values[i] = mev->count;
			mpx_events->stop_cycles[i] = mev->cycles;

			/* Read data only if it's not a rate measurement */
			if( mev == cur_event && !(mev->is_a_rate) ) {
				retval = PAPI_read(cur_event->papi_event,
						last_value);
				assert(retval == PAPI_OK);
				cycles_this_slice = (cur_event->pi.event_type
						 == PAPI_TOT_CYC)
						? last_value[0] : last_value[1];
				mpx_events->stop_values[i] += last_value[0];
				mpx_events->stop_cycles[i] += cycles_this_slice;
			}
		}

		mpx_events->stop_c = thread_data->total_c + cycles_this_slice;

		/* Restore the interrupt */
		mpx_release();
	}

        /* Compute the total time (in cycles) this measurement has run */
        time_interval = mpx_events->stop_c - mpx_events->start_c;

	/* Scale all the values and store in user array. */
	for( i = 0; i < mpx_events->num_events; i++ ) {
		long_long elapsed_cycles = mpx_events->stop_cycles[i]
			- mpx_events->start_cycles[i];

		/* Prevent division-by-zero if counters are zero */
		if( elapsed_cycles == 0 ) {
			values[i] = 0;
		}
		/* For rates, cycles contains the number of measurements,
		 * not the number of cycles, so just divide to compute
		 * an average value.  This assumes that the rate was
		 * constant over the whole measurement period.
		 */
		else if( mpx_events->mev[i]->is_a_rate ) {
			values[i] = (long_long)(
				(mpx_events->stop_values[i]
					- mpx_events->start_values[i])
				/ elapsed_cycles );

		/* For regular events, scale the value by the proportion
		 * of the total number of cycles during which this counter
		 * was active.
		 */
		} else {
			values[i] = (long_long)(
				(mpx_events->stop_values[i]
					- mpx_events->start_values[i])
					* (double)time_interval
				/ elapsed_cycles );
#if 0
			printf("events: %lld interval %lld: cycles %lld\n",
				(mpx_events->stop_values[i]
					- mpx_events->start_values[i]),
				time_interval, elapsed_cycles);
#endif
		}
	}

	return PAPI_OK;
}

int MPX_reset(MPX_EventSet * mpx_events)
{
	int i;
	int retval;
	long_long last_value[2];
	long_long cycles_this_slice = 0;
	MasterEvent * cur_event;
	Threadlist * thread_data;

	/* Disable timer interrupt */ 
	mpx_hold();

	thread_data = mpx_events->mythr;
	cur_event = thread_data->cur_event;

	/* Make counters read zero by setting the start values
	 * to the current counter values.
	 */
	for( i = 0; i < mpx_events->num_events; i++ ) {
		MasterEvent * mev = mpx_events->mev[i];

		mpx_events->start_values[i] = mev->count;
		mpx_events->start_cycles[i] = mev->cycles;
		if( mev == cur_event && !(mev->is_a_rate) ) {
			retval = PAPI_read(cur_event->papi_event,
					last_value);
			assert(retval == PAPI_OK);
			cycles_this_slice =
				(cur_event->pi.event_type == PAPI_TOT_CYC)
					? last_value[0] : last_value[1];
			mpx_events->start_values[i] += last_value[0];
			mpx_events->start_cycles[i] += cycles_this_slice;
		}
	}

	/* Set the start time for this set to the current cycle count */
	mpx_events->start_c = thread_data->total_c + cycles_this_slice;

	/* Restart the interrupt */
	mpx_release();
	
	return PAPI_OK;
}

int MPX_stop(MPX_EventSet * mpx_events, long_long * values)
{
	int i;
	int retval;
	long_long last_value[2];
	long_long cur_event_cycles, final_count, final_cycles;
	MasterEvent * cur_event, * head;
	Threadlist * thr;

	if( mpx_events == NULL || values == NULL ) return PAPI_EINVAL;

	if( mpx_events->status != MPX_RUNNING ) return PAPI_ENOTRUN;

	/* Block timer interrupts */
	mpx_hold();

	/* Get the master event list for this thread. */
	head = get_my_threads_master_event_list();

	/* Get this threads data structure */
	thr = head->mythr;

	/* Run through all the events decrement their activity counters. */
	for( i = 0; i < mpx_events->num_events; i++ ) {
		--mpx_events->mev[i]->active;
	}

	/* Now update the current event pointer for this thread; this is
	 * done in a separate loop from above to ensure that all the
	 * events' activity counters are up to date before we choose
	 * the next current event.
	 */
 	cur_event = thr->cur_event;
	for( i = 0; i < mpx_events->num_events; i++ ) {
		MasterEvent * tmp, * mev = mpx_events->mev[i];

		/* Find the current event and see if it's active; if not,
		 * move to the next event.  Otherwise, there's nothing
		 * more to do, just let it run.
		 */
		if( mev == cur_event ) {
			/* Event is now inactive; stop it and update master
			 * event set counters.
			 */
			if( mev->active == 0  ) {
				retval = PAPI_stop(mev->papi_event, last_value);
				assert(retval == PAPI_OK);

				cur_event_cycles
					= (mev->pi.event_type == PAPI_TOT_CYC)
						? last_value[0] : last_value[1];

				/* Include last measurement only if it's not
				 * a rate.  Rates measured over partial time
				 * intervals are potentially inaccurate.
				 */
				if( !(mev->is_a_rate) ) {
					mev->count += last_value[0];
					mev->cycles += cur_event_cycles;
				}
				thr->total_c += cur_event_cycles;
				mpx_events->stop_c = thr->total_c;
				final_count = mev->count;
				final_cycles = mev->cycles;

				/* Now find a new cur_event */
				for(tmp = ((cur_event->next == NULL)
						? head : cur_event->next);
				    tmp != cur_event;
				    tmp = (tmp->next == NULL)
				   		? head : tmp->next ) {
					/* Found the next one to start */
					if( tmp->active ) {
						thr->cur_event = tmp;
						retval = PAPI_start(tmp->
								papi_event);
						assert(retval == PAPI_OK);
						break;
					}
				}

				/* If we wrap around to tmp == cur_event, there 
				 * are no active events in the list, so there 
				 * is no current event.
				 */
				if( tmp == cur_event ) {
					thr->cur_event = NULL;
				}
			/* Event is still active in some other event set,
			 * so just read its current value.
			 */
			} else {
				/* Current event is still active in another
				 * running event set.
				 */
				if( !(mev->is_a_rate) ) {
					retval = PAPI_read(mev->papi_event,
							last_value);
					assert(retval == PAPI_OK);
					final_count = mev->count
						+ last_value[0];
					cur_event_cycles
						= (mev->pi.event_type
								== PAPI_TOT_CYC)
						? last_value[0] : last_value[1];

					final_cycles = mev->cycles +
						cur_event_cycles;
					mpx_events->stop_c = thr->total_c
						+ cur_event_cycles;
				} else {
					
					final_count = mev->count;
					final_cycles = mev->cycles;
				}
			}
		} else {
			final_count = mev->count;
			final_cycles = mev->cycles;
		}
		/* Get the latest count for each event */
		mpx_events->stop_values[i] = final_count;
		mpx_events->stop_cycles[i] = final_cycles;
	}

	mpx_events->status = MPX_STOPPED;

	if (thr->cur_event == NULL)
	  {
	    mpx_shutdown_itimer();
	  }

	/* Restore the timer (for other event sets that may be running) */
	mpx_release();

	/* Read the current data, then stop counting */
	retval = MPX_read(mpx_events, values);
	assert(retval == PAPI_OK);

	return PAPI_OK;
}

int MPX_cleanup(MPX_EventSet ** mpx_events)
{
#ifdef PTHREADS
	int retval;
#endif
	MPX_EventSet * tmp = *mpx_events;

	if( mpx_events == NULL || *mpx_events == NULL 
			|| (*mpx_events)->status == MPX_RUNNING )
		return PAPI_EINVAL;

	mpx_hold();

	/* Remove master events from this event set and from
	 * the master list, if necessary.
	 */
	mpx_delete_events(tmp);

	mpx_release();

	/* Free all the memory */

	free(tmp);

	*mpx_events = NULL;

	return PAPI_OK;
}

void MPX_shutdown(void)
{
#if 0
	Threadlist * t, * nextthr;
#endif

	mpx_shutdown_itimer();

#if 0
	PAPI_lock();

	for( t = tlist; t != NULL; t = nextthr ) {
		assert(t->cur_event == NULL);	/* should be no active events */
		nextthr = t->next;
		free(t);
	}

	PAPI_unlock();
#endif

	mpx_restore_signal();
}

int MPX_set_opt(int option, PAPI_option_t * ptr, MPX_EventSet * mpx_events)
{
#ifdef PTHREADS
	int retval;
#endif
#ifdef OLD
	int i;
	int granularity, domain;
	int * event_list;
#endif

	return(PAPI_EINVAL);

#ifdef OLD
	if( ptr == NULL || mpx_events == NULL ) return PAPI_EINVAL;

	switch(option) {
		/* options that are not per-eventset */
		case PAPI_SET_INHERIT:
			return PAPI_set_opt(option, ptr);
			break;

		/* options that are per-eventset */
		/* Changing domain or granularity causes the events
		 * in the set to be measured differently.  Conceivably,
		 * one might want to accumulate events measured
		 * differently into the same counter, but it's easier
		 * not to allow it.  So we'll handle new options by
		 * removing the old events and adding new ones with
		 * the new options.
		 */
		case PAPI_SET_DOMAIN:
		case PAPI_SET_GRANUL:
			/* Event set must not be running */
			if( mpx_events->status == MPX_RUNNING )
				return PAPI_EINVAL;

			/* Determine the option values to use */
			if( option == PAPI_SET_DOMAIN ) {
				domain = ptr->domain.domain;
				granularity
					= mpx_events->mev[0]->pi.granularity;
			} else if( option == PAPI_SET_GRANUL ) {
				domain = mpx_events->mev[0]->pi.domain;
				granularity = ptr->granularity.granularity;
			}

			/* If no change needed, just return */
			if( mpx_events->mev[0]->pi.domain == domain 
				&& mpx_events->mev[0]->pi.granularity
						== granularity ) 
				return PAPI_OK;

			/* Make a list of the events in the current set */
			event_list = (int *)malloc(mpx_events->num_events
					* sizeof(int));
			assert( event_list != NULL );
			for( i = 0; i < mpx_events->num_events; i++ )
				event_list[i] =
					mpx_events->mev[i]->pi.event_type;


			mpx_hold();

			/* Remove the events from the master list and the current set*/
			mpx_delete_events(mpx_events);

			/* Put the events back in the event set with the
			 * new options.
			 */
			mpx_insert_events(mpx_events, event_list, i,
					domain, granularity);

			mpx_release();

			free(event_list);

			break;
	}
	return PAPI_OK;
#endif
}

#ifdef ANY_THREAD_GETS_SIGNAL
void _papi_hwi_lookup_thread_symbols(void)
{
  int retval;
  char *error;
  void *symbol = NULL, *handle = dlopen(NULL,RTLD_LAZY);
  assert(handle != NULL);
#if defined(sun)
  symbol = dlsym(handle,"thr_self");
  if (symbol == NULL)
    symbol = dlsym(handle,"pthread_self");
#elif defined(_AIX)
  symbol = dlsym(handle,"pthread_self");
#endif
  error = dlerror();
  if ((error == NULL) && (symbol))
    {
      retval = PAPI_thread_init((unsigned long (*)(void))symbol, 0);
      assert(retval == 0);
    }
#if defined(sun)
  thread_kill_fn = (int (*)(int, int))dlsym(handle,"thr_kill");
  if (thread_kill_fn == NULL)
    thread_kill_fn = (int (*)(int, int))dlsym(handle,"pthread_kill");
#elif defined(_AIX)
  thread_kill_fn = (int (*)(int, int))dlsym(handle,"pthread_kill");
#endif
  assert(thread_kill_fn != NULL);
  dlclose(handle);
}
#endif

int mpx_init(int interval)
{
#ifdef PTHREADS
	int retval;
#endif

	mpx_init_timers(interval);

#ifdef OUTSIDE_PAPI
	/* Only want to initialize PAPI if it's not done already by
	 * some external library.  A crude test (but the best one I
	 * can think of) is to check for the existence of an  event
	 * that should be there.  If it is, PAPI is initialized;
	 * otherwise is isn't.
	 */
	retval = PAPI_query_event(PAPI_TOT_CYC);
	if( retval == PAPI_ENOEVNT ) {
		retval = PAPI_library_init(PAPI_VER_CURRENT);
		assert(retval == PAPI_VER_CURRENT);
#ifdef PTHREADS
		retval = PAPI_thread_init(
				(unsigned long(*)(void))pthread_self, 0);
		assert(retval == PAPI_OK);
#endif
	}
#endif

#ifdef PTHREADS
	retval = pthread_key_create(&master_events_key, NULL);
	assert(retval == 0);

	retval = pthread_key_create(&thread_record_key, NULL);
	assert(retval == 0);

	retval = pthread_mutex_init(&tlistlock, NULL);
	assert(retval == 0);
#endif
#if defined(ANY_THREAD_GETS_SIGNAL)
	_papi_hwi_lookup_thread_symbols();
#endif
	return(PAPI_OK);
}

/* Inserts a list of events into the master event list, 
   and add's new mev pointers to the MPX_EventSet. */

/* MUST BE CALLED WITH THE TIMER INTERRUPT DISABLED */

static int mpx_insert_events(MPX_EventSet *mpx_events, int * event_list,
		int num_events, int domain, int granularity)
{
	int i, retval=0, num_events_success = 0;
	MasterEvent * mev;
#if 0
	PAPI_option_t options;
#endif
	MasterEvent **head = &mpx_events->mythr->head;

	/* For each event, see if there is already a corresponding
	 * event in the master set for this thread.  If not, add it.
	 */
	for( i = 0; i < num_events; i++ ) {

		/* Look for a matching event in the master list */
		for( mev = *head; mev != NULL; mev = mev->next ) {
			if( mev->pi.event_type == event_list[i]
				&& mev->pi.domain == domain
				&& mev->pi.granularity == granularity ) break;
		}

		/* No matching event in the list; add a new one */
		if( mev == NULL ) {
			mev = (MasterEvent *)malloc(sizeof(MasterEvent));
			assert(mev != NULL);

			mev->pi.event_type = event_list[i];
			mev->pi.domain = domain;
			mev->pi.granularity = granularity;
			mev->uses = mev->active = 0;
			mev->count = mev->cycles = 0;
			mev->papi_event = PAPI_NULL;
			/* Scale rate measurements differently from counts */
			mev->is_a_rate = (event_list[i] == PAPI_FLOPS 
					|| event_list[i] == PAPI_IPS ); 

			retval = PAPI_create_eventset(&(mev->papi_event));
			if (retval != PAPI_OK)
			  {
#ifdef MPX_DEBUG
			    fprintf(stderr,"Event %d could not be counted.\n",event_list[i]);
#endif
			  bail:
			    PAPI_cleanup_eventset(&(mev->papi_event));
			    PAPI_destroy_eventset(&(mev->papi_event));
			    free(mev);
			    mev = NULL;
			    break;
			  }

			retval = PAPI_add_event(&(mev->papi_event),event_list[i]);
			if (retval != PAPI_OK)
			  {
#ifdef MPX_DEBUG
			    fprintf(stderr,"Event %d could not be counted.\n",event_list[i]);
#endif
			    goto bail;
			  }

			/* Always count total cycles so we can scale results.
			 * If user just requested cycles, don't add that event again. */

			if (event_list[i] != PAPI_TOT_CYC) 
			  {
			    retval = PAPI_add_event(&(mev->papi_event), PAPI_TOT_CYC); 
			    if (retval != PAPI_OK)
			      {
#ifdef MPX_DEBUG
				fprintf(stderr,"PAPI_TOT_CYC could not be counted at the same time.\n");
#endif
				goto bail;
			      }
			  }

			/* Set the options for the event set */
#if 0
			options.domain.eventset = mev->papi_event;
			options.domain.domain = domain;
			retval = PAPI_set_opt(PAPI_SET_DOMAIN, &options);
			if (retval != PAPI_OK)
			  {
#ifdef MPX_DEBUG
			    fprintf(stderr,"PAPI_set_opt(PAPI_SET_DOMAIN) failed.\n");
#endif
			    goto bail;
			  }
#endif
#if 0
			options.granularity.eventset = mev->papi_event;
			options.granularity.granularity = granularity;
			retval = PAPI_set_opt(PAPI_SET_GRANUL, &options);
			if (retval != PAPI_OK)
			  {
#ifdef MPX_DEBUG
			    fprintf(stderr,"PAPI_set_opt(PAPI_SET_GRANUL) failed.\n");
#endif
			    goto bail;
			  }
#endif

			/* Chain the event set into the 
			 * master list of event sets used in
			 * multiplexing. */

			mev->next = *head;
			*head = mev;
		}

		/* If we created a new event set, or we found a matching
		 * eventset already in the list, then add the pointer in
		 * the master list to this threads list. Then we bump the
		 * number of successfully added events. */

		mpx_events->mev[mpx_events->num_events+num_events_success] = mev;
		mpx_events->mev[mpx_events->num_events+num_events_success]->uses++;
		num_events_success++;
	}

	/* Always be sure the head master event points to the thread */
	if ( *head != NULL ) {
		(*head)->mythr = mpx_events->mythr;
	}
#ifdef MPX_DEBUG
	fprintf(stderr,"%d of %d events were added.\n",num_events_success,num_events);
#endif
	mpx_events->num_events += num_events_success;
	return(retval);
}

/* Remove revove master events from an mpx event set (and from the
 * master event set for this thread, if the events are unused).
 * MUST BE CALLED WITH THE SIGNAL HANDLER DISABLED
 */

static void mpx_delete_events(MPX_EventSet * mpx_events)
{
	int i;
	MasterEvent * mev, * lastmev = NULL, * nextmev;
	MasterEvent ** head = &mpx_events->mythr->head;
	Threadlist * thr = (*head == NULL) ? NULL : (*head)->mythr;

	/* First decrement the reference counter for each master
	 * event in this event set, then see if the master events
	 * can be deleted.
	 */
	for( i = 0; i < mpx_events->num_events; i++ ) {
		mev = mpx_events->mev[i];
		--mev->uses;
		
		/* If it's no longer used, it should not be active! */
		assert( mev->uses || !(mev->active) );
	}

	/* Clean up and remove unused master events. */
	for( mev = *head; mev != NULL; mev = nextmev ) {
		nextmev = mev->next;	/* get link before mev is freed*/
		if( !mev->uses ) {
			if( lastmev == NULL ) { /* this was the head event */
				*head = nextmev;
			} else {
				lastmev->next = nextmev;
			}
			PAPI_cleanup_eventset(&(mev->papi_event));
			PAPI_destroy_eventset(&(mev->papi_event));
			free(mev);
		} else {
			lastmev = mev;
		}
	}

	/* Always be sure the head master event points to the thread */
	if( *head != NULL ) {
		(*head)->mythr = thr;
	}
}
