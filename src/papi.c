/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
* Mods:    Kevin London
*	   london@cs.utk.edu
* Mods:    Per Ekman
*          pek@pdc.kth.se
* Mods:    <your name here>
*          <your email address>
*/

#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
#define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"

#include "papiStrings.h"        /* for language independent string support. */

/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

/* papi_internal.c */

#ifdef DEBUG
extern int _papi_hwi_debug;
#define papi_return(a) return(_papi_hwi_debug_handler(a))
#else
#define papi_return(a) return(a)
#endif


#ifdef ANY_THREAD_GETS_SIGNAL
extern int (*_papi_hwi_thread_kill_fn) (int, int);
#endif

extern unsigned long int (*_papi_hwi_thread_id_fn) (void);
extern int _papi_hwi_error_level;
extern char *_papi_hwi_errStr[];
extern PAPI_debug_handler_t _papi_hwi_debug_handler;
extern PAPI_overflow_handler_t _papi_hwi_dummy_handler;
extern papi_mdi_t _papi_hwi_system_info;

/* papi_data.c */

extern PAPI_event_info_t _papi_hwi_presets[];
extern int init_retval;
extern int init_level;

/* Defined by the substrate */
extern hwi_preset_data_t _papi_hwi_preset_data[];

/*****************************/
/* END EXTERNAL DECLARATIONS */
/*****************************/


/********************/
/*  BEGIN LOCALS    */
/********************/

/********************/
/*    END LOCALS    */
/********************/


int PAPI_thread_init(unsigned long int (*id_fn) (void), int flag)
{
   int changed_id_fn = 0;
/* Thread support not implemented on Alpha/OSF because the OSF pfm
 * counter device driver does not support per-thread counters.
 * When this is updated, we can remove this ifdef -KSL
 */
#if defined(__ALPHA) && defined(__osf__)
   papi_return(PAPI_ESBSTR);
#endif
   if (default_master_thread == NULL)
      papi_return(PAPI_EINVAL);

   /* They had better be compatible thread packages if you change thread thread packages without
      shutting down PAPI first. This means tid of package a = tid of package b */

   if ((id_fn != NULL) && (_papi_hwi_thread_id_fn != NULL)
       && (id_fn != _papi_hwi_thread_id_fn))
      changed_id_fn = 1;

   _papi_hwi_thread_id_fn = id_fn;

   /* When we shutdown threading, let's clear both function pointers. */

#if defined(ANY_THREAD_GETS_SIGNAL)
   if (id_fn == NULL)
      _papi_hwi_thread_kill_fn = NULL;
#endif

   /* Now change the master event's thread id from getpid() to the
      real thread id */

   /* By default, the initial master eventset has TID of getpid(). This will
      get changed if the user enables threads with PAPI_thread_init(). */

   if ((_papi_hwi_thread_id_fn) && (changed_id_fn == 0)) {
      default_master_thread->tid = (*_papi_hwi_thread_id_fn) ();
      _papi_hwi_insert_in_thread_list(default_master_thread);
   }

   papi_return(PAPI_OK);
}

unsigned long int PAPI_thread_id(void)
{
   if (_papi_hwi_thread_id_fn != NULL)
      return ((*_papi_hwi_thread_id_fn) ());
   else
      return (PAPI_EINVAL);
}

/* Thread Functions */

/* 
 * Notify PAPI that a thread has 'appeared'
 * We lookup the thread, if it does not exist we create it
 */
int PAPI_register_thread(void)
{
   ThreadInfo_t *thread = _papi_hwi_lookup_in_thread_list();
   int retval;

   if (thread == NULL) {
      retval = _papi_hwi_initialize_thread(&thread);
      if (retval)
         papi_return(retval);
      _papi_hwi_insert_in_thread_list(thread);
   }
   papi_return(PAPI_OK);
}

/*
 * Return a pointer to the stored thread information.
 */
int PAPI_get_thr_specific(int tag, void **ptr)
{
   ThreadInfo_t *thread;
   if (tag < 0 || tag > PAPI_MAX_THREAD_STORAGE)
      papi_return(PAPI_EINVAL);
   thread = _papi_hwi_lookup_in_thread_list();
   *ptr = thread->thread_storage[tag];
   papi_return(PAPI_OK);
}

/*
 * Store a pointer to memory provided by the thread
 */
int PAPI_set_thr_specific(int tag, void *ptr)
{
   ThreadInfo_t *thread;
   if (tag < 0 || tag > PAPI_MAX_THREAD_STORAGE)
      papi_return(PAPI_EINVAL);
   thread = _papi_hwi_lookup_in_thread_list();
   if (thread == NULL)
      papi_return(PAPI_EBUG);
   thread->thread_storage[tag] = ptr;
   papi_return(PAPI_OK);
}


int PAPI_library_init(int version)
{
   int i, j, tmp = 0;
   char *var;

#ifdef DEBUG
#ifdef _WIN32                   /* don't want to define an environment variable... */
   _papi_hwi_debug = 1;
#else
   _papi_hwi_debug = 0;
   if ((var = getenv("PAPI_DEBUG")) && var != NULL) {
      _papi_hwi_debug |= DEBUG_ON;
      if (strlen(var) > 1 && var[1] == 'x')
         sscanf(var, "%x", &tmp);
      else
         tmp = atoi(var);
      _papi_hwi_debug |= tmp;
   }
   if (getenv("PAPI_DEBUG_SUBSTRATE"))
      _papi_hwi_debug |= DEBUG_SUBSTRATE;
   if (getenv("PAPI_DEBUG_API"))
      _papi_hwi_debug |= DEBUG_API;
   if (getenv("PAPI_DEBUG_INTERNAL"))
      _papi_hwi_debug |= DEBUG_INTERNAL;
   if (getenv("PAPI_DEBUG_THREADS"))
      _papi_hwi_debug |= DEBUG_THREADS;
   if (getenv("PAPI_DEBUG_MULTIPLEX"))
      _papi_hwi_debug |= DEBUG_MULTIPLEX;
   if (getenv("PAPI_DEBUG_OVERFLOW"))
      _papi_hwi_debug |= DEBUG_OVERFLOW;
#endif
#endif

   if (init_retval != DEADBEEF)
      papi_return(init_retval);

   if (version != PAPI_VER_CURRENT) {
      init_retval = PAPI_EINVAL;
      papi_return(PAPI_EINVAL);
   }

   _papi_hwd_lock_init();

   if (_papi_hwi_mdi_init() != PAPI_OK) {
      papi_return(PAPI_EINVAL);
   }
   tmp = _papi_hwd_init_global();
   if (tmp) {
      init_retval = tmp;
      papi_return(init_retval);
   }

   if (_papi_hwi_allocate_eventset_map()) {
      _papi_hwd_shutdown_global();
      init_retval = PAPI_ENOMEM;
      papi_return(init_retval);
   }

   DBG((stderr, "Initialize master thread.\n"));
   tmp = _papi_hwi_initialize_thread(&default_master_thread);
   if (tmp) {
      _papi_hwd_shutdown_global();
      init_retval = tmp;
      papi_return(init_retval);
   }

   for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
      if (_papi_hwi_presets[i].symbol)  /* If the preset is part of the API */
         for (_papi_hwi_presets[i].count = 0, j = 0; j < MAX_COUNTER_TERMS; j++) {
            if (_papi_hwi_preset_data[i].native[j] == PAPI_NULL) break;
            _papi_hwi_presets[i].count++;
         }
   }

   init_level = PAPI_LOW_LEVEL_INITED;
   return (init_retval = PAPI_VER_CURRENT);
}

int PAPI_query_event(int EventCode)
{
   if (EventCode & PRESET_MASK) {
      EventCode &= PRESET_AND_MASK;
      if (EventCode >= PAPI_MAX_PRESET_EVENTS)
         papi_return(PAPI_ENOTPRESET);

      if (_papi_hwi_presets[EventCode].count)
         papi_return(PAPI_OK);
      else
         papi_return(PAPI_ENOEVNT);
   }

   if (EventCode & NATIVE_MASK) {
      papi_return(_papi_hwi_query_native_event(EventCode));
   }

   papi_return(PAPI_ENOTPRESET);
}

int PAPI_get_event_info(int EventCode, PAPI_event_info_t * info)
{
   if (info == NULL)
      papi_return(PAPI_EINVAL);

   if (EventCode & PRESET_MASK) {
      EventCode &= PRESET_AND_MASK;
      if (EventCode >= PAPI_MAX_PRESET_EVENTS)
         papi_return(PAPI_ENOTPRESET);

      if (_papi_hwi_presets[EventCode].symbol[0]) {
         memcpy(info, &_papi_hwi_presets[EventCode], sizeof(PAPI_event_info_t));
         papi_return(PAPI_OK);
      } else
         papi_return(PAPI_ENOEVNT);
   }

   if (EventCode & NATIVE_MASK) {
      papi_return(_papi_hwi_get_native_event_info(EventCode, info));
   }

   papi_return(PAPI_ENOTPRESET);
}

int PAPI_event_code_to_name(int EventCode, char *out)
{
   if (out == NULL)
      papi_return(PAPI_EINVAL);

   if (EventCode & PRESET_MASK) {
      EventCode &= PRESET_AND_MASK;
      if ((EventCode >= PAPI_MAX_PRESET_EVENTS)
          || (_papi_hwi_presets[EventCode].symbol == NULL))
         papi_return(PAPI_ENOTPRESET);

      strncpy(out, _papi_hwi_presets[EventCode].symbol, PAPI_MAX_STR_LEN);
      papi_return(PAPI_OK);
   }

   if (EventCode & NATIVE_MASK) {
      if (_papi_hwi_native_code_to_name(EventCode) != NULL) {
         strncpy(out, _papi_hwi_native_code_to_name(EventCode), PAPI_MAX_STR_LEN);
         papi_return(PAPI_OK);
      }
   }

   papi_return(PAPI_ENOEVNT);
}

int PAPI_event_name_to_code(char *in, int *out)
{
   int i;

   if ((in == NULL) || (out == NULL))
      papi_return(PAPI_EINVAL);

   if (strncmp(in, "PAPI", 4) == 0) {
      for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
         if ((_papi_hwi_presets[i].symbol)
             && (strcasecmp(_papi_hwi_presets[i].symbol, in) == 0)) {
            *out = _papi_hwi_presets[i].event_code;
            papi_return(PAPI_OK);
         }
      }
   }
   papi_return(_papi_hwi_native_name_to_code(in, out));
}

/* Updates EventCode to next valid value, or returns error; 
  modifer can specify {all / available} for presets, or other values for native tables 
  and may be platform specific (Major groups / all mask bits; P / M / E chip, etc) */
int PAPI_enum_event(int *EventCode, int modifier)
{
   int i = *EventCode;

   if (i & PRESET_MASK) {
      i &= PRESET_AND_MASK;
      while (++i < PAPI_MAX_PRESET_EVENTS) {
         if ((!modifier) || (_papi_hwi_presets[i].count)) {
            *EventCode = i | PRESET_MASK;
            if (_papi_hwi_presets[i].symbol[0])
               return (PAPI_OK);
            else
               return (PAPI_ENOEVNT);
         }
      }
   } else if (i & NATIVE_MASK) {
      return (_papi_hwd_ntv_enum_events((unsigned int *) EventCode, modifier));
   }
   return (PAPI_ENOEVNT);
}

int PAPI_create_eventset(int *EventSet)
{
   ThreadInfo_t *master = _papi_hwi_lookup_in_thread_list();
   int retval;
   if (master == NULL) {
      DBG((stderr, "PAPI_create_eventset(%p): new thread found\n", (void *) EventSet));
      retval = _papi_hwi_initialize_thread(&master);
      if (retval)
         return (retval);
      _papi_hwi_insert_in_thread_list(master);
   }

   return (_papi_hwi_create_eventset(EventSet, master));
}

int PAPI_add_pevent(int EventSet, int code, void *inout)
{
   EventSetInfo_t *ESI;

   /* Is the EventSet already in existence? */

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   /* Of course, it must be stopped in order to modify it. */

   if (!(ESI->state & PAPI_STOPPED))
      papi_return(PAPI_EISRUN);

   /* No multiplexing pevents. */

   if (ESI->state & PAPI_MULTIPLEXING)
      papi_return(PAPI_EINVAL);

   /* Now do the magic. */

   return (_papi_hwi_add_pevent(ESI, code, inout));
}

int PAPI_add_event(int EventSet, int EventCode)
{
   EventSetInfo_t *ESI;

   /* Is the EventSet already in existence? */

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   /* Of course, it must be stopped in order to modify it. */

   if (ESI->state & PAPI_RUNNING)
      papi_return(PAPI_EISRUN);

   /* Now do the magic. */

   papi_return(_papi_hwi_add_event(ESI, EventCode));
}

int PAPI_remove_event(int EventSet, int EventCode)
{
   EventSetInfo_t *ESI;
   int i;

   /* check for pre-existing ESI */

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   /* Of course, it must be stopped in order to modify it. */

   if (!(ESI->state & PAPI_STOPPED))
      papi_return(PAPI_EISRUN);

   /* if the state is PAPI_OVERFLOWING, you must first call
      PAPI_overflow with threshold=0 to remove the overflow flag */

   if (ESI->state & PAPI_OVERFLOWING) 
      papi_return(PAPI_EINVAL);


   /* Now do the magic. */

   papi_return(_papi_hwi_remove_event(ESI, EventCode));
}

int PAPI_destroy_eventset(int *EventSet)
{
   EventSetInfo_t *ESI;

   /* check for pre-existing ESI */

   if (EventSet == NULL)
      papi_return(PAPI_EINVAL);

   ESI = _papi_hwi_lookup_EventSet(*EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   if (!(ESI->state & PAPI_STOPPED))
      papi_return(PAPI_EISRUN);

   if (ESI->NumberOfEvents)
      papi_return(PAPI_EINVAL);

   _papi_hwi_remove_EventSet(ESI);
   *EventSet = PAPI_NULL;

   papi_return(PAPI_OK);
}

/* simply checks for valid EventSet, calls substrate start() call */

int PAPI_start(int EventSet)
{
   int retval;
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;

   DBG((stderr, "PAPI_start\n"));
   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if (!(ESI->state & PAPI_STOPPED))
      papi_return(PAPI_EISRUN);

   if (ESI->NumberOfEvents < 1)
      papi_return(PAPI_EINVAL);

   /* If multiplexing is enabled for this eventset,
      call John May's code. */

   if (ESI->state & PAPI_MULTIPLEXING) {
      retval = MPX_start(ESI->multiplex);
      if (retval != PAPI_OK)
         papi_return(retval);

      /* Update the state of this EventSet */

      ESI->state ^= PAPI_STOPPED;
      ESI->state |= PAPI_RUNNING;

      return (PAPI_OK);
   }

   /* Short circuit this stuff if there's nothing running */

/* commented out in Phil's papiv3
  if (thread->multistart.num_runners == 0)
    {
      for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
	{
	  ESI->hw_start[i] = 0;
	  thread->hw_start[i] = 0;
	}
    }
*/

   /* If overflowing is enabled, turn it on */

   if (ESI->state & PAPI_OVERFLOWING) {
      retval = _papi_hwi_start_overflow_timer(thread, ESI);
      if (retval < PAPI_OK)
         papi_return(retval);
   }

   if (ESI->state & PAPI_PROFILING)
      thread->event_set_profiling = ESI;

   /* Merge the control bits from the new EventSet into the active counter config. */

   retval = _papi_hwd_start(&thread->context, &ESI->machdep);
   if (retval != PAPI_OK)
      papi_return(retval);

   /* Update the state of this EventSet */

   ESI->state ^= PAPI_STOPPED;
   ESI->state |= PAPI_RUNNING;

   /* Update the number of active EventSets for this thread */

/* commented out in Phil's papiv3
  thread->multistart.num_runners++;
*/

   DBG((stderr, "PAPI_start returns %d\n", retval));
   return (retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */

int PAPI_stop(int EventSet, long_long * values)
{
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;
   int retval;

   DBG((stderr, "PAPI_stop\n"));
   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if (!(ESI->state & PAPI_RUNNING))
      papi_return(PAPI_ENOTRUN);

   if (ESI->state & PAPI_PROFILING) {
      if (_papi_hwi_system_info.supports_hw_profile) {
         retval = _papi_hwd_stop_profiling(thread, ESI);
         if (retval < PAPI_OK)
            papi_return(retval);
      }
   }

   /* If multiplexing is enabled for this eventset,
      call John May's code. */

   if (ESI->state & PAPI_MULTIPLEXING) {
      retval = MPX_stop(ESI->multiplex, values);
      if (retval != PAPI_OK)
         papi_return(retval);

      /* Update the state of this EventSet */

      ESI->state ^= PAPI_RUNNING;
      ESI->state |= PAPI_STOPPED;

      return (PAPI_OK);
   }

   /* Read the current counter values into the EventSet */

   retval = _papi_hwi_read(&thread->context, ESI, ESI->sw_stop);
   if (retval != PAPI_OK)
      papi_return(retval);

   /* If overflowing is enabled, turn it off */

   if (ESI->state & PAPI_OVERFLOWING) {
      ESI->overflow.count = 0;
      retval = _papi_hwi_stop_overflow_timer(thread, ESI);
      if (retval < PAPI_OK)
         papi_return(retval);
   }

   /* Remove the control bits from the active counter config. */

   retval = _papi_hwd_stop(&thread->context, &ESI->machdep);
   if (retval != PAPI_OK)
      papi_return(retval);
   if (values)
      memcpy(values, ESI->sw_stop, ESI->NumberOfEvents * sizeof(long_long));

   /* Update the state of this EventSet */

   ESI->state ^= PAPI_RUNNING;
   ESI->state |= PAPI_STOPPED;

   /* Update the number of active EventSets for this thread */

/* commented out in Phil's papiv3
  thread->multistart.num_runners --;
*/

#if defined(DEBUG)
   {
      int i;
      for (i = 0; i < ESI->NumberOfEvents; i++)
         DBG((stderr, "PAPI_stop ESI->sw_stop[%d]:\t%llu\n", i, ESI->sw_stop[i]));
   }
#endif

   DBG((stderr, "PAPI_stop returns %d\n", retval));

   return (retval);
}

int PAPI_reset(int EventSet)
{
   int retval = PAPI_OK;
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if (ESI->state & PAPI_RUNNING) {
      if (ESI->state & PAPI_MULTIPLEXING)
         retval = MPX_reset(ESI->multiplex);
      else {
         /* If we're not the only one running, then just
            read the current values into the ESI->start
            array. This holds the starting value for counters
            that are shared. */

         retval = _papi_hwd_reset(&thread->context, &ESI->machdep);

         if ((ESI->state & PAPI_OVERFLOWING) &&
             (_papi_hwi_system_info.supports_hw_overflow))
            ESI->overflow.count = 0;

         if ((ESI->state & PAPI_PROFILING) && (_papi_hwi_system_info.supports_hw_profile))
            ESI->profile.overflowcount = 0;
      }
   } else {
      memset(ESI->sw_stop, 0x00, ESI->NumberOfEvents * sizeof(long_long));
   }

   DBG((stderr, "PAPI_reset returns %d\n", retval));
   papi_return(retval);
}

int PAPI_read(int EventSet, long_long * values)
{
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;
   int retval = PAPI_OK;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if (values == NULL)
      papi_return(PAPI_EINVAL);

   if (ESI->state & PAPI_RUNNING) {
      if (ESI->state & PAPI_MULTIPLEXING)
         retval = MPX_read(ESI->multiplex, values);
      else
         retval = _papi_hwi_read(&thread->context, ESI, values);
      if (retval != PAPI_OK)
         papi_return(retval);
   } else {
      memcpy(values, ESI->sw_stop, ESI->NumberOfEvents * sizeof(long_long));
   }

#if defined(DEBUG)
   {
      int i;
      for (i = 0; i < ESI->NumberOfEvents; i++)
         DBG((stderr, "PAPI_read values[%d]:\t%llu\n", i, values[i]));
   }
#endif

   DBG((stderr, "PAPI_read returns %d\n", retval));
   return (retval);
}

int PAPI_accum(int EventSet, long_long * values)
{
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;
   int i, retval;
   long_long a, b, c;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if (values == NULL)
      papi_return(PAPI_EINVAL);

   if (ESI->state & PAPI_RUNNING) {
      if (ESI->state & PAPI_MULTIPLEXING)
         retval = MPX_read(ESI->multiplex, ESI->sw_stop);
      else
         retval = _papi_hwi_read(&thread->context, ESI, ESI->sw_stop);
      if (retval != PAPI_OK)
         papi_return(retval);
   }

   for (i = 0; i < ESI->NumberOfEvents; i++) {
      a = ESI->sw_stop[i];
      b = values[i];
      c = a + b;
      values[i] = c;
   }

   papi_return(PAPI_reset(EventSet));
}

int PAPI_write(int EventSet, long_long * values)
{
   int retval = PAPI_OK;
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if (values == NULL)
      papi_return(PAPI_EINVAL);

   if (ESI->state & PAPI_RUNNING) {
      retval = _papi_hwd_write(&thread->context, &ESI->machdep, values);
      if (retval != PAPI_OK)
         return (retval);
   }

   memcpy(ESI->hw_start, values, _papi_hwi_system_info.num_cntrs * sizeof(long_long));

   return (retval);
}

/*  The function PAPI_cleanup removes a stopped EventSet from existence. */

int PAPI_cleanup_eventset(int EventSet)
{
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;

   /* Is the EventSet already in existence? */

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   /* Of course, it must be stopped in order to modify it. */

   if (ESI->state & PAPI_RUNNING)
      papi_return(PAPI_EISRUN);

   /* Now do the magic */

   papi_return(_papi_hwi_cleanup_eventset(ESI));
}

int PAPI_multiplex_init(void)
{
   int retval;

   retval = mpx_init(PAPI_MPX_DEF_US);
   papi_return(retval);
}

int PAPI_state(int EventSet, int *status)
{
   EventSetInfo_t *ESI;

   if (status == NULL)
      papi_return(PAPI_EINVAL);

   /* check for good EventSetIndex value */

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   /*read status FROM ESI->state */

   *status = ESI->state;

   papi_return(PAPI_OK);
}

int PAPI_set_debug(int level)
{
   extern int _papi_hwi_error_level;
   switch (level) {
   case PAPI_QUIET:
   case PAPI_VERB_ESTOP:
   case PAPI_VERB_ECONT:
      _papi_hwi_error_level = level;
      papi_return(PAPI_OK);
   default:
      papi_return(PAPI_EINVAL);
   }
}

int PAPI_set_multiplex(int *EventSet)
{
   PAPI_option_t mpx;

   if (EventSet == NULL)
      papi_return(PAPI_EINVAL);

   mpx.multiplex.eventset = *EventSet;
   mpx.multiplex.us = PAPI_MPX_DEF_US;
   mpx.multiplex.max_degree = PAPI_MPX_DEF_DEG;

   return (PAPI_set_opt(PAPI_MULTIPLEX, &mpx));
}


int PAPI_set_opt(int option, PAPI_option_t * ptr)
{
   _papi_int_option_t internal;
   int retval;
   ThreadInfo_t *thread;

   if (ptr == NULL)
      papi_return(PAPI_EINVAL);

   memset(&internal, 0x0, sizeof(_papi_int_option_t));

   switch (option) {
   case PAPI_MAXMEM:
      {
         papi_return(_papi_hwd_setmaxmem());
      }
   case PAPI_MULTIPLEX:
      {
         EventSetInfo_t *ESI;

         if (ptr->multiplex.us < 1)
            papi_return(PAPI_EINVAL);
         if (ptr->multiplex.max_degree <= _papi_hwi_system_info.num_cntrs) {
            papi_return(PAPI_OK);
         }
         ESI = _papi_hwi_lookup_EventSet(ptr->multiplex.eventset);
         if (ESI == NULL)
            papi_return(PAPI_ENOEVST);
         if (!(ESI->state & PAPI_STOPPED))
            papi_return(PAPI_EISRUN);
         if (ESI->state & PAPI_MULTIPLEXING)
            papi_return(PAPI_EINVAL);

         papi_return(_papi_hwi_convert_eventset_to_multiplex(ESI));
      }
   case PAPI_DEBUG:
      papi_return(PAPI_set_debug(ptr->debug.level));
   case PAPI_DEFDOM:
      {
         int dom = ptr->defdomain.domain;
         if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
            papi_return(PAPI_EINVAL);

         thread = _papi_hwi_lookup_in_thread_list();

/* commented out in Phil's papiv3
	if (thread->multistart.num_runners)
          papi_return(PAPI_EISRUN);
*/

         /* Try to change the domain of the eventset in the hardware */

         internal.defdomain.defdomain = dom;
         retval = _papi_hwd_ctl(&thread->context, PAPI_DEFDOM, &internal);
         if (retval < PAPI_OK)
            papi_return(retval);

         /* Change the domain of the master eventset in this thread */

         thread->domain = dom;

         /* Change the global structure. This should be removed but is
            necessary since the _papi_hwd_init_control_state function 
            (formerly init_config) in the substrates
            gets information from the global structure instead of
            per-thread information. */

         _papi_hwi_system_info.default_domain = dom;

         return (retval);
      }
   case PAPI_DOMAIN:
      {
         int dom = ptr->domain.domain;
         if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
            papi_return(PAPI_EINVAL);

         internal.domain.ESI = _papi_hwi_lookup_EventSet(ptr->domain.eventset);
         if (internal.domain.ESI == NULL)
            papi_return(PAPI_ENOEVST);
         thread = internal.domain.ESI->master;

         if (!(internal.domain.ESI->state & PAPI_STOPPED))
            papi_return(PAPI_EISRUN);

         /* Try to change the domain of the eventset in the hardware */

         internal.domain.domain = dom;
         internal.domain.eventset = ptr->domain.eventset;
         retval = _papi_hwd_ctl(&thread->context, PAPI_DOMAIN, &internal);
         if (retval < PAPI_OK)
            papi_return(retval);

         /* Change the domain of the eventset in the library */

         internal.domain.ESI->domain.domain = dom;

         return (retval);
      }
#if 0
   case PAPI_GRANUL:
      {
         int grn = ptr->granularity.granularity;

         if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
            papi_return(PAPI_EINVAL);

         internal.granularity.ESI = _papi_hwi_lookup_EventSet(ptr->granularity.eventset);
         if (internal.granularity.ESI == NULL)
            papi_return(PAPI_ENOEVST);

         internal.granularity.granularity = grn;
         internal.granularity.eventset = ptr->granularity.eventset;
         retval = _papi_hwd_ctl(NULL, PAPI_GRANUL, &internal);
         if (retval < PAPI_OK)
            return (retval);

         internal.granularity.ESI->granularity.granularity = grn;
         return (retval);
      }
   case PAPI_INHERIT:
      {
         EventSetInfo_t *tmp = _papi_hwi_lookup_in_thread_list();
         if (tmp == NULL)
            return (PAPI_EINVAL);

         internal.inherit.inherit = ptr->inherit.inherit;
         internal.inherit.master = tmp;

         retval = _papi_hwd_ctl(tmp, PAPI_INHERIT, &internal);
         if (retval < PAPI_OK)
            return (retval);

         tmp->inherit.inherit = ptr->inherit.inherit;
         return (retval);
      }
#endif
   default:
      papi_return(PAPI_EINVAL);
   }
}

int PAPI_num_hwctrs(void)
{
   return (PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
}

int PAPI_get_multiplex(int EventSet)
{
   PAPI_option_t popt;
   int retval;

   popt.multiplex.eventset = EventSet;
   retval = PAPI_get_opt(PAPI_MULTIPLEX, &popt);
   if (retval < 0)
      retval = 0;
   return retval;
}


int PAPI_get_opt(int option, PAPI_option_t * ptr)
{
   switch (option) {
   case PAPI_MULTIPLEX:
      {
         EventSetInfo_t *ESI;

         ESI = _papi_hwi_lookup_EventSet(ptr->multiplex.eventset);
         if (ESI == NULL)
            papi_return(PAPI_ENOEVST);
         return (ESI->state & PAPI_MULTIPLEXING) != 0;
      }
      break;
   case PAPI_PRELOAD:
      strncpy(ptr->preload.lib_preload_env,
              _papi_hwi_system_info.exe_info.preload_info.lib_preload_env,
              PAPI_MAX_STR_LEN);
      ptr->preload.lib_preload_sep =
          _papi_hwi_system_info.exe_info.preload_info.lib_preload_sep;
      strncpy(ptr->preload.lib_dir_env,
              _papi_hwi_system_info.exe_info.preload_info.lib_dir_env, PAPI_MAX_STR_LEN);
      ptr->preload.lib_dir_sep = _papi_hwi_system_info.exe_info.preload_info.lib_dir_sep;
      break;
   case PAPI_DEBUG:
      ptr->debug.level = _papi_hwi_error_level;
      ptr->debug.handler = _papi_hwi_debug_handler;
      break;
   case PAPI_CLOCKRATE:
      return ((int) _papi_hwi_system_info.hw_info.mhz);
   case PAPI_MAX_CPUS:
      return (_papi_hwi_system_info.hw_info.ncpu);
   case PAPI_MAX_HWCTRS:
      return (_papi_hwi_system_info.num_cntrs);
   case PAPI_DEFDOM:
      return (_papi_hwi_system_info.default_domain);
   case PAPI_DEFGRN:
      return (_papi_hwi_system_info.default_granularity);
#if 0
   case PAPI_INHERIT:
      {
         EventSetInfo_t *tmp;
         tmp = _papi_hwi_lookup_in_thread_list();
         if (tmp == NULL)
            return (PAPI_EINVAL);

         return (tmp->inherit.inherit);
      }
   case PAPI_GRANUL:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      return (_papi_hwi_get_granularity(&ptr->granularity));
#endif
   case PAPI_SHLIBINFO:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      _papi_hwd_update_shlib_info();
      ptr->shlib_info = &_papi_hwi_system_info.shlib_info;
      break;
   case PAPI_EXEINFO:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      ptr->exe_info = &_papi_hwi_system_info.exe_info;
      break;
   case PAPI_HWINFO:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      ptr->hw_info = &_papi_hwi_system_info.hw_info;
      break;
   case PAPI_DOMAIN:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      return (_papi_hwi_get_domain(&ptr->domain));
   default:
      papi_return(PAPI_EINVAL);
   }
   papi_return(PAPI_OK);
}

int PAPI_num_hw_counters(void)
{
   return (PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
}

int PAPI_num_events(int EventSet)
{
   EventSetInfo_t *ESI;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (!ESI)
      papi_return(PAPI_ENOEVST);

#ifdef DEBUG
   /* Not necessary */
   if (ESI->NumberOfEvents == 0)
      papi_return(PAPI_EINVAL);
#endif

   return (ESI->NumberOfEvents);
}

void PAPI_shutdown(void)
{
   DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
   EventSetInfo_t *ESI;
   int i, j = 0, status;
   ThreadInfo_t *master;

   if (init_retval == DEADBEEF) {
      fprintf(stderr, PAPI_SHUTDOWN_str);
      return;
   }

   MPX_shutdown();

   master = _papi_hwi_lookup_in_thread_list();

   /* Count number of running EventSets AND */
   /* Stop any running EventSets in this thread */

 again:
   for (i = 0; i < map->totalSlots; i++) {
      ESI = map->dataSlotArray[i];
      if (ESI) {
         PAPI_state(i, &status);
         if (ESI->state & PAPI_RUNNING) {
            if (ESI->master == master) {
               PAPI_stop(i, NULL);
               PAPI_cleanup_eventset(i);
            } else
               j++;
         }
      }
   }

   /* No locking required, we're just waiting for the others
      to call shutdown or stop their eventsets. */

   if (j != 0) {
      fprintf(stderr, PAPI_SHUTDOWN_SYNC_str);
#if _WIN32
      Sleep(1);
#elif _CRAYT3E
      sleep(1);
#else
      usleep(1000);
#endif
      j = 0;
      goto again;
   }

   /* Here call shutdown on the other threads */

   _papi_hwi_shutdown_the_thread_list();
   _papi_hwi_cleanup_thread_list();

   /* Clean up thread stuff */

   PAPI_thread_init(NULL, 0);

   /* Free up some memory */

   free(map->dataSlotArray);
   memset(map, 0x0, sizeof(DynamicArray_t));

   /* Shutdown the entire substrate */

   _papi_hwd_shutdown_global();

   /* Clean up process stuff */

   memset(&_papi_hwi_system_info, 0x0, sizeof(_papi_hwi_system_info));

   /* Now it is safe to call re-init */

   init_retval = DEADBEEF;
}

char *PAPI_strerror(int errorCode)
{
   if ((errorCode > 0) || (-errorCode > PAPI_NUM_ERRORS))
      return (NULL);

   return ((char *) _papi_hwi_errStr[-errorCode]);
}

int PAPI_perror(int code, char *destination, int length)
{
   char *foo;

   foo = PAPI_strerror(code);
   if (foo == NULL)
      papi_return(PAPI_EINVAL);

   if (destination && (length >= 0))
      strncpy(destination, foo, length);
   else
      fprintf(stderr, "%s\n", foo);

   papi_return(PAPI_OK);
}

/* This function sets up an EventSet such that when it is PAPI_start()'ed, it
   begins to register overflows. This EventSet may only have multiple events
   in it, but only 1 can be an overflow trigger. Subsequent calls to PAPI_overflow
   replace earlier calls. To turn off overflow, set the threshold to 0 */

int PAPI_overflow(int EventSet, int EventCode, int threshold, int flags,
                  PAPI_overflow_handler_t handler)
{
   int retval, index, event_counter, i;
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
      papi_return(PAPI_EISRUN);

   if ((index = _papi_hwi_lookup_EventCodeIndex(ESI, EventCode)) < 0)
      papi_return(PAPI_ENOEVNT);

   if (threshold < 0)
      papi_return(PAPI_EINVAL);

   /* We do not support derived events in overflow */
   if (ESI->EventInfoArray[index].derived)
      papi_return(PAPI_EINVAL);

/* the first time to call PAPI_overflow function */
   if (!(ESI->state & PAPI_OVERFLOWING)) {
      if (handler == NULL)
         papi_return(PAPI_EINVAL);
      if (threshold == 0)
         papi_return(PAPI_EINVAL);
   }
   if (threshold > 0 && ESI->overflow.event_counter >= MAX_COUNTERS)
      papi_return(PAPI_ECNFLCT);

   if (threshold == 0) {
      for (i = 0; i < ESI->overflow.event_counter; i++) {
         if (ESI->overflow.EventCode[i] == EventCode)
            break;
      }
      /* EventCode not found */
      if (i == ESI->overflow.event_counter)
         papi_return(PAPI_EINVAL);
      /* compact these arrays */
      while (i < ESI->overflow.event_counter - 1) {
         ESI->overflow.deadline[i] = ESI->overflow.deadline[i + 1];
         ESI->overflow.threshold[i] = ESI->overflow.threshold[i + 1];
         ESI->overflow.EventIndex[i] = ESI->overflow.EventIndex[i + 1];
         ESI->overflow.EventCode[i] = ESI->overflow.EventCode[i + 1];
         i++;
      }
      ESI->overflow.deadline[i] = 0;
      ESI->overflow.threshold[i] = 0;
      ESI->overflow.EventIndex[i] = 0;
      ESI->overflow.EventCode[i] = 0;

      ESI->overflow.event_counter--;
   } else {
      ESI->overflow.event_counter++;
      event_counter = ESI->overflow.event_counter;
      ESI->overflow.deadline[event_counter - 1] = threshold;
      ESI->overflow.threshold[event_counter - 1] = threshold;
      ESI->overflow.EventIndex[event_counter - 1] = index;
      ESI->overflow.EventCode[event_counter - 1] = EventCode;
   }
   ESI->overflow.flags = flags;
   ESI->overflow.handler = handler;
   ESI->overflow.count = 0;

   /* Set up the option structure for the low level */

   if (_papi_hwi_system_info.supports_hw_overflow) {
/*
    retval = _papi_hwd_set_overflow(ESI, &ESI->overflow);
*/
      retval = _papi_hwd_set_overflow(ESI, index, threshold);
      if (retval < PAPI_OK)
         papi_return(retval);
   } else
      ESI->overflow.timer_ms = PAPI_ITIMER_MS;


   /* Toggle the overflow flag */
   if ((ESI->overflow.event_counter == 1 && threshold > 0) ||
       (ESI->overflow.event_counter == 0 && threshold == 0))
      ESI->state ^= PAPI_OVERFLOWING;

   papi_return(PAPI_OK);
}

int PAPI_sprofil(PAPI_sprofil_t * prof, int profcnt, int EventSet, int EventCode,
                 int threshold, int flags)
{
   EventSetInfo_t *ESI;
/*
  EventSetProfileInfo_t opt = { 0, };
*/
   int retval, index, i;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
      papi_return(PAPI_EISRUN);

   if ((index = _papi_hwi_lookup_EventCodeIndex(ESI, EventCode)) < 0)
      papi_return(PAPI_ENOEVNT);

   /* We do not support derived events in overflow */
   if (ESI->EventInfoArray[index].derived)
      papi_return(PAPI_EINVAL);

   if (threshold < 0)
      papi_return(PAPI_EINVAL);

   /* the first time to call PAPI_sprofil */
   if (!(ESI->state & PAPI_PROFILING)) {
      if (threshold == 0)
         papi_return(PAPI_EINVAL);
   }
   if (threshold > 0 && ESI->profile.event_counter >= MAX_COUNTERS)
      papi_return(PAPI_ECNFLCT);

   if (threshold == 0) {
      for (i = 0; i < ESI->profile.event_counter; i++) {
         if (ESI->profile.EventCode[i] == EventCode)
            break;
      }
      /* EventCode not found */
      if (i == ESI->profile.event_counter)
         papi_return(PAPI_EINVAL);
      /* compact these arrays */
      while (i < ESI->profile.event_counter - 1) {
         ESI->profile.prof[i] = ESI->profile.prof[i + 1];
         ESI->profile.count[i] = ESI->profile.count[i + 1];
         ESI->profile.threshold[i] = ESI->profile.threshold[i + 1];
         ESI->profile.EventIndex[i] = ESI->profile.EventIndex[i + 1];
         ESI->profile.EventCode[i] = ESI->profile.EventCode[i + 1];
         i++;
      }
      ESI->profile.prof[i] = NULL;
      ESI->profile.count[i] = 0;
      ESI->profile.threshold[i] = 0;
      ESI->profile.EventIndex[i] = 0;
      ESI->profile.EventCode[i] = 0;

      ESI->profile.event_counter--;
   } else {
      i = ESI->profile.event_counter;
      ESI->profile.prof[i] = prof;
      ESI->profile.count[i] = profcnt;
      ESI->profile.threshold[i] = threshold;
      ESI->profile.EventIndex[i] = index;
      ESI->profile.EventCode[i] = EventCode;
      ESI->profile.event_counter++;
   }

   if (flags & ~(PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM | PAPI_PROFIL_WEIGHTED
                 | PAPI_PROFIL_COMPRESS | PAPI_PROFIL_BUCKET_16 |
                 PAPI_PROFIL_BUCKET_32 | PAPI_PROFIL_BUCKET_64))
      papi_return(PAPI_EINVAL);

   /* Set up the option structure for the low level */

   ESI->profile.flags = flags;

   if (_papi_hwi_system_info.supports_hw_profile)
      retval = _papi_hwd_set_profile(ESI, index, threshold);
   else
      retval = PAPI_overflow(EventSet, EventCode, threshold, 0, _papi_hwi_dummy_handler);

   if (retval < PAPI_OK)
      return (retval);

   /* Toggle profiling flag */
   if ((ESI->profile.event_counter == 1 && threshold > 0) ||
       (ESI->profile.event_counter == 0 && threshold == 0))
      ESI->state ^= PAPI_PROFILING;

   papi_return(PAPI_OK);
}

int PAPI_profil(void *buf, unsigned bufsiz, unsigned long offset,
                unsigned scale, int EventSet, int EventCode, int threshold, int flags)
{
   if (scale > 65536 || scale < 0x0002)
      papi_return(PAPI_EINVAL);

   if (threshold > 0) {
      PAPI_sprofil_t *prof;

      prof = (PAPI_sprofil_t *) malloc(sizeof(PAPI_sprofil_t));
      memset(prof, 0x0, sizeof(PAPI_sprofil_t));
      prof->pr_base = buf;
      prof->pr_size = bufsiz;
      prof->pr_off = (caddr_t) offset;
      prof->pr_scale = scale;

      papi_return(PAPI_sprofil(prof, 1, EventSet, EventCode, threshold, flags));
   }
   papi_return(PAPI_sprofil(NULL, 0, EventSet, EventCode, 0, flags));

}

int PAPI_set_granularity(int granularity)
{
   PAPI_option_t ptr;

   ptr.defgranularity.granularity = granularity;
   papi_return(PAPI_set_opt(PAPI_GRANUL, &ptr));
}

/* This function sets the low level default counting domain
   for all newly manufactured eventsets */

int PAPI_set_domain(int domain)
{
   PAPI_option_t ptr;

   ptr.defdomain.domain = domain;
   papi_return(PAPI_set_opt(PAPI_DEFDOM, &ptr));
}

int PAPI_add_events(int EventSet, int *Events, int number)
{
   int i, retval;

   if ((Events == NULL) || (number < 0))
      papi_return(PAPI_EINVAL);

   for (i = 0; i < number; i++) {
      retval = PAPI_add_event(EventSet, Events[i]);
      if (retval != PAPI_OK)
         return (retval);
   }
   papi_return(PAPI_OK);
}

int PAPI_remove_events(int EventSet, int *Events, int number)
{
   int i, retval;

   if ((Events == NULL) || (number < 0))
      papi_return(PAPI_EINVAL);

   for (i = 0; i < number; i++) {
      retval = PAPI_remove_event(EventSet, Events[i]);
      if (retval != PAPI_OK)
         return (retval);
   }
   return (PAPI_OK);
}

int PAPI_list_events(int EventSet, int *Events, int *number)
{
   EventSetInfo_t *ESI;
   int num;
   int i;

   if ((!Events) || (!number))
      papi_return(PAPI_EINVAL);

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (!ESI)
      papi_return(PAPI_ENOEVST);

#ifdef DEBUG
   /* Not necessary */
   if (ESI->NumberOfEvents == 0)
      papi_return(PAPI_EINVAL);
#endif

   if (*number < ESI->NumberOfEvents)
      num = *number;
   else
      num = ESI->NumberOfEvents;

   for (i = 0; i < num; i++)
      Events[i] = ESI->EventInfoArray[i].event_code;

   *number = ESI->NumberOfEvents;

   papi_return(PAPI_OK);
}

long PAPI_get_dmem_info(int option)
{
   if (option != PAPI_GET_PAGESIZE) {
      return (_papi_hwd_get_dmem_info(option));
   } else
      return ((long) getpagesize());
}

const PAPI_exe_info_t *PAPI_get_executable_info(void)
{
   PAPI_option_t ptr;
   int retval;

   retval = PAPI_get_opt(PAPI_EXEINFO, &ptr);
   if (retval == PAPI_OK)
      return (ptr.exe_info);
   else
      return (NULL);
}

const PAPI_shlib_info_t *PAPI_get_shared_lib_info(void)
{
   PAPI_option_t ptr;
   int retval;

   retval = PAPI_get_opt(PAPI_SHLIBINFO, &ptr);
   if (retval == PAPI_OK)
      return (ptr.shlib_info);
   else
      return (NULL);
}

const PAPI_hw_info_t *PAPI_get_hardware_info(void)
{
   PAPI_option_t ptr;
   int retval;

   retval = PAPI_get_opt(PAPI_HWINFO, &ptr);
   if (retval == PAPI_OK)
      return (ptr.hw_info);
   else
      return (NULL);
}

long_long PAPI_get_real_cyc(void)
{
   return (_papi_hwd_get_real_cycles());
}

long_long PAPI_get_real_usec(void)
{
   return (_papi_hwd_get_real_usec());
}

u_long_long PAPI_get_virt_cyc(void)
{
   ThreadInfo_t *master = _papi_hwi_lookup_in_thread_list();

   if (master)
      return (_papi_hwd_get_virt_cycles(&master->context));
   else if (_papi_hwi_thread_id_fn != NULL) {
      int retval;
      DBG((stderr, "PAPI_get_virt_cyc(): new thread found\n"));
      retval = _papi_hwi_initialize_thread(&master);
      if (retval)
         papi_return(retval);

      _papi_hwi_insert_in_thread_list(master);
      return (_papi_hwd_get_virt_cycles(&master->context));
   }
   return PAPI_ECNFLCT;
}

u_long_long PAPI_get_virt_usec(void)
{
   ThreadInfo_t *master = _papi_hwi_lookup_in_thread_list();

   if (master)
      return (_papi_hwd_get_virt_usec(&master->context));
   else if (_papi_hwi_thread_id_fn != NULL) {
      int retval;
      DBG((stderr, "PAPI_get_virt_usec(): new thread found\n"));
      retval = _papi_hwi_initialize_thread(&master);
      if (retval)
         papi_return(retval);
      _papi_hwi_insert_in_thread_list(master);
      return (_papi_hwd_get_virt_usec(&master->context));
   } else
      return PAPI_ECNFLCT;
}

int PAPI_restore(void)
{
   fprintf(stderr, "PAPI_restore is currently not implemented\n");
   return (PAPI_ESBSTR);
}
int PAPI_save(void)
{
   fprintf(stderr, "PAPI_save is currently not implemented\n");
   return (PAPI_ESBSTR);
}

void PAPI_lock(int lck)
{
   _papi_hwd_lock(lck);
}

void PAPI_unlock(int lck)
{
   _papi_hwd_unlock(lck);
}

int PAPI_is_initialized(void)
{
   return (init_level);
}
