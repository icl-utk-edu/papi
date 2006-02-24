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

#include "papi.h"
#include "papi_internal.h"
#include "papi_protos.h"
#include "papi_vector.h"
#include "papi_vector_redefine.h"
#include "papi_memory.h"

/*******************************/
/* BEGIN EXTERNAL DECLARATIONS */
/*******************************/

#ifdef DEBUG
#define papi_return(a) return((_papi_hwi_debug_handler ? _papi_hwi_debug_handler(a) : a))
#else
#define papi_return(a) return(a)
#endif

#ifdef ANY_THREAD_GETS_SIGNAL
extern int (*_papi_hwi_thread_kill_fn) (int, int);
#endif

extern unsigned long int (*_papi_hwi_thread_id_fn) (void);
extern int _papi_hwi_error_level;
extern hwi_describe_t _papi_hwi_err[];
extern PAPI_debug_handler_t _papi_hwi_debug_handler;
extern papi_mdi_t _papi_hwi_system_info;
extern void _papi_hwi_dummy_handler(int,void*,long_long,void*);

/* papi_data.c */

extern hwi_presets_t _papi_hwi_presets;
extern const hwi_describe_t _papi_hwi_derived[];

extern int init_retval;
extern int init_level;

/* Defined by the substrate */
extern hwi_preset_data_t _papi_hwi_preset_data[];

/*****************************/
/* END EXTERNAL DECLARATIONS */
/*****************************/
int _papi_hwi_get_event_info(int EventCode, PAPI_event_info_t * info);
int _papi_hwi_encode_preset_event(PAPI_event_info_t * info, int *EventCode);


/********************/
/*  BEGIN LOCALS    */
/********************/

/********************/
/*    END LOCALS    */
/********************/

int PAPI_thread_init(unsigned long int (*id_fn) (void))
{
  /* Thread support not implemented on Alpha/OSF because the OSF pfm
   * counter device driver does not support per-thread counters.
   * When this is updated, we can remove this if statement
   */
   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_EINVAL);

   if ((init_level&PAPI_THREAD_LEVEL_INITED))
      papi_return(PAPI_OK);

   if (! _papi_hwi_system_info.supports_multiple_threads)
      papi_return(PAPI_ESBSTR);

   init_level |= PAPI_THREAD_LEVEL_INITED;
   papi_return(_papi_hwi_set_thread_id_fn(id_fn));
}

unsigned long PAPI_thread_id(void)
{
   if (_papi_hwi_thread_id_fn != NULL)
     return ((*_papi_hwi_thread_id_fn) ());
   else
     papi_return (PAPI_EMISC);
}

/* Thread Functions */

/* 
 * Notify PAPI that a thread has 'appeared'
 * We lookup the thread, if it does not exist we create it
 */

int PAPI_register_thread(void)
{
   ThreadInfo_t *thread;

   papi_return(_papi_hwi_lookup_or_create_thread(&thread));
}

/* 
 * Notify PAPI that a thread has 'disappeared'
 * We lookup the thread, if it does not exist we return an error
 */

int PAPI_unregister_thread(void)
{
   ThreadInfo_t *thread = _papi_hwi_lookup_thread();
   
   if (thread)
     papi_return(_papi_hwi_shutdown_thread(thread));

   papi_return(PAPI_EMISC);
}

/*
 * Return a pointer to the stored thread information.
 */
int PAPI_get_thr_specific(int tag, void **ptr)
{
   ThreadInfo_t *thread;
   int retval = PAPI_OK;

   if ((tag < 0) || (tag > PAPI_NUM_TLS))
      papi_return(PAPI_EINVAL);

   retval = _papi_hwi_lookup_or_create_thread(&thread);
   if (retval == PAPI_OK)
     *ptr = thread->thread_storage[tag];
   else
     return(retval);

   return(retval);
}

/*
 * Store a pointer to memory provided by the thread
 */
int PAPI_set_thr_specific(int tag, void *ptr)
{
   ThreadInfo_t *thread;
   int retval = PAPI_OK;

   if ((tag < 0) || (tag > PAPI_NUM_TLS))
      papi_return(PAPI_EINVAL);

   retval = _papi_hwi_lookup_or_create_thread(&thread);
   if (retval == PAPI_OK)
     thread->thread_storage[tag] = ptr;
   else
     return(retval);

   return(PAPI_OK);
}


int PAPI_library_init(int version)
{
   int tmp = 0,i;
   /* This is a poor attempt at a lock. 
      For 3.1 this should be replaced with a 
      true UNIX semaphore. We cannot use PAPI
      locks here because they are not initialized yet */
   static int _in_papi_library_init_cnt = 0;
#ifdef DEBUG
   char *var;
#endif

   if (version != PAPI_VER_CURRENT)
      papi_return(PAPI_EINVAL);

   ++_in_papi_library_init_cnt;
   while (_in_papi_library_init_cnt > 1)
     {
       PAPIERROR("Multiple callers of PAPI_library_init");
      sleep(1);
     }

#ifndef _WIN32
   /* This checks to see if we have forked or called init more than once.
      If we have forked, then we continue to init. If we have not forked, 
      we check to see the status of initialization. */

   APIDBG("Initializing library: current PID %d, old PID %d\n", getpid(), _papi_hwi_system_info.pid);
   if (_papi_hwi_system_info.pid == getpid())
#endif
     {
       /* If the magic environment variable PAPI_ALLOW_STOLEN is set,
	  we call shutdown if PAPI has been initialized. This allows
	  tools that use LD_PRELOAD to run on applications that use PAPI.
	  In this circumstance, PAPI_ALLOW_STOLEN will be set to 'stolen'
	  so the tool can check for this case. */

       if (getenv("PAPI_ALLOW_STOLEN"))
	 {
	   char buf[PAPI_HUGE_STR_LEN];
	   if (init_level != PAPI_NOT_INITED)
	     PAPI_shutdown();
	   sprintf(buf,"%s=%s","PAPI_ALLOW_STOLEN","stolen");
	   putenv(buf);
	 }

       /* If the library has been successfully initialized *OR*
	  the library attempted initialization but failed. */
       
       else if ((init_level != PAPI_NOT_INITED) || (init_retval != DEADBEEF))
	 {
	   _in_papi_library_init_cnt--;
	   if (init_retval < PAPI_OK)
	     papi_return(init_retval); 
	   else
	     return(init_retval); 
	 }

       APIDBG("system_info was initialized, but init did not succeed\n");
     }

#ifdef DEBUG
   var = (char *)getenv("PAPI_DEBUG");
   _papi_hwi_debug = 0;

   if (var != NULL) 
     {
       if (strlen(var) != 0)
	 {
	   if (strstr(var,"SUBSTRATE"))
	     _papi_hwi_debug |= DEBUG_SUBSTRATE;
	   if (strstr(var,"API"))
	     _papi_hwi_debug |= DEBUG_API;
	   if (strstr(var,"INTERNAL"))
	     _papi_hwi_debug |= DEBUG_INTERNAL;
	   if (strstr(var,"THREADS"))
	     _papi_hwi_debug |= DEBUG_THREADS;
	   if (strstr(var,"MULTIPLEX"))
	     _papi_hwi_debug |= DEBUG_MULTIPLEX;
	   if (strstr(var,"OVERFLOW"))
	     _papi_hwi_debug |= DEBUG_OVERFLOW;
	   if (strstr(var,"PROFILE"))
	     _papi_hwi_debug |= DEBUG_PROFILE;
	   if (strstr(var,"ALL"))
	     _papi_hwi_debug |= DEBUG_ALL;
	 }

       if (_papi_hwi_debug == 0)
	 _papi_hwi_debug |= DEBUG_API;
     }
#endif

   if (_papi_hwi_init_global_internal() != PAPI_OK) {
     _in_papi_library_init_cnt--;
      papi_return(PAPI_EINVAL);
   }

   /* Initialize substrate globals */

   tmp = _papi_hwi_init_global();
   if (tmp) {
      init_retval = tmp;
      _papi_hwi_shutdown_global_internal();
      _in_papi_library_init_cnt--;
      papi_return(init_retval);
   }

   /* Initialize internal globals */


   /* Initialize thread globals, including the main threads
      substrate */

   tmp = _papi_hwi_init_global_threads();
   if (tmp) {
      init_retval = tmp;
      _papi_hwi_shutdown_global_internal();
      for(i=0;i<papi_num_substrates;i++){
        _papi_vector_table[i]._vec_papi_hwd_shutdown_global();
      }
      _in_papi_library_init_cnt--;
      papi_return(init_retval);
   }

   init_level = PAPI_LOW_LEVEL_INITED;
   _in_papi_library_init_cnt--;
   return (init_retval = PAPI_VER_CURRENT);
}

int PAPI_query_event(int EventCode)
{
   if (EventCode & PAPI_PRESET_MASK) {
      EventCode &= PAPI_PRESET_AND_MASK;
      if (EventCode >= PAPI_MAX_PRESET_EVENTS)
         papi_return(PAPI_ENOTPRESET);

      if (_papi_hwi_presets.count[EventCode])
         papi_return(PAPI_OK);
      else
         return(PAPI_ENOEVNT);
   }

   if (EventCode & PAPI_NATIVE_MASK) {
      papi_return(_papi_hwi_query_native_event(EventCode));
   }

   papi_return(PAPI_ENOTPRESET);
}

int PAPI_get_sbstr_info(int idx, PAPI_substrate_info_t *info)
{
  if ( idx < 0 || idx >= papi_num_substrates )
    return(PAPI_EINVAL);

  if ( _papi_hwi_substrate_info[idx].num_cntrs == -1 ){
     info->name[0] = '\0';
     info->initialized = 0;
     info->num_cntrs = 0;
     info->version = 0.0;
  }
  else {
     strcpy(info->name,_papi_hwi_substrate_info[idx].substrate);
     info->initialized = 1;
     info->num_cntrs = _papi_hwi_substrate_info[idx].num_cntrs;
     info->version = _papi_hwi_substrate_info[idx].version; 
  }
  return(PAPI_OK);
}

/* PAPI_get_event_info:
   tests input EventCode and returns a filled in PAPI_event_info_t 
   structure containing descriptive strings and values for the 
   specified event. Handles both preset and native events by 
   calling either _papi_hwi_get_event_info or 
   _papi_hwi_get_native_event_info.
*/
int PAPI_get_event_info(int EventCode, PAPI_event_info_t * info)
{
   int i = EventCode & PAPI_PRESET_AND_MASK;

   if (info == NULL)
      papi_return(PAPI_EINVAL);

   if (EventCode & PAPI_PRESET_MASK) {
      if (i >= PAPI_MAX_PRESET_EVENTS)
         papi_return(PAPI_ENOTPRESET);
      papi_return(_papi_hwi_get_event_info(EventCode, info));
   }
 
   if (EventCode & PAPI_NATIVE_MASK) {
      papi_return(_papi_hwi_get_native_event_info(EventCode, info));
   }

   papi_return(PAPI_ENOTPRESET);
}


/* PAPI_set_event_info:
   info -- input only
      info->event_code can have the following bits set
         - PAPI_PRESET_MASK for preset events
         - PAPI_NATIVE_MASK for native events
      the index portion of the event_code is ignored; 
      matching occurs based on names only
   EventCode -- output only: contains the new event code
   replace -- true if existing events can be replaced

   This function is symmetric with PAPI_get_event_info, except that it
   doesn't yet support native events. It allows a user or tool to add
   or override definitions of PAPI events on the fly. Bails if events 
   to be modified have been added to existing EventSets.
*/
int PAPI_set_event_info(PAPI_event_info_t * info, int *EventCode, int replace)
{
   int i, type;
   int code;
   EventSetInfo_t *ESI;

   if (info == NULL)
      papi_return(PAPI_EINVAL);

   if (PAPI_event_name_to_code(info->symbol, &code) == PAPI_OK) { /* event already exists */
      if (!replace) papi_return(PAPI_EPERM); /* we don't have permission to modify it */

      /* Are any EventSets already in existence? */
      for (i=0; i<_papi_hwi_system_info.global_eventset_map.totalSlots; i++) {
         ESI = _papi_hwi_lookup_EventSet(i);
         if (ESI) { /* found an existing EventSet */
            if(_papi_hwi_lookup_EventCodeIndex(ESI, (unsigned int)code) != PAPI_EINVAL)
               papi_return(PAPI_EISRUN); /* can't handle Events in use */
         }
      }
   }

   if (info->event_code & PAPI_PRESET_MASK) {
      /* do some sanity checks */
      if (info->derived) {
         type = _papi_hwi_derived_type(info->derived);

         if(type == -1)
            papi_return(PAPI_EINVAL); /* not a valid Derived type string */
         if(type != NOT_DERIVED && info->count < 2)
            papi_return(PAPI_EINVAL); /* can't be derived with only one term */
         if(type == NOT_DERIVED && info->count > 1)
            papi_return(PAPI_EINVAL); /* must be derived if more than one term */
      }
      papi_return(_papi_hwi_set_event_info(info, EventCode));
   }
 
/* Someday we might also handle native events...
   if (info->event_code & PAPI_NATIVE_MASK) {
      papi_return(_papi_hwi_set_native_event(info, EventCode));
}*/

   papi_return(PAPI_ENOEVNT);
}

/* PAPI_encode_events:
   event_file -- pathname for a csv file containing event definitions
   replace -- true if existing events can be replaced

   This function reads event definitions from a comma separated values file.
   It allows a user or tool to add or override sets of PAPI events 
   on the fly. Bails if events to be modified have been added to existing EventSets.
*/

/* copy the contents of a potentially quoted string */
static char *quotcpy(char *d, char *s) {
   if (s && *s == '\"') {
      s++;
      s[strlen(s)-1]=0;
   }
   strcpy(d, s);
   return (d);
}

int PAPI_encode_events(char * event_file, int replace)
{
   int i, j, line, field;
   int retval;
   PAPI_event_info_t info;
	FILE *file = NULL;
   char buf[PAPI_HUGE_STR_LEN];
   char *b, *token[20];
   int quote;

   if (!event_file) papi_return(PAPI_EINVAL);
   file = fopen(event_file, "r");
	if (file == NULL) papi_return(PAPI_EINVAL);

   line = 0;
   while (fgets(buf, sizeof(buf), file)) {
      field = 0;
      quote = 0;
      b = buf;
      token[field] = b;
      do { /* split line into tokens (arbitrarily up to 20) */
         switch (*b) {
            case '\r':
            case '\n':
               /* strip line endings if not inside quotes */
               if (!quote) *b = 0;
               break;
            case '\"':
               /* toggle quoted state */
               quote = !quote;
               break;
            case ',':
               /* change fields on comma if not inside quotes */
               if (!quote) {
                  *b = 0;
                  token[++field] = b+1;
                  token[field+1] = NULL;
               }
               break;
            default:
               break;
         }
      } while (*(++b) && field < 20);

      if (line == 0) {
         /* line 0 contains field identifiers.
            This info is currently ignored,
            but could be used for location matching.
            Line 1 is assumed blank.
         */
      }
      else if (line > 1) {
         /* tokens are currently assumed to appear mostly as they 
            occur in the info structure. The exception is that the
            native event names have been moved to the end, and the
            postfix string has been moved after the derived string. */
         quotcpy(info.symbol, token[0]);
         quotcpy(info.derived, token[1]);
         quotcpy(info.postfix, token[2]);
         quotcpy(info.short_descr, token[3]);
         quotcpy(info.long_descr, token[4]);
         quotcpy(info.note, token[5]);
         info.event_code = PAPI_PRESET_MASK;
         info.count = 0;
         for (j = 0; j < PAPI_MAX_INFO_TERMS; j++) {
            if (!token[j+6]) break;
            quotcpy(info.name[j], token[j+6]);
            info.count++;
         }
         retval = PAPI_set_event_info(&info, &i, replace);
      }
      line++;
   }
	fclose(file);
   papi_return(PAPI_OK);
}

int PAPI_event_code_to_name(int EventCode, char *out)
{
   if (out == NULL)
      papi_return(PAPI_EINVAL);

   if (EventCode & PAPI_PRESET_MASK) {
      EventCode &= PAPI_PRESET_AND_MASK;
      if ((EventCode >= PAPI_MAX_PRESET_EVENTS)
          || (_papi_hwi_presets.info[EventCode].symbol == NULL))
         papi_return(PAPI_ENOTPRESET);

      strncpy(out, _papi_hwi_presets.info[EventCode].symbol, PAPI_MAX_STR_LEN);
      papi_return(PAPI_OK);
   }

   if (EventCode & PAPI_NATIVE_MASK) {
      return(_papi_hwi_native_code_to_name(EventCode, out, PAPI_MAX_STR_LEN));
   }

   papi_return(PAPI_ENOEVNT);
}

int PAPI_event_name_to_code(char *in, int *out)
{
   int i;

   if ((in == NULL) || (out == NULL))
      papi_return(PAPI_EINVAL);

   /* With user definable events, we can no longer assume
      presets begin with "PAPI"...
   if (strncmp(in, "PAPI", 4) == 0) {
   */
   for (i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
      if ((_papi_hwi_presets.info[i].symbol)
            && (strcasecmp(_papi_hwi_presets.info[i].symbol, in) == 0)) {
         *out = (i | PAPI_PRESET_MASK);
         papi_return(PAPI_OK);
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
   int idx = PAPI_SUBSTRATE_INDEX(*EventCode);

   if ( idx < 0 || idx > papi_num_substrates ) 
       return (PAPI_ENOEVNT);

   if (i & PAPI_PRESET_MASK) {
      i &= PAPI_PRESET_AND_MASK;
      while (++i < PAPI_MAX_PRESET_EVENTS) {
         if ((!modifier) || (_papi_hwi_presets.count[i])) {
            *EventCode = i | PAPI_PRESET_MASK;
            if (_papi_hwi_presets.info[i].symbol == NULL)
                return (PAPI_ENOEVNT); /* NULL pointer terminates list */
             else
                return (PAPI_OK);
         }
      }
   } else if (i & PAPI_NATIVE_MASK) {
      return (_papi_hwd_ntv_enum_events((unsigned int *) EventCode, modifier, idx));
   }
   return (PAPI_ENOEVNT);
}

/* Deprecated, use PAPI_create_sbstr_eventset */
int PAPI_create_eventset(int *EventSet)
{
  return (PAPI_create_sbstr_eventset(EventSet, 0));
}

int PAPI_create_sbstr_eventset(int *EventSet, int substrate)
{
   ThreadInfo_t *master;
   int retval;

   if ( substrate < 0 || substrate >= papi_num_substrates )
     return (PAPI_EINVAL);

   retval = _papi_hwi_lookup_or_create_thread(&master);
   if (retval)
     return(retval);

   return (_papi_hwi_create_eventset(EventSet, master, substrate));
}

int PAPI_add_event(int EventSet, int EventCode)
{
   EventSetInfo_t *ESI;

   /* Is the EventSet already in existence? */

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   /* Should we do it based on events or substrate here? */
   if ( ESI->SubstrateIndex < 0 )
      papi_return(PAPI_EMISC);

   /* Check argument for validity */

   if ((((EventCode & PAPI_PRESET_MASK) == 0) && 
       ((EventCode & PAPI_NATIVE_MASK) == 0)) || 
       PAPI_SUBSTRATE_INDEX(EventCode) != ESI->SubstrateIndex)
     papi_return(PAPI_EINVAL);

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

   /* Check argument for validity */

   if (((EventCode & PAPI_PRESET_MASK) == 0) && 
       (EventCode & PAPI_NATIVE_MASK) == 0)
     papi_return(PAPI_EINVAL);

   /* Of course, it must be stopped in order to modify it. */

   if (!(ESI->state & PAPI_STOPPED))
      papi_return(PAPI_EISRUN);

   /* if the state is PAPI_OVERFLOWING, you must first call
      PAPI_overflow with threshold=0 to remove the overflow flag */

   /* Turn off the even that is overflowing */
   if (ESI->state & PAPI_OVERFLOWING) {
      for(i=0; i<ESI->overflow.event_counter; i++ ) {
        if ( ESI->overflow.EventCode[i] == EventCode ){
           PAPI_overflow( EventSet, EventCode, 0, 0, ESI->overflow.handler);
           break;
        }
      } 
   }
   
   /* force the user to call PAPI_profil to clear the PAPI_PROFILING flag */
   if (ESI->state & PAPI_PROFILING)  {
     for (i=0; i < ESI->profile.event_counter; i++ ){
       if ( ESI->profile.EventCode[i] == EventCode ){
         PAPI_sprofil(NULL,0,EventSet,EventCode, 0, 0);
         break;
       }
     }
   }

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

   return(PAPI_OK);
}

/* simply checks for valid EventSet, calls substrate start() call */

int PAPI_start(int EventSet)
{
   int retval;
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;

   APIDBG("PAPI_start\n");

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
     papi_return(PAPI_ENOEVST);

   if ( ESI->SubstrateIndex < 0 ) 
     papi_return(PAPI_EMISC);

   thread = ESI->master;

   /* only one event set can be running at any time, so if another event
      set is running, the user must stop that event set explicitly */

   if (thread->running_eventset[ESI->SubstrateIndex])
      papi_return(PAPI_EISRUN);

   /* Check that there are added events */

   if (ESI->NumberOfEvents < 1)
      papi_return(PAPI_EINVAL);

   /* If multiplexing is enabled for this eventset,
      call John May's code. */

   if (ESI->state & PAPI_MULTIPLEXING) 
     {
      retval = MPX_start(ESI->multiplex);
      if (retval != PAPI_OK)
         papi_return(retval);

      /* Update the state of this EventSet */

      ESI->state ^= PAPI_STOPPED;
      ESI->state |= PAPI_RUNNING;

      return (PAPI_OK);
   }

   /* Short circuit this stuff if there's nothing running */

   /* If overflowing is enabled, turn it on */

   if (ESI->state & PAPI_OVERFLOWING) 
     {
       if (!(ESI->overflow.flags&PAPI_OVERFLOW_HARDWARE))
	 {
           APIDBG("Overflow using: %s with a interval of %d ms\n", (ESI->overflow.flags&PAPI_OVERFLOW_FORCE_SW)?"[Forced Software]":"Software", ESI->overflow.timer_ms);
	   retval = _papi_hwi_start_signal(PAPI_SIGNAL, NEED_CONTEXT);
	   if (retval != PAPI_OK)
	     papi_return(retval);
	   retval = _papi_hwi_start_timer(ESI->overflow.timer_ms);
	   if (retval != PAPI_OK)
	     {
               APIDBG("Error starting _papi_hwi_start_timer: %d\n", retval);
	       _papi_hwi_stop_signal(PAPI_SIGNAL);
	       papi_return(retval);
	     }
	 }
        else {
           APIDBG("Overflow using: [Hardware]\n");
        }
     }

   /* Merge the control bits from the new EventSet into the active counter config. */

   retval = _papi_hwd_start(thread->context[ESI->SubstrateIndex], ESI->machdep, ESI->SubstrateIndex);
   if (retval != PAPI_OK)
      papi_return(retval);

   /* Update the state of this EventSet */

   ESI->state ^= PAPI_STOPPED;
   ESI->state |= PAPI_RUNNING;

   /* Update the running event set  for this thread */
   thread->running_eventset[ESI->SubstrateIndex] = ESI;

   APIDBG("PAPI_start returns %d\n", retval);
   return (retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */

int PAPI_stop(int EventSet, long_long * values)
{
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;
   int retval;

   APIDBG("PAPI_stop\n");
   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   if ( ESI->SubstrateIndex < 0 ) 
      papi_return(PAPI_EMISC);

   thread = ESI->master;

   if (!(ESI->state & PAPI_RUNNING))
      papi_return(PAPI_ENOTRUN);

   if (ESI->state & PAPI_PROFILING) {
      if (_papi_hwi_substrate_info[ESI->SubstrateIndex].supports_hw_profile && !(ESI->profile.flags&PAPI_PROFIL_FORCE_SW)) {
         retval = _papi_hwd_stop_profiling(thread, ESI, ESI->SubstrateIndex);
         if (retval < PAPI_OK)
            papi_return(retval);
      }
   }

   /* If overflowing is enabled, turn it off */

   if (ESI->state & PAPI_OVERFLOWING) 
     {
       ESI->overflow.count = 0;
       if (!(ESI->overflow.flags&PAPI_OVERFLOW_HARDWARE))
	 {
	   retval = _papi_hwi_stop_timer();
	   if (retval != PAPI_OK)
	     papi_return(retval);
	   _papi_hwi_stop_signal(PAPI_SIGNAL);
	 }
     }

   /* If multiplexing is enabled for this eventset, turn if off */

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

   retval = _papi_hwi_read(thread->context[ESI->SubstrateIndex], ESI, ESI->sw_stop);
   if (retval != PAPI_OK)
      papi_return(retval);

   /* Remove the control bits from the active counter config. */

   retval = _papi_hwd_stop(thread->context[ESI->SubstrateIndex], ESI->machdep, ESI->SubstrateIndex);
   if (retval != PAPI_OK)
      papi_return(retval);
   if (values)
      memcpy(values, ESI->sw_stop, ESI->NumberOfEvents * sizeof(long_long));

   /* Update the state of this EventSet */

   ESI->state ^= PAPI_RUNNING;
   ESI->state |= PAPI_STOPPED;

   /* Update the running event set  for this thread */
   thread->running_eventset[ESI->SubstrateIndex] = NULL ;

#if defined(DEBUG)
     if (_papi_hwi_debug & DEBUG_API)
       {
	 int i;
	 for (i = 0; i < ESI->NumberOfEvents; i++)
	   APIDBG("PAPI_stop ESI->sw_stop[%d]:\t%llu\n", i, ESI->sw_stop[i]);
       }
#endif

   APIDBG("PAPI_stop returns %d\n", retval);

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

   if ( ESI->SubstrateIndex < 0 )
      papi_return(PAPI_EMISC);

   thread = ESI->master;

   if (ESI->state & PAPI_RUNNING) {
      if (ESI->state & PAPI_MULTIPLEXING)
         retval = MPX_reset(ESI->multiplex);
      else {
         /* If we're not the only one running, then just
            read the current values into the ESI->start
            array. This holds the starting value for counters
            that are shared. */

         retval = _papi_hwd_reset(thread->context[ESI->SubstrateIndex], ESI->machdep, ESI->SubstrateIndex);

         if ((ESI->state & PAPI_OVERFLOWING) &&
             (ESI->overflow.flags&PAPI_OVERFLOW_HARDWARE))
            ESI->overflow.count = 0;

         if ((ESI->state & PAPI_PROFILING) && (_papi_hwi_substrate_info[ESI->SubstrateIndex].supports_hw_profile) &&
             !(ESI->profile.flags&PAPI_PROFIL_FORCE_SW))
            ESI->profile.overflowcount = 0;
      }
   } else {
      memset(ESI->sw_stop, 0x00, ESI->NumberOfEvents * sizeof(long_long));
   }

   APIDBG("PAPI_reset returns %d\n", retval);
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
         retval = _papi_hwi_read(thread->context[ESI->SubstrateIndex], ESI, values);
      if (retval != PAPI_OK)
         papi_return(retval);
   } else {
      memcpy(values, ESI->sw_stop, ESI->NumberOfEvents * sizeof(long_long));
   }

#if defined(DEBUG)
   if (ISLEVEL(DEBUG_API))
   {
      int i;
      for (i = 0; i < ESI->NumberOfEvents; i++)
         APIDBG("PAPI_read values[%d]:\t%lld\n", i, values[i]);
   }
#endif

   APIDBG("PAPI_read returns %d\n", retval);
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
         retval = _papi_hwi_read(thread->context[ESI->SubstrateIndex], ESI, ESI->sw_stop);
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

   if ( ESI->SubstrateIndex < 0 )
      papi_return(PAPI_EMISC);

   thread = ESI->master;

   if (values == NULL)
      papi_return(PAPI_EINVAL);

   if (ESI->state & PAPI_RUNNING) {
      retval = _papi_hwd_write(thread->context[ESI->SubstrateIndex], ESI->machdep, values, ESI->SubstrateIndex);
      if (retval != PAPI_OK)
         return (retval);
   }

   memcpy(ESI->hw_start, values, _papi_hwi_substrate_info[ESI->SubstrateIndex].num_cntrs * sizeof(long_long));

   return (retval);
}

/*  The function PAPI_cleanup removes a stopped EventSet from existence. */

int PAPI_cleanup_eventset(int EventSet)
{
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;
   int i, total, retval;

   /* Is the EventSet already in existence? */

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   /* Of course, it must be stopped in order to modify it. */

   if (ESI->state & PAPI_RUNNING)
      papi_return(PAPI_EISRUN);

   /* clear overflow flag and turn off hardware overflow handler */
   if (ESI->state & PAPI_OVERFLOWING ) {
      total=ESI->overflow.event_counter;
      for (i = 0; i < total; i++) {
         retval = PAPI_overflow(EventSet,  
                 ESI->overflow.EventCode[0], 0, 0, NULL);
         if (retval != PAPI_OK)
            papi_return(retval);
      }
   }
   /* clear profile flag and turn off hardware profile handler */
   if ( (ESI->state & PAPI_PROFILING) && 
          _papi_hwi_substrate_info[ESI->SubstrateIndex].supports_hw_profile && !(ESI->profile.flags&PAPI_PROFIL_FORCE_SW)) {
      total=ESI->profile.event_counter;
      for (i = 0; i < total; i++) {
         retval = PAPI_sprofil(NULL,0,EventSet,ESI->profile.EventCode[0],0,
                               PAPI_PROFIL_POSIX);
         if (retval != PAPI_OK)
            papi_return(retval);
      }
   }

   if (ESI->state & PAPI_MULTIPLEXING) {
      retval = MPX_cleanup(&ESI->multiplex);
      if (retval != PAPI_OK)
         papi_return(retval);
   }

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

   return(PAPI_OK);
}

int PAPI_set_debug(int level)
{
  PAPI_option_t option;

  memset(&option,0x0,sizeof(option));
  option.debug.level = level;
  option.debug.handler = _papi_hwi_debug_handler;
  papi_return(PAPI_set_opt(PAPI_DEBUG,&option));
}

int PAPI_set_multiplex(int EventSet)
{
   PAPI_option_t mpx;

   mpx.multiplex.eventset = EventSet;
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
   case PAPI_MULTIPLEX:
      {
         EventSetInfo_t *ESI;

         if (ptr->multiplex.us < 1)
            papi_return(PAPI_EINVAL);
         ESI = _papi_hwi_lookup_EventSet(ptr->multiplex.eventset);
         if (ESI == NULL)
            papi_return(PAPI_ENOEVST);
         if (!(ESI->state & PAPI_STOPPED))
            papi_return(PAPI_EISRUN);
         if (ptr->multiplex.max_degree <= _papi_hwi_substrate_info[ESI->SubstrateIndex].num_cntrs) {
            return(PAPI_OK);
         }
         if (ESI->state & PAPI_MULTIPLEXING)
            papi_return(PAPI_EINVAL);

         papi_return(_papi_hwi_convert_eventset_to_multiplex(ESI));
      }
   case PAPI_DEBUG: 
      {
	int level = ptr->debug.level;
	switch (level) {
	case PAPI_QUIET:
	case PAPI_VERB_ESTOP:
	case PAPI_VERB_ECONT:
	  _papi_hwi_error_level = level;
	  break;
	default:
	  papi_return(PAPI_EINVAL); }
	_papi_hwi_debug_handler = ptr->debug.handler;
	return(PAPI_OK);
      }
   case PAPI_DEFDOM:
      {
         int dom = ptr->defdomain.domain;
         if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
            papi_return(PAPI_EINVAL);

         /* Change the global structure. The _papi_hwd_init_control_state function 
	    in the substrates gets information from the global structure instead of
            per-thread information. */

         /* XXX Need to add way to set default domain */
         _papi_hwi_substrate_info[0].default_domain = dom;

         return (PAPI_OK);
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
         retval = _papi_hwd_ctl(thread->context[0], PAPI_DOMAIN, &internal);
         if (retval < PAPI_OK)
            papi_return(retval);

         /* Change the domain of the eventset in the library */

         internal.domain.ESI->domain.domain = dom;

         return (retval);
      }
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
   case PAPI_DATA_ADDRESS:
   case PAPI_INSTR_ADDRESS:
      {
         internal.address_range.start = ptr->addr.start;
         internal.address_range.end = ptr->addr.end;
         retval = _papi_hwd_ctl(NULL, option, &internal);
      }

   default:
      papi_return(PAPI_EINVAL);
   }
}

/* This is the deprecated PAPI 3 num hwctrs interface.
   It is preserved for backward compatibility. It calls
   PAPI_get_sbstr_opt() with a substrate index of 0
*/
int PAPI_num_hwctrs(void)
{
   return (PAPI_get_sbstr_opt(PAPI_MAX_HWCTRS, NULL, 0));
}

int PAPI_num_sbstr_hwctrs(int idx)
{
   return (PAPI_get_sbstr_opt(PAPI_MAX_HWCTRS, NULL, idx));
}

int PAPI_get_multiplex(int EventSet)
{
   PAPI_option_t popt;
   int retval;

   popt.multiplex.eventset = EventSet;
   retval = PAPI_get_sbstr_opt(PAPI_MULTIPLEX, &popt, 0);
   if (retval < 0)
      retval = 0;
   return retval;
}

/* This is the deprecated PAPI 3 get option interface.
   It is preserved for backward compatibility. It calls
   PAPI_get_sbstr_opt() with a substrate index of 0
*/
int PAPI_get_opt(int option, PAPI_option_t * ptr)
{
   return PAPI_get_sbstr_opt(option, ptr, 0);
}

/* This entry point is new for PAPI 4. It implements a per-substrate
   option function. It supercedes the PAPI_get_opt call
*/
int PAPI_get_sbstr_opt(int option, PAPI_option_t *ptr, int sidx)
{
   switch (option) {
   case PAPI_MAX_CPUS:
      return (_papi_hwi_system_info.hw_info.ncpu);
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
     memcpy(&ptr->preload,&_papi_hwi_system_info.preload_info,sizeof(PAPI_preload_info_t));
      break;
   case PAPI_DEBUG:
      ptr->debug.level = _papi_hwi_error_level;
      ptr->debug.handler = _papi_hwi_debug_handler;
      break;
   case PAPI_CLOCKRATE:
      return ((int) _papi_hwi_system_info.hw_info.mhz);
   case PAPI_GRANUL:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      return (_papi_hwi_get_granularity(&ptr->granularity));
   case PAPI_SHLIBINFO:
      {
         int retval;

         if (ptr == NULL)
            papi_return(PAPI_EINVAL);
         retval = _papi_hwd_update_shlib_info();
         ptr->shlib_info = &_papi_hwi_system_info.shlib_info;
         papi_return(retval);
      }
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
   case PAPI_LIB_VERSION:
      return (PAPI_VERSION);
   case PAPI_MAX_HWCTRS:
      return (_papi_hwi_substrate_info[sidx].num_cntrs);
   case PAPI_DEFDOM:
      return (_papi_hwi_substrate_info[sidx].default_domain);
   case PAPI_DEFGRN:
      return (_papi_hwi_substrate_info[sidx].default_granularity);
   case PAPI_SUBSTRATE_SUPPORT:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      ptr->sub_info.supports_program = _papi_hwi_substrate_info[sidx].supports_program;
      ptr->sub_info.supports_write = _papi_hwi_substrate_info[sidx].supports_write;
      ptr->sub_info.supports_hw_overflow = _papi_hwi_substrate_info[sidx].supports_hw_overflow;
      ptr->sub_info.supports_hw_profile = _papi_hwi_substrate_info[sidx].supports_hw_profile;
      ptr->sub_info.supports_multiple_threads = _papi_hwi_system_info.supports_multiple_threads;
      ptr->sub_info.supports_64bit_counters = _papi_hwi_substrate_info[sidx].supports_64bit_counters;
      ptr->sub_info.supports_inheritance = _papi_hwi_substrate_info[sidx].supports_inheritance;
      ptr->sub_info.supports_attach = _papi_hwi_substrate_info[sidx].supports_attach;
      ptr->sub_info.supports_real_usec = _papi_hwi_system_info.supports_real_usec;
      ptr->sub_info.supports_virt_usec = _papi_hwi_system_info.supports_virt_usec;
      ptr->sub_info.supports_virt_cyc = _papi_hwi_system_info.supports_virt_cyc;
      return(PAPI_OK);
   default:
      papi_return (PAPI_EINVAL);
   }
   return(PAPI_OK);
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
   int i, j = 0;
   ThreadInfo_t *master;

   APIDBG("Enter\n");
   if (init_retval == DEADBEEF) {
      PAPIERROR(PAPI_SHUTDOWN_str);
      return;
   }

   MPX_shutdown();

   master = _papi_hwi_lookup_thread();

   /* Count number of running EventSets AND */
   /* Stop any running EventSets in this thread */

#ifdef DEBUG
 again:
#endif
   for (i = 0; i < map->totalSlots; i++) {
     ESI = map->dataSlotArray[i];
     if (ESI) {
       if (ESI->master == master) {
	 if (ESI->state & PAPI_RUNNING) 
	   PAPI_stop(i, NULL);
	 PAPI_cleanup_eventset(i);
       } else if (ESI->state & PAPI_RUNNING) 
	 j++;
     }
   }

   /* No locking required, we're just waiting for the others
      to call shutdown or stop their eventsets. */

#ifdef DEBUG
   if (j != 0) {
      PAPIERROR(PAPI_SHUTDOWN_SYNC_str);
      sleep(1);
      j = 0;
      goto again;
   }
#endif

   /* Shutdown the entire substrate */

   _papi_hwi_shutdown_highlevel();
   _papi_hwi_shutdown_global_internal();
   _papi_hwi_shutdown_global_threads();

   for(i=0;i<papi_num_substrates;i++){
     _papi_vector_table[i]._vec_papi_hwd_shutdown_global();
   }
   papi_free(_papi_vector_table);
   papi_num_substrates = 0;


   /* Now it is safe to call re-init */

   init_retval = DEADBEEF;
   init_level = PAPI_NOT_INITED;
   papi_mem_cleanup_all();
}

char *PAPI_strerror(int errorCode)
{
   if ((errorCode > 0) || (-errorCode > PAPI_NUM_ERRORS))
      return (NULL);

   return ((char *) _papi_hwi_err[-errorCode].name);
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

   return(PAPI_OK);
}

/* This function sets up an EventSet such that when it is PAPI_start()'ed, it
   begins to register overflows. This EventSet may only have multiple events
   in it and can set multiple events to register overflow, but need to call 
   this function multiple times. To turn off overflow, set the threshold to 0 */

int PAPI_overflow(int EventSet, int EventCode, int threshold, int flags,
                  PAPI_overflow_handler_t handler)
{
   int retval, index, c, i;
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;
   EventSetOverflowInfo_t *o;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   if ( ESI->SubstrateIndex < 0 ) 
      papi_return(PAPI_EMISC);

   thread = ESI->master;

   if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
      papi_return(PAPI_EISRUN);

   if ((index = _papi_hwi_lookup_EventCodeIndex(ESI, EventCode)) < 0)
      papi_return(PAPI_ENOEVNT);

   if (threshold < 0)
      papi_return(PAPI_EINVAL);

   /* We do not support derived events in overflow */
   /* Unless it's DERIVED_CMPD in which no calculations are done */
   if ( !(flags&PAPI_OVERFLOW_FORCE_SW)&& threshold != 0 &&
       (ESI->EventInfoArray[index].derived) && 
       (ESI->EventInfoArray[index].derived != DERIVED_CMPD))
      papi_return(PAPI_EINVAL);

/* the first time to call PAPI_overflow function */
   if (!(ESI->state & PAPI_OVERFLOWING)) {
      if (handler == NULL)
         papi_return(PAPI_EINVAL);
      if (threshold == 0)
         papi_return(PAPI_EINVAL);
   }
   
   o = &ESI->overflow; /* dereference the overflow structure */

   if (threshold > 0 && o->event_counter >= _papi_hwi_substrate_info[ESI->SubstrateIndex].num_cntrs)
      papi_return(PAPI_ECNFLCT);

   if (threshold == 0) {
      for (i = 0; i < o->event_counter; i++) {
         if (o->EventCode[i] == EventCode)
            break;
      }
      /* EventCode not found */
      if (i == o->event_counter)
         papi_return(PAPI_EINVAL);
      /* compact these arrays */
      while (i < o->event_counter - 1) {
         o->deadline[i] = o->deadline[i + 1];
         o->threshold[i] = o->threshold[i + 1];
         o->EventIndex[i] = o->EventIndex[i + 1];
         o->EventCode[i] = o->EventCode[i + 1];
         i++;
      }
      o->deadline[i] = 0;
      o->threshold[i] = 0;
      o->EventIndex[i] = 0;
      o->EventCode[i] = 0;

      o->event_counter--;
   } else {
      if ( o->event_counter > 0 ){
         if ( (flags&PAPI_OVERFLOW_FORCE_SW) && (o->flags&PAPI_OVERFLOW_HARDWARE))
            papi_return(PAPI_ECNFLCT);
         if ( !(flags&PAPI_OVERFLOW_FORCE_SW) && (o->flags&PAPI_OVERFLOW_FORCE_SW))
            papi_return(PAPI_ECNFLCT);
      }

      for (i = 0; i < o->event_counter; i++) {
         if (o->EventCode[i] == EventCode)
            break;
      }

      if (i == o->event_counter){
         c = o->event_counter;
         o->deadline[c] = threshold;
         o->threshold[c] = threshold;
         o->EventIndex[c] = index;
         o->EventCode[c] = EventCode;
         o->flags = flags;
         o->event_counter++;
      }
      else {
         o->deadline[i] = threshold;
         o->threshold[i] = threshold;
         o->EventIndex[i] = index;
         o->flags = flags;
      }
   }
   o->handler = handler;
   o->count = 0;

   /* Set up the option structure for the low level */

   if (_papi_hwi_substrate_info[ESI->SubstrateIndex].supports_hw_overflow && 
       !(o->flags&PAPI_OVERFLOW_FORCE_SW)) {
      if ( threshold != 0 )
         o->flags |= PAPI_OVERFLOW_HARDWARE;
      retval = _papi_hwd_set_overflow(ESI, index, threshold,ESI->SubstrateIndex);
      if ( !(o->flags&PAPI_OVERFLOW_HARDWARE) )
         o->timer_ms = PAPI_ITIMER_MS;
      else if (retval < PAPI_OK){
         if ( o->event_counter == 0 )
            o->flags = 0;
         papi_return(retval);
      }
   } else{
      o->timer_ms = PAPI_ITIMER_MS;
      o->flags &= ~(PAPI_OVERFLOW_HARDWARE);
   }

   APIDBG("Overflow using: %s\n", (o->flags&PAPI_OVERFLOW_HARDWARE?"[Hardware]":o->flags&PAPI_OVERFLOW_FORCE_SW?"[Forced Software]":"[Software]"));
   /* Toggle the overflow flag */
   if ((o->event_counter == 1 && threshold > 0) ||
       (o->event_counter == 0 && threshold == 0))
      ESI->state ^= PAPI_OVERFLOWING;

   if ( o->event_counter == 0 )
      o->flags = 0;

   return(PAPI_OK);
}

int PAPI_sprofil(PAPI_sprofil_t * prof, int profcnt, int EventSet, 
                    int EventCode, int threshold, int flags)
{
   EventSetInfo_t *ESI;
   int retval, index, i, buckets;
   int forceSW=0;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
      papi_return(PAPI_EISRUN);

   if ((index = _papi_hwi_lookup_EventCodeIndex(ESI, EventCode)) < 0)
      papi_return(PAPI_ENOEVNT);

   /* We do not support derived events in overflow */
   /* Unless it's DERIVED_CMPD in which no calculations are done */
   if ((ESI->EventInfoArray[index].derived) && 
       (ESI->EventInfoArray[index].derived != DERIVED_CMPD) &&
       !(flags&PAPI_PROFIL_FORCE_SW) ) 
      papi_return(PAPI_EINVAL);

   if ( prof == NULL )
     profcnt = 0;

   /* check all profile regions for valid scale factors of:
      2 (131072/65536),
      1 (65536/65536),
      or < 1 (65535 -> 2) as defined in unix profil()
      2/65536 is reserved for single bucket profiling
      {0,1}/65536 are traditionally used to terminate profiling
      but are unused here since PAPI uses threshold instead
   */
   for(i=0;i<profcnt;i++) {
      if (!((prof[i].pr_scale == 131072) ||
           ((prof[i].pr_scale <= 65536 && prof[i].pr_scale > 1)))) {
         APIDBG("Improper scale factor: %d\n", prof[i].pr_scale);
         papi_return(PAPI_EINVAL);
      }
   }

   if (threshold < 0)
      papi_return(PAPI_EINVAL);

   /* the first time to call PAPI_sprofil */
   if (!(ESI->state & PAPI_PROFILING)) {
      if (threshold == 0)
         papi_return(PAPI_EINVAL);
   }
   if (threshold > 0 && ESI->profile.event_counter >= _papi_hwi_substrate_info[ESI->SubstrateIndex].num_cntrs)
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
      if ( ESI->profile.event_counter > 0 ) {
        if ( (flags&PAPI_PROFIL_FORCE_SW) && !(ESI->profile.flags&PAPI_PROFIL_FORCE_SW) )
          papi_return(PAPI_ECNFLCT);
        if ( !(flags&PAPI_PROFIL_FORCE_SW) && (ESI->profile.flags&PAPI_PROFIL_FORCE_SW) )
          papi_return(PAPI_ECNFLCT);
      }

      for (i = 0; i < ESI->profile.event_counter; i++) {
         if (ESI->profile.EventCode[i] == EventCode)
            break;
      }

      if (i == ESI->profile.event_counter){
         i = ESI->profile.event_counter;
         ESI->profile.event_counter++;
         ESI->profile.EventCode[i] = EventCode;
      }
      ESI->profile.prof[i] = prof;
      ESI->profile.count[i] = profcnt;
      ESI->profile.threshold[i] = threshold;
      ESI->profile.EventIndex[i] = index;
   }

   /* make sure no invalid flags are set */
   if (flags & ~(PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM | PAPI_PROFIL_WEIGHTED
               | PAPI_PROFIL_COMPRESS | PAPI_PROFIL_BUCKETS | PAPI_PROFIL_FORCE_SW))
      papi_return(PAPI_EINVAL);

   if ( (flags&PAPI_PROFIL_FORCE_SW) ) 
      forceSW = PAPI_OVERFLOW_FORCE_SW;

   /* make sure one and only one bucket size is set */
   buckets = flags & PAPI_PROFIL_BUCKETS;
   if (!buckets) flags |= PAPI_PROFIL_BUCKET_16; /* default to 16 bit if nothing set */
   else { /* return error if more than one set */
      if (!((buckets == PAPI_PROFIL_BUCKET_16) ||
            (buckets == PAPI_PROFIL_BUCKET_32) || 
            (buckets == PAPI_PROFIL_BUCKET_64)))
         papi_return(PAPI_EINVAL);
   }

   /* Set up the option structure for the low level */

   ESI->profile.flags = flags;

   if (_papi_hwi_substrate_info[ESI->SubstrateIndex].supports_hw_profile && !forceSW)
      retval = _papi_hwd_set_profile(ESI, index, threshold, ESI->SubstrateIndex);
   else
      retval = PAPI_overflow(EventSet, EventCode, threshold, forceSW, _papi_hwi_dummy_handler);

   if (retval < PAPI_OK)
      return (retval);

   /* Toggle profiling flag */
   if ((ESI->profile.event_counter == 1 && threshold > 0) ||
       (ESI->profile.event_counter == 0 && threshold == 0))
      ESI->state ^= PAPI_PROFILING;

   if ( ESI->profile.event_counter == 0 )
     ESI->profile.flags = 0;

   return(PAPI_OK);
}

int PAPI_profil(void *buf, unsigned bufsiz, caddr_t offset,
                unsigned scale, int EventSet, int EventCode, int threshold, int flags)
{
   EventSetInfo_t *ESI;
   int i;
   int retval;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   /* scale factors are checked for validity in PAPI_sprofil */

   if (threshold > 0) {
      PAPI_sprofil_t *prof;

      for (i = 0; i < ESI->profile.event_counter; i++) {
         if (ESI->profile.EventCode[i] == EventCode)
            break;
      }

      if (i == ESI->profile.event_counter){
        prof = (PAPI_sprofil_t *) papi_malloc(sizeof(PAPI_sprofil_t));
        memset(prof, 0x0, sizeof(PAPI_sprofil_t));
        prof->pr_base = buf;
        prof->pr_size = bufsiz;
        prof->pr_off = offset;
        prof->pr_scale = scale;

        retval = PAPI_sprofil(prof, 1, EventSet, EventCode, threshold, flags);
        if ( retval != PAPI_OK )
           papi_free(prof);
      }
      else{
        prof = ESI->profile.prof[i];
        prof->pr_base = buf;
        prof->pr_size = bufsiz;
        prof->pr_off = offset;
        prof->pr_scale = scale;
        retval = PAPI_sprofil(prof, 1, EventSet, EventCode, threshold, flags);
      }
      papi_return(retval);
   }

   for (i = 0; i < ESI->profile.event_counter; i++) {
      if (ESI->profile.EventCode[i] == EventCode)
         break;
   }
   /* EventCode not found */
   if (i == ESI->profile.event_counter)
      papi_return(PAPI_EINVAL);

   papi_free(ESI->profile.prof[i]);
   ESI->profile.prof[i] = NULL;

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

   if ((Events == NULL) || (number <= 0))
      papi_return(PAPI_EINVAL);

   for (i = 0; i < number; i++) 
     {
       retval = PAPI_add_event(EventSet, Events[i]);
       if (retval != PAPI_OK)
	 {
	   if (i == 0)
	     papi_return(retval);
	   else
	     return(i);
	 }
     }
   return(PAPI_OK);
}

int PAPI_remove_events(int EventSet, int *Events, int number)
{
   int i, retval;

   if ((Events == NULL) || (number <= 0))
      papi_return(PAPI_EINVAL);

   for (i = 0; i < number; i++) 
     {
       retval = PAPI_remove_event(EventSet, Events[i]);
       if (retval != PAPI_OK)
	 {
	   if (i == 0)
	     papi_return(retval);
	   else
	     return(i);
	 }
     }
   return(PAPI_OK);
}

int PAPI_list_events(int EventSet, int *Events, int *number)
{
   EventSetInfo_t *ESI;
   int i,j;

   if ((Events == NULL) || (*number <= 0))
      papi_return(PAPI_EINVAL);

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (!ESI)
      papi_return(PAPI_ENOEVST);

   for (i=0,j=0; j < ESI->NumberOfEvents; i++) 
     {
       if (ESI->EventInfoArray[i].event_code != PAPI_NULL)
	 {
	   Events[j] = ESI->EventInfoArray[i].event_code;
	   j++;
	   if (j == *number)
	     break;
	 }
     }

  *number = j;

  return(PAPI_OK);
}

#ifdef PAPI_DMEM_INFO
long PAPI_get_dmem_info(int option)
{
   if (option != PAPI_GET_PAGESIZE) {
      return (_papi_hwd_get_dmem_info(option));
   } else
      return ((long) getpagesize());
}
#endif

const PAPI_exe_info_t *PAPI_get_executable_info(void)
{
   PAPI_option_t ptr;
   int retval;

   retval = PAPI_get_sbstr_opt(PAPI_EXEINFO, &ptr, 0);
   if (retval == PAPI_OK)
      return (ptr.exe_info);
   else
      return (NULL);
}

const PAPI_shlib_info_t *PAPI_get_shared_lib_info(void)
{
   PAPI_option_t ptr;
   int retval;

   retval = PAPI_get_sbstr_opt(PAPI_SHLIBINFO, &ptr, 0);
   if (retval == PAPI_OK)
      return (ptr.shlib_info);
   else
      return (NULL);
}

const PAPI_hw_info_t *PAPI_get_hardware_info(void)
{
   PAPI_option_t ptr;
   int retval;

   retval = PAPI_get_sbstr_opt(PAPI_HWINFO, &ptr, 0);
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

long_long PAPI_get_virt_cyc(void)
{
   ThreadInfo_t *master;
   int retval;
   
   if ((retval = _papi_hwi_lookup_or_create_thread(&master)) != PAPI_OK)
     papi_return(retval);

   return ((long_long)_papi_hwd_get_virt_cycles(&master->context[0]));
}

long_long PAPI_get_virt_usec(void)
{
   ThreadInfo_t *master;
   int retval;

   if ((retval = _papi_hwi_lookup_or_create_thread(&master)) != PAPI_OK)
     papi_return(retval);

   return ((long_long)_papi_hwd_get_virt_usec(&master->context[0]));
}

int PAPI_lock(int lck)
{
  if ((lck < 0) || (lck >= PAPI_NUM_LOCK))
    papi_return(PAPI_EINVAL);

  papi_return(_papi_hwi_lock(lck));
}

int PAPI_unlock(int lck)
{
  if ((lck < 0) || (lck >= PAPI_NUM_LOCK))
    papi_return(PAPI_EINVAL);

  papi_return(_papi_hwi_unlock(lck));
}

int PAPI_is_initialized(void)
{
   return (init_level);
}

/* This function maps the overflow_vector to event indexes in the event
   set, so that user can know which PAPI event overflowed.
   int *array---- an array of event indexes in eventset; the first index
                  maps to the highest set bit in overflow_vector
   int *number--- this is an input/output parameter, user should put the
                  size of the array into this parameter, after the function
                  is executed, the number of indexes in *array is written
                  to this parameter
*/
int PAPI_get_overflow_event_index(int EventSet, long_long overflow_vector, int *array, int *number)
{
   EventSetInfo_t *ESI;
   int set_bit, j, pos;
   int count = 0, k;

   if (overflow_vector == (long_long)0)
      papi_return(PAPI_EINVAL);

   if ((array == NULL) || (number == NULL))
      papi_return(PAPI_EINVAL);

   if (*number < 1 ) 
      papi_return(PAPI_EINVAL);

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   /* in case the eventset is empty */
   if (ESI->NumberOfEvents == 0 )
      papi_return(PAPI_EINVAL);

   while ((set_bit = ffsll(overflow_vector)))
   {
	   set_bit -= 1;
	   overflow_vector ^= (long_long)1 << set_bit;
	   for(j=0; j< ESI->NumberOfEvents; j++ )
	   {
	      for(k = 0, pos = 0; k < PAPI_MAX_COUNTER_TERMS && pos >= 0; k++) 
		  {
		     pos = ESI->EventInfoArray[j].pos[k];
		     if ((set_bit == pos) && 
               ((ESI->EventInfoArray[j].derived == NOT_DERIVED) || 
                (ESI->EventInfoArray[j].derived == DERIVED_CMPD))) 
		     { 
		        array[count++] = j;
                if (count == *number) 
                   return(PAPI_OK);

		        break;
		     }
		  }
	   }
   }
   *number = count;
   return(PAPI_OK);
}
