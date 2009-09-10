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
extern PAPI_debug_handler_t _papi_hwi_debug_handler;
extern papi_mdi_t _papi_hwi_system_info;
extern void _papi_hwi_dummy_handler(int,void*,long long,void*);

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
      papi_return(PAPI_ENOINIT);

   if ((init_level&PAPI_THREAD_LEVEL_INITED))
      papi_return(PAPI_OK);

   if (!SUPPORTS_MULTIPLE_THREADS(_papi_hwi_system_info))
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

   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_ENOINIT);
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

int PAPI_list_threads(PAPI_thread_id_t *id, int *num)
{
  PAPI_all_thr_spec_t tmp;
  int retval;

  /* If id == NULL, then just count the threads, don't gather a list. */
  /* If id != NULL, then we need the length of the id array in num. */

  if ((num == NULL) || (id && (*num <= 0)))
    papi_return(PAPI_EINVAL);

  memset(&tmp,0x0,sizeof(tmp));

  /* data == NULL, since we don't want the thread specific pointers. */
  /* id may be NULL, if the user doesn't want the thread ID's. */

  tmp.num = *num;
  tmp.id = id;
  tmp.data = NULL;

  retval = _papi_hwi_gather_all_thrspec_data(0,&tmp);
  if (retval == PAPI_OK)
    *num = tmp.num;

  papi_return(retval);
}

/*
 * Return a pointer to the stored thread information.
 */
int PAPI_get_thr_specific(int tag, void **ptr)
{
   ThreadInfo_t *thread;
   int doall = 0, retval = PAPI_OK;

   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_ENOINIT);
   if (tag & PAPI_TLS_ALL_THREADS)
     {
       tag = tag ^ PAPI_TLS_ALL_THREADS;
       doall = 1;
     }
   if ((tag < 0) || (tag > PAPI_TLS_NUM))
      papi_return(PAPI_EINVAL);

   if (doall)
     papi_return(_papi_hwi_gather_all_thrspec_data(tag,(PAPI_all_thr_spec_t *)ptr));
   
   retval = _papi_hwi_lookup_or_create_thread(&thread);
   if (retval == PAPI_OK)
     *ptr = thread->thread_storage[tag];
   else
     papi_return(retval);

   return(PAPI_OK);
}

/*
 * Store a pointer to memory provided by the thread
 */
int PAPI_set_thr_specific(int tag, void *ptr)
{
   ThreadInfo_t *thread;
   int retval = PAPI_OK;

   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_ENOINIT);
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
  int tmp = 0, tmpel;
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
	   if (strstr(var,"MEMORY"))
	     _papi_hwi_debug |= DEBUG_MEMORY;
	   if (strstr(var,"LEAK"))
	     _papi_hwi_debug |= DEBUG_LEAK;
	   if (strstr(var,"ALL"))
	     _papi_hwi_debug |= DEBUG_ALL;
	 }

       if (_papi_hwi_debug == 0)
	 _papi_hwi_debug |= DEBUG_API;
     }
#endif

   /* Be verbose for now */

   tmpel = _papi_hwi_error_level;
   _papi_hwi_error_level = PAPI_VERB_ECONT;

   /* Initialize internal globals */

   if (_papi_hwi_init_global_internal() != PAPI_OK) {
     _in_papi_library_init_cnt--;
     _papi_hwi_error_level = tmpel;
     papi_return(PAPI_EINVAL);
   }

   /* Initialize substrate globals */

   tmp = _papi_hwi_init_global();
   if (tmp) {
      init_retval = tmp;
      _papi_hwi_shutdown_global_internal();
      _in_papi_library_init_cnt--;
     _papi_hwi_error_level = tmpel;
      papi_return(init_retval);
   }

#ifdef __CATAMOUNT__
	/*	Set default for signal to ignore to prevent stray
	 *	interrupts from coming in at the end or profiling
	 *	or multiplexing. Allows multiplexing, overflow, and
	 *	profile tests to execute (although some do FAIL).
	 */
	do {
		struct sigaction sigi;
		sigi.sa_sigaction = (void *) SIG_IGN;
		sigi.sa_flags = SA_RESTART;
		sigemptyset (&sigi.sa_mask);
		(void) sigaction (SIGPROF, &sigi, NULL);
		(void) sigaction (SIGALRM, &sigi, NULL);
	} while (0);
#endif

   /* Initialize thread globals, including the main threads
      substrate */

   tmp = _papi_hwi_init_global_threads();
   if (tmp) {
      init_retval = tmp;
      _papi_hwi_shutdown_global_internal();
      _papi_hwd_shutdown_global();
      _in_papi_library_init_cnt--;
     _papi_hwi_error_level = tmpel;
      papi_return(init_retval);
   }

   init_level = PAPI_LOW_LEVEL_INITED;
   _in_papi_library_init_cnt--;
   _papi_hwi_error_level = tmpel;
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
#ifndef PAPI_MOD_EVENTS /* defined if a modifiable event table is supported */
      papi_return(PAPI_ESBSTR);
#else
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
	if (_papi_hwi_derived_type(info->derived,&type) != PAPI_OK)
	  return(PAPI_EINVAL);
      }
      papi_return(_papi_hwi_set_event_info(info, EventCode));
   }
 
/* Someday we might also handle native events...
   if (info->event_code & PAPI_NATIVE_MASK) {
      papi_return(_papi_hwi_set_native_event(info, EventCode));
}*/

   papi_return(PAPI_ENOEVNT);
#endif /* PAPI_MOD_EVENTS */
}

/* PAPI_encode_events:
   event_file -- pathname for a csv file containing event definitions
   replace -- true if existing events can be replaced

   This function reads event definitions from a comma separated values file.
   It allows a user or tool to add or override sets of PAPI events 
   on the fly. Bails if events to be modified have been added to existing EventSets.
*/

#ifdef PAPI_MOD_EVENTS /* defined if a modifiable event table is supported */
/* copy the contents of a potentially quoted string */
static char *quotcpy(char *d, char *s) {
   if (s && *s == '\"') {
      s++;
      s[strlen(s)-1]=0;
   }
   strcpy(d, s);
   return (d);
}
#endif

int PAPI_encode_events(char * event_file, int replace)
{
#ifndef PAPI_MOD_EVENTS /* defined if a modifiable event table is supported */
      papi_return(PAPI_ESBSTR);
#else

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
   #endif /* PAPI_MOD_EVENTS */
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

   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_ENOINIT);

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
  modifier can specify {all / available} for presets, or other values for native tables 
  and may be platform specific (Major groups / all mask bits; P / M / E chip, etc) */
int PAPI_enum_event(int *EventCode, int modifier)
{
   int i = *EventCode;

   if (i & PAPI_PRESET_MASK) {
	   if (modifier == PAPI_ENUM_FIRST) {
			 *EventCode = PAPI_PRESET_MASK;
			 return (PAPI_OK);
	   }
       i &= PAPI_PRESET_AND_MASK;
       while (++i < PAPI_MAX_PRESET_EVENTS) 
	 {
	   if (_papi_hwi_presets.info[i].symbol == NULL)
	     return (PAPI_ENOEVNT); /* NULL pointer terminates list */
	   if (modifier & PAPI_PRESET_ENUM_AVAIL)
	     {
	       if (_papi_hwi_presets.count[i] == 0)
		 continue;
	     }
	   *EventCode = i | PAPI_PRESET_MASK;
	   return (PAPI_OK);
         }
   }
   else if (i & PAPI_NATIVE_MASK) {
       /* Should check against num native events here */
       return (_papi_hwd_ntv_enum_events((unsigned int *) EventCode, modifier));
   }
   papi_return (PAPI_EINVAL);
}

int PAPI_create_eventset(int *EventSet)
{
   ThreadInfo_t *master;
   int retval;

   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_ENOINIT);
   retval = _papi_hwi_lookup_or_create_thread(&master);
   if (retval)
     papi_return(retval);

   papi_return (_papi_hwi_create_eventset(EventSet, master));
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

   papi_return (_papi_hwi_add_pevent(ESI, code, inout));
}

int PAPI_add_event(int EventSet, int EventCode)
{
   EventSetInfo_t *ESI;

   /* Is the EventSet already in existence? */

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);

   /* Check argument for validity */

   if (((EventCode & PAPI_PRESET_MASK) == 0) && 
       (EventCode & PAPI_NATIVE_MASK) == 0)
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

   thread = ESI->master;

   /* only one event set can be running at any time, so if another event
      set is running, the user must stop that event set explicitly */

   if (thread->running_eventset)
      papi_return(PAPI_EISRUN);

   /* Check that there are added events */

   if (ESI->NumberOfEvents < 1)
      papi_return(PAPI_EINVAL);

   /* If multiplexing is enabled for this eventset,
      call John May's code. */

   if (_papi_hwi_is_sw_multiplex(ESI))
     {
      retval = MPX_start(ESI->multiplex.mpx_evset);
      if (retval != PAPI_OK)
         papi_return(retval);

      /* Update the state of this EventSet */

      ESI->state ^= PAPI_STOPPED;
      ESI->state |= PAPI_RUNNING;

      return (PAPI_OK);
   }

   /* Short circuit this stuff if there's nothing running */

   /* If overflowing is enabled, turn it on */

   if ((ESI->state & PAPI_OVERFLOWING) && !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE))
     {
       APIDBG("Overflow using %s with timer %d, signal %d, ns %d\n", 
	      (ESI->overflow.flags&PAPI_OVERFLOW_FORCE_SW)?"[Forced Software]":"Software",
	      _papi_hwi_system_info.sub_info.itimer_num, 
	      _papi_hwi_system_info.sub_info.itimer_sig,
	      _papi_hwi_system_info.sub_info.itimer_ns);

       retval = _papi_hwi_start_signal(_papi_hwi_system_info.sub_info.itimer_sig, NEED_CONTEXT);
       if (retval != PAPI_OK)
	 papi_return(retval);

       /* Update the state of this EventSet and thread before to avoid races */
       ESI->state ^= PAPI_STOPPED;
       ESI->state |= PAPI_RUNNING;
       if (!(ESI->state & PAPI_ATTACHED))
	 thread->running_eventset = ESI;

       retval = _papi_hwd_start(&thread->context, &ESI->machdep);
       if (retval != PAPI_OK) {
	 _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.itimer_sig);
	 ESI->state ^= PAPI_RUNNING;
	 ESI->state |= PAPI_STOPPED;
	 if (!(ESI->state & PAPI_ATTACHED))
	   thread->running_eventset = NULL;
	 papi_return(retval);
       }

       retval = _papi_hwi_start_timer(_papi_hwi_system_info.sub_info.itimer_num, 
				      _papi_hwi_system_info.sub_info.itimer_sig,
				      _papi_hwi_system_info.sub_info.itimer_ns);
       if (retval != PAPI_OK) {
	 _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.itimer_sig);
	 _papi_hwd_stop(&thread->context, &ESI->machdep);
	 ESI->state ^= PAPI_RUNNING;
	 ESI->state |= PAPI_STOPPED;
	 if (!(ESI->state & PAPI_ATTACHED))
	   thread->running_eventset = NULL;
	 papi_return(retval);
       }
     }
   else
     {
       /* Update the state of this EventSet and thread before to avoid races */
       ESI->state ^= PAPI_STOPPED;
       ESI->state |= PAPI_RUNNING;
       if (!(ESI->state & PAPI_ATTACHED))
	 thread->running_eventset = ESI;

       retval = _papi_hwd_start(&thread->context, &ESI->machdep);
       if (retval != PAPI_OK) {
	 _papi_hwd_stop(&thread->context, &ESI->machdep);
	 ESI->state ^= PAPI_RUNNING;
	 ESI->state |= PAPI_STOPPED;
	 if (!(ESI->state & PAPI_ATTACHED))
	   thread->running_eventset = NULL;
	 papi_return(retval);
       }
     }

   return (retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */

int PAPI_stop(int EventSet, long long * values)
{
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;
   int retval;

   APIDBG("PAPI_stop\n");
   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if (!(ESI->state & PAPI_RUNNING))
      papi_return(PAPI_ENOTRUN);

   /* If multiplexing is enabled for this eventset, turn if off */

   if (_papi_hwi_is_sw_multiplex(ESI)) {
      retval = MPX_stop(ESI->multiplex.mpx_evset, values);
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

   /* Remove the control bits from the active counter config. */

   retval = _papi_hwd_stop(&thread->context, &ESI->machdep);
   if (retval != PAPI_OK)
      papi_return(retval);
   if (values)
      memcpy(values, ESI->sw_stop, ESI->NumberOfEvents * sizeof(long long));

   /* If kernel profiling is in use, flush and process the kernel buffer */

   if (ESI->state & PAPI_PROFILING) {
     if (_papi_hwi_system_info.sub_info.kernel_profile && !(ESI->profile.flags&PAPI_PROFIL_FORCE_SW)) {
         retval = _papi_hwd_stop_profiling(thread, ESI);
         if (retval < PAPI_OK)
            papi_return(retval);
      }
   }

   /* If overflowing is enabled, turn it off */

   if (ESI->state & PAPI_OVERFLOWING) 
     {
       if (!(ESI->overflow.flags&PAPI_OVERFLOW_HARDWARE))
	 {
	   retval = _papi_hwi_stop_timer(_papi_hwi_system_info.sub_info.itimer_num, _papi_hwi_system_info.sub_info.itimer_sig);
	   if (retval != PAPI_OK)
	     papi_return(retval);
	   _papi_hwi_stop_signal(_papi_hwi_system_info.sub_info.itimer_sig);
	 }
     }

   /* Update the state of this EventSet */

   ESI->state ^= PAPI_RUNNING;
   ESI->state |= PAPI_STOPPED;

   /* Update the running event set for this thread */

   /* #warning "I don't know why this is here" */

   if (!(ESI->state & PAPI_ATTACHED))
     thread->running_eventset = NULL ;

#if defined(DEBUG)
     if (_papi_hwi_debug & DEBUG_API)
       {
	 int i;
	 for (i = 0; i < ESI->NumberOfEvents; i++)
	   APIDBG("PAPI_stop ESI->sw_stop[%d]:\t%llu\n", i, ESI->sw_stop[i]);
       }
#endif

   return (PAPI_OK);
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
      if (_papi_hwi_is_sw_multiplex(ESI))
         retval = MPX_reset(ESI->multiplex.mpx_evset);
      else {
         /* If we're not the only one running, then just
            read the current values into the ESI->start
            array. This holds the starting value for counters
            that are shared. */

         retval = _papi_hwd_reset(&thread->context, &ESI->machdep);

      }
   } else {
      memset(ESI->sw_stop, 0x00, ESI->NumberOfEvents * sizeof(long long));
   }

   APIDBG("PAPI_reset returns %d\n", retval);
   papi_return(retval);
}

int PAPI_read(int EventSet, long long * values)
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
      if (_papi_hwi_is_sw_multiplex(ESI))
         retval = MPX_read(ESI->multiplex.mpx_evset, values);
      else
         retval = _papi_hwi_read(&thread->context, ESI, values);
      if (retval != PAPI_OK)
         papi_return(retval);
   } else {
      memcpy(values, ESI->sw_stop, ESI->NumberOfEvents * sizeof(long long));
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
   return (PAPI_OK);
}

int PAPI_read_ts(int EventSet, long long * values, long long *cyc)
{
   int retval;

   retval = PAPI_read(EventSet, values);
   *cyc = _papi_hwd_get_real_cycles();
   papi_return(retval);
}

int PAPI_accum(int EventSet, long long * values)
{
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;
   int i, retval;
   long long a, b, c;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if (values == NULL)
      papi_return(PAPI_EINVAL);

   if (ESI->state & PAPI_RUNNING) {
      if (_papi_hwi_is_sw_multiplex(ESI))
         retval = MPX_read(ESI->multiplex.mpx_evset, ESI->sw_stop);
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

int PAPI_write(int EventSet, long long * values)
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

   memcpy(ESI->hw_start, values, _papi_hwi_system_info.sub_info.num_cntrs * sizeof(long long));

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
          _papi_hwi_system_info.sub_info.hardware_intr && !(ESI->profile.flags&PAPI_PROFIL_FORCE_SW)) {
      total=ESI->profile.event_counter;
      for (i = 0; i < total; i++) {
         retval = PAPI_sprofil(NULL,0,EventSet,ESI->profile.EventCode[0],0,
                               PAPI_PROFIL_POSIX);
         if (retval != PAPI_OK)
            papi_return(retval);
      }
   }

   if (_papi_hwi_is_sw_multiplex(ESI)) {
      retval = MPX_cleanup(&ESI->multiplex.mpx_evset);
      if (retval != PAPI_OK)
         papi_return(retval);
   }

   /* Now do the magic */

   papi_return(_papi_hwi_cleanup_eventset(ESI));
}

int PAPI_multiplex_init(void)
{
   int retval;

   retval = mpx_init(_papi_hwi_system_info.sub_info.itimer_ns);
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
  return(PAPI_set_opt(PAPI_DEBUG,&option));
}

/* Attaches to or detaches from the specified thread id */
inline_static int _papi_set_attach(int option, int EventSet, unsigned long tid)
{
  PAPI_option_t attach;

  memset(&attach,0x0,sizeof(attach));
  attach.attach.eventset = EventSet;
  attach.attach.tid = tid;
  return(PAPI_set_opt(option,&attach));
}

int PAPI_attach(int EventSet, unsigned long tid)
{
  return(_papi_set_attach(PAPI_ATTACH, EventSet, tid));
}

int PAPI_detach(int EventSet)
{
  return(_papi_set_attach(PAPI_DETACH, EventSet, 0));
}

int PAPI_set_multiplex(int EventSet)
{
   PAPI_option_t mpx;
   int ret;

   if ((ret = mpx_check(EventSet)) != PAPI_OK)
	papi_return(ret);
   memset(&mpx,0x0,sizeof(mpx));
   mpx.multiplex.eventset = EventSet;
   mpx.multiplex.flags = PAPI_MULTIPLEX_DEFAULT;
   mpx.multiplex.ns = _papi_hwi_system_info.sub_info.itimer_ns;
   return(PAPI_set_opt(PAPI_MULTIPLEX, &mpx));
}

int PAPI_set_opt(int option, PAPI_option_t * ptr)
{
   _papi_int_option_t internal;
   int retval = PAPI_OK;
   ThreadInfo_t *thread = NULL;

   if ((option != PAPI_DEBUG) && (init_level == PAPI_NOT_INITED))
      papi_return(PAPI_ENOINIT);
   if (ptr == NULL)
      papi_return(PAPI_EINVAL);

   memset(&internal, 0x0, sizeof(_papi_int_option_t));

   switch (option) {
   case PAPI_DETACH:
     {
       if (_papi_hwi_system_info.sub_info.attach == 0)
	 papi_return(PAPI_ESBSTR);

       internal.attach.ESI = _papi_hwi_lookup_EventSet(ptr->attach.eventset);
       if (internal.attach.ESI == NULL)
	 papi_return(PAPI_ENOEVST);

       if ((internal.attach.ESI->state & PAPI_STOPPED) == 0)
	 papi_return(PAPI_EISRUN);

       if ((internal.attach.ESI->state & PAPI_ATTACHED) == 0)
	 papi_return(PAPI_EINVAL);

       internal.attach.tid = internal.attach.ESI->attach.tid;
       thread = internal.attach.ESI->master;
       retval = _papi_hwd_ctl(&thread->context, PAPI_DETACH, &internal);
       if (retval != PAPI_OK)
	 papi_return(retval);

       internal.attach.ESI->state ^= PAPI_ATTACHED;
       internal.attach.ESI->attach.tid = 0;
       return(PAPI_OK);
     }
   case PAPI_ATTACH:
     {
       if (_papi_hwi_system_info.sub_info.attach == 0)
	 papi_return(PAPI_ESBSTR);

       internal.attach.ESI = _papi_hwi_lookup_EventSet(ptr->attach.eventset);
       if (internal.attach.ESI == NULL)
	 papi_return(PAPI_ENOEVST);

       if ((internal.attach.ESI->state & PAPI_STOPPED) == 0)
	 papi_return(PAPI_EISRUN);

       if (internal.attach.ESI->state & PAPI_ATTACHED)
	 papi_return(PAPI_EINVAL);

       internal.attach.tid = ptr->attach.tid;
       thread = internal.attach.ESI->master;
       retval = _papi_hwd_ctl(&thread->context, PAPI_ATTACH, &internal);
       if (retval != PAPI_OK)
	 papi_return(retval);

       internal.attach.ESI->state |= PAPI_ATTACHED;
       internal.attach.ESI->attach.tid = ptr->attach.tid;
       return(PAPI_OK);
     }
   case PAPI_DEF_MPX_NS:
     {
       if (ptr->multiplex.ns < 0)
	 papi_return(PAPI_EINVAL);
       internal.multiplex.ns = ptr->multiplex.ns;
       /* Low level just checks/adjusts the args for this substrate */
       retval = _papi_hwd_ctl(&thread->context, PAPI_DEF_MPX_NS, &internal);
       if (retval == PAPI_OK)
	 {
	   _papi_hwi_system_info.sub_info.itimer_ns = internal.multiplex.ns;
	   ptr->multiplex.ns = internal.multiplex.ns;
	 }
       papi_return(retval);
     }
   case PAPI_DEF_ITIMER_NS:
     {
       if (ptr->itimer.ns < 0)
	 papi_return(PAPI_EINVAL);
       internal.itimer.ns = ptr->itimer.ns;
       /* Low level just checks/adjusts the args for this substrate */
       retval = _papi_hwd_ctl(NULL, PAPI_DEF_ITIMER_NS, &internal);
       if (retval == PAPI_OK)
	 {
	   _papi_hwi_system_info.sub_info.itimer_ns = internal.itimer.ns;
	   ptr->itimer.ns = internal.itimer.ns;
	 }
       papi_return(retval);
     }
   case PAPI_DEF_ITIMER:
     {
       if (ptr->itimer.ns < 0)
	 papi_return(PAPI_EINVAL);
       memcpy(&internal.itimer,&ptr->itimer,sizeof(PAPI_itimer_option_t));
       /* Low level just checks/adjusts the args for this substrate */
       retval = _papi_hwd_ctl(NULL, PAPI_DEF_ITIMER, &internal);
       if (retval == PAPI_OK) {
	 _papi_hwi_system_info.sub_info.itimer_num = ptr->itimer.itimer_num;
	 _papi_hwi_system_info.sub_info.itimer_sig = ptr->itimer.itimer_sig;
	 if (ptr->itimer.ns > 0)
	   _papi_hwi_system_info.sub_info.itimer_ns = ptr->itimer.ns;
	 /* flags are currently ignored, eventually the flags will be able
	    to specify whether or not we use POSIX itimers (clock_gettimer) */
       }
       papi_return(retval);
     }
   case PAPI_MULTIPLEX:
      {
         EventSetInfo_t *ESI;
         ESI = _papi_hwi_lookup_EventSet(ptr->multiplex.eventset);
         if (ESI == NULL)
            papi_return(PAPI_ENOEVST);
         if (!(ESI->state & PAPI_STOPPED))
            papi_return(PAPI_EISRUN);
         if (ESI->state & PAPI_MULTIPLEXING)
            papi_return(PAPI_EINVAL);
	 if (ptr->multiplex.ns < 0)
	   papi_return(PAPI_EINVAL);
	 internal.multiplex.ESI = ESI;
	 internal.multiplex.ns = ptr->multiplex.ns;
	 internal.multiplex.flags = ptr->multiplex.flags;
	 if ((_papi_hwi_system_info.sub_info.kernel_multiplex) && ((ptr->multiplex.flags & PAPI_MULTIPLEX_FORCE_SW) == 0))
	   {
       	    thread = internal.multiplex.ESI->master;
            retval = _papi_hwd_ctl(&thread->context, PAPI_MULTIPLEX, &internal);
	}
	 /* Kernel or PAPI may have changed this value so send it back out to the user */
	 ptr->multiplex.ns = internal.multiplex.ns;
	    if (retval == PAPI_OK)
	   	papi_return(_papi_hwi_convert_eventset_to_multiplex(&internal.multiplex));
		return(retval);
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

	 /* Check what the substrate supports */

	 if (dom == PAPI_DOM_ALL)
	   dom = _papi_hwi_system_info.sub_info.available_domains;

	 if (dom & ~_papi_hwi_system_info.sub_info.available_domains)
	   papi_return(PAPI_EINVAL);

         _papi_hwi_system_info.sub_info.default_domain = dom;

         return (PAPI_OK);
      }
   case PAPI_DOMAIN:
      {
         int dom = ptr->domain.domain;
         if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
            papi_return(PAPI_EINVAL);

	 /* Check what the substrate supports */

	 if (dom == PAPI_DOM_ALL)
	   dom = _papi_hwi_system_info.sub_info.available_domains;

	 if (dom & ~_papi_hwi_system_info.sub_info.available_domains)
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
   case PAPI_DEFGRN:
      {
         int grn = ptr->defgranularity.granularity;
         if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
            papi_return(PAPI_EINVAL);

         /* Change the global structure. The _papi_hwd_init_control_state function 
	    in the substrates gets information from the global structure instead of
            per-thread information. */

	 /* Check what the substrate supports */

	 if (grn & ~_papi_hwi_system_info.sub_info.available_granularities)
	   papi_return(PAPI_EINVAL);

	 /* Make sure there is only 1 set. */
	 if (grn ^ (1 << (ffs(grn) - 1)))
	   papi_return(PAPI_EINVAL);
	 
         _papi_hwi_system_info.sub_info.default_granularity = grn;

         return (PAPI_OK);
      }
   case PAPI_GRANUL:
      {
         int grn = ptr->granularity.granularity;

         if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
            papi_return(PAPI_EINVAL);

	 /* Check what the substrate supports */

	 if (grn & ~_papi_hwi_system_info.sub_info.available_granularities)
	   papi_return(PAPI_EINVAL);

	 /* Make sure there is only 1 set. */
	 if (grn ^ (1 << (ffs(grn) - 1)))
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
#if 0
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
   case PAPI_DATA_ADDRESS:
   case PAPI_INSTR_ADDRESS:
      {

         EventSetInfo_t *ESI;

         ESI = _papi_hwi_lookup_EventSet(ptr->addr.eventset);
         if (ESI == NULL)
            papi_return(PAPI_ENOEVST);
         thread = ESI->master;

         internal.address_range.ESI = ESI;

         if (!(internal.address_range.ESI->state & PAPI_STOPPED))
            papi_return(PAPI_EISRUN);

         /*set domain to be PAPI_DOM_USER*/
         internal.address_range.domain = PAPI_DOM_USER;

         internal.address_range.start = ptr->addr.start;
         internal.address_range.end = ptr->addr.end;
         retval = _papi_hwd_ctl(&thread->context, option, &internal);
         ptr->addr.start_off = internal.address_range.start_off;
         ptr->addr.end_off = internal.address_range.end_off;
         papi_return (retval);
     }
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
  EventSetInfo_t *ESI;
	 
   if ((option != PAPI_DEBUG) && (init_level == PAPI_NOT_INITED))
      papi_return(PAPI_ENOINIT);

   switch (option) {
   case PAPI_DETACH:
     {
       if (ptr == NULL)
	 papi_return(PAPI_EINVAL);
       ESI = _papi_hwi_lookup_EventSet(ptr->attach.eventset);
       if (ESI == NULL)
	 papi_return(PAPI_ENOEVST);
       ptr->attach.tid = ESI->attach.tid;
       return ((ESI->state & PAPI_ATTACHED) == 0);
     }
   case PAPI_ATTACH:
     {
       if (ptr == NULL)
	 papi_return(PAPI_EINVAL);
       ESI = _papi_hwi_lookup_EventSet(ptr->attach.eventset);
       if (ESI == NULL)
	 papi_return(PAPI_ENOEVST);
       ptr->attach.tid = ESI->attach.tid;
       return ((ESI->state & PAPI_ATTACHED) != 0);
     }
   case PAPI_DEF_MPX_NS:
     {
       if (ptr == NULL)
	 papi_return(PAPI_EINVAL);
       ptr->multiplex.ns = _papi_hwi_system_info.sub_info.itimer_ns;
       return(PAPI_OK);
     }
   case PAPI_DEF_ITIMER_NS:
     {
       if (ptr == NULL)
	 papi_return(PAPI_EINVAL);
       ptr->itimer.ns = _papi_hwi_system_info.sub_info.itimer_ns;
       return(PAPI_OK);
     }
   case PAPI_DEF_ITIMER:
     {
       if (ptr == NULL)
	 papi_return(PAPI_EINVAL);
       ptr->itimer.itimer_num = _papi_hwi_system_info.sub_info.itimer_num;
       ptr->itimer.itimer_sig = _papi_hwi_system_info.sub_info.itimer_sig;
       ptr->itimer.ns = _papi_hwi_system_info.sub_info.itimer_ns;
       ptr->itimer.flags = 0;
       return(PAPI_OK);
     }
   case PAPI_MULTIPLEX:
      {
	 if (ptr == NULL)
	   papi_return(PAPI_EINVAL);
         ESI = _papi_hwi_lookup_EventSet(ptr->multiplex.eventset);
         if (ESI == NULL)
            papi_return(PAPI_ENOEVST);
	 ptr->multiplex.ns = ESI->multiplex.ns;
	 ptr->multiplex.flags = ESI->multiplex.flags;
         return (ESI->state & PAPI_MULTIPLEXING) != 0;
      }
   case PAPI_PRELOAD:
     if (ptr == NULL)
       papi_return(PAPI_EINVAL);
     memcpy(&ptr->preload,&_papi_hwi_system_info.preload_info,sizeof(PAPI_preload_info_t));
     break;
   case PAPI_DEBUG:
     if (ptr == NULL)
       papi_return(PAPI_EINVAL);
     ptr->debug.level = _papi_hwi_error_level;
     ptr->debug.handler = _papi_hwi_debug_handler;
     break;
   case PAPI_CLOCKRATE:
      return ((int) _papi_hwi_system_info.hw_info.mhz);
   case PAPI_MAX_CPUS:
      return (_papi_hwi_system_info.hw_info.ncpu);
  /* For now, MAX_HWCTRS and MAX CTRS are identical.
     At some future point, they may map onto different values.
  */
   case PAPI_MAX_HWCTRS:
      return (_papi_hwi_system_info.sub_info.num_cntrs);
   case PAPI_MAX_MPX_CTRS:
      return (_papi_hwi_system_info.sub_info.num_mpx_cntrs);
   case PAPI_DEFDOM:
      return (_papi_hwi_system_info.sub_info.default_domain);
   case PAPI_DEFGRN:
      return (_papi_hwi_system_info.sub_info.default_granularity);
#if 0
   case PAPI_INHERIT:
      {
         EventSetInfo_t *tmp;
         tmp = _papi_hwi_lookup_in_thread_list();
         if (tmp == NULL)
            return (PAPI_EINVAL);

         return (tmp->inherit.inherit);
      }
#endif
   case PAPI_GRANUL:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      ESI = _papi_hwi_lookup_EventSet(ptr->granularity.eventset);
      if (ESI == NULL)
	papi_return(PAPI_ENOEVST);
      ptr->granularity.granularity = ESI->granularity.granularity;
	  break;
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
   case PAPI_SUBSTRATEINFO:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      ptr->sub_info = &_papi_hwi_system_info.sub_info;
      return(PAPI_OK);
   case PAPI_DOMAIN:
      if (ptr == NULL)
         papi_return(PAPI_EINVAL);
      ESI = _papi_hwi_lookup_EventSet(ptr->domain.eventset);
      if (ESI == NULL)
	papi_return(PAPI_ENOEVST);
      ptr->domain.domain = ESI->domain.domain;
      return(PAPI_OK);
   case PAPI_LIB_VERSION:
      return (PAPI_VERSION);
   default:
      papi_return(PAPI_EINVAL);
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
   _papi_hwd_shutdown_global();

   /* Now it is safe to call re-init */

   init_retval = DEADBEEF;
   init_level = PAPI_NOT_INITED;
   _papi_cleanup_all_memory();
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
   int retval, index, i;
   EventSetInfo_t *ESI;
   ThreadInfo_t *thread;

   ESI = _papi_hwi_lookup_EventSet(EventSet);
   if (ESI == NULL)
      papi_return(PAPI_ENOEVST);
   thread = ESI->master;

   if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
      papi_return(PAPI_EISRUN);

   if (ESI->state & PAPI_ATTACHED)
      papi_return(PAPI_EINVAL);

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
      if ( ESI->overflow.event_counter > 0 ){
         if ( (flags&PAPI_OVERFLOW_FORCE_SW) && (ESI->overflow.flags&PAPI_OVERFLOW_HARDWARE))
            papi_return(PAPI_ECNFLCT);
         if ( !(flags&PAPI_OVERFLOW_FORCE_SW) && (ESI->overflow.flags&PAPI_OVERFLOW_FORCE_SW))
            papi_return(PAPI_ECNFLCT);
      }
      for (i = 0; i < ESI->overflow.event_counter; i++) {
         if (ESI->overflow.EventCode[i] == EventCode)
            break;
      }
      /* A new entry */
      if (i == ESI->overflow.event_counter) {
         ESI->overflow.EventCode[i] = EventCode;
         ESI->overflow.event_counter++;
      }
      /* New or existing entry */
      ESI->overflow.deadline[i] = threshold;
      ESI->overflow.threshold[i] = threshold;
      ESI->overflow.EventIndex[i] = index;
      ESI->overflow.flags = flags;
   }

   /* If overflowing is already active, we should check to
      make sure that we don't specify a different handler
      or different flags here. You can't mix them. */

   ESI->overflow.handler = handler;

   /* Set up the option structure for the low level.
      If we have hardware interrupts and we are not using
      forced software emulated interrupts */

   if (_papi_hwi_system_info.sub_info.hardware_intr && 
       !(ESI->overflow.flags & PAPI_OVERFLOW_FORCE_SW)) {
     retval = _papi_hwd_set_overflow(ESI, index, threshold);
     if (retval == PAPI_OK) 
       ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;
     else {
       papi_return (retval); /* We should undo stuff here */
     }
   } else {
     /* Make sure hardware overflow is not set */
     ESI->overflow.flags &= ~(PAPI_OVERFLOW_HARDWARE);
   }

   APIDBG("Overflow using: %s\n", (ESI->overflow.flags&PAPI_OVERFLOW_HARDWARE?"[Hardware]":ESI->overflow.flags&PAPI_OVERFLOW_FORCE_SW?"[Forced Software]":"[Software]"));

   /* Toggle the overflow flags and ESI state */

   if (ESI->overflow.event_counter >= 1)
      ESI->state |= PAPI_OVERFLOWING;
   else {
      ESI->state ^= PAPI_OVERFLOWING;
      ESI->overflow.flags = 0;
      ESI->overflow.handler = NULL;
   }

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

   if (ESI->state & PAPI_ATTACHED)
      papi_return(PAPI_EINVAL);

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

   APIDBG("Profile event counter is %d\n",ESI->profile.event_counter);

   /* Clear out old flags */
   if (threshold == 0) 
     flags |= ESI->profile.flags;

   /* make sure no invalid flags are set */
   if (flags & ~(PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM | PAPI_PROFIL_WEIGHTED
               | PAPI_PROFIL_COMPRESS | PAPI_PROFIL_BUCKETS | PAPI_PROFIL_FORCE_SW | PAPI_PROFIL_INST_EAR | PAPI_PROFIL_DATA_EAR))
      papi_return(PAPI_EINVAL);

   if ((flags & (PAPI_PROFIL_INST_EAR | PAPI_PROFIL_DATA_EAR)) && (_papi_hwi_system_info.sub_info.profile_ear == 0))
     papi_return(PAPI_ESBSTR);

   /* if we have kernel-based profiling, then we're just asking for signals on interrupt. */
   /* if we don't have kernel-based profiling, then we're asking for emulated PMU interrupt */
   if ((flags & PAPI_PROFIL_FORCE_SW) && (_papi_hwi_system_info.sub_info.kernel_profile == 0)) 
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

   if (_papi_hwi_system_info.sub_info.kernel_profile &&
       !(ESI->profile.flags & PAPI_PROFIL_FORCE_SW)) {
     retval = _papi_hwd_set_profile(ESI, index, threshold);
     if ((retval == PAPI_OK) && (threshold > 0)) {
       /* We need overflowing because we use the overflow dispatch handler */
       ESI->state |= PAPI_OVERFLOWING;
       ESI->overflow.flags |= PAPI_OVERFLOW_HARDWARE;
     }
   } else {
       retval = PAPI_overflow(EventSet, EventCode, threshold, forceSW, _papi_hwi_dummy_handler);
   }
   if (retval < PAPI_OK)
       papi_return (retval); /* We should undo stuff here */

   /* Toggle the profiling flags and ESI state */

   if (ESI->profile.event_counter >= 1)
     ESI->state |= PAPI_PROFILING;
   else {
     ESI->state ^= PAPI_PROFILING;
     ESI->profile.flags = 0;
   }

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

      if (i == ESI->profile.event_counter) {
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

   memset (&ptr, 0, sizeof(ptr));
   ptr.defgranularity.granularity = granularity;
   return(PAPI_set_opt(PAPI_DEFGRN, &ptr));
}

/* This function sets the low level default counting domain
   for all newly manufactured eventsets */

int PAPI_set_domain(int domain)
{
   PAPI_option_t ptr;

   memset (&ptr, 0, sizeof(ptr));
   ptr.defdomain.domain = domain;
   return(PAPI_set_opt(PAPI_DEFDOM, &ptr));
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

int PAPI_get_dmem_info(PAPI_dmem_info_t *dest)
{
  if (dest == NULL)
    return PAPI_EINVAL;

  memset((void *)dest,0x0,sizeof(PAPI_dmem_info_t));
  return(_papi_hwd_get_dmem_info(dest));
}


const PAPI_exe_info_t *PAPI_get_executable_info(void)
{
   PAPI_option_t ptr;
   int retval;

   memset (&ptr, 0, sizeof(ptr));
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

   memset (&ptr, 0, sizeof(ptr));
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

   memset (&ptr, 0, sizeof(ptr));
   retval = PAPI_get_opt(PAPI_HWINFO, &ptr);
   if (retval == PAPI_OK)
      return (ptr.hw_info);
   else
      return (NULL);
}

const PAPI_substrate_info_t *PAPI_get_substrate_info(void)
{
   PAPI_option_t ptr;
   int retval;

   memset (&ptr, 0, sizeof(ptr));
   retval = PAPI_get_opt(PAPI_SUBSTRATEINFO, &ptr);
   if (retval == PAPI_OK)
      return (ptr.sub_info);
   else
      return (NULL);
}

long long PAPI_get_real_cyc(void)
{
   return (_papi_hwd_get_real_cycles());
}

long long PAPI_get_real_nsec(void)
{
  return ((_papi_hwd_get_real_cycles()*1000LL)/(long long)_papi_hwi_system_info.hw_info.mhz);
}

long long PAPI_get_real_usec(void)
{
   return (_papi_hwd_get_real_usec());
}

long long PAPI_get_virt_cyc(void)
{
   ThreadInfo_t *master;
   int retval;
   
   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_ENOINIT);
   if ((retval = _papi_hwi_lookup_or_create_thread(&master)) != PAPI_OK)
     papi_return(retval);

   return (_papi_hwd_get_virt_cycles(&master->context));
}

long long PAPI_get_virt_nsec(void)
{
   ThreadInfo_t *master;
   int retval;

   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_ENOINIT);
   if ((retval = _papi_hwi_lookup_or_create_thread(&master)) != PAPI_OK)
     papi_return(retval);

   return ((_papi_hwd_get_virt_cycles(&master->context)*1000LL)/(long long)_papi_hwi_system_info.hw_info.mhz);
}

long long PAPI_get_virt_usec(void)
{
   ThreadInfo_t *master;
   int retval;

   if (init_level == PAPI_NOT_INITED)
      papi_return(PAPI_ENOINIT);
   if ((retval = _papi_hwi_lookup_or_create_thread(&master)) != PAPI_OK)
     papi_return(retval);

   return (_papi_hwd_get_virt_usec(&master->context));
}

int PAPI_restore(void)
{
   PAPIERROR("PAPI_restore is currently not implemented");
   return (PAPI_ESBSTR);
}
int PAPI_save(void)
{
  PAPIERROR("PAPI_save is currently not implemented");
   return (PAPI_ESBSTR);
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
int PAPI_get_overflow_event_index(int EventSet, long long overflow_vector, int *array, int *number)
{
   EventSetInfo_t *ESI;
   int set_bit, j, pos;
   int count = 0, k;

   if (overflow_vector == (long long)0)
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
	   overflow_vector ^= (long long)1 << set_bit;
	   for(j=0; j< ESI->NumberOfEvents; j++ )
	   {
	      for(k = 0, pos = 0; k < MAX_COUNTER_TERMS && pos >= 0; k++) 
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
