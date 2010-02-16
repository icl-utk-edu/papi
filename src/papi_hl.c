/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi_hl.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:      Kevin London
*           london@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file contains the 'high level' interface to PAPI. 
   BASIC is a high level language. ;-) */

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

/* high level papi functions*/

/*
 * Which high-level interface are we using?
 */
#define HL_START_COUNTERS	1
#define HL_FLIPS		2
#define HL_IPC			3
#define HL_FLOPS		4

/* Definitions for reading */
#define PAPI_HL_READ		1
#define PAPI_HL_ACCUM		2

/* 
 * This is stored per thread
 */
typedef struct _HighLevelInfo {
   int EventSet;                /* EventSet of the thread */
   short int num_evts;
   short int running;
   long long initial_time;    /* Start time */
   float total_proc_time;       /* Total processor time */
   float total_ins;             /* Total instructions */
} HighLevelInfo;

extern int init_level;
int _hl_rate_calls(float *real_time, float *proc_time, long long * ins, float *rate,
                   unsigned int EVENT, HighLevelInfo * state);
void _internal_cleanup_hl_info(HighLevelInfo * state);
int _internal_check_state(HighLevelInfo ** state);
int _internal_start_hl_counters(HighLevelInfo * state);
int _internal_hl_read_cnts(long long * values, int array_len, int flag);

/* CHANGE LOG:
  - ksl 10/17/03
   Pretty much a complete rewrite of the high level interface.  Now
   the interface is thread safe and you don't have to worry as much
   about mixing the various high level calls.

  - dkt 11/19/01:
   After much discussion with users and developers, removed FMA and SLOPE
   fudge factors. SLOPE was not being used, and we decided the place to
   apply FMA was at a higher level where there could be a better understanding
   of platform discrepancies and code implications.
   ALL PAPI CALLS NOW RETURN EXACTLY WHAT THE HARDWARE REPORTS
  - dkt 08/14/01:
   Added reinitialization of values and proc_time to new reinit code.
   Added SLOPE and FMA constants to correct for systemic errors on a
   platform-by-platform basis.
   SLOPE is a factor subtracted from flpins on each call to compensate
   for platform overhead in the call.
   FMA is a shifter that doubles floating point counts on platforms that
   count FMA as one op instead of two.
   NOTE: We are making the FLAWED assumption that ALL flpins are FMA!
   This will result in counts that are TOO HIGH on the affected platforms
   in instances where the code is NOT mostly FMA.
  - dkt 08/01/01:
   NOTE: Calling semantics have changed!
   Now, if flpins < 0 (an invalid value) a PAPI_reset is issued to reset the
   counter values. The internal start time is also reset. This should be a 
   benign change, exept in the rare case where a user passes an uninitialized
   (and possibly negative) value for flpins to the routine *AFTER* it has been
   called the first time. This is unlikely, since the first call clears and
   returns th is value.
  - dkt 08/01/01:
   Internal sequencing changes:
   -- initial PAPI_get_real_usec() call moved above PAPI_start to avoid unwanted flops.
   -- PAPI_accum() replaced with PAPI_start() / PAPI_stop pair for same reason.
*/

/*
 * This function is called to determine the state of the system.
 * We may as well set the HighLevelInfo so you don't have to look it
 * up again.
 */
int _internal_check_state(HighLevelInfo ** outgoing)
{
   int retval;
   HighLevelInfo *state = NULL;

   /* Only allow one thread at a time in here */
   if (init_level == PAPI_NOT_INITED) {
      retval = PAPI_library_init(PAPI_VER_CURRENT);
      if (retval != PAPI_VER_CURRENT) {
         return (retval);
      } else {
	 _papi_hwi_lock(HIGHLEVEL_LOCK);
         init_level = PAPI_HIGH_LEVEL_INITED;
         _papi_hwi_unlock(HIGHLEVEL_LOCK);
      }
   }

   /*
    * Do we have the thread specific data setup yet?
    */
   if ((retval = PAPI_get_thr_specific(PAPI_HIGH_LEVEL_TLS, (void *) &state))
       != PAPI_OK || state == NULL) {
      state = (HighLevelInfo *) papi_malloc(sizeof(HighLevelInfo));
      if (state == NULL)
         return (PAPI_ENOMEM);

      memset(state, 0, sizeof(HighLevelInfo));
      state->EventSet = -1;

      if ((retval = PAPI_create_eventset(&state->EventSet)) != PAPI_OK)
         return (retval);

      if ((retval = PAPI_set_thr_specific(PAPI_HIGH_LEVEL_TLS, state)) != PAPI_OK)
         return (retval);
   }
   *outgoing = state;
   return (PAPI_OK);
}

/*
 * Make sure to allocate space for values 
 */
int _internal_start_hl_counters(HighLevelInfo * state)
{
   return (PAPI_start(state->EventSet));
}

void _internal_cleanup_hl_info(HighLevelInfo * state)
{
   state->num_evts        = 0;
   state->running         = 0;
   state->initial_time    = -1;
   state->total_proc_time = 0;
   state->total_ins       = 0;
   return;
}

/*
 * The next three calls all use _hl_rate_calls() to return an instruction rate value.
 * PAPI_flips returns information related to floating point instructions using 
 * the PAPI_FP_INS event. This is intended to measure instruction rate through the 
 * floating point pipe with no massaging.
 * PAPI_flops return information related to theoretical floating point operations
 * rather than simple instructions. It uses the PAPI_FP_OPS event which attempts to 
 * 'correctly' account for, e.g., FMA undercounts and FP Store overcounts, etc.
 * PAPI_ipc returns information on the instruction rate using the PAPI_TOT_INS event.
 */
int PAPI_flips(float *rtime, float *ptime, long long * flpins, float *mflips)
{
   HighLevelInfo *state = NULL;
   int retval;

   if ((retval = _internal_check_state(&state)) != PAPI_OK)
      return (retval);

   if ((retval =
        _hl_rate_calls(rtime, ptime, flpins, mflips, (unsigned int)PAPI_FP_INS, state)) != PAPI_OK)
      return (retval);

   return (PAPI_OK);
}

int PAPI_flops(float *rtime, float *ptime, long long * flpops, float *mflops)
{
   HighLevelInfo *state = NULL;
   int retval;

   if ((retval = _internal_check_state(&state)) != PAPI_OK)
      return (retval);

   if ((retval =
        _hl_rate_calls(rtime, ptime, flpops, mflops, (unsigned int)PAPI_FP_OPS, state)) != PAPI_OK)
      return (retval);

   return (PAPI_OK);
}

int PAPI_ipc(float *rtime, float *ptime, long long * ins, float *ipc)
{
   HighLevelInfo *state = NULL;
   int retval;

   if ((retval = _internal_check_state(&state)) != PAPI_OK)
      return (retval);

   return _hl_rate_calls(rtime, ptime, ins, ipc, (unsigned int)PAPI_TOT_INS, state);
}

int _hl_rate_calls(float *real_time, float *proc_time, long long * ins, float *rate,
                   unsigned int EVENT, HighLevelInfo * state)
{
   long long values[2] = { 0, 0 };
   int retval = 0;
   int level = 0;


   if (EVENT == (unsigned int)PAPI_FP_INS)
      level = HL_FLIPS;
   else if (EVENT == (unsigned int)PAPI_TOT_INS)
      level = HL_IPC;
   else if (EVENT == (unsigned int)PAPI_FP_OPS)
      level = HL_FLOPS;

   if (state->running != 0 && state->running != level)
      return (PAPI_EINVAL);

   if (state->running == 0) {
      if (PAPI_query_event((int)EVENT) != PAPI_OK)
         return (PAPI_ENOEVNT);

      if ((retval = PAPI_add_event(state->EventSet, (int)EVENT)) != PAPI_OK) {
         _internal_cleanup_hl_info(state);
         PAPI_cleanup_eventset(state->EventSet);
         return (retval);
      }

      if (PAPI_query_event((int)PAPI_TOT_CYC) != PAPI_OK)
         return (PAPI_ENOEVNT);

      if ((retval = PAPI_add_event(state->EventSet, (int)PAPI_TOT_CYC)) != PAPI_OK) {
         _internal_cleanup_hl_info(state);
         PAPI_cleanup_eventset(state->EventSet);
         return (retval);
      }

      state->initial_time = PAPI_get_real_usec();
      if ((retval = PAPI_start(state->EventSet)) != PAPI_OK)
         return (retval);
      state->running = (short)level;
   } else {
      if ((retval = PAPI_stop(state->EventSet, values)) != PAPI_OK)
         return (retval);
      /* Use Multiplication because it is much faster */
      *real_time = (float)((double)(PAPI_get_real_usec() - state->initial_time) * .000001);
      *proc_time = (float)((double)values[1]*.000001/((_papi_hwi_system_info.hw_info.mhz==0)?1:_papi_hwi_system_info.hw_info.mhz));
      if (*proc_time > 0)
	*rate = (float)((float)values[0]*(EVENT==(unsigned int)PAPI_TOT_INS?1:_papi_hwi_system_info.hw_info.mhz)/(float)(values[1]==0?1:values[1]));
      state->total_proc_time += *proc_time;
      state->total_ins += (float)values[0];
      *proc_time = state->total_proc_time;
      *ins = (long long)state->total_ins;
      if ((retval = PAPI_start(state->EventSet)) != PAPI_OK) {
         state->running = 0;
         return (retval);
      }
   }
   return PAPI_OK;
}

/*
 * How many hardware counters does this platform support?
 */
int PAPI_num_counters(void)
{
   int retval;
   HighLevelInfo *tmp = NULL;

   /* Make sure the Library is initialized, etc... */
   if ((retval = _internal_check_state(&tmp)) != PAPI_OK)
      return (retval);

   return (PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
}

/*========================================================================*/
/* int PAPI_start_counters(int *events, int array_len)                    */
/* from draft standard:                                                   */
/* Start counting the events named in the events array.                   */
/* If the events array is already running, then you must call             */
/* PAPI_stop_counters to stop the events before you call this function    */
/* again. It is the user's                                                */
/* responsibility to choose events that can be counted simultaneously     */
/* by reading the vendor's documentation. The length of this array        */
/* should be no longer than PAPI_num_counters()                           */
/* This will fail if flips or ipc is already running			  */
/*========================================================================*/

int PAPI_start_counters(int *events, int array_len)
{
   int i, retval;
   HighLevelInfo *state = NULL;

   if ((retval = _internal_check_state(&state)) != PAPI_OK)
      return (retval);

   if(state->running != 0)
	return(PAPI_EINVAL);

   /* load events to the new EventSet */
   for (i = 0; i < array_len; i++) {
      retval = PAPI_add_event(state->EventSet, events[i]);
      if (retval == PAPI_EISRUN)
         return (retval);

      if (retval) {
         /* remove any prior events that may have been added 
          * and cleanup the high level information
          */
         _internal_cleanup_hl_info(state);
         PAPI_cleanup_eventset(state->EventSet);
         return (retval);
      }
   }
   /* start the EventSet */
   if ((retval = _internal_start_hl_counters(state)) == PAPI_OK) {
      state->running = HL_START_COUNTERS;
      state->num_evts = (short)array_len;
   }
   return (retval);
}

/*========================================================================*/
/* int PAPI_read_counters(long long *values, int array_len)      */
/*                                                                        */
/* Read the running counters into the values array. This call             */
/* implicitly initializes the internal counters to zero and allows        */
/* them continue to run upon return.                                      */
/*========================================================================*/

int _internal_hl_read_cnts(long long * values, int array_len, int flag)
{
   int retval;
   HighLevelInfo *state = NULL;

   if ((retval = _internal_check_state(&state)) != PAPI_OK)
      return (retval);

   if (state->running != HL_START_COUNTERS || array_len < state->num_evts)
      return (PAPI_EINVAL);

   if (flag == PAPI_HL_ACCUM)
      return (PAPI_accum(state->EventSet, values));
   else if (flag == PAPI_HL_READ) {
      if ((retval = PAPI_read(state->EventSet, values)) != PAPI_OK)
         return (retval);
      return (PAPI_reset(state->EventSet));
   }

   /* Invalid flag passed in */
   return (PAPI_EINVAL);
}

int PAPI_read_counters(long long * values, int array_len)
{
   return (_internal_hl_read_cnts(values, array_len, PAPI_HL_READ));
}

int PAPI_accum_counters(long long * values, int array_len)
{
   return (_internal_hl_read_cnts(values, array_len, PAPI_HL_ACCUM));
}


/*========================================================================*/
/* int PAPI_stop_counters(long long *values, int array_len)               */
/*                                                                        */
/* Stop the running counters and copy the counts into the values array.   */
/* Reset the counters to 0.                                               */
/*========================================================================*/

int PAPI_stop_counters(long long * values, int array_len)
{
   int retval;
   HighLevelInfo *state = NULL;

   if ((retval = _internal_check_state(&state)) != PAPI_OK)
      return (retval);

   if (state->running == 0)
      return (PAPI_ENOTRUN);

   if (state->running == HL_FLOPS || state->running == HL_FLIPS || state->running == HL_IPC) {
      long long tmp_values[2];
      retval = PAPI_stop(state->EventSet, tmp_values);
   } 
   else if(state->running != HL_START_COUNTERS || array_len < state->num_evts)
      return (PAPI_EINVAL);
   else
      retval = PAPI_stop(state->EventSet, values);

   if (retval==PAPI_OK) {
      _internal_cleanup_hl_info(state);
      PAPI_cleanup_eventset(state->EventSet);
   }
   APIDBG("PAPI_stop_counters returns %d\n", retval);
   return retval;
}

void _papi_hwi_shutdown_highlevel(){
   HighLevelInfo *state = NULL;

   if ( PAPI_get_thr_specific(PAPI_HIGH_LEVEL_TLS, (void *) &state)==PAPI_OK){
      if ( state ) papi_free(state);
   }
}
