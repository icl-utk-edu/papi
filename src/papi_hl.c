/* 
* File:    papi_hl.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:	   Kevin London
* 	       london@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* This file contains the 'high level' interface to PAPI. 
   BASIC is a high level language. ;-) */

#ifndef _WIN32
  #include SUBSTRATE
#else
  #include "win32.h"
#endif

/* high level papi functions*/

static int PAPI_EVENTSET_INUSE = PAPI_NULL;
static int initialized = 0;
static int hl_max_counters = 0;

/* CHANGE LOG:
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

int PAPI_flops(float *real_time, float *proc_time, long_long *flpins, float *mflops)
{
   static float total_proc_time=0.0; 
   static int EventSet = PAPI_NULL;
   static float mhz;
   static long_long start_us = 0;
   static long_long total_flpins = 0;
   const PAPI_hw_info_t *hwinfo = NULL;
   long_long values[2] = {0,0};
   char buf[500];
   int retval;

   if ( !initialized ) {
	mhz = 0.0;
	*mflops = 0.0;
 	*real_time = 0.0;
 	*proc_time = 0.0;
	*flpins = 0;
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
	   return(retval);
	if ( (hwinfo = PAPI_get_hardware_info()) == NULL ) {
	   printf("Error getting hw_info\n");
	   return -1;
        } 
	mhz = hwinfo->mhz;
	PAPI_create_eventset( &EventSet );
	retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
	PAPI_perror( retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     return retval;
	}
	retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC);
	PAPI_perror(retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     return retval;
	}
	initialized = 1;
	start_us = PAPI_get_real_usec();
	retval = PAPI_start(EventSet);
	PAPI_perror(retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     return retval;
	}
   }
   else {
	retval = PAPI_stop( EventSet, values );
	/* If fp instuction count is negative, re-initialize */
	if ( *flpins < 0 ) {
		total_flpins = 0;
		total_proc_time = 0.0;
		*mflops = 0.0;
		*real_time = 0.0;
		*proc_time = 0.0;
		*flpins = 0;
		start_us = PAPI_get_real_usec();
	} else {
		*real_time = (float)((PAPI_get_real_usec()-start_us)/1000000.0);
		PAPI_perror( retval, buf, 500);
		if ( retval < PAPI_OK ) {
			 PAPI_shutdown();
			 initialized = 0;
			 return retval;
		}

		*proc_time = (float)(values[1]/(mhz*1000000.0));
		*mflops = (float)((values[0])/(*proc_time*1000000.0));
		total_proc_time += *proc_time;
		total_flpins += values[0];
		*proc_time = total_proc_time;
		*flpins = total_flpins;
	}
	retval = PAPI_start(EventSet);
	PAPI_perror(retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     return retval;
	}
   }
   return PAPI_OK;
}

int PAPI_num_counters(void) 
{
  int retval;

  if (!initialized)
    {
      retval = PAPI_library_init(PAPI_VER_CURRENT);
      if (retval != PAPI_VER_CURRENT)
	return(PAPI_EINVAL);

      retval = PAPI_create_eventset(&PAPI_EVENTSET_INUSE);
      if (retval)
	return(retval);
      
      hl_max_counters = PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL);
      
      initialized = 1;
    }

  return(hl_max_counters);
}

/*========================================================================*/
/* int PAPI_start_counters(int *events, int array_len)                    */
/* from draft standard:                                                   */ 
/* Start counting the events named in the events array. This function     */
/* implicitly stops and initializes any counters running as the result    */
/* of a previous call to PAPI_start_counters(). It is the user's          */
/* responsibility to choose events that can be counted simultaneously     */
/* by reading the vendor's documentation. The length of this array        */
/* should be no longer than PAPI_num_counters()                           */ 
/*========================================================================*/

int PAPI_start_counters(int *events, int array_len) 
{
  int i,retval;

  if (!initialized)
    {
      PAPI_num_counters();
      if (hl_max_counters < 1)
	return(PAPI_ENOCNTR);
    }

  if (array_len > hl_max_counters)
    return(PAPI_EINVAL);

  /* load events to the new EventSet */   

  for (i=0;i<array_len;i++) 
    {
      /* retval = PAPI_query_event(events[i]);
      if (retval)
	return(retval); */

      retval = PAPI_add_event(&PAPI_EVENTSET_INUSE,events[i]);
      if (retval)
	return(retval);
    }

  /* start the EventSet*/

  retval = PAPI_start(PAPI_EVENTSET_INUSE);
  if (retval) 
    return(retval);

  return(PAPI_OK);
}

/*========================================================================*/
/* int PAPI_read_counters(long long *values, int array_len)      */
/*                                                                        */ 
/* Read the running counters into the values array. This call             */
/* implicitly initializes the internal counters to zero and allows        */
/* them continue to run upon return.                                      */
/*========================================================================*/

int PAPI_read_counters(long_long *values, int array_len) 
{
  int retval;

  if (!initialized)
    return(PAPI_EINVAL);

  if (array_len > hl_max_counters)
    return(PAPI_EINVAL);

  retval = PAPI_read(PAPI_EVENTSET_INUSE,values);
  if (retval)
    return(retval);

  return(PAPI_reset(PAPI_EVENTSET_INUSE));
}

int PAPI_accum_counters(long_long *values, int array_len) 
{
  int retval;

  if (!initialized)
    return(PAPI_EINVAL);

  if (array_len > hl_max_counters)
    return(PAPI_EINVAL);

  retval = PAPI_accum(PAPI_EVENTSET_INUSE,values);

  return(retval);
  /* PAPI_accum implies a PAPI_reset so no explicit PAPI_reset needed */
}

/*========================================================================*/
/* int PAPI_stop_counters(long long *values, int array_len)               */
/*                                                                        */ 
/* Stop the running counters and copy the counts into the values array.   */ 
/* Reset the counters to 0.                                               */       
/*========================================================================*/

int PAPI_stop_counters(long_long *values, int array_len) 
{
  int retval;

  if (!initialized)
    return(PAPI_EINVAL);

  hl_max_counters = PAPI_num_counters();

  if (array_len > hl_max_counters)
    return(PAPI_EINVAL);

  retval = PAPI_stop(PAPI_EVENTSET_INUSE, values);
  if (retval) 
    return(retval);

  return(PAPI_cleanup_eventset(&PAPI_EVENTSET_INUSE));
}

