/* 
* File:    papi_hl.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* This file contains the 'high level' interface to PAPI. 
   BASIC is a high level language. ;-) */

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>
#include <strings.h>
#include <errno.h>

#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

/* high level papi functions*/

static int PAPI_EVENTSET_INUSE = PAPI_NULL;
static int initialized = 0;
static int MAX_COUNTERS = 0;

int PAPI_num_counters(void) 
{
  int retval;

  if (!initialized)
    {
      retval = PAPI_library_init(PAPI_VER_CURRENT);
      if (retval != PAPI_VER_CURRENT)
	return(retval);

      retval = PAPI_create_eventset(&PAPI_EVENTSET_INUSE);
      if (retval)
	return(retval);
      
      MAX_COUNTERS = PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL);
      
      initialized = 1;
    }

  return(MAX_COUNTERS);
}

/*========================================================================*/
/* int PAPI_start_counters(int *events, int array_len)                    */
/* from draft standard:                                                   */ 
/* Start counting the events named in the events array. This function     */
/* implicitly stops and initializes any counters running as the result    */
/* of a previous call to PAPI_start_counters(). It is the user's          */
/* responsibility to choose events that can be counted simultaneously     */
/* by reading the vendor's documentation. The length of this array        */
/* should be no longer than PAPI_MAX_EVNTS.                               */ 
/*========================================================================*/

int PAPI_start_counters(int *events, int array_len) 
{
  int i,retval;

  if (!initialized)
    {
      PAPI_num_counters();
      if (MAX_COUNTERS < 1)
	return(PAPI_ENOCNTR);
    }

  if (array_len > MAX_COUNTERS)
    return(PAPI_EINVAL);

  /* load events to the new EventSet */   

  for (i=0;i<array_len;i++) 
    {
      retval = PAPI_query_event(events[i]);
      if (retval)
	return(retval);

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

int PAPI_read_counters(long long *values, int array_len) 
{
  int retval;

  if (!initialized)
    return(PAPI_EINVAL);

  if (array_len > MAX_COUNTERS)
    return(PAPI_EINVAL);

  retval = PAPI_read(PAPI_EVENTSET_INUSE,values);
  if (retval)
    return(retval);

  return(PAPI_reset(PAPI_EVENTSET_INUSE));
}

/*========================================================================*/
/* int PAPI_stop_counters(long long *values, int array_len)               */
/*                                                                        */ 
/* Stop the running counters and copy the counts into the values array.   */ 
/* Reset the counters to 0.                                               */       
/*========================================================================*/

int PAPI_stop_counters(long long *values, int array_len) 
{
  int retval;

  if (!initialized)
    return(PAPI_EINVAL);

  MAX_COUNTERS = PAPI_num_counters();

  if (array_len > MAX_COUNTERS)
    return(PAPI_EINVAL);

  retval = PAPI_stop(PAPI_EVENTSET_INUSE, values);
  if (retval) 
    return(retval);

  return(PAPI_cleanup_eventset(&PAPI_EVENTSET_INUSE));
} 




