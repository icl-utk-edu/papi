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

/*========================================================================*/
/* int PAPI_num_counters()                                                  */
/*                                                                        */ 
/* This function returns the optimal length of the values                 */
/* array used in the high level routines.                                 */ 
/*========================================================================*/

int PAPI_num_counters(void) 
{
  return(PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL));
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
  int EventSet;
  int i,retval;
  int MAX_COUNTERS;

  if (!initialized)
    {
      retval = PAPI_library_init(PAPI_VER_CURRENT);
      if (retval != PAPI_VER_CURRENT)
	return(retval);
      initialized = 1;
    }

  MAX_COUNTERS = PAPI_num_counters();
  if (MAX_COUNTERS < 1)
    return(PAPI_ENOCNTR);
  if (array_len > MAX_COUNTERS)
    return(PAPI_EINVAL);
  if (PAPI_EVENTSET_INUSE != PAPI_NULL) 
    return(PAPI_EISRUN);

  /*initialize value for EventSet integer*/

  EventSet = PAPI_NULL;

  /* load events to the new EventSet */   

  for (i=0;i<array_len;i++) 
    {
      retval = PAPI_add_event(&EventSet,events[i]);
      if (retval)
	return(retval);
    }

  /* start the EventSet*/

  retval = PAPI_start(EventSet);
  if (retval) 
    return(retval);

  PAPI_EVENTSET_INUSE = EventSet;

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
  int retval, MAX_COUNTERS;

  if (!initialized)
    {
      retval = PAPI_library_init(PAPI_VER_CURRENT);
      if (retval != PAPI_VER_CURRENT)
	return(retval);
      initialized = 1;
    }

  MAX_COUNTERS = PAPI_num_counters();

  if (MAX_COUNTERS < 1)
    return(PAPI_ENOCNTR);
  if (array_len > MAX_COUNTERS)
    return(PAPI_EINVAL);
  if (PAPI_EVENTSET_INUSE == PAPI_NULL)
    return(PAPI_ENOTRUN);

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
  int retval, MAX_COUNTERS;

  if (!initialized)
    {
      retval = PAPI_library_init(PAPI_VER_CURRENT);
      if (retval != PAPI_VER_CURRENT)
	return(retval);
      initialized = 1;
    }

  MAX_COUNTERS = PAPI_num_counters();

  if (MAX_COUNTERS < 1)
    return(PAPI_ENOCNTR);
  if (array_len > MAX_COUNTERS)
    return(PAPI_EINVAL);
  if (PAPI_EVENTSET_INUSE == PAPI_NULL)
    return(PAPI_ENOTRUN);

  retval = PAPI_stop(PAPI_EVENTSET_INUSE, values);
  if (retval) 
    return(retval);

  return(PAPI_cleanup_eventset(&PAPI_EVENTSET_INUSE));
} 




