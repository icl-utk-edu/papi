/* file: papi_hl.c */

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

/* do NOT set this to PAPI_NULL=0 */
static int PAPI_EVENTSET_INUSE=PAPI_EINVAL;

/*========================================================================*/
/* int PAPI_num_events()                                                  */
/*                                                                        */ 
/* This function returns the optimal length of the values                 */
/* array used in the high level routines.                                 */ 
/*========================================================================*/
int PAPI_num_events(void) {
return(PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL));
}


/*  */
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
int PAPI_start_counters(int *events, int array_len) {

int EventSet;
int i,r;

int MAX_COUNTERS;

MAX_COUNTERS=PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL);


if(PAPI_EVENTSET_INUSE > PAPI_NULL ) {
PAPI_perror(PAPI_EINVAL,"attempt to start new event set while prev one running",0);
return(PAPI_EINVAL);
}/* end if*/


if(array_len>MAX_COUNTERS) {
PAPI_perror(PAPI_EINVAL,"PAPI_start_counters failed because array_len > MAX_COUNTERS",0);
return(PAPI_EINVAL);
}

/*initialize value for EventSet integer*/

EventSet=PAPI_EINVAL;

/* load *events to the new EventSet */   

for(i=0;i<array_len;i++) {
r=PAPI_add_event(&EventSet,events[i]);
if( r < PAPI_OK ) {
  fprintf(stderr,
   "PAPI warning: PAPI_start_counters failed to load events[%d]",i);
   }
}



if ( EventSet == PAPI_EINVAL ) {
    PAPI_perror(PAPI_EINVAL,
     "PAPI_start_counters failed to create the EventSet",0);
     return(PAPI_EINVAL);
}

/* start the EventSet*/


r=PAPI_start(EventSet);
if( r < PAPI_OK ) {
  fprintf(stderr,"PAPI warning: PAPI_start failed ");
  return(PAPI_EINVAL);
}


PAPI_EVENTSET_INUSE=EventSet;

return(PAPI_OK);
}
/*  */
/*========================================================================*/
/* int PAPI_read_counters(unsigned long long *values, int array_len)      */
/*                                                                        */ 
/* Read the running counters into the values array. This call             */
/* implicitly initializes the internal counters to zero and allows        */
/* them continue to run upon return.                                      */
/*========================================================================*/
int PAPI_read_counters(unsigned long long *values, int array_len) {

if(PAPI_EVENTSET_INUSE != PAPI_EINVAL) {
PAPI_read(PAPI_EVENTSET_INUSE,values);/* cp cntrs to values*/
return(PAPI_OK);
}

else {
PAPI_perror(PAPI_EINVAL,"PAPI_read_counters failed to locate active EventSet",0);
}
return(PAPI_EINVAL);

}


/*========================================================================*/
/* int PAPI_stop_counters(unsigned long long *values, int array_len)      */
/*                                                                        */ 
/* Stop the running counters and copy the counts into the values array.   */ 
/* Reset the counters to 0.                                               */       
/*========================================================================*/
int PAPI_stop_counters(unsigned long long *values, int array_len) {

if( PAPI_EVENTSET_INUSE != PAPI_EINVAL ) {
PAPI_stop(PAPI_EVENTSET_INUSE,values);/* cp cntrs to values*/
return(PAPI_OK);
}

else {
PAPI_perror(PAPI_EINVAL,"PAPI_stop_counters failed to locate active EventSet",0);
}
return(PAPI_EINVAL);

} 



/* end papi_hl.c*/
