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

static int PAPI_EVENTSET_INUSE;

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
int status;
int i,r;
int MAX_COUNTERS;
unsigned long long *ct;

MAX_COUNTERS=PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL);
ct=(unsigned long long *)malloc(MAX_COUNTERS*sizeof(unsigned long long));

/* turn off the running EventSet */
i=1;
while(i) {
	r=PAPI_state(i,&status);
	if(r==PAPI_EMISC)break;/* no running counters found*/
        if(r==PAPI_OK) {
        if(status==PAPI_RUNNING) {
            /* cp hwd_ctrs to ct, turn off EventSet, 
               set ctrs to zero, status to PAPI_STOPPED*/
	    PAPI_stop(i,ct);
	    /* cp ct to hwd_ctrs so preserved*/
            PAPI_write(i,ct);
            break;
            } 
	}
        i++; 
}

if(array_len>MAX_COUNTERS) {
PAPI_perror(PAPI_EINVAL,"PAPI_start_counters failed because array_len > MAX_COUNTERS",0);
return(PAPI_EINVAL);
}

/* load *events to the new EventSet */   
EventSet=PAPI_NULL;
for(i=0;i<array_len;i++) {
r=PAPI_add_event(&EventSet,events[i]);
if(r<PAPI_OK) {
  fprintf(stderr,"PAPI warning: PAPI_start_counters failed to load events[%d] (%x) ", i,events[i]);
   }
}

if (EventSet==PAPI_NULL) {
PAPI_perror(PAPI_EINVAL,"PAPI_start_counters failed to create the EventSet",0);
return(PAPI_EINVAL);
}

PAPI_EVENTSET_INUSE=EventSet;

/* start the EventSet*/
r=PAPI_start(PAPI_EVENTSET_INUSE);
if(r<PAPI_OK) {
PAPI_perror(PAPI_EINVAL,"PAPI_start_counters failed to start the EventSet",0);
return(PAPI_EINVAL);
}
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

int status;
int i,r;

/* locate the EventSet with running counters and read it*/

i=1;
while(i) {
	r=PAPI_state(i,&status);
	if(r==PAPI_EMISC)break;/* no running counters found*/
        if(r==PAPI_OK) {
          if(status==PAPI_RUNNING) {
		PAPI_EVENTSET_INUSE=i;
		PAPI_read(PAPI_EVENTSET_INUSE,values);/* cp cntrs to values*/
                PAPI_reset(PAPI_EVENTSET_INUSE); /* reset cntrs to zero*/
		printf("\n PAPI_read_counters:");
		printf("\n ct[0]=%lld ct[2]=%lld",values[0],values[2]);
		PAPI_start(PAPI_EVENTSET_INUSE);
		return(PAPI_OK);
		}
          }
	}	

PAPI_perror(PAPI_EINVAL,"PAPI_read_counters failed to locate active EventSet",0);
return(PAPI_EINVAL);

}


/*========================================================================*/
/* int PAPI_stop_counters(unsigned long long *values, int array_len)      */
/*                                                                        */ 
/* Stop the running counters and copy the counts into the values array.   */ 
/* Reset the counters to 0.                                               */       
/*========================================================================*/
int PAPI_stop_counters(unsigned long long *values, int array_len) {

int status;
int i,r;

/* locate the EventSet with running counters and stop it*/
i=1;
while(i) {
	r=PAPI_state(i,&status);
	if(r==PAPI_EMISC)break;/* no running counters found*/
        if(r==PAPI_OK) {
          if(status==PAPI_RUNNING) {
		PAPI_EVENTSET_INUSE=i;
		PAPI_stop(PAPI_EVENTSET_INUSE,values);/* cp cntrs to values*/
                PAPI_reset(PAPI_EVENTSET_INUSE);/* reset cntrs to zero*/
                return(PAPI_OK);
		}
          }
	}	


PAPI_perror(PAPI_EINVAL,"PAPI_stop_counters failed to locate active EventSet",0);
return(PAPI_EINVAL);

} 



/* end papi_hl.c*/
