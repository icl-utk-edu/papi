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

static int PAPI_EVENTSET_INUSE=PAPI_EINVAL;/* initialize global constant */


int PAPI_num_events(void) 
{
  return(PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL));
}



int PAPI_start_counters(int *events, int array_len) 
{

  int EventSet;
  int i,r;
  char *eventName;




  if(PAPI_EVENTSET_INUSE != PAPI_EINVAL ) 
   {
     PAPI_perror(PAPI_EINVAL,"attempt to start new event set while prev one running",0);
     return(PAPI_EINVAL);
   }


/* see if there are enough counters to do requested events*/
  if( array_len > PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL) ) 
  {
    PAPI_perror(PAPI_EINVAL,"err: PAPI_start_counters request for too many hw counters",0);
    return(PAPI_EINVAL);
  }



/* see if all the values in the events array are papi standard values*/
  eventName=(char *)malloc(32*sizeof(char));
  for(i=0;i<array_len;i++) 
   {  
     *eventName=NULL;
     r=PAPI_describe_event(eventName,events[i],NULL);
     if(r==0) 
         {
          printf("\n events[%d] not a papi standard event\n",i);
	  exit(0);
         }
   }



/* see if all the values in events array are supported on local platform */ 
/* 
  ---------------------------------------------------------------------
  This code to be added once all of the query interfaces are completed.
  for(i=0;i<array_len;i++) 
   { 
    r=PAPI_query_event(events[i]);
    if(r<PAPI_OK) 
    	{
	  printf("\n events[%d] not supported on this platform\n",i);
          exit(0);
	}
   }
  ---------------------------------------------------------------------
*/


  /*initialize value for EventSet integer*/

  EventSet=PAPI_EINVAL;

  /* load *events to the new EventSet */   

  for(i=0;i<array_len;i++) 
  {
    r=PAPI_add_event(&EventSet,events[i]);
    if( r < PAPI_OK ) 
      {
      fprintf(stderr,
	      "PAPI warning: PAPI_start_counters failed to load events[%d]",i);
      }
  }



  if ( EventSet == PAPI_EINVAL ) 
  {
    PAPI_perror(PAPI_EINVAL,
		"PAPI_start_counters failed to create the EventSet",0);
    return(PAPI_EINVAL);
  }

  /* start the EventSet*/


  r=PAPI_start(EventSet);
  if( r < PAPI_OK ) 
  {
    fprintf(stderr,"PAPI warning: PAPI_start failed ");
    return(PAPI_EINVAL);
  }


  PAPI_EVENTSET_INUSE=EventSet;/* reset global constant*/

  return(PAPI_OK);
}


int PAPI_read_counters(unsigned long long *values, int array_len) {

  if(PAPI_EVENTSET_INUSE != PAPI_EINVAL) 
   {
     PAPI_read(PAPI_EVENTSET_INUSE,values);/* cp cntrs to values*/
     return(PAPI_OK);
   }

  else 
   {
     PAPI_perror(PAPI_EINVAL,"PAPI_read_counters failed to locate active EventSet",0);
   }

  return(PAPI_EINVAL);

}



int PAPI_stop_counters(unsigned long long *values, int array_len) {

  if( PAPI_EVENTSET_INUSE != PAPI_EINVAL ) 
   {
     PAPI_stop(PAPI_EVENTSET_INUSE,values);/* cp cntrs to values*/
	PAPI_EVENTSET_INUSE = PAPI_EINVAL; /*reset global constant*/
     return(PAPI_OK);
   }

  else 
   {
     PAPI_perror(PAPI_EINVAL,"PAPI_stop_counters failed to locate active EventSet",0);
   }

  return(PAPI_EINVAL);

} 


