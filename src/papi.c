/* $Id$ */

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

/* Static prototypes */

static int expand_dynamic_array();
static EventSetInfo *allocate_EventSet(void);
static int add_EventSet(EventSetInfo *);
static int remove_EventSet(int);
static void free_EventSet(EventSetInfo *);
static int handle_error(int, char *);
static char *get_error_string(int);
static EventSetInfo *lookup_EventSet(int eventset);
static int lookup_EventCodeIndex(EventSetInfo *ESI,int Event);
static int nullify_event(EventSetInfo *ESI,int Event);

/* Global variables */
/* These will eventually be encapsulated into per thread structures. */ 

/* Our integer to EventSetInfo * mapping */

DynamicArray PAPI_EVENTSET_MAP = { 0, };    

/* Behavior of handle_error(). Changed to the default behavior of PAPI_QUIET in PAPI_init
after initialization is successful. */

int          PAPI_ERR_LEVEL = PAPI_VERB_ESTOP; 

/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */    
/* static void PAPI_init(DynamicArray *EM, int ERROR_LEVEL_CHOICE );      */
/*                                                                        */
/* This function must be called at the beginning of the program.          */
/* This function performs all initializations to set up PAPI environment. */
/* The user selects the level of error handling here.                     */
/* Failure of this function should shutdown the PAPI tool.                */
/*                                                                        */
/* Initialize EM.                                                         */
/* Set pointer to GLOBAL variable PAPI_EVENTSET_MAP.                      */
/* Since PAPI_EVENTSET_MAP is declared at the top of the program          */
/* no malloc for EM is needed.                                            */
/* But the pointer EM->dataSlotArray must be malloced here.               */
/*                                                                        */
/* Initialize PAPI_ERR_LEVEL.                                             */
/* The user selects error handling with ERROR_LEVEL_CHOICE.               */  
/* ERROR_LEVEL_CHOICE may have one of two values:                         */
/*   a. PAPI_VERB_ECONT [print error message, then continue processing ]  */
/*   b. PAPI_VERB_ESTOP [print error message, then shutdown ]             */
/*========================================================================*/

/* This function returns the number of counters that a read(ESI->machdep)
   returns */

static int num_counters(EventSetInfo *ESI)
{
  return(_papi_system_info.num_cntrs);
}

/* This function returns true and the memory allocated for counters */

static long long *get_space_for_counters(EventSetInfo *ESI)
{
  int num;

  num = num_counters(ESI);
  return((long long *)malloc(num*sizeof(long long)));
}

static void initialize(void)
{
   memset(&PAPI_EVENTSET_MAP,0x00,sizeof(PAPI_EVENTSET_MAP));

   /* initialize values in PAPI_EVENTSET_MAP */ 

   PAPI_EVENTSET_MAP.dataSlotArray=(EventSetInfo **)malloc(PAPI_INIT_SLOTS*sizeof(void *));
   if(!PAPI_EVENTSET_MAP.dataSlotArray) 
     handle_error(PAPI_ENOMEM,"Initialization failed.");

   memset(&PAPI_EVENTSET_MAP.dataSlotArray,0x00, PAPI_INIT_SLOTS*sizeof(void *));

   PAPI_EVENTSET_MAP.totalSlots = PAPI_INIT_SLOTS;
   PAPI_EVENTSET_MAP.availSlots = PAPI_INIT_SLOTS - 1;
   PAPI_EVENTSET_MAP.lowestEmptySlot = 1;

   PAPI_ERR_LEVEL = PAPI_QUIET;
}

/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */    
/* static void PAPI_shutdown (void);                                      */
/*                                                                        */
/* This function provides a graceful exit to the PAPI tool.               */
/* a. All memory associated with the PAPI tool is freed.                  */
/*  b. a shutdown message is written to stderr                            */ 
/*                                                                        */
/*========================================================================*/

void PAPI_shutdown(void) {

    int i;
    /* close all memory pointed to by xEM */
    /* this code under construction       */
    /* note: do we need to make a special case for PAPI_EVENTSET_MAP.dataSlotArray[0]?*/


    /* free all the EventInfo Structures in the PAPI_EVENTSET_MAP.dataSlotArray*/
    for(i=0;i<PAPI_EVENTSET_MAP.totalSlots;i++) {
	if(PAPI_EVENTSET_MAP.dataSlotArray[i]) {
 	  free_EventSet(PAPI_EVENTSET_MAP.dataSlotArray[i]); 
	  }/* end if */
	}/* end for */ 
		 
	free(PAPI_EVENTSET_MAP.dataSlotArray);


    /* shutdown message */
    fprintf(stderr,"\n\n PAPI SHUTDOWN. \n\n");

    return;
}
/*========================================================================*/
/*end function: PAPI_shutdown                                             */
/*========================================================================*/
/*========================================================================*/


/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* static int handle_error(int PAPI_errorCode, char *errorMessage);       */
/*                                                                        */
/* The function PAPI_error (int, char *) provides error handling for      */
/* both the user and the internal functions.                              */
/*                                                                        */
/* A one line or multiple line error message is printed to stderr.        */ 
/*                                                                        */
/* The first line of the error message will be the character string       */
/* from errStr[N] where N is the absolute value of the PAPI_errorCode     */
/* passed to the function.                                                */  
/*                                                                        */
/* If *errorMessage is set to NULL, there is no further error message.    */
/* If *errorMessage points to a character string, this will be printed    */
/* to stderr.                                                             */
/*                                                                        */
/* The global value PAPI_ERR_LEVEL determines whether this error will     */
/* cause shutdown of the papi tool.                                       */
/*========================================================================*/

static char *papi_errStr[PAPI_NUM_ERRORS] = {
"No error",
"Invalid argument",
"Insufficient memory",
"A System/C library call failed",
"Substrate returned an error",
"Access to the counters was lost or interrupted",
"Internal error, please send mail to the developers",
"Hardware Event does not exist",
"Hardware Event exists, but cannot be counted due to counter resource limits",
"No Events or EventSets are currently counting",
"Unknown error code"
};

int PAPI_perror(int code, char *destination, int length)
{
  char *foo;

  foo = get_error_string(code);

  if ((destination) && (length >= 0))
    strncpy(destination,foo,length);
  else
    fprintf(stderr,"%s\n",foo);

  return(PAPI_OK);
}

static char *get_error_string(int code)
{
  code = - code;

  if ((code < 0) || (code >= PAPI_NUM_ERRORS))
    code = PAPI_EMISC;

  return(papi_errStr[code]);
}

/* Never call this with PAPI_OK */

static int handle_error(int PAPI_errorCode, char *errorMessage) 
{
  if (PAPI_ERR_LEVEL) 
    {
      /* print standard papi error message */
      fprintf(stderr, "%s", get_error_string(PAPI_errorCode));

      /* check for failed C library call*/
      if ( PAPI_errorCode==PAPI_ESYS ) perror(errorMessage);
      /* this not compile:
      if ( PAPI_errorCode==PAPI_ESYS ) fprintf(stderr,": %s",strerror(errno));
      */

      /* check for user supplied error message */
      if (errorMessage) fprintf(stderr, ": %s", errorMessage);

      fprintf(stderr,"\n");

      if (PAPI_ERR_LEVEL==PAPI_VERB_ESTOP) 
	PAPI_shutdown();
    }
  return(PAPI_errorCode);

}

/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* static int _papi_expandDA(DynamicArray *EM);                           */
/*                                                                        */
/* The function _papi_expandDA expands PAPI_EVENTSET_MAP.dataSlotArray    */
/* when the array has become full.                                        */
/* The function also resets:                                              */
/*       PAPI_EVENTSET_MAP.totalSlots                                     */
/*       PAPI_EVENTSET_MAP.availSlots                                     */
/*       PAPI_EVENTSET_MAP.lowestEmptySlot                                */
/* This enables the user to load as many events as needed.                */
/*                                                                        */
/* The DynamicArray data structure is defined in papi_internal.h.         */
/* typedef struct _dynamic_array {                                        */
/*	void   **dataSlotArray; ** ptr to array of ptrs to EventSets      */
/*	int    totalSlots;      ** number of slots in dataSlotArrays      */
/*	int    availSlots;      ** number of open slots in dataSlotArrays */
/*	int    lowestEmptySlot; ** index of lowest empty dataSlotArray    */
/* } DynamicArray;                                                        */
/*                                                                        */
/* Error handling should be done in the calling function.                 */  
/*========================================================================*/

static int expand_dynamic_array(DynamicArray *DA)
{
  int  prevTotal;	
  EventSetInfo **n;

  /*realloc existing PAPI_EVENTSET_MAP.dataSlotArray*/
    
  n = (EventSetInfo **)realloc(DA->dataSlotArray,DA->totalSlots*sizeof(EventSetInfo *));
  if (n==NULL) 
    return(handle_error(PAPI_ENOMEM,NULL));   

  /* Need to assign this value, what if realloc moved it? */
  DA->dataSlotArray = n;
       
  /* bookkeeping to accomodate successful realloc operation*/
  prevTotal           = DA->totalSlots; 
  DA->totalSlots     += prevTotal;
  DA->availSlots      = prevTotal;
  DA->lowestEmptySlot = prevTotal;

  return(PAPI_OK);
}

/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* int PAPI_state(int EventSetIndex, int *status)                         */
/*                                                                        */ 
/* This function reports the state of the entire EventSet designated by   */
/* EventSetIndex by setting the value of *status.                         */ 
/* If the call succeeds, then the value of *status is set to:             */
/*   a.  PAPI_RUNNING=1.                                                  */
/*   b.  PAPI_STOPPED=2.                                                  */
/*                                                                        */ 
/* The return value of this function tells if the call succeeded.         */ 
/*   a.  return(PAPI_OK) [success]                                        */
/*   b.  return(PAPI_EINVAL) [invalid argument]                           */
/*========================================================================*/

int PAPI_state(int EventSetIndex, int *status) 
{

 /* EventSetIndex is an integer, the index N of PAPI_EVENTSET_MAP.dataSlotArray[N] */ 
 /* Check if EventSetIndex is a valid value 4 different ways. */ 

 /* 1.   invalid array index less than zero */
 /*      if (EventSetIndex < 0 ) return( PAPI_EINVAL );*/

 /* 2.   invalid array index 0, reserved for internal use only */ 
 /*      if(PAPI_EVENTSET_MAP.dataSlotArray[EventSet]==0) return (PAPI_EINVAL); */

 /* 3.   invalid array index greater than highest possible value*/
 /*      if (EventSetIndex => PAPI_EVENTSET_MAP.totalSlots ) return (PAPI_EINVAL); */

 /* 4.   valid array index value, but not assigned to any event yet*/
 /*      if (PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]==NULL) return(PAPI_EINVAL); */


    /* combine all of the above ifs */

    if ((EventSetIndex < 1) || 
	(EventSetIndex >= PAPI_EVENTSET_MAP.totalSlots) ||
	(PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]==NULL)) 
           return (handle_error(PAPI_EINVAL,NULL));

    /* Good value for PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]-> state */

    *status = (PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex])->state;
     return(PAPI_OK);
}
/*========================================================================*/
/*end function: PAPI_state                                                */
/*========================================================================*/
/*========================================================================*/





/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* static EventSetInfo *papi_allocate_EventSet(void);                     */
/*                                                                        */
/* This function allocates space for one EventSetInfo structure and for   */
/* all of the pointers in this structure.  If any malloc in this function */
/* fails, all memory malloced to the point of failure is freed, and NULL  */
/* is returned.  Upon success, a pointer to the EventSetInfo data         */
/* structure is returned.                                                 */
/*========================================================================*/

static EventSetInfo *allocate_EventSet(void) 
{
  EventSetInfo *ESI;
  int counterArrayLength;
  
  counterArrayLength=_papi_system_info.num_cntrs;

  ESI=(EventSetInfo *)malloc(sizeof(EventSetInfo));
  if (ESI==NULL) 
    return(NULL); 
  memset(&ESI,0x00,sizeof(ESI));

  ESI->machdep=(void *)malloc(_papi_system_info.size_machdep);
  ESI->start = (long long *)malloc(counterArrayLength*sizeof(long long));
  ESI->stop = (long long *)malloc(counterArrayLength*sizeof(long long));
  ESI->latest = (long long *)malloc(counterArrayLength*sizeof(long long));
  ESI->EventCodeArray = (int *)malloc(counterArrayLength*sizeof(int));

  if ((ESI->machdep == NULL) || 
      (ESI->start == NULL) || 
      (ESI->stop == NULL) || 
      (ESI->latest == NULL) ||
      (ESI->EventCodeArray))
    {
      if (ESI->machdep) free(ESI->machdep);
      if (ESI->start) free(ESI->start);
      if (ESI->stop) free(ESI->stop);
      if (ESI->latest) free(ESI->latest);
      if (ESI->EventCodeArray) free(ESI->EventCodeArray);
      free(ESI);
      return(NULL);
    }
  memset(ESI->machdep,0x00,_papi_system_info.size_machdep);
  memset(ESI->start,0x00,counterArrayLength*sizeof(long long));
  memset(ESI->stop,0x00,counterArrayLength*sizeof(long long));
  memset(ESI->latest,0x00,counterArrayLength*sizeof(long long));
  memset(ESI->EventCodeArray,0x00,counterArrayLength*sizeof(int));

  ESI->state = PAPI_STOPPED; 

  return(ESI);
}

/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* static int free_EventSet();                                            */
/*                                                                        */ 
/* This function should free memory for one EventSetInfo structure.       */
/* The argument list consists of a pointer to the EventSetInfo            */
/* structure, *ESI.                                                       */
/* The calling function should check  for ESI==NULL.                      */
/*========================================================================*/

static void free_EventSet(EventSetInfo *ESI) 
{
  if (ESI->EventCodeArray) free(ESI->EventCodeArray);
  if (ESI->machdep)        free(ESI->machdep);
  if (ESI->start)          free(ESI->start);
  if (ESI->stop)           free(ESI->stop);
  if (ESI->latest)         free(ESI->latest);
  free(ESI);
}

/*========================================================================*/
/* This function should nullify memory pertaining to a single event in an
   EventSetInfo structure, by setting array elements to PAPI_NULL.
   No memory is freed.
*/

static int nullify_event(EventSetInfo *ESI,int Event) {

int k;

  /* determine index of target event k */
      k=lookup_EventCodeIndex(ESI,Event);
      if(k<PAPI_OK)
        return(handle_error(PAPI_EINVAL,NULL));


     ESI->EventCodeArray[k]=PAPI_NULL;
     ESI->start[k]         =PAPI_NULL;
     ESI->stop[k]          =PAPI_NULL;
     ESI->latest[k]        =PAPI_NULL;
    

     return(PAPI_OK); 
}

	
/*========================================================================*/
/* begin function:                                                        */
/* int PAPI_add_event(int *EventSet, int Event)                           */
/*                                                                        */
/* from the papi draft standard:
int PAPI_add_event(int *EventSet, int Event) 

          This function sets up a new EventSet or modifies an existing
     one. To create a new EventSet, EventSet must be set to
     PAPI_NULL. Separate EventSets containing events that require
     use of the same hardware may exist, but may not be started if a
     conflicting EventSet is running. Returns PAPI_ENOEVNT if
     Event cannot be counted on this platform. The addition of a
     conflicting event to an event set will return an error unless
     PAPI_SET_MPXRES has been set. Note: EventSet 0 may not be
     used; it has been reserved for internal use.
*/
int PAPI_add_event(int *EventSet, int Event) 
{
  int errorCode;
  int allocated_a_new_one = 0;
  EventSetInfo *ESI;

  if ( EventSet == NULL ) 
    return(handle_error(PAPI_EINVAL,NULL));

  if ( *EventSet == PAPI_NULL ) /* We need a new one */
    {
      ESI = allocate_EventSet();
      if (!ESI) 
	return(handle_error(PAPI_ENOMEM,NULL));
      allocated_a_new_one = 1;
    }
  else /* One exists already */
    {
      ESI = lookup_EventSet(*EventSet);
      if ( ESI == NULL ) 
	return(handle_error(PAPI_EINVAL,NULL));
    }

  /* Nor we have a valid ESI, try to add HW event */
  errorCode = _papi_hwd_add_event(ESI->machdep,Event);
  if (errorCode < PAPI_OK) 
    {
damnit:      /* If we've allocated one, we must free it */
      if (allocated_a_new_one)
	free_EventSet(ESI);
      return(handle_error(errorCode,NULL));
    }

  errorCode=add_EventSet(ESI);
  if (errorCode<PAPI_OK) 
    goto damnit;

  return(errorCode);
}

/*========================================================================*/
/* int PAPI_add_events(int *EventSet, int *Events, int number) 

   Same as PAPI_add_event() for a vector of events with length of vector
   equal to number. If one or more of Events cannot be counted on this 
   platform, then this call fails and PAPI_ENOEVNT is returned after _all_ 
   of the values in the Events array have been checked. In addition, the 
   invalid entries in the Events array are set to PAPI_NULL such that the 
   user can successfully reissue the call.
*/

int PAPI_add_events(int *EventSet, int *Events, int number) 
{
    int all_events_valid=1;
    int i;
    int errorCode;
    int allocated_a_new_one = 0;
    EventSetInfo *ESI;

    if ( EventSet == NULL )
       return(handle_error(PAPI_EINVAL,NULL));

  if ( *EventSet == PAPI_NULL ) /* We need a new one */
    {
      ESI = allocate_EventSet();
      if (!ESI)
        return(handle_error(PAPI_ENOMEM,NULL));
      allocated_a_new_one = 1;
    }
  else /* One exists already */
    {
      ESI = lookup_EventSet(*EventSet);
      if ( ESI == NULL )
        return(handle_error(PAPI_EINVAL,NULL));
    }

 /* Now we have a valid ESI, try to add HW events    */
 /* If any Events[i]==NULL, skip with no error       */
 /* If any Events[i] are invalid, all_events_valid=0 */  
 
    for( i=0; i<number; i++ ) {
    
    if( Events[i] != PAPI_NULL ) {/* only act if not PAPI_NULL */
       errorCode = _papi_hwd_add_event(ESI->machdep,Events[i]);
       if ( errorCode < PAPI_OK ) {
            all_events_valid=0; 
            Events[i]=PAPI_NULL;
	    }
       }
    }

 /* If (all_events_valid==1), try add_EventSet(ESI)         */
 /* errorCode==PAPI_OK     for no error simple              */
 /* errorCode==PAPI_OK_MPX for no error multiplex is active */ 

    if ( all_events_valid == 1 ) {
        errorCode=add_EventSet(ESI); 
	if (errorCode >= PAPI_OK ) /*add_EventSet was successful*/
 	  return(errorCode);
	}

    /* fall through to here if anything has gone wrong */
    /*    if (all_events_valid==0)                     */
    /*    if (errorCode<PAPI_OK)                       */
    /* get rid of allocations if they were new here    */
      
     if (allocated_a_new_one)
	   free_EventSet(ESI);
            
     return(errorCode);
}



/*========================================================================*/
/* low-level function:                                                    */
/* static int PAPI_add_pevent(int *EventSet, int code, void *inout )      */
/*                                                                        */
/*
    from the papi draft standard:
    int PAPI_add_event(int *EventSet, int Event)

   PAPI_add_pevent() may be implemented as a call to the
   PAPI_set_opt().

*/
/*========================================================================*/

int PAPI_add_pevent(int *EventSet, int code, void *inout)
{

EventSetInfo *ESI;
int errorCode;
void *extra; /*needed to call _papi_hwd_add_prog_event*/
int allocated_a_new_one = 0;

 if ( EventSet == NULL )
    return(handle_error(PAPI_EINVAL,NULL));

  if ( *EventSet == PAPI_NULL ) /* We need a new one */
    {
      ESI = allocate_EventSet();
      if (!ESI)
        return(handle_error(PAPI_ENOMEM,NULL));
      allocated_a_new_one = 1;
    }
  else /* One exists already */
    {
      ESI = lookup_EventSet(*EventSet);
      if ( ESI == NULL )
        return(handle_error(PAPI_EINVAL,NULL));
    }


  /* Now we have a valid ESI, try to add HW event */
  errorCode = _papi_hwd_add_prog_event(ESI->machdep,code,extra);
  if (errorCode < PAPI_OK)
    {
damnit:      /* If we've allocated one, we must free it */
      if (allocated_a_new_one)
        free_EventSet(ESI);
      return(handle_error(errorCode,NULL));
    }

  errorCode=add_EventSet(ESI);
  if (errorCode<PAPI_OK)
    goto damnit;

  return(errorCode);

}








/*========================================================================*/
/* low-level function:                                                    */
/* int PAPI_rem_event(int EventSet, int Event)                            */ 

/* from the draft standard:
   This function removes the hardware counter Event from EventSet.
*/

int PAPI_rem_event(int EventSet, int Event)
{
  EventSetInfo *ESI;
  int errorCode;


  /* determine target ESI structure */
  ESI=lookup_EventSet(EventSet);
  if ( ESI == NULL )
      return(handle_error(PAPI_EINVAL,NULL));


  /* nullify all array values for ESI */
      errorCode=nullify_event(ESI,Event);
      if(errorCode<PAPI_OK) 
      return(handle_error(PAPI_EINVAL,NULL));

  errorCode=_papi_hwd_rem_event(ESI->machdep,Event);
  if(errorCode<PAPI_OK)
      return(handle_error(PAPI_EINVAL,NULL));

  return(PAPI_OK);
}

/*========================================================================*/
/* low-level function:                                                    */
/* static int PAPI_rem_events(int EventSet, int *Events, int number)      */
/*
   from the draft standard:

    This function performs the same as above [PAPI_rem_event]
    except for a vector of hardware Events.
    number is the number of events to remove.
*/
/*========================================================================*/

int PAPI_rem_events(int EventSet, int *RemEvents, int number)
{
  EventSetInfo *ESI;
  int k,j;
  int nActive;

 
  /* determine target ESI structure */
  ESI=lookup_EventSet(EventSet);
  if ( ESI == NULL )
      return(handle_error(PAPI_EINVAL,NULL));

  nActive=0;/*count number of active events*/


  j=0;
  while(RemEvents[j]) {
     /* determine index of target event k */
      k=lookup_EventCodeIndex(ESI,RemEvents[j]);
      if(k<PAPI_OK)
        return(handle_error(PAPI_EINVAL,NULL));
      else {
      PAPI_rem_event(EventSet,RemEvents[j]);
      nActive--;
      }
  j++;
  }/*end j*/

  number=nActive;

  return(PAPI_OK);
}

/*========================================================================*/
/* low-level function:                                                    */
/* static int PAPI_list_events(int EventSet, int *Events, int *number)    */
/*
   from the draft standard:

   This function decomposes EventSet into the hardware
   Events it contains. number is both an input and output.

   ---------------------------------------------------------------------
   number as input:  total of all events ever added [active + inactive ]
   number as output: total of all active events at this time
*/
/*========================================================================*/

int PAPI_list_events(int EventSet, int *Events, int *number)
{
   EventSetInfo *ESI;
   int i,k,nActive;
   /*char *standardEventDef_STR[25]    added to papiStdDefs.h*/
   /*int  standardEventDef_NUM[25] added to papiStdDefs.h*/

  /* determine target ESI structure */
  ESI=lookup_EventSet(EventSet);
  if ( ESI == NULL )
      return(handle_error(PAPI_EINVAL,NULL));

  nActive=0;/*count number of active events*/

     for(i=0;i<*number;i++) {

  /* determine index of target event k */
      k=lookup_EventCodeIndex(ESI,Events[i]);
      if(k<PAPI_OK) {
        printf("\n EventCodeArray[%d]:  no value", i);
        }
      else {
        printf("\n ESI->EventCodeArray[%d]: ESI->latest[%d] : %lld",
        i,i, ESI->latest[i]);
        nActive++;
        }
     }/* end for i */

   *number=nActive;
  
   return(PAPI_OK);
}
  




/*========================================================================*/
static int lookup_EventCodeIndex(EventSetInfo *ESI,int Event)
{
  /* This function returns the index of the EventCodeArray element
     that matches the value of Event. 
     If no match is found, PAPI_NULL is returned.
     This function assumes that the length of ESI->EventCodeArray
     is equal to ESI->NumberOfCounters
  */

  int i;

  for(i=0;i<ESI->NumberOfCounters;i++) {
  if(ESI->EventCodeArray[i]==Event) return (i);
  }

  return(PAPI_NULL);

} 





/*========================================================================*/
/* static int add_EventSet(EventSetInfo *ESI)                             */
/*
   Not in draft standard.

  Called by:    PAPI_add_event
                PAPI_add_events
                PAPI_add_pevent
*/

int add_EventSet(EventSetInfo *ESI)
{
  int N; /* temp value for bookkeeping */
  int errorCode;

  if (PAPI_EVENTSET_MAP.availSlots==0)
  errorCode=expand_dynamic_array(&PAPI_EVENTSET_MAP);
  if(errorCode<PAPI_OK) 
     return(errorCode);

  ESI->EventSetIndex=PAPI_EVENTSET_MAP.lowestEmptySlot;
  PAPI_EVENTSET_MAP.dataSlotArray[PAPI_EVENTSET_MAP.lowestEmptySlot] = ESI;
 
  /* Update the values for lowestEmptySlot, num of availSlots */

  N=PAPI_EVENTSET_MAP.lowestEmptySlot;
  while(PAPI_EVENTSET_MAP.dataSlotArray[N]!=NULL)
    N++;
  PAPI_EVENTSET_MAP.lowestEmptySlot=N;
  PAPI_EVENTSET_MAP.availSlots--;
 
  return(PAPI_OK);
}


/*========================================================================*/
/* low-level function:                                                    */
/* static int remove_EventSet(int eventset)                               */
/* eventset is a value like: PAPI_L1_ICM                                  */


static int remove_EventSet(int eventset)
{

   /* Determine if target eventset value is valid*/
      if(PAPI_EVENTSET_MAP.dataSlotArray[eventset]==NULL){
        return(handle_error(PAPI_EINVAL,NULL));
      }

   /* Free target EventSet*/
      free_EventSet(PAPI_EVENTSET_MAP.dataSlotArray[eventset]);

   /* do bookkeeping for PAPI_EVENTSET_MAP */
        PAPI_EVENTSET_MAP.dataSlotArray[eventset]=NULL;
     if(PAPI_EVENTSET_MAP.lowestEmptySlot < eventset)
        PAPI_EVENTSET_MAP.lowestEmptySlot = eventset;
        PAPI_EVENTSET_MAP.availSlots++;

  return(PAPI_OK);
}


/*========================================================================*/
static EventSetInfo *lookup_EventSet(int eventset)
{
  if ((eventset > 1) && (eventset < PAPI_EVENTSET_MAP.totalSlots))
    return(PAPI_EVENTSET_MAP.dataSlotArray[eventset]);
  else
    return(NULL);
}

/*========================================================================*/
static int event_is_in_eventset(int event, EventSetInfo *ESI)
{
  int i = ESI->NumberOfCounters;
  int *events_in_set = ESI->EventCodeArray;

  while ((--i) >= 0)
    {
      if (events_in_set[i] == event)
	return(i);
    }
  
  return(handle_error(PAPI_EINVAL,"Event not in EventSet"));
}

/* There's more damn comments in this file than code! Let's go guys. */

/*========================================================================*/
static int set_multiplex(int eventset, PAPI_option_t *ptr)
{
  EventSetInfo *ESI = lookup_EventSet(eventset);

  if (!ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  return(_papi_hwd_setopt(PAPI_SET_MPXRES,ESI,ptr));
}

/*========================================================================*/
static int overflow_is_active(EventSetInfo *ESI)
{
  if (ESI->overflow.eventindex >= 0) /* No overflow active for this EventSet */
    return(1);
  else
    return(0);
}

/*========================================================================*/
static int set_overflow(int eventset, PAPI_option_t *ptr)
{
  int retval, ind;
  EventSetInfo *ESI = lookup_EventSet(eventset);

  /* Check the arguments */

  if (!ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  ind = event_is_in_eventset(ptr->overflow.event, ESI);
  if (ind < PAPI_OK)
    return(ind);

  if (ptr->overflow.threshold < 0)
    return(handle_error(PAPI_EINVAL,"Threshold cannot be less than zero"));

  if (ptr->overflow.threshold > 0)
    if (!ptr->overflow.handler)
      return(handle_error(PAPI_EINVAL,"Overflow handler not specified"));
					
  /* Args are good. Is overflow active? */

  if ((!overflow_is_active(ESI)) && (ptr->overflow.threshold == 0))
    return(PAPI_OK);
    
  retval = _papi_hwd_setopt(PAPI_SET_OVRFLO,ESI,ptr);
  if (retval < 0)
    return(retval);

  /* ESI->overflow.eventindex = ind;
  ESI->overflow.deadline = ;
  ESI->overflow.milliseconds =;
  memcpy(&ESI->overflow.option,&ptr->overflow,sizeof(ptr->overflow)) */
  return(PAPI_OK);
}

/*========================================================================*/
static int get_multiplex(int *eventset, PAPI_option_t *ptr)
{
  EventSetInfo *ESI;

  if ((!eventset) || (!ptr))
    return(handle_error(PAPI_EINVAL,"Invalid pointer"));

  ESI = lookup_EventSet(*eventset);
  if (!ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  return(_papi_hwd_getopt(PAPI_GET_MPXRES,ESI,ptr));
}

/*========================================================================*/
static int get_overflow(int *eventset, PAPI_option_t *ptr)
{
  EventSetInfo *ESI;

  if ((!eventset) || (!ptr))
    return(handle_error(PAPI_EINVAL,"Invalid pointer"));

  ESI = lookup_EventSet(*eventset);
  if (!ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  return(_papi_hwd_getopt(PAPI_GET_OVRFLO,ESI,ptr));
}

/*========================================================================*/
int PAPI_set_opt(int option, int value, PAPI_option_t *ptr)
{
  switch (option)
    {
    case PAPI_SET_MPXRES:
      return(set_multiplex(value,ptr)); 
    case PAPI_SET_OVRFLO:
      return(set_overflow(value,ptr));
    case PAPI_DEBUG:
      if ((value < PAPI_QUIET) || (value > PAPI_VERB_ESTOP)) 
	return(handle_error(PAPI_EINVAL,NULL));
      PAPI_ERR_LEVEL = value;
      return(PAPI_OK);
    default:
      return(handle_error(PAPI_EINVAL,"No such option"));
    }
}

/*========================================================================*/
int PAPI_get_opt(int option, int *value, PAPI_option_t *ptr)
{
  switch (option)
    {
    case PAPI_GET_MPXRES:
      return(get_multiplex(value,ptr)); 
    case PAPI_GET_OVRFLO:
      return(get_overflow(value,ptr));
    case PAPI_DEBUG:
      if (!value)
	return(handle_error(PAPI_EINVAL,"Invalid pointer"));
      *value = PAPI_ERR_LEVEL;
      return(PAPI_OK);
    default:
      return(handle_error(PAPI_EINVAL,NULL));
    }
}

/*========================================================================*/
int PAPI_start(int EventSet)
{ 
  int retval;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;

  retval = _papi_hwd_start(this_machdep);
  if(retval) return(PAPI_EBUG);
  return 0;
}

/*========================================================================*/
int PAPI_stop(int EventSet, long long *values)
{ int retval, i;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;

  retval = _papi_hwd_stop(this_machdep, values);
  if(retval) return(PAPI_EBUG);

  for(i=0; i<(_papi_system_info.num_gp_cntrs +
              _papi_system_info.num_sp_cntrs); i++)
  { if(values[i] >= 0)
    { printf("\tCounter %d : %lld\n", i, values[i]);
    }
  }

  retval = _papi_hwd_reset(this_machdep);
  if(retval) return(PAPI_EBUG);
  return 0;
}

/*========================================================================*/
int PAPI_read(int EventSet, long long *values)
{ int retval, i;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;

  retval = _papi_hwd_read(this_machdep, values);
  if(retval) return(PAPI_EBUG);

  for(i=0; i<(_papi_system_info.num_gp_cntrs + 
              _papi_system_info.num_sp_cntrs); i++)
  { if(values[i] >= 0) 
    { printf("\tCounter %d : %lld\n", i, values[i]);
    }
  }
  return 0;
}

/*========================================================================*/

int PAPI_accum(int EventSet, long long *values)
{ 
  EventSetInfo *ESI;
  int retval, i, bound;
  long long a,b,c,*increase;

  ESI = lookup_EventSet(EventSet);
  if ( ESI == NULL ) 
    return(handle_error(PAPI_EINVAL,NULL));

  increase = ESI->latest;
  bound = num_counters(ESI);

  for ( i=0 ; i < bound; i++)
    { 
      a = increase[i];
      b = values[i];
      c = a + b;
      values[i] = c;
    }

  retval = _papi_hwd_read(ESI->machdep, increase);
  if (retval < PAPI_OK) 
    return(handle_error(retval,NULL));

  retval = _papi_hwd_reset(ESI->machdep);
  if (retval < PAPI_OK)
    return(handle_error(retval,NULL));
  return(retval);
}

/*========================================================================*/
int PAPI_write(int EventSet, long long *values)
{ int retval;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;

  retval = _papi_hwd_write(this_machdep, values);
  if(retval) return(PAPI_EBUG);
  return 0;
}

/*========================================================================*/
int PAPI_reset(int EventSet)
{ int retval;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;

  retval = _papi_hwd_reset(this_machdep);
  if(retval) return(PAPI_EBUG);
  return 0;
}

