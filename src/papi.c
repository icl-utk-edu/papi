/* papi.c */
#include <stdio.h>
#include <malloc.h>
#include <memory.h>
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
EventSetInfo *allocate_EventSet(void);
int add_EventSet(EventSetInfo *);
static int remove_EventSet(int);
static void free_EventSet(EventSetInfo *);
static int handle_error(int, char *);
static char *get_error_string(int);

/* Global variables */
/* These will eventually be encapsulated into per thread structures. */ 

/* Our integer to EventSetInfo * mapping */

DynamicArray PAPI_EVENTSET_MAP = { 0, };    

/* Behavior of handle_error(). 
Changed to the default behavior of PAPI_QUIET in PAPI_init
after initialization is successful. */

int          PAPI_ERR_LEVEL = PAPI_VERB_ESTOP; 

/*========================================================================*/
/* low-level function:                                                    */    
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

static void initialize(void)
{
   memset(&PAPI_EVENTSET_MAP,0x00,sizeof(PAPI_EVENTSET_MAP));

   /* initialize values in PAPI_EVENTSET_MAP */ 

   PAPI_EVENTSET_MAP.dataSlotArray=
    (EventSetInfo **)malloc(PAPI_INIT_SLOTS*sizeof(void *));
   if(!PAPI_EVENTSET_MAP.dataSlotArray) 
     handle_error(PAPI_ENOMEM,"Initialization failed.");

   memset(&PAPI_EVENTSET_MAP.dataSlotArray,0x00, PAPI_INIT_SLOTS*sizeof(void *));

   PAPI_EVENTSET_MAP.totalSlots = PAPI_INIT_SLOTS;
   PAPI_EVENTSET_MAP.availSlots = PAPI_INIT_SLOTS - 1;
   PAPI_EVENTSET_MAP.lowestEmptySlot = 1;

   PAPI_ERR_LEVEL = PAPI_QUIET;
}

/*========================================================================*/
/* low-level function:                                                    */    
/* static void PAPI_shutdown (void);                                      */
/*                                                                        */
/* This function provides a graceful exit to the PAPI tool.               */
/* a. All memory associated with the PAPI tool is freed.                  */
/*  b. a shutdown message is written to stderr                            */ 
/*                                                                        */
/* Free the elements of PAPI_EVENTSET_MAP.dataSlotArray by calling        */
/* papi_freeEventSet.                                                     */
/* The function _papi_free_EventSet frees the _EventSetInfo structure     */ 
/* in two stages:                                                         */
/*    1. Free the internal pointers.                                      */  
/*    2. Free the pointer to the _EventSetInfo structure itself.          */ 
/*                                                                        */
/* Once the PAPI_EVENTSET_MAP.dataSlotArray has had all its elements      */
/* removed, then the PAPI_EVENTSET_MAP.dataSlotArray itself may be freed. */
/* The EM pointer itself does not have to be freed because it points to   */
/* the static memory location of PAPI_EVENTSET_MAP.                       */
/*========================================================================*/

void PAPI_shutdown(void) {

    int i;

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
/* low-level functions for error handling:                                */
/*	int PAPI_perror(int code, char *destination, int length)          */
/*	static char *get_error_string(int code)                           */ 
/*	static int handle_error(int PAPI_errorCode, char *errorMessage)   */
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
       if(PAPPI_errorCode==PAPU_ESYS)fprintf(stderr,": %s",strerror(errno));*/

      /* check for user supplied error message */
      if (errorMessage) fprintf(stderr, ": %s", errorMessage);

      fprintf(stderr,"\n");

      if (PAPI_ERR_LEVEL==PAPI_VERB_ESTOP) 
	PAPI_shutdown();
    }
  return(PAPI_errorCode);

}

/*========================================================================*/
/* low-level function:                                                    */
/* static int _papi_expandDA(DynamicArray *EM);                           */
/*                                                                        */
/* The function _papi_expandDA expands PAPI_EVENTSET_MAP.dataSlotArray    */
/* when the array has become full. The function also updates totalSlots,  */
/* availSlots, and lowestEmptySlot.                                       */ 
/* This enables the user to load as many events as needed.                */
/*                                                                        */
/* DynamicArray PAPI_EVENTSET_MAP is global.                              */ 
/* PAPI_EVENTSET_MAP.dataSlotArray[0] is reserved for internal use.       */
/*                                                                        */
/* PAPI_EVENTSET_MAP.dataSlotArray[1] holds ptr to EventSetInfo number 1  */
/* PAPI_EVENTSET_MAP.dataSlotArray[2] holds ptr to EventSetInfo number 2  */
/* PAPI_EVENTSET_MAP.dataSlotArray[3] holds ptr to EventSetInfo number 3  */ 
/*  ...                                                                   */
/*  ...                                                                   */
/* PAPI_EVENTSET_MAP.dataSlotArray[N] holds ptr to EventSetInfo number N, */
/* where: N < PAPI_EVENTSET_MAP.totalSlots                                */
/*                                                                        */
/* The function _papi_expandDA returns PAPI_OK upon success               */
/* or PAPI_ENOMEM on failure.                                             */
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
/* low-level function:                                                    */
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
/* Check if EventSetIndex is a valid value 4 different ways. 
   1. if (EventSetIndex < 0 ) => bad value of N 
   2. if(PAPI_EVENTSET_MAP.dataSlotArray[EventSet]==0) => N=0 reserved 
   3. if (EventSetIndex => PAPI_EVENTSET_MAP.totalSlots ) => N out of range
   4. if (PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]==NULL) => N not assigned yet
*/
    if ((EventSetIndex < 1) || 
	(EventSetIndex >= PAPI_EVENTSET_MAP.totalSlots) ||
	(PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]== NULL)) 
		return (handle_error(PAPI_EINVAL,NULL));

    /* Good value for PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]-> state */

    *status = (PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex])->state;
     		return(PAPI_OK);
}


/*========================================================================*/
/* low-level function:                                                    */
/* static EventSetInfo *allocate_EventSet(void);                          */
/*                                                                        */
/* This function allocates space for one EventSetInfo structure and for   */
/* all of the pointers in this structure.  If any malloc in this function */
/* fails, all memory malloced to the point of failure is freed, and NULL  */
/* is returned.  Upon success, a pointer to the EventSetInfo data         */
/* structure is returned.                                                 */
/*========================================================================*/

EventSetInfo *allocate_EventSet(void) 
{
  EventSetInfo *ESI;
  int counterArrayLength;
  /****PAPI_option_t *ptr;  needed for overflow****/
  
  counterArrayLength=_papi_system_info.num_gp_cntrs
		    +_papi_system_info.num_sp_cntrs;

  ESI=(EventSetInfo *)malloc(sizeof(EventSetInfo));
  if (ESI==NULL) 
    return(NULL); 
  memset(&ESI,0x00,sizeof(ESI));

  ESI->EventCodeArray=(int *)malloc(counterArrayLength*sizeof(int));
  ESI->machdep=(void *)malloc(_papi_system_info.size_machdep);
  ESI->start =(long long *)malloc(counterArrayLength*sizeof(long long));
  ESI->stop  =(long long *)malloc(counterArrayLength*sizeof(long long));
  ESI->latest=(long long *)malloc(counterArrayLength*sizeof(long long));

  if ((ESI->machdep   == NULL) || 
      (ESI->start     == NULL) || 
      (ESI->stop      == NULL) || 
      (ESI->latest    == NULL) ||
      (ESI->EventCodeArray == NULL))
    {
      if (ESI->machdep) free(ESI->machdep);
      if (ESI->start) free(ESI->start);
      if (ESI->stop) free(ESI->stop);
      if (ESI->latest) free(ESI->latest);
      if (ESI->EventCodeArray) free(ESI->EventCodeArray);
      free(ESI);
      return(NULL);
    }
  memset(ESI->EventCodeArray,0x00,counterArrayLength*sizeof(int));
  memset(ESI->machdep,0x00,_papi_system_info.size_machdep);
  memset(ESI->start,  0x00,counterArrayLength*sizeof(long long));
  memset(ESI->stop,   0x00,counterArrayLength*sizeof(long long));
  memset(ESI->latest, 0x00,counterArrayLength*sizeof(long long));

  ESI->state = PAPI_STOPPED; 
  ESI->NumberOfCounters=0;
  /****ESI->overflow=get_overflow(EventSet,ptr);****/                   

  return(ESI);
}



/*========================================================================*/
/* low-level function:                                                    */
/* static int free_EventSet(EventSetInfo *ESI);                           */
/*                                                                        */ 
/* This function should free memory for one EventSetInfo structure.       */
/* The argument list consists of a pointer to the EventSetInfo            */
/* structure, *ESI.                                                       */
/* The calling function should check  for ESI==NULL.                      */
/*========================================================================*/

static void free_EventSet(EventSetInfo *ESI) 
{
  if(!ESI)return;
  
  if (ESI->EventCodeArray) free(ESI->EventCodeArray);
  if (ESI->machdep)        free(ESI->machdep);
  if (ESI->start)          free(ESI->start);
  if (ESI->stop)           free(ESI->stop);
  if (ESI->latest)         free(ESI->latest);

  free(ESI);
}



/*========================================================================*/
/* low-level function:                                                    */
/* static int PAPI_add_event(int *EventSet, int EventSet)                 */
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
/*========================================================================*/
int PAPI_add_event(int *EventSet, int Event) {
EventSetInfo *ESI;
int errorCode;
int k;
/****PAPI_option_t *ptr;  needed for overflow****/


   /* Determine if target Event value is valid standard value*/ 
      errorCode=checkTargetEventValue(Event);
      if(errorCode!=PAPI_OK) PAPI_perror(PAPI_EINVAL,NULL,0);

if(EventSet==NULL) {/*create new ESI*/
ESI=allocate_EventSet();
if(!ESI) return(PAPI_ENOMEM);

errorCode=add_EventSet(ESI);
if(errorCode!=PAPI_OK) return(errorCode);

}/* new ESI created and loaded to PAPI_EVENTSET_MAP.dataSlotArray */

/*determine the lowest open slot in EventCodeArray*/
k=locateTargetIndexECA(ESI,-1);
/*check if you ran out of counters*/
if(k<0)
return(handle_error(PAPI_EINVAL," ran out of counters "));

/*set EventCodeArray[k] to hold value of Event*/
ESI->EventCodeArray[k]=Event;
ESI->NumberOfCounters++;

errorCode=_papi_hwd_add_event(ESI->machdep,Event);
if(errorCode!=PAPI_OK) return(errorCode);
ESI->start[k]=0;
ESI->stop[k]=0;
ESI->latest[k]=0;
/****ESI->overflow=get_overflow(EventSet,ptr);****/                   

return(PAPI_OK);

}/* end PAPI_add_event */

/*========================================================================*/
/* This function determines which standardEventDef_INT[j] matches the     */
/* value EventID.  

   The index value "j" is returned.  
   If no match is found, -1 is returned. 
*/

	
int locateIndexStdEventDef(int EventID) {

int j=0;
for(j=0;j<24;j++)
if(EventID==standardEventDef_INT[j])return(j);

return(-1);

}


/*========================================================================*/
/* This function returns the index of the lowest empty array slot in      */
/* ESI->EventCodeArray                                                    */ 
/* The length of ESI->EventCodeArray is the value:
  counterArrayLength=_papi_system_info.num_gp_cntrs
		    +_papi_system_info.num_sp_cntrs;

  The function returns the integer value of the target slot index.

  If (ID==-1), look for lowest empty slot.
  If there are NO empty slots, a value of -1 is returned.

  If (ID>0) return the index of the slot that matches ID.
  If no match is found, return a value of -1.
*/

int locateTargetIndexECA (EventSetInfo *ESI,int EventID) {

int j, counterArrayLength;
counterArrayLength=_papi_system_info.num_gp_cntrs
		  +_papi_system_info.num_sp_cntrs;
j=0;

if(EventID==-1) {
while (j<counterArrayLength) {
if(  (ESI->EventCodeArray[j]<1) 
   ||(ESI->EventCodeArray[j]==NULL) ) return(j);
j++;
}
}

else {
while (j<counterArrayLength) {
if(ESI->EventCodeArray[j]==EventID) return(j);
j++;
}
}

return(-1);
}

/*========================================================================*/
/* This function checks to see if the target Event or Event[n] value      */
/* is a valid standard value. 

   Detection of valid standard value causes return of PAPI_OK.
  
   No detection of valid standard value causes return of PAPI_EINVAL.
*/

int checkTargetEventValue(int Event) {

      int j;
      j=0;

      for(j=0;j<24;j++) 
      if(Event==standardEventDef_INT[j])return(PAPI_OK);
 
      return(PAPI_EINVAL);
}




/*========================================================================*/
/* low-level function:                                                    */
/* static int PAPI_add_events(int *EventSet, int *Events, int number)     */
/*                                                                        */ 
/*
   from the draft standard:

   Same as above [ PAPI_add_event ] for a vector of events. 
   If one or more of Events cannot be counted on this platform, 
   then this call fails and PAPI_ENOEVNT is returned. In addition, 
   the invalid entries in the Events array are set to PAPI_NULL
   such that the user can successfully reissue the call.
*/
/*========================================================================*/


int PAPI_add_events(int *EventSet, int *Events, int number) {
EventSetInfo *ESI;
int errorCode;
int k,m;

if(EventSet==NULL) {/*create new ESI*/
ESI=allocate_EventSet();
if(!ESI) return(PAPI_ENOMEM);
}

for(m=0;m<number;m++) {/* add Events[m]*/

/* Determine if target Events[m] value is valid standard value*/ 
      errorCode=checkTargetEventValue(Events[m]);
      if(errorCode!=PAPI_OK) PAPI_perror(PAPI_EINVAL,NULL,0);

/*determine the lowest open slot in EventCodeArray*/
      k=locateTargetIndexECA(ESI,-1);
/*check if you ran out of counters*/
if(k<0)
return(handle_error(PAPI_EINVAL," ran out of counters "));

/*set ESI->EventCodeArray[k] to hold value of Event m*/
ESI->EventCodeArray[k]=Events[m];

errorCode=_papi_hwd_add_event(ESI->machdep,Events[m]);
if(errorCode!=PAPI_OK) return(errorCode);
ESI->start[k]=0;
ESI->stop[k]=0;
ESI->latest[k]=0;
/****ESI->overflow=get_overflow(EventSet,ptr);****/                   

ESI->NumberOfCounters++;
}/*end for m*/

return(PAPI_OK);

}/* end PAPI_add_events */




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
/* This is the function where I think we should add a registry
   of user-prgrammable events. chd.*/

   
EventSetInfo *ESI;
int errorCode,k;
void *extra; /*needed to call _papi_hwd_add_prog_event*/
/****PAPI_option_t *ptr;  needed for overflow****/

if(EventSet==NULL) {/*create new ESI*/
ESI=allocate_EventSet();
if(!ESI) return(PAPI_ENOMEM);

errorCode=add_EventSet(ESI);
if(errorCode!=PAPI_OK) return(errorCode);

}/* new ESI created and loaded to PAPI_EVENTSET_MAP.dataSlotArray */

/* determine lowest empty slot in ESI->EventCodeArray*/
k=locateTargetIndexECA(ESI,-1);
/*check if you ran out of counters*/
if(k<0)
return(handle_error(PAPI_EINVAL," ran out of counters "));

/*set ESI->EventCodeArray[k] to hold value of Event*/
ESI->EventCodeArray[k]=code;

/* Add ESI for (pevent) code to PAPI_EVENTSET_MAP.dataSlotArray*/

errorCode=_papi_hwd_add_prog_event(ESI->machdep,code,extra);
if(errorCode!=PAPI_OK) return(errorCode);
ESI->start[k]=0;
ESI->stop[k]=0;
ESI->latest[k]=0;
/****ESI->overflow=get_overflow(EventSet,ptr);****/                   
ESI->NumberOfCounters++;

return(PAPI_OK);
}



/*========================================================================*/
/* low-level function:                                                    */
/* static int add_EventSet(EventSetInfo *ESI)                             */
/*
   Not in draft standard.

  Called by:    PAPI_add_event
		PAPI_add_events
		PAPI_add_pevent
*/
/*========================================================================*/
int add_EventSet(EventSetInfo *ESI) 
{
  int N; /* temp value for bookkeeping */
  int errorCode;

  if (PAPI_EVENTSET_MAP.availSlots==0) 
  errorCode=expand_dynamic_array(&PAPI_EVENTSET_MAP);
  if(errorCode!=PAPI_OK) return(errorCode);

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
/* static int PAPI_rem_event(int EventSet, int Event)                     */
/* 
   from the draft standard:

   This function removes the hardware counter Event from EventSet.


*/
/*========================================================================*/

int PAPI_rem_event(int EventSet, int Event) 
{
  EventSetInfo *ESI;
  int k;
  int errorCode;

  /* Would you like this function to also remove Pevents????*/

   /*int  standardEventDef_INT[25] added to papiStdDefs.h*/

   /* Determine if target Event value is valid standard value*/ 
      errorCode=checkTargetEventValue(Event);
      if(errorCode!=PAPI_OK) PAPI_perror(PAPI_EINVAL,NULL,0);
        
 
  /* determine target ESI structure */
  ESI=PAPI_EVENTSET_MAP.dataSlotArray[EventSet];

  /* determine index of target event k */
      k=locateTargetIndexECA(ESI,Event);
      if(k<0) return(PAPI_EINVAL);

  *ESI->EventCodeArray= -1; /* EventCodeArray not active*/
  errorCode=_papi_hwd_rem_event(ESI->machdep,Event);
  if(errorCode!=PAPI_OK)return(errorCode);
  ESI->start [k]=-1; /* remove start */ 
  ESI->stop  [k]=-1; /* remove stop  */
  ESI->latest[k]=-1; /* remove latest*/

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

  nActive=number;
  
  /* determine target ESI structure */
  ESI=PAPI_EVENTSET_MAP.dataSlotArray[EventSet];


  nActive=0;/*count number of active events*/

  j=0;
  while(RemEvents[j]) { 
     /* determine index of target event k */
      k=locateTargetIndexECA(ESI,RemEvents[j]);
      if(k>0) {
      PAPI_rem_event(EventSet,RemEvents[j]);
      ESI->EventCodeArray[k]=-1;
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
   int i,j,k,nActive,errorCode;
   /*char *standardEventDef[25]    added to papiStdDefs.h*/
   /*int  standardEventDef_INT[25] added to papiStdDefs.h*/

    /*This is only done for standard events, not for user defined events.*/

   /* Determine target EventSet */ 
      ESI=PAPI_EVENTSET_MAP.dataSlotArray[EventSet];
    

     nActive=0;
     for(i=0;i<*number;i++) {

   /* Determine if target Event value is valid standard value*/ 
      errorCode=checkTargetEventValue(Events[i]);
      if(errorCode!=PAPI_OK) PAPI_perror(PAPI_EINVAL,NULL,0);
     
     else {
    
     
     j=locateTargetIndexECA(ESI,Events[i]);
     if(j==-1){
     printf("\n Events[%d]=%d not found in EventCodeArray", i,Events[i]);
     }
     else {

     /* get index k for print string value */
     k=locateIndexStdEventDef(Events[i]);
     printf("\n EventCodeArray[%d]: %s : %lld", 
     i, standardEventDef[k],ESI->latest[i]); 
      nActive++;
     }
     }/* end else on good Events[i] */
     }/* end for i */
	
   *number=nActive;
   
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
      PAPI_perror(PAPI_EINVAL,NULL,0);
      return(PAPI_OK);
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


/*========================================================================*/
/* There's more damn comments in this file than code! Let's go guys. */

#if 0

static int set_multiplex(int value, PAPI_option_t *ptr)
{
  return(_papi_hwd_setopt(PAPI_SET_MPXRES,value,ptr));
}


/*========================================================================*/
static int get_multiplex(int *value, PAPI_option_t *ptr)
{
  if ((!value) || (!ptr))
    return(handle_error(PAPI_EINVAL,"Invalid pointer"));

  return(_papi_hwd_getopt(PAPI_SET_MPXRES,value,ptr));
}


/*========================================================================*/
static int set_overflow(int eventset, PAPI_option_t *ptr)
{
  int retval, ind;
  EventSetInfo *ESI = lookup_EventSet(eventset);

  if (!ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  if (ESI->overflow.eventindex < 0) /* No overflow is active for this EventSet */
    {
      if (ptr == NULL)    /* Turning off overflow that's already off */
	return(PAPI_OK);
    }
    
  ind = event_is_in_eventset(ptr->overflow.event, ESI);
  if (ind < 0)
    return(ind);

  retval = _papi_hwd_setopt(PAPI_SET_OVRFLO,ESI->EventCodeArray[ind],ptr);
  if (retval < 0)
    return(retval);

  ESI->overflow.eventindex = ind;
  return(PAPI_OK);
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

  memcpy(ptr,&ESI->overflow.option,sizeof(*ptr));
  return(PAPI_OK);
}
#endif


/*========================================================================*/
int PAPI_set_opt(int option, int value, PAPI_option_t *ptr)
{
  switch (option)
    {
    case PAPI_SET_MPXRES:
      /* return(set_multiplex(value,ptr)); */
      return(PAPI_OK);
    case PAPI_SET_OVRFLO:
      /* return(set_overflow(value,ptr)); */
      return(PAPI_OK);
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
      /* return(get_multiplex(value,ptr)); */
      return(PAPI_OK);
    case PAPI_GET_OVRFLO:
    /* return(get_overflow(value,ptr)); */
      return(PAPI_OK);
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
{ int retval;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;

  retval = _papi_hwd_start(this_machdep);
  if(retval) return(PAPI_EBUG);
  return 0;
}


/*========================================================================*/
int PAPI_stop(int EventSet, long long *values)
{ int retval, machnum;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;

  retval = _papi_hwd_stop(this_machdep, values);
  if(retval) return(PAPI_EBUG);

  if(machnum == 0) return(PAPI_ENOTRUN);

  if(machnum >= 4)
  { printf("\tCounter 0 : %lld\n", values[0]);
  }
  if((machnum == 3) || (machnum == 7))
  { printf("\tCounter 1 : %lld\n", values[1]);
    printf("\tCounter 2 : %lld\n", values[2]);
  }
  if((machnum == 2) || (machnum == 6))
  { printf("\tCounter 1 : %lld\n", values[1]);
  }
  else printf("\tCounter 2 : %lld\n", values[2]);

  retval = _papi_hwd_reset(this_machdep);
  if(retval) return(PAPI_EBUG);
  return 0;
}


/*========================================================================*/
int PAPI_read(int EventSet, long long *values)
{ int retval, machnum;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;

  retval = _papi_hwd_read(this_machdep, values, &machnum);
  if(retval) return(PAPI_EBUG);
  if(machnum == 0) return(PAPI_ENOTRUN); 

  if(machnum >= 4)
  { printf("\tCounter 0 : %lld\n", values[0]);
  }
  if((machnum == 3) || (machnum == 7))
  { printf("\tCounter 1 : %lld\n", values[1]);
    printf("\tCounter 2 : %lld\n", values[2]);
  }
  if((machnum == 2) || (machnum == 6))
  { printf("\tCounter 1 : %lld\n", values[1]);
  }
  else printf("\tCounter 2 : %lld\n", values[2]);
  return 0;
}



/*========================================================================*/
int PAPI_accum(int EventSet, long long *values)
{ int retval, machnum, i;
  void *this_machdep = PAPI_EVENTSET_MAP.dataSlotArray[EventSet]->machdep;
  long long increase[3];   /*should be a variable known to library containing*/
				/*number of counters*/

  retval = _papi_hwd_read(this_machdep, increase, &machnum);
  if(retval) return(PAPI_EBUG);
  if(machnum == 0) return(PAPI_ENOTRUN);

  for(i=0; i<3; i++)
  { values[i] += increase[i];
  }

  if(machnum >= 4)
  { printf("\tCounter 0 : %lld\n", values[0]);
  }
  if((machnum == 3) || (machnum == 7))
  { printf("\tCounter 1 : %lld\n", values[1]);
    printf("\tCounter 2 : %lld\n", values[2]);
  }
  if((machnum == 2) || (machnum == 6))
  { printf("\tCounter 1 : %lld\n", values[1]);
  }
  else printf("\tCounter 2 : %lld\n", values[2]);

  retval = _papi_hwd_reset(this_machdep);
  if(retval) return(PAPI_EBUG);
  return 0;
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

