/* file: papi.c */ 

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>
#include <strings.h>
#include <errno.h>

#include "papi_internal.h"
#include "papi.h"

/* There are two global variables.    */ 
/* Their values are set in PAPI_init. */  

DynamicArray PAPI_EVENTSET_MAP;    
int          PAPI_ERR_LEVEL; 

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
/* Set pointer to GLOBAL variable PAPI_EVENTSET_MAP.                         */
/* Since PAPI_EVENTSET_MAP is declared at the top of the program             */
/* no malloc for EM is needed.                                            */
/* But the pointer EM->dataSlotArray must be malloced here.               */
/*                                                                        */
/* Initialize PAPI_ERR_LEVEL.                                             */
/* The user selects error handling with ERROR_LEVEL_CHOICE.               */  
/* ERROR_LEVEL_CHOICE may have one of two values:                         */
/*   a. PAPI_VERB_ECONT [print error message, then continue processing ]  */
/*   b. PAPI_VERB_ESTOP [print error message, then shutdown ]             */
/*========================================================================*/
static void PAPI_init(int ERROR_LEVEL_CHOICE) 
{
   memset(&PAPI_EVENTSET_MAP,0x00,sizeof(PAPI_EVENTSET_MAP));

/* initialize values in PAPI_EVENTSET_MAP */ 

   PAPI_EVENTSET_MAP.dataSlotArray=(EventSetInfo **)malloc(PAPI_EVENTSET_MAP.totalSlots*sizeof(void *));
   if(!PAPI_EVENTSET_MAP.dataSlotArray) PAPI_shutdown();
   bzero(PAPI_EVENTSET_MAP.dataSlotArray,sizeof(PAPI_EVENTSET_MAP.dataSlotArray));

   PAPI_EVENTSET_MAP.totalSlots = PAPI_INIT_SLOTS;
   PAPI_EVENTSET_MAP.availSlots = PAPI_INIT_SLOTS - 1;
   PAPI_EVENTSET_MAP.lowestEmptySlot = 1;

/* initialize PAPI_ERR_LEVEL */

   if(   (ERROR_LEVEL_CHOICE!=PAPI_VERB_ECONT)
       &&(ERROR_LEVEL_CHOICE!=PAPI_VERB_ESTOP) ) 
          PAPI_shutdown(); 

   PAPI_ERR_LEVEL=ERROR_LEVEL_CHOICE;

   return;
   }/***/
/*========================================================================*/
/*end function: PAPI_init                                                 */
/*========================================================================*/
/*========================================================================*/

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
/* Free the elements of PAPI_EVENTSET_MAP.dataSlotArray by calling papi_freeEventSet.   */
/* The function _papi_free_EventSet frees the _EventSetInfo structure     */ 
/* in two stages:                                                         */
/*    1. Free the internal pointers.                                      */  
/*    2. Free the pointer to the _EventSetInfo structure itself.          */ 
/*                                                                        */
/* Once the PAPI_EVENTSET_MAP.dataSlotArray has had all its elements removed, then      */
/* the PAPI_EVENTSET_MAP.dataSlotArray itself may be freed.                             */
/* The EM pointer itself does not have to be freed because it points to   */
/* the static memory location of PAPI_EVENTSET_MAP.                          */
/*========================================================================*/

void PAPI_shutdown(void) {

    int i;
    /* close all memory pointed to by xEM */
    /* this code under construction       */
    /* note: do we need to make a special case for PAPI_EVENTSET_MAP.dataSlotArray[0]?*/


    /* free all the EventInfo Structures in the PAPI_EVENTSET_MAP.dataSlotArray*/
    for(i=0;i<PAPI_EVENTSET_MAP.totalSlots;i++) {
	if(PAPI_EVENTSET_MAP.dataSlotArray[i]) {
 	  _papi_free_EventSet(PAPI_EVENTSET_MAP.dataSlotArray[i]); 
	  }/* end if */
	}/* end for */ 
		 
	free(PAPI_EVENTSET_MAP.dataSlotArray);



    /* shutdown message */
    fprintf(stderr,"\n\n PAPI SHUTDOWN. \n\n");


    return;
}/***/
/*========================================================================*/
/*end function: PAPI_shutdown                                             */
/*========================================================================*/
/*========================================================================*/


/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* static int PAPI_error(int PAPI_errorCode, char *errorMessage);         */
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
"Call to PAPI_error made with no error",
"Invalid argument",
"Insufficient memory",
"A System/C library call failed",
"Substrate returned an error",
"Access to the counters was lost or interrupted",
"Internal error, please send mail to the developers",
"Hardware Event does not exist",
"Hardware Event exists, but cannot be counted due to counter resource limits",
"No Events or EventSets are currently counting" 
"Call to PAPI_error with unknown error code",
};

int PAPI_error (int PAPI_errorCode, char *errorMessage) 
{
  if (PAPI_ERR_LEVEL) 
    {
      if (PAPI_errorCode > 0) 
	PAPI_errorCode = 0;
      else 
	PAPI_errorCode = - PAPI_errorCode;

      if (PAPI_errorCode > PAPI_NUM_ERRORS)
	PAPI_errorCode = PAPI_EMISC;

      /* print standard papi error message */
      fprintf(stderr, "%s", papi_errStr[PAPI_errorCode]);

      /* check for failed C library call*/
      if ( PAPI_errorCode==PAPI_ESYS ) fprintf(stderr,": %s",strerror(errno));

      /* check for user supplied error message */
      if (errorMessage) fprintf(stderr, ": %s", errorMessage);

      fprintf(stderr,"\n");

      if (PAPI_ERR_LEVEL==PAPI_VERB_ESTOP) 
	PAPI_shutdown();
    }
  return(PAPI_errorCode);

}/***/
/*========================================================================*/
/*end function: PAPI_error                                                */
/*========================================================================*/
/*========================================================================*/




/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* static int _papi_expandDA(DynamicArray *EM);                           */
/*                                                                        */
/* The function _papi_expandDA expands PAPI_EVENTSET_MAP.dataSlotArray when the array   */
/* has become full. The function also does all of the bookkeeping chores  */
/* [reset PAPI_EVENTSET_MAP.totalSlots, PAPI_EVENTSET_MAP.availSlots, PAPI_EVENTSET_MAP.lowestEmptySlot].           */
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
/* The EM structure holds the dataSlotArray. Each element of the          */ 
/* PAPI_EVENTSET_MAP.dataSlotArray is a pointer to an EventSetInfo structure.           */
/* The zero element of the PAPI_EVENTSET_MAP.dataSlotArray is reserved and is set       */
/* during initilization, [see papi_init]                                  */
/*                                                                        */
/* PAPI_EVENTSET_MAP.dataSlotArray[1] holds ptr to EventSetInfo number 1                */
/* PAPI_EVENTSET_MAP.dataSlotArray[2] holds ptr to EventSetInfo number 2                */
/* PAPI_EVENTSET_MAP.dataSlotArray[3] holds ptr to EventSetInfo number 3                */ 
/*  ...                                                                   */
/*  ...                                                                   */
/* PAPI_EVENTSET_MAP.dataSlotArray[N] holds ptr to EventSetInfo number N,               */
/* where                                                                  */     
/*	N < PAPI_EVENTSET_MAP.totalSlots                                                */
/*                                                                        */
/* The function _papi_expandDA returns PAPI_OK upon success               */
/* or PAPI_ENOMEM on failure.                                             */
/* Error handling should be done in the calling function.                 */  
/*========================================================================*/

static int _papi_expandDA(void)
{
  int  prevTotal;	
  EventSetInfo **n;

  /*realloc existing PAPI_EVENTSET_MAP.dataSlotArray*/
    
  n = (EventSetInfo **)realloc(PAPI_EVENTSET_MAP.dataSlotArray,PAPI_EVENTSET_MAP.totalSlots*sizeof(EventSetInfo *));
  if (n==NULL) 
    return(PAPI_error(PAPI_ENOMEM,NULL));   

  /* Need to assign this value, what if realloc moved it? */
  PAPI_EVENTSET_MAP.dataSlotArray = n;
       
  /* bookkeeping to accomodate successful realloc operation*/
  prevTotal           = PAPI_EVENTSET_MAP.totalSlots; 
  PAPI_EVENTSET_MAP.totalSlots     += prevTotal;
  PAPI_EVENTSET_MAP.availSlots      = prevTotal;
  PAPI_EVENTSET_MAP.lowestEmptySlot = prevTotal;

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
	(PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]==NULL)) return (PAPI_error(PAPI_EINVAL,NULL));

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
  
  counterArrayLength=_papi_system_info.num_gp_cntrs+_papi_system_info.num_sp_cntrs;

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
/*end function: _papi_free_EventSet                                       */
/*========================================================================*/
/*========================================================================*/

/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* int PAPI_load_event(EventSetInfo *thisEventSet, int EventSetIndex)     */
/*                                                                        */
/* The function PAPI_load_event is derived from PAPI_add_event,           */ 
/* which is described below [from papi draft standard ].                  */ 
/*                                                                        */
/* Note that the first argument for PAPI_load_Event is a ptr to a         */
/* previously allocated EventSetInfo structure that was returned by       */
/* PAPI_allocate_EventSet.                                                */
/*                                                                        */
/* PAPI_load_event sets up a new EventSetInfo structure or modifies an    */
/* existing EventSetInfo structure.  In the case of a new EventSetInfo    */
/* structure, PAPI_load_event loads the new EventSetInfo pointer to the   */
/* PAPI_EVENTSET_MAP.dataSlotArray by calling PAPI_load_dataSlotArrayElement.           */
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


int PAPI_load_event(EventSetInfo *thisESI, int EventSetIndex) 
{
int returnCode;

/* if thisESI is new */
if(thisESI->EventSetIndex==0) { 
	returnCode=PAPI_load_dataSlotArrayElement(thisESI);
	if(returnCode!=PAPI_OK) {
	PAPI_error(returnCode," PAPI_load_event error on new *EventSetInfo");
	}}

/* hardware dependent information */

/*
-------------------------------------------------------------------
Some of these are ints, some are ptrs.  I put the data type to the
right of the assignment statement.

The writing of this function needs to be coordinated with the
hardware dependent information.  I need help to do this. c! 
-------------------------------------------------------------------
thisESI->NumberOfCounters= _______;  int 
thisESI->EventCodeArray  = _______;  int *
thisESI->machdep         = _______;  void *
thisESI->start           = _______;  long long *
thisESI->stop            = _______;  long long *
thisESI->latest          = _______;  long long *
thisESI->state           = PAPI_RUNNING; [ or PAPI_STOPPED ]   int
-------------------------------------------------------------------
*/


return (PAPI_OK);
}
/*========================================================================*/
/*end function: PAPI_load_event                                           */
/*========================================================================*/
/*========================================================================*/

/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* int PAPI_load_dataSlotArrayElement(EventSetInfo *ESI)                  */
/*                                                                        */
/* This function determines the value of the lowestEmptySlot in the       */
/* PAPI_EVENTSET_MAP.dataSlotArray.  If the dataSlotArray is full, a call to expand is  */
/* made to papi_expandDA.  The EventSetInfo *ESI is loaded to the lowest  */
/* empty slot, and the value of ESI->EventSetIndex set to self-reference  */
/* PAPI_EVENTSET_MAP.lowestEmptySlot.                                                   */
/* bookkeeping is performed on the DynamicArray *EM to update the values  */
/* of PAPI_EVENTSET_MAP.lowestEmptySlot and PAPI_EVENTSET_MAP.availSlots.                             */
/*========================================================================*/
   
static int load_dataSlotArrayElement(EventSetInfo *ESI) 
{
  int N; /* temp value for bookkeeping */

  if (PAPI_EVENTSET_MAP.availSlots==0) _papi_expandDA();

  ESI->EventSetIndex=PAPI_EVENTSET_MAP.lowestEmptySlot;
  PAPI_EVENTSET_MAP.dataSlotArray[PAPI_EVENTSET_MAP.lowestEmptySlot] = ESI;
  
  /*bookkeeping*/
  N=PAPI_EVENTSET_MAP.lowestEmptySlot;
  while(PAPI_EVENTSET_MAP.dataSlotArray[N]!=NULL)
    N++;
  PAPI_EVENTSET_MAP.lowestEmptySlot=N;
  PAPI_EVENTSET_MAP.availSlots--;
  
  return(PAPI_OK);
}

/* There's more damn comments in this file than code! Let's go guys. */

int PAPI_set_opt(int option, int value, PAPI_option_t *ptr)
{
  switch (option)
    {
    case PAPI_SET_MPXRES:
    case PAPI_SET_OVRFLO:
      return(_papi_hwd_setopt(option,value,ptr));
    case PAPI_DEBUG:
      if ((value < PAPI_QUIET) || (value > PAPI_VERB_ESTOP)) return(PAPI_error(PAPI_EINVAL,NULL));
      PAPI_ERR_LEVEL = value;
      return(PAPI_OK);
    default:
      return(PAPI_error(PAPI_EINVAL,NULL));
    }
}

int PAPI_get_opt(int option, int *value, PAPI_option_t *ptr)
{
  switch (option)
    {
    case PAPI_GET_MPXRES:
    case PAPI_GET_OVRFLO:
      return(_papi_hwd_getopt(option,value,ptr));
    case PAPI_DEBUG:
      *value = PAPI_ERR_LEVEL;
      return(PAPI_OK);
    default:
      return(PAPI_error(PAPI_EINVAL,NULL));
    }
}
