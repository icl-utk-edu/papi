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

#define DEBUG

/* Static prototypes */

static int check_initialize(void);
static int initialize(void);
static int expand_dynamic_array(DynamicArray *);
static EventSetInfo *allocate_EventSet(void);
static int add_EventSet(EventSetInfo *);
static int add_event(EventSetInfo *ESI,int Event);
static int remove_event(EventSetInfo *ESI,int Event);
static int remove_EventSet(EventSetInfo *);
static void free_EventSet(EventSetInfo *);
static int handle_error(int, char *);
static char *get_error_string(int);
static EventSetInfo *lookup_EventSet(int eventset);
static int lookup_EventCodeIndex(EventSetInfo *ESI,int Event);

EventSetInfo *get_valid_ESI(int *EventSetIndex, int *allocated_a_new_one );

/* Global variables */
/* These will eventually be encapsulated into per thread structures. */ 

/* Our integer to EventSetInfo * mapping */

DynamicArray PAPI_EVENTSET_MAP = { 0, };    

/* Behavior of handle_error(). 
   Changed to the default behavior of PAPI_QUIET in PAPI_init
   after initialization is successful. */

int PAPI_ERR_LEVEL = PAPI_VERB_ESTOP; 

static int check_initialize(void) 
{
  /* see if initialization needed */

  if (PAPI_EVENTSET_MAP.totalSlots) 
    return(PAPI_OK);
   
  return(initialize()); 
} 

/*=+=*/ 
/*========================================================================*/
/* begin function:                                                        */    
/* static int initialize(void);                                           */
/*                                                                        */
/* This function performs all initializations to set up PAPI environment. */
/* The user selects the level of error handling here.                     */
/* Failure of this function should shutdown the PAPI tool.                */
/*                                                                        */
/* This function initializes the values in the global structure           */
/* DynamicArray PAPI_EVENTSET_MAP.                                        */
/*                                                                        */
/* The DynamicArray data structure is defined in papi_internal.h.         */
/* typedef struct _dynamic_array {                                        */
/*	void   **dataSlotArray; ** ptr to array of ptrs to EventSets      */
/*	int    totalSlots;      ** number of slots in dataSlotArrays      */
/*	int    availSlots;      ** number of open slots in dataSlotArrays */
/*	int    fullSlots;       ** number of full slots in dataSlotArrays */
/*	int    lowestEmptySlot; ** index of lowest empty dataSlotArray    */
/* } DynamicArray;                                                        */
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

static int initialize(void)
{
   memset(&PAPI_EVENTSET_MAP,0x00,sizeof(PAPI_EVENTSET_MAP));

   /* initialize values in PAPI_EVENTSET_MAP */ 

   PAPI_EVENTSET_MAP.dataSlotArray=
    (EventSetInfo **)malloc(PAPI_INIT_SLOTS*sizeof(EventSetInfo *));
   if(!PAPI_EVENTSET_MAP.dataSlotArray) 
     return(PAPI_ENOMEM);

   memset(&PAPI_EVENTSET_MAP.dataSlotArray,0x00, 
           PAPI_INIT_SLOTS*sizeof(EventSetInfo *));

   PAPI_EVENTSET_MAP.totalSlots = PAPI_INIT_SLOTS;
   PAPI_EVENTSET_MAP.availSlots = PAPI_INIT_SLOTS - 1;
   PAPI_EVENTSET_MAP.fullSlots  = 1;
   PAPI_EVENTSET_MAP.lowestEmptySlot = 1;

   PAPI_ERR_LEVEL = PAPI_QUIET;

   return(PAPI_OK);
}

/*========================================================================*/
/* This function returns the number of counters that a read(ESI->machdep)
   returns */

static int num_counters(EventSetInfo *ESI)
{
  return(_papi_system_info.num_cntrs);
}


/*========================================================================*/
/* This function returns true and the memory allocated for counters 

static unsigned long long *get_space_for_counters(EventSetInfo *ESI)
{
  int num;

  num = num_counters(ESI);
  return((unsigned long long *)malloc(num*sizeof(unsigned long long)));
}
*/

/*========================================================================*/
/* begin function:                                                        */    
/* static void PAPI_shutdown (void);                                      */
/*                                                                        */
/* This function provides a graceful exit to the PAPI tool.               */
/*  a. All memory associated with the PAPI tool is freed.                 */
/*  b. a shutdown message is written to stderr                            */ 
/*                                                                        */
/*========================================================================*/

void PAPI_shutdown(void) {

    int i;
    /* close all memory pointed to by xEM */
    /* this code under construction       */
    /* note: do we need to make a special case for 
       PAPI_EVENTSET_MAP.dataSlotArray[0]?*/


    /* free all the EventInfo Structures in the PAPI_EVENTSET_MAP.dataSlotArray*/
    for(i=0;i<PAPI_EVENTSET_MAP.totalSlots;i++) {
	if(PAPI_EVENTSET_MAP.dataSlotArray[i]) {
           /*free all memory associated with ptrs in the ESI structure*/
           /*and then free the ESI ptr itself                         */
 	   free_EventSet(PAPI_EVENTSET_MAP.dataSlotArray[i]); 
	  }/* end if */
	}/* end for */ 
		 
	free(PAPI_EVENTSET_MAP.dataSlotArray);


    /* shutdown message */
    fprintf(stderr,"\n\n PAPI SHUTDOWN. \n\n");

    return;
}


/*=+=*/ 
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
/*                                                                        */
/* Never call this with PAPI_OK                                           */
/*========================================================================*/
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

/*=+=*/ 
/*========================================================================*/
static char *get_error_string(int errorCode)
{

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

  /* set message for PAPI_OK_MPX same as PAPI_OK*/
  if(errorCode>PAPI_OK) errorCode=PAPI_OK;
  errorCode = - errorCode;

  if ((errorCode < 0) || ( errorCode >= PAPI_NUM_ERRORS))
    errorCode = PAPI_EMISC;

  return(papi_errStr[errorCode]);
}


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


/*=+=*/ 
/*========================================================================*/
/* begin function:                                                        */
/* static int _papi_expandDA(DynamicArray *EM);                           */
/*                                                                        */
/* The function _papi_expandDA expands PAPI_EVENTSET_MAP.dataSlotArray    */
/* when the array has become full.                                        */
/* The function also resets:                                              */
/*       PAPI_EVENTSET_MAP.totalSlots                                     */
/*       PAPI_EVENTSET_MAP.availSlots                                     */
/*       PAPI_EVENTSET_MAP.fullSlots (no change)                          */
/*       PAPI_EVENTSET_MAP.lowestEmptySlot                                */
/* This enables the user to load as many events as needed.                */
/*                                                                        */
/* The DynamicArray data structure is defined in papi_internal.h.         */
/* typedef struct _dynamic_array {                                        */
/*	void   **dataSlotArray; ** ptr to array of ptrs to EventSets      */
/*	int    totalSlots;      ** number of slots in dataSlotArrays      */
/*	int    availSlots;      ** number of open slots in dataSlotArrays */
/*	int    fullSlots;       ** number of full slots in dataSlotArrays */
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
    
  n = (EventSetInfo **)realloc
      (DA->dataSlotArray,DA->totalSlots*sizeof(EventSetInfo *));
  if (n==NULL) 
    return(handle_error(PAPI_ENOMEM,NULL));   

   memset(&PAPI_EVENTSET_MAP.dataSlotArray,0x00, 
           PAPI_INIT_SLOTS*sizeof(EventSetInfo *));

  /* Need to assign this value, what if realloc moved it? */
  DA->dataSlotArray = n;
       
  /* bookkeeping to accomodate successful realloc operation*/
  prevTotal           = DA->totalSlots; 
  memset(DA->dataSlotArray[prevTotal],0x00, 
           prevTotal*sizeof(EventSetInfo *));
  DA->totalSlots     += prevTotal;
  DA->availSlots      = prevTotal;
/*DA->fullSlots       = no change;*/
  DA->lowestEmptySlot = prevTotal;

  return(PAPI_OK);
}

/*=+=*/ 
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

 /* EventSetIndex is an integer, 
    the index N of PAPI_EVENTSET_MAP.dataSlotArray[N] */ 
 /* Check if EventSetIndex is a valid value 4 different ways. */ 

 /* 1.   invalid array index less than zero */
 /*      if (EventSetIndex < 0 ) 
            return( PAPI_EINVAL );*/

 /* 2.   invalid array index 0, reserved for internal use only */ 
 /*      if(PAPI_EVENTSET_MAP.dataSlotArray[EventSet]==0) 
           return (PAPI_EINVAL); */

 /* 3.   invalid array index greater than highest possible value*/
 /*      if (EventSetIndex => PAPI_EVENTSET_MAP.totalSlots ) 
            return (PAPI_EINVAL); */

 /* 4.   valid array index value, but not assigned to any event yet*/
 /*      if (PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]==NULL) 
            return(PAPI_EINVAL); */


    /* combine all of the above ifs */

    if ((EventSetIndex < 1) || 
	(EventSetIndex >= PAPI_EVENTSET_MAP.totalSlots) ||
	(PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]==NULL)) 
           return (handle_error(PAPI_EINVAL,NULL));

   /* if( PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex]->state )
      is a good value, assign *status */

    *status = (PAPI_EVENTSET_MAP.dataSlotArray[EventSetIndex])->state;
     return(PAPI_OK);
}

/*=+=*/ 
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
  int A=_papi_system_info.num_gp_cntrs;
  int B=_papi_system_info.num_sp_cntrs;

  counterArrayLength=A+B;

  ESI=(EventSetInfo *)malloc(sizeof(EventSetInfo));
  if (ESI==NULL) 
    return(NULL); 
  /*memset(&ESI,0x00,sizeof(ESI)); incorrect arg 1*/
  memset(ESI,0x00,sizeof(ESI));

  ESI->machdep=(void *)malloc(_papi_system_info.size_machdep);

  ESI->start =    (unsigned long long *)malloc(counterArrayLength*sizeof(unsigned long long));
  ESI->stop =     (unsigned long long *)malloc(counterArrayLength*sizeof(unsigned long long));
  ESI->latest =   (unsigned long long *)malloc(counterArrayLength*sizeof(unsigned long long));
  ESI->EventCodeArray = (int *)malloc(counterArrayLength*sizeof(int));

  if ((ESI->machdep        == NULL )  || 
      (ESI->start          == NULL )  || 
      (ESI->stop           == NULL )  || 
      (ESI->latest         == NULL )  ||
      (ESI->EventCodeArray == NULL ))
    {
      if (ESI->machdep)        free(ESI->machdep);
      if (ESI->start)          free(ESI->start);
      if (ESI->stop)           free(ESI->stop);
      if (ESI->latest)         free(ESI->latest);
      if (ESI->EventCodeArray) free(ESI->EventCodeArray);
      free(ESI);
      return(NULL);
    }
  memset(ESI->machdep,       0x00,_papi_system_info.size_machdep);
  memset(ESI->start,         0x00,counterArrayLength*sizeof(unsigned long long));
  memset(ESI->stop,          0x00,counterArrayLength*sizeof(unsigned long long));
  memset(ESI->latest,        0x00,counterArrayLength*sizeof(unsigned long long));
  memset(ESI->EventCodeArray,0x00,counterArrayLength*sizeof(int));

  ESI->state = PAPI_STOPPED; 

  return(ESI);
}

/*=+=*/ 
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

/*=+=*/ 
/*========================================================================*/
/* begin function:                                                        */
/* static int add_event(EventSetInfo *ESI, int EventCode)                 */
/*                                                                        */ 
/* This function is called by PAPI_add_event and by PAPI_add_events.      */ 
/* Those functions take care of checking for valid *ESI and valid         */ 
/* EventCode.                                                             */ 
/*                                                                        */ 
/* This function adds the EventCode to the "lowest open slot" in the      */ 
/* ESI->EventCodeArray and sets the corresponding counters for start,     */ 
/* stop, and latest equal to zero.                                        */ 
/*                                                                        */ 
/* If no open slot is found, PAPI_ECNFLCT is returned.                    */ 
/* PAPI_ECNFLCT means HardwareEvent exists but cannot be counted due to   */ 
/* counter resource limitations.                                          */ 
/*========================================================================*/
static int add_event(EventSetInfo *ESI, int EventCode) 
{
  int k;

  /*set EventCode value to 0 because this is value in open slot*/
  k=lookup_EventCodeIndex(ESI,0);

  if(k==PAPI_NULL) { /* no open slot found */

  return(PAPI_ECNFLCT);
  }

  /* add the eventCode and initialize its counters*/
  ESI->EventCodeArray[k] = EventCode;
  ESI->start[k]          = 0;
  ESI->stop[k]           = 0;
  ESI->latest[k]         = 0;

  ESI->NumberOfCounters++;

  return(PAPI_OK); 
}

/*=+=*/ 
/*========================================================================*/
/* begin function:                                                        */
/* static int remove_event(EventSetInfo *ESI, int EventCode)              */
/*                                                                        */ 
/* This function is called by PAPI_rem_event, which does the checking     */ 
/* for valid *ESI and valid EventCode.                                    */ 
/*                                                                        */ 
/* This function determines the index k associated with EventCode and     */ 
/* resets the values of EventCodeArray, start, stop, latest for the       */ 
/* elements at index k to zero.  This is the same as the initial values   */ 
/* when the EventSetInfo structure was first allocated.                   */ 
/*                                                                        */ 
/*========================================================================*/
static int remove_event(EventSetInfo *ESI, int EventCode) 
{
  int k;

  /* determine index of target event k */

  k = lookup_EventCodeIndex(ESI,EventCode);
  if (k < 0)
    return(PAPI_EINVAL);

  ESI->EventCodeArray[k] = 0;
  ESI->start[k]          = 0;
  ESI->stop[k]           = 0;
  ESI->latest[k]         = 0;
  ESI->NumberOfCounters--;

  return(PAPI_OK); 
}

/*=+=*/ 
/*========================================================================*/
/* begin function:                                                        */
/* EventSetInfo *get_valid_ESI                                            */ 
/*              (int *EventSetIndex,int *allocated_a_new_one)             */
/*                                                                        */
/* This function determines if the first argument (int *EventSetIndex)    */ 
/* refers to an existing PAPI_EVENTSET_MAP.dataSlotArray[*EventSetIndex]  */
/* exists or if it must be created.                                       */
/*                                                                        */ 
/* If a valid *ESI exists for *EventSetIndex, the ptr to this ESI         */ 
/* is returned.                                                           */ 
/*                                                                        */ 
/* If the *ESI needs to be created, the function attempts to create it.   */ 
/* Upon successful creation, the ptr to new ESI is returned.              */
/* Upon failure, null is returned.                                        */ 
/*========================================================================*/
EventSetInfo *get_valid_ESI(int *EventSetIndex, int *allocated_a_new_one )
{
 int errorCode;
 EventSetInfo *ESI;
    
        /*check for initialization*/
        errorCode=check_initialize();
        if(errorCode<PAPI_OK)
           return(NULL);

      /* check for pre-existing ESI*/
      ESI = lookup_EventSet(*EventSetIndex);

      if ( ESI ){/*found it*/
         *allocated_a_new_one=0;
         return(ESI);
         }

      else {/*need new one*/
      ESI = allocate_EventSet();
            if (!ESI) {/*allocate_EventSet failed*/
            *allocated_a_new_one=0; 
             return(NULL);
             }
      *allocated_a_new_one=1;
      return(ESI);
    }

}

/*========================================================================*/
/* begin function:                                                        */
/* int d_handler(char *errorName,int errorNum, int allocated_a_new_one)   */
/*                                                                        */
/* This function care of "damnit:" code and is called by                  */
/*    PAPI_add_event                                                      */
/*    PAPI_add_events                                                     */
/*    PAPI_add_pevent                                                     */
/*                                                                        */
/*========================================================================*/
int d_handler(char *errorName, int errorNum, 
      int allocated_a_new_one,EventSetInfo *ESI) 
{

   /*damnit:*/

    return(PAPI_EINVAL);
}


/*=+=*/ 
/*========================================================================*/
/* begin function:                                                        */
/* int PAPI_add_event(int *EventSetIndex, int EventCode)                  */
/*                                                                        */
/* This function determines if the first argument (int *EventSetIndex)    */ 
/* refers to an existing PAPI_EVENTSET_MAP.dataSlotArray[*EventSetIndex]  */
/* exists or if it must be created by calling get_valid_ESI. If new       */
/* ESI created, allocated_a_new_one set to equal one.                     */ 
/*                                                                        */ 
/* Once a valid target ESI is in place, this function determines if       */
/* the hardware event designated by argument 2 (EventCode) is supported   */
/* by this platform and if it is not in conflict with other EventCodes    */
/* already being counted with this ESI. EventCode must hava a value       */
/* corresponding to one of the standard event definitions [PAPI_L1_ICM,   */ 
/* PAPI_TOT_CYC, etc.] To add a user defined event, use add_pevent.       */ 
/*                                                                        */ 
/* This function returns PAPI_ENOEVNT if an event cannot be counted       */ 
/* on this platform.                                                      */ 
/*                                                                        */ 
/* The attempt to add a conflicting event to an event set will cause      */ 
/* return of error unless PAPI_SETA_MPXRES has been set.                  */ 
/*                                                                        */ 
/* A value of 0 for *EventSetIndex may not be used. This has been         */ 
/* reserved for internal use.                                             */ 
/*                                                                        */ 
/* Given a good value for argument 2 (EventCode), this function adds      */
/* counters for the designated hardware event to the target ESI.          */ 
/*                                                                        */ 
/*========================================================================*/

int PAPI_add_event(int *EventSetIndex, int EventCode) 
{
  int add_errorCode,hwd_errorCode,glo_errorCode,esi_errorCode;
  int allocated_a_new_one = 0;
  EventSetInfo *ESI;

  /* initialize error codes */
  add_errorCode=PAPI_OK; /*for add_event          */
  esi_errorCode=PAPI_OK; /*for get_valid_ESI      */
  hwd_errorCode=PAPI_OK; /*for _papi_hwd_add_event*/
  glo_errorCode=PAPI_OK; /*for add_Event_set      */ 


  if ( EventSetIndex == NULL ) 
    return(handle_error(PAPI_EINVAL,NULL));

  ESI=get_valid_ESI(EventSetIndex,&allocated_a_new_one);
  if(!ESI) {
     esi_errorCode=PAPI_ENOMEM;
     goto damnit;
     }

  /*=+=*/
  /* Now we have a valid ESI, try _papi_hwd_add_event with eventCode */
  hwd_errorCode = _papi_hwd_add_event(ESI->machdep,EventCode);

   /* if good _papi_hwd_add_event, see if it can be added to the eventSet*/
   /* Update the machine Independent information                         */

   if ( hwd_errorCode >= PAPI_OK) {

       add_errorCode=add_event(ESI,EventCode);

   /*if good errorCode for add_event, and allocated_a_new_one =1, 
         need to add this ESI to the global table. */ 

         if(   (add_errorCode >= PAPI_OK )
            && (allocated_a_new_one==1) ) {
            glo_errorCode=add_EventSet(ESI);
            if (glo_errorCode<PAPI_OK) goto damnit;
            }

         /* if you get to here, everything has worked*/
         /* Always return the hwd_errorCode from hwd ops */

            return(hwd_errorCode);/*leave with success*/

     }/*end if ( hwd_errorCode >= PAPI_OK */


    /* to be replaced by call to function d_handler */
    damnit:      
     /*     not good get_valid_ESI
        or  not good _papi_hwd_add_event 
        or  not good add_event
        or  not good add_EventSet */
	
    /* If we've allocated a new ESI, we must free it */
    if (allocated_a_new_one)
       free_EventSet(ESI);  
        
    /* Always return the hwd_errorCode from hwd ops */
    /* add message string to further define error   */
    if(esi_errorCode<PAPI_OK)
        return(handle_error(PAPI_ENOMEM,"failure on get_valid_ESI call"));
    if(add_errorCode<PAPI_OK) 
        return(handle_error(hwd_errorCode,"failure on add_event call"));
    if(glo_errorCode<PAPI_OK)
        return(handle_error(hwd_errorCode,"failure on add_EventSet call"));
    /* error in hwd_errorCode*/
    return(handle_error(hwd_errorCode,"failure on _papi_hwd_add_event call"));

}

/*=+=*/ 
/*========================================================================*/
/* begin function:                                                        */
/* int PAPI_add_events(int *EventSetIndex, int *EventCodes, int number)   */
/*                                                                        */
/* This function determines if the first argument (int *EventSetIndex)    */
/* refers to an existing PAPI_EVENTSET_MAP.dataSlotArray[*EventSetIndex]  */
/* exists or if it must be created.                                       */
/*                                                                        */
/* Once a valid target ESI is in place, this function goes through the    */
/* loop "number" of times to add EventCodes[k] to  ESI->EventCodeArray.   */
/* If one or more of Events cannot be counted on this platform, then this */
/* call fails and PAPI_ENOEVNT is returned after _all_ of the values in   */
/* the EventCodes array have been checked. In addition, the invalid       */
/* entries in the EventCodes array are set to PAPI_NULL such that the     */
/* user can successfully reissue the call with the revised EventCodes.    */
/*========================================================================*/

int PAPI_add_events(int *EventSetIndex, int *EventCodes, int number) 
{
  int add_errorCode,hwd_errorCode,HWD_errorCode,glo_errorCode,esi_errorCode;
  int allocated_a_new_one = 0;
  EventSetInfo *ESI;
  int  i;

  /* initialize error codes */
  add_errorCode=PAPI_OK; /*for add_event                     */
  esi_errorCode=PAPI_OK; /*for get_valid_ESI                 */
  hwd_errorCode=PAPI_OK; /*for individual _papi_hwd_add_event*/
  HWD_errorCode=PAPI_OK; /*for aggregate  _papi_hwd_add_event*/
  glo_errorCode=PAPI_OK; /*for add_Event_set                 */ 

  if ( EventSetIndex == NULL ) 
    return(handle_error(PAPI_EINVAL,NULL));

  if ( number <= 0)
    return(handle_error(PAPI_EINVAL,NULL));

  ESI=get_valid_ESI(EventSetIndex,&allocated_a_new_one);
  if(!ESI) {
     esi_errorCode=PAPI_ENOMEM;
     goto damnit;
     }

  /* Now we have a valid ESI, try to add HW events    */
  /* If any Events[i]==NULL, skip with no error       */
  /* If any Events[i] are invalid, all_events_valid=0 */  
 
  for ( i=0; i<number; i++ ) 
    {
      if ( EventCodes[i] != PAPI_NULL ) /* only act if not PAPI_NULL */
	{
	  hwd_errorCode = _papi_hwd_add_event(ESI->machdep,EventCodes[i]);
	  if (hwd_errorCode < PAPI_OK) 
	    {
	      EventCodes[i]  = PAPI_NULL;
              HWD_errorCode   = hwd_errorCode; /* only gets last bad one*/ 
	    }
	}
    }


  if (HWD_errorCode == PAPI_OK) { /* all _papi_hwd_add_event good */

  /* add the EventCodes to the ESI->EventCodeArray */
  /* Update the machine Independent information */

   for ( i=0; i<number; i++ ){
     add_errorCode=add_event(ESI,EventCodes[i]);
     if(add_errorCode<PAPI_OK)
       goto damnit;
     } 
  
  /* If new ESI, we need to insert it into the global table */
  
  if (allocated_a_new_one)
    {
      glo_errorCode = add_EventSet(ESI);
      if (glo_errorCode < PAPI_OK)
	goto damnit;
    }

   /* if you got to here, it all worked.      */
  /* Always return the errorCode from hwd ops */

  return(hwd_errorCode);

  }/* end if (HWD_errorCode == PAPI_OK) */


    /* to be replaced by call to function d_handler */
    damnit:      
     /*     not good get_valid_ESI
        or  not good _papi_hwd_add_event 
        or  not good add_event
        or  not good add_EventSet */
	
    /* If we've allocated a new ESI, we must free it */
    if (allocated_a_new_one)
       free_EventSet(ESI);  
        
    /* Always return the hwd_errorCode from hwd ops */
    /* add message string to further define error   */
    if(esi_errorCode<PAPI_OK)
        return(handle_error(PAPI_ENOMEM,"failure on get_valid_ESI call"));
    if(add_errorCode<PAPI_OK) 
        return(handle_error(hwd_errorCode,"failure on add_event call"));
    if(glo_errorCode<PAPI_OK)
        return(handle_error(hwd_errorCode,"failure on add_EventSet call"));
    /* error in HWD_errorCode*/
    return(handle_error(HWD_errorCode,"failure on _papi_hwd_add_event call"));


}

/*=+=*/ 
/*========================================================================*/
/* low-level function:                                                    */
/* static int PAPI_add_pevent(int *EventSetIndex, int p_code, void *inout)*/
/*                                                                        */
/* This function determines if the first argument (int *EventSetIndex)    */
/* refers to an existing PAPI_EVENTSET_MAP.dataSlotArray[*EventSetIndex]  */
/* exists or if it must be created.                                       */
/*                                                                        */
/*
    from the papi draft standard:
    int PAPI_add_event(int *EventSet, int Event)

   PAPI_add_pevent() may be implemented as a call to the
   PAPI_set_opt().
*/

int PAPI_add_pevent(int *EventSetIndex, int p_code, void *inout)
{
  int add_errorCode,hwd_errorCode,glo_errorCode,esi_errorCode;
  int allocated_a_new_one = 0;
  EventSetInfo *ESI;

  /* initialize error codes */
  add_errorCode=PAPI_OK; /*for add_event                     */
  esi_errorCode=PAPI_OK; /*for get_valid_ESI                 */
  hwd_errorCode=PAPI_OK; /*for individual _papi_hwd_add_event*/
  glo_errorCode=PAPI_OK; /*for add_Event_set                 */ 

  if ( EventSetIndex == NULL ) 
    return(handle_error(PAPI_EINVAL,NULL));

  if ( p_code <= 0)
    return(handle_error(PAPI_EINVAL,NULL));

  /* make sure there is a valid ESI for this EventSetIndex*/
  ESI=get_valid_ESI(EventSetIndex,&allocated_a_new_one);
  if(!ESI) {
     esi_errorCode=PAPI_ENOMEM;
     goto damnit;
     }

  /* Now we have a valid ESI, try _papi_hwd_add_prog_event with p_code */
  hwd_errorCode = _papi_hwd_add_prog_event(ESI->machdep,p_code,inout);

   /* if good _papi_hwd_add_event, see if it can be added to the eventSet*/
   /* Update the machine Independent information                         */

   if ( hwd_errorCode >= PAPI_OK) {

       add_errorCode=add_event(ESI,p_code);

   /*if good errorCode for add_event, and allocated_a_new_one =1,
         need to add this ESI to the global table. */

         if(   (add_errorCode >= PAPI_OK )
            && (allocated_a_new_one==1) ) {
            glo_errorCode=add_EventSet(ESI);
            if (glo_errorCode<PAPI_OK) goto damnit;
            }

         /* if you get to here, everything has worked*/
         /* Always return the hwd_errorCode from hwd ops */

            return(hwd_errorCode);/*leave with success*/

     }/*end if ( hwd_errorCode >= PAPI_OK */

    /* to be replaced by call to function d_handler */
    damnit:
     /*     not good get_valid_ESI
        or  not good _papi_hwd_add_event
        or  not good add_event
        or  not good add_EventSet */

    /* If we've allocated a new ESI, we must free it */
    if (allocated_a_new_one)
       free_EventSet(ESI);

    /* Always return the hwd_errorCode from hwd ops */
    /* add message string to further define error   */
    if(esi_errorCode<PAPI_OK)
        return(handle_error(PAPI_ENOMEM,"failure on get_valid_ESI call"));
    if(add_errorCode<PAPI_OK)
        return(handle_error(hwd_errorCode,"failure on add_event call"));
    if(glo_errorCode<PAPI_OK)
        return(handle_error(hwd_errorCode,"failure on add_EventSet call"));
    /* error in hwd_errorCode*/
    return(handle_error(hwd_errorCode,"failure on _papi_hwd_add_event call"));

}

 

/*=+=*/ 
/*========================================================================*/
/* low-level function:                                                    */
/* int PAPI_rem_event(int EventSet, int Event)                            */ 

/* from the draft standard:
   This function removes the hardware counter Event from EventSet.
*/

int PAPI_rem_event(int EventSet, int Event)
{
  EventSetInfo *ESI;
  int errorCode, hwd_errorCode;

  /* determine target ESI structure */

  ESI = lookup_EventSet(EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  if (ESI->NumberOfCounters < 1)
    return(handle_error(PAPI_EINVAL,"No counters present"));

  /* Remove Event from machine independent structures */

  errorCode = remove_event(ESI,Event);
  if (errorCode < PAPI_OK) 
    return(handle_error(errorCode,NULL));

  /* Remove Event from machine dependent structures */

  hwd_errorCode = _papi_hwd_rem_event(ESI->machdep,Event);
  if (hwd_errorCode < PAPI_OK)
    return(handle_error(hwd_errorCode,NULL));

  if (ESI->NumberOfCounters == 0)
    remove_EventSet(ESI); 

  /* Always return the hwd_errorCode from hwd ops */

  return(errorCode);
}

/*=+=*/ 
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
  int i, retval;

  /* determine target ESI structure */

  ESI = lookup_EventSet(EventSet);
  if (( ESI == NULL ) || (number <= 0))
    return(handle_error(PAPI_EINVAL,NULL));

  for (i=0; i<number;i++)
    {
      retval = PAPI_rem_event(EventSet,RemEvents[i]);
      if (retval < PAPI_OK)
	return(retval); /* Errors are handled by rem_event */
    }

  return(retval);
}

/*=+=*/ 
/*========================================================================*/
/* low-level function:                                                    */
/* static int PAPI_list_events(int EventSet,int *EventCodes,int *number)  */
/*
   from the draft standard:

   This function decomposes EventSet into the hardware
   Events it contains. number is both an input and output.

   ---------------------------------------------------------------------
   number as input:  total of all events ever added [active + inactive ]
   number as output: total of all active events at this time
*/
/*========================================================================*/

int PAPI_list_events(int EventSet, int *EventCodes, int *number)
{
   EventSetInfo *ESI;
   int i,nActive;

  /* determine target ESI structure */
  ESI=lookup_EventSet(EventSet);
  if ( ESI == NULL )
      return(handle_error(PAPI_EINVAL,NULL));

  nActive=0;/*count number of active events*/

  for(i=0;i<*number;i++) {
     EventCodes[i]=ESI->EventCodeArray[i];
     if(EventCodes[i]>0)nActive++;
  }/* end for i */

  *number=nActive;
  
   return(PAPI_OK);
}

/*=+=*/ 
/*========================================================================*/
/* phil's handy phunction                                                 */
/* This function takes a target EventSetInfo *ESI and a designated        */
/* EventCode [a value like PAPI_FP_INS, PAPI_TOT_CYC, etc.] and           */
/* determines if this EventCode has been loaded into the EventCodeArray   */
/* for the target *ESI.  If yes, the first index where the designated     */
/* EventCode resides is returned.  If a single EventCode has been loaded  */
/* more than once into an EventCodeArray, only the first instance will    */
/* be detected by this function.                                          */
/* If NO instance of the designated EventCode is detected, a value of     */
/* PAPI_NULL ( =-1 ) is returned.                                         */
/*========================================================================*/

static int lookup_EventCodeIndex(EventSetInfo *ESI,int EventCode)
{
  int i;

  for(i=0;i<ESI->NumberOfCounters;i++) 
    {
      if (ESI->EventCodeArray[i]==EventCode) 
	return(i);
    }

  return(PAPI_NULL);
} 

/*========================================================================*/
static EventSetInfo *lookup_EventSet(int eventset)
{
  if ((eventset > 1) && (eventset < PAPI_EVENTSET_MAP.totalSlots))
    return(PAPI_EVENTSET_MAP.dataSlotArray[eventset]);
  else
    return(NULL);
}





/*=+=*/ 
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
  PAPI_EVENTSET_MAP.fullSlots++; 
 
  return(PAPI_OK);
}


/*=+=*/ 
/*========================================================================*/
/* low-level function:                                                    */
/* static int remove_EventSet(int eventset)                               */
/* eventset is the index of the ESI in the dataSlotArray                  */
/*                                                                        */
/* This function is called by PAPI_rem_event when the value for           */
/* ESI->NumberOfCounters goes to zero.                                    */
/*========================================================================*/
static int remove_EventSet(EventSetInfo *ESI)
{

   int I;

   /* get value of Index I for this ESI in 
      PAPI_EVENTSET_MAP.dataSlotArray[I]    */
      I=ESI->EventSetIndex;
   

   /* free all the memory from the target EventSet*/ 
      free_EventSet(ESI);


   /* do bookkeeping for PAPI_EVENTSET_MAP */
        PAPI_EVENTSET_MAP.dataSlotArray[I]=NULL;
     if(PAPI_EVENTSET_MAP.lowestEmptySlot < I)
        PAPI_EVENTSET_MAP.lowestEmptySlot = I;
        PAPI_EVENTSET_MAP.availSlots++;
        PAPI_EVENTSET_MAP.fullSlots--;

  return(PAPI_OK);
}



/*=+=*/ 
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
static int set_multiplex(PAPI_option_t *ptr)
{
  _papi_int_option_t internal_option;
  int retval;

  internal_option.multiplex.ESI = lookup_EventSet(ptr->multiplex.eventset);
  if (!internal_option.multiplex.ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  memcpy(&internal_option.multiplex.multiplex,&ptr->multiplex,sizeof(PAPI_multiplex_option_t));
  retval = _papi_hwd_ctl(PAPI_SET_MPXRES,&internal_option);
  if (retval < PAPI_OK)
    return(retval);

  memcpy(&internal_option.multiplex.ESI->all_options.multiplex,
	 &internal_option.multiplex,sizeof(_papi_int_multiplex_t));
  return(PAPI_OK);
  
}

/*========================================================================*/
static int overflow_is_active(EventSetInfo *ESI)
{
  if (ESI->all_options.overflow.eventindex >= 0) /* No overflow active for this EventSet */
    return(1);
  else
    return(0);
}

/*========================================================================*/
static int set_overflow(PAPI_option_t *ptr)
{
  int retval, ind;
  _papi_int_option_t internal_option;

  internal_option.overflow.ESI = lookup_EventSet(ptr->overflow.eventset);
  if (!internal_option.overflow.ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  ind = event_is_in_eventset(ptr->overflow.event, internal_option.overflow.ESI);
  if (ind < PAPI_OK)
    return(ind);

  if (ptr->overflow.threshold > 0)
    if (!ptr->overflow.handler)
      return(handle_error(PAPI_EINVAL,"Overflow handler not specified"));
					
  /* Args are good. Is overflow active? */

  if ((!overflow_is_active(internal_option.overflow.ESI)) && (ptr->overflow.threshold == 0))
    return(PAPI_OK);
    
  retval = _papi_hwd_ctl(PAPI_SET_OVRFLO,&internal_option);
  if (retval < 0)
    return(retval);

  /* ESI->overflow.eventindex = ind;
  ESI->overflow.deadline = ;
  ESI->overflow.milliseconds =;
  memcpy(&ESI->overflow.option,&ptr->overflow,sizeof(ptr->overflow)) */
  return(PAPI_OK);
}

static int get_overflow(PAPI_option_t *ptr)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(ptr->overflow.eventset);
  if (!ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  memcpy(&ptr->overflow,&ESI->all_options.overflow,sizeof(ESI->all_options.overflow));

  return(PAPI_OK);
}

static int get_multiplex(PAPI_option_t *ptr)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(ptr->multiplex.eventset);
  if (!ESI)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  memcpy(&ptr->multiplex,&ESI->all_options.multiplex,sizeof(ESI->all_options.multiplex));

  return(PAPI_OK);
}

static int set_domain(PAPI_option_t *ptr)
{
  _papi_int_option_t opt;
  int arg, retval;

  /* Check the args */

  if (ptr == NULL)
    return(handle_error(PAPI_EINVAL,"Invalid option pointer"));

  arg = ptr->domain.domain;

  if ((arg < PAPI_DOM_MIN) && (arg > PAPI_DOM_MAX))
    return(handle_error(PAPI_EINVAL,"Invalid domain"));

  /* Lookup the eventset */
  
  opt.domain.ESI = lookup_EventSet(ptr->domain.eventset);
  if (opt.domain.ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));
  
  /* Fill in the internal structure */
  
  memcpy(&opt.domain.domain,&ptr->domain,sizeof(PAPI_domain_option_t));
  
  /* Do the low level call */

  retval = _papi_hwd_ctl(PAPI_SET_DOMAIN,&opt);
  if (retval < PAPI_OK)
    return(retval);

  /* Store this ESI's options if the above is successful */
  
  memcpy(&opt.domain.ESI->all_options.domain,&opt.domain,sizeof(_papi_int_domain_t));
  return(PAPI_OK);
}

static int get_defdomain(PAPI_option_t *ptr)
{
  return(PAPI_OK);
}

static int get_defgranularity(PAPI_option_t *ptr)
{
  return(PAPI_OK);
}

static int get_granularity(PAPI_option_t *ptr)
{
  return(PAPI_OK);
}

static int get_domain(PAPI_option_t *ptr)
{
  EventSetInfo *ESI;

  /* Check the args */

  if (ptr == NULL)
    return(handle_error(PAPI_EINVAL,"Invalid option pointer"));

  /* Lookup the eventset */
  
  ESI = lookup_EventSet(ptr->domain.eventset);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));
  
  /* Copy this ESI's options if the above is successful */
  
  memcpy(&ptr->domain,&ESI->all_options.domain.domain,sizeof(PAPI_domain_option_t));

  return(PAPI_OK);
}

static int set_defdomain(PAPI_option_t *ptr)
{
  return(PAPI_OK);
}

static int set_granularity(PAPI_option_t *ptr)
{
  _papi_int_option_t opt;
  int arg, retval;

  /* Check the args */

  if (ptr == NULL)
    return(handle_error(PAPI_EINVAL,"Invalid option pointer"));

  arg = ptr->granularity.granularity;

  if ((arg < PAPI_GRN_MIN) && (arg > PAPI_GRN_MAX))
    return(handle_error(PAPI_EINVAL,"Invalid granularity"));

  /* Lookup the eventset */
  
  opt.granularity.ESI = lookup_EventSet(ptr->granularity.eventset);
  if (opt.granularity.ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));
  
  /* Fill in the internal structure */
  
  memcpy(&opt.granularity.granularity,&ptr->granularity,sizeof(PAPI_granularity_option_t));
  
  /* Do the low level call */

  retval = _papi_hwd_ctl(PAPI_SET_GRANUL,&opt);
  if (retval < PAPI_OK)
    return(retval);

  /* Store this ESI's options if the above is successful */
  
  memcpy(&opt.granularity.ESI->all_options.granularity,&opt.granularity,sizeof(_papi_int_granularity_t));
  return(PAPI_OK);
}

static int set_defgranularity(PAPI_option_t *ptr)
{
  return(PAPI_OK);
}

int PAPI_set_granularity(int granularity)
{ 
  PAPI_option_t opt;

  opt.defgranularity.granularity = granularity;
  return(PAPI_set_opt(PAPI_SET_DEFGRN,&opt));
}

int PAPI_set_domain(int domain)
{ 
  PAPI_option_t opt;

  opt.defdomain.domain = domain;
  return(PAPI_set_opt(PAPI_SET_DEFDOM,&opt));
}

int PAPI_set_opt(int option, PAPI_option_t *ptr)
{
  int errorCode;

  /* check for initialization */

  errorCode = check_initialize();
  if (errorCode < PAPI_OK)
    return(handle_error(errorCode,NULL));

  switch (option)
    {
    case PAPI_SET_DEFDOM:
      return(set_defdomain(ptr));
    case PAPI_SET_DOMAIN:
      return(set_domain(ptr));
    case PAPI_SET_DEFGRN:
      return(set_defgranularity(ptr));
    case PAPI_SET_GRANUL:
      return(set_granularity(ptr));
    case PAPI_SET_MPXRES:
      return(set_multiplex(ptr)); 
    case PAPI_SET_OVRFLO:
      return(set_overflow(ptr));
    case PAPI_DEBUG:
      if ((ptr->debug < PAPI_QUIET) || (ptr->debug > PAPI_VERB_ESTOP)) 
	return(handle_error(PAPI_EINVAL,NULL));
      PAPI_ERR_LEVEL = ptr->debug;
      return(PAPI_OK);
    default:
      return(handle_error(PAPI_EINVAL,"No such option"));
    }
}

int PAPI_get_opt(int option, PAPI_option_t *ptr)
{
  int errorCode;

  if (ptr == NULL)
    return(handle_error(PAPI_EINVAL,"Invalid pointer"));

  errorCode = check_initialize();
  if (errorCode<PAPI_OK)
    return(handle_error(errorCode,NULL));

  switch (option)
    {
    case PAPI_GET_MPXRES:
      return(get_multiplex(ptr)); 
    case PAPI_GET_OVRFLO:
      return(get_overflow(ptr));
    case PAPI_GET_DOMAIN:
      return(get_domain(ptr));
    case PAPI_GET_DEFDOM:
      return(get_defdomain(ptr));
    case PAPI_GET_GRANUL:
      return(get_granularity(ptr));
    case PAPI_GET_DEFGRN:
      return(get_defgranularity(ptr));
    case PAPI_DEBUG:
      ptr->debug = PAPI_ERR_LEVEL;
      return(PAPI_OK);
    default:
      return(handle_error(PAPI_EINVAL,"Invalid option"));
    }
}

/*========================================================================*/

int PAPI_start(int EventSet)
{ 
  int retval;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL) 
    return(handle_error(PAPI_EINVAL, NULL));

  retval = _papi_hwd_start(ESI);
  if(retval<PAPI_OK) 
    return(handle_error(retval, NULL));

  return(retval);
}

/*========================================================================*/

int PAPI_stop(int EventSet, unsigned long long *values)
{ 
  int retval, i;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if (ESI==NULL) 
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  retval = _papi_hwd_stop(ESI->machdep, values);
  if (retval<PAPI_OK) 
    return(handle_error(retval, NULL));

#ifdef DEBUG
  {
    int bound;
    bound = num_counters(ESI);

    for(i=0; i<bound; i++)
      { 
	printf("DEBUG: Counter %d : %lld\n", i, values[i]);
      }
  }
#endif

  retval = _papi_hwd_reset(ESI->machdep);
  if (retval<PAPI_OK) 
    return(handle_error(retval, NULL));

  return(retval);
}

/*========================================================================*/

int PAPI_read(int EventSet, unsigned long long *values)
{ 
  int retval, i;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  retval = _papi_hwd_read(ESI->machdep, values);
  if (retval<PAPI_OK) 
    return(handle_error(retval, NULL));

#ifdef DEBUG
  {
    int bound;
    bound = num_counters(ESI);

    for(i=0; i<bound; i++)
      { 
	printf("DEBUG: Counter %d : %lld\n", i, values[i]);
      }
  }
#endif

  return(retval);
}

/*=+=*/ 
/*========================================================================*/

int PAPI_accum(int EventSet, unsigned long long *values)
{ 
  EventSetInfo *ESI;
  int retval, i, bound;
  unsigned long long a,b,c,*increase;

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

/*=+=*/ 
/*========================================================================*/
int PAPI_write(int EventSet, unsigned long long *values)
{ int retval;
  EventSetInfo *ESI; 

  ESI = lookup_EventSet(EventSet);
  if ( ESI == NULL )
    return(handle_error(PAPI_EINVAL,NULL));

  retval = _papi_hwd_write(ESI->machdep, values);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));
  return(retval);
}

/*=+=*/ 
/*========================================================================*/
int PAPI_reset(int EventSet)
{ int retval;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL) return(handle_error(PAPI_EINVAL, NULL));

  retval = _papi_hwd_reset(ESI->machdep);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));
  return(retval);
}


/*========================================================================
  This function calls the setopt with the option to set the default granularity
  it doesn't operate on an EventSet. */

/* int PAPI_set_granularity(int granularity)
{ int retval;

  retval = PAPI_set_opt(PAPI_SET_DEFGRN,granularity,NULL); 
  if(retval<PAPI_OK) return(handle_error(retval, NULL));
  return(retval);
}

int PAPI_set_domain(int domain)
{ int retval;

  retval = PAPI_set_opt(PAPI_SET_DEFDOM,domain,NULL);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));
  return(retval);
} */
