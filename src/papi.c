/* file: papi.c */ 

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>
#include <strings.h>

#include "papi_internal.h"
#include "papi.h"

/* There are two global variables.    */ 
/* Their values are set in PAPI_init. */  
DynamicArray PAPI_EVENT_MAP;    
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
/* Set pointer to GLOBAL variable PAPI_EVENT_MAP.                         */
/* Since PAPI_EVENT_MAP is declared at the top of the program             */
/* no malloc for EM is needed.                                            */
/* But the pointer EM->dataSlotArray must be malloced here.               */
/*                                                                        */
/* Initialize PAPI_ERR_LEVEL.                                             */
/* The user selects error handling with ERROR_LEVEL_CHOICE.               */  
/* ERROR_LEVEL_CHOICE may have one of two values:                         */
/*   a. PAPI_VERB_ECONT [print error message, then continue processing ]  */
/*   b. PAPI_VERB_ESTOP [print error message, then shutdown ]             */
/*========================================================================*/
static void PAPI_init(DynamicArray *EM, int ERROR_LEVEL_CHOICE) {

   EM=&PAPI_EVENT_MAP;
   bzero(EM,sizeof(PAPI_EVENT_MAP));

/* initialize values in PAPI_EVENT_MAP */ 

   EM->dataSlotArray=(void **)malloc(EM->totalSlots*sizeof(void *));
   if(!EM->dataSlotArray) PAPI_shutdown();
   bzero(EM->dataSlotArray,sizeof(EM->dataSlotArray));

   EM->totalSlots = PAPI_INIT_SLOTS;
   EM->availSlots = PAPI_INIT_SLOTS - 1;
   EM->lowestEmptySlot = 1;

/* initialize PAPI_ERR_LEVEL */

   if(   (ERROR_LEVEL_CHOICE!=PAPI_VERB_ECONT)
       &&(ERROR_LEVEL_CHOICE!=PAPI_VERB_ESTOP) ) 
          PAPI_shutdown(); 

   PAPI_ERR_LEVEL=ERROR_LEVEL_CHOICE;


/* from papi_internal.h
typedef struct _papi_options{
	int error_level;
	} papi_options;
*/

 	papi_options.error_level=PAPI_ERR_LEVEL;

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
/* Free the elements of EM->dataSlotArray by calling papi_freeEventSet.   */
/* The function _papi_free_EventSet frees the _EventSetInfo structure     */ 
/* in two stages:                                                         */
/*    1. Free the internal pointers.                                      */  
/*    2. Free the pointer to the _EventSetInfo structure itself.          */ 
/*                                                                        */
/* Once the EM->dataSlotArray has had all its elements removed, then      */
/* the EM->dataSlotArray itself may be freed.                             */
/* The EM pointer itself does not have to be freed because it points to   */
/* the static memory location of PAPI_EVENT_MAP.                          */
/*========================================================================*/
static void PAPI_shutdown(void) {

    int i;
    DynamicArray *EM;
    EM=&PAPI_EVENT_MAP;
    /* close all memory pointed to by xEM */
    /* this code under construction       */
    /* note: do we need to make a special case for EM->dataSlotArray[0]?*/


    /* free all the EventInfo Structures in the EM->dataSlotArray*/
    for(i=0;i<EM->totalSlots;i++) {
	if(EM->dataSlotArray[i]) {
 	  _papi_free_EventSet(EM->dataSlotArray[i]); 
	  }/* end if */
	}/* end for */ 
		 
	free(EM->dataSlotArray);



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

int PAPI_error (int PAPI_errorCode, char *errorMessage) {

int N; /* absolute value of PAPI_errorCode */

char *papi_errStr[11] = {
" Call to PAPI_error made with no error ",
" Invalid argument ",
" Insufficient memory ",
" A System C library call failed: \n\t\t",
" Substrate returned an error ",
" Access to the counters was lost or interrupted ",
" Internal error, please send mail to the developers ",
" Hardware Event does not exist ",
" Hardware Event exists, but cannot be counted due to counter resource limits ",
" No Events or EventSets are currently counting " 
" Invalid Error Code ",
};

/* check for valid error code */
N=PAPI_errorCode; 
if (N==1)  N = 0;
if (N<0)   N*= (-1);
if (N>9)   N = 10;

/* print standard papi error message */
fprintf(stderr, " %s\n", papi_errStr[N]);

/* check for failed C library call*/
if ( N==3 ) fprintf(" %s\n",strerror(errno));

/* check for user supplied error message */
if(errorMessage) fprintf(stderr, " %s\n", errorMessage);


if (PAPI_ERR_LEVEL==PAPI_VERB_STOP) {
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
/* The function _papi_expandDA expands EM->dataSlotArray when the array   */
/* has become full. The function also does all of the bookkeeping chores  */
/* [reset EM->totalSlots, EM->availSlots, EM->lowestEmptySlot].           */
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
/* EM->dataSlotArray is a pointer to an EventSetInfo structure.           */
/* The zero element of the EM->dataSlotArray is reserved and is set       */
/* during initilization, [see papi_init]                                  */
/*                                                                        */
/* EM->dataSlotArray[1] holds ptr to EventSetInfo number 1                */
/* EM->dataSlotArray[2] holds ptr to EventSetInfo number 2                */
/* EM->dataSlotArray[3] holds ptr to EventSetInfo number 3                */ 
/*  ...                                                                   */
/*  ...                                                                   */
/* EM->dataSlotArray[N] holds ptr to EventSetInfo number N,               */
/* where                                                                  */     
/*	N < EM->totalSlots                                                */
/*                                                                        */
/* The function _papi_expandDA returns PAPI_OK upon success               */
/* or PAPI_ENOMEM on failure.                                             */
/* Error handling should be done in the calling function.                 */  
/*========================================================================*/



static int _papi_expandDA(DynamicArray *EM) {

int          prevTotal;	

	/*realloc existing EM->dataSlotArray*/
        if( (void **)realloc(EM->dataSlotArray,EM->totalSlots*sizeof(void *))
        ==NULL) return(PAPI_ENOMEM);   

       
        /* bookkeeping to accomodate successful realloc operation*/
	prevTotal           = EM->totalSlots; 
	EM->totalSlots     += prevTotal;
	EM->availSlots      = prevTotal;
	EM->lowestEmptySlot = prevTotal;


	}/***/
/* 
PAPI_OK denotes no error; 
PAPI_OK_MPX denotes no error, multiplexing enabled.
*/
return(PAPI_OK);

}
/*========================================================================*/
/*end function: _papi_expandDA                                            */
/*========================================================================*/
/*========================================================================*/

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

int PAPI_state(int EventSetIndex, int *status) {

	DynamicArray *EM=&PAPI_EVENT_MAP;

 /* EventSetIndex is an integer, the index N of EM->dataSlotArray[N] */ 
 /* Check if EventSetIndex is a valid value 4 different ways. */ 

    /* 1.   invalid array index less than zero */
    /*      if (EventSetIndex < 0 ) return( PAPI_EINVAL );*/

    /* 2.   invalid array index 0, reserved for internal use only */ 
    /*      if(EM->dataSlotArray[EventSet]==0) return (PAPI_EINVAL); */

    /* 3.   invalid array index greater than highest possible value*/
    /*      if (EventSetIndex => EM->totalSlots ) return (PAPI_EINVAL); */

    /* 4.   valid array index value, but not assigned to any event yet*/
    /*      if (EM->dataSlotArray[EventSetIndex]==NULL) return(PAPI_EINVAL); */


    /* combine all of the above ifs */

    if(   (EventSetIndex < 1 )
        ||(EventSetIndex => EM->totalSlots )
	||(EM->dataSlotArray[EventSetIndex]==NULL )) return (PAPI_EINVAL);

    /* check for error in value of 
       EM->dataSlotArray[EventSetIndex]-> state, 
       which would be an internal error */

    if(   (EM->dataSlotArray[EventSetIndex])->state !=PAPI_RUNNING)
        &&(EM->dataSlotArray[EventSetIndex])->state !=PAPI_STOPPED) ) {
           /* if this error is ignored, need to set value of *status */ 
	   *status=PAPI_EINVAL;
           return(PAPI_EBUG); 
	   }
	   
    /* Good value for EM->dataSlotArray[EventSetIndex]-> state */

    *status = (EM->dataSlotArray[EventSetIndex])->state;
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

static EventSetInfo *papi_allocate_EventSet(void) {

EventSetInfo *ESI;
int counterArrayLength;

counterArrayLength=_papi_system_info.num_gp_cntrs+_papi_system_info.num_sp_cntrs;

if(!ESI=(EventSetInfo *)malloc(sizeof(EventSetInfo))) return(NULL); 
bzero(ESI,sizeof(ESI));

ESI->EventSetIndex=0;     /*this is N index from EM->dataSlotArray[N] */
ESI->NumberOfCounters=0;  /*number of counters*/

/*this ptr needs more work ******************************************************/
ESI->EventCodeArray=NULL; /* array of codes for events in this set from AddEvent*/

if(!ESI->machdep)=(void *)malloc(_papi_system_info.size_machdep)) {
	free (ESI);
	return(NULL);
	}
bzero(ESI->machdep,sizeof(ESI->machdep));

if(!ESI->start =(long long *)malloc(counterArrayLength*sizeof(long long))){
	free (ESI->machdep);
	free (ESI);
	return(NULL);
	}
bzero(ESI->start,sizeof(ESI->start));

if(!ESI->stop  =(long long *)malloc(counterArrayLength*sizeof(long long))){
	free (ESI->machdep);
	free (ESI->start);
	free (ESI);
	return(NULL);
	}
bzero(ESI->stop,sizeof(ESI->stop));

if(!ESI->latest=(long long *)malloc(counterArrayLength*sizeof(long long))){
	free (ESI->machdep);
	free (ESI->start);
	free (ESI->stop);
	free (ESI);
	return(NULL);
	}
bzero(ESI->latest,sizeof(ESI->latest));

ESI->state=PAPI_RUNNING; /*[or PAPI_STOPED]*/

return(ESI);
}
/*========================================================================*/
/*end function: papi_allocate_EventSet                                    */
/*========================================================================*/
/*========================================================================*/



/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function:                                                        */
/* static int _papi_free_EventSet();                                      */
/*                                                                        */ 
/* This function should free memory for one EventSetInfo structure.       */
/* The argument list consists of a pointer to the EventSetInfo            */
/* structure, *ESI.                                                       */
/* The calling function should check  for ESI==NULL.                      */
/*========================================================================*/

static void free_EventSet(EventSetInfo *ESI) {


if(ESI->EventCodeArray) free(ESI->EventCodeArray);
if(ESI->machdep)        free(ESI->machdep);
if(ESI->start)          free(ESI->start);
if(ESI->stop)           free(ESI->stop);
if(ESI->latest)         free(ESI->latest);

free(ESI);

return; /* normal return */
}
/*========================================================================*/
/*end function: _papi_free_EventSet                                       */
/*========================================================================*/
/*========================================================================*/
