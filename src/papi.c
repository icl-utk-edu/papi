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
/*  a. all memory associated with the PAPI tool is freed.                 */
/*  b. a shutdown message is written to stderr                            */
/*                                                                        */
/* Need to free:     

1. typedef struct _EventSetInfo {
  int EventSetIndex;       ** Index of the EventSet in the array  **

  int NumberOfCounters;    ** Number of counters used- usu. the number of 
                              events added **
  int *EventCodeArray;     ** PAPI/Native codes for events in this set from 
                              AddEvent **
  void *machdep;      ** A pointer to memory of size 
                         _papi_system_info.size_machdep bytes. This 
                         will contain the encoding necessary for the 
                         hardware to set the counters to the appropriate
                         conditions**
  long long *start;   ** Array of length _papi_system_info.num_gp_cntrs
                         + _papi_system_info.num_sp_cntrs 
                         This will most likely be zero for most cases**
  long long *stop;    ** Array of the same length as above, but 
                         containing the values of the counters when 
                         stopped **
  long long *latest;  ** Array of the same length as above, containing 
                         the values of the counters when last read ** 
} EventSetInfo;
*/
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
	  if(EM->dataSlotArray[i]->EventCodeArray)
	    free(EM->dataSlotArray[i]->EventCodeArray);
	  if(EM->dataSlotArray[i]->machdep)
	    free(EM->dataSlotArray[i]->machdep);
	  if(EM->dataSlotArray[i]->start)
	    free(EM->dataSlotArray[i]->start);
	  if(EM->dataSlotArray[i]->stop)
	    free(EM->dataSlotArray[i]->stop);
	  if(EM->dataSlotArray[i]->latest)
	    free(EM->dataSlotArray[i]->latest);
	  free(EM->dataSlotArray[i]);
	  }/***/
	}/***/
		 
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
/* begin function: static int PAPI_perror(int, char *, int );             */
/*                                                                        */
/* The function PAPI_perror (int, char *, int) is similar to the unix     */
/* function perror(const char *msg).                                      */
/* The function PAPI_perror gets the error description string by calling  */
/* PAPI_strerror(int code), which returns a null terminated string.       */
/*                                                                        */
/* If the calling function specifies int length greater than zero,        */
/* "length" number of characters from "code" are copied to the buffer     */
/* named "destination".  If int length equals zero, the "code" string     */
/* is both printed to stderr and copied to "destination.                  */ 
/*                                                                        */
/* The global value PAPI_ERR_LEVEL determines whether this error will     */
/* shutdown the program.                                                  */
/*========================================================================*/

int PAPI_perror (int code, char *destination, int length) {

int icode;
char buff[100];

strcpy(buff, PAPI_strerror(code));

if(length==0) {
fprintf(stderr, " %s\n", PAPI_strerror(code));
strcpy(destination,PAPI_strerror(code));
}/**/

else
strncpy(destination,PAPI_strerror(code),length);

if (PAPI_ERR_LEVEL==PAPI_VERB_STOP) {
	PAPI_shutdown();
	}
return(code);

return(1);

}/***/
/*========================================================================*/
/*end function: PAPI_error                                                */
/*========================================================================*/
/*========================================================================*/


/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function: char *PAPI_strerror (int errnum)                       */
/*========================================================================*/
/* The function PAPI_strerror(int errnum) is analogous to the unix        */
/* function strerror(int errnum).  Here, the errnum maps to a PAPI error  */
/* code defined in papi.h.  The function PAPI_strerror returns a char *   */
/* pointer to the appropriate PAPI error code.                            */
/*                                                                        */
/* This function is meant to be called by PAPI_perror(), but can also be  */
/* called directly by the user for specialized error handling.            */
/*========================================================================*/
char *PAPI_strerror (int errnum) {

int   ecode;
char *retStr;

char *errStr[11] = {
" Call to PAPI_strerror made with no error ",
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


retStr=(char *)malloc(120*sizeof(char));

/* check for valid error code */
if (ecode==1)  ecode=0;
ecode=errnum*(-1);
if (ecode>9)  ecode=10;


strcpy(retStr,errStr[ecode]);

/* check for failed C library call*/
if ( ecode==3 ) {
	strcat(retStr,strerror(errno));
	}

return(retStr);

}/* end PAPI_strerror */
/*========================================================================*/
/*end function: PAPI_strerror                                             */
/*========================================================================*/
/*========================================================================*/




/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function: static int _papi_expandDA(DynamicArray *EM);           */
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
/* Low-level machine-independent routines                                 */
/*========================================================================*/
/*========================================================================*/

int PAPI_state(int EventSet, int *status)
{
    if (EventSet < 0 || EventSet >= evmap->totalSlots || 
                          evmap->dataSlotArray[EventSet] == NULL) {
        /* invalid argument */
        if (_papi_err_level == PAPI_VERB_ECONT ||
                     _papi_err_level == PAPI_VERB_ESTOP) 
            PAPI_perror(PAPI_EINVAL, NULL, 0);
        if (_papi_err_level == PAPI_VERB_ESTOP) 
            exit(PAPI_ERROR);
        return(PAPI_EINVAL);
    }
    *status = (evmap->dataSlotArray[EventSet])->state;
    if (*status != PAPI_RUNNING && *status != PAPI_STOPPED) {
        /* internal error */
        if (_papi_err_level == PAPI_VERB_ECONT ||
                     _papi_err_level == PAPI_VERB_ESTOP) 
            PAPI_perror(PAPI_EBUG, NULL, 0);
        if (_papi_err_level == PAPI_VERB_ESTOP)
            exit(PAPI_ERROR);
        return(PAPI_EBUG);
    }
    return(PAPI_OK);
}


/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function: static EventSetInfo *papi_allocate_EventSet(void);     */
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

return(ESI);
}
/*========================================================================*/
/*end function: papi_allocate_EventSet                                    */
/*========================================================================*/
/*========================================================================*/



/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function: static int _papi_free_EventSet();                      */
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
