/* file: papi.c */ 
/*file: PAPI_expandDA.c*/
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <siginfo.h>

#include "papi.h"


/*========================================================================*/
/*========================================================================*/
/*========================================================================*/
/* begin function: static int _papi_expandDA(DynamicArray *DA);           */
/*========================================================================*/

----------------------------------------------------------------------------
the _dynamic_array struct defined in papi_internal.h
typedef struct _dynamic_array {
	void   **dataSlotArray; ** ptr to array of ptrs to EventSets      **
	int    totalSlots;      ** number of slots in dataSlotArrays      **
	int    availSlots;      ** number of open slots in dataSlotArrays **
	int    lowestEmptySlot; ** index of lowest empty dataSlotArray    **
} DynamicArray;
----------------------------------------------------------------------------
The function _papi_expandDA initializes or expands the global 
data structure DA, which is type DynamicArray defined above.

The function _papi_expandDA takes one argument:
	DynamicArray *DA

If the initialization case is detected [ if(DA->totalSlots==0) ],
the DA structure is initialized.

If the expansion case is detected [ if (DA->totalSlots!=0) ],
the DA->dataSlotArray is expanded. 

The DA structure holds the dataSlotArray.  Each element of dataSlotArray  
is a pointer to an EventSetInfo structure.  The zero element of the
dataSlotArray is reserved and is set during initilization [see below].

DA->dataSlotArray[1] holds ptr to EventSetInfo number 1 
DA->dataSlotArray[2] holds ptr to EventSetInfo number 2 
DA->dataSlotArray[3] holds ptr to EventSetInfo number 3 
  ...
  ...
DA->dataSlotArray[N] holds ptr to EventSetInfo number N,
where 
	N<DA->totalSlots

The function _papi_expandDA is only called when one of
the following is detected by a function seeking to
add an EventSet to the list:

	1. initialization condition 
	2. N==DA->totalSlots

The function _papi_expandDA returns a pointer to the
expanded DA structure.

	return (1) indicates failure
	return (0) indicates success
----------------------------------------------------------------------------
*/


static int _papi_expandDA(DynamicArray *DA) {

int          prevTotal;	

if(DA->totalSlots==0) { /*initialization of first DA structure*/
	
	DA->totalSlots = 64;
	DA->availSlots = 63;
	DA->lowestEmptySlot = 1;
	DA->dataSlotArray=(void **)malloc(DA->totalSlots*sizeof(void *));

	}/***/

else	{/* expansion of existing DA structure */
	prevTotal           = DA->totalSlots; 
	DA->totalSlots     += prevTotal;
	DA->availSlots      = prevTotal;
	DA->lowestEmptySlot = prevTotal;

	/*realloc existing array*/
        if( (void **)realloc(DA->dataSlotArray,DA->totalSlots*sizeof(void *))
        ==NULL) return(1);   

        /* would anyone prefer:  return(PAPI_ENOMEM);
           instead of:           return(1);
         */  

	}/***/

return(0);

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
/* begin function: static EventSetInfo *_papi_allocate_EventSet(void);    */
/*                                                                        */
/* This function allocates space for one EventSetInfo structure and for   */
/* all of the pointers in this structure.  If any malloc in this function */
/* fails, NULL is returned.  Upon success, a pointer to the EventSetInfo  */
/* data structure is returned.                                            */
/*========================================================================*/

static EventSetInfo *_papi_allocate_EventSet(void) {

EventSetInfo *ESI;
int counterArrayLength;

counterArrayLength=_papi_system_info.num_gp_cntrs+_papi_system_info.num_sp_cntrs;

if(!ESI=(EventSetInfo *)malloc(sizeof(EventSetInfo))) return(NULL); 

ESI->EventSetIndex=0;     /*this is N index from DA->dataSlotArray[N] */
ESI->NumberOfCounters=0;  /*number of counters*/
ESI->EventCodeArray=NULL; /* array of codes for events in this set from AddEvent*/

if(!ESI->machdep)=(void *)malloc(_papi_system_info.size_machdep)) return(NULL);

if(!ESI->start =(long long *)malloc(counterArrayLength*sizeof(long long)))
	return(NULL);
if(!ESI->stop  =(long long *)malloc(counterArrayLength*sizeof(long long)))
	return(NULL);
if(!ESI->latest=(long long *)malloc(counterArrayLength*sizeof(long long)))
	return(NULL);

return(ESI);
}
/*========================================================================*/
/*end function: _papi_allocate_EventSet                                   */
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
/* If ESI==NULL	before free operation, return 0.                          */
/* Else, return 1.                                                        */
/*========================================================================*/

static int free_EventSet(EventSetInfo *ESI) {

if(!ESI) return 0; /*ESI not exist*/

if(ESI->EventCodeArray) free(ESI->EventCodeArray);
if(ESI->machdep)        free(ESI->machdep);
if(ESI->start)          free(ESI->start);
if(ESI->stop)           free(ESI->stop);
if(ESI->latest)         free(ESI->latest);

free(ESI);

return(1); /* normal return */
}
/*========================================================================*/
/*end function: _papi_free_EventSet                                       */
/*========================================================================*/
/*========================================================================*/
