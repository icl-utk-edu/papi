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

static int num_counters(EventSetInfo *);
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
static int lookup_EventCodeIndex(EventSetInfo *ESI,int EventCode);

/* Global variables */
/* These will eventually be encapsulated into per thread structures. */ 

/* Our integer to EventSetInfo * mapping */

static DynamicArray PAPI_EVENTSET_MAP;    

/* Behavior of handle_error(). 
   Changed to the default behavior of PAPI_QUIET in PAPI_init
   after initialization is successful. */

static int PAPI_ERR_LEVEL = PAPI_VERB_ESTOP; 

/* Utility functions */

static int num_counters(EventSetInfo *ESI)
{
  return(ESI->NumberOfCounters);
}

static int check_initialize(void) 
{
  /* see if initialization needed */

  if (PAPI_EVENTSET_MAP.totalSlots) 
    return(PAPI_OK);
   
  return(initialize()); 
} 

static int initialize(void)
{
  int retval;
  EventSetInfo *zero;

  /* Clear the Dynamic Array structure */

  memset(&PAPI_EVENTSET_MAP,0x00,sizeof(PAPI_EVENTSET_MAP));
   
  /* Allocate space for the EventSetInfo pointers */

  PAPI_EVENTSET_MAP.dataSlotArray =
    (EventSetInfo **)malloc(PAPI_INIT_SLOTS*sizeof(EventSetInfo *));
  if(!PAPI_EVENTSET_MAP.dataSlotArray) 
    return(PAPI_ENOMEM);
  memset(PAPI_EVENTSET_MAP.dataSlotArray,0x00, 
	 PAPI_INIT_SLOTS*sizeof(EventSetInfo *));

   PAPI_EVENTSET_MAP.totalSlots = PAPI_INIT_SLOTS;
   PAPI_EVENTSET_MAP.availSlots = PAPI_INIT_SLOTS - 1;
   PAPI_EVENTSET_MAP.fullSlots  = 1;
   PAPI_EVENTSET_MAP.lowestEmptySlot = 1;

   /* Remember that EventSet zero is reserved */

   PAPI_ERR_LEVEL = PAPI_QUIET;
   
   zero = (EventSetInfo *)malloc(sizeof(EventSetInfo));
   if (zero == NULL)
     {
heck:
       free(PAPI_EVENTSET_MAP.dataSlotArray);
       memset(&PAPI_EVENTSET_MAP,0x00,sizeof(PAPI_EVENTSET_MAP));
       return(PAPI_ENOMEM);
     }
   memset(zero,0x00,sizeof(EventSetInfo));

   retval = _papi_hwd_init(zero);
   if (retval < PAPI_OK)
     {
       free(zero);
       goto heck;
     }

   return(retval);
}

static int expand_dynamic_array(DynamicArray *DA)
{
  int number;	
  EventSetInfo **n;

  /*realloc existing PAPI_EVENTSET_MAP.dataSlotArray*/
    
  number = DA->totalSlots*2;
  n = (EventSetInfo **)realloc(DA->dataSlotArray,number*sizeof(EventSetInfo *));
  if (n==NULL) 
    return(handle_error(PAPI_ENOMEM,NULL));   

  /* Need to assign this value, what if realloc moved it? */

  DA->dataSlotArray = n;

  memset(DA->dataSlotArray[DA->totalSlots],0x00,DA->totalSlots*sizeof(EventSetInfo *));

  DA->totalSlots = number;
  DA->availSlots = number - DA->fullSlots;
  DA->lowestEmptySlot = DA->totalSlots/2;

  return(PAPI_OK);
}

/*========================================================================*/
/* This function allocates space for one EventSetInfo structure and for   */
/* all of the pointers in this structure.  If any malloc in this function */
/* fails, all memory malloced to the point of failure is freed, and NULL  */
/* is returned.  Upon success, a pointer to the EventSetInfo data         */
/* structure is returned.                                                 */
/*========================================================================*/

static EventSetInfo *allocate_EventSet(void) 
{
  EventSetInfo *ESI;
  int i, max_counters;
  
  ESI=(EventSetInfo *)malloc(sizeof(EventSetInfo));
  if (ESI==NULL) 
    return(NULL); 
  memset(ESI,0x00,sizeof(EventSetInfo));

  max_counters = _papi_system_info.num_cntrs;
  ESI->machdep = (void *)malloc(_papi_system_info.size_machdep);
  ESI->start = (unsigned long long *)malloc(max_counters*sizeof(unsigned long long));
  ESI->stop = (unsigned long long *)malloc(max_counters*sizeof(unsigned long long));
  ESI->latest = (unsigned long long *)malloc(max_counters*sizeof(unsigned long long));
  ESI->EventCodeArray = (int *)malloc(max_counters*sizeof(int));

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
  memset(ESI->start,         0x00,max_counters*sizeof(unsigned long long));
  memset(ESI->stop,          0x00,max_counters*sizeof(unsigned long long));
  memset(ESI->latest,        0x00,max_counters*sizeof(unsigned long long));

  for (i=0;i<max_counters;i++)
    ESI->EventCodeArray[i] = PAPI_NULL;

  ESI->state = PAPI_STOPPED; 
  ESI->all_options.domain.domain.domain = PAPI_DOM_DEFAULT;

  return(ESI);
}

/*========================================================================*/
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
#ifdef DEBUG
  memset(ESI,0x00,sizeof(EventSetInfo));
#endif
  free(ESI);
}

static int add_EventSet(EventSetInfo *ESI)
{
  int i, errorCode;

  /* Update the values for lowestEmptySlot, num of availSlots */

  ESI->EventSetIndex=PAPI_EVENTSET_MAP.lowestEmptySlot;
  PAPI_EVENTSET_MAP.dataSlotArray[ESI->EventSetIndex] = ESI;
  PAPI_EVENTSET_MAP.availSlots--;
  PAPI_EVENTSET_MAP.fullSlots++; 

  if (PAPI_EVENTSET_MAP.availSlots == 0)
    {
      errorCode = expand_dynamic_array(&PAPI_EVENTSET_MAP);
      if (errorCode<PAPI_OK) 
	return(errorCode);
    }

  i = ESI->EventSetIndex + 1;
  while (PAPI_EVENTSET_MAP.dataSlotArray[i]) i++;
  DBG((stderr,"Empty slot for EventSetInfo at %d\n",i));
  PAPI_EVENTSET_MAP.lowestEmptySlot = i;
 
  return(PAPI_OK);
}

/* add_event checks for event already added, zeroes the counters
   values in the ESI structure */

static int add_event(EventSetInfo *ESI, int EventCode) 
{ 
  int k;

  k = lookup_EventCodeIndex(ESI, EventCode);
  if (k<PAPI_OK) 
    return k; 

  ESI->EventCodeArray[k] = EventCode;
  ESI->start[k]          = 0;
  ESI->stop[k]           = 0;
  ESI->latest[k]         = 0;
  ESI->NumberOfCounters++;

  return(PAPI_OK); 
}

/* add_event checks to see whether the ESI structure has been 
   created already for this EventSet, adds the event */

int PAPI_add_event(int *EventSet, int EventCode) 
{ 
  int retval;
  EventSetInfo *ESI,*n = NULL;

  retval = check_initialize();
  if (retval < PAPI_OK)
    return(retval);
  
  /* check for pre-existing ESI*/
  
  ESI = lookup_EventSet(*EventSet);
  if (ESI == NULL)
    {
      n = allocate_EventSet();
      if (n == NULL)
	return(PAPI_ENOMEM);
      ESI = n;
    }

  retval = _papi_hwd_add_event(ESI,EventCode);
  if (retval < PAPI_OK)
    {
      heck:
      if (n)
	free_EventSet(ESI);
      return(retval);
    }

  retval = add_event(ESI,EventCode);
  if (retval < PAPI_OK)
    goto heck;

  if (n)
    {
      retval = add_EventSet(ESI);
      if (retval < PAPI_OK)
	goto heck;

      *EventSet = ESI->EventSetIndex;
    }
  return(retval);
}

static int lookup_EventCodeIndex(EventSetInfo *ESI,int EventCode)
{
  int i;

  for(i=0;i<ESI->NumberOfCounters;i++) 
    { 
      if (ESI->EventCodeArray[i]==EventCode) 
	return(PAPI_ECNFLCT);
    }

  return(PAPI_OK);
} 

static EventSetInfo *lookup_EventSet(int eventset)
{
  if ((eventset >= 1) && (eventset < PAPI_EVENTSET_MAP.totalSlots))
    return(PAPI_EVENTSET_MAP.dataSlotArray[eventset]);
  else
    return(NULL);
}

static int remove_EventSet(EventSetInfo *ESI)
{
  int i;

  /* get value of Index I for this ESI in 
     PAPI_EVENTSET_MAP.dataSlotArray[I]    */

  i = ESI->EventSetIndex;

  free_EventSet(ESI);

  /* do bookkeeping for PAPI_EVENTSET_MAP */

  PAPI_EVENTSET_MAP.dataSlotArray[i] = NULL;
  if (PAPI_EVENTSET_MAP.lowestEmptySlot < i)
    PAPI_EVENTSET_MAP.lowestEmptySlot = i;
  PAPI_EVENTSET_MAP.availSlots++;
  PAPI_EVENTSET_MAP.fullSlots--;

  return(PAPI_OK);
}

static int remove_event(EventSetInfo *ESI, int EventCode)
{
  int k;

  k = lookup_EventCodeIndex(ESI,EventCode);
  if (k < 0)
    return(PAPI_EINVAL);

  ESI->EventCodeArray[k] = PAPI_NULL;
  ESI->start[k]          = 0;
  ESI->stop[k]           = 0;
  ESI->latest[k]         = 0;
  ESI->NumberOfCounters--;

  return(PAPI_OK);
}

int PAPI_rem_event(int EventSet, int Event)
{
  EventSetInfo *ESI;
  int errorCode, hwd_errorCode;

  ESI = lookup_EventSet(EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,NULL));

  errorCode = remove_event(ESI,Event);
  if (errorCode < PAPI_OK)
    return(handle_error(errorCode,NULL));

  hwd_errorCode = _papi_hwd_rem_event(ESI,Event);
  if (hwd_errorCode < PAPI_OK)
    return(handle_error(hwd_errorCode,NULL));

  if (ESI->NumberOfCounters == 0)
    remove_EventSet(ESI);

  return(hwd_errorCode);
}

/* simply checks for valid EventSet, calls substrate start() call */

int PAPI_start(int EventSet)
{ 
  int retval;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL) return(handle_error(PAPI_EINVAL, NULL));

  retval = _papi_hwd_start(ESI);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));

  DBG((stderr,"PAPI_start returns %d\n",retval));
  return(retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */
int PAPI_stop(int EventSet, unsigned long long *values)
{ 
  int retval;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI==NULL) return(handle_error(PAPI_EINVAL, NULL));

  retval = _papi_hwd_stop(ESI, values);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));

  retval = _papi_hwd_reset(ESI);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));

#if defined(DEBUG)
  if (values)
    { 
      int i;
      for (i=0;i<ESI->NumberOfCounters;i++)
	DBG((stderr,"PAPI_stop values[%d]:\t\t%lld\n",i,values[i]));
    }
#endif

  return(retval);
}

int PAPI_reset(int EventSet)
{ int retval;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL) return(handle_error(PAPI_EINVAL, NULL));

  retval = _papi_hwd_reset(ESI);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));

  DBG((stderr,"PAPI_reset returns %d\n",retval));
  return(retval);
}

int PAPI_read(int EventSet, unsigned long long *values)
{ 
  int retval;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if ( ESI == NULL )
    return(handle_error(PAPI_EINVAL,NULL));

  retval = _papi_hwd_read(ESI, values);
  if(retval<PAPI_OK) 
    return(handle_error(retval, NULL));

  return(retval);
}

int PAPI_accum(int EventSet, unsigned long long *values)
{ EventSetInfo *ESI;
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

  retval = _papi_hwd_read(ESI, increase);
  if (retval < PAPI_OK) 
    return(handle_error(retval,NULL));
  retval = _papi_hwd_reset(ESI);
  if (retval < PAPI_OK)
    return(handle_error(retval,NULL));
  return(retval);
}

int PAPI_write(int EventSet, unsigned long long *values)
{ int retval;
  EventSetInfo *ESI; 

  ESI = lookup_EventSet(EventSet);
  if ( ESI == NULL )
    return(handle_error(PAPI_EINVAL,NULL));

  retval = _papi_hwd_write(ESI, values);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));
  return(retval);
}

int PAPI_set_opt(int option, PAPI_option_t *ptr)
{ int retval;
  _papi_int_option_t internal;

  switch(option)
  { case PAPI_SET_MPXRES:
    case PAPI_SET_OVRFLO:
    case PAPI_GET_MPXRES:
    case PAPI_GET_OVRFLO:
    case PAPI_SET_DEFDOM:
    case PAPI_SET_DEFGRN:
    case PAPI_SET_DOMAIN:
    { internal.domain.domain.eventset=ptr->domain.eventset;
      internal.domain.domain.domain=ptr->domain.domain;
      internal.domain.ESI=lookup_EventSet(ptr->domain.eventset);
      retval = _papi_hwd_ctl(PAPI_SET_DOMAIN, &internal);
      return(retval);
    }
    case PAPI_SET_GRANUL:
    case PAPI_GET_DEFDOM:
    case PAPI_GET_DEFGRN:
    case PAPI_GET_DOMAIN:
    case PAPI_GET_GRANUL:
    default:
      return(PAPI_EINVAL);
    }
}
 
void PAPI_shutdown(void) {
    int i;

    for(i=0;i<PAPI_EVENTSET_MAP.totalSlots;i++) {
        if(PAPI_EVENTSET_MAP.dataSlotArray[i]) {
           free_EventSet(PAPI_EVENTSET_MAP.dataSlotArray[i]);
          }
        }
    free(PAPI_EVENTSET_MAP.dataSlotArray);
    fprintf(stderr,"\n\n PAPI SHUTDOWN. \n\n");
    return;
}

static int handle_error(int PAPI_errorCode, char *errorMessage)
{
  if (PAPI_ERR_LEVEL)
    {
      fprintf(stderr, "%s", get_error_string(PAPI_errorCode));
      if ( PAPI_errorCode==PAPI_ESYS ) perror(errorMessage);
      if (errorMessage) fprintf(stderr, ": %s", errorMessage);
      fprintf(stderr,"\n");
      if (PAPI_ERR_LEVEL==PAPI_VERB_ESTOP)
        PAPI_shutdown();
    }
  return(PAPI_errorCode);
}

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

