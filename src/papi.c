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

static int check_initialize(void);
static int initialize(void);
static EventSetInfo *allocate_EventSet(int *EventSet);
static int add_EventSet(EventSetInfo *);
static int add_event(EventSetInfo *ESI,int Event);
static int remove_event(EventSetInfo *ESI,int Event);
static int remove_EventSet(EventSetInfo *);
static void free_EventSet(EventSetInfo *);
static int handle_error(int, char *);
static char *get_error_string(int);
static EventSetInfo *lookup_EventSet(int eventset);
static int lookup_EventCodeIndex(EventSetInfo *ESI,int EventCode);

EventSetInfo *get_valid_ESI(int *EventSetIndex, int *allocated_a_new_one );

DynamicArray PAPI_EVENTSET_MAP;    

int PAPI_ERR_LEVEL = PAPI_VERB_ESTOP; 

static int num_counters(EventSetInfo *ESI)
{
  return(_papi_system_info.num_cntrs);
}

static int check_initialize(void) {
  int errorCode;
    if(PAPI_EVENTSET_MAP.totalSlots==0) {
        errorCode=initialize(); 
        if(errorCode<PAPI_OK)
	return(handle_error(errorCode,NULL));
	}
   return(PAPI_OK);
} 

static int initialize(void)
{ int i, j, num_counters;
  int a[(_papi_system_info.num_gp_cntrs+_papi_system_info.num_sp_cntrs)];
  long long b[(_papi_system_info.num_gp_cntrs+_papi_system_info.num_sp_cntrs)];
  EventSetInfo tmp;
  num_counters=
    (_papi_system_info.num_gp_cntrs+_papi_system_info.num_sp_cntrs);

  for(i=0; i<PAPI_INIT_SLOTS; i++)
  { PAPI_EVENTSET_MAP.dataSlotArray[i]=(EventSetInfo *)malloc(sizeof(tmp)); 
    PAPI_EVENTSET_MAP.dataSlotArray[i]->EventSetIndex=i;
    PAPI_EVENTSET_MAP.dataSlotArray[i]->all_options.domain.domain.domain=PAPI_DOM_DEFAULT;
    PAPI_EVENTSET_MAP.dataSlotArray[i]->NumberOfCounters=0;
    PAPI_EVENTSET_MAP.dataSlotArray[i]->machdep=
                     (void *)malloc(sizeof(_papi_system_info.size_machdep));
    PAPI_EVENTSET_MAP.dataSlotArray[i]->EventCodeArray=(int *)malloc(sizeof(a));
    PAPI_EVENTSET_MAP.dataSlotArray[i]->start=(long long *)malloc(sizeof(b));
    PAPI_EVENTSET_MAP.dataSlotArray[i]->stop=(long long *)malloc(sizeof(b));
    PAPI_EVENTSET_MAP.dataSlotArray[i]->latest=(long long *)malloc(sizeof(b));
    for(j=0;j<num_counters;j++)
    { PAPI_EVENTSET_MAP.dataSlotArray[i]->EventCodeArray[j]=-1;
      PAPI_EVENTSET_MAP.dataSlotArray[i]->start[j]=0;
      PAPI_EVENTSET_MAP.dataSlotArray[i]->stop[j]=0;
      PAPI_EVENTSET_MAP.dataSlotArray[i]->latest[j]=0;
    }
  }
  PAPI_EVENTSET_MAP.totalSlots = PAPI_INIT_SLOTS;
  PAPI_EVENTSET_MAP.availSlots = PAPI_INIT_SLOTS - 1;
  PAPI_EVENTSET_MAP.fullSlots  = 1;
  PAPI_EVENTSET_MAP.lowestEmptySlot = 1;
  PAPI_ERR_LEVEL = PAPI_QUIET;
  return(PAPI_OK);
}

/* allocate_EventSet() only called for creating a new EventSet */
static EventSetInfo *allocate_EventSet(int *EventSet) 
{ EventSetInfo *ESI;
  int i;
 
  i=PAPI_EVENTSET_MAP.lowestEmptySlot;
  ESI=PAPI_EVENTSET_MAP.dataSlotArray[i];
  *EventSet=i; 
  PAPI_EVENTSET_MAP.availSlots = PAPI_EVENTSET_MAP.availSlots - 1;
  PAPI_EVENTSET_MAP.fullSlots = PAPI_EVENTSET_MAP.fullSlots + 1;   
  PAPI_EVENTSET_MAP.lowestEmptySlot = PAPI_EVENTSET_MAP.lowestEmptySlot + 1;
  return(ESI);
}

/* add_event checks for event already added, zeroes the counters
   values in the ESI structure */
static int add_event(EventSetInfo *ESI, int EventCode) 
{ int k,A,B,counterArrayLength;
  
  A=_papi_system_info.num_gp_cntrs;
  B=_papi_system_info.num_sp_cntrs;
  counterArrayLength=A+B;

  k=lookup_EventCodeIndex(ESI, EventCode);
  if(k<PAPI_OK) return k; 
  ESI->EventCodeArray[k] = EventCode;
  ESI->start[k]          = 0;
  ESI->stop[k]           = 0;
  ESI->latest[k]         = 0;
  ESI->NumberOfCounters++;
  return(PAPI_OK); 
}

EventSetInfo *get_valid_ESI(int *EventSetIndex, int *allocated_a_new_one )
{ int retval;
  EventSetInfo *ESI;
    
  retval=check_initialize();
  if(retval<PAPI_OK) return(NULL);
  ESI = lookup_EventSet(*EventSetIndex);

  if ( ESI!=NULL )
  { *allocated_a_new_one=0;
    return(ESI);
  }
  else 
  { ESI = allocate_EventSet(EventSetIndex);
    if (!ESI) 
    { *allocated_a_new_one=0; 
      return(NULL);
    }
    *allocated_a_new_one=1;
    return(ESI);
  }
}

/* add_event checks to see whether the ESI structure has been 
   created already for this EventSet, adds the event */
int PAPI_add_event(int *EventSet, int EventCode) 
{ int retval;
  int allocated_a_new_one = 0;
  int EventSetIndex;
  EventSetInfo *ESI;

  EventSetIndex = *EventSet;
  ESI=get_valid_ESI(&EventSetIndex,&allocated_a_new_one);
  retval = _papi_hwd_add_event(ESI,EventCode);

  if ( retval >= PAPI_OK) 
  { retval=add_event(ESI,EventCode);
    (int *)*EventSet = (int *)EventSetIndex;
    return(retval);
  }
}

static int lookup_EventCodeIndex(EventSetInfo *ESI,int EventCode)
{
  int i;
  for(i=0;i<ESI->NumberOfCounters;i++) 
  { if (ESI->EventCodeArray[i]==EventCode) 
      return(PAPI_ECNFLCT);
    if(ESI->EventCodeArray[i]<0) return i;
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

/* simply checks for valid EventSet, calls substrate start() call */
int PAPI_start(int EventSet)
{ 
  int retval;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL) return(handle_error(PAPI_EINVAL, NULL));

  retval = _papi_hwd_start(ESI);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));
  return(retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */
int PAPI_stop(int EventSet, unsigned long long *values)
{ int retval, i, bound;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI==NULL) return(handle_error(PAPI_EINVAL, NULL));

  retval = _papi_hwd_stop(ESI, values);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));

  bound = num_counters(ESI);

  for(i=0; i<bound; i++)
  { if(values[i] >= 0)
    { printf("\tCounter %d : %lld\n", i, values[i]);
    }
  }

  retval = _papi_hwd_reset(ESI);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));
  return(retval);
}

int PAPI_reset(int EventSet)
{ int retval;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL) return(handle_error(PAPI_EINVAL, NULL));

  retval = _papi_hwd_reset(ESI);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));
  return(retval);
}

int PAPI_read(int EventSet, unsigned long long *values)
{ int retval, i, bound;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if ( ESI == NULL )
    return(handle_error(PAPI_EINVAL,NULL));

  retval = _papi_hwd_read(ESI, values);
  if(retval<PAPI_OK) return(handle_error(retval, NULL));

  bound = num_counters(ESI);
  for(i=0; i<bound; i++)
  { if(values[i] >= 0) 
    { printf("\tCounter %d : %lld\n", i, values[i]);
    }
  }
  return(retval);
}

int PAPI_accum(int EventSet, unsigned long long *values)
{ EventSetInfo *ESI;
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

static int remove_EventSet(EventSetInfo *ESI)
{
   int I;
      I=ESI->EventSetIndex;
      free_EventSet(ESI);
        PAPI_EVENTSET_MAP.dataSlotArray[I]=NULL;
     if(PAPI_EVENTSET_MAP.lowestEmptySlot < I)
        PAPI_EVENTSET_MAP.lowestEmptySlot = I;
        PAPI_EVENTSET_MAP.availSlots++;
        PAPI_EVENTSET_MAP.fullSlots--;
  return(PAPI_OK);
}

int PAPI_rem_event(int EventSet, int Event)
{
  EventSetInfo *ESI;
  int errorCode, hwd_errorCode;

  ESI = lookup_EventSet(EventSet);
  if ( ESI == NULL )
    return(handle_error(PAPI_EINVAL,NULL));

  errorCode = remove_event(ESI,Event);
  if (errorCode < PAPI_OK)
    return(handle_error(errorCode,NULL));

  hwd_errorCode = _papi_hwd_rem_event(ESI,Event);
  if (hwd_errorCode < PAPI_OK)
    return(handle_error(hwd_errorCode,NULL));

  if (ESI->NumberOfCounters == 0)
    remove_EventSet(ESI);

  return(errorCode);
}

static int remove_event(EventSetInfo *ESI, int EventCode)
{
  int k;

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

static void free_EventSet(EventSetInfo *ESI)
{
  if (ESI->EventCodeArray) free(ESI->EventCodeArray);
  if (ESI->machdep)        free(ESI->machdep);
  if (ESI->start)          free(ESI->start);
  if (ESI->stop)           free(ESI->stop);
  if (ESI->latest)         free(ESI->latest);
  free(ESI);
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

