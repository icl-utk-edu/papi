/* $Id$ */

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <assert.h>

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
static EventSetInfo *lookup_Zero(int eventset);

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
{/* returns number of counters currently loaded 
    in specified eventset*/

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
  int retval, i;
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

#ifdef DEBUG
   PAPI_ERR_LEVEL = PAPI_VERB_ECONT;
#else
   PAPI_ERR_LEVEL = PAPI_QUIET;
#endif

   /* Remember that EventSet zero is reserved */
   
   zero = (EventSetInfo *)malloc(sizeof(EventSetInfo));
   if (zero == NULL)
     {
heck:
       free(PAPI_EVENTSET_MAP.dataSlotArray);
       memset(&PAPI_EVENTSET_MAP,0x00,sizeof(PAPI_EVENTSET_MAP));
       return(PAPI_ENOMEM);
     }
   memset(zero,0x00,sizeof(EventSetInfo));

   zero->machdep = (void *)malloc(_papi_system_info.size_machdep);

   PAPI_EVENTSET_MAP.dataSlotArray[0] = zero;

   retval = _papi_hwd_init(zero);
   if (retval < PAPI_OK)
     {
       free(zero);
       goto heck;
     }
   zero->all_options.multistart.multistart.SharedDepth =
    (int **)malloc(_papi_system_info.num_cntrs*sizeof(int *));
   zero->all_options.multistart.multistart.EvSetArray =
    (int **)malloc(PAPI_INIT_SLOTS*sizeof(int *));
   zero->all_options.multistart.multistart.virtual_machdep =
    (void *)malloc(_papi_system_info.size_machdep);
   zero->all_options.multistart.multistart.num_runners =0;
   for(i=0; i<_papi_system_info.num_cntrs; i++)
     zero->all_options.multistart.multistart.SharedDepth[i] = 0;

   return(retval);
}

int PAPI_init(void)
{ 
  return(check_initialize());
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
  ESI->EventSelectArray = (int *)malloc(max_counters*sizeof(int));

  if ((ESI->machdep        == NULL )  || 
      (ESI->start          == NULL )  || 
      (ESI->stop           == NULL )  || 
      (ESI->latest         == NULL )  ||
      (ESI->EventCodeArray == NULL )  ||
      (ESI->EventSelectArray == NULL ))
    {
      if (ESI->machdep)        free(ESI->machdep);
      if (ESI->start)          free(ESI->start);
      if (ESI->stop)           free(ESI->stop);
      if (ESI->latest)         free(ESI->latest);
      if (ESI->EventCodeArray) free(ESI->EventCodeArray);
      if (ESI->EventSelectArray)  free(ESI->EventSelectArray);
      free(ESI);
      return(NULL);
    }
  memset(ESI->machdep,       0x00,_papi_system_info.size_machdep);
  memset(ESI->start,         0x00,max_counters*sizeof(unsigned long long));
  memset(ESI->stop,          0x00,max_counters*sizeof(unsigned long long));
  memset(ESI->latest,        0x00,max_counters*sizeof(unsigned long long));

  for (i=0;i<max_counters;i++)
    {
      ESI->EventCodeArray[i] = PAPI_NULL;
      ESI->EventSelectArray[i] = PAPI_NULL;
    }

  ESI->state = PAPI_STOPPED; 
  ESI->all_options.domain.domain.domain = _papi_system_info.default_domain;
  ESI->all_options.multiplex.multiplex.milliseconds=0;

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
  if (ESI->EventSelectArray)  free(ESI->EventSelectArray);
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
   values in the ESI structure. This function now returns the index
   of the slot in EventCodeArray and EventSelectArray which is used for
   the added EventCode. */

static int add_event(EventSetInfo *ESI, int EventCode) 
{ 
  int k;

  if (ESI->NumberOfCounters == _papi_system_info.num_cntrs)
    return(PAPI_ECNFLCT);

  k = lookup_EventCodeIndex(ESI, EventCode);
  if (k != PAPI_EINVAL)
    return(PAPI_ECNFLCT); 

  /* Take the lowest empty slot */

  for (k=0;k<_papi_system_info.num_cntrs;k++) {
    if (ESI->EventCodeArray[k] == -1)
      break;
  } 

  if (k == _papi_system_info.num_cntrs)
    return(PAPI_ECNFLCT);

  ESI->EventCodeArray[k] = EventCode;
  ESI->start[k]          = 0;
  ESI->stop[k]           = 0;
  ESI->latest[k]         = 0;
  ESI->NumberOfCounters++;

  return(k); 
}

/* Describe the event. name is input/output, eventcode is input/output,
   description is output */

typedef struct pre_info {
  char *name;
  unsigned int code;
  char *descr; } preset_info_t; 

static preset_info_t papi_preset_info[PAPI_MAX_PRESET_EVENTS] = { 
  { "PAPI_L1_DCM", 0x80000000, "Level 1 Data Cache Misses" },
  { "PAPI_L1_ICM", 0x80000001, "Level 1 Instruction Cache Misses" },
  { "PAPI_L2_DCM", 0x80000002, "Level 2 Data Cache Misses" },
  { "PAPI_L2_ICM", 0x80000003, "Level 2 Instruction Cache Misses" },
  { "PAPI_L3_DCM", 0x80000004, "Level 3 Data Cache Misses" },
  { "PAPI_L3_ICM", 0x80000005, "Level 3 Instruction Cache Misses" },
  { NULL, 0x80000006, NULL },
  { NULL, 0x80000007, NULL },
  { NULL, 0x80000008, NULL },
  { NULL, 0x80000009, NULL },
  { "PAPI_CA_SHR", 0x8000000A, "Requests for Shared Cache Line" },
  { "PAPI_CA_CLN", 0x8000000B, "Requests for Clean Cache Line" },
  { "PAPI_CA_INV", 0x8000000C, "Cache Line Invalidation Requests" },
  { NULL, 0x8000000D, NULL },
  { NULL, 0x8000000E, NULL },
  { NULL, 0x8000000F, NULL },
  { NULL, 0x80000010, NULL },
  { NULL, 0x80000011, NULL },
  { NULL, 0x80000012, NULL },
  { NULL, 0x80000013, NULL },
  { "PAPI_TLB_DM", 0x80000014, "Data Translation Lookaside Buffer Misses" },
  { "PAPI_TLB_IM", 0x80000015, "Instruction Translation Lookaside Buffer Misses" },
  { "PAPI_TLB_TOT", 0x80000016, "Total Translation Lookaside Buffer Misses" },
  { NULL,  0x80000017, NULL },
  { NULL,  0x80000018, NULL },
  { NULL,  0x80000019, NULL },
  { NULL,  0x8000001A, NULL },
  { NULL,  0x8000001B, NULL },
  { NULL,  0x8000001C, NULL },
  { NULL,  0x8000001D, NULL },
  { "PAPI_TLB_SD", 0x8000001E, "Translation Lookaside Buffer Shootdowns" },
  { NULL,  0x8000001F, NULL },
  { NULL,  0x80000020, NULL },
  { NULL,  0x80000021, NULL },
  { "PAPI_MEM_SCY",  0x80000022, "Cycles Stalled Waiting for Memory Access" },
  { "PAPI_MEM_RCY",  0x80000023, "Cycles Stalled Waiting for Memory Read" },
  { "PAPI_MEM_WCY",  0x80000024, "Cycles Stalled Waiting for Memory Write" },
  { "PAPI_STL_SCY",  0x80000025, "Cycles with No Instruction Issue" },
  { "PAPI_FUL_CYC",  0x80000026, "Cycles with Maximum Instruction Issue" },
  { NULL,  0x80000027, NULL },
  { NULL,  0x80000028, NULL },
  { NULL,  0x80000029, NULL },
  { "PAPI_BR_UCN", 0x8000002A, "Unconditional Branch Instructions" },
  { "PAPI_BR_CN", 0x8000002B, "Conditional Branch Instructions" },
  { "PAPI_BR_TKN", 0x8000002C, "Conditional Branch Instructions Taken" }, 
  { "PAPI_BR_NTK", 0x8000002D, "Conditional Branch Instructions Not Taken" }, 
  { "PAPI_BR_MSP", 0x8000002E, "Conditional Branch Instructions Mispredicted" },
  { NULL,  0x8000002F, NULL },
  { NULL,  0x80000030, NULL },
  { NULL,  0x80000031, NULL },
  { "PAPI_TOT_INS", 0x80000032, "Total Instructions" },
  { "PAPI_INT_INS", 0x80000033, "Integer Instructions" },
  { "PAPI_FP_INS", 0x80000034, "Floating Point Instructions" },
  { "PAPI_LD_INS", 0x80000035, "Load Instructions" },
  { "PAPI_SR_INS", 0x80000036, "Store Instructions" },
  { "PAPI_BR_INS", 0x80000037, "Branch Instructions" },
  { "PAPI_VEC_INS", 0x80000038, "Vector Instructions" },
  { "PAPI_FLOPS", 0x80000039, "Floating Point Instructions per Second" },
  { NULL,  0x8000003A, NULL },
  { NULL,  0x8000003B, NULL },
  { "PAPI_TOT_CYC",  0x8000003C, "Total Cycles" },
  { "PAPI_MIPS", 0x8000003D, "Millions of Instructions per Second" },
  { NULL,  0x8000003E, NULL },
  { NULL,  0x8000003, NULL } };

int PAPI_describe_event(char *name, int *EventCode, char *description)
{
  if (name)
    {
      int i;
      
      for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
	{
	  if (strcmp(papi_preset_info[i].name,name) == 0)
	    {
	      if (description)
		strcpy(description,papi_preset_info[i].descr);
	      *EventCode = papi_preset_info[i].code;
	      return(PAPI_OK);
	    }
	}
      return(PAPI_EINVAL);
    }
  if ((*EventCode > 0) && (*EventCode < PAPI_MAX_PRESET_EVENTS))
    {
      if (description)
	strcpy(description,papi_preset_info[*EventCode].descr);
      if (name)
	strcpy(name,papi_preset_info[*EventCode].name);
      return(PAPI_OK);
    }
  return(PAPI_EINVAL);
}

int PAPI_query_event(int EventCode)
{
  int retval;

  retval = _papi_hwd_query(EventCode);
  if (retval != PAPI_OK)
    return(handle_error(retval,"Event does not exist on this substrate"));

  DBG((stderr,"PAPI_query returns %d\n",retval));
  return(retval);
}

/* add_event checks to see whether the ESI structure has been 
   created already for this EventSet, adds the event */

int PAPI_add_event(int *EventSet, int EventCode) 
{ 
  int retval,indextohw;
  EventSetInfo *ESI,*n = NULL;

  PAPI_init();
  
  /* check for pre-existing ESI*/
  
  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(*EventSet);
  if (ESI == NULL)
    {
      n = allocate_EventSet();
      if (n == NULL)
	return(handle_error(PAPI_ENOMEM,"Error allocating memory for new EventSet"));
      ESI = n;
    }

  /* This returns index into the map array. Note that this routine
     increments ESI->NumberOfCounters. */

  retval = add_event(ESI,EventCode);
  if (retval < PAPI_OK)
    {
    heck:
      if (n)
	free_EventSet(ESI);
      return(handle_error(retval,NULL));
    }

  indextohw = retval;
  retval = _papi_hwd_add_event(ESI,indextohw,EventCode);
  if (retval < PAPI_OK)
    {
      remove_event(ESI,EventCode);
      goto heck;
    }

  if (n)
    {
      retval = add_EventSet(ESI);
      if (retval < PAPI_OK)
	goto heck;

      *EventSet = ESI->EventSetIndex;
      DBG((stderr,"PAPI_add_event new EventSet in slot %d\n",*EventSet));
    }
  return(retval);
}

/* This function should return the index of the EventCode and counter
   value in the arrays inside ESI. */

static int lookup_EventCodeIndex(EventSetInfo *ESI, int EventCode)
{
  int i;

  for(i=0;i<_papi_system_info.num_cntrs;i++) 
    { 
      if (ESI->EventCodeArray[i]==EventCode) 
	return(i);
    }

  return(PAPI_EINVAL);
} 

static EventSetInfo *lookup_EventSet(int eventset)
{
  if ((eventset >= 1) && (eventset < PAPI_EVENTSET_MAP.totalSlots))
    return(PAPI_EVENTSET_MAP.dataSlotArray[eventset]);
  else
    return(NULL);
}

static EventSetInfo *lookup_Zero(int eventset)
{ if(eventset == 0) return(PAPI_EVENTSET_MAP.dataSlotArray[0]);
  return(NULL);
}

/* This function only removes empty EventSets */

static int remove_EventSet(EventSetInfo *ESI)
{
  int i;

  assert(ESI->NumberOfCounters == 0);

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

static int cleanup_EventSet(EventSetInfo *ESI)
{
  int i;

  /* first remove all of the Events from this EventSet*/
  /* ignore return vals */

  for(i=0;i<_papi_system_info.num_cntrs;i++) 
    {
      remove_event(ESI,ESI->EventCodeArray[i]);
    }

  return(remove_EventSet(ESI));
}

static int remove_event(EventSetInfo *ESI, int EventCode)
{
  int k;

  k = lookup_EventCodeIndex(ESI,EventCode);
  if (k < 0)
    return(k);

  ESI->EventCodeArray[k] = PAPI_NULL;
  ESI->EventSelectArray[k] = PAPI_NULL;
  ESI->start[k]          = 0;
  ESI->stop[k]           = 0;
  ESI->latest[k]         = 0;
  ESI->NumberOfCounters--;

  return(PAPI_OK);
}

int PAPI_rem_event(int *EventSet, int EventCode)
{
  EventSetInfo *ESI;
  int retval;

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(*EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

 if (ESI->state != PAPI_STOPPED)
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  retval = remove_event(ESI,EventCode);
  if (retval < PAPI_OK)
    return(handle_error(retval,NULL));

  retval = _papi_hwd_rem_event(ESI,EventCode);
  if (retval < PAPI_OK)
    return(handle_error(retval,NULL));

  if (ESI->NumberOfCounters == 0)
    {
      remove_EventSet(ESI);
      *EventSet = PAPI_NULL;
    }

  return(retval);
}

/* simply checks for valid EventSet, calls substrate start() call */

int PAPI_start(int EventSet)
{ 
  int retval;
  EventSetInfo *ESI, *zero;

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL) 
     return(handle_error(PAPI_EINVAL, NULL));

  if (ESI->state != PAPI_STOPPED)
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  zero = lookup_Zero(0);
  if(zero == NULL) return(handle_error(PAPI_EINVAL, NULL)); 

  if(zero->all_options.multistart.multistart.num_runners >0)
  { retval=_papi_hwd_merge(ESI, zero);
    if(retval<PAPI_OK) return(handle_error(retval, NULL));
  }
  else
  { retval = _papi_hwd_start(ESI);
    if(retval<PAPI_OK) return(handle_error(retval, NULL));
  }
  ESI->state=PAPI_RUNNING;
  zero->all_options.multistart.multistart.num_runners ++;

  DBG((stderr,"PAPI_start returns %d\n",retval));
  return(retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */
int PAPI_stop(int EventSet, unsigned long long *values)
{ 
  int retval, i;
  EventSetInfo *ESI, *zero;

  ESI = lookup_EventSet(EventSet);
  if(ESI==NULL) 
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (ESI->state != PAPI_RUNNING)
    return(handle_error(PAPI_EINVAL, "EventSet is not running"));

  zero = lookup_Zero(0);
  if(zero==NULL) return(handle_error(PAPI_EINVAL, NULL));

  if(zero->all_options.multistart.multistart.num_runners >1)
  { retval=_papi_hwd_unmerge(ESI, zero);
    if(retval<PAPI_OK) return(handle_error(retval, NULL));
    for (i=0;i<_papi_system_info.num_cntrs;i++) values[i] = ESI->stop[i]; 
  }
  else
  { retval = _papi_hwd_stop(ESI, values);
    if(retval<PAPI_OK) return(handle_error(retval, NULL));
  }

  if (values)
    memcpy(values,ESI->latest,_papi_system_info.num_cntrs*sizeof(unsigned long long));

#if defined(DEBUG)
  if (values)
    {
      int i;
      for (i=0;i<ESI->NumberOfCounters;i++)
        DBG((stderr,"PAPI_stop values[%d]:\t%lld\n",i,values[i]));
    }
#endif

  ESI->state=PAPI_STOPPED;
  DBG((stderr,"PAPI_stop returns %d\n",retval));

  zero->all_options.multistart.multistart.num_runners --;
  return(retval);
}

int PAPI_reset(int EventSet)
{ int retval = PAPI_OK;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL) 
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (ESI->state == PAPI_RUNNING)
    {
      retval = _papi_hwd_reset(ESI);
      if (retval<PAPI_OK)
        return(handle_error(retval, NULL));
    }
  else
    {
      memset(ESI->latest,0x00,_papi_system_info.num_cntrs*sizeof(unsigned long long));
    }

  DBG((stderr,"PAPI_reset returns %d\n",retval));
  return(retval);
}

int PAPI_read(int EventSet, unsigned long long *values)
{ 
  int retval = PAPI_OK;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state == PAPI_RUNNING)
    {
      retval = _papi_hwd_read(ESI, ESI->latest);
      if (retval<PAPI_OK)
        return(handle_error(retval, NULL));
    }
  memcpy(values,ESI->latest,_papi_system_info.num_cntrs*sizeof(unsigned long long));

  DBG((stderr,"PAPI_read returns %d\n",retval));
  return(retval);
}

int PAPI_accum(int EventSet, unsigned long long *values)
{ EventSetInfo *ESI;
  int retval = PAPI_OK, i;
  unsigned long long a,b,c;

  ESI = lookup_EventSet(EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state == PAPI_RUNNING)
    {
      retval = _papi_hwd_read(ESI, ESI->latest);
      if (retval < PAPI_OK)
        return(handle_error(retval,NULL));

      retval = _papi_hwd_reset(ESI);
      if (retval < PAPI_OK)
        return(handle_error(retval,NULL));
    }

  for (i=0 ; i < ESI->NumberOfCounters; i++)
    {
      a = ESI->latest[i];
      b = values[i];
      c = a + b;
      values[i] = c;
    }

  memset(ESI->latest,0x0,_papi_system_info.num_cntrs*sizeof(unsigned long long));

  return(retval);
}

int PAPI_write(int EventSet, unsigned long long *values)
{
  int retval = PAPI_OK;
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state == PAPI_RUNNING)
    {
      retval = _papi_hwd_write(ESI, values);
      if (retval<PAPI_OK)
        return(handle_error(retval, NULL));
    }

  memcpy(ESI->latest,values,_papi_system_info.num_cntrs*sizeof(unsigned long long));

  return(retval);
}

/*  The function PAPI_cleanup removes a stopped 
    EventSet from existence. */

int PAPI_cleanup(int *EventSet) 
{ 
  int retval;
  EventSetInfo *ESI;
  
  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(*EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  if (ESI->state != PAPI_STOPPED) 
    return(handle_error(PAPI_EINVAL,"EventSet is still running"));
  
  retval = cleanup_EventSet(ESI);
  if (retval < PAPI_OK)
    return(handle_error(PAPI_EMISC,NULL));

  *EventSet = PAPI_NULL;
  return(retval);
}
 
int PAPI_state(int EventSet, int *status)
{
  EventSetInfo *ESI;
  
  if (status == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  /* check for good EventSetIndex value*/
  
  ESI = lookup_EventSet(EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));
  
  /*read status FROM ESI->state*/
  
  *status=ESI->state;
  
  return(PAPI_OK);
}


int PAPI_set_opt(int option, PAPI_option_t *ptr)
{ 
  int retval;
  _papi_int_option_t internal;
  
  switch(option)
    { 
    case PAPI_SET_DOMAIN:
      { 
	int dom = ptr->defdomain.domain;

	if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
	  return(handle_error(PAPI_EINVAL,"Domain out of range"));
	internal.domain.domain.domain = dom;

	internal.domain.domain.eventset = ptr->domain.eventset;
	internal.domain.ESI = lookup_EventSet(ptr->domain.eventset);
	if (internal.domain.ESI == NULL)
	  return(handle_error(PAPI_EINVAL,"No such EventSet"));

	retval = _papi_hwd_ctl(PAPI_SET_DOMAIN, &internal);
	if (retval < PAPI_OK)
	  return(handle_error(retval,NULL));
	else
	  return(retval);
      }
    case PAPI_SET_DEFDOM:
      {
	int dom = ptr->defdomain.domain;

	if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
	  return(handle_error(PAPI_EINVAL,"Domain out of range"));

	internal.defdomain.defdomain.domain = dom;

	retval = _papi_hwd_ctl(PAPI_SET_DEFDOM, &internal);
	if (retval < PAPI_OK)
	  return(handle_error(retval,NULL));
	else
	  return(retval);
      }
    case PAPI_SET_GRANUL:
      {
	int grn = ptr->defgranularity.granularity;

	if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
	  return(handle_error(PAPI_EINVAL,"Granularity out of range"));

	internal.defgranularity.defgranularity.granularity = grn;

	retval = _papi_hwd_ctl(PAPI_SET_DEFGRN, &internal);
	if (retval < PAPI_OK)
	  return(handle_error(retval,NULL));
	else
	  return(retval);
      }
    case PAPI_SET_DEFGRN:
      {
	int grn = ptr->defgranularity.granularity;

	if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
	  return(handle_error(PAPI_EINVAL,"Granularity out of range"));

	internal.defgranularity.defgranularity.granularity = grn;

	retval = _papi_hwd_ctl(PAPI_SET_DEFGRN, &internal);
	if (retval < PAPI_OK)
	  return(handle_error(retval,NULL));
	else
	  return(retval);
      }
    case PAPI_SET_MPXRES:
    case PAPI_SET_OVRFLO:
    case PAPI_GET_MPXRES:
    case PAPI_GET_OVRFLO:
    case PAPI_GET_DEFGRN:
    case PAPI_GET_DOMAIN:
    case PAPI_GET_GRANUL:
    default:
      return(PAPI_EINVAL);
    }
}


int PAPI_get_opt(int option, PAPI_option_t *ptr) 
{ 
  PAPI_init();

  switch(option)
    {
    case PAPI_GET_CLOCKRATE:
      return( _papi_system_info.mhz ); 
    case PAPI_GET_MAX_HWCTRS: 
      return( _papi_system_info.num_cntrs  ); 
    default:
      return(PAPI_EINVAL);
    }
} 

void PAPI_shutdown(void) 
{
  int i;
  
  for (i=0;i<PAPI_EVENTSET_MAP.totalSlots;i++) 
    {
      if (PAPI_EVENTSET_MAP.dataSlotArray[i]) 
	{
	  free_EventSet(PAPI_EVENTSET_MAP.dataSlotArray[i]);
	}
    }
  free(PAPI_EVENTSET_MAP.dataSlotArray);
}

static int handle_error(int PAPI_errorCode, char *errorMessage)
{
  if (PAPI_ERR_LEVEL)
    {
      fprintf(stderr, "%s", get_error_string(PAPI_errorCode));
      if (PAPI_errorCode==PAPI_ESYS)
        perror(errorMessage);
      if (errorMessage)
        fprintf(stderr, ": %s", errorMessage);
      fprintf(stderr,"\n");
      if (PAPI_ERR_LEVEL==PAPI_VERB_ESTOP)
        PAPI_shutdown();
    }
  return(PAPI_errorCode);
}

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

static char *get_error_string(int errorCode)
{
  if (errorCode>PAPI_OK)
    errorCode = PAPI_OK;
  errorCode = - errorCode;

  if ((errorCode < 0) || ( errorCode >= PAPI_NUM_ERRORS))
    errorCode = PAPI_EMISC;

  return(papi_errStr[errorCode]);
}

int PAPI_perror(int code, char *destination, int length)
{
  char *foo;

  foo = get_error_string(code);

  if (destination && (length >= 0))
    strncpy(destination,foo,length);
  else
    fprintf(stderr,"%s\n",foo);

  return(PAPI_OK);
}

