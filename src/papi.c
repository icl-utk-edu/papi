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

/* Global variables:  These will eventually be encapsulated into per thread structures. */

EventSetInfo *event_set_zero = NULL;
EventSetInfo *event_set_overflowing = NULL;

/* Our integer to EventSetInfo * mapping */

static DynamicArray PAPI_EVENTSET_MAP;    

/* Behavior of handle_error(). 
   Changed to the default behavior of PAPI_QUIET in PAPI_init
   after initialization is successful. */

static int PAPI_ERR_LEVEL = PAPI_VERB_ESTOP; 

/* Utility functions */

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

   /* Remember that EventSet zero is reserved. We allocate it here. */
   
   event_set_zero = (EventSetInfo *)malloc(sizeof(EventSetInfo));
   if (event_set_zero == NULL)
     {
     heck:
       free(PAPI_EVENTSET_MAP.dataSlotArray);
       memset(&PAPI_EVENTSET_MAP,0x00,sizeof(PAPI_EVENTSET_MAP));
       return(PAPI_ENOMEM);
     }
   memset(event_set_zero,0x00,sizeof(EventSetInfo));

   /* Allocate the machine dependent control block for EventSet zero. */

   event_set_zero->machdep = (void *)malloc(_papi_system_info.size_machdep);
   if (event_set_zero->machdep == NULL)
     {
     heck2:
       free(event_set_zero);
       event_set_zero = NULL;
       goto heck;
     }
   memset(event_set_zero->machdep,0x00,_papi_system_info.size_machdep);

   /* Initialize any global options stored in EventSet zero. */

   event_set_zero->domain.domain = _papi_system_info.default_domain;
   event_set_zero->granularity.granularity = _papi_system_info.default_granularity;

   /* Hook it into our data structure. */

   PAPI_EVENTSET_MAP.dataSlotArray[0] = event_set_zero;

   /* Here we initialize the goodies that help us keep track of multiple
      running eventsets. We don't need much... */

   event_set_zero->multistart.SharedDepth = (int *)malloc(_papi_system_info.num_cntrs*sizeof(int));
   if (event_set_zero->multistart.SharedDepth == NULL)
     {
     heck3:
       free(event_set_zero->machdep);
       goto heck2;
     }
   memset(event_set_zero->multistart.SharedDepth,0x0,_papi_system_info.num_cntrs*sizeof(int));
   event_set_zero->multistart.num_runners = 0;

   /* Call the substrate to fill in anything special. */

   retval = _papi_hwd_init(event_set_zero);
   if (retval < PAPI_OK)
     {
       free(event_set_zero->multistart.SharedDepth);
       goto heck3;
     }

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

  /* ESI->domain.domain = 0;
     ESI->granularity.granularity = 0; */

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

static int get_domain(PAPI_domain_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(opt->eventset);
  if(ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  opt->domain = ESI->domain.domain;
  return(PAPI_OK);
}

static int get_granularity(PAPI_granularity_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(opt->eventset);
  if(ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  opt->granularity = ESI->granularity.granularity;
  return(PAPI_OK);
}

/* Describe the event. name is input/output, eventcode is input/output,
   description is output */

typedef struct pre_info {
  char *name;
  unsigned int code;
  char *descr; } preset_info_t; 

static preset_info_t papi_preset_info[PAPI_MAX_PRESET_EVENTS] = { 
  { "PAPI_L1_DCM", 0, "Level 1 Data Cache Misses" },
  { "PAPI_L1_ICM", 1, "Level 1 Instruction Cache Misses" },
  { "PAPI_L2_DCM", 2, "Level 2 Data Cache Misses" },
  { "PAPI_L2_ICM", 3, "Level 2 Instruction Cache Misses" },
  { "PAPI_L3_DCM", 4, "Level 3 Data Cache Misses" },
  { "PAPI_L3_ICM", 5, "Level 3 Instruction Cache Misses" },
  { "PAPI_NULL", 6, NULL },
  { "PAPI_NULL", 7, NULL },
  { "PAPI_NULL", 8, NULL },
  { "PAPI_NULL", 9, NULL },
  { "PAPI_CA_SHR", 10, "Requests for Shared Cache Line" },
  { "PAPI_CA_CLN", 11, "Requests for Clean Cache Line" },
  { "PAPI_CA_INV", 12, "Cache Line Invalidation Requests" },
  { "PAPI_NULL", 13, NULL },
  { "PAPI_NULL", 14, NULL },
  { "PAPI_NULL", 15, NULL },
  { "PAPI_NULL", 16, NULL },
  { "PAPI_NULL", 17, NULL },
  { "PAPI_NULL", 18, NULL },
  { "PAPI_NULL", 19, NULL },
  { "PAPI_TLB_DM", 20, "Data Translation Lookaside Buffer Misses" },
  { "PAPI_TLB_IM", 21, "Instruction Translation Lookaside Buffer Misses" },
  { "PAPI_TLB_TOT", 22, "Total Translation Lookaside Buffer Misses" },
  { "PAPI_NULL",  23, NULL },
  { "PAPI_NULL",  24, NULL },
  { "PAPI_NULL",  25, NULL },
  { "PAPI_NULL",  26, NULL },
  { "PAPI_NULL",  27, NULL },
  { "PAPI_NULL",  28, NULL },
  { "PAPI_NULL",  29, NULL },
  { "PAPI_TLB_SD", 30, "Translation Lookaside Buffer Shootdowns" },
  { "PAPI_NULL",  31, NULL },
  { "PAPI_NULL",  32, NULL },
  { "PAPI_NULL",  33, NULL },
  { "PAPI_MEM_SCY",  34, "Cycles Stalled Waiting for Memory Access" },
  { "PAPI_MEM_RCY",  35, "Cycles Stalled Waiting for Memory Read" },
  { "PAPI_MEM_WCY",  36, "Cycles Stalled Waiting for Memory Write" },
  { "PAPI_STL_SCY",  37, "Cycles with No Instruction Issue" },
  { "PAPI_FUL_CYC",  38, "Cycles with Maximum Instruction Issue" },
  { "PAPI_NULL",  39, NULL },
  { "PAPI_NULL",  40, NULL },
  { "PAPI_NULL",  41, NULL },
  { "PAPI_BR_UCN", 42, "Unconditional Branch Instructions" },
  { "PAPI_BR_CN", 43, "Conditional Branch Instructions" },
  { "PAPI_BR_TKN", 44, "Conditional Branch Instructions Taken" }, 
  { "PAPI_BR_NTK", 45, "Conditional Branch Instructions Not Taken" }, 
  { "PAPI_BR_MSP", 46, "Conditional Branch Instructions Mispredicted" },
  { "PAPI_NULL",  47, NULL },
  { "PAPI_NULL",  48, NULL },
  { "PAPI_NULL",  49, NULL },
  { "PAPI_TOT_INS", 50, "Total Instructions" },
  { "PAPI_INT_INS", 51, "Integer Instructions" },
  { "PAPI_FP_INS", 52, "Floating Point Instructions" },
  { "PAPI_LD_INS", 53, "Load Instructions" },
  { "PAPI_SR_INS", 54, "Store Instructions" },
  { "PAPI_BR_INS", 55, "Branch Instructions" },
  { "PAPI_VEC_INS", 56, "Vector Instructions" },
  { "PAPI_FLOPS", 57, "Floating Point Instructions per Second" },
  { "PAPI_NULL",  58, NULL },
  { "PAPI_NULL",  59, NULL },
  { "PAPI_TOT_CYC",  60, "Total Cycles" },
  { "PAPI_MIPS", 61, "Millions of Instructions per Second" },
  { "PAPI_NULL",  62, NULL },
  { "PAPI_NULL",  63, NULL } };

int PAPI_describe_event(char *name, int *EventCode, char *description)
{ 
  int i;
  if ((*EventCode >= 0) && (*EventCode < PAPI_MAX_PRESET_EVENTS))
  { 
    strcpy(name,papi_preset_info[*EventCode].name);
    strcpy(description,papi_preset_info[*EventCode].descr);
    return(PAPI_OK);
  }
  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
  { if (strcmp(papi_preset_info[i].name,name) == 0)
    { if(description)
        strcpy(description,papi_preset_info[i].descr);
      *EventCode = papi_preset_info[i].code;
      return(PAPI_OK);
    }
  }
  return(PAPI_EINVAL);
}

int PAPI_query_event(int EventCode)
{ int retval;

  retval = _papi_hwd_query(EventCode);
  if (retval != PAPI_OK)
    return(handle_error(retval,"Event does not exist on this substrate"));

  DBG((stderr,"PAPI_query returns %d\n",retval));
  return(retval);
}

/* add_event checks to see whether the ESI structure has been 
   created already for this EventSet, adds the event */

int PAPI_add_event(int *EventSet, int EventCode) 
{ int retval,indextohw;
  EventSetInfo *ESI,*n = NULL;

  retval =  PAPI_init();
  if(retval < PAPI_OK) return retval;
  
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

  if(!(ESI->state & PAPI_STOPPED))
    {
      if (n) free(n);
      return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));
    }

  /* This returns index into the map array. Note that this routine
     increments ESI->NumberOfCounters. */

  retval = add_event(ESI,EventCode);
  if (retval < PAPI_OK)
    {
    heck:
      if (n)
        {
          free_EventSet(ESI);
          free(n);
        }
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

/* add_event checks for event already added, zeroes the counters
   values in the ESI structure. This function now returns the index
   of the slot in EventCodeArray and EventSelectArray which is used for
   the added EventCode. */

static int add_event(EventSetInfo *ESI, int EventCode)
{
  int k;

  if (ESI->NumberOfCounters == _papi_system_info.num_cntrs)
    return(PAPI_ECNFLCT);

  /* Check for duplicate events */

  k = lookup_EventCodeIndex(ESI, EventCode);
  if (k != PAPI_EINVAL)
    return(PAPI_ECNFLCT);

  /* Take the lowest empty slot */

  for (k=0;k<_papi_system_info.num_cntrs;k++) {
    if (ESI->EventCodeArray[k] == -1)
      break;
  }

  /* If no space, return error */

  if (k == _papi_system_info.num_cntrs)
    return(PAPI_ECNFLCT);

  /* Fill in everything but EventSelectArray, which is
     filled by the low level _papi_hwd_add_event call. */

  ESI->EventCodeArray[k] = EventCode;
  ESI->start[k]          = 0;
  ESI->stop[k]           = 0;
  ESI->latest[k]         = 0;
  ESI->NumberOfCounters++;

  /* Return the index of this event. */
  return(k);
}

static int remove_event(EventSetInfo *ESI, int EventCode)
{
  int k;

  /* Make sure the event is preset. */

  k = lookup_EventCodeIndex(ESI,EventCode);
  if (k < 0)
    return(k);

  /* Zero everything but EventSelectArray, which is
     zeroed by the low level _papi_hwd_rem_event call. */

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
  int retval, indextohw;

  /* check for pre-existing ESI */

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(*EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));


 if (!(ESI->state & PAPI_STOPPED))
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  /* This returns index into the map array.
     Note that this routine decrements ESI->NumberOfCounters. */

  retval = remove_event(ESI,EventCode);
  if (retval < PAPI_OK)
    return(handle_error(retval,NULL));

  indextohw = retval;

  /* Call the low level. This function needs ESI->EventSelectArray[indextohw]
     to determine which events to remove. */

  retval = _papi_hwd_rem_event(ESI, indextohw, EventCode);
  if (retval < PAPI_OK)
    return(handle_error(retval,NULL));

  /* Always clean up empty EventSets. */

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
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (!(ESI->state & PAPI_STOPPED))
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  retval = _papi_hwd_merge(ESI, event_set_zero);
  if (retval<PAPI_OK)
    return(handle_error(retval, NULL));

  ESI->state ^= PAPI_STOPPED;
  ESI->state |= PAPI_RUNNING;
  event_set_zero->multistart.num_runners++;

  DBG((stderr,"PAPI_start returns %d\n",retval));
  return(retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */
int PAPI_stop(int EventSet, unsigned long long *values)
{ 
  int retval
  EventSetInfo *ESI;

  ESI = lookup_EventSet(EventSet);
  if (ESI==NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (!(ESI->state & PAPI_RUNNING))
    return(handle_error(PAPI_EINVAL, "EventSet is not running"));

  retval = _papi_hwd_unmerge(ESI, event_set_zero);
  if (retval<PAPI_OK)
    return(handle_error(retval, NULL));

  if (values)
    memcpy(values,ESI->stop,ESI->NumberOfCounters*sizeof(unsigned long long));

  ESI->state ^= PAPI_RUNNING;
  ESI->state |= PAPI_STOPPED;
  event_set_zero->multistart.num_runners --;

#if defined(DEBUG)
  {
    int i;
    for (i=0;i<ESI->NumberOfCounters;i++)
    DBG((stderr,"PAPI_stop ESI->stop[%d]:\t%llu\n",i,ESI->stop[i]));
  }
#endif

  DBG((stderr,"PAPI_stop returns %d\n",retval));

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
      /* If we're not the only one running, then just
         read the current values into the ESI->start
         array. This holds the starting value for counters
         that are shared. */

      if (event_set_zero->multistart.num_runners > 1)
        {
          retval = _papi_hwd_read(ESI, event_set_zero, ESI->start);
          if (retval < PAPI_OK)
            return(handle_error(retval, NULL));
        }
      else
        {
          retval = _papi_hwd_reset(ESI);
          if (retval<PAPI_OK)
            return(handle_error(retval, NULL));
        }
    }
  else
    {
      memset(ESI->start,0x00,ESI->NumberOfCounters*sizeof(unsigned long long));
      memset(ESI->stop,0x00,ESI->NumberOfCounters*sizeof(unsigned long long));
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
      retval = _papi_hwd_read(ESI, event_set_zero, ESI->latest);
      if (retval<PAPI_OK)
        return(handle_error(retval, NULL));
      memcpy(values,ESI->latest,ESI->NumberOfCounters*sizeof(unsigned long long));
    }
  else
    {
      memcpy(values,ESI->stop,ESI->NumberOfCounters*sizeof(unsigned long long));
    }

#if defined(DEBUG)
  {
    int i;
    for (i=0;i<ESI->NumberOfCounters;i++)
    DBG((stderr,"PAPI_read values[%d]:\t%llu\n",i,values[i]));
  }
#endif

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
      retval = _papi_hwd_read(ESI, event_set_zero, ESI->latest);
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

        internal.domain.ESI = lookup_EventSet(ptr->domain.eventset);
        if (internal.domain.ESI == NULL)
          return(handle_error(PAPI_EINVAL,"No such EventSet"));

        if (!(internal.domain.ESI->state & PAPI_STOPPED))
          return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

        internal.domain.domain = dom;
        internal.domain.eventset = ptr->domain.eventset;
        retval = _papi_hwd_ctl(PAPI_SET_DOMAIN, &internal);
        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        internal.domain.ESI->domain.domain = dom;
        return(retval);
      }
    case PAPI_SET_GRANUL:
      {
        int grn = ptr->granularity.granularity;

        if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
          return(handle_error(PAPI_EINVAL,"Granularity out of range"));

        internal.granularity.ESI = lookup_EventSet(ptr->granularity.eventset);
        if (internal.granularity.ESI == NULL)
          return(handle_error(PAPI_EINVAL,"No such EventSet"));

        if (!(internal.granularity.ESI->state & PAPI_STOPPED))
          return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

        internal.granularity.granularity = grn;
        internal.granularity.eventset = ptr->granularity.eventset;
        retval = _papi_hwd_ctl(PAPI_SET_GRANUL, &internal);
        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        internal.granularity.ESI->granularity.granularity = grn;
        return(retval);
      }
    case PAPI_SET_DEFDOM:
      {
        int dom = ptr->domain.domain;

        if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
          return(handle_error(PAPI_EINVAL,"Domain out of range"));

        internal.domain.domain = dom;
        retval = _papi_hwd_ctl(PAPI_SET_DEFDOM, &internal);

        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        event_set_zero->domain.domain = dom;
        return(retval);
      }
    case PAPI_SET_DEFGRN:
      {
        int grn = ptr->granularity.granularity;

        if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
          return(handle_error(PAPI_EINVAL,"Granularity out of range"));

        internal.granularity.granularity = grn;
        retval = _papi_hwd_ctl(PAPI_SET_DEFGRN, &internal);

        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        event_set_zero->granularity.granularity = grn;
        return(retval);
      }
    case PAPI_SET_INHERIT:
      {
        internal.inherit.inherit = ptr->inherit.inherit;
        retval = _papi_hwd_ctl(PAPI_SET_INHERIT, &internal);
        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        event_set_zero->inherit.inherit = (ptr->inherit.inherit != 0);
        return(retval);
      }
    default:
      return(handle_error(PAPI_EINVAL,"Invalid option type"));
    }
}

int PAPI_get_opt(int option, PAPI_option_t *ptr) 
{ 
  int retval;

  retval = PAPI_init();
  if (retval < PAPI_OK)
    return retval;

  switch(option)
    {
    case PAPI_GET_CLOCKRATE:
      return(_papi_system_info.mhz);
    case PAPI_GET_MAX_HWCTRS:
      return(_papi_system_info.num_cntrs);
    case PAPI_GET_DEFDOM:
      return(event_set_zero->domain.domain);
    case PAPI_GET_DEFGRN:
      return(event_set_zero->granularity.granularity);
    case PAPI_GET_INHERIT:
      return(event_set_zero->inherit.inherit);
    case PAPI_GET_DOMAIN:
      if (ptr == NULL)
        return(handle_error(PAPI_EINVAL, "Invalid option pointer"));
      return(get_domain(&ptr->domain));
    case PAPI_GET_GRANUL:
      if (ptr == NULL)
        return(handle_error(PAPI_EINVAL, "Invalid option pointer"));
      return(get_granularity(&ptr->granularity));
    default:
      return(handle_error(PAPI_EINVAL,"Invalid option type"));
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

/* This function sets up an EventSet such that when it is PAPI_start()'ed, it
   begins to register overflows. This EventSet may only have multiple events
   in it, but only 1 can be an overflow trigger. Subsequent calls to PAPI_overflow
   replace earlier calls. To turn off overflow, set the handler to NULL. */

int PAPI_overflow(int EventSet, int EventCode, int threshold, int flags, PAPI_overflow_handler_t handler)
{
  int retval, index;
  EventSetInfo *ESI;
  EventSetOverflowInfo_t opt = { 0, };

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL)
     return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  if ((index = lookup_EventCodeIndex(ESI, EventCode)) < 0)
    return(handle_error(PAPI_ENOEVNT, "Hardware event is not in EventSet"));

  if (threshold < 0)
    return(handle_error(PAPI_EINVAL, "Threshold is too low.\n"));

  if (ESI->state & PAPI_OVERFLOWING)
    {
      if (threshold)
        return(handle_error(PAPI_EINVAL, "Overflow is already enabled for this EventSet.\n"));
    }
  else
    {
      if (handler == NULL)
        return(handle_error(PAPI_EINVAL, "Invalid handler address.\n"));
      if (threshold == 0)
        return(handle_error(PAPI_EINVAL, "Overflow is not enabled for this EventSet.\n"));
    }

  /* Set up the option structure for the low level */

  opt.deadline = threshold;
  opt.threshold = threshold;
  opt.EventIndex = index;
  opt.EventCode = EventCode;
  opt.flags = flags;
  opt.handler = handler;

  retval = _papi_hwd_set_overflow(ESI, &opt);
  if (retval < PAPI_OK)
    return(retval);

  /* Toggle the overflow flag */

  ESI->state ^= PAPI_OVERFLOWING;

  if (ESI->state & PAPI_OVERFLOWING)
    {
      /* Copy the machine independent options into the ESI */
      memcpy(&ESI->overflow, &opt, sizeof(EventSetOverflowInfo_t));
    }
  return(PAPI_OK);
}

static void dummy_handler(int EventSet, int count, int eventcode,
                          unsigned long long value, int *threshold, void *context)
{
  abort();
}

int PAPI_profil(void *buf, int bufsiz, caddr_t offset, int scale,
                int EventSet, int EventCode, int threshold, int flags)
{
  int retval, index;
  EventSetInfo *ESI;
  EventSetProfileInfo_t opt = { 0, };

  ESI = lookup_EventSet(EventSet);
  if(ESI == NULL)
     return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  if ((index = lookup_EventCodeIndex(ESI, EventCode)) < 0)
    return(handle_error(PAPI_ENOEVNT, "Hardware event is not in EventSet"));

  if (threshold < 0)
    return(handle_error(PAPI_EINVAL, "Threshold is too low.\n"));

  if (ESI->state & PAPI_PROFILING)
    {
      if (threshold)
        return(handle_error(PAPI_EINVAL, "Profiling is already enabled for this EventSet.\n"));
    }
  else
    {
      if (threshold == 0)
        return(handle_error(PAPI_EINVAL, "Profiling is not enabled for this EventSet.\n"));
    }

  /* Set up the option structure for the low level */

  opt.buf = buf;
  opt.bufsiz = bufsiz;
  opt.offset = offset;
  opt.scale = scale;
  opt.flags = flags;

  switch (flags)
    {
    case PAPI_PROFIL_POSIX:
    default:
      opt.divisor = 65536;
    }

  if (_papi_system_info.needs_profile_emul)
    retval = PAPI_overflow(EventSet, EventCode, threshold, 0, dummy_handler);
  else
    retval = _papi_hwd_set_profile(ESI, &opt);

  if (retval < PAPI_OK)
    return(retval);

  /* Toggle profiling flag */

  ESI->state ^= PAPI_PROFILING;

  if (ESI->state & PAPI_PROFILING)
    {
      /* Copy the machine independent options into the ESI */
      memcpy(&ESI->profile, &opt, sizeof(EventSetProfileInfo_t));
    }
  return(PAPI_OK);
}

int PAPI_set_granularity(int granularity)
{ PAPI_option_t ptr;

  ptr.defgranularity.granularity = granularity;
  return(PAPI_set_opt(PAPI_SET_GRANUL, &ptr));
}

int PAPI_set_domain(int domain)
{ PAPI_option_t ptr;

  ptr.defdomain.domain = domain;
  return(PAPI_set_opt(PAPI_SET_DOMAIN, &ptr));
}

int PAPI_add_pevent(int *EventSet, int code, void *inout)
{ int retval, indextohw;
  EventSetInfo *ESI, *n = NULL;
  
  retval = PAPI_init();
  if (retval < PAPI_OK) return retval;

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

  if(!(ESI->state & PAPI_STOPPED))
    {
      if (n) free(n);
      return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));
    }

  /* This returns index into the map array. Note that this routine
     increments ESI->NumberOfCounters. */

  retval = add_event(ESI,code);
  if (retval < PAPI_OK)
    {
    heck:
      if (n)
        {
          free_EventSet(ESI);
          free(n);
        }
      return(handle_error(retval,NULL));
    }

  indextohw = retval;
  retval = _papi_hwd_add_prog_event(ESI->machdep,indextohw,code,inout);
  if (retval < PAPI_OK)
    {
      remove_event(ESI,code);
      goto heck;
    }

  if (n)
    {
      retval = add_EventSet(ESI);
      if (retval < PAPI_OK)
        goto heck;

      *EventSet = ESI->EventSetIndex;
      DBG((stderr,"PAPI_add_pevent new EventSet in slot %d\n",*EventSet));
    }
  return(PAPI_OK);
}

int PAPI_add_events(int *EventSet, int *Events, int number)
{
  int i, retval;

  for (i=0;i<number;i++)
    {
      retval = PAPI_add_event(EventSet, Events[i]);
      if (retval<PAPI_OK) return(retval);
    }
  return(PAPI_OK);
}

int PAPI_rem_events(int *EventSet, int *Events, int number)
{
  int i, retval;
  EventSetInfo *ESI;

  if ((!EventSet) || (!Events))
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI=lookup_EventSet(*EventSet);
  if (!ESI)
    return(handle_error(PAPI_EINVAL, "Not a valid EventSet"));

#ifdef DEBUG
  /* Not necessary */
  if (ESI->NumberOfCounters == 0)
    return(handle_error(PAPI_EINVAL, "No events have been added"));
#endif

  if (number > ESI->NumberOfCounters)
    return(handle_error(PAPI_EINVAL, "Too many events requested"));

  for (i=0; i<number; i++)
    {
      retval=PAPI_rem_event(EventSet, Events[i]);
      if(retval<PAPI_OK) return(retval);
    }
  return(PAPI_OK);
}


int PAPI_list_events(int EventSet, int *Events, int *number)
{
  EventSetInfo *ESI;
  int i;

  if ((!Events) || (!number))
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI=lookup_EventSet(EventSet);
  if (!ESI)
    return(handle_error(PAPI_EINVAL, "Not a valid EventSet"));

#ifdef DEBUG
  /* Not necessary */
  if (ESI->NumberOfCounters == 0)
    return(handle_error(PAPI_EINVAL, "No events have been added"));
#endif

  for(i=0; i<ESI->NumberOfCounters; i++)
    {
      Events[i] = (PRESET_MASK ^ ESI->EventCodeArray[i]);
    }
  *number = ESI->NumberOfCounters;
  return(PAPI_OK);
}

int PAPI_save(void)
{
  return(PAPI_OK);
}

int PAPI_restore(void)
{
  return(PAPI_OK);
}

