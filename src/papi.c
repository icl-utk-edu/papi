#ifdef PTHREADS
#include <pthread.h>
#endif

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

static int expand_dynamic_array(DynamicArray *);
static int handle_error(int, char *);
static char *get_error_string(int);

/* EventSet handling functions */

static EventSetInfo *allocate_EventSet(void);
static int add_EventSet(DynamicArray *map, EventSetInfo *);
static EventSetInfo *lookup_EventSet(DynamicArray *map, int eventset);
static int remove_EventSet(DynamicArray *map, EventSetInfo *);
static void free_EventSet(EventSetInfo *);

/* Event handling functions */

static int add_event(EventSetInfo *ESI, int EventCode);
static int add_pevent(EventSetInfo *ESI, int EventCode, void *inout);
static int get_free_EventCodeIndex(EventSetInfo *ESI, int EventCode);
static int lookup_EventCodeIndex(EventSetInfo *ESI,int EventCode);
static int remove_event(EventSetInfo *ESI, int EventCode);

/* Global variables */

#ifdef PTHREADS
pthread_key_t theKey;
pthread_once_t once_block = PTHREAD_ONCE_INIT;
pthread_mutex_t global_mutex;
PAPI_notify_handler_t thread_notifier = NULL;
#elif defined SMPTHREADS
volatile void **thread_specific = NULL;
volatile int num_threads = 0;
volatile int once_block = 0;
volatile int global_mutex = 0;
PAPI_notify_handler_t thread_notifier = NULL;
#else
DynamicArray *PAPI_EVENTSET_MAP; /* Integer to EventSetInfo * mapping */
#endif
static int initialize_process_retval = 0xdeadbeef;

/* Behavior of handle_error() */

static int PAPI_ERR_LEVEL = PAPI_VERB_ESTOP; 

/* Our informative table */

static PAPI_preset_info_t papi_presets[PAPI_MAX_PRESET_EVENTS] = { 
  { "PAPI_L1_DCM", PAPI_L1_DCM, "Level 1 data cache misses", 0, NULL, 0 },
  { "PAPI_L1_ICM", PAPI_L1_ICM, "Level 1 instruction cache misses", 0, NULL, 0 },
  { "PAPI_L2_DCM", PAPI_L2_DCM, "Level 2 data cache misses", 0, NULL, 0 },
  { "PAPI_L2_ICM", PAPI_L2_ICM, "Level 2 instruction cache misses", 0, NULL, 0 },
  { "PAPI_L3_DCM", PAPI_L3_DCM, "Level 3 data cache misses", 0, NULL, 0 },
  { "PAPI_L3_ICM", PAPI_L3_ICM, "Level 3 instruction cache misses", 0, NULL, 0 },
  { "PAPI_L1_TCM", PAPI_L1_TCM, "Level 1 cache misses", 0, NULL, 0 },
  { "PAPI_L2_TCM", PAPI_L2_TCM, "Level 2 cache misses", 0, NULL, 0 },
  { "PAPI_L3_TCM", PAPI_L3_TCM, "Level 3 cache misses", 0, NULL, 0 },
  { "PAPI_CA_SNP", PAPI_CA_SNP, "Requests for a snoop", 0, NULL, 0 },
  { "PAPI_CA_SHR", PAPI_CA_SHR, "Requests for shared cache line", 0, NULL, 0 },
  { "PAPI_CA_CLN", PAPI_CA_CLN, "Requests for clean cache line", 0, NULL, 0 },
  { "PAPI_CA_INV", PAPI_CA_INV, "Requests for cache line invalidation", 0, NULL, 0 },
  { "PAPI_CA_ITV", PAPI_CA_ITV, "Requests for cache line intervention", 0, NULL, 0 },
  { "PAPI_L3_LDM", PAPI_L3_LDM, "Level 3 load misses", 0, NULL, 0 },
  { "PAPI_L3_STM", PAPI_L3_STM, "Level 3 store misses", 0, NULL, 0 },
  { "PAPI_BRU_IDL", PAPI_BRU_IDL, "Cycles branch units are idle", 0, NULL, 0 },
  { "PAPI_FXU_IDL", PAPI_FXU_IDL, "Cycles integer units are idle", 0, NULL, 0 },
  { "PAPI_FPU_IDL", PAPI_FPU_IDL, "Cycles floating point units are idle", 0, NULL, 0 },
  { "PAPI_LSU_IDL", PAPI_LSU_IDL, "Cycles load/store units are idle", 0, NULL, 0 },
  { "PAPI_TLB_DM", PAPI_TLB_DM, "Data translation lookaside buffer misses", 0, NULL, 0 },
  { "PAPI_TLB_IM", PAPI_TLB_IM, "Instruction translation lookaside buffer misses", 0, NULL, 0 },
  { "PAPI_TLB_TL", PAPI_TLB_TL, "Total translation lookaside buffer misses", 0, NULL, 0 },
  { "PAPI_L1_LDM", PAPI_L1_LDM, "Level 1 load misses", 0, NULL, 0 },
  { "PAPI_L1_STM", PAPI_L1_STM, "Level 1 store misses", 0, NULL, 0 },
  { "PAPI_L2_LDM", PAPI_L2_LDM, "Level 2 load misses", 0, NULL, 0 },
  { "PAPI_L2_STM", PAPI_L2_STM, "Level 2 store misses", 0, NULL, 0 },
  { "PAPI_BTAC_M", PAPI_BTAC_M, "Branch target address cache misses", 0, NULL, 0 },
  { "PAPI_PRF_DM", PAPI_PRF_DM, "Data prefetch cache misses", 0, NULL, 0 },
  { NULL, PAPI_NULL, "unused", 0, NULL, 0 },
  { "PAPI_TLB_SD", PAPI_TLB_SD, "Translation lookaside buffer shootdowns", 0, NULL, 0 },
  { "PAPI_CSR_FAL", PAPI_CSR_FAL, "Failed store conditional instructions", 0, NULL, 0 },
  { "PAPI_CSR_SUC", PAPI_CSR_SUC, "Successful store conditional instructions", 0, NULL, 0 },
  { "PAPI_CSR_TOT", PAPI_CSR_TOT, "Total store conditional instructions", 0, NULL, 0 },
  { "PAPI_MEM_SCY", PAPI_MEM_SCY, "Cycles Stalled Waiting for Memory Access", 0, NULL, 0 },
  { "PAPI_MEM_RCY", PAPI_MEM_RCY, "Cycles Stalled Waiting for Memory Read", 0, NULL, 0 },
  { "PAPI_MEM_WCY", PAPI_MEM_WCY, "Cycles Stalled Waiting for Memory Write", 0, NULL, 0 },
  { "PAPI_STL_ICY", PAPI_STL_ICY, "Cycles with no Instruction Issue", 0, NULL, 0 },
  { "PAPI_FUL_ICY", PAPI_FUL_ICY, "Cycles with maximum Instruction Issue", 0, NULL, 0 },
  { "PAPI_STL_CCY", PAPI_STL_CCY, "Cycles with no instructions completed", 0, NULL, 0 },
  { "PAPI_FUL_CCY", PAPI_FUL_CCY, "Cycles with maximum instructions completed", 0, NULL, 0 },
  { NULL, 0, NULL, 0, NULL, 0 },
  { "PAPI_BR_UCN", PAPI_BR_UCN, "Unconditional Branch Instructions", 0, NULL, 0 },
  { "PAPI_BR_CN", PAPI_BR_CN, "Conditional Branch Instructions", 0, NULL, 0 },
  { "PAPI_BR_TKN", PAPI_BR_TKN, "Conditional Branch Instructions Taken", 0, NULL, 0 }, 
  { "PAPI_BR_NTK", PAPI_BR_NTK, "Conditional Branch Instructions Not Taken", 0, NULL, 0 }, 
  { "PAPI_BR_MSP", PAPI_BR_MSP, "Conditional Branch Instructions Mispredicted", 0, NULL, 0 },
  { "PAPI_BR_PRC", PAPI_BR_PRC, "Conditional branch instructions correctly predicted", 0, NULL, 0 },
  { "PAPI_FMA_INS", PAPI_FMA_INS, "FMA instructions completed", 0, NULL, 0 },
  { "PAPI_TOT_IIS", PAPI_TOT_IIS, "Instructions issued", 0, NULL, 0 },
  { "PAPI_TOT_INS", PAPI_TOT_INS, "Instructions completed", 0, NULL, 0 },
  { "PAPI_INT_INS", PAPI_INT_INS, "Integer Instructions", 0, NULL, 0 },
  { "PAPI_FP_INS", PAPI_FP_INS, "Floating Point Instructions", 0, NULL, 0 },
  { "PAPI_LD_INS", PAPI_LD_INS, "Load Instructions", 0, NULL, 0 },
  { "PAPI_SR_INS", PAPI_SR_INS, "Store Instructions", 0, NULL, 0 },
  { "PAPI_BR_INS", PAPI_BR_INS, "Branch Instructions", 0, NULL, 0 },
  { "PAPI_VEC_INS", PAPI_VEC_INS, "Vector Instructions", 0, NULL, 0 },
  { "PAPI_FLOPS", PAPI_FLOPS, "Floating point instructions per second", 0, NULL, 0 },
  { "PAPI_RES_STL", PAPI_RES_STL, "Cycles stalled on any resource", 0, NULL, 0 },
  { "PAPI_FP_STAL", PAPI_FP_STAL, "Cycles the FP unit(s) are stalled", 0, NULL, 0 },
  { "PAPI_TOT_CYC", PAPI_TOT_CYC, "Total cycles", 0, NULL, 0 },
  { "PAPI_IPS", PAPI_IPS, "Instructions per second", 0, NULL, 0 },
  { "PAPI_LST_INS", PAPI_LST_INS, "Load/store instructions completed", 0, NULL, 0 },
  { "PAPI_SYC_INS", PAPI_SYC_INS, "Synchronization instructions completed", 0, NULL, 0 }
};

/* Utility functions */

static DynamicArray *allocate_eventset_map(void)
{
  DynamicArray *map;

  /* Allocate and clear the Dynamic Array structure */
  
  map = (DynamicArray *)malloc(sizeof(DynamicArray));
  if (map == NULL)
    return(NULL);
  memset(map,0x00,sizeof(DynamicArray));

  /* Allocate space for the EventSetInfo pointers */

  map->dataSlotArray = 
    (EventSetInfo **)malloc(PAPI_INIT_SLOTS*sizeof(EventSetInfo *));
  if(map->dataSlotArray == NULL) 
    {
      free(map);
      return(NULL);
    }
  memset(map->dataSlotArray,0x00, 
	 PAPI_INIT_SLOTS*sizeof(EventSetInfo *));

  map->totalSlots = PAPI_INIT_SLOTS;
  map->availSlots = PAPI_INIT_SLOTS - 1;
  map->fullSlots  = 1;
  map->lowestEmptySlot = 1;
  
  return(map);
}

static EventSetInfo *allocate_master_eventset(void)
{
  EventSetInfo *master;
  
  /* Remember that EventSet zero is reserved. We allocate it here. */
  
  master = (EventSetInfo *)malloc(sizeof(EventSetInfo));
  if (master == NULL)
    return(NULL);
  memset(master,0x00,sizeof(EventSetInfo));
  
  /* Allocate the machine dependent control block for EventSet zero. */
  
  master->machdep = (void *)malloc(_papi_system_info.size_machdep);
  if (master->machdep == NULL)
    {
      free(master);
      return(NULL);
    }
  memset(master->machdep,0x00,_papi_system_info.size_machdep);
  
  /* Allocate the holding area for the global counter values */
  
  master->hw_start = (long long *)malloc(_papi_system_info.num_cntrs*sizeof(long long));
  if (master->hw_start == NULL)
    {
      free(master->machdep);
      free(master);
      return(NULL);
    }
  memset(master->hw_start,0x00,_papi_system_info.num_cntrs*sizeof(long long));
   
  /* Here we initialize the goodies that help us keep track of multiple
     running eventsets. We don't need much... */
  
  master->multistart.SharedDepth = (int *)malloc(_papi_system_info.num_cntrs*sizeof(int));
  if (master->multistart.SharedDepth == NULL)
    {
      free(master->hw_start);
      free(master->machdep);
      free(master);
      return(NULL);
     }
  memset(master->multistart.SharedDepth,0x0,_papi_system_info.num_cntrs*sizeof(int));
  return(master);
}

static int initialize_thread(void)
{
  int retval;
  DynamicArray *map;
  EventSetInfo *master;
  
#ifdef PTHREADS
  if (pthread_getspecific(theKey))
    {
      fprintf(stderr,"Thread 0x%x already initialized\n",(unsigned int)pthread_self());
      abort();
    }
  if (thread_notifier)
    {
      int self = pthread_self();
      thread_notifier(PAPI_THREAD_CREATE, &self);
    }
#elif defined(SMPTHREADS)
  if (thread_specific[mp_my_threadnum()])
    {
      fprintf(stderr,"Thread 0x%x already initialized\n",(unsigned int)pthread_self());
      abort();
    }
  if (thread_notifier)
    {
      int self = mp_my_threadid();
      thread_notifier(PAPI_THREAD_CREATE, &self);
    }
#endif  

  if ((map = allocate_eventset_map()) == NULL)
    return(PAPI_ENOMEM);

  if ((master = allocate_master_eventset()) == NULL)
    return(PAPI_ENOMEM);

  /* Call the substrate to fill in anything special. */
  
  retval = _papi_hwd_init(master);
  if (retval)
    return(retval);

  /* Initialize any global options stored in EventSet zero. */
  
  master->domain.domain = _papi_system_info.default_domain;
  master->granularity.granularity = _papi_system_info.default_granularity;

  /* Hook it into our data structure. */
  
  map->dataSlotArray[0] = master;
  
#ifdef PTHREADS
  /* Set this map to be per thread */

  if ((retval = pthread_setspecific(theKey, (void *)map)))
    return(retval);
#else
  PAPI_EVENTSET_MAP = map;
#endif
  
  return(PAPI_OK);
}

static void initialize_process(void)
{
  int i;

#ifdef PTHREADS
  initialize_process_retval = pthread_mutex_init(&global_mutex, NULL);
  if (initialize_process_retval)
    return;
  
  initialize_process_retval = pthread_key_create(&theKey, NULL);
  if (initialize_process_retval)
    return;
#elif defined(SMPTHREADS)
  if (__lock_test_and_set(&once_block, 1) != 0)
    {
      DBG((stderr,"initialize_process(): thread %d booted.\n",mp_my_threadnum()));
      return;
    }
  thread_specific = (void **)malloc(_papi_system_info.hw_info.totalcpus * sizeof(void *));
  memset(thread_specific,0x0,_papi_system_info.hw_info.totalcpus * sizeof(void *));
#endif

  initialize_process_retval = _papi_hwd_init_global();
  if (initialize_process_retval)
    return;

  /* Set up the query structure */
  
  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    if (papi_presets[i].event_name) /* If the preset is part of the API */
      papi_presets[i].avail = _papi_hwd_query(papi_presets[i].event_code ^ PRESET_MASK,
					      &papi_presets[i].flags,
					      &papi_presets[i].event_note);
}

int _papi_hwi_initialize(DynamicArray **map)
{
  int retval;

#ifdef PTHREADS
  pthread_once(&once_block,initialize_process);
  if (initialize_process_retval == PAPI_OK)
    {
      *map = pthread_getspecific(theKey); 
      if (*map)
	{
	  DBG((stderr,"Existing thread 0x%x detected, data is at %p\n",(int)pthread_self(),*map));
	  return(PAPI_OK);
	}
      DBG((stderr,"New thread 0x%x detected\n",(int)pthread_self()));
      retval = initialize_thread();
      if (retval == PAPI_OK)
	{
	  *map = pthread_getspecific(theKey); 
	  DBG((stderr,"New data for thread 0x%x is at %p\n",(int)pthread_self(),*map));
	}
      return(retval);
    }
#elif defined(SMPTHREADS)
  initialize_process();
  if (initialize_process_retval == PAPI_OK)
    {
      while (__lock_test_and_set(&global_mutex, 1, thread_specific) != 0);

      *map = (DynamicArray *)thread_specific[mp_my_threadnum()]; 
      if (*map)
	{
	  DBG((stderr,"Existing thread 0x%x detected, data is at %p\n",
	       (int)mp_my_threadnum(),*map));
	  return(PAPI_OK);
	}

      DBG((stderr,"New thread 0x%x detected\n",(int)pthread_self()));
      retval = initialize_thread();
      if (retval == PAPI_OK)
	{
	  *map = (DynamicArray *)thread_specific[mp_my_threadnum()];  
	  DBG((stderr,"New data for thread 0x%x is at %p\n",(int)pthread_self(),*map));
	}
      return(retval);
    }
#else
  if (initialize_process_retval == PAPI_OK)
    {
      *map = PAPI_EVENTSET_MAP;
      return(PAPI_OK);
    }
  if (initialize_process_retval == 0xdeadbeef)
    {
      initialize_process();
      if (initialize_process_retval == PAPI_OK)
	{
	  retval = initialize_thread();
	  if (retval == PAPI_OK)
	    {
	      *map = PAPI_EVENTSET_MAP;
	    }
	  return(retval);
	}
    }
#endif
  return(initialize_process_retval);
}

int PAPI_init(void)
{ 
  DynamicArray *map;
  return(_papi_hwi_initialize(&map)); 
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
  ESI->sw_stop = (long long *)malloc(max_counters*sizeof(long long)); 
  ESI->hw_start = (long long *)malloc(max_counters*sizeof(long long));
  ESI->EventInfoArray = (EventInfo_t *)malloc(max_counters*sizeof(EventInfo_t));

  if ((ESI->machdep        == NULL )  || 
      (ESI->sw_stop           == NULL )  || 
      (ESI->hw_start         == NULL )  ||
      (ESI->EventInfoArray == NULL ))
    {
      if (ESI->machdep)        free(ESI->machdep);
      if (ESI->sw_stop)           free(ESI->sw_stop); 
      if (ESI->hw_start)         free(ESI->hw_start);
      if (ESI->EventInfoArray) free(ESI->EventInfoArray);
      free(ESI);
      return(NULL);
    }
  memset(ESI->machdep,       0x00,_papi_system_info.size_machdep);
  memset(ESI->sw_stop,          0x00,max_counters*sizeof(long long)); 
  memset(ESI->hw_start,        0x00,max_counters*sizeof(long long));

  for (i=0;i<max_counters;i++)
    {
      ESI->EventInfoArray[i].code = PAPI_NULL;
      ESI->EventInfoArray[i].selector = 0;
      ESI->EventInfoArray[i].command = NOT_DERIVED;
      ESI->EventInfoArray[i].operand_index = -1;
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
  if (ESI->EventInfoArray) free(ESI->EventInfoArray);
  if (ESI->machdep)        free(ESI->machdep);
  if (ESI->sw_stop)        free(ESI->sw_stop); 
  if (ESI->hw_start)       free(ESI->hw_start);
#ifdef DEBUG
  memset(ESI,0x00,sizeof(EventSetInfo));
#endif
  free(ESI);
}

static int add_EventSet(DynamicArray *map, EventSetInfo *ESI)
{
  int i, errorCode;

  /* Update the values for lowestEmptySlot, num of availSlots */

  ESI->EventSetIndex = map->lowestEmptySlot;
  map->dataSlotArray[ESI->EventSetIndex] = ESI;
  map->availSlots--;
  map->fullSlots++; 

  if (map->availSlots == 0)
    {
      errorCode = expand_dynamic_array(map);
      if (errorCode!=PAPI_OK) 
	return(errorCode);
    }

  i = ESI->EventSetIndex + 1;
  while (map->dataSlotArray[i]) i++;
  DBG((stderr,"Empty slot for EventSetInfo at %d\n",i));
  map->lowestEmptySlot = i;
 
  return(PAPI_OK);
}

static int get_domain(DynamicArray *map, PAPI_domain_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(map, opt->eventset);
  if(ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  opt->domain = ESI->domain.domain;
  return(PAPI_OK);
}

static int get_granularity(DynamicArray *map, PAPI_granularity_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(map, opt->eventset);
  if(ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  opt->granularity = ESI->granularity.granularity;
  return(PAPI_OK);
}

int PAPI_query_event(int EventCode)
{ 
  int retval;
  if ((retval = PAPI_init()) < PAPI_OK)
    return retval;

  if (EventCode & PRESET_MASK)
    { 
      EventCode ^= PRESET_MASK;
      if (EventCode >= PAPI_MAX_PRESET_EVENTS)
	return(handle_error(PAPI_EINVAL,"Event is not a valid preset"));
	
      if (papi_presets[EventCode].avail)
	return(PAPI_OK);
      else
	return(PAPI_EINVAL);
    }
  return(handle_error(PAPI_EINVAL,"Event is not a valid preset"));
}

int PAPI_query_event_verbose(int EventCode, PAPI_preset_info_t *info)
{ 
  int retval;
  if ((retval = PAPI_init()) < PAPI_OK)
    return retval;

  if (info == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (EventCode & PRESET_MASK)
    { 
      EventCode ^= PRESET_MASK;
      if (EventCode >= PAPI_MAX_PRESET_EVENTS)
	return(handle_error(PAPI_EINVAL,"Event is not a valid preset"));
	
      if (papi_presets[EventCode].avail)
	{
	  memcpy(info,&papi_presets[EventCode],sizeof(PAPI_preset_info_t));
	  return(PAPI_OK);
	}
      else
	return(PAPI_EINVAL);
    }
  return(handle_error(PAPI_EINVAL,"Event is not a valid preset"));
}

const PAPI_preset_info_t *PAPI_query_all_events_verbose(void)
{
  if (PAPI_init() < PAPI_OK)
    return NULL;

  return(papi_presets);
}

int PAPI_event_code_to_name(int EventCode, char *out)
{
  if (out == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (EventCode & PRESET_MASK)
    { 
      EventCode ^= PRESET_MASK;
      if ((EventCode >= PAPI_MAX_PRESET_EVENTS) || (papi_presets[EventCode].event_name == NULL))
	return(handle_error(PAPI_EINVAL,"Event is not a valid preset"));
	
      strcpy(out,papi_presets[EventCode].event_name);
      return(PAPI_OK);
    }
  return(handle_error(PAPI_EINVAL,"Event is not a valid preset"));
}

int PAPI_event_name_to_code(char *in, int *out)
{
  int i;
  
  if ((in == NULL) || (out == NULL))
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    {
      if ((papi_presets[i].event_name) && (strcasecmp(papi_presets[i].event_name,in) == 0))
	{ 
	  *out = papi_presets[i].event_code;
	  return(PAPI_OK);
	}
    }
  return(handle_error(PAPI_EINVAL,"Event is not a valid preset"));
}

int PAPI_add_pevent(int *EventSet, int code, void *inout)
{ 
  EventSetInfo *ESI, *n = NULL;
  INIT_MAP;

  retval = PAPI_init();
  if (retval < PAPI_OK) return retval;

  /* Is the EventSet already in existence? */

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(map, *EventSet);
  if (ESI == NULL)
    {
      /* Well, then allocate a new one. Use n to keep track of a NEW EventSet */

      n = allocate_EventSet();
      if (n == NULL)
        return(handle_error(PAPI_ENOMEM,"Error allocating memory for new EventSet"));
      ESI = n;
    }

  /* Of course, it must be stopped in order to modify it. */

  if (!(ESI->state & PAPI_STOPPED))
    {
      if (n) free(n);
      return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));
    }

  /* Now do the magic. */

  retval = add_pevent(ESI,code,inout);
  if (retval < PAPI_OK)
    {
    heck:
      if (n) free_EventSet(ESI);
      return(handle_error(retval,NULL));
    }

  /* If it's a new one, add it to the global table */

  if (n)
    {
      retval = add_EventSet(map, ESI);
      if (retval < PAPI_OK)
        goto heck;

      *EventSet = ESI->EventSetIndex;
      DBG((stderr,"PAPI_add_pevent new EventSet in slot %d\n",*EventSet));
    }
  return(retval);
}

int PAPI_create_eventset(int *EventSet)
{
  EventSetInfo *ESI;
  INIT_MAP;

  retval = PAPI_init();
  if (retval < PAPI_OK) return retval;
  
  /* Is the EventSet already in existence? */
  
  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (*EventSet != PAPI_NULL)
    return(handle_error(PAPI_EINVAL, "EventSet must be initialized to PAPI_NULL"));

  /* Well, then allocate a new one. Use n to keep track of a NEW EventSet */
  
  ESI = allocate_EventSet();
  if (ESI == NULL)
    return(handle_error(PAPI_ENOMEM,"Error allocating memory for new EventSet"));

  /* Add it to the global table */

  retval = add_EventSet(map, ESI);
  if (retval < PAPI_OK)
    {
      free_EventSet(ESI);
      return(handle_error(retval,"Error adding EventSet to global array"));
    }
  
  *EventSet = ESI->EventSetIndex;
  DBG((stderr,"PAPI_add_event new EventSet in slot %d\n",*EventSet));

  return(retval);
}

int PAPI_add_event(int *EventSet, int EventCode) 
{ 
  EventSetInfo *ESI, *n = NULL;
  INIT_MAP;

  retval = PAPI_init();
  if (retval < PAPI_OK) return retval;
  
  /* Is the EventSet already in existence? */
  
  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(map, *EventSet);
  if (ESI == NULL)
    {
      /* Well, then allocate a new one. Use n to keep track of a NEW EventSet */

      n = allocate_EventSet();
      if (n == NULL)
	return(handle_error(PAPI_ENOMEM, "Error allocating memory for new EventSet"));
      ESI = n;
    }

  /* Of course, it must be stopped in order to modify it. */

  if (!(ESI->state & PAPI_STOPPED))
    {
      if (n) free(n);
      return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));
    }

  /* Now do the magic. */

  retval = add_event(ESI,EventCode);
  if (retval < PAPI_OK)
    {
    heck:
      if (n) free_EventSet(ESI);
      return(handle_error(retval,NULL));
    }

  /* If it's a new one, add it to the global table */

  if (n)
    {
      retval = add_EventSet(map, ESI);
      if (retval < PAPI_OK)
	goto heck;

      *EventSet = ESI->EventSetIndex;
      DBG((stderr,"PAPI_add_event new EventSet in slot %d\n",*EventSet));
    }
  return(retval);
}

/* This function returns the index of the EventCode or error */
/* Index to what? The index to everything stored EventCode in the */
/* EventSet. */  

static int get_free_EventCodeIndex(EventSetInfo *ESI, int EventCode)
{
  int k;
  int lowslot = -1;
  
  /* Check for duplicate events and get the lowest empty slot */
  
  for (k=0;k<_papi_system_info.num_cntrs;k++) 
    {
      if (ESI->EventInfoArray[k].code == EventCode)
	return(PAPI_ECNFLCT);
      if ((ESI->EventInfoArray[k].code == PAPI_NULL) && (lowslot == -1))
	lowslot = k;
    }
  
  if (lowslot != -1)
    return(lowslot);
  else
    return(PAPI_ECNFLCT);
}

/* This function returns the index of the EventCode or error */
/* Index to what? The index to everything stored EventCode in the */
/* EventSet. */  

static int lookup_EventCodeIndex(EventSetInfo *ESI, int EventCode)
{
  int i;

  for(i=0;i<_papi_system_info.num_cntrs;i++) 
    { 
      if (ESI->EventInfoArray[i].code == EventCode) 
	return(i);
    }

  return(PAPI_EINVAL);
} 

static EventSetInfo *lookup_EventSet(DynamicArray *map, int eventset)
{
  if ((eventset >= 1) && (eventset < map->totalSlots))
    return(map->dataSlotArray[eventset]);
  else
    return(NULL);
}

/* This function only removes empty EventSets */

static int remove_EventSet(DynamicArray *map, EventSetInfo *ESI)
{
  int i;

  assert(ESI->NumberOfCounters == 0);

  i = ESI->EventSetIndex;

  free_EventSet(ESI);

  /* do bookkeeping for PAPI_EVENTSET_MAP */

  map->dataSlotArray[i] = NULL;
  if (map->lowestEmptySlot < i)
    map->lowestEmptySlot = i;
  map->availSlots++;
  map->fullSlots--;

  return(PAPI_OK);
}

static int cleanup_EventSet(DynamicArray *map, EventSetInfo *ESI)
{
  int i;

  /* first remove all of the Events from this EventSet*/
  /* ignore return vals */

  for(i=0;i<_papi_system_info.num_cntrs;i++) 
    {
      remove_event(ESI,ESI->EventInfoArray[i].code);
    }

  return(remove_EventSet(map, ESI));
}

static int add_event(EventSetInfo *ESI, int EventCode)
{
  int hwindex, retval;

  /* Make sure the event is not present and get a free slot. */

  retval = get_free_EventCodeIndex(ESI,EventCode);
  if (retval < PAPI_OK)
    return(retval);
    
  /* Fill in machine depending info including the EventInfoArray. */

  hwindex = retval;
  retval = _papi_hwd_add_event(ESI,hwindex,EventCode);
  if (retval < PAPI_OK)
    return(retval);

  /* Initialize everything left over. */

  /* ESI->sw_stop[hwindex]     = 0; */
  ESI->hw_start[hwindex]   = 0;
  ESI->NumberOfCounters++;

  return(retval);
}

static int add_pevent(EventSetInfo *ESI, int EventCode, void *inout)
{
  int hwindex, retval;

  /* Make sure the event is not present and get a free slot. */

  retval = get_free_EventCodeIndex(ESI,EventCode);
  if (retval < PAPI_OK)
    return(retval);
    
  /* Fill in machine depending info including the EventInfoArray. */

  hwindex = retval;
  retval = _papi_hwd_add_prog_event(ESI,hwindex,EventCode,inout);
  if (retval < PAPI_OK)
    return(retval);

  /* Initialize everything left over. */

  /* ESI->sw_stop[hwindex]     = 0; */
  ESI->hw_start[hwindex]   = 0;
  ESI->NumberOfCounters++;

  return(retval);
}

static int remove_event(EventSetInfo *ESI, int EventCode)
{
  int retval, hwindex;

  /* Make sure the event is preset. */

  retval = lookup_EventCodeIndex(ESI,EventCode);
  if (retval < PAPI_OK)
    return(retval);

  hwindex = retval;
  retval = _papi_hwd_rem_event(ESI,hwindex,EventCode);
  if (retval < PAPI_OK)
    return(retval);

  /* Zero everything left over. */

  ESI->EventInfoArray[hwindex].code = PAPI_NULL;
  ESI->EventInfoArray[hwindex].selector = 0;
  ESI->EventInfoArray[hwindex].command = NOT_DERIVED;
  ESI->EventInfoArray[hwindex].operand_index = -1;
  /* ESI->sw_stop[hwindex]           = 0; */
  ESI->hw_start[hwindex]         = 0;
  ESI->NumberOfCounters--;

  return(retval);
}

int PAPI_rem_event(int *EventSet, int EventCode)
{
  EventSetInfo *ESI;
  INIT_MAP;

  /* check for pre-existing ESI */

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(map, *EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

 if (!(ESI->state & PAPI_STOPPED))
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  retval = remove_event(ESI,EventCode);
  if (retval < PAPI_OK)
    return(handle_error(retval,NULL));

  return(retval);
}

int PAPI_destroy_eventset(int *EventSet)
{
  EventSetInfo *ESI;
  INIT_MAP;

  /* check for pre-existing ESI */

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(map, *EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  if (!(ESI->state & PAPI_STOPPED))
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  if (ESI->NumberOfCounters)
    return(handle_error(PAPI_EINVAL, "EventSet is not empty"));

  remove_EventSet(map, ESI);
  *EventSet = PAPI_NULL;

  return(PAPI_OK);
}

/* simply checks for valid EventSet, calls substrate start() call */

int PAPI_start(int EventSet)
{ 
  int i;
  EventSetInfo *ESI;
  INIT_MAP;

  ESI = lookup_EventSet(map, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (!(ESI->state & PAPI_STOPPED))
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  /* Short circuit this stuff if there's nothing running */

  if (master_event_set->multistart.num_runners == 0)
    {
      for (i=0;i<_papi_system_info.num_cntrs;i++)
	{
	  ESI->hw_start[i] = 0;
	  master_event_set->hw_start[i] = 0;
	}
    }

  /* If overflowing is enabled, turn it on */
  
  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_start_overflow_timer(ESI, master_event_set);
      if (retval < PAPI_OK)
	return(PAPI_EBUG);
      master_event_set->event_set_overflowing = ESI;
    }

  retval = _papi_hwd_merge(ESI, master_event_set);
  if (retval != PAPI_OK)
    return(handle_error(retval, NULL));

  ESI->state ^= PAPI_STOPPED;
  ESI->state |= PAPI_RUNNING;
  master_event_set->multistart.num_runners++;

  DBG((stderr,"PAPI_start returns %d\n",retval));
  return(retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */

int PAPI_stop(int EventSet, long long *values)
{ 
  EventSetInfo *ESI;
  INIT_MAP;

  ESI = lookup_EventSet(map, EventSet);
  if (ESI==NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (!(ESI->state & PAPI_RUNNING))
    return(handle_error(PAPI_EINVAL, "EventSet is not running"));

  retval = _papi_hwd_read(ESI, master_event_set, ESI->sw_stop);
  if (retval != PAPI_OK)
    return(handle_error(retval,NULL)); 

  /* If overflowing is enabled, turn it off */

  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_stop_overflow_timer(ESI, master_event_set);
      if (retval < PAPI_OK)
	return(PAPI_EBUG);
      master_event_set->event_set_overflowing = NULL;
    }
  
  retval = _papi_hwd_unmerge(ESI, master_event_set);
  if (retval != PAPI_OK)
    return(handle_error(retval, NULL));

  if (values)
    memcpy(values,ESI->sw_stop,ESI->NumberOfCounters*sizeof(long long)); 

  ESI->state ^= PAPI_RUNNING;
  ESI->state |= PAPI_STOPPED;
  master_event_set->multistart.num_runners --;

#if defined(DEBUG)
  {
    int i;
    for (i=0;i<ESI->NumberOfCounters;i++)
      DBG((stderr,"PAPI_stop ESI->sw_stop[%d]:\t%llu\n",i,ESI->sw_stop[i]));
  }
#endif 

  DBG((stderr,"PAPI_stop returns %d\n",retval));

  return(retval);
}

int PAPI_reset(int EventSet)
{ 
  EventSetInfo *ESI;
  INIT_MAP;

  ESI = lookup_EventSet(map, EventSet);
  if(ESI == NULL) 
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (ESI->state & PAPI_RUNNING)
    {
      /* If we're not the only one running, then just
         read the current values into the ESI->start
         array. This holds the starting value for counters
         that are shared. */

      retval = _papi_hwd_reset(ESI, master_event_set);
      if (retval != PAPI_OK)
	return(handle_error(retval, NULL));
    }
  else
    {
      memset(ESI->sw_stop,0x00,ESI->NumberOfCounters*sizeof(long long)); 
    }

  DBG((stderr,"PAPI_reset returns %d\n",retval));
  return(retval);
}

int PAPI_read(int EventSet, long long *values)
{ 
  EventSetInfo *ESI;
  INIT_MAP;

  ESI = lookup_EventSet(map, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state & PAPI_RUNNING)
    {
      retval = _papi_hwd_read(ESI, master_event_set, values);
      if (retval != PAPI_OK)
        return(handle_error(retval, NULL));
    }
  else
    {
      memcpy(values,ESI->sw_stop,ESI->NumberOfCounters*sizeof(long long)); 
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

int PAPI_accum(int EventSet, long long *values)
{ 
  EventSetInfo *ESI;
  int i;
  long long a,b,c;
  INIT_MAP;

  ESI = lookup_EventSet(map, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state & PAPI_RUNNING)
    {
      retval = _papi_hwd_read(ESI, master_event_set, ESI->sw_stop);
      if (retval != PAPI_OK)
        return(handle_error(retval, NULL));
    }
  
  for (i=0 ; i < ESI->NumberOfCounters; i++)
    {
      a = ESI->sw_stop[i];
      b = values[i];
      c = a + b;
      values[i] = c;
    } 

  return(PAPI_reset(EventSet));
}

int PAPI_write(int EventSet, long long *values)
{
  EventSetInfo *ESI;
  INIT_MAP;

  ESI = lookup_EventSet(map, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state == PAPI_RUNNING)
    {
      retval = _papi_hwd_write(ESI, values);
      if (retval!=PAPI_OK)
        return(handle_error(retval, NULL));
    }

  memcpy(ESI->hw_start,values,_papi_system_info.num_cntrs*sizeof(long long));

  return(retval);
}

/*  The function PAPI_cleanup removes a stopped 
    EventSet from existence. */

int PAPI_cleanup(int *EventSet) 
{ 
  EventSetInfo *ESI;
  INIT_MAP;

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(map, *EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  if (ESI->state != PAPI_STOPPED) 
    return(handle_error(PAPI_EINVAL,"EventSet is still running"));
  
  retval = cleanup_EventSet(map, ESI);
  if (retval < PAPI_OK)
    return(handle_error(PAPI_EMISC,NULL));

  *EventSet = PAPI_NULL;
  return(retval);
}
 
int PAPI_state(int EventSet, int *status)
{
  EventSetInfo *ESI;
  INIT_MAP;

  if (status == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  /* check for good EventSetIndex value*/
  
  ESI = lookup_EventSet(map, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));
  
  /*read status FROM ESI->state*/
  
  *status=ESI->state;
  
  return(PAPI_OK);
}


int PAPI_set_opt(int option, PAPI_option_t *ptr)
{ 
  _papi_int_option_t internal;
  INIT_MAP;

  switch(option)
    { 
    case PAPI_SET_DOMAIN:
      { 
	int dom = ptr->defdomain.domain;

	if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
	  return(handle_error(PAPI_EINVAL,"Domain out of range"));

        internal.domain.ESI = lookup_EventSet(map, ptr->domain.eventset);
        if (internal.domain.ESI == NULL)
          return(handle_error(PAPI_EINVAL,"No such EventSet"));

        if (!(internal.domain.ESI->state & PAPI_STOPPED))
          return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

        internal.domain.domain = dom;
        internal.domain.eventset = ptr->domain.eventset;
        retval = _papi_hwd_ctl(master_event_set, PAPI_SET_DOMAIN, &internal);
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

        internal.granularity.ESI = lookup_EventSet(map, ptr->granularity.eventset);
        if (internal.granularity.ESI == NULL)
          return(handle_error(PAPI_EINVAL,"No such EventSet"));

        if (!(internal.granularity.ESI->state & PAPI_STOPPED))
          return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

        internal.granularity.granularity = grn;
        internal.granularity.eventset = ptr->granularity.eventset;
        retval = _papi_hwd_ctl(master_event_set, PAPI_SET_GRANUL, &internal);
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
        retval = _papi_hwd_ctl(master_event_set, PAPI_SET_DEFDOM, &internal);

        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        master_event_set->domain.domain = dom;
        return(retval);
      }
    case PAPI_SET_DEFGRN:
      {
        int grn = ptr->granularity.granularity;

        if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
          return(handle_error(PAPI_EINVAL,"Granularity out of range"));

        internal.granularity.granularity = grn;
        retval = _papi_hwd_ctl(master_event_set, PAPI_SET_DEFGRN, &internal);

        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        master_event_set->granularity.granularity = grn;
        return(retval);
      }
    case PAPI_SET_INHERIT:
      {
        internal.inherit.inherit = ptr->inherit.inherit;
        retval = _papi_hwd_ctl(master_event_set, PAPI_SET_INHERIT, &internal);
        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        master_event_set->inherit.inherit = (ptr->inherit.inherit != 0);
        return(retval);
      }
    default:
      return(handle_error(PAPI_EINVAL,"Invalid option type"));
    }
}

int PAPI_get_opt(int option, PAPI_option_t *ptr) 
{ 
  INIT_MAP;

  retval = PAPI_init();
  if (retval < PAPI_OK)
    return retval;

  switch(option)
    {
    case PAPI_GET_CLOCKRATE:
      return(_papi_system_info.hw_info.mhz);
    case PAPI_GET_MAX_HWCTRS:
      return(_papi_system_info.num_cntrs);
    case PAPI_GET_DEFDOM:
      return(master_event_set->domain.domain);
    case PAPI_GET_DEFGRN:
      return(master_event_set->granularity.granularity);
    case PAPI_GET_INHERIT:
      return(master_event_set->inherit.inherit);
    case PAPI_GET_EXEINFO:
      if (ptr == NULL)
	return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));
      ptr->exe_info = &_papi_system_info.exe_info;
      break;
    case PAPI_GET_HWINFO:
      if (ptr == NULL)
	return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));
      ptr->hw_info = &_papi_system_info.hw_info;
      break;
    case PAPI_GET_DOMAIN:
      if (ptr == NULL)
	return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));
      return(get_domain(map, &ptr->domain));
    case PAPI_GET_GRANUL:
      if (ptr == NULL)
	return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));
      return(get_granularity(map, &ptr->granularity));
    default:
      return(handle_error(PAPI_EINVAL,"Invalid option type"));
    }
  return(PAPI_OK);
} 

void PAPI_shutdown(void) 
{
  int i;
  INIT_MAP_VOID;

  for (i=0;i<map->totalSlots;i++) 
    {
      if (map->dataSlotArray[i]) 
	{
	  free_EventSet(map->dataSlotArray[i]);
	  map->dataSlotArray[i] = NULL;
	}
    }
  free(map->dataSlotArray);
  memset(map,0x0,sizeof(DynamicArray));
  free(map);
  map = NULL;
}

static int handle_error(int PAPI_errorCode, char *errorMessage)
{
#if 0
  if (PAPI_ERR_LEVEL)
    {
      char tmp[80];
      char *s;

      s = get_error_string(PAPI_errorCode, tmp);
      if (PAPI_errorCode == PAPI_ESYS)
        perror(errorMessage);
      else if (errorMessage)
        fprintf(stderr, "%s : %s", s, errorMessage);
      else
	fprintf(stderr,"%s",s);

      fprintf(stderr,"\n");

      if (PAPI_ERR_LEVEL==PAPI_VERB_ESTOP)
        PAPI_shutdown();
    }
#endif
  char *s;

  s = get_error_string(PAPI_errorCode);
  fprintf(stderr,"PAPI Error Code %d: %s\n",PAPI_errorCode,s);
  if (PAPI_errorCode == PAPI_ESYS)
    fprintf(stderr,"System Error Code %d: %s\n",errno,strerror(errno));
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
  if ((errorCode > 0) || (-errorCode > PAPI_NUM_ERRORS))
    errorCode = PAPI_EMISC;
    
  return(papi_errStr[-errorCode]);
}

int PAPI_perror(int code, char *destination, int length)
{
  char *foo;
  char tmp[80];

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
  int index;
  EventSetInfo *ESI;
  EventSetOverflowInfo_t opt = { 0, };
  INIT_MAP;

  ESI = lookup_EventSet(map, EventSet);
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

  /* Copy the machine independent options into the ESI */

  memcpy(&ESI->overflow, &opt, sizeof(EventSetOverflowInfo_t));

  return(PAPI_OK);
}

static void dummy_handler(int EventSet, int EventCode, int EventIndex,
                          long long *values, int *threshold, void *context)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

int PAPI_profil(void *buf, int bufsiz, caddr_t offset, int scale,
                int EventSet, int EventCode, int threshold, int flags)
{
  EventSetInfo *ESI;
  EventSetProfileInfo_t opt = { 0, };
  INIT_MAP;

  ESI = lookup_EventSet(map, EventSet);
  if(ESI == NULL)
     return(handle_error(PAPI_EINVAL, "No such EventSet"));

  if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  if (lookup_EventCodeIndex(ESI, EventCode) < 0)
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

  if (flags & ~(PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM | PAPI_PROFIL_WEIGHTED))
    return(PAPI_EINVAL);

  /* Set up the option structure for the low level */

  opt.buf = buf;
  opt.bufsiz = bufsiz;
  opt.offset = offset;
  opt.scale = scale;
  opt.flags = flags;

  
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

int PAPI_add_events(int *EventSet, int *Events, int number)
{
  int i, retval;

  for (i=0;i<number;i++)
    {
      retval = PAPI_add_event(EventSet, Events[i]);
      if (retval!=PAPI_OK) return(retval);
    }
  return(PAPI_OK);
}

int PAPI_rem_events(int *EventSet, int *Events, int number)
{
  int i;
  EventSetInfo *ESI;
  INIT_MAP;

  if ((!EventSet) || (!Events))
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI=lookup_EventSet(map, *EventSet);
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
      if(retval!=PAPI_OK) return(retval);
    }
  return(PAPI_OK);
}


int PAPI_list_events(int EventSet, int *Events, int *number)
{
  EventSetInfo *ESI;
  int num;
  int i;
  INIT_MAP;

  if ((!Events) || (!number))
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(map, EventSet);
  if (!ESI)
    return(handle_error(PAPI_EINVAL, "Not a valid EventSet"));

#ifdef DEBUG
  /* Not necessary */
  if (ESI->NumberOfCounters == 0)
    return(handle_error(PAPI_EINVAL, "No events have been added"));
#endif

  if (*number < ESI->NumberOfCounters)
    num = *number;
  else
    num = ESI->NumberOfCounters;

  for(i=0; i<num; i++)
    Events[i] = ESI->EventInfoArray[i].code;

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

void *PAPI_get_overflow_address(void *context)
{
  return(_papi_hwd_get_overflow_address(context));
}

const PAPI_exe_info_t *PAPI_get_executable_info(void)
{
  PAPI_option_t ptr;
  int retval;

  retval = PAPI_get_opt(PAPI_GET_EXEINFO,&ptr);
  if (retval == PAPI_OK)
    return(ptr.exe_info);
  else
    return(NULL);
}

const PAPI_hw_info_t *PAPI_get_hardware_info(void)
{
  PAPI_option_t ptr;
  int retval;

  retval = PAPI_get_opt(PAPI_GET_HWINFO,&ptr);
  if (retval == PAPI_OK)
    return(ptr.hw_info);
  else
    return(NULL);
}

long long PAPI_get_real_cyc(void)
{
  INIT_MAP_QUICK_LL;

  return(_papi_hwd_get_real_cycles());
}

long long PAPI_get_real_usec(void)
{
  INIT_MAP_QUICK_LL;

  return(_papi_hwd_get_real_usec());
}

int PAPI_notify(int what, int flags, PAPI_notify_handler_t handler)
{
#ifdef PTHREADS
  switch (what)
    {
    case PAPI_THREAD_CREATE:
      if (pthread_mutex_lock(&global_mutex))
	return(PAPI_ESYS);
      thread_notifier = handler;
      if (pthread_mutex_unlock(&global_mutex))
	return(PAPI_ESYS);
      return(PAPI_OK);
    case PAPI_THREAD_DESTROY:
    default:
      return(PAPI_EINVAL);
    }
#else
  return(PAPI_EINVAL);
#endif
}

int PAPI_sample(int EventSet, int EventCode, int ms, int flags, PAPI_sample_handler_t handler)
{
  return(PAPI_OK);
}
