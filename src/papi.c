#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>

#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

/* Error handling functions */

static int handle_error(int, char *);
static char *get_error_string(int);

/* Utility functions */

static int expand_dynamic_array(DynamicArray *);

/* EventSet handling functions */

static EventSetInfo *allocate_EventSet(void);
static int add_EventSet(DynamicArray *map, EventSetInfo *created, EventSetInfo *master);
static EventSetInfo *lookup_EventSet(const DynamicArray *map, int eventset);
static int remove_EventSet(DynamicArray *map, EventSetInfo *);
static void free_EventSet(EventSetInfo *);

/* Event handling functions */

static int add_event(EventSetInfo *ESI, int EventCode);
static int add_pevent(EventSetInfo *ESI, int EventCode, void *inout);
static int get_free_EventCodeIndex(EventSetInfo *ESI, int EventCode);
static int lookup_EventCodeIndex(EventSetInfo *ESI,int EventCode);
static int remove_event(EventSetInfo *ESI, int EventCode);

/* Global variables that may be modified by any thread */

#define PAPI_EVENTSET_MAP (&_papi_system_info.global_eventset_map)

EventSetInfo *default_master_eventset = NULL; /* For non threaded apps */

static int PAPI_ERR_LEVEL = PAPI_VERB_ECONT; /* Behavior of handle_error() */

#ifdef DEBUG
int papi_debug = 1;
#endif

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
  { "PAPI_CA_SHR", PAPI_CA_SHR, "Requests for exclusive access to shared cache line", 0, NULL, 0 },
  { "PAPI_CA_CLN", PAPI_CA_CLN, "Requests for exclusive access to clean cache line", 0, NULL, 0 },
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
  { "PAPI_L3_DCH", PAPI_L3_DCH, "Level 3 Data Cache Hits", 0, NULL, 0 },
  { "PAPI_TLB_SD", PAPI_TLB_SD, "Translation lookaside buffer shootdowns", 0, NULL, 0 },
  { "PAPI_CSR_FAL", PAPI_CSR_FAL, "Failed store conditional instructions", 0, NULL, 0 },
  { "PAPI_CSR_SUC", PAPI_CSR_SUC, "Successful store conditional instructions", 0, NULL, 0 },
  { "PAPI_CSR_TOT", PAPI_CSR_TOT, "Total store conditional instructions", 0, NULL, 0 },
  { "PAPI_MEM_SCY", PAPI_MEM_SCY, "Cycles Stalled Waiting for memory accesses", 0, NULL, 0 },
  { "PAPI_MEM_RCY", PAPI_MEM_RCY, "Cycles Stalled Waiting for memory Reads", 0, NULL, 0 },
  { "PAPI_MEM_WCY", PAPI_MEM_WCY, "Cycles Stalled Waiting for memory writes", 0, NULL, 0 },
  { "PAPI_STL_ICY", PAPI_STL_ICY, "Cycles with no instruction issue", 0, NULL, 0 },
  { "PAPI_FUL_ICY", PAPI_FUL_ICY, "Cycles with maximum instruction issue", 0, NULL, 0 },
  { "PAPI_STL_CCY", PAPI_STL_CCY, "Cycles with no instructions completed", 0, NULL, 0 },
  { "PAPI_FUL_CCY", PAPI_FUL_CCY, "Cycles with maximum instructions completed", 0, NULL, 0 },
  { "PAPI_HW_INT", PAPI_HW_INT, "Hardware interrupts", 0, NULL, 0 },
  { "PAPI_BR_UCN", PAPI_BR_UCN, "Unconditional branch instructions", 0, NULL, 0 },
  { "PAPI_BR_CN", PAPI_BR_CN, "Conditional branch instructions", 0, NULL, 0 },
  { "PAPI_BR_TKN", PAPI_BR_TKN, "Conditional branch instructions taken", 0, NULL, 0 }, 
  { "PAPI_BR_NTK", PAPI_BR_NTK, "Conditional branch instructions not taken", 0, NULL, 0 }, 
  { "PAPI_BR_MSP", PAPI_BR_MSP, "Conditional branch instructions mispredicted", 0, NULL, 0 },
  { "PAPI_BR_PRC", PAPI_BR_PRC, "Conditional branch instructions correctly predicted", 0, NULL, 0 },
  { "PAPI_FMA_INS", PAPI_FMA_INS, "FMA instructions completed", 0, NULL, 0 },
  { "PAPI_TOT_IIS", PAPI_TOT_IIS, "Instructions issued", 0, NULL, 0 },
  { "PAPI_TOT_INS", PAPI_TOT_INS, "Instructions completed", 0, NULL, 0 },
  { "PAPI_INT_INS", PAPI_INT_INS, "Integer instructions", 0, NULL, 0 },
  { "PAPI_FP_INS", PAPI_FP_INS, "Floating point instructions", 0, NULL, 0 },
  { "PAPI_LD_INS", PAPI_LD_INS, "Load instructions", 0, NULL, 0 },
  { "PAPI_SR_INS", PAPI_SR_INS, "Store instructions", 0, NULL, 0 },
  { "PAPI_BR_INS", PAPI_BR_INS, "Branch instructions", 0, NULL, 0 },
  { "PAPI_VEC_INS", PAPI_VEC_INS, "Vector/SIMD instructions", 0, NULL, 0 },
  { "PAPI_FLOPS", PAPI_FLOPS, "Floating point instructions per second", 0, NULL, 0 },
  { "PAPI_RES_STL", PAPI_RES_STL, "Cycles stalled on any resource", 0, NULL, 0 },
  { "PAPI_FP_STAL", PAPI_FP_STAL, "Cycles the FP unit(s) are stalled", 0, NULL, 0 },
  { "PAPI_TOT_CYC", PAPI_TOT_CYC, "Total cycles", 0, NULL, 0 },
  { "PAPI_IPS", PAPI_IPS, "Instructions per second", 0, NULL, 0 },
  { "PAPI_LST_INS", PAPI_LST_INS, "Load/store instructions completed", 0, NULL, 0 },
  { "PAPI_SYC_INS", PAPI_SYC_INS, "Synchronization instructions completed", 0, NULL, 0 },
  { "PAPI_L1_DCH", PAPI_L1_DCH, "L1 data cache hits", 0, NULL, 0 },
  { "PAPI_L2_DCH", PAPI_L2_DCH, "L2 data cache hits", 0, NULL, 0 },
  { "PAPI_L1_DCA", PAPI_L1_DCA, "L1 data cache accesses", 0, NULL, 0 },
  { "PAPI_L2_DCA", PAPI_L2_DCA, "L2 data cache accesses", 0, NULL, 0 },
  { "PAPI_L3_DCA", PAPI_L3_DCA, "L3 data cache accesses", 0, NULL, 0 },
  { "PAPI_L1_DCR", PAPI_L1_DCR, "L1 data cache reads", 0, NULL, 0 },
  { "PAPI_L2_DCR", PAPI_L2_DCR, "L2 data cache reads", 0, NULL, 0 },
  { "PAPI_L3_DCR", PAPI_L3_DCR, "L3 data cache reads", 0, NULL, 0 },
  { "PAPI_L1_DCW", PAPI_L1_DCW, "L1 data cache writes", 0, NULL, 0 },
  { "PAPI_L2_DCW", PAPI_L2_DCW, "L2 data cache writes", 0, NULL, 0 },
  { "PAPI_L3_DCW", PAPI_L3_DCW, "L3 data cache writes", 0, NULL, 0 },
  { "PAPI_L1_ICH", PAPI_L1_ICH, "L1 instruction cache hits", 0, NULL, 0 },
  { "PAPI_L2_ICH", PAPI_L2_ICH, "L2 instruction cache hits", 0, NULL, 0 },
  { "PAPI_L3_ICH", PAPI_L3_ICH, "L3 instruction cache hits", 0, NULL, 0 },
  { "PAPI_L1_ICA", PAPI_L1_ICA, "L1 instruction cache accesses", 0, NULL, 0 },
  { "PAPI_L2_ICA", PAPI_L2_ICA, "L2 instruction cache accesses", 0, NULL, 0 },
  { "PAPI_L3_ICA", PAPI_L3_ICA, "L3 instruction cache accesses", 0, NULL, 0 },
  { "PAPI_L1_ICR", PAPI_L1_ICR, "L1 instruction cache reads", 0, NULL, 0 },
  { "PAPI_L2_ICR", PAPI_L2_ICR, "L2 instruction cache reads", 0, NULL, 0 },
  { "PAPI_L3_ICR", PAPI_L3_ICR, "L3 instruction cache reads", 0, NULL, 0 },
  { "PAPI_L1_ICW", PAPI_L1_ICW, "L1 instruction cache writes", 0, NULL, 0 },
  { "PAPI_L2_ICW", PAPI_L2_ICW, "L2 instruction cache writes", 0, NULL, 0 },
  { "PAPI_L3_ICW", PAPI_L3_ICW, "L3 instruction cache writes", 0, NULL, 0 },
  { "PAPI_L1_TCH", PAPI_L1_TCH, "L1 total cache hits", 0, NULL, 0 },
  { "PAPI_L2_TCH", PAPI_L2_TCH, "L2 total cache hits", 0, NULL, 0 },
  { "PAPI_L3_TCH", PAPI_L3_TCH, "L3 total cache hits", 0, NULL, 0 },
  { "PAPI_L1_TCA", PAPI_L1_TCA, "L1 total cache accesses", 0, NULL, 0 },
  { "PAPI_L2_TCA", PAPI_L2_TCA, "L2 total cache accesses", 0, NULL, 0 },
  { "PAPI_L3_TCA", PAPI_L3_TCA, "L3 total cache accesses", 0, NULL, 0 },
  { "PAPI_L1_TCR", PAPI_L1_TCR, "L1 total cache reads", 0, NULL, 0 },
  { "PAPI_L2_TCR", PAPI_L2_TCR, "L2 total cache reads", 0, NULL, 0 },
  { "PAPI_L3_TCR", PAPI_L3_TCR, "L3 total cache reads", 0, NULL, 0 },
  { "PAPI_L1_TCW", PAPI_L1_TCW, "L1 total cache writes", 0, NULL, 0 },
  { "PAPI_L2_TCW", PAPI_L2_TCW, "L2 total cache writes", 0, NULL, 0 },
  { "PAPI_L3_TCW", PAPI_L3_TCW, "L3 total cache writes", 0, NULL, 0 },
  { "PAPI_FML_INS", PAPI_FML_INS, "Floating point multiply instructions", 0, NULL, 0 },
  { "PAPI_FAD_INS", PAPI_FAD_INS, "Floating point add instructions", 0, NULL, 0 },
  { "PAPI_FDV_INS", PAPI_FDV_INS, "Floating point divide instructions", 0, NULL, 0 },
  { "PAPI_FSQ_INS", PAPI_FSQ_INS, "Floating point square root instructions", 0, NULL, 0 },
  { "PAPI_FNV_INS", PAPI_FNV_INS, "Floating point inverse instructions", 0, NULL, 0 },
};

/* Utility functions */

static int allocate_eventset_map(DynamicArray *map)
{
  /* Allocate and clear the Dynamic Array structure */
  
  memset(map,0x00,sizeof(DynamicArray));

  /* Allocate space for the EventSetInfo pointers */

  map->dataSlotArray = 
    (EventSetInfo **)malloc(PAPI_INIT_SLOTS*sizeof(EventSetInfo *));
  if(map->dataSlotArray == NULL) 
    {
      free(map);
      return(1);
    }
  memset(map->dataSlotArray,0x00, 
	 PAPI_INIT_SLOTS*sizeof(EventSetInfo *));

  map->totalSlots = PAPI_INIT_SLOTS;
  map->availSlots = PAPI_INIT_SLOTS;
  map->fullSlots  = 0;
  map->lowestEmptySlot = 0;
  
  return(0);
}

static void free_master_eventset(EventSetInfo *master)
{
  free_EventSet(master);
}

static EventSetInfo *allocate_master_eventset(void)
{
  EventSetInfo *master;
  
  /* The Master EventSet is special. It is not in the EventSet list, but is pointed
     to by each EventSet of that particular thread. */
  
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

int PAPI_thread_init(void **handle, int flag)
{
  int retval;
  EventSetInfo *master;

  if ((master = allocate_master_eventset()) == NULL)
    return(PAPI_ENOMEM);

  /* Call the substrate to fill in anything special. */
  
  retval = _papi_hwd_init(master);
  if (retval)
    {
      free_master_eventset(master);
      return(retval);
    }

  if (handle == NULL)
    default_master_eventset = master;
  else
    *handle = master;

  return(PAPI_OK);
}

#if defined(linux)
#define __SMP__
#include <asm/atomic.h>
atomic_t lock;
#elif defined(sun) && defined(sparc)
#include <synch.h>
rwlock_t lock;
#elif defined(sgi) && defined(mips)
int lock = 0;
#elif defined(_CRAYT3E)
volatile int lock = 0;
#elif defined(_AIX)
#include <sys/atomic_op.h>
int lock_var = 0;
atomic_p lock = &lock_var;
#endif

void PAPI_lock(void)
{
#if defined(linux)
  atomic_inc(&lock);
  while (atomic_read(&lock) > 1)
    {
      DBG((stderr,"Waiting..."));
      usleep(1000);
    }
#elif defined(_AIX)
  while (_check_lock(lock,0,1) == TRUE)
    {
      DBG((stderr,"Waiting..."));
      usleep(1000);
    }
#elif defined(sgi) && defined(mips)
  while (__lock_test_and_set(&lock,1) != 0)
    {
      DBG((stderr,"Waiting..."));
      usleep(1000);
    }
#elif defined(_CRAYT3E)
  _cmr();
  lock++;
  _cmr();
  while (lock != 1)
    {
      DBG((stderr,"Waiting..."));
      _cmr();
    }
#elif defined(sun) && defined(sparc)
  rw_wrlock(&lock);
#endif
}

void PAPI_unlock(void)
{
#if defined(linux)
  atomic_dec(&lock);
#elif defined(sun) && defined(sparc)
  rw_unlock(&lock);
#elif defined(_AIX)
  _clear_lock(lock, 0);
#elif defined(sgi) && defined(mips)
  __lock_release(&lock);
#elif defined(_CRAYT3E)
  _cmr();
  lock--;
  _cmr();
#endif
}

int PAPI_library_init(int version)
{
  int init_retval, i, tmp;

  if (version != PAPI_VER_CURRENT)
    return(PAPI_EMISC);

#ifdef DEBUG
  if (getenv("PAPI_NDEBUG"))
    papi_debug = 0;
#endif

#if defined(linux)
  atomic_set(&lock,0);
#endif

  tmp = _papi_hwd_init_global();
  if (tmp)
    return(tmp);

  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    if (papi_presets[i].event_name) /* If the preset is part of the API */
      papi_presets[i].avail = 
	_papi_hwd_query(papi_presets[i].event_code ^ PRESET_MASK,
			&papi_presets[i].flags,
			&papi_presets[i].event_note);

  if (allocate_eventset_map(PAPI_EVENTSET_MAP))
    return(PAPI_ENOMEM);

  init_retval = PAPI_VER_CURRENT;
  return(init_retval);
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

static int add_EventSet(DynamicArray *map, EventSetInfo *ESI, EventSetInfo *master)
{
  int i, errorCode;

  PAPI_lock();

  /* Update the values for lowestEmptySlot, num of availSlots */

  ESI->master = master;
  ESI->EventSetIndex = map->lowestEmptySlot;
  map->dataSlotArray[ESI->EventSetIndex] = ESI;
  map->availSlots--;
  map->fullSlots++; 

  if (map->availSlots == 0)
    {
      errorCode = expand_dynamic_array(map);
      if (errorCode!=PAPI_OK) 
	{
	  PAPI_unlock();
	  return(errorCode);
	}
    }

  i = ESI->EventSetIndex + 1;
  while (map->dataSlotArray[i]) i++;
  DBG((stderr,"Empty slot for lowest available EventSet is at %d\n",i));
  map->lowestEmptySlot = i;
 
  PAPI_unlock();
  return(PAPI_OK);
}

static int get_domain(DynamicArray *map, PAPI_domain_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, opt->eventset);
  if(ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  opt->domain = ESI->domain.domain;
  return(PAPI_OK);
}

static int get_granularity(DynamicArray *map, PAPI_granularity_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, opt->eventset);
  if(ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));

  opt->granularity = ESI->granularity.granularity;
  return(PAPI_OK);
}

int PAPI_query_event(int EventCode)
{ 
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
  int retval;

  /* Is the EventSet already in existence? */

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
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
      retval = add_EventSet(PAPI_EVENTSET_MAP, ESI, default_master_eventset);
      if (retval < PAPI_OK)
        goto heck;

      *EventSet = ESI->EventSetIndex;
      DBG((stderr,"PAPI_add_pevent new EventSet in slot %d\n",*EventSet));
    }
  return(retval);
}

int PAPI_create_eventset_r(int *EventSet, void *handle)
{
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset = (EventSetInfo *)handle;
  int retval;

  /* Is the EventSet already in existence? */
  
  if ((EventSet == NULL) || (handle == NULL))
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (*EventSet != PAPI_NULL)
    return(handle_error(PAPI_EINVAL, "EventSet must be initialized to PAPI_NULL"));

  /* Well, then allocate a new one. Use n to keep track of a NEW EventSet */
  
  ESI = allocate_EventSet();
  if (ESI == NULL)
    return(handle_error(PAPI_ENOMEM,"Error allocating memory for new EventSet"));

  /* Add it to the global table */

  retval = add_EventSet(PAPI_EVENTSET_MAP, ESI, thread_master_eventset);
  if (retval < PAPI_OK)
    {
      free_EventSet(ESI);
      return(handle_error(retval,"Error adding EventSet to global array"));
    }
  
  *EventSet = ESI->EventSetIndex;
  DBG((stderr,"PAPI_add_event new EventSet in slot %d\n",*EventSet));

  return(retval);
}

int PAPI_create_eventset(int *EventSet)
{
  return(PAPI_create_eventset_r(EventSet, default_master_eventset));
}

int PAPI_add_event(int *EventSet, int EventCode) 
{ 
  int retval;
  EventSetInfo *ESI, *n = NULL;

  /* Is the EventSet already in existence? */
  
  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
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
      retval = add_EventSet(PAPI_EVENTSET_MAP, ESI, default_master_eventset);
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

static EventSetInfo *lookup_EventSet(const DynamicArray *map, int eventset)
{
  if ((eventset < 0) || (eventset >= map->totalSlots))
    return(NULL);
  return(map->dataSlotArray[eventset]);
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
  if (i < map->lowestEmptySlot)
    map->lowestEmptySlot = i;
  map->availSlots++;
  map->fullSlots--;

  return(PAPI_OK);
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
  int retval;

  /* check for pre-existing ESI */

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
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

  /* check for pre-existing ESI */

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));

  if (!(ESI->state & PAPI_STOPPED))
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  if (ESI->NumberOfCounters)
    return(handle_error(PAPI_EINVAL, "EventSet is not empty"));

  remove_EventSet(PAPI_EVENTSET_MAP, ESI);
  *EventSet = PAPI_NULL;

  return(PAPI_OK);
}

/* simply checks for valid EventSet, calls substrate start() call */

int PAPI_start(int EventSet)
{ 
  int i, retval;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));
  thread_master_eventset = ESI->master;

  if (!(ESI->state & PAPI_STOPPED))
    return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

  if (ESI->NumberOfCounters < 1)
    return(handle_error(PAPI_EINVAL, "EventSet is empty"));

  /* Short circuit this stuff if there's nothing running */

  if (thread_master_eventset->multistart.num_runners == 0)
    {
      for (i=0;i<_papi_system_info.num_cntrs;i++)
	{
	  ESI->hw_start[i] = 0;
	  thread_master_eventset->hw_start[i] = 0;
	}
    }

  /* If overflowing is enabled, turn it on */
  
  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_start_overflow_timer(ESI, thread_master_eventset);
      if (retval < PAPI_OK)
	return(retval);
    }

  retval = _papi_hwd_merge(ESI, thread_master_eventset);
  if (retval != PAPI_OK)
    return(handle_error(retval, NULL));

  ESI->state ^= PAPI_STOPPED;
  ESI->state |= PAPI_RUNNING;
  thread_master_eventset->multistart.num_runners++;

  DBG((stderr,"PAPI_start returns %d\n",retval));
  return(retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */

int PAPI_stop(int EventSet, long long *values)
{ 
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;
  int retval;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI==NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));
  thread_master_eventset = ESI->master;

  if (!(ESI->state & PAPI_RUNNING))
    return(handle_error(PAPI_EINVAL, "EventSet is not running"));

  retval = _papi_hwd_read(ESI, thread_master_eventset, ESI->sw_stop);
  if (retval != PAPI_OK)
    return(handle_error(retval,NULL)); 

  /* If overflowing is enabled, turn it off */

  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_stop_overflow_timer(ESI, thread_master_eventset);
      if (retval < PAPI_OK)
	return(retval);
    }
  
  retval = _papi_hwd_unmerge(ESI, thread_master_eventset);
  if (retval != PAPI_OK)
    return(handle_error(retval, NULL));

  if (values)
    memcpy(values,ESI->sw_stop,ESI->NumberOfCounters*sizeof(long long)); 

  ESI->state ^= PAPI_RUNNING;
  ESI->state |= PAPI_STOPPED;
  thread_master_eventset->multistart.num_runners --;

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
  int retval = PAPI_OK;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if(ESI == NULL) 
    return(handle_error(PAPI_EINVAL, "No such EventSet"));
  thread_master_eventset = ESI->master;

  if (ESI->state & PAPI_RUNNING)
    {
      /* If we're not the only one running, then just
         read the current values into the ESI->start
         array. This holds the starting value for counters
         that are shared. */

      retval = _papi_hwd_reset(ESI, thread_master_eventset);
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
  EventSetInfo *thread_master_eventset;
  int retval = PAPI_OK;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));
  thread_master_eventset = ESI->master;

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state & PAPI_RUNNING)
    {
      retval = _papi_hwd_read(ESI, thread_master_eventset, values);
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
  EventSetInfo *thread_master_eventset;
  int i, retval;
  long long a,b,c;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));
  thread_master_eventset = ESI->master;

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state & PAPI_RUNNING)
    {
      retval = _papi_hwd_read(ESI, thread_master_eventset, ESI->sw_stop);
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
  int retval = PAPI_OK;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL, "No such EventSet"));
  thread_master_eventset = ESI->master;

  if (values == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  if (ESI->state == PAPI_RUNNING)
    {
      retval = _papi_hwd_write(thread_master_eventset, ESI, values);
      if (retval!=PAPI_OK)
        return(handle_error(retval, NULL));
    }

  memcpy(ESI->hw_start,values,_papi_system_info.num_cntrs*sizeof(long long));

  return(retval);
}

/*  The function PAPI_cleanup removes a stopped EventSet from existence. */

int PAPI_cleanup_eventset(int *EventSet) 
{ 
  int i;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  if (EventSet == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));
  thread_master_eventset = ESI->master;

  if (ESI->state != PAPI_STOPPED) 
    return(handle_error(PAPI_EINVAL,"EventSet is still running"));
  
  for(i=0;i<ESI->NumberOfCounters;i++) 
    {
      if (remove_event(ESI, ESI->EventInfoArray[i].code))
	return(handle_error(PAPI_EBUG,NULL));
    }
  
  return(PAPI_OK);
}
 
int PAPI_state(int EventSet, int *status)
{
  EventSetInfo *ESI;

  if (status == NULL)
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  /* check for good EventSetIndex value*/
  
  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    return(handle_error(PAPI_EINVAL,"No such EventSet"));
  
  /*read status FROM ESI->state*/
  
  *status=ESI->state;
  
  return(PAPI_OK);
}

int PAPI_set_debug(int level)
{
  switch(level)
    {
    case PAPI_QUIET:
    case PAPI_VERB_ESTOP:
    case PAPI_VERB_ECONT:
      PAPI_ERR_LEVEL = level;
      return(PAPI_OK);
    default:
      return(PAPI_EINVAL);
    }
}

int PAPI_set_opt(int option, PAPI_option_t *ptr)
{ 
  _papi_int_option_t internal;
  int retval;
  EventSetInfo *thread_master_eventset;

  switch(option)
    { 
    case PAPI_SET_DOMAIN:
      { 
	int dom = ptr->defdomain.domain;
	
	if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
	  return(handle_error(PAPI_EINVAL,"Domain out of range"));

        internal.domain.ESI = lookup_EventSet(PAPI_EVENTSET_MAP, ptr->domain.eventset);
        if (internal.domain.ESI == NULL)
          return(handle_error(PAPI_EINVAL,"No such EventSet"));
	thread_master_eventset = internal.domain.ESI->master;

        if (!(internal.domain.ESI->state & PAPI_STOPPED))
          return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

        internal.domain.domain = dom;
        internal.domain.eventset = ptr->domain.eventset;
        retval = _papi_hwd_ctl(thread_master_eventset, PAPI_SET_DOMAIN, &internal);
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

        internal.granularity.ESI = lookup_EventSet(PAPI_EVENTSET_MAP, ptr->granularity.eventset);
        if (internal.granularity.ESI == NULL)
          return(handle_error(PAPI_EINVAL,"No such EventSet"));
	thread_master_eventset = internal.granularity.ESI->master;

        if (!(internal.granularity.ESI->state & PAPI_STOPPED))
          return(handle_error(PAPI_EINVAL, "EventSet is not stopped"));

        internal.granularity.granularity = grn;
        internal.granularity.eventset = ptr->granularity.eventset;
        retval = _papi_hwd_ctl(thread_master_eventset, PAPI_SET_GRANUL, &internal);
        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        internal.granularity.ESI->granularity.granularity = grn;
        return(retval);
      }
    /* case PAPI_SET_DEFDOM:
      {
        int dom = ptr->domain.domain;

        if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
          return(handle_error(PAPI_EINVAL,"Domain out of range"));

        internal.domain.domain = dom;
        retval = _papi_hwd_ctl(thread_master_eventset, PAPI_SET_DEFDOM, &internal);

        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        thread_master_eventset->domain.domain = dom;
        return(retval);
      }
    case PAPI_SET_DEFGRN:
      {
        int grn = ptr->granularity.granularity;

        if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
          return(handle_error(PAPI_EINVAL,"Granularity out of range"));

        internal.granularity.granularity = grn;
        retval = _papi_hwd_ctl(thread_master_eventset, PAPI_SET_DEFGRN, &internal);

        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        thread_master_eventset->granularity.granularity = grn;
        return(retval);
      }
    case PAPI_SET_INHERIT:
      {
        internal.inherit.inherit = ptr->inherit.inherit;
        retval = _papi_hwd_ctl(thread_master_eventset, PAPI_SET_INHERIT, &internal);
        if (retval < PAPI_OK)
          return(handle_error(retval,NULL));

        thread_master_eventset->inherit.inherit = (ptr->inherit.inherit != 0);
        return(retval);
      } */
    default:
      return(handle_error(PAPI_EINVAL,"Invalid option type"));
    }
}

int PAPI_get_opt(int option, PAPI_option_t *ptr) 
{ 
  switch(option)
    {
    case PAPI_GET_CLOCKRATE:
      return(_papi_system_info.hw_info.mhz);
    case PAPI_GET_MAX_HWCTRS:
      return(_papi_system_info.num_cntrs);
    case PAPI_GET_DEFDOM:
      return(_papi_system_info.default_domain);
    case PAPI_GET_DEFGRN:
      return(_papi_system_info.default_granularity);
#if 0
    case PAPI_GET_INHERIT:
      return(thread_master_eventset->inherit.inherit); 
#endif
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
      return(get_domain(PAPI_EVENTSET_MAP, &ptr->domain));
    case PAPI_GET_GRANUL:
      if (ptr == NULL)
	return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));
      return(get_granularity(PAPI_EVENTSET_MAP, &ptr->granularity));
    default:
      return(handle_error(PAPI_EINVAL,"Invalid option type"));
    }
  return(PAPI_OK);
} 

void PAPI_shutdown(void) 
{
  int i, status;

  for (i=0;i<PAPI_EVENTSET_MAP->totalSlots;i++) 
    {
      if (PAPI_EVENTSET_MAP->dataSlotArray[i]) 
	{
	  PAPI_state(i,&status);
	  if (status & PAPI_RUNNING)
	    PAPI_stop(i,NULL);
	  free_EventSet(PAPI_EVENTSET_MAP->dataSlotArray[i]);
	  PAPI_EVENTSET_MAP->dataSlotArray[i] = NULL;
	}
    }
  free(PAPI_EVENTSET_MAP->dataSlotArray);
#ifdef DEBUG
  memset(PAPI_EVENTSET_MAP,0x0,sizeof(DynamicArray));
#endif
#if 0
  free(PAPI_EVENTSET_MAP);
  PAPI_EVENTSET_MAP = NULL;
#endif
}

static int handle_error(int PAPI_errorCode, char *errorMessage)
{
  if (PAPI_ERR_LEVEL)
    {
      char *s;
      s = get_error_string(PAPI_errorCode);
      fprintf(stderr,"PAPI Error Code %d: %s\n",PAPI_errorCode,s);
      if (PAPI_errorCode == PAPI_ESYS)
	fprintf(stderr,"System Error Code %d: %s\n",errno,strerror(errno));
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
  if ((errorCode > 0) || (-errorCode > PAPI_NUM_ERRORS))
    errorCode = PAPI_EMISC;
    
  return(papi_errStr[-errorCode]);
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
  EventSetInfo *thread_master_eventset;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if(ESI == NULL)
     return(handle_error(PAPI_EINVAL, "No such EventSet"));
  thread_master_eventset = ESI->master;

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

  if (_papi_system_info.supports_hw_overflow)
    {
      retval = _papi_hwd_set_overflow(ESI, &opt);
      if (retval < PAPI_OK)
	return(retval);
    }
  else
    opt.timer_ms = PAPI_ITIMER_MS;
    
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

int PAPI_profil(void *buf, int bufsiz, caddr_t offset, unsigned int scale,
                int EventSet, int EventCode, int threshold, int flags)
{
  EventSetInfo *ESI;
  EventSetProfileInfo_t opt = { 0, };
  EventSetInfo *thread_master_eventset;
  int retval;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if(ESI == NULL)
     return(handle_error(PAPI_EINVAL, "No such EventSet"));
  thread_master_eventset = ESI->master;

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

  if (flags & ~(PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM | PAPI_PROFIL_WEIGHTED | PAPI_PROFIL_COMPRESS))
    return(PAPI_EINVAL);

  /* Set up the option structure for the low level */

  opt.buf = buf;
  opt.bufsiz = bufsiz;
  opt.offset = offset;
  opt.scale = scale;
  opt.flags = flags;
  
  if (_papi_system_info.supports_hw_profile)
    retval = _papi_hwd_set_profile(ESI, &opt);
  else 
    retval = PAPI_overflow(EventSet, EventCode, threshold, 0, dummy_handler);
  
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
{ 
  PAPI_option_t ptr;

  ptr.defgranularity.granularity = granularity;
  return(PAPI_set_opt(PAPI_SET_GRANUL, &ptr));
}

int PAPI_set_domain(int domain)
{ 
  PAPI_option_t ptr;

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
  int i, retval;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  if ((!EventSet) || (!Events))
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI=lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (!ESI)
    return(handle_error(PAPI_EINVAL, "Not a valid EventSet"));
  thread_master_eventset = ESI->master;

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

  if ((!Events) || (!number))
    return(handle_error(PAPI_EINVAL, "Null pointer is an invalid argument"));

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
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
  return(_papi_hwd_get_real_cycles());
}

long long PAPI_get_real_usec(void)
{
  return(_papi_hwd_get_real_usec());
}

long long PAPI_get_virt_cyc(void)
{
  return(_papi_hwd_get_virt_cycles());
}

long long PAPI_get_virt_usec(void)
{
  return(_papi_hwd_get_virt_usec());
}

int PAPI_notify(int what, int flags, PAPI_notify_handler_t handler)
{
  switch (what)
    {
    case PAPI_THREAD_CREATE:
    case PAPI_THREAD_DESTROY:
    default:
      return(PAPI_EINVAL);
    }
}

int PAPI_sample(int EventSet, int EventCode, int ms, int flags, PAPI_sample_handler_t handler)
{
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

