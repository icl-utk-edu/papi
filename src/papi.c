/* 
* File:    papi.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

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

/********************/
/* BEGIN PROTOTYPES */
/********************/

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
static int default_error_handler(int errorCode);

/********************/
/*  END PROTOTYPES  */
/********************/

/********************/
/*  BEGIN GLOBALS   */ 
/********************/

#define PAPI_EVENTSET_MAP (&_papi_system_info.global_eventset_map)
EventSetInfo *default_master_eventset = NULL; 
unsigned long int (*thread_id_fn)(void) = NULL;
#ifdef DEBUG
int papi_debug = 0;
#endif

/********************/
/*    END GLOBALS   */
/********************/

/********************/
/*  BEGIN LOCALS    */ 
/********************/

static int PAPI_ERR_LEVEL = PAPI_QUIET; /* Behavior of handle_error() */
static PAPI_debug_handler_t PAPI_ERR_HANDLER = default_error_handler;

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

const char *papi_errNam[PAPI_NUM_ERRORS] = {
  "PAPI_OK",
  "PAPI_EINVAL",
  "PAPI_ENOMEM",
  "PAPI_ESYS",
  "PAPI_ESBSTR",
  "PAPI_ECLOST",
  "PAPI_EBUG",
  "PAPI_ENOEVNT",
  "PAPI_ECNFLCT",
  "PAPI_ENOTRUN",
  "PAPI_EISRUN",
  "PAPI_ENOEVST",
  "PAPI_ENOTPRESET",
  "PAPI_ENOCNTR",
  "PAPI_EMISC" 
};

const char *papi_errStr[PAPI_NUM_ERRORS] = {
  "No error",
  "Invalid argument",
  "Insufficient memory",
  "A System/C library call failed",
  "Not supported by substrate",
  "Access to the counters was lost or interrupted",
  "Internal error, please send mail to the developers",
  "Event does not exist",
  "Event exists, but cannot be counted due to hardware resource limits",
  "EventSet is currently running",
  "EventSet is currently counting",
  "No such EventSet available",
  "Event in argument is not a valid preset",
  "Hardware does not support performance counters",
  "Unknown error code"
};

/********************/
/*    END LOCALS    */ 
/********************/

/* Utility functions */

static int default_error_handler(int errorCode)
{
  if (errorCode == PAPI_OK)
    return(errorCode);

  if ((errorCode > 0) || (-errorCode > PAPI_NUM_ERRORS))
    abort();

  switch (PAPI_ERR_LEVEL)
    {
    case PAPI_VERB_ECONT:
      fprintf(stderr,"PAPI Error Code %d: %s: %s\n",errorCode,papi_errNam[-errorCode],papi_errStr[-errorCode]);
      return errorCode;
      break;
    case PAPI_VERB_ESTOP:
      fprintf(stderr,"PAPI Error Code %d: %s: %s\n",errorCode,papi_errNam[-errorCode],papi_errStr[-errorCode]);
      exit(-errorCode);
      break;
    case PAPI_QUIET:
      return errorCode;
    default:
      abort();
    }
}

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

int PAPI_thread_init(unsigned long int (*id_fn)(void), int flag)
{
  if ((id_fn == NULL) || (flag != 0) || (default_master_eventset == NULL))
    papi_return(PAPI_EINVAL);
    
  thread_id_fn = id_fn;
  
  /* Now change the master event's thread id from 0 to the
     real thread id */

  /* By default, the initial master eventset has TID of -1. This will
     get changed if the user enables threads with PAPI_thread_init(). */

  default_master_eventset->tid = (*thread_id_fn)();

  _papi_hwi_insert_in_master_list(default_master_eventset);
  
  papi_return(PAPI_OK);
}

unsigned long int PAPI_thread_id(void)
{
  if (thread_id_fn != NULL)
    return((*thread_id_fn)());
  else
    return(-1);
}

static int initialize_master_eventset(EventSetInfo **master)
{
  int retval;

  if ((*master = allocate_master_eventset()) == NULL)
    papi_return(PAPI_ENOMEM);

  /* Call the substrate to fill in anything special. */
  
  retval = _papi_hwd_init(*master);
  if (retval)
    {
      free_master_eventset(*master);
      *master = NULL;
      return(retval);
    }

  if (thread_id_fn)
    (*master)->tid = (*thread_id_fn)();

  papi_return(PAPI_OK);
}

void PAPI_lock(void)
{
  _papi_hwd_lock();
}

void PAPI_unlock(void)
{
  _papi_hwd_unlock();
}

int PAPI_library_init(int version)
{
  static int init_retval = 0xdedbeef, i, tmp;

#ifdef DEBUG
  if (getenv("PAPI_DEBUG"))
    papi_debug = 1;
#endif

  if (init_retval != 0xdedbeef)
    return(init_retval);

  if (version != PAPI_VER_CURRENT) {
    init_retval = PAPI_EINVAL;
    papi_return(PAPI_EINVAL);
  }

  _papi_hwd_lock_init();

  tmp = _papi_hwd_init_global();
  if (tmp) {
    init_retval = tmp;
    return(init_retval);
  }

  if (allocate_eventset_map(PAPI_EVENTSET_MAP)) 
    {
      _papi_hwd_shutdown_global();
      init_retval = PAPI_ENOMEM;
      return(init_retval);
    }

  tmp = initialize_master_eventset(&default_master_eventset);
  if (tmp)
    {
      _papi_hwd_shutdown_global();
      init_retval = tmp;
      return(init_retval); 
    }

  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    if (papi_presets[i].event_name) /* If the preset is part of the API */
      papi_presets[i].avail = 
	_papi_hwd_query(papi_presets[i].event_code ^ PRESET_MASK,
			&papi_presets[i].flags,
			&papi_presets[i].event_note);

  return(init_retval = PAPI_VER_CURRENT);
}

static int expand_dynamic_array(DynamicArray *DA)
{
  int number;	
  EventSetInfo **n;

  /*realloc existing PAPI_EVENTSET_MAP.dataSlotArray*/
    
  number = DA->totalSlots*2;
  n = (EventSetInfo **)realloc(DA->dataSlotArray,number*sizeof(EventSetInfo *));
  if (n==NULL)
    papi_return(PAPI_ENOMEM);

  /* Need to assign this value, what if realloc moved it? */

  DA->dataSlotArray = n;

  memset(DA->dataSlotArray+DA->totalSlots,0x00,DA->totalSlots*sizeof(EventSetInfo *));

  DA->totalSlots = number;
  DA->availSlots = number - DA->fullSlots;
  DA->lowestEmptySlot = DA->totalSlots/2;

  papi_return(PAPI_OK);
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

  _papi_hwd_lock();

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
	  _papi_hwd_unlock();
	  return(errorCode);
	}
    }

  i = ESI->EventSetIndex + 1;
  while (map->dataSlotArray[i]) i++;
  DBG((stderr,"Empty slot for lowest available EventSet is at %d\n",i));
  map->lowestEmptySlot = i;
 
  _papi_hwd_unlock();
  papi_return(PAPI_OK);
}

static int get_domain(DynamicArray *map, PAPI_domain_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, opt->eventset);
  if(ESI == NULL)
    papi_return(PAPI_ENOEVST);

  opt->domain = ESI->domain.domain;
  papi_return(PAPI_OK);
}

static int get_granularity(DynamicArray *map, PAPI_granularity_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, opt->eventset);
  if(ESI == NULL)
    papi_return(PAPI_ENOEVST);

  opt->granularity = ESI->granularity.granularity;
  papi_return(PAPI_OK);
}

int PAPI_query_event(int EventCode)
{ 
  if (EventCode & PRESET_MASK)
    { 
      EventCode ^= PRESET_MASK;
      if (EventCode >= PAPI_MAX_PRESET_EVENTS)
	papi_return(PAPI_ENOTPRESET);
	
      if (papi_presets[EventCode].avail)
	papi_return(PAPI_OK);
      else
	papi_return(PAPI_ENOEVNT);
    }
  papi_return(PAPI_ENOTPRESET);
}

int PAPI_query_event_verbose(int EventCode, PAPI_preset_info_t *info)
{ 
  if (info == NULL)
    papi_return(PAPI_EINVAL);

  if (EventCode & PRESET_MASK)
    { 
      EventCode ^= PRESET_MASK;
      if (EventCode >= PAPI_MAX_PRESET_EVENTS)
	papi_return(PAPI_ENOTPRESET);
	
      if (papi_presets[EventCode].avail)
	{
	  memcpy(info,&papi_presets[EventCode],sizeof(PAPI_preset_info_t));
	  papi_return(PAPI_OK);
	}
      else
	papi_return(PAPI_ENOEVNT);
    }
  papi_return(PAPI_ENOTPRESET);
}

const PAPI_preset_info_t *PAPI_query_all_events_verbose(void)
{
  return(papi_presets);
}

int PAPI_event_code_to_name(int EventCode, char *out)
{
  if (out == NULL)
    papi_return(PAPI_EINVAL);

  if (EventCode & PRESET_MASK)
    { 
      EventCode ^= PRESET_MASK;
      if ((EventCode >= PAPI_MAX_PRESET_EVENTS) || (papi_presets[EventCode].event_name == NULL))
	papi_return(PAPI_ENOTPRESET);
	
      strcpy(out,papi_presets[EventCode].event_name,PAPI_MAX_STR_LEN);
      papi_return(PAPI_OK);
    }
  papi_return(PAPI_ENOTPRESET);
}

int PAPI_event_name_to_code(char *in, int *out)
{
  int i;
  
  if ((in == NULL) || (out == NULL))
    papi_return(PAPI_EINVAL);

  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    {
      if ((papi_presets[i].event_name) && (strcasecmp(papi_presets[i].event_name,in) == 0))
	{ 
	  *out = papi_presets[i].event_code;
	  papi_return(PAPI_OK);
	}
    }
  papi_return(PAPI_ENOTPRESET);
}

int create_eventset(int *EventSet, void *handle)
{
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset = (EventSetInfo *)handle;
  int retval;

  /* Is the EventSet already in existence? */
  
  if ((EventSet == NULL) || (handle == NULL))
    return(PAPI_EINVAL);

  /* Well, then allocate a new one. Use n to keep track of a NEW EventSet */
  
  ESI = allocate_EventSet();
  if (ESI == NULL)
    return(PAPI_ENOMEM);

  /* Add it to the global table */

  retval = add_EventSet(PAPI_EVENTSET_MAP, ESI, thread_master_eventset);
  if (retval < PAPI_OK)
    {
      free_EventSet(ESI);
      return(retval);
    }
  
  *EventSet = ESI->EventSetIndex;
  DBG((stderr,"create_eventset(%p,%p): new EventSet in slot %d\n",EventSet,handle,*EventSet));

  return(retval);
}

int PAPI_create_eventset(int *EventSet)
{
  EventSetInfo *master = _papi_hwi_lookup_in_master_list();
  int retval;
  if (master == NULL)
    {
      DBG((stderr,"PAPI_create_eventset(%p): new thread found\n",EventSet));
      retval = initialize_master_eventset(&master);
      if (retval)
	return(retval);
      _papi_hwi_insert_in_master_list(master);
    }

  return(create_eventset(EventSet, master));
}

int PAPI_add_pevent(int *EventSet, int code, void *inout)
{ 
  EventSetInfo *ESI;

  /* Is the EventSet already in existence? */

  if (EventSet == NULL)
    papi_return(PAPI_EINVAL);

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);

  /* Of course, it must be stopped in order to modify it. */

  if (!(ESI->state & PAPI_STOPPED))
    papi_return(PAPI_EISRUN);

  /* Now do the magic. */

  return(add_pevent(ESI,code,inout));
}

int PAPI_add_event(int *EventSet, int EventCode) 
{ 
  EventSetInfo *ESI;

  /* Is the EventSet already in existence? */
  
  if (EventSet == NULL)
    papi_return(PAPI_EINVAL);

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);

  /* Of course, it must be stopped in order to modify it. */

  if (!(ESI->state & PAPI_STOPPED))
    papi_return(PAPI_EISRUN);

  /* Now do the magic. */

  return(add_event(ESI,EventCode));
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
	papi_return(PAPI_ECNFLCT);
      if ((ESI->EventInfoArray[k].code == PAPI_NULL) && (lowslot == -1))
	lowslot = k;
    }
  
  if (lowslot != -1)
    return(lowslot);
  else
    papi_return(PAPI_ECNFLCT);
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

  papi_return(PAPI_EINVAL);
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

  papi_return(PAPI_OK);
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
    papi_return(PAPI_EINVAL);

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);

  if (!(ESI->state & PAPI_STOPPED))
    papi_return(PAPI_EISRUN);

  retval = remove_event(ESI,EventCode);
  if (retval < PAPI_OK)
    return(retval);

  return(retval);
}

int PAPI_destroy_eventset(int *EventSet)
{
  EventSetInfo *ESI;

  /* check for pre-existing ESI */

  if (EventSet == NULL)
    papi_return(PAPI_EINVAL);

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);

  if (!(ESI->state & PAPI_STOPPED))
    papi_return(PAPI_EISRUN);

  if (ESI->NumberOfCounters)
    papi_return(PAPI_EINVAL);

  remove_EventSet(PAPI_EVENTSET_MAP, ESI);
  *EventSet = PAPI_NULL;

  papi_return(PAPI_OK);
}

/* simply checks for valid EventSet, calls substrate start() call */

int PAPI_start(int EventSet)
{ 
  int i, retval;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (!(ESI->state & PAPI_STOPPED))
    papi_return(PAPI_EISRUN);

  if (ESI->NumberOfCounters < 1)
    papi_return(PAPI_EINVAL);

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
    return(retval);

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
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (!(ESI->state & PAPI_RUNNING))
    papi_return(PAPI_EISRUN);

  retval = _papi_hwd_read(ESI, thread_master_eventset, ESI->sw_stop);
  if (retval != PAPI_OK)
    return(retval);

  /* If overflowing is enabled, turn it off */

  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_stop_overflow_timer(ESI, thread_master_eventset);
      if (retval < PAPI_OK)
	return(retval);
    }
  
  retval = _papi_hwd_unmerge(ESI, thread_master_eventset);
  if (retval != PAPI_OK)
    return(retval);

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
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (ESI->state & PAPI_RUNNING)
    {
      /* If we're not the only one running, then just
         read the current values into the ESI->start
         array. This holds the starting value for counters
         that are shared. */

      retval = _papi_hwd_reset(ESI, thread_master_eventset);
      if (retval != PAPI_OK)
	return(retval);
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
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (values == NULL)
    papi_return(PAPI_EINVAL);

  if (ESI->state & PAPI_RUNNING)
    {
      retval = _papi_hwd_read(ESI, thread_master_eventset, values);
      if (retval != PAPI_OK)
        return(retval);
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
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (values == NULL)
    papi_return(PAPI_EINVAL);

  if (ESI->state & PAPI_RUNNING)
    {
      retval = _papi_hwd_read(ESI, thread_master_eventset, ESI->sw_stop);
      if (retval != PAPI_OK)
        return(retval);
    }
  
  for (i=0 ; i < ESI->NumberOfCounters; i++)
    {
      a = ESI->sw_stop[i];
      b = values[i];
      c = a + b;
      values[i] = c;
    } 

  papi_return(PAPI_reset(EventSet));
}

int PAPI_write(int EventSet, long long *values)
{
  int retval = PAPI_OK;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (values == NULL)
    papi_return(PAPI_EINVAL);

  if (ESI->state == PAPI_RUNNING)
    {
      retval = _papi_hwd_write(thread_master_eventset, ESI, values);
      if (retval!=PAPI_OK)
        return(retval);
    }

  memcpy(ESI->hw_start,values,_papi_system_info.num_cntrs*sizeof(long long));

  return(retval);
}

/*  The function PAPI_cleanup removes a stopped EventSet from existence. */

int PAPI_cleanup_eventset(int *EventSet) 
{ 
  int i, tmp;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  if (EventSet == NULL)
    papi_return(PAPI_EINVAL);

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (ESI->state != PAPI_STOPPED) 
    papi_return(PAPI_EISRUN);
  
  tmp = ESI->NumberOfCounters;
  for(i=0;i<tmp;i++) 
    {
      if (remove_event(ESI, ESI->EventInfoArray[i].code))
	papi_return(PAPI_EBUG);
    }
  
  papi_return(PAPI_OK);
}
 
int PAPI_state(int EventSet, int *status)
{
  EventSetInfo *ESI;

  if (status == NULL)
    papi_return(PAPI_EINVAL);

  /* check for good EventSetIndex value*/
  
  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);
  
  /*read status FROM ESI->state*/
  
  *status=ESI->state;
  
  papi_return(PAPI_OK);
}

int PAPI_set_debug(int level)
{
  switch(level)
    {
    case PAPI_QUIET:
    case PAPI_VERB_ESTOP:
    case PAPI_VERB_ECONT:
      PAPI_ERR_LEVEL = level;
      papi_return(PAPI_OK);
    default:
      papi_return(PAPI_EINVAL);
    }
}

int PAPI_set_opt(int option, PAPI_option_t *ptr)
{ 
  _papi_int_option_t internal;
  int retval;
  EventSetInfo *thread_master_eventset;

  if (ptr == NULL)
    papi_return(PAPI_EINVAL);

  switch(option)
    { 
    case PAPI_SET_DEBUG:
      papi_return(PAPI_set_debug(ptr->debug.level));
    case PAPI_SET_DOMAIN:
      { 
	int dom = ptr->defdomain.domain;
	
	if ((dom < PAPI_DOM_MIN) || (dom > PAPI_DOM_MAX))
	  papi_return(PAPI_EINVAL);

        internal.domain.ESI = lookup_EventSet(PAPI_EVENTSET_MAP, ptr->domain.eventset);
        if (internal.domain.ESI == NULL)
          papi_return(PAPI_ENOEVST);
	thread_master_eventset = internal.domain.ESI->master;

        if (!(internal.domain.ESI->state & PAPI_STOPPED))
          papi_return(PAPI_EISRUN);

        internal.domain.domain = dom;
        internal.domain.eventset = ptr->domain.eventset;
        retval = _papi_hwd_ctl(thread_master_eventset, PAPI_SET_DOMAIN, &internal);
        if (retval < PAPI_OK)
          return(retval);

        internal.domain.ESI->domain.domain = dom;
        return(retval);
      }
    case PAPI_SET_GRANUL:
      {
        int grn = ptr->granularity.granularity;

        if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
          papi_return(PAPI_EINVAL);

        internal.granularity.ESI = lookup_EventSet(PAPI_EVENTSET_MAP, ptr->granularity.eventset);
        if (internal.granularity.ESI == NULL)
          papi_return(PAPI_ENOEVST);
	thread_master_eventset = internal.granularity.ESI->master;

        if (!(internal.granularity.ESI->state & PAPI_STOPPED))
          papi_return(PAPI_EISRUN);

        internal.granularity.granularity = grn;
        internal.granularity.eventset = ptr->granularity.eventset;
        retval = _papi_hwd_ctl(thread_master_eventset, PAPI_SET_GRANUL, &internal);
        if (retval < PAPI_OK)
          return(retval);

        internal.granularity.ESI->granularity.granularity = grn;
        return(retval);
      }
    case PAPI_SET_INHERIT:
      {
	EventSetInfo *tmp = _papi_hwi_lookup_in_master_list();
	if (tmp == NULL)
	  return(PAPI_EINVAL);

        internal.inherit.inherit = ptr->inherit.inherit;
	internal.inherit.master = tmp;

        retval = _papi_hwd_ctl(tmp, PAPI_SET_INHERIT, &internal);
        if (retval < PAPI_OK)
          return(retval);

	tmp->inherit.inherit = ptr->inherit.inherit;
        return(retval);
      } 
    default:
      papi_return(PAPI_EINVAL);
    }
}

int PAPI_get_opt(int option, PAPI_option_t *ptr) 
{ 
  switch(option)
    {
    case PAPI_GET_PRELOAD:
      strncpy(ptr->preload.lib_preload_env,_papi_system_info.exe_info.lib_preload_env,
	      PAPI_MAX_STR_LEN);
      break;
    case PAPI_GET_DEBUG:
      ptr->debug.level = PAPI_ERR_LEVEL;
      ptr->debug.handler = PAPI_ERR_HANDLER;
      break;
    case PAPI_GET_CLOCKRATE:
      return(_papi_system_info.hw_info.mhz);
    case PAPI_GET_MAX_HWCTRS:
      return(_papi_system_info.num_cntrs);
    case PAPI_GET_DEFDOM:
      return(_papi_system_info.default_domain);
    case PAPI_GET_DEFGRN:
      return(_papi_system_info.default_granularity);
    case PAPI_GET_INHERIT:
      {
	EventSetInfo *tmp;
	tmp = _papi_hwi_lookup_in_master_list();
	if (tmp == NULL)
	  return(PAPI_EINVAL);
	
	return(tmp->inherit.inherit); 
      }
    case PAPI_GET_EXEINFO:
      if (ptr == NULL)
	papi_return(PAPI_EINVAL);
      ptr->exe_info = &_papi_system_info.exe_info;
      break;
    case PAPI_GET_HWINFO:
      if (ptr == NULL)
	papi_return(PAPI_EINVAL);
      ptr->hw_info = &_papi_system_info.hw_info;
      break;
    case PAPI_GET_DOMAIN:
      if (ptr == NULL)
	papi_return(PAPI_EINVAL);
      return(get_domain(PAPI_EVENTSET_MAP, &ptr->domain));
    case PAPI_GET_GRANUL:
      if (ptr == NULL)
	papi_return(PAPI_EINVAL);
      return(get_granularity(PAPI_EVENTSET_MAP, &ptr->granularity));
    default:
      papi_return(PAPI_EINVAL);
    }
  papi_return(PAPI_OK);
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

char *PAPI_strerror(int errorCode)
{
  if ((errorCode > 0) || (-errorCode > PAPI_NUM_ERRORS))
    return(NULL);
    
  return((char *)papi_errStr[-errorCode]);
}

int PAPI_perror(int code, char *destination, int length)
{
  char *foo;

  foo = PAPI_strerror(code);

  if (destination && (length >= 0))
    strncpy(destination,foo,length);
  else
    fprintf(stderr,"%s\n",foo);

  papi_return(PAPI_OK);
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
     papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
    papi_return(PAPI_EISRUN);

  if ((index = lookup_EventCodeIndex(ESI, EventCode)) < 0)
    papi_return(PAPI_ENOEVNT);

  if (threshold < 0)
    papi_return(PAPI_EINVAL);

  if (ESI->state & PAPI_OVERFLOWING)
    {
      if (threshold)
        papi_return(PAPI_EINVAL);
    }
  else
    {
      if (handler == NULL)
        papi_return(PAPI_EINVAL);
      if (threshold == 0)
        papi_return(PAPI_EINVAL);
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

  papi_return(PAPI_OK);
}

static void dummy_handler(int EventSet, int EventCode, int EventIndex,
                          long long *values, int *threshold, void *context)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

int PAPI_sprofil(PAPI_sprofil_t *prof, int profcnt, int EventSet, int EventCode, int threshold, int flags)
{
  EventSetInfo *ESI;
  EventSetProfileInfo_t opt = { 0, };
  int retval;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
     papi_return(PAPI_ENOEVST);

  if ((ESI->state & PAPI_STOPPED) != PAPI_STOPPED)
    papi_return(PAPI_EISRUN);

  if (lookup_EventCodeIndex(ESI, EventCode) < 0)
    papi_return(PAPI_ENOEVNT);

  if (threshold < 0)
    papi_return(PAPI_EINVAL);

  if (ESI->state & PAPI_PROFILING)
    {
      if (threshold)
        papi_return(PAPI_EINVAL);
    }
  else
    {
      if (threshold == 0)
        papi_return(PAPI_EINVAL);
    }

  if (flags & ~(PAPI_PROFIL_POSIX | PAPI_PROFIL_RANDOM | PAPI_PROFIL_WEIGHTED | PAPI_PROFIL_COMPRESS))
    papi_return(PAPI_EINVAL);

  /* Set up the option structure for the low level */

  opt.prof = prof;
  opt.count = profcnt;
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
  papi_return(PAPI_OK);
}

int PAPI_profil(unsigned short *buf, unsigned bufsiz, unsigned long offset, unsigned scale,
                int EventSet, int EventCode, int threshold, int flags)
{
  if (threshold > 0)
    {
      PAPI_sprofil_t *prof;

      prof = (PAPI_sprofil_t *)malloc(sizeof(PAPI_sprofil_t));
      memset(prof,0x0,sizeof(PAPI_sprofil_t));
      prof->pr_base = buf;
      prof->pr_size = bufsiz;
      prof->pr_off = offset;
      prof->pr_scale = scale;
      papi_return(PAPI_sprofil(prof,1,EventSet,EventCode,threshold,flags));
    }

  papi_return(PAPI_sprofil(NULL,0,EventSet,EventCode,0,flags));
}

int PAPI_set_granularity(int granularity)
{ 
  PAPI_option_t ptr;

  ptr.defgranularity.granularity = granularity;
  papi_return(PAPI_set_opt(PAPI_SET_GRANUL, &ptr));
}

int PAPI_set_domain(int domain)
{ 
  PAPI_option_t ptr;

  ptr.defdomain.domain = domain;
  papi_return(PAPI_set_opt(PAPI_SET_DOMAIN, &ptr));
}

int PAPI_add_events(int *EventSet, int *Events, int number)
{
  int i, retval;

  if (Events == NULL)
    papi_return(PAPI_EINVAL);

  for (i=0;i<number;i++)
    {
      retval = PAPI_add_event(EventSet, Events[i]);
      if (retval!=PAPI_OK) return(retval);
    }
  papi_return(PAPI_OK);
}

int PAPI_rem_events(int *EventSet, int *Events, int number)
{
  int i, retval;
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  if ((!EventSet) || (!Events))
    papi_return(PAPI_EINVAL);

  ESI=lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (!ESI)
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

#ifdef DEBUG
  /* Not necessary */
  if (ESI->NumberOfCounters == 0)
    papi_return(PAPI_EINVAL);
#endif

  if (number > ESI->NumberOfCounters)
    papi_return(PAPI_EINVAL);

  for (i=0; i<number; i++)
    {
      retval=PAPI_rem_event(EventSet, Events[i]);
      if(retval!=PAPI_OK) return(retval);
    }
  papi_return(PAPI_OK);
}


int PAPI_list_events(int EventSet, int *Events, int *number)
{
  EventSetInfo *ESI;
  int num;
  int i;

  if ((!Events) || (!number))
    papi_return(PAPI_EINVAL);

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (!ESI)
    papi_return(PAPI_ENOEVST);

#ifdef DEBUG
  /* Not necessary */
  if (ESI->NumberOfCounters == 0)
    papi_return(PAPI_EINVAL);
#endif

  if (*number < ESI->NumberOfCounters)
    num = *number;
  else
    num = ESI->NumberOfCounters;

  for(i=0; i<num; i++)
    Events[i] = ESI->EventInfoArray[i].code;

  *number = ESI->NumberOfCounters;

  papi_return(PAPI_OK);
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
  EventSetInfo *master = _papi_hwi_lookup_in_master_list();
  if (master)
    return(_papi_hwd_get_virt_cycles(master));
  return(-1);
}

long long PAPI_get_virt_usec(void)
{
  EventSetInfo *master = _papi_hwi_lookup_in_master_list();
  if (master)
    return(_papi_hwd_get_virt_usec(master));
  return(-1);
}
