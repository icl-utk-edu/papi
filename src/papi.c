/* 
* File:    papi.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

#ifndef _WIN32	/* for Linux/Unix systems */
  #include SUBSTRATE
#else			
  /* I couldn't figure out how to assign a string value 
	to a preprocessor directive in Windows, so we just
	include what we need... */
  #include "win32.h"
#endif

#include "papiStrings.h" /* for language independent string support. */

/********************/
/* BEGIN PROTOTYPES */
/********************/

#define DEADBEEF 0xdedbeef
#define PAPI_EVENTSET_MAP (&_papi_system_info.global_eventset_map)

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
static int get_free_EventCodeIndex(const EventSetInfo *ESI, int EventCode);
static int lookup_EventCodeIndex(const EventSetInfo *ESI,int EventCode);
static int remove_event(EventSetInfo *ESI, int EventCode);
static int default_error_handler(int errorCode);

/********************/
/*  END PROTOTYPES  */
/********************/

/********************/
/*  BEGIN GLOBALS   */ 
/********************/

EventSetInfo *default_master_eventset = NULL; 
unsigned long int (*thread_id_fn)(void) = NULL;
static int init_retval = DEADBEEF;
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
#ifdef DEBUG
#define papi_return(a) return(PAPI_ERR_HANDLER(a))
#else
#define papi_return(a) return(a)
#endif

/* Our informative table */

#define PAPI_PRESET(function)\
	function##_nm, function, function##_dsc, function##_lbl, 0, NULL, 0

static PAPI_preset_info_t papi_presets[PAPI_MAX_PRESET_EVENTS] = { 
  { PAPI_PRESET(PAPI_L1_DCM) },
  { PAPI_PRESET(PAPI_L1_ICM) },
  { PAPI_PRESET(PAPI_L2_DCM) },
  { PAPI_PRESET(PAPI_L2_ICM) },
  { PAPI_PRESET(PAPI_L3_DCM) },
  { PAPI_PRESET(PAPI_L3_ICM) },
  { PAPI_PRESET(PAPI_L1_TCM) },
  { PAPI_PRESET(PAPI_L2_TCM) },
  { PAPI_PRESET(PAPI_L3_TCM) },
  { PAPI_PRESET(PAPI_CA_SNP) },
  { PAPI_PRESET(PAPI_CA_SHR) },
  { PAPI_PRESET(PAPI_CA_CLN) },
  { PAPI_PRESET(PAPI_CA_INV) },
  { PAPI_PRESET(PAPI_CA_ITV) },
  { PAPI_PRESET(PAPI_L3_LDM) },
  { PAPI_PRESET(PAPI_L3_STM) },
  { PAPI_PRESET(PAPI_BRU_IDL) },
  { PAPI_PRESET(PAPI_FXU_IDL) },
  { PAPI_PRESET(PAPI_FPU_IDL) },
  { PAPI_PRESET(PAPI_LSU_IDL) },
  { PAPI_PRESET(PAPI_TLB_DM) },
  { PAPI_PRESET(PAPI_TLB_IM) },
  { PAPI_PRESET(PAPI_TLB_TL) },
  { PAPI_PRESET(PAPI_L1_LDM) },
  { PAPI_PRESET(PAPI_L1_STM) },
  { PAPI_PRESET(PAPI_L2_LDM) },
  { PAPI_PRESET(PAPI_L2_STM) },
  { PAPI_PRESET(PAPI_BTAC_M) },
  { PAPI_PRESET(PAPI_PRF_DM) },
  { PAPI_PRESET(PAPI_L3_DCH) },
  { PAPI_PRESET(PAPI_TLB_SD) },
  { PAPI_PRESET(PAPI_CSR_FAL) },
  { PAPI_PRESET(PAPI_CSR_SUC) },
  { PAPI_PRESET(PAPI_CSR_TOT) },
  { PAPI_PRESET(PAPI_MEM_SCY) },
  { PAPI_PRESET(PAPI_MEM_RCY) },
  { PAPI_PRESET(PAPI_MEM_WCY) },
  { PAPI_PRESET(PAPI_STL_ICY) },
  { PAPI_PRESET(PAPI_FUL_ICY) },
  { PAPI_PRESET(PAPI_STL_CCY) },
  { PAPI_PRESET(PAPI_FUL_CCY) },
  { PAPI_PRESET(PAPI_HW_INT) },
  { PAPI_PRESET(PAPI_BR_UCN) },
  { PAPI_PRESET(PAPI_BR_CN) },
  { PAPI_PRESET(PAPI_BR_TKN) },
  { PAPI_PRESET(PAPI_BR_NTK) },
  { PAPI_PRESET(PAPI_BR_MSP) },
  { PAPI_PRESET(PAPI_BR_PRC) },
  { PAPI_PRESET(PAPI_FMA_INS) },
  { PAPI_PRESET(PAPI_TOT_IIS) },
  { PAPI_PRESET(PAPI_TOT_INS) },
  { PAPI_PRESET(PAPI_INT_INS) },
  { PAPI_PRESET(PAPI_FP_INS) },
  { PAPI_PRESET(PAPI_LD_INS) },
  { PAPI_PRESET(PAPI_SR_INS) },
  { PAPI_PRESET(PAPI_BR_INS) },
  { PAPI_PRESET(PAPI_VEC_INS) },
  { PAPI_PRESET(PAPI_FLOPS) },
  { PAPI_PRESET(PAPI_RES_STL) },
  { PAPI_PRESET(PAPI_FP_STAL) },
  { PAPI_PRESET(PAPI_TOT_CYC) },
  { PAPI_PRESET(PAPI_IPS) },
  { PAPI_PRESET(PAPI_LST_INS) },
  { PAPI_PRESET(PAPI_SYC_INS) },
  { PAPI_PRESET(PAPI_L1_DCH) },
  { PAPI_PRESET(PAPI_L2_DCH) },
  { PAPI_PRESET(PAPI_L1_DCA) },
  { PAPI_PRESET(PAPI_L2_DCA) },
  { PAPI_PRESET(PAPI_L3_DCA) },
  { PAPI_PRESET(PAPI_L1_DCR) },
  { PAPI_PRESET(PAPI_L2_DCR) },
  { PAPI_PRESET(PAPI_L3_DCR) },
  { PAPI_PRESET(PAPI_L1_DCW) },
  { PAPI_PRESET(PAPI_L2_DCW) },
  { PAPI_PRESET(PAPI_L3_DCW) },
  { PAPI_PRESET(PAPI_L1_ICH) },
  { PAPI_PRESET(PAPI_L2_ICH) },
  { PAPI_PRESET(PAPI_L3_ICH) },
  { PAPI_PRESET(PAPI_L1_ICA) },
  { PAPI_PRESET(PAPI_L2_ICA) },
  { PAPI_PRESET(PAPI_L3_ICA) },
  { PAPI_PRESET(PAPI_L1_ICR) },
  { PAPI_PRESET(PAPI_L2_ICR) },
  { PAPI_PRESET(PAPI_L3_ICR) },
  { PAPI_PRESET(PAPI_L1_ICW) },
  { PAPI_PRESET(PAPI_L2_ICW) },
  { PAPI_PRESET(PAPI_L3_ICW) },
  { PAPI_PRESET(PAPI_L1_TCH) },
  { PAPI_PRESET(PAPI_L2_TCH) },
  { PAPI_PRESET(PAPI_L3_TCH) },
  { PAPI_PRESET(PAPI_L1_TCA) },
  { PAPI_PRESET(PAPI_L2_TCA) },
  { PAPI_PRESET(PAPI_L3_TCA) },
  { PAPI_PRESET(PAPI_L1_TCR) },
  { PAPI_PRESET(PAPI_L2_TCR) },
  { PAPI_PRESET(PAPI_L3_TCR) },
  { PAPI_PRESET(PAPI_L1_TCW) },
  { PAPI_PRESET(PAPI_L2_TCW) },
  { PAPI_PRESET(PAPI_L3_TCW) },
  { PAPI_PRESET(PAPI_FML_INS) },
  { PAPI_PRESET(PAPI_FAD_INS) },
  { PAPI_PRESET(PAPI_FDV_INS) },
  { PAPI_PRESET(PAPI_FSQ_INS) },
  { PAPI_PRESET(PAPI_FNV_INS) },
};

const char *papi_errNam[PAPI_NUM_ERRORS] = {
  PAPI_OK_nm,
  PAPI_EINVAL_nm,
  PAPI_ENOMEM_nm,
  PAPI_ESYS_nm,
  PAPI_ESBSTR_nm,
  PAPI_ECLOST_nm,
  PAPI_EBUG_nm,
  PAPI_ENOEVNT_nm,
  PAPI_ECNFLCT_nm,
  PAPI_ENOTRUN_nm,
  PAPI_EISRUN_nm,
  PAPI_ENOEVST_nm,
  PAPI_ENOTPRESET_nm,
  PAPI_ENOCNTR_nm,
  PAPI_EMISC_nm 
};

const char *papi_errStr[PAPI_NUM_ERRORS] = {
  PAPI_OK_dsc,
  PAPI_EINVAL_dsc,
  PAPI_ENOMEM_dsc,
  PAPI_ESYS_dsc,
  PAPI_ESBSTR_dsc,
  PAPI_ECLOST_dsc,
  PAPI_EBUG_dsc,
  PAPI_ENOEVNT_dsc,
  PAPI_ECNFLCT_dsc,
  PAPI_ENOTRUN_dsc,
  PAPI_EISRUN_dsc,
  PAPI_ENOEVST_dsc,
  PAPI_ENOTPRESET_dsc,
  PAPI_ENOCNTR_dsc,
  PAPI_EMISC_dsc
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
      fprintf(stderr,"%s %d: %s: %s\n",PAPI_ERROR_CODE_str,errorCode,papi_errNam[-errorCode],papi_errStr[-errorCode]);
      if (errorCode == PAPI_ESYS)
	perror("");
      return errorCode;
      break;
    case PAPI_VERB_ESTOP:
      fprintf(stderr,"%s %d: %s: %s\n",PAPI_ERROR_CODE_str,errorCode,papi_errNam[-errorCode],papi_errStr[-errorCode]);
      if (errorCode == PAPI_ESYS)
	perror("");
      exit(-errorCode);
      break;
    case PAPI_QUIET:
      return errorCode;
    default:
      abort();
    }
  return(PAPI_EBUG);
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
  
  master->hw_start = (long_long *)malloc(_papi_system_info.num_cntrs*sizeof(long_long));
  if (master->hw_start == NULL)
    {
      free(master->machdep);
      free(master);
      return(NULL);
    }
  memset(master->hw_start,0x00,_papi_system_info.num_cntrs*sizeof(long_long));
   
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
/* Thread support not implemented on Alpha/OSF because the OSF pfm
 * counter device driver does not support per-thread counters.
 * When this is updated, we can remove this ifdef -KSL
 */
#if defined(__ALPHA) && defined(__osf__)
    papi_return(PAPI_ESBSTR);
#endif
  if ((id_fn == NULL) || (flag != 0) || (default_master_eventset == NULL))
    papi_return(PAPI_EINVAL);
    
  if (thread_id_fn != NULL)
    {
      fprintf(stderr, PAPI_THREAD_INIT_str);
      exit(1);
    }

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
    return(PAPI_EINVAL);
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
  int i, tmp;

#ifdef DEBUG
  #ifdef _WIN32	/* don't want to define an environment variable... */
	papi_debug = 1;
  #else
	if (getenv("PAPI_DEBUG"))
	  papi_debug = 1;
  #endif
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

  _papi_system_info.total_events = 0 ;
  tmp = 0;  /* Count number of derived events */
  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++){
    if(papi_presets[i].avail>0) _papi_system_info.total_events += 1;
    if(papi_presets[i].flags & PAPI_DERIVED) tmp += 1;
  }
  _papi_system_info.total_presets = _papi_system_info.total_events - tmp;

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

static int EventInfoArrayLength(const EventSetInfo *ESI)
{
  if (ESI->state & PAPI_MULTIPLEXING)
    return(PAPI_MPX_DEF_DEG);
  else
    return(_papi_system_info.num_cntrs);
}
 
static void initialize_EventInfoArray(EventSetInfo *ESI)
{
  int i, limit = EventInfoArrayLength(ESI);

  for (i=0;i<limit;i++)
    {
      ESI->EventInfoArray[i].code = PAPI_NULL;
      ESI->EventInfoArray[i].selector = 0;
      ESI->EventInfoArray[i].command = NOT_DERIVED;
      ESI->EventInfoArray[i].operand_index = -1;
    }
}

static EventSetInfo *allocate_EventSet(void) 
{
  EventSetInfo *ESI;
  int max_counters;
  
  ESI=(EventSetInfo *)malloc(sizeof(EventSetInfo));
  if (ESI==NULL) 
    return(NULL); 
  memset(ESI,0x00,sizeof(EventSetInfo));

  max_counters = _papi_system_info.num_cntrs;
  ESI->machdep = (void *)malloc(_papi_system_info.size_machdep);
  ESI->sw_stop = (long_long *)malloc(max_counters*sizeof(long_long)); 
  ESI->hw_start = (long_long *)malloc(max_counters*sizeof(long_long));
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
  memset(ESI->sw_stop,          0x00,max_counters*sizeof(long_long)); 
  memset(ESI->hw_start,        0x00,max_counters*sizeof(long_long));

  initialize_EventInfoArray(ESI);

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

#if 0
static int get_granularity(DynamicArray *map, PAPI_granularity_option_t *opt)
{
  EventSetInfo *ESI;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, opt->eventset);
  if(ESI == NULL)
    papi_return(PAPI_ENOEVST);

  opt->granularity = ESI->granularity.granularity;
  papi_return(PAPI_OK);
}
#endif

/* From Anders Nilsson's (anni@pdc.kth.se) */

int PAPI_describe_event(char *name, int *EventCode, char *description)
{
  int retval;

  if (name == NULL)
   papi_return(PAPI_EINVAL);

  if (((int)strlen(name) == 0) && *EventCode == 0)
    papi_return(PAPI_EINVAL);

  if ((int)strlen(name) == 0)
    retval = PAPI_event_code_to_name(*EventCode, name);
  else
    retval = PAPI_event_name_to_code(name, EventCode);

  if (retval != PAPI_OK)
    papi_return(retval);

  if (description != NULL)
    {
      strncpy(description, papi_presets[*EventCode].event_descr, PAPI_MAX_STR_LEN);
    }
  papi_return(PAPI_OK);
}

int PAPI_label_event(int EventCode, char *label)
{
  if (EventCode == 0 || label == NULL)
    papi_return(PAPI_EINVAL);

  strncpy(label, papi_presets[EventCode].event_label, PAPI_MAX_STR_LEN);
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
	
      strncpy(out,papi_presets[EventCode].event_name,PAPI_MAX_STR_LEN);
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
  DBG((stderr,"create_eventset(%p,%p): new EventSet in slot %d\n",(void *)EventSet,handle,*EventSet));

  return(retval);
}

int PAPI_create_eventset(int *EventSet)
{
  EventSetInfo *master = _papi_hwi_lookup_in_master_list();
  int retval;
  if (master == NULL)
    {
      DBG((stderr,"PAPI_create_eventset(%p): new thread found\n",(void *)EventSet));
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

  /* No multiplexing pevents. */

  if (ESI->state & PAPI_MULTIPLEXING)
    papi_return(PAPI_EINVAL);

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

  if (ESI->state & PAPI_RUNNING)
    papi_return(PAPI_EISRUN);

  /* Now do the magic. */

  papi_return(add_event(ESI,EventCode));
}

/* This function returns the index of the the next free slot
   in the EventInfoArray. If EventCode is already in the list,
   it returns PAPI_ECNFLCT. */

static int get_free_EventCodeIndex(const EventSetInfo *ESI, int EventCode)
{
  int k;
  int lowslot = PAPI_ECNFLCT;
  int limit = EventInfoArrayLength(ESI);

  /* Check for duplicate events and get the lowest empty slot */
  
  for (k=0;k<limit;k++) 
    {
      if (ESI->EventInfoArray[k].code == EventCode)
	papi_return(PAPI_ECNFLCT);
      if ((ESI->EventInfoArray[k].code == PAPI_NULL) && (lowslot == PAPI_ECNFLCT))
	lowslot = k;
    }
  
  return(lowslot);
}

/* This function returns the index of the EventCode or error */
/* Index to what? The index to everything stored EventCode in the */
/* EventSet. */  

static int lookup_EventCodeIndex(const EventSetInfo *ESI, int EventCode)
{
  int i;
  int limit = EventInfoArrayLength(ESI);

  for(i=0;i<limit;i++) 
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

  assert(ESI->NumberOfEvents == 0);

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
  int thisindex, retval;

  /* Make sure the event is not present and get the next
     free slot. */

  retval = get_free_EventCodeIndex(ESI,EventCode);
  if (retval < PAPI_OK)
    return(retval);
  thisindex = retval;

  /* If it is a MPX EventSet, add it to the multiplex data structure and
     this threads multiplex list */

  if (ESI->state & PAPI_MULTIPLEXING)
    {
      retval = mpx_add_event(&ESI->multiplex,EventCode);
      if (retval < PAPI_OK)
	return(retval);

      /* just fill in the EventSetInfo array
	 with the relevant information. */

      ESI->EventInfoArray[thisindex].code = EventCode;      /* Relevant */
      ESI->EventInfoArray[thisindex].command = NOT_DERIVED; 
      ESI->EventInfoArray[thisindex].selector = 0;          
      ESI->EventInfoArray[thisindex].operand_index = -1;    
    }
  else
    {

      /* If this is a real EventSet, predecode the event into the 
	 machine dependent structure and then fill in the
	 EventInfoArray. */
      
      retval = _papi_hwd_add_event(ESI->machdep,EventCode,&ESI->EventInfoArray[thisindex]);
      if (retval < PAPI_OK)
	return(retval);

      /* The following may not be necessary */ 

      /* ESI->hw_start[thisindex]   = 0; */
      /* ESI->sw_stop[hwindex]     = 0; */
    }

  ESI->NumberOfEvents++;

  return(retval);
}

static int add_pevent(EventSetInfo *ESI, int EventCode, void *inout)
{
  int thisindex, retval;

  /* Make sure the event is not present and get a free slot. */

  retval = get_free_EventCodeIndex(ESI,EventCode);
  if (retval < PAPI_OK)
    return(retval);
  thisindex = retval;

  /* Fill in machine depending info including the EventInfoArray. */

  retval = _papi_hwd_add_prog_event(ESI->machdep,EventCode,inout,&ESI->EventInfoArray[thisindex]);
  if (retval < PAPI_OK)
    return(retval);

  /* Initialize everything left over. */

  /* ESI->sw_stop[thisindex]     = 0; */
  /* ESI->hw_start[thisindex]   = 0; */

  ESI->NumberOfEvents++;
  return(retval);
}

static int remove_event(EventSetInfo *ESI, int EventCode)
{
  int retval, thisindex;

  /* Make sure the event is preset. */

  retval = lookup_EventCodeIndex(ESI,EventCode);
  if (retval < PAPI_OK)
    return(retval);
  thisindex = retval;

  /* If it is a MPX EventSet, remove it from the multiplex data structure and
     this threads multiplex list */

  if (ESI->state & PAPI_MULTIPLEXING)
    {
      retval = mpx_remove_event(&ESI->multiplex,EventCode); 
      if (retval < PAPI_OK)
	return(retval);
   }
  else    
    /* Remove the events hardware dependant stuff from the EventSet */
    {
      retval = _papi_hwd_rem_event(ESI->machdep,&ESI->EventInfoArray[thisindex]);
      if (retval < PAPI_OK)
	return(retval);
    }

  /* Zero the EventInfoArray. */

  ESI->EventInfoArray[thisindex].code = PAPI_NULL;
  ESI->EventInfoArray[thisindex].command = NOT_DERIVED;
  ESI->EventInfoArray[thisindex].selector = 0;
  ESI->EventInfoArray[thisindex].operand_index = -1;

  /* ESI->sw_stop[hwindex]           = 0; */
  /* ESI->hw_start[hwindex]         = 0; */

  ESI->NumberOfEvents--;

  return(retval);
}

int PAPI_rem_event(int *EventSet, int EventCode)
{
  EventSetInfo *ESI;

  /* check for pre-existing ESI */

  if (EventSet == NULL)
    papi_return(PAPI_EINVAL);

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);

  /* Of course, it must be stopped in order to modify it. */

  if (!(ESI->state & PAPI_STOPPED))
    papi_return(PAPI_EISRUN);

  /* If it is a MPX EventSet, do the right thing */

  if (ESI->state & PAPI_MULTIPLEXING)
    papi_return(mpx_remove_event(&ESI->multiplex,EventCode));

  /* Now do the magic. */

  papi_return(remove_event(ESI,EventCode));
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

  if (ESI->NumberOfEvents)
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

  if (ESI->NumberOfEvents < 1)
    papi_return(PAPI_EINVAL);

  /* If multiplexing is enabled for this eventset,
     call John May's code. */

  if (ESI->state & PAPI_MULTIPLEXING)
    {
      retval = MPX_start(ESI->multiplex);
      if (retval != PAPI_OK)
	papi_return(retval);

      /* Update the state of this EventSet */
      
      ESI->state ^= PAPI_STOPPED;
      ESI->state |= PAPI_RUNNING;

      return(PAPI_OK);
    }

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
	papi_return(retval);
    }
  
  /* Merge the control bits from the new EventSet into the active counter config. */
  
  retval = _papi_hwd_merge(ESI, thread_master_eventset);
  if (retval != PAPI_OK)
    papi_return(retval);
  
  /* Update the state of this EventSet */
  
  ESI->state ^= PAPI_STOPPED;
  ESI->state |= PAPI_RUNNING;
  
  /* Update the number of active EventSets for this thread */
  
  thread_master_eventset->multistart.num_runners++;
  
  DBG((stderr,"PAPI_start returns %d\n",retval));
  return(retval);
}

/* checks for valid EventSet, calls substrate stop() fxn. */

int PAPI_stop(int EventSet, long_long *values)
{ 
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;
  int retval;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI==NULL)
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (!(ESI->state & PAPI_RUNNING))
    papi_return(PAPI_ENOTRUN);

  /* If multiplexing is enabled for this eventset,
     call John May's code. */

  if (ESI->state & PAPI_MULTIPLEXING)
    {
      retval = MPX_stop(ESI->multiplex, values);
      if (retval != PAPI_OK)
	papi_return(retval);

      /* Update the state of this EventSet */

      ESI->state ^= PAPI_RUNNING;
      ESI->state |= PAPI_STOPPED;

      return(PAPI_OK);
    }
      
  /* Read the current counter values into the EventSet */

  retval = _papi_hwd_read(ESI, thread_master_eventset, ESI->sw_stop);
  if (retval != PAPI_OK)
    papi_return(retval);

  /* If overflowing is enabled, turn it off */

  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_stop_overflow_timer(ESI, thread_master_eventset);
      if (retval < PAPI_OK)
	papi_return(retval);
    }
  
  /* Remove the control bits from the active counter config. */

  retval = _papi_hwd_unmerge(ESI, thread_master_eventset);
  if (retval != PAPI_OK)
    papi_return(retval);

  if (values)
    memcpy(values,ESI->sw_stop,ESI->NumberOfEvents*sizeof(long_long)); 

  /* Update the state of this EventSet */

  ESI->state ^= PAPI_RUNNING;
  ESI->state |= PAPI_STOPPED;

  /* Update the number of active EventSets for this thread */

  thread_master_eventset->multistart.num_runners --;

#if defined(DEBUG)
  {
    int i;
    for (i=0;i<ESI->NumberOfEvents;i++)
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
      if (ESI->state & PAPI_MULTIPLEXING)
	retval = MPX_reset(ESI->multiplex);
      else
	{
	  /* If we're not the only one running, then just
	     read the current values into the ESI->start
	     array. This holds the starting value for counters
	     that are shared. */
	  
	  retval = _papi_hwd_reset(ESI, thread_master_eventset);
	}
      if (retval != PAPI_OK)
	papi_return(retval);
    }
  else
    {
      memset(ESI->sw_stop,0x00,ESI->NumberOfEvents*sizeof(long_long)); 
    }

  DBG((stderr,"PAPI_reset returns %d\n",retval));
  return(retval);
}

int PAPI_read(int EventSet, long_long *values)
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
      if (ESI->state & PAPI_MULTIPLEXING)
	retval = MPX_read(ESI->multiplex, values);
      else
	retval = _papi_hwd_read(ESI, thread_master_eventset, values);
      if (retval != PAPI_OK)
        papi_return(retval);
    }
  else
    {
      memcpy(values,ESI->sw_stop,ESI->NumberOfEvents*sizeof(long_long)); 
    }

#if defined(DEBUG)
  {
    int i;
    for (i=0;i<ESI->NumberOfEvents;i++)
    DBG((stderr,"PAPI_read values[%d]:\t%llu\n",i,values[i]));
  }
#endif

  DBG((stderr,"PAPI_read returns %d\n",retval));
  return(retval);
}

int PAPI_accum(int EventSet, long_long *values)
{ 
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;
  int i, retval;
  long_long a,b,c;

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  if (values == NULL)
    papi_return(PAPI_EINVAL);

  if (ESI->state & PAPI_RUNNING)
    {
      if (ESI->state & PAPI_MULTIPLEXING)
        retval = MPX_read(ESI->multiplex, ESI->sw_stop);
      else
        retval = _papi_hwd_read(ESI, thread_master_eventset, ESI->sw_stop);
      if (retval != PAPI_OK)
        papi_return(retval);
    }
  
  for (i=0 ; i < ESI->NumberOfEvents; i++)
    {
      a = ESI->sw_stop[i];
      b = values[i];
      c = a + b;
      values[i] = c;
    } 

  papi_return(PAPI_reset(EventSet));
}

int PAPI_write(int EventSet, long_long *values)
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

  if (ESI->state & PAPI_RUNNING)
    {
      retval = _papi_hwd_write(thread_master_eventset, ESI, values);
      if (retval!=PAPI_OK)
        return(retval);
    }

  memcpy(ESI->hw_start,values,_papi_system_info.num_cntrs*sizeof(long_long));

  return(retval);
}

static int cleanup_eventset(EventSetInfo *ESI)
{
  int retval, i, tmp = ESI->NumberOfEvents;

  if (ESI->state & PAPI_MULTIPLEXING)
    {
      retval = MPX_cleanup(&ESI->multiplex);
      if (retval != PAPI_OK)
	return(retval);
    }
  
  for(i=0;i<tmp;i++) 
    {
      retval = remove_event(ESI, ESI->EventInfoArray[i].code);
      if (retval != PAPI_OK)
	return(retval);
    }

  return(PAPI_OK);
}

/*  The function PAPI_cleanup removes a stopped EventSet from existence. */

int PAPI_cleanup_eventset(int *EventSet) 
{ 
  EventSetInfo *ESI;
  EventSetInfo *thread_master_eventset;

  /* Is the EventSet already in existence? */

  if (EventSet == NULL)
    papi_return(PAPI_EINVAL);

  ESI = lookup_EventSet(PAPI_EVENTSET_MAP, *EventSet);
  if (ESI == NULL)
    papi_return(PAPI_ENOEVST);
  thread_master_eventset = ESI->master;

  /* Of course, it must be stopped in order to modify it. */

  if (ESI->state & PAPI_RUNNING) 
    papi_return(PAPI_EISRUN);
  
  /* Now do the magic */

  papi_return(cleanup_eventset(ESI));
}
 
int PAPI_multiplex_init(void)
{
  int retval;

  retval = mpx_init(PAPI_MPX_DEF_US);
  papi_return(retval);
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

int PAPI_set_multiplex(int *EventSet)
{
  PAPI_option_t mpx;

  if (EventSet == NULL) 
    papi_return(PAPI_EINVAL);

  mpx.multiplex.eventset = *EventSet;
  mpx.multiplex.us = PAPI_MPX_DEF_US;
  mpx.multiplex.max_degree = PAPI_MPX_DEF_DEG;
  
  return(PAPI_set_opt(PAPI_SET_MULTIPLEX,&mpx));
}

static int convert_eventset_to_multiplex(EventSetInfo *ESI)
{
  EventInfo_t *tmp;

  tmp = (EventInfo_t *)malloc(PAPI_MPX_DEF_DEG*sizeof(EventInfo_t));
  if (tmp == NULL)
    return(PAPI_ENOMEM);

  /* If there are any events in the EventSet, 
     convert them to multiplex events */

  if (ESI->NumberOfEvents)
    {
      int retval, i, *mpxlist = (int *)malloc(sizeof(int)*ESI->NumberOfEvents);
      if (mpxlist == NULL)
	{
	  free(tmp);
	  return(PAPI_ENOMEM);
	}

      /* Build the args to MPX_add_events(). */
      
      for (i=0;i<ESI->NumberOfEvents;i++)
	mpxlist[i] = ESI->EventInfoArray[i].code;	      
      
      retval = MPX_add_events(&ESI->multiplex,mpxlist,i);
      if (retval != PAPI_OK)
	{
	  free(mpxlist);
	  free(tmp);
	  return(retval);
	}
      free(mpxlist);
    }
  
  /* Resize the EventInfo_t array */
  
  free(ESI->EventInfoArray);
  ESI->EventInfoArray = tmp;
  ESI->state |= PAPI_MULTIPLEXING;

  /* Update the state */
  /* Bug fix, state must be enabled, as ESI is an argument. */

  initialize_EventInfoArray(ESI);
  
  return(PAPI_OK);
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
    case PAPI_SET_MULTIPLEX:
      {
	EventSetInfo *ESI;

	if (ptr->multiplex.us < 1)
	  papi_return(PAPI_EINVAL);
	if (ptr->multiplex.max_degree <= _papi_system_info.num_cntrs) {
	  papi_return(PAPI_OK);
        }
        ESI = lookup_EventSet(PAPI_EVENTSET_MAP, ptr->multiplex.eventset);
	if (ESI == NULL)
	  papi_return(PAPI_ENOEVST);
	if (!(ESI->state & PAPI_STOPPED))
	  papi_return(PAPI_EISRUN);
	if (ESI->state & PAPI_MULTIPLEXING)
	  papi_return(PAPI_EINVAL);

	papi_return(convert_eventset_to_multiplex(ESI));
      }
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
#if 0
    case PAPI_SET_GRANUL:
      {
        int grn = ptr->granularity.granularity;

        if ((grn < PAPI_GRN_MIN) || (grn > PAPI_GRN_MAX))
          papi_return(PAPI_EINVAL);

        internal.granularity.ESI = lookup_EventSet(PAPI_EVENTSET_MAP, ptr->granularity.eventset);
        if (internal.granularity.ESI == NULL)
          papi_return(PAPI_ENOEVST);

        internal.granularity.granularity = grn;
        internal.granularity.eventset = ptr->granularity.eventset;
        retval = _papi_hwd_ctl(NULL, PAPI_SET_GRANUL, &internal);
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
#endif
    default:
      papi_return(PAPI_EINVAL);
    }
}

int PAPI_get_opt(int option, PAPI_option_t *ptr) 
{ 
  switch(option)
    {
    case PAPI_GET_MULTIPLEX:
      break;
    case PAPI_GET_PRELOAD:
      strncpy(ptr->preload.lib_preload_env,_papi_system_info.exe_info.lib_preload_env,
	      PAPI_MAX_STR_LEN);
      break;
    case PAPI_GET_DEBUG:
      ptr->debug.level = PAPI_ERR_LEVEL;
      ptr->debug.handler = PAPI_ERR_HANDLER;
      break;
    case PAPI_GET_CLOCKRATE:
      return((int)_papi_system_info.hw_info.mhz);
    case PAPI_GET_MAX_CPUS:
      return(_papi_system_info.hw_info.ncpu);
    case PAPI_GET_MAX_HWCTRS:
      return(_papi_system_info.num_cntrs);
    case PAPI_GET_DEFDOM:
      return(_papi_system_info.default_domain);
    case PAPI_GET_DEFGRN:
      return(_papi_system_info.default_granularity);
#if 0
    case PAPI_GET_INHERIT:
      {
	EventSetInfo *tmp;
	tmp = _papi_hwi_lookup_in_master_list();
	if (tmp == NULL)
	  return(PAPI_EINVAL);
	
	return(tmp->inherit.inherit); 
      }
    case PAPI_GET_GRANUL:
      if (ptr == NULL)
	papi_return(PAPI_EINVAL);
      return(get_granularity(PAPI_EVENTSET_MAP, &ptr->granularity));
#endif
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
    default:
      papi_return(PAPI_EINVAL);
    }
  papi_return(PAPI_OK);
} 

void PAPI_shutdown(void) 
{
  int i, status;

  if(init_retval == DEADBEEF) {
    fprintf(stderr, PAPI_SHUTDOWN_str);
    return;
  }

  for (i=0;i<PAPI_EVENTSET_MAP->totalSlots;i++) 
    {
      if (PAPI_EVENTSET_MAP->dataSlotArray[i]) 
	{
	  PAPI_state(i,&status);
	  if (status & PAPI_RUNNING)
	    PAPI_stop(i,NULL);
          _papi_hwd_shutdown(PAPI_EVENTSET_MAP->dataSlotArray[i]);
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
  _papi_hwd_shutdown_global();
  init_retval = DEADBEEF;

  /* Clean up thread stuff */

  thread_id_fn = NULL;
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
  if (foo == NULL)
    papi_return(PAPI_EINVAL);

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
                          long_long *values, int *threshold, void *context)
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

  if ((Events == NULL) || (number < 0))
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
  if (ESI->NumberOfEvents == 0)
    papi_return(PAPI_EINVAL);
#endif

  if ((number > ESI->NumberOfEvents) || (number < 0))
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
  if (ESI->NumberOfEvents == 0)
    papi_return(PAPI_EINVAL);
#endif

  if (*number < ESI->NumberOfEvents)
    num = *number;
  else
    num = ESI->NumberOfEvents;

  for(i=0; i<num; i++)
    Events[i] = ESI->EventInfoArray[i].code;

  *number = ESI->NumberOfEvents;

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

long_long PAPI_get_real_cyc(void)
{
  return(_papi_hwd_get_real_cycles());
}

long_long PAPI_get_real_usec(void)
{
  return(_papi_hwd_get_real_usec());
}

long_long PAPI_get_virt_cyc(void)
{
  EventSetInfo *master = _papi_hwi_lookup_in_master_list();
  if (master)
    return(_papi_hwd_get_virt_cycles(master));
  return(-1);
}

long_long PAPI_get_virt_usec(void)
{
  EventSetInfo *master = _papi_hwi_lookup_in_master_list();
  if (master)
    return(_papi_hwd_get_virt_usec(master));
  return(-1);
}

int PAPI_restore(void)
{
  fprintf(stderr,"PAPI_restore is currently not implemented\n");
  return(PAPI_ESBSTR);
}
int PAPI_save(void)
{
  fprintf(stderr,"PAPI_save is currently not implemented\n");
  return(PAPI_ESBSTR);
}
