/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi_internal.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
* Mods:    Kevin London
*	   london@cs.utk.edu
* Mods:    Per Ekman
*          pek@pdc.kth.se
* Mods:    <your name here>
*          <your email address>
*/  

#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
  #define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

#include "papiStrings.h"


/********************/
/* BEGIN PROTOTYPES */
/********************/

/* Defined in this file */
static int default_error_handler(int errorCode);

extern unsigned long int (*_papi_hwi_thread_id_fn)(void);


/********************/
/*  END PROTOTYPES  */
/********************/

/********************/
/*  BEGIN GLOBALS   */ 
/********************/

/* Defined by the substrate */
extern hwd_preset_t _papi_hwd_preset_map[];

/* Defined in papi_data.c */
extern PAPI_preset_info_t _papi_hwi_presets[];
extern int _papi_hwi_debug;

ThreadInfo_t *default_master_thread = NULL; 

/* Machine dependent info structure */
extern papi_mdi_t _papi_hwi_system_info;

/********************/
/*  BEGIN LOCALS    */ 
/********************/

int _papi_hwi_error_level = PAPI_QUIET; 
PAPI_debug_handler_t _papi_hwi_debug_handler = default_error_handler;

#ifdef DEBUG
#define papi_return(a) return(_papi_hwi_debug_handler(a))
#else
#define papi_return(a) return(a)
#endif

/********************/
/*    END LOCALS    */ 
/********************/

/* Utility functions */

int default_error_handler(int errorCode)
{
  extern char *_papi_hwi_errNam[], *_papi_hwi_errStr[];

  if (errorCode == PAPI_OK)
    return(errorCode);

  if ((errorCode > 0) || (-errorCode > PAPI_NUM_ERRORS))
    abort();

  switch (_papi_hwi_error_level)
    {
    case PAPI_VERB_ECONT:
    case PAPI_VERB_ESTOP:
      fprintf(stderr,"%s %d: ",PAPI_ERROR_CODE_str,errorCode);
      /* gcc 2.96 bug fix, do not change */
      /* fprintf(stderr,"%s %d: %s: %s\n",PAPI_ERROR_CODE_str,errorCode,_papi_hwi_errNam[-errorCode],_papi_hwi_errStr[-errorCode]); */
      fputs(_papi_hwi_errNam[-errorCode],stderr);
      fputs(", ",stderr);
      fputs(_papi_hwi_errStr[-errorCode],stderr);
      if (errorCode == PAPI_ESYS)
	{
	  fprintf(stderr,": ");
	  perror("");
	}
      else
	fprintf(stderr,"\n");
      if (_papi_hwi_error_level == PAPI_VERB_ESTOP)
	exit(-errorCode);
      else
	return errorCode;
      break;
    case PAPI_QUIET:
      return errorCode;
    default:
      abort();
    }
  return(PAPI_EBUG);
}

int _papi_hwi_allocate_eventset_map(void)
{
  DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;

  /* Allocate and clear the Dynamic Array structure */
  
  memset(map,0x00,sizeof(DynamicArray_t));

  /* Allocate space for the EventSetInfo_t pointers */

  map->dataSlotArray = 
    (EventSetInfo_t **)malloc(PAPI_INIT_SLOTS*sizeof(EventSetInfo_t *));
  if(map->dataSlotArray == NULL) 
    {
      free(map);
      return(1);
    }
  memset(map->dataSlotArray,0x00, 
	 PAPI_INIT_SLOTS*sizeof(EventSetInfo_t *));

  map->totalSlots = PAPI_INIT_SLOTS;
  map->availSlots = PAPI_INIT_SLOTS;
  map->fullSlots  = 0;
  map->lowestEmptySlot = 0;
  
  return(0);
}

static void free_thread(ThreadInfo_t **master)
{
  memset(*master,0x00,sizeof(ThreadInfo_t));
  free(*master);
  *master = NULL;
}

static ThreadInfo_t *allocate_new_thread(void)
{
  ThreadInfo_t *master;
  
  /* The Master EventSet is special. It is not in the EventSet list, but is pointed
     to by each EventSet of that particular thread. */
  
  master = (ThreadInfo_t *)malloc(sizeof(ThreadInfo_t));
  if (master == NULL)
    return(NULL);
  memset(master,0x00,sizeof(ThreadInfo_t));

  return(master);
}

int _papi_hwi_initialize_thread(ThreadInfo_t **master)
{
  int retval;

  if ((*master = allocate_new_thread()) == NULL)
    papi_return(PAPI_ENOMEM);

  /* Call the substrate to fill in anything special. */
  
  retval = _papi_hwd_init(&((*master)->context));
  if (retval)
    {
      free_thread(master);
      return(retval);
    }

  if (_papi_hwi_thread_id_fn)
    (*master)->tid = (*_papi_hwi_thread_id_fn)();
  return(PAPI_OK);
}

static int expand_dynamic_array(DynamicArray_t *DA)
{
  int number;	
  EventSetInfo_t **n;

  /*realloc existing PAPI_EVENTSET_MAP.dataSlotArray*/
    
  number = DA->totalSlots*2;
  n = (EventSetInfo_t **)realloc(DA->dataSlotArray,number*sizeof(EventSetInfo_t *));
  if (n==NULL)
    papi_return(PAPI_ENOMEM);

  /* Need to assign this value, what if realloc moved it? */

  DA->dataSlotArray = n;

  memset(DA->dataSlotArray+DA->totalSlots,0x00,DA->totalSlots*sizeof(EventSetInfo_t *));

  DA->totalSlots = number;
  DA->availSlots = number - DA->fullSlots;
  DA->lowestEmptySlot = DA->totalSlots/2;

  return(PAPI_OK);
}

/*========================================================================*/
/* This function allocates space for one EventSetInfo_t structure and for */
/* all of the pointers in this structure.  If any malloc in this function */
/* fails, all memory malloced to the point of failure is freed, and NULL  */
/* is returned.  Upon success, a pointer to the EventSetInfo_t data       */
/* structure is returned.                                                 */
/*========================================================================*/

static int EventInfoArrayLength(const EventSetInfo_t *ESI)
{
  if (ESI->state & PAPI_MULTIPLEXING)
    return(PAPI_MPX_DEF_DEG);
  else
    return(_papi_hwi_system_info.num_cntrs);
}
 
static void initialize_EventInfoArray(EventSetInfo_t *ESI)
{
  int i, limit = EventInfoArrayLength(ESI);

  for (i=0;i<limit;i++)
    {
      ESI->EventInfoArray[i].event_code = PAPI_NULL;
      ESI->EventInfoArray[i].hardware_selector = 0;
      ESI->EventInfoArray[i].command = NOT_DERIVED;
      ESI->EventInfoArray[i].operand_index = -1;
      ESI->EventInfoArray[i].index = i;
    }
}

EventSetInfo_t *_papi_hwi_allocate_EventSet(void) 
{
  EventSetInfo_t *ESI;
  int max_counters;
  
  ESI=(EventSetInfo_t *)malloc(sizeof(EventSetInfo_t));
  if (ESI==NULL) 
    return(NULL); 
  memset(ESI,0x00,sizeof(EventSetInfo_t));

  max_counters = _papi_hwi_system_info.num_cntrs;
/*  ESI->machdep = (hwd_control_state_t *)malloc(sizeof(hwd_control_state_t)); */
  ESI->sw_stop = (long_long *)malloc(max_counters*sizeof(long_long)); 
  ESI->hw_start = (long_long *)malloc(max_counters*sizeof(long_long));
  ESI->EventInfoArray = (EventInfo_t *)malloc(max_counters*sizeof(EventInfo_t));

  if (
/*    (ESI->machdep        == NULL )  || */
      (ESI->sw_stop           == NULL )  || 
      (ESI->hw_start         == NULL )  ||
      (ESI->EventInfoArray == NULL ))
    {
/*      if (ESI->machdep)        free(ESI->machdep); */
      if (ESI->sw_stop)           free(ESI->sw_stop); 
      if (ESI->hw_start)         free(ESI->hw_start);
      if (ESI->EventInfoArray) free(ESI->EventInfoArray);
      free(ESI);
      return(NULL);
    }
/*  memset(ESI->machdep,       0x00,_papi_system_info.size_machdep); */
  memset(ESI->sw_stop,          0x00,max_counters*sizeof(long_long)); 
  memset(ESI->hw_start,        0x00,max_counters*sizeof(long_long));

  initialize_EventInfoArray(ESI);

  ESI->state = PAPI_STOPPED; 

  /* ESI->domain.domain = 0;
     ESI->granularity.granularity = 0; */

  return(ESI);
}

/*========================================================================*/
/* This function should free memory for one EventSetInfo_t structure.     */
/* The argument list consists of a pointer to the EventSetInfo_t          */
/* structure, *ESI.                                                       */
/* The calling function should check  for ESI==NULL.                      */
/*========================================================================*/

static void free_EventSet(EventSetInfo_t *ESI) 
{
  if (ESI->EventInfoArray) free(ESI->EventInfoArray);
/*  if (ESI->machdep)        free(ESI->machdep); */
  if (ESI->sw_stop)        free(ESI->sw_stop); 
  if (ESI->hw_start)       free(ESI->hw_start);
  if ((ESI->state&PAPI_MULTIPLEXING)&&ESI->multiplex)
    free(ESI->multiplex);

#ifdef DEBUG
  memset(ESI,0x00,sizeof(EventSetInfo_t));
#endif
  free(ESI);
}

static int add_EventSet(EventSetInfo_t *ESI, ThreadInfo_t *master)
{
  DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
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

EventSetInfo_t *_papi_hwi_lookup_EventSet(int eventset)
{
  const DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;

  if ((eventset < 0) || (eventset >= map->totalSlots))
    return(NULL);
  return(map->dataSlotArray[eventset]);
}

int _papi_hwi_create_eventset(int *EventSet, ThreadInfo_t *handle)
{
  EventSetInfo_t *ESI;
  int retval;

  /* Is the EventSet already in existence? */
  
  if ((EventSet == NULL) || (handle == NULL))
    return(PAPI_EINVAL);

  /* Well, then allocate a new one. Use n to keep track of a NEW EventSet */
  
  ESI = _papi_hwi_allocate_EventSet();
  if (ESI == NULL)
    return(PAPI_ENOMEM);

  /* Add it to the global table */

  retval = add_EventSet(ESI, handle);
  if (retval < PAPI_OK)
    {
      free_EventSet(ESI);
      return(retval);
    }
  
  *EventSet = ESI->EventSetIndex;
  DBG((stderr,"_papi_hwi_create_eventset(%p,%p): new EventSet in slot %d\n",(void *)EventSet,handle,*EventSet));
if (*EventSet > 100) exit(0);
  return(retval);
}

int _papi_hwi_get_domain(PAPI_domain_option_t *opt)
{
  EventSetInfo_t *ESI;

  ESI = _papi_hwi_lookup_EventSet(opt->eventset);
  if(ESI == NULL)
    papi_return(PAPI_ENOEVST);

  opt->domain = ESI->domain.domain;
  return(PAPI_OK);
}

#if 0
int _papi_hwi_get_granularity(PAPI_granularity_option_t *opt)
{
  EventSetInfo_t *ESI;

  ESI = _papi_hwi_lookup_EventSet(opt->eventset);
  if(ESI == NULL)
    papi_return(PAPI_ENOEVST);

  opt->granularity = ESI->granularity.granularity;
  papi_return(PAPI_OK);
}
#endif

static int add_preset_event(hwd_control_state_t *machdep, hwd_preset_t *preset, EventInfo_t *evi)
{
  hwd_register_map_t *result = &evi->bits;

  memset(result,0x0,sizeof(hwd_register_map_t));

  /* The EventSet must keep an overall 'map' of which registers are in
     use. This map is machine dependent, and the structure is defined
     in the substrate include file. Then it is 'typedef'ed to a
     hwd_register_t. On most systems, this is a single bit mask
     with an additional field containing the number of bits that
     are on. On the P4, there are 2 registers to allocate, so it
     needs 2 masks, and 2 total counts. */

  /* get_allocated_counters_in_eventset(machdep,&already_taken); */

  /* The preset also keeps a 'map', but it is different. The preset's
     map contains ALL the possible registers that this preset can use. 
     For events that require more than 1 hardware event, (like derived
     events and/or the P4 tagging mechanism) we need to have multiple
     masks, one per register used. So the hwd_register_map_t is 
     actually an array of hwd_register_t's. */

  /* num = get_needed_counters_in_preset(preset,&needed); */

  /* Now we call a machine dependent function to compare if there are
     enough registers available to use this preset. Store the allocated
     bits in 'result'. */

  if (_papi_hwd_allocate_registers(machdep, preset, result))
    return(PAPI_ECNFLCT);

  /* Now we call a machine dependent function to use the allocated
     register bits to stuff the proper values into our counter control
     structure */

  evi->hardware_index = _papi_hwd_add_event(result, preset, machdep);

  /* After this function is called, ESI->machdep has everything it 
     needs to do a start/read/stop as quickly as possible */

  return(PAPI_OK);
}

/* This function returns the index of the the next free slot
   in the EventInfoArray. If EventCode is already in the list,
   it returns PAPI_ECNFLCT. */

static int get_free_EventCodeIndex(const EventSetInfo_t *ESI, unsigned int EventCode)
{
  int k;
  int lowslot = PAPI_ECNFLCT;
  int limit = EventInfoArrayLength(ESI);

  /* Check for duplicate events and get the lowest empty slot */
  
  for (k=0;k<limit;k++) 
    {
      if (ESI->EventInfoArray[k].event_code == EventCode)
	papi_return(PAPI_ECNFLCT);
      if ((ESI->EventInfoArray[k].event_code == PAPI_NULL) && (lowslot == PAPI_ECNFLCT))
	lowslot = k;
    }
  
  return(lowslot);
}

/* This function returns the index of the EventCode or error */
/* Index to what? The index to everything stored EventCode in the */
/* EventSet. */  

int _papi_hwi_lookup_EventCodeIndex(const EventSetInfo_t *ESI, unsigned int EventCode)
{
  int i;
  int limit = EventInfoArrayLength(ESI);

  for(i=0;i<limit;i++) 
    { 
      if (ESI->EventInfoArray[i].event_code == EventCode) 
	return(i);
    }

  return(PAPI_EINVAL);
} 

/* Return the EventSetInfo_t to which this EventInfo belongs */
EventSetInfo_t *get_my_EventSetInfo(EventInfo_t *me) {
    const DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
    EventSetInfo_t *ESI;
    int ei, i;
    
    for (ei = 0; ei < map->totalSlots; ei++) {
       if (( ESI = _papi_hwi_lookup_EventSet(ei)) != NULL) {
           for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
               if (&ESI->EventInfoArray[i] == me) 
                   return ESI;
           }
       }
    }

    return NULL;
}


/* This function only removes empty EventSets */

int _papi_hwi_remove_EventSet(EventSetInfo_t *ESI)
{
  DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
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

  return(PAPI_OK);
}

int _papi_hwi_add_event(EventSetInfo_t *ESI, int EventCode)
{
  int thisindex, retval;

  /* Make sure the event is not present and get the next
     free slot. */

  thisindex = get_free_EventCodeIndex(ESI,EventCode);
  if (thisindex < PAPI_OK)
    return(thisindex);

  /* If it is a MPX EventSet, add it to the multiplex data structure and
     this threads multiplex list */

  if (!(ESI->state & PAPI_MULTIPLEXING))
    {
      int preset_index = EventCode ^ PRESET_MASK;

      if (EventCode & PRESET_MASK)
	{
	  /* Check if it's within the valid range */

	  if ((preset_index < 0) || (preset_index >= PAPI_MAX_PRESET_EVENTS))
	    return(PAPI_EINVAL);

	  /* Check if event exists */

	  if (!_papi_hwi_presets[preset_index].avail)
	    return(PAPI_ENOEVNT);

	  /* Try to add the preset. */

	  retval = add_preset_event((hwd_control_state_t *)(&ESI->machdep), 
				    &_papi_hwd_preset_map[preset_index], 
				    &ESI->EventInfoArray[thisindex]);
	  if (retval < PAPI_OK)
	    return(retval);

	  /* Fill in the EventCode (machine independent) information */

	  ESI->EventInfoArray[thisindex].event_code = EventCode; 
	}
      else
	{
	  /* Native events that can be encoded in sizeof(int) go here. */

	  return(PAPI_EBUG);
	}
    }
  else
    {
      /* Multiplexing is special. See multiplex.c */
      retval = mpx_add_event(&ESI->multiplex,EventCode);
      if (retval < PAPI_OK)
	return(retval);
    }

  /* Bump the number of events */
  ESI->NumberOfEvents++;

  return(retval);
}

int _papi_hwi_add_pevent(EventSetInfo_t *ESI, int EventCode, void *inout)
{
  int thisindex, retval;

  /* Make sure the event is not present and get a free slot. */

  thisindex = get_free_EventCodeIndex(ESI,EventCode);
  if (thisindex < PAPI_OK)
    return(thisindex);

  /* Fill in machine depending info including the EventInfoArray. */

  retval = _papi_hwd_add_prog_event(&ESI->machdep,EventCode,inout,&ESI->EventInfoArray[thisindex]);
  if (retval < PAPI_OK)
    return(retval);

  /* Initialize everything left over. */

  /* ESI->sw_stop[thisindex]     = 0; */
  /* ESI->hw_start[thisindex]   = 0; */

  ESI->NumberOfEvents++;
  return(retval);
}

int _papi_hwi_remove_event(EventSetInfo_t *ESI, int EventCode)
{
  int i, j = 0, retval, thisindex;

  /* Make sure the event is preset. */

  thisindex = _papi_hwi_lookup_EventCodeIndex(ESI,EventCode);
  if (thisindex < PAPI_OK)
    return(thisindex);

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
      retval = _papi_hwd_remove_event(&ESI->EventInfoArray[thisindex].bits, ESI->EventInfoArray[thisindex].hardware_index, &ESI->machdep);
      if (retval < PAPI_OK)
	return(retval);
    }

  /* Move the hardware_index's around. */

  for (i=0;i<EventInfoArrayLength(ESI);i++)
    {
      if (ESI->EventInfoArray[i].hardware_index < ESI->EventInfoArray[thisindex].hardware_index)
	;
      else if (ESI->EventInfoArray[i].hardware_index == ESI->EventInfoArray[thisindex].hardware_index)
	{
	  ESI->EventInfoArray[i].event_code = PAPI_NULL;
	  ESI->EventInfoArray[i].hardware_index = -1;
	}
      else
	{
	  ESI->EventInfoArray[i].hardware_index = -1;
	}

      if (++j == ESI->NumberOfEvents)
	break;
    }
	
  /* ESI->EventInfoArray[thisindex].command = NOT_DERIVED; */
  /* ESI->EventInfoArray[thisindex].selector = 0; */
  /* ESI->EventInfoArray[thisindex].operand_index = -1; */

  /* ESI->sw_stop[hwindex]           = 0; */
  /* ESI->hw_start[hwindex]         = 0; */

  ESI->NumberOfEvents--;

  return(retval);
}

int _papi_hwi_read(hwd_context_t *context, EventSetInfo_t *ESI, u_long_long *values)
{
  register int i, j = 0;
  int retval;
  u_long_long *dp;

  retval = _papi_hwd_read(context, &ESI->machdep, &dp);
  if (retval != PAPI_OK)
    return(retval);

  for (i=0;i<EventInfoArrayLength(ESI);i++)
    {
#ifdef DEBUG
      DBG((stderr,"PAPI counter %d is at hardware index %d, %lld\n",i,ESI->EventInfoArray[i].hardware_index,dp[ESI->EventInfoArray[i].hardware_index]));
#endif
      values[j] = dp[ESI->EventInfoArray[i].hardware_index];

      /* Early exit! */
      
      if (++j == ESI->NumberOfEvents)
	return(PAPI_OK);
    }

  return(PAPI_EBUG);
}

int _papi_hwi_cleanup_eventset(EventSetInfo_t *ESI)
{
  int retval, i, tmp = EventInfoArrayLength(ESI);

  if (ESI->state & PAPI_MULTIPLEXING)
    {
      retval = MPX_cleanup(&ESI->multiplex);
      if (retval != PAPI_OK)
	return(retval);
    }
  
  for(i=(tmp-1);i>=0;i--) 
    {
      if (ESI->EventInfoArray[i].event_code != PAPI_NULL)
	{
	  retval = _papi_hwi_remove_event(ESI, ESI->EventInfoArray[i].event_code);
	  if (retval != PAPI_OK)
	    return(retval);
	}
    }

  return(PAPI_OK);
}

int _papi_hwi_convert_eventset_to_multiplex(EventSetInfo_t *ESI)
{
  EventInfo_t *tmp;
  int retval, i, j=0, *mpxlist = NULL;

  tmp = (EventInfo_t *)malloc(PAPI_MPX_DEF_DEG*sizeof(EventInfo_t));
  if (tmp == NULL)
    return(PAPI_ENOMEM);

  /* If there are any events in the EventSet, 
     convert them to multiplex events */

  if (ESI->NumberOfEvents)
    {
      mpxlist = (int *)malloc(sizeof(int)*ESI->NumberOfEvents);
      if (mpxlist == NULL)
	{
	  free(tmp);
	  return(PAPI_ENOMEM);
	}

      /* Build the args to MPX_add_events(). */

      /* Remember the EventInfoArray can be sparse
	 and the data can be non-contiguous */

      for (i=0;i<EventInfoArrayLength(ESI);i++)
	if (ESI->EventInfoArray[i].event_code != PAPI_NULL)
	  mpxlist[j++] = ESI->EventInfoArray[i].event_code;	      
      
      retval = MPX_add_events(&ESI->multiplex,mpxlist,j);
      if (retval != PAPI_OK)
	{
	  free(mpxlist);
	  free(tmp);
	  return(retval);
	}
    }
  
  /* Resize the EventInfo_t array */
  
  free(ESI->EventInfoArray);
  ESI->EventInfoArray = tmp;

  /* Update the state before initialization! */

  ESI->state |= PAPI_MULTIPLEXING;

  /* Initialize it */

  initialize_EventInfoArray(ESI);
  
  /* Copy only the relevant contents of EventInfoArray to
     this multiplexing eventset. This allows PAPI_list_events
     to work transparently and allows quick lookups of what's
     in this eventset without having to iterate through all
     it's 'sub-eventsets'. */

  if (mpxlist != NULL) {
    for (i=0;i<ESI->NumberOfEvents;i++)
	ESI->EventInfoArray[i].event_code = mpxlist[i];      
    free(mpxlist);
  }

  return(PAPI_OK);
}

int _papi_hwi_query(int preset_index, int *flags, char **note)
{ 
  if (_papi_hwd_preset_map[preset_index].number == 0)
    return(0);
  DBG((stderr,"preset_index: %d derived: %d\n", preset_index, _papi_hwd_preset_map[preset_index].derived));
  if (_papi_hwd_preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  DBG((stderr,"note: %s\n", _papi_hwd_preset_map[preset_index].note));
  if (_papi_hwd_preset_map[preset_index].note)
    *note = _papi_hwd_preset_map[preset_index].note;
  return(1);
}

void _papi_hwi_dummy_handler(int EventSet, int EventCode, int EventIndex,
                          long_long *values, int *threshold, void *context)
{
  /* This function is not used and shouldn't be called. */

  abort();
}
