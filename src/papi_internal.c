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
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"

#include "papiStrings.h"


/********************/
/* BEGIN PROTOTYPES */
/********************/

/* Defined in this file */
static int default_error_handler(int errorCode);
static int counter_reorder(EventSetInfo_t *ESI, long_long *hw_counter, long_long *events);

extern unsigned long int (*_papi_hwi_thread_id_fn)(void);


/********************/
/*  END PROTOTYPES  */
/********************/

/********************/
/*  BEGIN GLOBALS   */ 
/********************/

/* Defined by the substrate */
extern hwi_preset_t _papi_hwi_preset_map[];

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
  int i, j, limit = EventInfoArrayLength(ESI);

  for (i=0;i<limit;i++)
    {
      ESI->EventInfoArray[i].ESIhead = ESI; /* always points to EventSetInfo_t *ESI */
      ESI->EventInfoArray[i].event_code = PAPI_NULL;
      ESI->EventInfoArray[i].counter_index = -1;
      for(j=0;j<_papi_hwi_system_info.num_cntrs;j++)
     	  ESI->EventInfoArray[i].pos[j] = -1;
      ESI->EventInfoArray[i].ops = NULL;
      ESI->EventInfoArray[i].hwd_selector = 0;
      ESI->EventInfoArray[i].derived = NOT_DERIVED;
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

/* this function is called after mapping is done
    if(remap) refill info for every added events
	else     fill current EventInfo_t *out
 */
void _papi_hwi_allocate_after(hwd_control_state_t *tmp_state, EventInfo_t *out, int remap)
{
	EventInfo_t *head;
	int i, j, k, n, preset_index, nix, total_events;

	head=out->ESIhead->EventInfoArray;
	total_events=out->ESIhead->NumberOfEvents;

	if(!remap){
		out->counter_index = out->pos[0];
		i=0;
		while(out->pos[i]>=0 && i<MAX_COUNTERS){
			out->hwd_selector |= 1<<out->pos[i];
			i++;
		}
		return;
	}
	
	j=0;
	for(i=0;i<=total_events;i++){
		while(head[j].event_code==PAPI_NULL) /* find the added event in EventInfoArray */
			j++;
		/* fill in the new information */
		head[j].hwd_selector=0;
		if(head[j].event_code & PRESET_MASK){
			preset_index = head[j].event_code & PRESET_AND_MASK;
			for(k=0;k<_papi_hwi_preset_map[preset_index].metric_count;k++){
				nix=_papi_hwi_preset_map[preset_index].natIndex[k];
				for(n=0;n<tmp_state->native_idx;n++){
					if(nix==tmp_state->native[n].index){
						head[j].pos[k]=tmp_state->native[n].position;
						head[j].hwd_selector |= 1<<tmp_state->native[n].position;
						break;
					}
				}
			}
			head[j].counter_index=head[j].pos[0];
		}
		else{
			nix = head[j].event_code & NATIVE_AND_MASK;
			for(n=0;n<tmp_state->native_idx;n++){
				if(nix==tmp_state->native[n].index){
					head[j].pos[0]=tmp_state->native[n].position;
					head[j].hwd_selector |= 1<<tmp_state->native[n].position;
					head[j].counter_index=tmp_state->native[n].position;
					break;
				}
				
			}
		}
		j++;
	}
}

int _papi_hwi_add_event(EventSetInfo_t *ESI, int EventCode)
{
	int thisindex, remap, retval=PAPI_OK;
	
	/* Make sure the event is not present and get the next
	free slot. */
	
	thisindex = get_free_EventCodeIndex(ESI,EventCode);
	if (thisindex < PAPI_OK)
		return(thisindex);
	
		/* If it is a MPX EventSet, add it to the multiplex data structure and
	this threads multiplex list */
	
	if (!(ESI->state & PAPI_MULTIPLEXING))
    {
		
		if (EventCode & PRESET_MASK)
		{
			int preset_index = EventCode & PRESET_AND_MASK;
			
			/* Check if it's within the valid range */
			
			if ((preset_index < 0) || (preset_index >= PAPI_MAX_PRESET_EVENTS))
				return(PAPI_EINVAL);
			
			/* Check if event exists */
			
			if (!_papi_hwi_presets[preset_index].avail)
				return(PAPI_ENOEVNT);
			
			/* Try to add the preset. */
			
			remap = _papi_hwd_add_event(&ESI->machdep, _papi_hwi_preset_map[preset_index].natIndex, _papi_hwi_preset_map[preset_index].metric_count, &ESI->EventInfoArray[thisindex]);
			if (remap < 0)
				return(PAPI_ECNFLCT);
			else{
				/* Fill in the EventCode (machine independent) information */
			
				ESI->EventInfoArray[thisindex].event_code = EventCode; 
				ESI->EventInfoArray[thisindex].derived = _papi_hwi_preset_map[preset_index].derived; 
				ESI->EventInfoArray[thisindex].ops = _papi_hwi_preset_map[preset_index].operation; 
				_papi_hwi_allocate_after(&ESI->machdep, &ESI->EventInfoArray[thisindex], remap);
			}
		}
		else if(EventCode & NATIVE_MASK)
		{
			int native_index = EventCode & NATIVE_AND_MASK;

			/* Check if it's within the valid range */
			
			if ((native_index < 0) || (native_index >= PAPI_MAX_NATIVE_EVENTS))
				return(PAPI_EINVAL);

			/* Try to add the native. */
			
			remap = _papi_hwd_add_event(&ESI->machdep, &native_index, 1, &ESI->EventInfoArray[thisindex]);
			if (remap < 0)
				return(PAPI_ECNFLCT);
			else{
				/* Fill in the EventCode (machine independent) information */
			
				ESI->EventInfoArray[thisindex].event_code = EventCode; 
				_papi_hwi_allocate_after(&ESI->machdep, &ESI->EventInfoArray[thisindex], remap);
			}
		}
		else
		{
			/* not Native and Preset events */
			
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
  int j = 0, retval, thisindex;

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
		if (EventCode & PRESET_MASK)
		{
			int preset_index = EventCode & PRESET_AND_MASK;
			
			/* Check if it's within the valid range */
			
			if ((preset_index < 0) || (preset_index >= PAPI_MAX_PRESET_EVENTS))
				return(PAPI_EINVAL);
			
			/* Check if event exists */
			
			if (!_papi_hwi_presets[preset_index].avail)
				return(PAPI_ENOEVNT);
			
			/* Try to remove the preset. */
			
			_papi_hwd_remove_event(&ESI->machdep, _papi_hwi_preset_map[preset_index].natIndex, _papi_hwi_preset_map[preset_index].metric_count);
		}
		else if(EventCode & NATIVE_MASK)
		{
			int native_index = EventCode & NATIVE_AND_MASK;

			/* Check if it's within the valid range */
			
			if ((native_index < 0) || (native_index >= PAPI_MAX_NATIVE_EVENTS))
				return(PAPI_EINVAL);

			/* Try to add the native. */
			
			_papi_hwd_remove_event(&ESI->machdep, &native_index, 1);
		}
		else
			return(PAPI_ENOEVNT);

		/* clean the EventCode (machine independent) information */

		ESI->EventInfoArray[thisindex].event_code = PAPI_NULL;
		ESI->EventInfoArray[thisindex].counter_index = -1;
		for(j=0;j<_papi_hwi_system_info.num_cntrs;j++)
     		ESI->EventInfoArray[thisindex].pos[j] = -1;
		ESI->EventInfoArray[thisindex].ops = NULL;
		ESI->EventInfoArray[thisindex].hwd_selector = 0;
		ESI->EventInfoArray[thisindex].derived = NOT_DERIVED;
	
    }


  /* Move the counter_index's around. */

 /* for (i=0;i<EventInfoArrayLength(ESI);i++)
    {
      if (ESI->EventInfoArray[i].counter_index < ESI->EventInfoArray[thisindex].counter_index)
	;
      else if (ESI->EventInfoArray[i].counter_index == ESI->EventInfoArray[thisindex].counter_index)
	{
	  ESI->EventInfoArray[i].event_code = PAPI_NULL;
	  ESI->EventInfoArray[i].counter_index = -1;
	}
      else
	{
	  ESI->EventInfoArray[i].counter_index = -1;
	}

      if (++j == ESI->NumberOfEvents)
	break;
    }
	*/
  /* ESI->EventInfoArray[thisindex].derived = NOT_DERIVED; */
  /* ESI->EventInfoArray[thisindex].selector = 0; */
  /* ESI->EventInfoArray[thisindex].operand_index = -1; */

  /* ESI->sw_stop[hwindex]           = 0; */
  /* ESI->hw_start[hwindex]         = 0; */

  ESI->NumberOfEvents--;

  return(PAPI_OK);
}

int _papi_hwi_read(hwd_context_t *context, EventSetInfo_t *ESI, long_long *values)
{
  register int i;
  int retval, selector;
  long_long *dp;

  retval = _papi_hwd_read(context, &ESI->machdep, &dp);
  if (retval != PAPI_OK)
    return(retval);

  if ((ESI->state & PAPI_OVERFLOWING) && 
          (_papi_hwi_system_info.supports_hw_overflow)) 
  {
    selector = ESI->EventInfoArray[ESI->overflow.EventIndex].hwd_selector;
    while ((i=ffs(selector)))
    {
      dp[i-1]+= ESI->overflow.threshold;
      selector ^= 1<< (i-1);
    }
  }  
  if ((ESI->state & PAPI_PROFILING) && 
          (_papi_hwi_system_info.supports_hw_profile)) 
  {
    selector = ESI->EventInfoArray[ESI->profile.EventIndex].hwd_selector;
    while ((i=ffs(selector)))
    {
      dp[i-1]+= ESI->profile.threshold;
      selector ^= 1<< (i-1);
    }
    selector = ESI->EventInfoArray[ESI->profile.EventIndex].hwd_selector;
    i = ffs(selector);
    if ( i ) 
      dp[i-1] += ESI->profile.threshold * ESI->profile.overflowcount;
  }  

  retval = counter_reorder(ESI, dp, values);
  if (retval != PAPI_OK)
    return(retval);

  return(PAPI_OK);
#if 0
  for (i=0;i<EventInfoArrayLength(ESI);i++)
    {
#ifdef DEBUG
      DBG((stderr,"PAPI counter %d is at hardware index %d, %lld\n",i,ESI->EventInfoArray[i].counter_index,dp[ESI->EventInfoArray[i].counter_index]));
#endif
      values[j] = dp[ESI->EventInfoArray[i].counter_index];

      /* Early exit! */
      
      if (++j == ESI->NumberOfEvents)
	return(PAPI_OK);
    }
  return(PAPI_EBUG);
#endif
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

#if 0
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
#endif

/* Machine info struct initialization using defaults */
/* See _papi_mdi definition in papi_internal.h       */
int _papi_hwi_mdi_init() {
   _papi_hwi_system_info.substrate[0]           = '\0'; /* Name of the substrate we're using */
   _papi_hwi_system_info.version                = 1.0; /* version */
   _papi_hwi_system_info.pid                    = 0;   /* Process identifier */

  /* The PAPI_hw_info_t struct defined in papi.h */
   _papi_hwi_system_info.hw_info.ncpu               = -1;    /* ncpu */
   _papi_hwi_system_info.hw_info.nnodes             =  1;    /* nnodes */
   _papi_hwi_system_info.hw_info.totalcpus          = -1;    /* totalcpus */
   _papi_hwi_system_info.hw_info.vendor             = -1;    /* vendor */
   _papi_hwi_system_info.hw_info.vendor_string[0]   = '\0';  /* vendor_string */
   _papi_hwi_system_info.hw_info.model              = -1;    /* model */
   _papi_hwi_system_info.hw_info.model_string[0]    = '\0';  /* model_string */
   _papi_hwi_system_info.hw_info.revision           = 0.0;   /* revision */
   _papi_hwi_system_info.hw_info.mhz                = 0.0;   /* mhz */
   _papi_hwi_system_info.hw_info.max_native_events  = PAPI_MAX_NATIVE_EVENTS;

  /* The PAPI_exe_info_t struct defined in papi.h */
   memset(&_papi_hwi_system_info.exe_info, sizeof(PAPI_exe_info_t), 0);

  /* The PAPI_mem_info_t struct defined in papi.h */
    memset(&_papi_hwi_system_info.mem_info, sizeof(PAPI_mem_info_t), 0);

  /* The PAPI_shlib_info_t struct defined in papi.h */
    memset(&_papi_hwi_system_info.shlib_info, sizeof(PAPI_shlib_info_t), 0);
   _papi_hwi_system_info.shlib_info.map = (PAPI_address_map_t *)malloc(sizeof(PAPI_address_map_t));

  /* The following variables define the length of the arrays in the
     EventSetInfo_t structure. Each array is of length num_gp_cntrs +
     num_sp_cntrs * sizeof(long_long) */
   _papi_hwi_system_info.num_cntrs                      = -1;
   _papi_hwi_system_info.num_gp_cntrs                   = -1;
   _papi_hwi_system_info.grouped_counters               = -1;
   _papi_hwi_system_info.num_sp_cntrs                   = -1;
   _papi_hwi_system_info.total_presets                  = -1;
   _papi_hwi_system_info.total_events                   = -1;
   _papi_hwi_system_info.default_domain                 = PAPI_DOM_USER;
   _papi_hwi_system_info.default_granularity            = PAPI_GRN_THR;

  /* Public feature flags */
   _papi_hwi_system_info.supports_program               = 0;
   _papi_hwi_system_info.supports_write                 = 0;
   _papi_hwi_system_info.supports_hw_overflow           = 0;
   _papi_hwi_system_info.supports_hw_profile            = 0;
   _papi_hwi_system_info.supports_64bit_counters        = 0;
   _papi_hwi_system_info.supports_inheritance           = 0;
   _papi_hwi_system_info.supports_attach                = 0;
   _papi_hwi_system_info.supports_real_usec             = 0;
   _papi_hwi_system_info.supports_real_cyc              = 0;
   _papi_hwi_system_info.supports_virt_usec             = 0;
   _papi_hwi_system_info.supports_virt_cyc              = 0;

  /* Does the read call from the kernel reset the counters? */
   _papi_hwi_system_info.supports_read_reset            = 0;  /* Private flag */

  /* Size of the substrate's control struct in bytes */
   _papi_hwi_system_info.size_machdep = sizeof(hwd_control_state_t);

  /* Global struct to maintain EventSet mapping */
   _papi_hwi_system_info.global_eventset_map.dataSlotArray   = NULL;
   _papi_hwi_system_info.global_eventset_map.totalSlots      = 0;
   _papi_hwi_system_info.global_eventset_map.availSlots      = 0;
   _papi_hwi_system_info.global_eventset_map.fullSlots       = 0;
   _papi_hwi_system_info.global_eventset_map.lowestEmptySlot = 0;

  return(PAPI_OK);
}

void _papi_hwi_dummy_handler(int EventSet, int EventCode, int EventIndex,
                          long_long *values, int *threshold, void *context)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

static long_long handle_derived_add(int selector, long_long *from)
{
  int pos;
  long_long retval = 0;

  while ((pos = ffs(selector)))
    {
      DBG((stderr,"Compound event, adding %lld to %lld\n",from[pos-1],retval));
      retval += from[pos-1];
      selector ^= 1 << (pos-1);
    }
  return(retval);
}

static long_long handle_derived_subtract(int counter_index, int selector, long_long *from)
{
  int pos;
  long_long retval = from[counter_index];

  DBG((stderr,"counter_index: %d   selector: 0x%x\n",counter_index,selector));
  selector = selector ^ (1 << counter_index);
  while ((pos = ffs(selector)))
    {
      DBG((stderr,"Compound event, subtracting pos=%d  %lld to %lld\n",pos, from[pos-1],retval));
      retval -= from[pos-1];
      selector ^= 1 << (pos-1);
    }
  return(retval);
}

static long_long units_per_second(long_long units, long_long cycles)
{
  float tmp;

  tmp = (float)units * _papi_hwi_system_info.hw_info.mhz * 1000000.0;
  tmp = tmp / (float) cycles;
  return((u_long_long)tmp);
}

static long_long handle_derived_ps(int counter_index, int selector, long_long *from)
{
  int pos;

  pos = ffs(selector ^ (1 << counter_index)) - 1;
  assert(pos >= 0);

  return(units_per_second(from[pos],from[counter_index]));
}

static long_long handle_derived_add_ps(int counter_index, int selector, long_long *from)
{
  int add_selector = selector ^ (1 << counter_index);
  long_long tmp = handle_derived_add(add_selector, from);
  return(units_per_second(tmp, from[counter_index]));
}

static long_long handle_derived(EventInfo_t *evi, long_long *from)
{
  switch (evi->derived)
  {
    case DERIVED_ADD: 
      return(handle_derived_add(evi->hwd_selector, from));
    case DERIVED_ADD_PS:
      return(handle_derived_add_ps(evi->counter_index, evi->hwd_selector, from));
    case DERIVED_SUB:
      return(handle_derived_subtract(evi->counter_index, evi->hwd_selector, from));
    case DERIVED_PS:
      return(handle_derived_ps(evi->counter_index, evi->hwd_selector, from));
    default:
      abort();
  }
}

static int counter_reorder(EventSetInfo_t *ESI, long_long *hw_counter, long_long *values)
{
  int i, j=0, selector, index;

  /* This routine distributes hardware counters to software counters in the
     order that they were added. Note that the higher level
     EventInfoArray[i] entries may not be contiguous because the user
     has the right to remove an event.
     But if we do compaction after remove event, this function can be 
     changed.  ----Min
   */

  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
  {
    selector = ESI->EventInfoArray[i].hwd_selector;
    if (selector == 0)
      continue;
    index = ESI->EventInfoArray[i].counter_index;

    DBG((stderr,"Event index %d, selector is 0x%x\n",j,selector));

    /* If this is not a derived event */

    if (ESI->EventInfoArray[i].derived == NOT_DERIVED)
    {
      DBG((stderr,"counter_index is %d\n", index));
      values[j] = hw_counter[index];
    }
    else /* If this is a derived event */
      values[j]= handle_derived(&ESI->EventInfoArray[i], hw_counter);

    DBG((stderr, "read value is =%lld \n", values[j]));
    /* Early exit! */
    if (++j == ESI->NumberOfEvents)
      break;
  }
  return(PAPI_OK);
}
