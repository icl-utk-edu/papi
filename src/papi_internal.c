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
      for(j=0;j<_papi_hwi_system_info.num_cntrs;j++)
     	  ESI->EventInfoArray[i].pos[j] = -1;
      ESI->EventInfoArray[i].ops = NULL;
      ESI->EventInfoArray[i].derived = NOT_DERIVED;
    }
}


static void initialize_NativeInfoArray(EventSetInfo_t *ESI)
{
  int i;

  for (i = 0; i < MAX_COUNTERS; i++) {
 	ESI->NativeInfoArray[i].ni_index = -1;
 	ESI->NativeInfoArray[i].ni_position = -1;
 	ESI->NativeInfoArray[i].ni_owners = 0;
  }
  ESI->NativeCount = 0;
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
  initialize_NativeInfoArray(ESI);
  _papi_hwd_init_control_state(&ESI->machdep); /* this used to be init_config */

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
  
  for (k=0;k<limit;k++){
      if (ESI->EventInfoArray[k].event_code == EventCode)
		papi_return(PAPI_ECNFLCT);
      /*if ((ESI->EventInfoArray[k].event_code == PAPI_NULL) && (lowslot == PAPI_ECNFLCT))*/
	  if (ESI->EventInfoArray[k].event_code == PAPI_NULL){
		  lowslot = k;
		  break;
	  }
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


/* this function try to find out whether native event has already been mapped. 
     Success, return hwd_native_t array index
     Fail,    return -1;                                                             
*/
int _papi_hwi_add_native_precheck(EventSetInfo_t *ESI, int nix)
{
	int i;

	/* to find the native event from the native events list */
	for(i=0; i<ESI->NativeCount;i++){
		if(nix==ESI->NativeInfoArray[i].ni_index){
			ESI->NativeInfoArray[i].ni_owners++;
			DBG((stderr,"found native event already mapped: %s\n", _papi_hwd_native_code_to_name(nix & NATIVE_MASK)));
			return i;
		}
	}
	return -1;
}	  


/* this function is called after mapping is done
   refill info for every added events
 */
static void remap_event_position(EventSetInfo_t *ESI, int thisindex)
{
	EventInfo_t *out, *head;
	int i, j, k, n, preset_index, nix, total_events;

	head = ESI->EventInfoArray;
	out  = &head[thisindex];
	total_events=ESI->NumberOfEvents;

	j=0;
	for(i=0;i<=total_events;i++)
    {
      /* find the added event in EventInfoArray */
	  while(head[j].event_code==PAPI_NULL) 
	    j++;
	  /* fill in the new information */
	  if(head[j].event_code & PRESET_MASK)
      {
	    preset_index = head[j].event_code & PRESET_AND_MASK;
	    for(k=0;k<_papi_hwi_preset_map[preset_index].metric_count;k++)
        {
		  nix=_papi_hwi_preset_map[preset_index].natIndex[k];
		  for(n=0;n<ESI->NativeCount;n++)
          {
		    if(nix==ESI->NativeInfoArray[n].ni_index)
            {
		      head[j].pos[k]=ESI->NativeInfoArray[n].ni_position;
			  break;
			}
		  }
		}
		/*head[j].pos[k]=-1;*/
	  }
	  else
      {
	    nix = head[j].event_code & NATIVE_AND_MASK;
	    for(n=0;n<ESI->NativeCount;n++)
        {
		  if(nix==ESI->NativeInfoArray[n].ni_index)
          {
		    head[j].pos[0]=ESI->NativeInfoArray[n].ni_position;
            /*head[j].pos[1]=-1;*/
		    break;
		  }
		}
	  }  /* end of if */
	  j++;
    }  /* end of for loop */
}


static int add_native_fail_clean(EventSetInfo_t *ESI, int nix)
{
	int i;
	
	/* to find the native event from the native events list */
	for(i=0; i<ESI->NativeCount;i++){
		if(nix==ESI->NativeInfoArray[i].ni_index){
			ESI->NativeInfoArray[i].ni_owners--;
			/* to clean the entry in the nativeInfo array */
			if(ESI->NativeInfoArray[i].ni_owners==0){
				ESI->NativeInfoArray[i].ni_index = -1;
				ESI->NativeInfoArray[i].ni_position = -1;
				ESI->NativeCount --;
			}
			DBG((stderr,"add_events fail, and remove added native events of the event: %s\n", _papi_hwd_native_code_to_name(nix & NATIVE_MASK)));
			return i;
		}
	}
	return -1;
}	  

/* this function is called by _papi_hwi_add_event when adding native events 
nix: pointer to array of native event table indexes from the preset entry
size: number of native events to add
*/
static int add_native_events(EventSetInfo_t *ESI, int *nix, int size, EventInfo_t *out)
{
	int nidx, i, j, remap=0;
    int retval;
	
	/* Need to decide what needs to be preserved so we can roll back state
	   if the add event fails...
	*/
	
	/* if the native event is already mapped, fill in */
	for(i=0;i<size;i++){
		if((nidx=_papi_hwi_add_native_precheck(ESI, nix[i]))>=0){
			out->pos[i]=ESI->NativeInfoArray[nidx].ni_position;
		}
		else{
			/* all counters have been used, add_native fail */
			if(ESI->NativeCount==MAX_COUNTERS){
				/* to clean owners for previous added native events */
				for(j=0;j<i;j++){
					if((nidx=add_native_fail_clean(ESI, nix[j]))>=0){
						out->pos[j]=-1;
						continue;
					}
					DBG((stderr,"should not happen!\n"));
				}
				DBG((stderr,"counters are full!\n"));
				return -1;
			}
			/* there is an empty slot for the native event;
			   initialize the native index for the new added event */
			ESI->NativeInfoArray[ESI->NativeCount].ni_index=nix[i];
		    ESI->NativeInfoArray[ESI->NativeCount].ni_owners=1;
			ESI->NativeCount++;
			remap++; 
		}
	}
	
	/* if remap!=0, we need reallocate counters */
	if(remap){
		if(_papi_hwd_allocate_registers(ESI)){
			retval=_papi_hwd_update_control_state(&ESI->machdep, ESI->NativeInfoArray, ESI->NativeCount);
            if (retval != PAPI_OK)
              return(retval);

		    return 1;
		}
		else{
			for(i=0;i<size;i++){
				if((nidx=add_native_fail_clean(ESI, nix[i]))>=0){
					out->pos[j]=-1;
					continue;
				}
				DBG((stderr,"should not happen!\n"));
			}
			return -1;
		}
	}
	
	return 0;
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
			
			remap = add_native_events(ESI, _papi_hwi_preset_map[preset_index].natIndex, _papi_hwi_preset_map[preset_index].metric_count, &ESI->EventInfoArray[thisindex]);
			if (remap < 0)
				return(PAPI_ECNFLCT);
			else{
				/* Fill in the EventCode (machine independent) information */
			
				ESI->EventInfoArray[thisindex].event_code = EventCode; 
				ESI->EventInfoArray[thisindex].derived = _papi_hwi_preset_map[preset_index].derived; 
				ESI->EventInfoArray[thisindex].ops = _papi_hwi_preset_map[preset_index].operation; 
				if(remap)
					remap_event_position(ESI, thisindex);
			}
		}
		else if(EventCode & NATIVE_MASK)
		{
			int native_index = EventCode & NATIVE_AND_MASK;

			/* Check if it's within the valid range */
			
			if ((native_index < 0) || (native_index >= PAPI_MAX_NATIVE_EVENTS))
				return(PAPI_EINVAL);

			/* Check if native event exists */

			if (_papi_hwi_query_native_event(EventCode) != PAPI_OK)
				return(PAPI_ENOEVNT);

			/* Try to add the native. */
			
			remap = add_native_events(ESI, &native_index, 1, &ESI->EventInfoArray[thisindex]);
			if (remap < 0)
				return(PAPI_ECNFLCT);
			else{
				/* Fill in the EventCode (machine independent) information */
			
				ESI->EventInfoArray[thisindex].event_code = EventCode; 
				if(remap)
					remap_event_position(ESI, thisindex);
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
        ESI->EventInfoArray[thisindex].event_code = EventCode;  /* Relevant */
        ESI->EventInfoArray[thisindex].derived = NOT_DERIVED;

    }
	
	/* Bump the number of events */
	ESI->NumberOfEvents++;
			/*print_state(ESI);*/
	
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


int remove_native_events(EventSetInfo_t *ESI, int *nix, int size)
{
    hwd_control_state_t *this_state= &ESI->machdep;
    NativeInfo_t *native = ESI->NativeInfoArray;
    int i, j, zero=0, retval;

    /* Remove the references to this event from the native events:
       for all the metrics in this event,
	compare to each native event in this event set,
	and decrement owners if they match  */
    for(i=0;i<size;i++){
		for(j=0;j<ESI->NativeCount;j++){ 
			if(native[j].ni_index==nix[i]){
				native[j].ni_owners--;
				if(native[j].ni_owners==0){
					zero++;
				}
				break;
			}
		}
    }

    /* Remove any native events from the array if owners dropped to zero.
	The NativeInfoArray must be dense, with no empty slots, so if we
	remove an element, we must compact the list */
    for(i=0;i<ESI->NativeCount;i++){
		if(native[i].ni_index==-1)
			continue;
	
		if(native[i].ni_owners==0){
			int copy=0;
			for(j=ESI->NativeCount-1;j>i;j--){
				if(native[j].ni_index==-1 || native[j].ni_owners==0)
					continue;
				else{
					memcpy(native+i, native+j, sizeof(NativeInfo_t));
					memset(native+j, 0, sizeof(NativeInfo_t));
					native[j].ni_index=-1;
					native[j].ni_position=-1;
					copy++;
					break;
				}
			}
			if(copy==0){
				memset(native+i, 0, sizeof(NativeInfo_t));
				native[j].ni_index=-1;
				native[i].ni_position=-1;
			}
		}
    }

	/* to reset hwd_control_state values */
	ESI->NativeCount-=zero;

    /* If we removed any elements, 
	clear the now empty slots, reinitialize the index, and update the count.
	Then send the info down to the substrate to update the hwd control structure. */
    if (zero) {
      retval=_papi_hwd_update_control_state(this_state, native, ESI->NativeCount);
      if (retval != PAPI_OK)
        return(retval);
    }

    return(PAPI_OK);
}

int _papi_hwi_remove_event(EventSetInfo_t *ESI, int EventCode)
{
  int j = 0, retval, thisindex;
  EventInfo_t *array;

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
  /* Remove the events hardware dependent stuff from the EventSet */
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
	  
	  /* Remove the preset event. */
	  retval=remove_native_events(ESI, _papi_hwi_preset_map[preset_index].natIndex, _papi_hwi_preset_map[preset_index].metric_count);
      if (retval != PAPI_OK)
  	    return(retval);
      }
      else if(EventCode & NATIVE_MASK)
      {
	  int native_index = EventCode & NATIVE_AND_MASK;

	  /* Check if it's within the valid range */
	  if ((native_index < 0) || (native_index >= PAPI_MAX_NATIVE_EVENTS))
		  return(PAPI_EINVAL);

	  /* Remove the native event. */
	  retval=remove_native_events(ESI, &native_index, 1);
      if (retval != PAPI_OK)
  	    return(retval);
      }
      else
		return(PAPI_ENOEVNT);

      /* dereference a couple values for cleaner code */
      /*count = ESI->NumberOfEvents;*/
      array = ESI->EventInfoArray;

     /* Compact the Event Info Array list if it's not the last event */
     /* if (thisindex < (count - 1))
	  memcpy(&array[thisindex], &array[thisindex+1], sizeof(EventInfo_t) * (count - thisindex - 1));
*/
      /* clear the newly empty slot in the array */
      array[thisindex].event_code = PAPI_NULL;
      for(j=0;j<_papi_hwi_system_info.num_cntrs;j++)
		array[thisindex].pos[j] = -1;
      array[thisindex].ops = NULL;
      array[thisindex].derived = NOT_DERIVED;
  }
  ESI->NumberOfEvents--;
  /*print_state(ESI);*/

  return(PAPI_OK);
}

int _papi_hwi_read(hwd_context_t *context, EventSetInfo_t *ESI, long_long *values)
{
  int retval;
  long_long *dp;
/*
  int *pos, threshold, multiplier;
*/


/*
  pos = NULL;
  threshold = multiplier = 0;
  if ((ESI->state & PAPI_OVERFLOWING) && 
          (_papi_hwi_system_info.supports_hw_overflow)) 
  {
    pos = ESI->EventInfoArray[ESI->overflow.EventIndex].pos;
    threshold= ESI->overflow.threshold;
    multiplier = ESI->overflow.count;
  }  
  else 
    if ((ESI->state & PAPI_PROFILING) && 
          (_papi_hwi_system_info.supports_hw_profile)) 
    {
      pos = ESI->EventInfoArray[ESI->profile.EventIndex].pos;
      threshold = ESI->profile.threshold;
      multiplier =  ESI->profile.overflowcount;
    };
*/

  retval = _papi_hwd_read(context, &ESI->machdep, &dp);
  if (retval != PAPI_OK)
    return(retval);

  counter_read(ESI, dp, values);
  
  return(PAPI_OK);
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

static long_long handle_derived_add(int *position, long_long *from)
{
  int pos, i;
  long_long retval = 0;

  i=0;
  pos=position[i++];
  while (pos != -1 )
  {
    DBG((stderr,"Compound event, adding %lld to %lld\n",from[pos],retval));
    retval += from[pos];
    pos=position[i++];
  }
  return(retval);
}

static long_long handle_derived_subtract(int *position, long_long *from)
{
  int pos, i;
  long_long retval = from[position[0]];

  i=1;
  pos=position[i++];
  while (pos != -1)
  {
    DBG((stderr,"Compound event, subtracting pos=%d  %lld to %lld\n",pos, from[pos],retval));
    retval -= from[pos];
    pos=position[i++];
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

static long_long handle_derived_ps(int *position, long_long *from)
{
  return(units_per_second(from[position[1]],from[position[0]]));
}

static long_long handle_derived_add_ps(int *position, long_long *from)
{
  long_long tmp = handle_derived_add(position+1, from);
  return(units_per_second(tmp, from[position[0]]));
}

static long_long handle_derived(EventInfo_t *evi, long_long *from)
{
  switch (evi->derived)
  {
    case DERIVED_ADD: 
      return(handle_derived_add(evi->pos, from));
    case DERIVED_ADD_PS:
      return(handle_derived_add_ps(evi->pos, from));
    case DERIVED_SUB:
      return(handle_derived_subtract(evi->pos, from));
    case DERIVED_PS:
      return(handle_derived_ps(evi->pos, from));
    default:
      abort();
  }
}

static int counter_read(EventSetInfo_t *ESI, long_long *hw_counter, long_long *values)
{
  int i, j=0, index;

  /* This routine distributes hardware counters to software counters in the
     order that they were added. Note that the higher level
     EventInfoArray[i] entries may not be contiguous because the user
     has the right to remove an event.
     But if we do compaction after remove event, this function can be 
     changed.  
   */

  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
  {
    index = ESI->EventInfoArray[i].pos[0];
    if (index == -1)
      continue;

    DBG((stderr,"Event index %d, position is 0x%x\n",j,index));

    /* If this is not a derived event */

    if (ESI->EventInfoArray[i].derived == NOT_DERIVED)
    {
      DBG((stderr,"counter index is %d\n", index));
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

/*
void print_state(EventSetInfo_t *ESI)
{
  int i;
  
  fprintf(stderr,"\n\n-----------------------------------------\n");
  fprintf(stderr,"numEvent: %d    numNative: %d\n", ESI->NumberOfEvents, ESI->NativeCount);

  fprintf(stderr,"\nnative_event_name       ");
  for(i=0;i<MAX_COUNTERS;i++)
	  fprintf(stderr,"%15s",native_table[ESI->NativeInfoArray[i].ni_index].name);
  fprintf(stderr,"\n");

  fprintf(stderr,"native_event_selectors    ");
  for(i=0;i<MAX_COUNTERS;i++)
	  fprintf(stderr,"%15d",native_table[ESI->NativeInfoArray[i].ni_index].resources.selector);
  fprintf(stderr,"\n");

  fprintf(stderr,"native_event_position     ");
  for(i=0;i<MAX_COUNTERS;i++)
	  fprintf(stderr,"%15d",ESI->NativeInfoArray[i].ni_position);
  fprintf(stderr,"\n");

  fprintf(stderr,"counter_cmd               ");
  for(i=0;i<MAX_COUNTERS;i++)
	  fprintf(stderr,"%15d",(&(ESI->machdep))->counter_cmd.events[i]);
  fprintf(stderr,"\n");

  fprintf(stderr,"native links              ");
  for(i=0;i<MAX_COUNTERS;i++)
	  fprintf(stderr,"%15d",ESI->NativeInfoArray[i].ni_owners);
  fprintf(stderr,"\n");
  
}
*/


