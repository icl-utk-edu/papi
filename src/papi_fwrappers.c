#include <stdio.h>
#include <assert.h>
#include <string.h>
#if defined ( _CRAYT3E ) 
#include  <stdlib.h>
#include  <fortran.h>  
#endif
#include "papi.h"

/* Lets use defines to rename all the files */

#ifdef FORTRANUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##_##args
#elif FORTRANDOUBLEUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##__##args
#elif FORTRANCAPS
#define PAPI_FCALL(function,caps,args) void caps##args
#else
#define PAPI_FCALL(function,caps,args) void function##args
#endif

/* The Low Level Wrappers */

PAPI_FCALL(papif_accum,PAPIF_ACCUM,(int *EventSet, long long *values, int *check))
{
  *check = PAPI_accum(*EventSet, values);
}

PAPI_FCALL(papif_add_event,PAPIF_ADD_EVENT,(int *EventSet, int *Event, int *check))
{
  *check = PAPI_add_event(EventSet, *Event);
}

PAPI_FCALL(papif_add_events,PAPIF_ADD_EVENTS,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_add_events(EventSet, Events, *number);
}

PAPI_FCALL(papif_cleanup_eventset,PAPIF_CLEANUP_EVENTSET,(int *EventSet, int *check))
{
  *check = PAPI_cleanup_eventset(EventSet);
}

PAPI_FCALL(papif_library_init,PAPIF_LIBRARY_INIT,(int *check))
{
  int tmp;
  tmp = PAPI_library_init(PAPI_VER_CURRENT);
  if (tmp != PAPI_VER_CURRENT)
    *check = PAPI_EBUG;
  else
    *check = PAPI_VER_CURRENT;
}

/* This must be passed an EXTERNAL or INTRINISIC FUNCTION not a SUBROUTINE */

PAPI_FCALL(papif_thread_init,PAPIF_THREAD_INIT,(unsigned long int (*handle)(), int *flag, int *check))
{
  *check = PAPI_thread_init(handle, *flag);
}

PAPI_FCALL(papif_list_events,PAPIF_LIST_EVENTS,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_list_events(*EventSet, Events, number);
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_perror,PAPIF_PERROR,(int *code, _fcd destination, int *length, int *check))
#else
PAPI_FCALL(papif_perror,PAPIF_PERROR,(int *code, char *destination, int *length, int *check))
#endif
{
#if defined( _CRAYT3E )
  char tmp[PAPI_MAX_STR_LEN];
  *check = PAPI_perror(*code, tmp, *length);
  strncpy( _fcdtocp(destination), tmp, strlen(tmp));
#else
  *check = PAPI_perror(*code, destination, *length);
#endif
}

PAPI_FCALL(papif_query_event,PAPIF_QUERY_EVENT,(int *EventCode, int *check))
{
  *check = PAPI_query_event(*EventCode);
}
 
#if defined ( _CRAYT3E )
PAPI_FCALL(papif_event_code_to_name,PAPIF_EVENT_CODE_TO_NAME,(int *EventCode, _fcd out, int *check))
#else
PAPI_FCALL(papif_event_code_to_name,PAPIF_EVENT_CODE_TO_NAME,(int *EventCode, char *out, int *check))
#endif
{
#if defined( _CRAYT3E )
  char tmp[PAPI_MAX_STR_LEN];
  *check = PAPI_event_code_to_name(*EventCode, tmp);
  strncpy( _fcdtocp(out), tmp, strlen(tmp));
#else
  *check = PAPI_event_code_to_name(*EventCode, out);
#endif
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_event_name_to_code,PAPIF_EVENT_NAME_TO_CODE,(_fcd in, int *out, int *check))
#else
PAPI_FCALL(papif_event_name_to_code,PAPIF_EVENT_NAME_TO_CODE,(char *in, int *out, int *check))
#endif
{
#if defined( _CRAYT3E )
  char tmpin[PAPI_MAX_STR_LEN]; 
  int slen;
  
  slen = _fcdlen(in);
  strncpy( tmpin, _fcdtocp(in), slen );
  *check= PAPI_event_name_to_code(tmpin, out);
#else
  *check= PAPI_event_name_to_code(in, out);
#endif
}

PAPI_FCALL(papif_read,PAPIF_READ,(int *EventSet, long long *values, int *check))
{
  *check = PAPI_read(*EventSet, values);
}

PAPI_FCALL(papif_rem_event,PAPIF_REM_EVENT,(int *EventSet, int *Event, int *check))
{
  *check = PAPI_rem_event(EventSet, *Event);
}

PAPI_FCALL(papif_rem_events,PAPIF_REM_EVENTS,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_rem_events(EventSet, Events, *number);
}

PAPI_FCALL(papif_reset,PAPIF_RESET,(int *EventSet, int *check))
{
  *check = PAPI_reset(*EventSet);
}

PAPI_FCALL(papif_set_debug,PAPIF_SET_DEBUG,(int *debug, int *check))
{
  *check = PAPI_set_debug(*debug);
}

PAPI_FCALL(papif_set_domain,PAPIF_SET_DOMAIN,(int *domain, int *check))
{
  *check = PAPI_set_domain(*domain);
}

PAPI_FCALL(papif_set_granularity,PAPIF_SET_GRANULARITY,(int *granularity, int *check))
{
  *check = PAPI_set_granularity(*granularity);
}

PAPI_FCALL(papif_start,PAPIF_START,(int *EventSet, int *check))
{
  *check = PAPI_start(*EventSet);
}

PAPI_FCALL(papif_state,PAPIF_STATE,(int *EventSet, int *status, int *check))
{
  *check = PAPI_state(*EventSet, status);
}

PAPI_FCALL(papif_stop,PAPIF_STOP,(int *EventSet, long long *values, int *check))
{
  *check = PAPI_stop(*EventSet, values);
}

PAPI_FCALL(papif_write,PAPIF_WRITE,(int *EventSet, long long *values, int *check))
{
  *check = PAPI_write(*EventSet, values);
}

PAPI_FCALL(papif_shutdown,PAPIF_SHUTDOWN,(void))
{
  PAPI_shutdown();
}

#if defined ( _CRAYT3E )
PAPI_FCALL(papif_get_hardware_info,PAPIF_GET_HARDWARE_INFO,(int *ncpu, 
	   int *nnodes, int *totalcpus, int *vendor, _fcd vendor_string, 
	   int *model, _fcd model_string, float *revision, float *mhz))
#else
PAPI_FCALL(papif_get_hardware_info,PAPIF_GET_HARDWARE_INFO,(int *ncpu, 
	   int *nnodes, int *totalcpus, int *vendor, char *vendor_string, 
	   int *model, char *model_string, float *revision, float *mhz))
#endif
{
  const PAPI_hw_info_t *hwinfo;
  hwinfo = PAPI_get_hardware_info();
  if ( hwinfo == NULL ){
    *ncpu = 0;
  }
  else {
    *ncpu = hwinfo->ncpu;
    *nnodes = hwinfo->nnodes;
    *totalcpus = hwinfo->totalcpus;
    *vendor = hwinfo->vendor;
    *model = hwinfo->model;
#if defined ( _CRAYT3E )
    strncpy( _fcdtocp( vendor_string), hwinfo->vendor_string,
	     strlen( hwinfo->vendor_string) );
    strncpy( _fcdtocp( model_string), hwinfo->model_string,
	     strlen( hwinfo->model_string ) );
#else
    strcpy( vendor_string, hwinfo->vendor_string );
    strcpy( model_string, hwinfo->model_string );
#endif    
    *revision = hwinfo->revision;
    *mhz = hwinfo->mhz;
  }
  return;
}

PAPI_FCALL(papif_create_eventset,PAPIF_CREATE_EVENTSET,(int *EventSet, int *check))
{
  *check = PAPI_create_eventset(EventSet);
}

PAPI_FCALL(papif_destroy_eventset,PAPIF_DESTROY_EVENTSET,(int *EventSet, int *check))
{
  *check = PAPI_destroy_eventset(EventSet);
}

/* The High Level API Wrappers */

PAPI_FCALL(papif_num_counters,PAPIF_NUM_COUNTERS,(int *numevents))
{
  *numevents = PAPI_num_counters();
}

PAPI_FCALL(papif_start_counters,PAPIF_START_COUNTERS,(int *events, int *array_len, int *check))
{
  *check = PAPI_start_counters(events, *array_len);
}

PAPI_FCALL(papif_read_counters,PAPIF_READ_COUNTERS,(long long *values, int *array_len, int *check))
{
  *check = PAPI_read_counters(values, *array_len);
}

PAPI_FCALL(papif_stop_counters,PAPIF_STOP_COUNTERS,(long long *values, int *array_len, int *check))
{
  *check = PAPI_stop_counters(values, *array_len);
}

PAPI_FCALL(papif_get_real_usec,PAPIF_GET_REAL_USEC,( long long *time))
{
  *time = PAPI_get_real_usec();
}

PAPI_FCALL(papif_get_real_cyc,PAPIF_GET_REAL_CYC,(long long *real_cyc))
{
  *real_cyc = PAPI_get_real_cyc();
}

PAPI_FCALL(papif_get_virt_usec,PAPIF_GET_VIRT_USEC,( long long *time))
{
  *time = PAPI_get_virt_usec();
}

PAPI_FCALL(papif_get_virt_cyc,PAPIF_GET_VIRT_CYC,(long long *virt_cyc))
{
  *virt_cyc = PAPI_get_virt_cyc();
}
