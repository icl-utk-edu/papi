#include <stdio.h>
#include <assert.h>
#include <string.h>
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

/* Rule for Fortran is, if *handle == 0, then no threads. */
/* This *may* be a problem for 64 bit machines where 
   sizeof(int) != sizeof(int *) */

PAPI_FCALL(papif_thread_init,PAPIF_THREAD_INIT,(int *handle, int *flag, int *check))
{
  if (*handle == 0)
    handle = NULL;
  assert(sizeof(int *) <= sizeof(int));
  *check = PAPI_thread_init((void **)handle, *flag);
}

PAPI_FCALL(papif_list_events,PAPIF_LIST_EVENTS,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_list_events(*EventSet, Events, number);
}

PAPI_FCALL(papif_perror,PAPIF_PERROR,(int *code, char *destination, int *length, int *check))
{
  *check = PAPI_perror(*code, destination, *length);
}

PAPI_FCALL(papif_query_event,PAPIF_QUERY_EVENT,(int *EventCode, int *check))
{
  *check = PAPI_query_event(*EventCode);
}
 
PAPI_FCALL(papif_event_code_to_name,PAPIF_EVENT_CODE_TO_NAME,(int *EventCode, char *out, int *check))
{
  *check = PAPI_event_code_to_name(*EventCode, out);
}

PAPI_FCALL(papif_event_name_to_code,PAPIF_EVENT_NAME_TO_CODE,(char *in, int *out, int *check))
{
  *check= PAPI_event_name_to_code(in, out);
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

PAPI_FCALL(papif_get_hardware_info,PAPIF_GET_HARDWARE_INFO,(int *ncpu, int *nnodes, int *totalcpus, int *vendor,
				    char *vendor_string, int *model, char *model_string, float *revision,
				    float *mhz))
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
    strcpy( vendor_string, hwinfo->vendor_string );
    *model = hwinfo->model;
    strcpy( model_string, hwinfo->model_string );
    *revision = hwinfo->revision;
    *mhz = hwinfo->mhz;
  }
  return;
}

PAPI_FCALL(papif_create_eventset,PAPIF_CREATE_EVENTSET,(int *EventSet, int *check))
{
  *check = PAPI_create_eventset(EventSet);
}

PAPI_FCALL(papif_create_eventset_r,PAPIF_CREATE_EVENTSET_R,(int *EventSet, int *handle, int *check))
{
  assert(sizeof(int *) <= sizeof(int));
  *check = PAPI_create_eventset_r(EventSet, (void **)handle);
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
