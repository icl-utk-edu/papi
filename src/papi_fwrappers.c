#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "papi.h"

/* Lets use defines to rename all the files */

#ifdef FORTRANUNDERSCORE
#define PAPI_FCALL(function,args) void function##_##args
#elif FORTRANDOUBLEUNDERSCORE
#define PAPI_FCALL(function,args) void function##__##args
#else
#define PAPI_FCALL(function,args) void function##args
#endif

/* The Low Level Wrappers */

PAPI_FCALL(papif_accum,(int *EventSet, long long *values, int *check))
{
  *check = PAPI_accum(*EventSet, values);
}

PAPI_FCALL(papif_add_event,(int *EventSet, int *Event, int *check))
{
  *check = PAPI_add_event(EventSet, *Event);
}

PAPI_FCALL(papif_add_events,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_add_events(EventSet, Events, *number);
}

PAPI_FCALL(papif_cleanup_eventset,(int *EventSet, int *check))
{
  *check = PAPI_cleanup_eventset(EventSet);
}

PAPI_FCALL(papif_library_init,(int *check))
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

PAPI_FCALL(papif_thread_init,(int *handle, int *flag, int *check))
{
  if (*handle == 0)
    handle = NULL;
  assert(sizeof(int *) <= sizeof(int));
  *check = PAPI_thread_init((void **)handle, *flag);
}

PAPI_FCALL(papif_list_events,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_list_events(*EventSet, Events, number);
}

PAPI_FCALL(papif_perror,(int *code, char *destination, int *length, int *check))
{
  *check = PAPI_perror(*code, destination, *length);
}

PAPI_FCALL(papif_query_event,(int *EventCode, int *check))
{
  *check = PAPI_query_event(*EventCode);
}
 
PAPI_FCALL(papif_event_code_to_name,(int *EventCode, char *out, int *check))
{
  *check = PAPI_event_code_to_name(*EventCode, out);
}

PAPI_FCALL(papif_event_name_to_code,(char *in, int *out, int *check))
{
  *check= PAPI_event_name_to_code(in, out);
}

PAPI_FCALL(papif_read,(int *EventSet, long long *values, int *check))
{
  *check = PAPI_read(*EventSet, values);
}

PAPI_FCALL(papif_rem_event,(int *EventSet, int *Event, int *check))
{
  *check = PAPI_rem_event(EventSet, *Event);
}

PAPI_FCALL(papif_rem_events,(int *EventSet, int *Events, int *number, int *check))
{
  *check = PAPI_rem_events(EventSet, Events, *number);
}

PAPI_FCALL(papif_reset,(int *EventSet, int *check))
{
  *check = PAPI_reset(*EventSet);
}

PAPI_FCALL(papif_set_debug,(int *debug, int *check))
{
  *check = PAPI_set_debug(*debug);
}

PAPI_FCALL(papif_set_domain,(int *domain, int *check))
{
  *check = PAPI_set_domain(*domain);
}

PAPI_FCALL(papif_set_granularity,(int *granularity, int *check))
{
  *check = PAPI_set_granularity(*granularity);
}

PAPI_FCALL(papif_start,(int *EventSet, int *check))
{
  *check = PAPI_start(*EventSet);
}

PAPI_FCALL(papif_state,(int *EventSet, int *status, int *check))
{
  *check = PAPI_state(*EventSet, status);
}

PAPI_FCALL(papif_stop,(int *EventSet, long long *values, int *check))
{
  *check = PAPI_stop(*EventSet, values);
}

PAPI_FCALL(papif_write,(int *EventSet, long long *values, int *check))
{
  *check = PAPI_write(*EventSet, values);
}

PAPI_FCALL(papif_shutdown,(void))
{
  PAPI_shutdown();
}

PAPI_FCALL(papif_get_hardware_info,(int *ncpu, int *nnodes, int *totalcpus, int *vendor,
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

PAPI_FCALL(papif_create_eventset,(int *EventSet, int *check))
{
  *check = PAPI_create_eventset(EventSet);
}

PAPI_FCALL(papif_create_eventset_r,(int *EventSet, int *handle, int *check))
{
  assert(sizeof(int *) <= sizeof(int));
  *check = PAPI_create_eventset_r(EventSet, (void **)handle);
}

PAPI_FCALL(papif_destroy_eventset,(int *EventSet, int *check))
{
  *check = PAPI_destroy_eventset(EventSet);
}

/* The High Level API Wrappers */

PAPI_FCALL(papif_num_counters,(int *numevents))
{
  *numevents = PAPI_num_counters();
}

PAPI_FCALL(papif_start_counters,(int *events, int *array_len, int *check))
{
  *check = PAPI_start_counters(events, *array_len);
}

PAPI_FCALL(papif_read_counters,(long long *values, int *array_len, int *check))
{
  *check = PAPI_read_counters(values, *array_len);
}

PAPI_FCALL(papif_stop_counters,(long long *values, int *array_len, int *check))
{
  *check = PAPI_stop_counters(values, *array_len);
}

PAPI_FCALL(papif_get_real_usec,( long long *time))
{
  *time = PAPI_get_real_usec();
}

PAPI_FCALL(papif_get_real_cyc,(long long *real_cyc))
{
  *real_cyc = PAPI_get_real_cyc();
}

PAPI_FCALL(papif_get_virt_usec,( long long *time))
{
  *time = PAPI_get_virt_usec();
}

PAPI_FCALL(papif_get_virt_cyc,(long long *virt_cyc))
{
  *virt_cyc = PAPI_get_virt_cyc();
}

/* this routine makes a mess of fortran strings. Grabbed from MPI_Connect */

int fstr2c (char *in, char *out, int outmaxlen)
{
  int i, j;

  j = strlen (in);
  if (j >= outmaxlen ) j = outmaxlen -1; /* truncate with space for null */

  for(i=0;i<j;i++)
    if (in[i]<' ' || in[i]>122) break;

  strncpy (out, in, i);
  out[i] = '\0';
  return (i);
}
