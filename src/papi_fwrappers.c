#include "papi.h"
#include <stdio.h>

/* Lets use defines to rename all the files */
#ifdef FORTRANUNDERSCORE
#define PAPIf_accum papif_accum_
#define PAPIf_add_event papif_add_event_
#define PAPIf_add_events papif_add_events_
#define PAPIf_cleanup papif_cleanup_
#define PAPIf_list_events papif_list_events_
#define PAPIf_perror papif_perror_
#define PAPIf_query_event papif_query_event_
#define PAPIf_event_name_to_code papif_event_name_to_code_
#define PAPIf_read papif_read_
#define PAPIf_rem_event papif_rem_event_
#define PAPIf_rem_events papif_rem_events_
#define PAPIf_reset papif_reset_
#define PAPIf_restore papif_restore_
#define PAPIf_save papif_save_
#define PAPIf_set_domain papif_set_domain_
#define PAPIf_set_granularity papif_set_granularity_
#define PAPIf_start papif_start_
#define PAPIf_state papif_state_
#define PAPIf_stop papif_stop_
#define PAPIf_write papif_write_
#define PAPIf_shutdown papif_shutdown_
#define PAPIf_get_hardware_info papif_get_hardware_info_
#define PAPIf_create_eventset papif_create_eventset_
#define PAPIf_destroy_eventset papif_destroy_eventset_
#define PAPIf_num_events papif_num_events_
#define PAPIf_start_counters papif_start_counters_
#define PAPIf_read_counters papif_read_counters_
#define PAPIf_stop_counters papif_stop_counters_
#define PAPIf_get_real_usec papif_get_real_usec_
#define PAPIf_get_real_cyc papif_get_real_cyc_

#elif FORTRANDOUBLEUNDERSCORE
#define PAPIf_accum papif_accum__
#define PAPIf_add_event papif_add_event__
#define PAPIf_add_events papif_add_events__
#define PAPIf_cleanup papif_cleanup__
#define PAPIf_init papif_init__
#define PAPIf_list_events papif_list_events__
#define PAPIf_perror papif_perror__
#define PAPIf_query_event papif_query_event__
#define PAPIf_event_name_to_code papif_event_name_to_code__
#define PAPIf_read papif_read__
#define PAPIf_rem_event papif_rem_event__
#define PAPIf_rem_events papif_rem_events__
#define PAPIf_reset papif_reset__
#define PAPIf_restore papif_restore__
#define PAPIf_save papif_save__
#define PAPIf_set_domain papif_set_domain__
#define PAPIf_set_granularity papif_set_granularity__
#define PAPIf_start papif_start__
#define PAPIf_state papif_state__
#define PAPIf_stop papif_stop__
#define PAPIf_write papif_write__
#define PAPIf_shutdown papif_shutdown__
#define PAPIf_get_hardware_info papif_get_hardware_info__
#define PAPIf_create_eventset papif_create_eventset__
#define PAPIf_destroy_eventset papif_destroy_eventset__
#define PAPIf_num_events papif_num_events__
#define PAPIf_start_counters papif_start_counters__
#define PAPIf_read_counters papif_read_counters__
#define PAPIf_stop_counters papif_stop_counters__
#define PAPIf_get_real_usec papif_get_real_usec__
#define PAPIf_get_real_cyc papif_get_real_cyc__

#elif FORTRANALLCAPS
#define PAPIf_accum PAPIf_ACCUM
#define PAPIf_add_event PAPIf_ADD_EVENT
#define PAPIf_add_events PAPIf_ADD_EVENTS
#define PAPIf_cleanup PAPIf_CLEANUP
#define PAPIf_init PAPIf_INIT
#define PAPIf_list_events PAPIf_LIST_EVENTS
#define PAPIf_perror PAPIf_PERROR
#define PAPIf_query_event PAPIf_QUERY_EVENT
#define PAPIf_event_name_to_code PAPIf_EVENT_NAME_TO_CODE
#define PAPIf_read PAPIf_READ
#define PAPIf_rem_event PAPIf_REM_EVENT
#define PAPIf_rem_events PAPIf_REM_EVENTS
#define PAPIf_reset PAPIf_RESET
#define PAPIf_restore PAPIf_RESTORE
#define PAPIf_save PAPIf_SAVE
#define PAPIf_set_domain PAPIf_SET_DOMAIN
#define PAPIf_set_granularity PAPIf_SET_GRANULARITY
#define PAPIf_start PAPIf_START
#define PAPIf_state PAPIf_STATE
#define PAPIf_stop PAPIf_STOP
#define PAPIf_write PAPIf_WRITE
#define PAPIf_shutdown PAPIf_SHUTDOWN
#define PAPIf_get_hardware_info PAPIf_GET_HARDWARE_INFO
#define PAPIf_create_eventset PAPIf_CREATE_EVENTSET
#define PAPIf_destroy_eventset PAPIf_DESTROY_EVENTSET
#define PAPIf_num_events PAPIf_NUM_EVENTS
#define PAPIf_start_counters PAPIf_START_COUNTERS
#define PAPIf_read_counters PAPIf_READ_COUNTERS
#define PAPIf_stop_counters PAPIf_STOP_COUNTERS
#define PAPIf_get_real_usec PAPIf_GET_REAL_USEC
#define PAPIf_get_real_cyc PAPIf_GET_REAL_CYC
#endif


/* Fortran wrapper stuff */
#ifndef FOREVENTS
#define FOREVENTS 1
static int fortran_events[64] =  { 
0x80000000, 0x80000001, 0x80000002, 0x80000003, 0x80000004, 0x80000005,
0x80000006, 0x80000007, 0x80000008, 0x80000009, 0x8000000A, 0x8000000B,
0x8000000C, 0x8000000D, 0x8000000E, 0x8000000F, 0x80000010, 0x80000011,
0x80000012, 0x80000013, 0x80000014, 0x80000015, 0x80000016, 0x80000017,
0x80000018, 0x80000019, 0x8000001A, 0x8000001B, 0x8000001C, 0x8000001D,
0x8000001E, 0x8000001F, 0x80000020, 0x80000021, 0x80000022, 0x80000023,
0x80000024, 0x80000025, 0x80000026, 0x80000027, 0x80000028, 0x80000029,
0x8000002A, 0x8000002B, 0x8000002C, 0x8000002D, 0x8000002E, 0x8000002F,
0x80000030, 0x80000031, 0x80000032, 0x80000033, 0x80000034, 0x80000035,
0x80000036, 0x80000037, 0x80000038, 0x80000039, 0x8000003A, 0x8000003B,
0x8000003C, 0x8000003D, 0x8000003E, 0x8000003F
};
#endif

/* The Low Level Wrappers */
PAPIf_accum(int *EventSet, long long *values, int *check) {
   *check = PAPI_accum(*EventSet, values);
}

PAPIf_add_event(int *EventSet, int *Event, int *check){
   *Event = convert_event( *Event );
   *check = PAPI_add_event(EventSet, *Event);
}

PAPIf_add_events(int *EventSet, int *Events, int *number, int *check){
   int i;
   for ( i = 0; i < *number; i++ )
	Events[i] = convert_event(Events[i]);
   *check = PAPI_add_events(EventSet, Events, *number);
}

PAPIf_cleanup(int *EventSet, int *check){
   *check = PAPI_cleanup(EventSet);
}

PAPIf_init(int *check){
   *check = PAPI_init();
}

PAPIf_list_events(int *EventSet, int *Events, int *number, int *check){
   *check = PAPI_list_events(*EventSet, Events, number);
}

PAPIf_perror(int *code, char *destination, int *length, int *check){
   *check = PAPI_perror(*code, destination, *length);
}

PAPIf_query_event(int *EventCode, int *check) {
   *check = PAPI_query_event(*EventCode);
}
 
PAPIf_event_code_to_name(int *EventCode, char *out, int *check){
   *check = PAPI_event_code_to_name(*EventCode, out);
}

PAPIf_event_name_to_code(char *in, int *out, int *check){
   *check= PAPI_event_name_to_code(in, out);
}

PAPIf_read(int *EventSet, long long *values, int *check){
   *check = PAPI_read(*EventSet, values);
}

PAPIf_rem_event(int *EventSet, int *Event, int *check){
   *Event = convert_event( *Event );
   *check = PAPI_rem_event(EventSet, *Event);
}

PAPIf_rem_events(int *EventSet, int *Events, int *number, int *check){
   int i;
   for ( i = 0; i < *number; i++ ) 
	Events[i] = convert_event(Events[i]);

   *check = PAPI_rem_events(EventSet, Events, *number);
}

PAPIf_reset(int *EventSet, int *check){
   *check = PAPI_reset(*EventSet);
}

PAPIf_restore(int *check){
   *check = PAPI_restore();
}

PAPIf_save(int *check){
   *check = PAPI_save();
}

PAPIf_set_domain(int *domain, int *check){
   *check = PAPI_set_domain(*domain);
}

PAPIf_set_granularity(int *granularity, int *check){
   *check = PAPI_set_granularity(*granularity);
}

PAPIf_start(int *EventSet, int *check){
   *check = PAPI_start(*EventSet);
}

PAPIf_state(int *EventSet, int *status, int *check){
   *check = PAPI_state(*EventSet, status);
}

PAPIf_stop(int *EventSet, long long *values, int *check){
   *check = PAPI_stop(*EventSet, values);
}

PAPIf_write(int *EventSet, long long *values, int *check){
   *check = PAPI_write(*EventSet, values);
}

PAPIf_shutdown(){
   PAPI_shutdown();
}

PAPIf_get_hardware_info(int *ncpu, int *nnodes, int *totalcpus, int *vendor,
    char *vendor_string, int *model, char *model_string, float *revision,
    float *mhz){
   const PAPI_hw_info_t *hwinfo;
   hwinfo = PAPI_get_hardware_info();
   if ( hwinfo == NULL ){
	*ncpu = 0;
   }
   *ncpu = hwinfo->ncpu;
   *nnodes = hwinfo->nnodes;
   *totalcpus = hwinfo->totalcpus;
   *vendor = hwinfo->vendor;
   strcpy( vendor_string, hwinfo->vendor_string );
   *model = hwinfo->model;
   strcpy( model_string, hwinfo->model_string );
   *revision = hwinfo->revision;
   *mhz = hwinfo->mhz;
   return;
}

PAPIf_create_eventset(int *EventSet, int *check){
   *check = PAPI_create_eventset(EventSet);
}

PAPIf_destroy_eventset(int *EventSet, int *check){
   *check = PAPI_destroy_eventset(EventSet);
}


/* The High Level API Wrappers */
PAPIf_num_events(int *numevents){
   *numevents = PAPI_num_events();
}

PAPIf_start_counters(int *events, int *array_len, int *check){
   int i;
   for ( i=0; i<*array_len;i++ ) 
	events[i] = convert_event(events[i]);
   *check = PAPI_start_counters(events, *array_len);
}

PAPIf_read_counters(long long *values, int *array_len, int *check){
   *check = PAPI_read_counters(values, *array_len);
}

PAPIf_stop_counters(long long *values, int *array_len, int *check){
   *check = PAPI_stop_counters(values, *array_len);
}

PAPIf_get_real_usec( long long *time){
   *time = PAPI_get_real_usec();
}

PAPIf_get_real_cyc(long long *real_cyc){
   *real_cyc = PAPI_get_real_cyc();
}


/*
 * Macro for getting the real event value
 * 192 is PAPI_FORTRAN_MAX as defined in fpapi.h
 */
int convert_event( int value ) {
  if ( value>127 && value<192) value = fortran_events[value-128];
  return value;
}

/* this routine makes a mess of fortran strings. Grabbed from MPI_Connect */

int fstr2c (char *in, char *out, int outmaxlen)
{
int i, j, k;

j = strlen (in);
if (j >= outmaxlen ) j = outmaxlen -1; /* truncate with space for null */

for(i=0;i<j;i++)
        if (in[i]<' ' || in[i]>122) break;

strncpy (out, in, i);
out[i] = '\0';
return (i);
}
