#include "papi.h"
#include <stdio.h>

/* Lets use defines to rename all the files */
#ifdef FORTRANUNDERSCORE
#define PAPI_accum papi_accum_
#define PAPI_add_event papi_add_event_
#define PAPI_add_events papi_add_events_
#define PAPI_cleanup papi_cleanup_
#define PAPI_list_events papi_list_events_
#define PAPI_perror papi_perror_
#define PAPI_query_event papi_query_event_
#define PAPI_event_name_to_code papi_event_name_to_code_
#define PAPI_read papi_read_
#define PAPI_rem_event papi_rem_event_
#define PAPI_rem_events papi_rem_events_
#define PAPI_reset papi_reset_
#define PAPI_restore papi_restore_
#define PAPI_save papi_save_
#define PAPI_set_domain papi_set_domain_
#define PAPI_set_granularity papi_set_granularity_
#define PAPI_start papi_start_
#define PAPI_state papi_state_
#define PAPI_stop papi_stop_
#define PAPI_write papi_write_
#define PAPI_shutdown papi_shutdown_
#define PAPI_get_hardware_info papi_get_hardware_info_
#define PAPI_create_eventset papi_create_eventset_
#define PAPI_destroy_eventset papi_destroy_eventset_
#define PAPI_num_events papi_num_events_
#define PAPI_start_counters papi_start_counters_
#define PAPI_read_counters papi_read_counters_
#define PAPI_stop_counters papi_stop_counters_
#define PAPI_get_real_usec papi_get_real_usec_
#define PAPI_get_real_cyc papi_get_real_cyc_

#elif FORTRANDOUBLEUNDERSCORE
#define PAPI_accum papi_accum__
#define PAPI_add_event papi_add_event__
#define PAPI_add_events papi_add_events__
#define PAPI_cleanup papi_cleanup__
#define PAPI_init papi_init__
#define PAPI_list_events papi_list_events__
#define PAPI_perror papi_perror__
#define PAPI_query_event papi_query_event__
#define PAPI_event_name_to_code papi_event_name_to_code__
#define PAPI_read papi_read__
#define PAPI_rem_event papi_rem_event__
#define PAPI_rem_events papi_rem_events__
#define PAPI_reset papi_reset__
#define PAPI_restore papi_restore__
#define PAPI_save papi_save__
#define PAPI_set_domain papi_set_domain__
#define PAPI_set_granularity papi_set_granularity__
#define PAPI_start papi_start__
#define PAPI_state papi_state__
#define PAPI_stop papi_stop__
#define PAPI_write papi_write__
#define PAPI_shutdown papi_shutdown__
#define PAPI_get_hardware_info papi_get_hardware_info__
#define PAPI_create_eventset papi_create_eventset__
#define PAPI_destroy_eventset papi_destroy_eventset__
#define PAPI_num_events papi_num_events__
#define PAPI_start_counters papi_start_counters__
#define PAPI_read_counters papi_read_counters__
#define PAPI_stop_counters papi_stop_counters__
#define PAPI_get_real_usec papi_get_real_usec__
#define PAPI_get_real_cyc papi_get_real_cyc__

#elif FORTRANALLCAPS
#define PAPI_accum PAPI_ACCUM
#define PAPI_add_event PAPI_ADD_EVENT
#define PAPI_add_events PAPI_ADD_EVENTS
#define PAPI_cleanup PAPI_CLEANUP
#define PAPI_init PAPI_INIT
#define PAPI_list_events PAPI_LIST_EVENTS
#define PAPI_perror PAPI_PERROR
#define PAPI_query_event PAPI_QUERY_EVENT
#define PAPI_event_name_to_code PAPI_EVENT_NAME_TO_CODE
#define PAPI_read PAPI_READ
#define PAPI_rem_event PAPI_REM_EVENT
#define PAPI_rem_events PAPI_REM_EVENTS
#define PAPI_reset PAPI_RESET
#define PAPI_restore PAPI_RESTORE
#define PAPI_save PAPI_SAVE
#define PAPI_set_domain PAPI_SET_DOMAIN
#define PAPI_set_granularity PAPI_SET_GRANULARITY
#define PAPI_start PAPI_START
#define PAPI_state PAPI_STATE
#define PAPI_stop PAPI_STOP
#define PAPI_write PAPI_WRITE
#define PAPI_shutdown PAPI_SHUTDOWN
#define PAPI_get_hardware_info PAPI_GET_HARDWARE_INFO
#define PAPI_create_eventset PAPI_CREATE_EVENTSET
#define PAPI_destroy_eventset PAPI_DESTROY_EVENTSET
#define PAPI_num_events PAPI_NUM_EVENTS
#define PAPI_start_counters PAPI_START_COUNTERS
#define PAPI_read_counters PAPI_READ_COUNTERS
#define PAPI_stop_counters PAPI_STOP_COUNTERS
#define PAPI_get_real_usec PAPI_GET_REAL_USEC
#define PAPI_get_real_cyc PAPI_GET_REAL_CYC
#endif

/* The Low Level Wrappers */
PAPI_accum(int *EventSet, long long *values, int *check) {
   *check = internal_PAPI_accum(*EventSet, values);
}

PAPI_add_event(int *EventSet, int *Event, int *check){
   *Event = convert_event( *Event );
   *check = internal_PAPI_add_event(EventSet, *Event);
}

PAPI_add_events(int *EventSet, int *Events, int *number, int *check){
   int i;
   for ( i = 0; i < *number; i++ )
	Events[i] = convert_event(Events[i]);
   *check = internal_PAPI_add_events(EventSet, Events, *number);
}

PAPI_cleanup(int *EventSet, int *check){
   *check = internal_PAPI_cleanup(EventSet);
}

PAPI_init(int *check){
   *check = internal_PAPI_init();
}

PAPI_list_events(int *EventSet, int *Events, int *number, int *check){
   *check = internal_PAPI_list_events(*EventSet, Events, number);
}

PAPI_perror(int *code, char *destination, int *length, int *check){
   *check = internal_PAPI_perror(*code, destination, *length);
}

PAPI_query_event(int *EventCode, int *check) {
   *check = internal_PAPI_query_event(*EventCode);
}
 
PAPI_event_code_to_name(int *EventCode, char *out, int *check){
   *check = internal_PAPI_event_code_to_name(*EventCode, out);
}

PAPI_event_name_to_code(char *in, int *out, int *check){
   *check= internal_PAPI_event_name_to_code(in, out);
}

PAPI_read(int *EventSet, long long *values, int *check){
   *check = internal_PAPI_read(*EventSet, values);
}

PAPI_rem_event(int *EventSet, int *Event, int *check){
   *Event = convert_event( *Event );
   *check = internal_PAPI_rem_event(EventSet, *Event);
}

PAPI_rem_events(int *EventSet, int *Events, int *number, int *check){
   int i;
   for ( i = 0; i < *number; i++ ) 
	Events[i] = convert_event(Events[i]);

   *check = internal_PAPI_rem_events(EventSet, Events, *number);
}

PAPI_reset(int *EventSet, int *check){
   *check = internal_PAPI_reset(*EventSet);
}

PAPI_restore(int *check){
   *check = internal_PAPI_restore();
}

PAPI_save(int *check){
   *check = internal_PAPI_save();
}

PAPI_set_domain(int *domain, int *check){
   *check = internal_PAPI_set_domain(*domain);
}

PAPI_set_granularity(int *granularity, int *check){
   *check = internal_PAPI_set_granularity(*granularity);
}

PAPI_start(int *EventSet, int *check){
   *check = internal_PAPI_start(*EventSet);
}

PAPI_state(int *EventSet, int *status, int *check){
   *check = internal_PAPI_state(*EventSet, status);
}

PAPI_stop(int *EventSet, long long *values, int *check){
   *check = internal_PAPI_stop(*EventSet, values);
}

PAPI_write(int *EventSet, long long *values, int *check){
   *check = internal_PAPI_write(*EventSet, values);
}

PAPI_shutdown(){
   internal_PAPI_shutdown();
}

PAPI_get_hardware_info(int *ncpu, int *nnodes, int *totalcpus, int *vendor,
    char *vendor_string, int *model, char *model_string, float *revision,
    float *mhz){
   const PAPI_hw_info_t *hwinfo;
   hwinfo = internal_PAPI_get_hardware_info();
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

PAPI_create_eventset(int *EventSet, int *check){
   *check = internal_PAPI_create_eventset(EventSet);
}

PAPI_destroy_eventset(int *EventSet, int *check){
   *check = internal_PAPI_destroy_eventset(EventSet);
}


/* The High Level API Wrappers */
PAPI_num_events(int *numevents){
   *numevents = internal_PAPI_num_events();
}

PAPI_start_counters(int *events, int *array_len, int *check){
   int i;
   for ( i=0; i<*array_len;i++ ) 
	events[i] = convert_event(events[i]);
   *check = internal_PAPI_start_counters(events, *array_len);
}

PAPI_read_counters(long long *values, int *array_len, int *check){
   *check = internal_PAPI_read_counters(values, *array_len);
}

PAPI_stop_counters(long long *values, int *array_len, int *check){
   *check = internal_PAPI_stop_counters(values, *array_len);
}

PAPI_get_real_usec( long long *time){
   *time = internal_PAPI_get_real_usec();
}

PAPI_get_real_cyc(long long *real_cyc){
   *real_cyc = internal_PAPI_get_real_cyc();
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
