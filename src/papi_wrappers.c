#include "papi.h"

/* The Low Level Wrappers */
int PAPI_accum(int EventSet, long long *values) {
   return internal_PAPI_accum(EventSet, values);
}

int PAPI_add_event(int *EventSet, int Event){
   return internal_PAPI_add_event(EventSet, Event);
}

int PAPI_add_events(int *EventSet, int *Events, int number){
   return internal_PAPI_add_events(EventSet, Events, number);
}

int PAPI_add_pevent(int *EventSet, int code, void *inout){
   return internal_PAPI_add_pevent(EventSet, code, inout);
}

int PAPI_cleanup(int *EventSet){
   return internal_PAPI_cleanup(EventSet);
}

int PAPI_get_opt(int option, PAPI_option_t *ptr){
   return internal_PAPI_get_opt(option, ptr);
}

int PAPI_init(void){
   return internal_PAPI_init();
}

int PAPI_list_events(int EventSet, int *Events, int *number){
   return internal_PAPI_list_events(EventSet, Events, number);
}

int PAPI_overflow(int EventSet, int EventCode, int threshold, int flags, 
	PAPI_overflow_handler_t handler){
   return internal_PAPI_overflow(EventSet, EventCode, threshold, 
		flags, handler);
}

int PAPI_perror(int code, char *destination, int length){
   return internal_PAPI_perror(code, destination, length);
}

int PAPI_profil(void *buf, int bufsiz, caddr_t offset, int scale, int EventSet,
	 int EventCode, int threshold, int flags){
   return internal_PAPI_profil(buf, bufsiz, offset, 
	scale, EventSet, EventCode, threshold, flags); 
}

int PAPI_query_event(int EventCode) {
   return internal_PAPI_query_event(EventCode);
}
 
int PAPI_query_event_verbose( int EventCode, PAPI_preset_info_t *info ) {
   return internal_PAPI_query_event_verbose( EventCode, info );
}

const PAPI_preset_info_t *PAPI_query_all_events_verbose(void){
   return internal_PAPI_query_all_events_verbose();
}

int PAPI_event_code_to_name(int EventCode, char *out){
   return internal_PAPI_event_code_to_name(EventCode, out);
}
int PAPI_event_name_to_code(char *in, int *out){
   return internal_PAPI_event_name_to_code(in, out);
}

int PAPI_read(int EventSet, long long *values){
   return internal_PAPI_read(EventSet, values);
}

int PAPI_rem_event(int *EventSet, int Event){
   return internal_PAPI_rem_event(EventSet, Event);
}

int PAPI_rem_events(int *EventSet, int *Events, int number){
   return internal_PAPI_rem_events(EventSet, Events, number);
}

int PAPI_reset(int EventSet){
   return internal_PAPI_reset(EventSet);
}

int PAPI_restore(void){
   return internal_PAPI_restore();
}
int PAPI_save(void){
   return internal_PAPI_save();
}

int PAPI_set_domain(int domain){
   return internal_PAPI_set_domain(domain);
}

int PAPI_set_granularity(int granularity){
   return internal_PAPI_set_granularity(granularity);
}

int PAPI_set_opt(int option, PAPI_option_t *ptr){
   return internal_PAPI_set_opt(option, ptr);
}

int PAPI_start(int EventSet){
   return internal_PAPI_start(EventSet);
}

int PAPI_state(int EventSet, int *status){
   return internal_PAPI_state(EventSet, status);
}

int PAPI_stop(int EventSet, long long *values){
   return internal_PAPI_stop(EventSet, values);
}

int PAPI_write(int EventSet, long long *values){
   return internal_PAPI_write(EventSet, values);
}

void PAPI_shutdown(void){
   internal_PAPI_shutdown();
}

void *PAPI_get_overflow_address(void *context){
   return internal_PAPI_get_overflow_address(context);
}

const PAPI_exe_info_t *PAPI_get_executable_info(void){
   return internal_PAPI_get_executable_info();
}

const PAPI_hw_info_t *PAPI_get_hardware_info(void){
   return internal_PAPI_get_hardware_info();
}

int PAPI_create_eventset(int *EventSet){
   return internal_PAPI_create_eventset(EventSet);
}

int PAPI_destroy_eventset(int *EventSet){
   return internal_PAPI_destroy_eventset(EventSet);
}


/* The High Level API Wrappers */
int PAPI_num_events(void){
   return internal_PAPI_num_events();
}

int PAPI_start_counters(int *events, int array_len){
   return internal_PAPI_start_counters(events, array_len);
}

int PAPI_read_counters(long long *values, int array_len){
   return internal_PAPI_read_counters(values, array_len);
}

int PAPI_stop_counters(long long *values, int array_len){
   return internal_PAPI_stop_counters(values, array_len);
}

long long PAPI_get_real_usec(void){
   return internal_PAPI_get_real_usec();
}

long long PAPI_get_real_cyc(void){
   return internal_PAPI_get_real_cyc();
}

