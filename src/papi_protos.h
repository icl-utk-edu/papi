#ifndef PAPI_PROTOS_H
#define PAPI_PROTOS_H

/* 
* File:    protos.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/  

/* The following PAPI internal functions are defined by the papi.c file. */

extern int _papi_hwi_read(hwd_context_t *context, EventSetInfo_t *ESI, unsigned long long *values);
extern int _papi_hwi_allocate_eventset_map(void);
extern int _papi_hwi_initialize_thread(ThreadInfo_t **master);
extern int _papi_hwi_create_eventset(int *EventSet, ThreadInfo_t *handle);
extern int _papi_hwi_add_event(EventSetInfo_t *ESI, int index);
extern int _papi_hwi_add_pevent(EventSetInfo_t *ESI, int EventCode, void *inout);
extern int _papi_hwi_remove_event(EventSetInfo_t *ESI, int EventCode);
extern int _papi_hwi_remove_EventSet(EventSetInfo_t *ESI);
extern int _papi_hwi_cleanup_eventset(EventSetInfo_t *ESI);
extern int _papi_hwi_get_domain(PAPI_domain_option_t *opt);
extern int _papi_hwi_convert_eventset_to_multiplex(EventSetInfo_t *ESI);
extern int _papi_hwi_lookup_EventCodeIndex(const EventSetInfo_t *ESI, int EventCode);
extern EventSetInfo_t *_papi_hwi_allocate_EventSet(void);
extern EventSetInfo_t *_papi_hwi_lookup_EventSet(int eventset);
extern int _papi_hwi_remove_EventSet(EventSetInfo_t *);
extern int _papi_hwi_query(int preset_index, int *flags, char **note);

/* The following PAPI internal functions are defined by the multiplex.c file. */

extern int mpx_init(int);
extern int mpx_add_event(MPX_EventSet **, int EventCode);
extern int mpx_remove_event(MPX_EventSet **, int EventCode);
extern int MPX_add_events(MPX_EventSet ** mpx_events, int * event_list, int num_events);
extern int MPX_stop(MPX_EventSet * mpx_events, long_long * values);
extern int MPX_cleanup(MPX_EventSet ** mpx_events);
extern void MPX_shutdown(void);
extern int MPX_reset(MPX_EventSet * mpx_events);
extern int MPX_read(MPX_EventSet * mpx_events, long_long * values);
extern int MPX_start(MPX_EventSet * mpx_events);

/* The following PAPI internal functions are defined by the threads.c file. */

extern void _papi_hwi_cleanup_thread_list(void);
extern int _papi_hwi_insert_in_thread_list(ThreadInfo_t *ptr);
extern ThreadInfo_t *_papi_hwi_lookup_in_thread_list();

/* The following PAPI internal functions are defined by the extras.c file. */

extern int _papi_hwi_stop_overflow_timer(ThreadInfo_t *master, EventSetInfo_t *ESI);
extern int _papi_hwi_start_overflow_timer(ThreadInfo_t *master, EventSetInfo_t *ESI);
extern int _papi_hwi_initialize(DynamicArray_t **);
extern void _papi_hwi_dispatch_overflow_signal(void *context);

#ifdef _WIN32
/* Callback routine for Windows timers */
void CALLBACK _papi_hwd_timer_callback(UINT wTimerID, UINT msg, DWORD dwUser, DWORD dw1, DWORD dw2);
#endif

/* The following PAPI internal functions are defined by the substrate file. */

extern int _papi_hwd_init(hwd_context_t *);
extern int _papi_hwd_add_event(hwd_register_map_t *chosen, hwd_preset_t *preset, hwd_control_state_t *out);
extern int _papi_hwd_add_prog_event(hwd_control_state_t *, int, void *, EventInfo_t *); 
extern int _papi_hwd_allocate_registers(hwd_control_state_t *, hwd_preset_t *, hwd_register_map_t *);
extern int _papi_hwd_read(hwd_context_t *, hwd_control_state_t *, unsigned long long **);
extern int _papi_hwd_shutdown(hwd_context_t *);
extern int _papi_hwd_remove_event(hwd_register_map_t *chosen, unsigned hardware_index, hwd_control_state_t *out);
extern unsigned long_long _papi_hwd_get_real_cycles (void);
extern unsigned long_long _papi_hwd_get_real_usec (void);
extern unsigned long_long _papi_hwd_get_virt_cycles (const hwd_context_t *);
extern unsigned long_long _papi_hwd_get_virt_usec (const hwd_context_t *);
extern int _papi_hwd_start(hwd_context_t *, hwd_control_state_t *);
extern int _papi_hwd_reset(hwd_context_t *, hwd_control_state_t *);
extern int _papi_hwd_stop(hwd_context_t *, hwd_control_state_t *);
extern int _papi_hwd_write(hwd_context_t *, hwd_control_state_t *, long_long events[]);
extern int _papi_hwd_ctl(hwd_context_t *, int code, _papi_int_option_t *option);
void _papi_hwd_dispatch_timer(int signal, siginfo_t *info, void *tmp);
extern int _papi_hwd_init_global(void);
extern int _papi_hwd_merge(EventSetInfo_t *ESI, EventSetInfo_t *zero);
extern int _papi_hwd_query(int preset, int *flags, char **note_loc);
extern int _papi_hwd_set_overflow(EventSetInfo_t *ESI, EventSetOverflowInfo_t *overflow_option);
extern int _papi_hwd_set_profile(EventSetInfo_t *ESI, EventSetProfileInfo_t *profile_option);
extern void *_papi_hwd_get_overflow_address(void *context);
extern void _papi_hwd_error(int error, char *);
extern void _papi_hwd_lock_init(void);
extern void _papi_hwd_lock(void);
extern void _papi_hwd_unlock(void);
extern int _papi_hwd_shutdown_global(void);
extern int _papi_hwd_set_domain(hwd_control_state_t *, int);

/* Defined by the OS substrate file */

extern int _papi_hwd_update_shlib_info(void);
extern int _papi_hwd_get_system_info(void);

#endif /* PAPI_PROTOS_H */
