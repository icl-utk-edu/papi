/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

#ifndef PAPI_PROTOS_H
#define PAPI_PROTOS_H

/* 
* File:    papi_protos.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Haihang You
*          you@cs.utk.edu
*          <your name here>
*          <your email address>
*/  

/* The following PAPI internal functions are defined by the papi_internal.c file. */
extern int _papi_hwi_read(hwd_context_t *context, EventSetInfo_t *ESI, long_long *values);
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
extern int _papi_hwi_lookup_EventCodeIndex(const EventSetInfo_t *ESI, unsigned int EventCode);
extern EventSetInfo_t *_papi_hwi_allocate_EventSet(void);
extern EventSetInfo_t *_papi_hwi_lookup_EventSet(int eventset);
extern int _papi_hwi_remove_EventSet(EventSetInfo_t *);
extern EventSetInfo_t *get_my_EventSetInfo(EventInfo_t *);
extern int _papi_hwi_mdi_init(void);
extern void print_state(EventSetInfo_t *ESI);

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

extern void _papi_hwi_shutdown_the_thread_list(void);
extern void _papi_hwi_cleanup_thread_list(void);
extern int _papi_hwi_insert_in_thread_list(ThreadInfo_t *ptr);
extern ThreadInfo_t *_papi_hwi_lookup_in_thread_list();
extern void _papi_hwi_shutdown_the_thread_list(void);

/* The following PAPI internal functions are defined by the extras.c file. */

extern int _papi_hwi_stop_overflow_timer(ThreadInfo_t *master, EventSetInfo_t *ESI);
extern int _papi_hwi_start_overflow_timer(ThreadInfo_t *master, EventSetInfo_t *ESI);
extern int _papi_hwi_initialize(DynamicArray_t **);
/*
extern void _papi_hwi_dispatch_overflow_signal(void *context);
*/
extern void _papi_hwi_dispatch_overflow_signal(void *context, int, long_long , int);

/* The following PAPI internal functions are defined by the substrate file. */

extern int _papi_hwd_init(hwd_context_t *);
/*
extern int _papi_hwd_add_event(hwd_register_map_t *chosen, hwd_preset_t *preset, hwd_control_state_t *out);
extern int _papi_hwd_remove_event(EventSetInfo_t *ESI, int *nix, int size);
extern int _papi_hwd_merge(EventSetInfo_t *ESI, EventSetInfo_t *zero);
extern void _papi_hwd_remove_native(hwd_control_state_t *this_state, NativeInfo_t *nativeInfo);
extern int _papi_hwd_add_event(hwd_control_state_t *this_state, int *nix, int size, EventInfo_t *out);
*/
extern void _papi_hwd_init_control_state(hwd_control_state_t *ptr);
extern int _papi_hwd_update_control_state(hwd_control_state_t *this_state, NativeInfo_t *native, int count);
extern int _papi_hwd_add_prog_event(hwd_control_state_t *, unsigned int, void *, EventInfo_t *); 
extern int _papi_hwd_allocate_registers(EventSetInfo_t *ESI);
extern int _papi_hwd_read(hwd_context_t *, hwd_control_state_t *, long_long **);
extern int _papi_hwd_shutdown(hwd_context_t *);
extern u_long_long _papi_hwd_get_real_cycles (void);
extern u_long_long _papi_hwd_get_real_usec (void);
extern u_long_long _papi_hwd_get_virt_cycles (const hwd_context_t *);
extern u_long_long _papi_hwd_get_virt_usec (const hwd_context_t *);
extern int _papi_hwd_start(hwd_context_t *, hwd_control_state_t *);
extern int _papi_hwd_reset(hwd_context_t *, hwd_control_state_t *);
extern int _papi_hwd_stop(hwd_context_t *, hwd_control_state_t *);
extern int _papi_hwd_write(hwd_context_t *, hwd_control_state_t *, long_long events[]);
extern int _papi_hwd_ctl(hwd_context_t *, int code, _papi_int_option_t *option);
extern int _papi_hwd_init_global(void);
extern int _papi_hwd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold);
extern int _papi_hwd_set_profile(EventSetInfo_t *ESI, EventSetProfileInfo_t *profile_option);
extern void *_papi_hwd_get_overflow_address(void *context);
extern void _papi_hwd_error(int error, char *);
extern void _papi_hwd_lock_init(void);
extern int _papi_hwd_shutdown_global(void);
extern int _papi_hwd_set_domain(hwd_control_state_t *, int);
extern int _papi_hwd_setmaxmem();
extern int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI);
extern int _papi_hwd_mdi_init(void);

#ifdef _WIN32
/* Callback routine for Windows timers */
void CALLBACK _papi_hwd_timer_callback(UINT wTimerID, UINT msg, DWORD dwUser, DWORD dw1, DWORD dw2);
#else
/* Callback routine for Linux/Unix timers */
void _papi_hwd_dispatch_timer(int signal, siginfo_t *info, void *tmp);
#endif

/* The following functions implement the native event query capability
   See extras.c or substrates for details... */

extern int _papi_hwi_query_native_event(unsigned int EventCode);
extern int _papi_hwi_query_native_event_verbose(unsigned int EventCode, PAPI_preset_info_t *info);
extern int _papi_hwi_native_name_to_code(char *in, int *out);
extern char *_papi_hwi_native_code_to_name(unsigned int EventCode);
extern char *_papi_hwi_native_code_to_descr(unsigned int EventCode);

/* The following functions implement the hardware dependent native event table access.
   The first four routines are required. The next two are optional.
   All six must at least be stubbed in the substrate file. 
*/

extern int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer);
extern char *_papi_hwd_ntv_code_to_name(unsigned int EventCode);
extern char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode);
extern int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits);
/* not completely defined yet... I'm partial to using XML -- dkt
    _papi_hwd_ntv_encode();
    _papi_hwd_ntv_decode();
*/

/* the following functions are counter allocation functions */
/* this function recusively does Modified Bipartite Graph counter allocation 
    success  return 1
    fail     return 0
	Author: Haihang You  you@cs.utk.edu
	Mods  : Dan Terpstra terpstra@cs.utk.edu
*/

extern int _papi_hwi_bipartite_alloc(hwd_reg_alloc_t *event_list, int count);

/* The following functions are called by _papi_hwi_bipartite_alloc().
   They are hardware dependent, but don't need to be implemented
   if _papi_hwi_bipartite_alloc() is not called.
 */

/* This function examines the event to determine
    if it can be mapped to counter ctr. 
    Returns true if it can, false if it can't.
*/
extern int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr);
/* This function forces the event to
    be mapped to only counter ctr. 
    Returns nothing.
*/
extern void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr);
/* This function examines the event to determine
    if it has a single exclusive mapping. 
    Returns true if exlusive, false if non-exclusive.
*/
extern int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t *dst);
/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
extern int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src);
/* This function removes the counters available to the src event
    from the counters available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
extern void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src);
/* This function updates the selection status of 
    the dst event based on information in the src event.
    Returns nothing.
*/
extern void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src);



/* The following functions are defined by the memory file. */

extern long _papi_hwd_get_dmem_info(int option);

/* Defined by the OS substrate file */

extern int _papi_hwd_update_shlib_info(void);
extern int _papi_hwd_get_system_info(void);

/* Defined in a memory file, could be processor or OS specific */
extern int _papi_hwd_get_memory_info( PAPI_hw_info_t *, int );

/* Linux defines; may also appear in substrates */
#ifdef linux
extern int sighold(int);
extern int sigrelse(int);
#endif

#endif /* PAPI_PROTOS_H */

