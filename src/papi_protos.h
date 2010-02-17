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
 int _papi_hwi_read(hwd_context_t * context, EventSetInfo_t * ESI,
                          long long * values);
 int _papi_hwi_create_eventset(int *EventSet, ThreadInfo_t * handle);
 int _papi_hwi_assign_eventset(EventSetInfo_t *ESI, int cidx);
 void _papi_hwi_free_EventSet(EventSetInfo_t * ESI);
 int _papi_hwi_add_event(EventSetInfo_t * ESI, int index);
 int _papi_hwi_add_pevent(EventSetInfo_t * ESI, int EventCode, void *inout);
 int _papi_hwi_remove_event(EventSetInfo_t * ESI, int EventCode);
 int _papi_hwi_remove_EventSet(EventSetInfo_t * ESI);
 int _papi_hwi_cleanup_eventset(EventSetInfo_t * ESI);
 int _papi_hwi_get_domain(PAPI_domain_option_t * opt);
 int _papi_hwi_get_granularity(PAPI_granularity_option_t * opt);
 int _papi_hwi_convert_eventset_to_multiplex(_papi_int_multiplex_t *opt);
 int _papi_hwi_lookup_EventCodeIndex(const EventSetInfo_t * ESI,
                                           unsigned int EventCode);
inline_static EventSetInfo_t *_papi_hwi_lookup_EventSet(int eventset);
 int _papi_hwi_remove_EventSet(EventSetInfo_t *);
 void print_state(EventSetInfo_t * ESI);
 int _papi_hwi_init_global_internal(void);
 int _papi_hwi_init_global(void);
 void _papi_hwi_shutdown_global_internal(void);
 int _papi_hwi_cleanup_all_presets(void);
 int _papi_hwi_get_event_info(int EventCode, PAPI_event_info_t * info);
 int _papi_hwi_set_event_info(PAPI_event_info_t * info, int *EventCode);

/* The following PAPI internal functions are defined by the papi_preset.c file. */

int _papi_hwi_setup_all_presets(hwi_search_t *findem, hwi_dev_notes_t *notes);
int _papi_hwi_cleanup_all_presets(void);
#ifdef XML
int _xml_papi_hwi_setup_all_presets(char *arch, hwi_dev_notes_t *notes);
#endif

/* The following PAPI internal functions are defined by the multiplex.c file. */

 int mpx_check(int EventSet);
 int mpx_init(int);
 int mpx_add_event(MPX_EventSet **, int EventCode, int domain, int granularity);
 int mpx_remove_event(MPX_EventSet **, int EventCode);
 int MPX_add_events(MPX_EventSet ** mpx_events, int *event_list, int num_events, int domain, int granularity);
 int MPX_stop(MPX_EventSet * mpx_events, long long * values);
 int MPX_cleanup(MPX_EventSet ** mpx_events);
 void MPX_shutdown(void);
 int MPX_reset(MPX_EventSet * mpx_events);
 int MPX_read(MPX_EventSet * mpx_events, long long * values);
 int MPX_start(MPX_EventSet * mpx_events);

/* The following PAPI internal functions are defined by the threads.c file. */

 void _papi_hwi_shutdown_the_thread_list(void);
 void _papi_hwi_cleanup_thread_list(void);
 int _papi_hwi_insert_in_thread_list(ThreadInfo_t * ptr);
 ThreadInfo_t *_papi_hwi_lookup_in_thread_list();
 void _papi_hwi_shutdown_the_thread_list(void);
 int  _papi_hwi_get_thr_context(void ** );
 int _papi_hwi_gather_all_thrspec_data(int tag, PAPI_all_thr_spec_t *where);

/* The following PAPI internal functions are defined by the extras.c file. */

 int _papi_hwi_stop_timer(int timer, int signal);
 int _papi_hwi_start_timer(int timer, int signal, int ms);
 int _papi_hwi_stop_signal(int signal);
 int _papi_hwi_start_signal(int signal, int need_context, int cidx);
 int _papi_hwi_initialize(DynamicArray_t **);
 int _papi_hwi_dispatch_overflow_signal(void *papiContext, caddr_t address, int *, long long, int, ThreadInfo_t **master, int cidx);
 void _papi_hwi_dispatch_profile(EventSetInfo_t * ESI, caddr_t address, long long over, int profile_index);

 /* The following PAPI internal functions are defined by the papi_data.c file. */

int _papi_hwi_derived_type(char *derived, int *code);
 int _papi_hwi_derived_string(int type, char *derived, int len);

/* The following PAPI internal functions are defined by the substrate file. */

 int _papi_hwd_get_system_info(void);
 int _papi_hwd_init_substrate(int cidx);
 int _papi_hwd_init(hwd_context_t *);
 int _papi_hwd_init_control_state(hwd_control_state_t * ptr);
 int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                                          NativeInfo_t * native, int count, hwd_context_t *);
 int _papi_hwd_add_prog_event(hwd_control_state_t *, unsigned int, void *,
                                    EventInfo_t *);
 int _papi_hwd_allocate_registers(EventSetInfo_t * ESI);
 int _papi_hwd_read(hwd_context_t *, hwd_control_state_t *, long long **, int);
 int _papi_hwd_shutdown(hwd_context_t *);
/* The following functions are now defined in the substrate header files to be inline_static */
 long long _papi_hwd_get_real_cycles(void);
 long long _papi_hwd_get_real_usec(void);
 long long _papi_hwd_get_virt_cycles(const hwd_context_t *);
 long long _papi_hwd_get_virt_usec(const hwd_context_t *);
/* End of above */
 int _papi_hwd_start(hwd_context_t *, hwd_control_state_t *);
 int _papi_hwd_reset(hwd_context_t *, hwd_control_state_t *);
 int _papi_hwd_stop(hwd_context_t *, hwd_control_state_t *);
 int _papi_hwd_write(hwd_context_t *, hwd_control_state_t *, long long events[]);
 int _papi_hwd_ctl(hwd_context_t *, int code, _papi_int_option_t * option);
 int _papi_hwd_init_global(void);
 int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold);
 int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold);
 void *_papi_hwd_get_overflow_address(void *context);
 int _papi_hwd_shutdown_global(void);
 int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI);


#ifdef _WIN32
/* Callback routine for Windows timers */
void CALLBACK _papi_hwd_timer_callback(UINT wTimerID, UINT msg, DWORD dwUser, DWORD dw1,
                                       DWORD dw2);
#else
/* Callback routine for Linux/Unix timers */
void _papi_hwd_dispatch_timer(int signal, hwd_siginfo_t * info, void *tmp);
#endif

/* The following functions implement the native event query capability
   See extras.c or substrates for details... */

 int _papi_hwi_query_native_event(unsigned int EventCode);
 int _papi_hwi_get_native_event_info(unsigned int EventCode,
                                           PAPI_event_info_t * info);
 int _papi_hwi_native_name_to_code(char *in, int *out);
 int _papi_hwi_native_code_to_name(unsigned int EventCode, char *hwi_name, int len);
 int _papi_hwi_native_code_to_descr(unsigned int EventCode, char *hwi_descr, int len);

/* The following functions implement the hardware dependent native event table access.
   The first four routines are required. The next two are optional.
   All six must at least be stubbed in the substrate file. 
*/

 int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier);
/* The following were the only two calls in the substrate API that return
    pointers to strings. They were initially intended to always reference
    const strings, but this is no longer universally true. Hence, they are
    not always thread-safe, and have been redefined and recoded to match 
    the calling sequence of the equivalent _hwi_ functions.
 char *_papi_hwd_ntv_code_to_name(unsigned int EventCode);
 char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode);
 */
 int _papi_hwd_ntv_code_to_name(unsigned int EventCode, char *hwd_name, int len);
 int _papi_hwd_ntv_code_to_descr(unsigned int EventCode, char *hwd_descr, int len);
 int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits);
 int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values,
                                      int name_len, int count);
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

 int _papi_hwi_bipartite_alloc(hwd_reg_alloc_t * event_list, int count, int cidx);

/* The following functions are called by _papi_hwi_bipartite_alloc().
   They are hardware dependent, but don't need to be implemented
   if _papi_hwi_bipartite_alloc() is not called.
 */

/* This function examines the event to determine
    if it can be mapped to counter ctr. 
    Returns true if it can, false if it can't.
*/
 int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr);
/* This function forces the event to
    be mapped to only counter ctr. 
    Returns nothing.
*/
 void _papi_hwd_bpt_map_set(hwd_reg_alloc_t * dst, int ctr);
/* This function examines the event to determine
    if it has a single exclusive mapping. 
    Returns true if exlusive, false if non-exclusive.
*/
 int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst);
/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
 int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src);
/* This function removes the counters available to the src event
    from the counters available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
 void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src);
/* This function updates the selection status of 
    the dst event based on information in the src event.
    Returns nothing.
*/
 void _papi_hwd_bpt_map_update(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src);

/* The following functions are defined by papi_hl.c */
 void _papi_hwi_shutdown_highlevel();


/* The following functions are defined by the memory file. */

 int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *);

/* Defined by the OS substrate file */
 int _papi_hwd_update_shlib_info(void);

/* Defined in a memory file, could be processor or OS specific */
 int _papi_hwd_get_memory_info(PAPI_hw_info_t *, int);

/* papi_internal.c global papi error function */
void PAPIERROR(char *format, ...);

#if (!defined(HAVE_FFSLL) || defined(_BGP))
 int ffsll(long long lli);
#endif

#endif                          /* PAPI_PROTOS_H */
