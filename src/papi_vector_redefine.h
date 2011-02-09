/* Redfine the calls to vector table lookups */

/* Component specific data structure */
//#define _papi_hwd_cmp_info        _papi_hwi_current_vector->cmp_info /* See definition in papi.h */

/* Private structure sizes for this component */
//#define _papi_hwd_cmp_size        _papi_hwi_current_vector->size

/* List of exposed function pointers for this component */
/*
#ifdef _WIN32
#define _papi_hwd_timer_callback	_papi_hwi_current_vector->timer_callback
#else
#define _papi_hwd_dispatch_timer	_papi_hwi_current_vector->dispatch_timer 
#endif
#define _papi_hwd_get_overflow_address	_papi_hwi_current_vector->get_overflow_address
#define _papi_hwd_start			_papi_hwi_current_vector->start 
#define _papi_hwd_stop			_papi_hwi_current_vector->stop
#define _papi_hwd_read			_papi_hwi_current_vector->read
#define _papi_hwd_reset			_papi_hwi_current_vector->reset
#define _papi_hwd_write			_papi_hwi_current_vector->write
#define _papi_hwd_destroy_eventset	_papi_hwi_current_vector->destroy_eventset
#define _papi_hwd_get_real_cycles	_papi_hwi_current_vector->get_real_cycles
#define _papi_hwd_get_real_usec		_papi_hwi_current_vector->get_real_usec
#define _papi_hwd_get_virt_cycles	_papi_hwi_current_vector->get_virt_cycles
#define _papi_hwd_get_virt_usec		_papi_hwi_current_vector->get_virt_usec
#define _papi_hwd_stop_profiling	_papi_hwi_current_vector->stop_profiling
#define _papi_hwd_init_substrate	_papi_hwi_current_vector->init_substrate
#define _papi_hwd_init			_papi_hwi_current_vector->init
#define _papi_hwd_init_control_state	_papi_hwi_current_vector->init_control_state
#define _papi_hwd_update_shlib_info	_papi_hwi_current_vector->update_shlib_info
#define _papi_hwd_get_system_info	_papi_hwi_current_vector->get_system_info
#define _papi_hwd_get_memory_info	_papi_hwi_current_vector->get_memory_info
#define _papi_hwd_update_control_state	_papi_hwi_current_vector->update_control_state
#define _papi_hwd_ctl			_papi_hwi_current_vector->ctl
#define _papi_hwd_set_overflow		_papi_hwi_current_vector->set_overflow
#define _papi_hwd_set_profile		_papi_hwi_current_vector->set_profile
#define _papi_hwd_add_prog_event	_papi_hwi_current_vector->add_prog_event
#define _papi_hwd_set_domain		_papi_hwi_current_vector->set_domain
#define _papi_hwd_ntv_enum_events	_papi_hwi_current_vector->ntv_enum_events
#define _papi_hwd_ntv_code_to_name	_papi_hwi_current_vector->ntv_code_to_name
#define _papi_hwd_ntv_code_to_descr	_papi_hwi_current_vector->ntv_code_to_descr
#define _papi_hwd_ntv_code_to_bits	_papi_hwi_current_vector->ntv_code_to_bits
#define _papi_hwd_ntv_bits_to_info	_papi_hwi_current_vector->ntv_bits_to_info
#define _papi_hwd_allocate_registers	_papi_hwi_current_vector->allocate_registers
#define _papi_hwd_bpt_map_avail		_papi_hwi_current_vector->bpt_map_avail
#define _papi_hwd_bpt_map_set		_papi_hwi_current_vector->bpt_map_set
#define _papi_hwd_bpt_map_exclusive	_papi_hwi_current_vector->bpt_map_exclusive
#define _papi_hwd_bpt_map_shared	_papi_hwi_current_vector->bpt_map_shared
#define _papi_hwd_bpt_map_preempt	_papi_hwi_current_vector->bpt_map_preempt
#define _papi_hwd_bpt_map_update	_papi_hwi_current_vector->bpt_map_update
#define _papi_hwd_get_dmem_info		_papi_hwi_current_vector->get_dmem_info
#define _papi_hwd_shutdown		_papi_hwi_current_vector->shutdown
#define _papi_hwd_shutdown_global	_papi_hwi_current_vector->shutdown_global
*/
