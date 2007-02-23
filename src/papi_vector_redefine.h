/* Redfine the calls to vector table lookups */
#define _papi_hwd_context_size		_PAPI_CURRENT_VECTOR->context_size
#define _papi_hwd_control_state_size	_PAPI_CURRENT_VECTOR->control_state_size
#define _papi_hwd_register_size		_PAPI_CURRENT_VECTOR->register_size
#define _papi_hwd_reg_alloc_size	_PAPI_CURRENT_VECTOR->reg_alloc_size

#ifdef _WIN32
#define _papi_hwd_timer_callback	_PAPI_CURRENT_VECTOR->timer_callback
#else
#define _papi_hwd_dispatch_timer	_PAPI_CURRENT_VECTOR->dispatch_timer 
#endif
#define _papi_hwd_get_overflow_address	_PAPI_CURRENT_VECTOR->get_overflow_address
#define _papi_hwd_start			_PAPI_CURRENT_VECTOR->start 
#define _papi_hwd_stop			_PAPI_CURRENT_VECTOR->stop
#define _papi_hwd_read			_PAPI_CURRENT_VECTOR->read
#define _papi_hwd_reset			_PAPI_CURRENT_VECTOR->reset
#define _papi_hwd_write			_PAPI_CURRENT_VECTOR->write
#define _papi_hwd_get_real_cycles	_PAPI_CURRENT_VECTOR->get_real_cycles
#define _papi_hwd_get_real_usec		_PAPI_CURRENT_VECTOR->get_real_usec
#define _papi_hwd_get_virt_cycles	_PAPI_CURRENT_VECTOR->get_virt_cycles
#define _papi_hwd_get_virt_usec		_PAPI_CURRENT_VECTOR->get_virt_usec
#define _papi_hwd_stop_profiling	_PAPI_CURRENT_VECTOR->stop_profiling
#define _papi_hwd_init			_PAPI_CURRENT_VECTOR->init
#define _papi_hwd_init_control_state	_PAPI_CURRENT_VECTOR->init_control_state
#define _papi_hwd_update_shlib_info	_PAPI_CURRENT_VECTOR->update_shlib_info
#define _papi_hwd_get_system_info	_PAPI_CURRENT_VECTOR->get_system_info
#define _papi_hwd_get_memory_info	_PAPI_CURRENT_VECTOR->get_memory_info
#define _papi_hwd_update_control_state	_PAPI_CURRENT_VECTOR->update_control_state
#define _papi_hwd_ctl			_PAPI_CURRENT_VECTOR->ctl
#define _papi_hwd_set_overflow		_PAPI_CURRENT_VECTOR->set_overflow
#define _papi_hwd_set_profile		_PAPI_CURRENT_VECTOR->set_profile
#define _papi_hwd_add_prog_event	_PAPI_CURRENT_VECTOR->add_prog_event
#define _papi_hwd_set_domain		_PAPI_CURRENT_VECTOR->set_domain
#define _papi_hwd_ntv_enum_events	_PAPI_CURRENT_VECTOR->ntv_enum_events
#define _papi_hwd_ntv_code_to_name	_PAPI_CURRENT_VECTOR->ntv_code_to_name
#define _papi_hwd_ntv_code_to_descr	_PAPI_CURRENT_VECTOR->ntv_code_to_descr
#define _papi_hwd_ntv_code_to_bits	_PAPI_CURRENT_VECTOR->ntv_code_to_bits
#define _papi_hwd_ntv_bits_to_info	_PAPI_CURRENT_VECTOR->ntv_bits_to_info
#define _papi_hwd_allocate_registers	_PAPI_CURRENT_VECTOR->allocate_registers
#define _papi_hwd_bpt_map_avail		_PAPI_CURRENT_VECTOR->bpt_map_avail
#define _papi_hwd_bpt_map_set		_PAPI_CURRENT_VECTOR->bpt_map_set
#define _papi_hwd_bpt_map_exclusive	_PAPI_CURRENT_VECTOR->bpt_map_exclusive
#define _papi_hwd_bpt_map_shared	_PAPI_CURRENT_VECTOR->bpt_map_shared
#define _papi_hwd_bpt_map_preempt	_PAPI_CURRENT_VECTOR->bpt_map_preempt
#define _papi_hwd_bpt_map_update	_PAPI_CURRENT_VECTOR->bpt_map_update
#define _papi_hwd_get_dmem_info		_PAPI_CURRENT_VECTOR->get_dmem_info
#define _papi_hwd_shutdown		_PAPI_CURRENT_VECTOR->shutdown
#define _papi_hwd_shutdown_global	_PAPI_CURRENT_VECTOR->shutdown_global
