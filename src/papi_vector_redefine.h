#ifndef PAPI_NO_VECTOR
/* Redfine the calls to vector table lookups */
#define _papi_hwd_read(a,b,c,d,idx) _papi_vector_table[idx]._vec_papi_hwd_read(a,b,c,d)
#define _papi_hwd_start(a,b,idx)  _papi_vector_table[idx]._vec_papi_hwd_start(a,b) 
#define _papi_hwd_stop(a,b,idx) _papi_vector_table[idx]._vec_papi_hwd_stop(a,b)
#define _papi_hwd_reset(a,b,idx) _papi_vector_table[idx]._vec_papi_hwd_reset(a,b)
#define _papi_hwd_write(a,b,c,idx) _papi_vector_table[idx]._vec_papi_hwd_write(a,b,c)
#define _papi_hwd_stop_profiling(a,b,idx) _papi_vector_table[idx]._vec_papi_hwd_stop_profiling(a,b)
#define _papi_hwd_update_control_state(a,b,c,d,idx) _papi_vector_table[idx]._vec_papi_hwd_update_control_state(a,b,c,d)
#define _papi_hwd_set_overflow(a,b,c,idx) _papi_vector_table[idx]._vec_papi_hwd_set_overflow(a,b,c)
#define _papi_hwd_set_profile(a,b,c,idx) _papi_vector_table[idx]._vec_papi_hwd_set_profile(a,b,c)
#define _papi_hwd_allocate_registers(a,idx) _papi_vector_table[idx]._vec_papi_hwd_allocate_registers(a)
#define _papi_hwd_init_control_state(a,idx) _papi_vector_table[idx]._vec_papi_hwd_init_control_state(a)

#define _papi_hwd_dispatch_timer _papi_vector_table[0]._vec_papi_hwd_dispatch_timer

/* These functions are all tied to the CPU substrate */ 
#define _papi_hwd_get_real_cycles() _papi_vector_table[0]._vec_papi_hwd_get_real_cycles()
#define _papi_hwd_get_real_usec() _papi_vector_table[0]._vec_papi_hwd_get_real_usec()
#define _papi_hwd_get_virt_cycles(a)  _papi_vector_table[0]._vec_papi_hwd_get_virt_cycles(a)
#define _papi_hwd_get_virt_usec(a) _papi_vector_table[0]._vec_papi_hwd_get_virt_usec(a)
#define _papi_hwd_update_shlib_info() _papi_vector_table[0]._vec_papi_hwd_update_shlib_info()
#define _papi_hwd_get_system_info() _papi_vector_table[0]._vec_papi_hwd_get_system_info()
#define _papi_hwd_get_memory_info(a,b) _papi_vector_table[0]._vec_papi_hwd_get_memory_info(a,b)
#define _papi_hwd_get_dmem_info(a) _papi_vector_table[0]._vec_papi_hwd_get_dmem_info(a)

/* Figure out what to do with these */
#define _papi_hwd_ctl(a,b,c) _papi_vector_table[0]._vec_papi_hwd_ctl(a,b,c)
#define _papi_hwd_ntv_enum_events(a,b) _papi_vector_table[0]._vec_papi_hwd_ntv_enum_events(a,b)
#define _papi_hwd_ntv_code_to_name(a) _papi_vector_table[0]._vec_papi_hwd_ntv_code_to_name(a)
#define _papi_hwd_ntv_code_to_descr(a) _papi_vector_table[0]._vec_papi_hwd_ntv_code_to_descr(a)
#define _papi_hwd_ntv_code_to_bits(a,b) _papi_vector_table[0]._vec_papi_hwd_ntv_code_to_bits(a,b)
#define _papi_hwd_ntv_bits_to_info(a,b,c,d,e) _papi_vector_table[0]._vec_papi_hwd_ntv_bits_to_info(a,b,c,d,e)
#define _papi_hwd_bpt_map_avail(a,b) _papi_vector_table[0]._vec_papi_hwd_bpt_map_avail(a,b)
#define _papi_hwd_bpt_map_set(a,b) _papi_vector_table[0]._vec_papi_hwd_bpt_map_set(a,b)
#define _papi_hwd_bpt_map_exclusive(a) _papi_vector_table[0]._vec_papi_hwd_bpt_map_exclusive(a)
#define _papi_hwd_bpt_map_shared(a,b) _papi_vector_table[0]._vec_papi_hwd_bpt_map_shared(a,b)
#define _papi_hwd_bpt_map_preempt(a,b) _papi_vector_table[0]._vec_papi_hwd_bpt_map_preempt(a,b)
#define _papi_hwd_bpt_map_update(a,b) _papi_vector_table[0]._vec_papi_hwd_bpt_map_update(a,b)
#define _papi_hwd_shutdown(a) _papi_vector_table[0]._vec_papi_hwd_shutdown(a)
#endif
