#ifndef _PAPI_VECTOR_H
#define _PAPI_VECTOR_H

/* Vector Table Stuff */
typedef struct papi_svector {
  void (*func)();
  int  func_type;
} papi_svector_t;

/* If not vector code, or if not in the substrate, VECTOR_STATIC maps to a null string.
	Thus, prototypes behave the same as always.
	If inside the substrate (as defined by IN_SUBSTRATE at the top of a substrate file)
	VECTOR_STATIC maps to static, which allows routines to be properly prototyped and
	declared as static within the substrate file. This removes the warning messages,
	while preserving prototype checking on both sides of the substrate boundary.
*/

#ifdef PAPI_NO_VECTOR
#define papi_vectors_t void *
#define VECTOR_STATIC	
#else
#ifndef IN_SUBSTRATE
#define VECTOR_STATIC 
#else
#define VECTOR_STATIC static
#endif
enum {
   VEC_PAPI_END=0,
   VEC_PAPI_HWD_READ,
   VEC_PAPI_HWD_DISPATCH_TIMER,
   VEC_PAPI_HWD_GET_OVERFLOW_ADDRESS,
   VEC_PAPI_HWD_START,
   VEC_PAPI_HWD_STOP,
   VEC_PAPI_HWD_GET_REAL_CYCLES,
   VEC_PAPI_HWD_GET_REAL_USEC,
   VEC_PAPI_HWD_GET_VIRT_CYCLES,
   VEC_PAPI_HWD_GET_VIRT_USEC,
   VEC_PAPI_HWD_RESET,
   VEC_PAPI_HWD_WRITE,
   VEC_PAPI_HWD_STOP_PROFILING,
   VEC_PAPI_HWD_INIT,
   VEC_PAPI_HWD_INIT_CONTROL_STATE,
   VEC_PAPI_HWD_UPDATE_SHLIB_INFO,
   VEC_PAPI_HWD_GET_SYSTEM_INFO,
   VEC_PAPI_HWD_GET_MEMORY_INFO,
   VEC_PAPI_HWD_UPDATE_CONTROL_STATE,
   VEC_PAPI_HWD_CTL,
   VEC_PAPI_HWD_SET_OVERFLOW,
   VEC_PAPI_HWD_SET_PROFILE,
   VEC_PAPI_HWD_ADD_PROG_EVENT,
   VEC_PAPI_HWD_SET_DOMAIN,
   VEC_PAPI_HWD_NTV_ENUM_EVENTS,
   VEC_PAPI_HWD_NTV_CODE_TO_NAME,
   VEC_PAPI_HWD_NTV_CODE_TO_DESCR,
   VEC_PAPI_HWD_NTV_CODE_TO_BITS,
   VEC_PAPI_HWD_NTV_BITS_TO_INFO,
   VEC_PAPI_HWD_ALLOCATE_REGISTERS,
   VEC_PAPI_HWD_BPT_MAP_AVAIL,
   VEC_PAPI_HWD_BPT_MAP_SET,
   VEC_PAPI_HWD_BPT_MAP_EXCLUSIVE,
   VEC_PAPI_HWD_BPT_MAP_SHARED,
   VEC_PAPI_HWD_BPT_MAP_PREEMPT,
   VEC_PAPI_HWD_BPT_MAP_UPDATE,
   VEC_PAPI_HWD_GET_DMEM_INFO,
   VEC_PAPI_HWD_SHUTDOWN,
   VEC_PAPI_HWD_SHUTDOWN_GLOBAL,
   VEC_PAPI_HWD_USER,
   VEC_MAX_ENTRIES
};
typedef struct papi_vectors{
  int (*_vec_papi_hwd_read) (void *, void *, long_long **, int);
  void (*_vec_papi_hwd_dispatch_timer) (int, siginfo_t *, void *);
  void *(*_vec_papi_hwd_get_overflow_address) (int, char *);
  int (*_vec_papi_hwd_start) (void *, void *);
  int (*_vec_papi_hwd_stop) (void *, void *);
  long_long (*_vec_papi_hwd_get_real_cycles) ();
  long_long (*_vec_papi_hwd_get_real_usec) ();
  long_long (*_vec_papi_hwd_get_virt_cycles) (void *);
  long_long (*_vec_papi_hwd_get_virt_usec) (void *);
  int (*_vec_papi_hwd_reset) (void *, void *);
  int (*_vec_papi_hwd_write) (void *, void *, long_long[]);
  int (*_vec_papi_hwd_stop_profiling) (ThreadInfo_t *, EventSetInfo_t *);
  int (*_vec_papi_hwd_init) (void *);
  void (*_vec_papi_hwd_init_control_state) (void *);
  int (*_vec_papi_hwd_update_shlib_info) (void);
  int (*_vec_papi_hwd_get_system_info) ();
  int (*_vec_papi_hwd_get_memory_info) (PAPI_hw_info_t *, int);
  int (*_vec_papi_hwd_update_control_state) (void *, NativeInfo_t *, int, void *);
  int (*_vec_papi_hwd_ctl) (void *, int, _papi_int_option_t *);
  int (*_vec_papi_hwd_set_overflow) (EventSetInfo_t *, int, int);
  int (*_vec_papi_hwd_set_profile) (EventSetInfo_t *, int, int);
  int (*_vec_papi_hwd_add_prog_event) (void *, unsigned int, void *, EventInfo_t *);
  int (*_vec_papi_hwd_set_domain) (void *, int);
  int (*_vec_papi_hwd_ntv_enum_events) (unsigned int *, int);
  char * (*_vec_papi_hwd_ntv_code_to_name) (unsigned int);
  char * (*_vec_papi_hwd_ntv_code_to_descr) (unsigned int);
  int (*_vec_papi_hwd_ntv_code_to_bits) (unsigned int, void *);
  int (*_vec_papi_hwd_ntv_bits_to_info) (void *, char *, unsigned int *, int, int);
  int (*_vec_papi_hwd_allocate_registers) (EventSetInfo_t *);
  int (*_vec_papi_hwd_bpt_map_avail) (void *, int);
  void (*_vec_papi_hwd_bpt_map_set) (void *, int);
  int (*_vec_papi_hwd_bpt_map_exclusive) (void *);
  int (*_vec_papi_hwd_bpt_map_shared) (void *, void *);
  void (*_vec_papi_hwd_bpt_map_preempt) (void *, void *);
  void (*_vec_papi_hwd_bpt_map_update) (void *, void *);
  long (*_vec_papi_hwd_get_dmem_info) (int);
  int (*_vec_papi_hwd_shutdown) (void *);
  int (*_vec_papi_hwd_shutdown_global) (void);
  int (*_vec_papi_hwd_user) (int, void *, void *);
}papi_vectors_t;

extern papi_vectors_t _papi_vector_table;

/* Prototypes */
int _papi_hwi_setup_vector_table(papi_vectors_t *table, papi_svector_t *stable);
int _papi_hwi_initialize_vector_table(papi_vectors_t *table);


#endif

#endif /* _PAPI_VECTOR_H */
