#ifndef _PAPI_SDE_COMMON_H
#define _PAPI_SDE_COMMON_H
extern papisde_control_t *get_global_struct(void);
extern sde_counter_t *ht_lookup_by_id(papisde_list_entry_t *hash_table, unsigned int uniq_id);
extern sde_counter_t *ht_lookup_by_name(papisde_list_entry_t *hash_table, const char *name);
extern sde_counter_t *ht_delete(papisde_list_entry_t *hash_table, int ht_key, unsigned int uniq_id);
extern void ht_insert(papisde_list_entry_t *hash_table, int ht_key, sde_counter_t *sde_counter);
extern unsigned long ht_hash_name(const char *str);
extern unsigned int ht_hash_id(unsigned int uniq_id);
extern papi_handle_t do_sde_init(const char *name_of_library, papisde_control_t *gctl);
extern sde_counter_t *allocate_and_insert(papisde_control_t *gctl, papisde_library_desc_t* lib_handle, const char *name, unsigned int uniq_id, int cntr_mode, int cntr_type, void *data, papi_sde_fptr_t func_ptr, void *param);
extern void recorder_data_to_contiguous(sde_counter_t *recorder, void *cont_buffer);
extern void SDE_ERROR( char *format, ... );
extern int _sde_be_verbose;
extern int _sde_debug;
#define SDEDBG(format, args...) { if(_sde_debug){fprintf(stderr,format, ## args);} }

#endif // _PAPI_SDE_COMMON_H
