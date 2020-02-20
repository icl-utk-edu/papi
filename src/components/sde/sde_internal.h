#ifndef SDE_H
#define SDE_H

// Enable the following line if you want to use PAPI_overflow()
#define SDE_HAVE_OVERFLOW

#define PAPI_SDE_THREAD_SAFE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#if defined(SDE_HAVE_OVERFLOW)
#include <ucontext.h>
#endif //defined(SDE_HAVE_OVERFLOW)


#define is_readonly(_X_)  (PAPI_SDE_RO      == ((_X_)&0x0F))
#define is_readwrite(_X_) (PAPI_SDE_RW      == ((_X_)&0x0F))
#define is_delta(_X_)     (PAPI_SDE_DELTA   == ((_X_)&0xF0))
#define is_instant(_X_)   (PAPI_SDE_INSTANT == ((_X_)&0xF0))

#define EXP_CONTAINER_ENTRIES 52
#define EXP_CONTAINER_MIN_SIZE 2048

#ifndef SDE_MAX_SIMULTANEOUS_COUNTERS
#define SDE_MAX_SIMULTANEOUS_COUNTERS 40
#endif

#define PAPISDE_HT_SIZE 512
#define REGISTERED_EVENT_MASK 0x2;

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"

#include "interface/papi_sde_interface.h"

#if defined(PAPI_SDE_THREAD_SAFE)
  #define papi_sde_lock() _papi_hwi_lock(COMPONENT_LOCK);
  #define papi_sde_unlock() _papi_hwi_unlock(COMPONENT_LOCK);
#else
  #warning "Thread safe locking is _NOT_ activated"
  #define papi_sde_lock()
  #define papi_sde_unlock()
#endif

papi_vector_t _sde_vector;

/* We do not use this structure, but the framework needs its size */
typedef struct sde_register
{
   int junk;
} sde_register_t;

/* We do not use this structure, but the framework needs its size */
typedef struct sde_reg_alloc
{
	sde_register_t junk;
} sde_reg_alloc_t;

/** 
 *  There's one of these per event-set to hold data specific to the EventSet, like
 *  counter start values, number of events in a set and counter uniq ids.
 */
typedef struct sde_control_state
{
  int num_events;
  unsigned int which_counter[SDE_MAX_SIMULTANEOUS_COUNTERS]; 
  long long counter[SDE_MAX_SIMULTANEOUS_COUNTERS];
  long long previous_value[SDE_MAX_SIMULTANEOUS_COUNTERS];
#if defined(SDE_HAVE_OVERFLOW)
  timer_t timerid;
  int has_timer;
#endif //defined(SDE_HAVE_OVERFLOW)
} sde_control_state_t;

typedef struct sde_context {
   long long junk;
} sde_context_t;

typedef struct sde_counter_s sde_counter_t;
typedef struct sde_sorting_params_s sde_sorting_params_t;
typedef struct papisde_list_entry_s papisde_list_entry_t;
typedef struct papisde_library_desc_s papisde_library_desc_t;
typedef struct papisde_control_s papisde_control_t;
typedef struct recorder_data_s recorder_data_t;

/* Hash table entry */
struct papisde_list_entry_s {
    sde_counter_t *item;
    papisde_list_entry_t *next;
};

struct recorder_data_s{
   void *exp_container[EXP_CONTAINER_ENTRIES];
   long long total_entries;
   long long used_entries;
   size_t typesize;
   void *sorted_buffer;
   long long sorted_entries;
};

/* The following type describes a counter, or a counter group, or a recording. */
struct sde_counter_s {
   unsigned int glb_uniq_id;
   char *name;
   char *description;
   void *data; 
   long long int previous_data;
   recorder_data_t *recorder_data;
   int is_created;
   int overflow;
   papi_sde_fptr_t func_ptr;   
   void *param;   
   int cntr_type;
   int cntr_mode;
   papisde_library_desc_t *which_lib;
   papisde_list_entry_t *counter_group_head;
   uint32_t counter_group_flags;
};

struct sde_sorting_params_s{
   sde_counter_t *recording;
   int (*cmpr_func_ptr)(const void *p1, const void *p2);
};

/* This type describes one library. This is the type of the handle returned by papi_sde_init(). */
struct papisde_library_desc_s {
    char* libraryName;
    papisde_list_entry_t lib_counters[PAPISDE_HT_SIZE];
    papisde_library_desc_t *next;
};

/* One global variable of this type holds pointers to all other SDE meta-data */
struct papisde_control_s {
    unsigned int num_reg_events; /* This number only increases, so it can be used as a uniq id */
    unsigned int num_live_events; /* This number decreases at unregister() */
    papisde_library_desc_t *lib_list_head;
    unsigned int activeLibCount;
    papisde_list_entry_t all_reg_counters[PAPISDE_HT_SIZE];
};

/** This global variable points to the head of the control state list **/
static papisde_control_t *_papisde_global_control = NULL;

/* All of the following functions are for internal use only. */
static int _sde_reset( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_write( hwd_context_t *ctx, hwd_control_state_t *ctl, long long *events );
static int _sde_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events, int flags );
static int _sde_stop( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_start( hwd_context_t *ctx, hwd_control_state_t *ctl );
static int _sde_update_control_state( hwd_control_state_t *ctl, NativeInfo_t *native, int count, hwd_context_t *ctx );
static int _sde_init_control_state( hwd_control_state_t * ctl );
static int _sde_init_thread( hwd_context_t *ctx );
static int _sde_init_component( int cidx );
static int _sde_shutdown_component(void);
static int _sde_shutdown_thread( hwd_context_t *ctx );
static int _sde_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option );
static int _sde_set_domain( hwd_control_state_t * cntrl, int domain );
static int _sde_ntv_enum_events( unsigned int *EventCode, int modifier );
static int _sde_ntv_code_to_name( unsigned int EventCode, char *name, int len );
static int _sde_ntv_code_to_descr( unsigned int EventCode, char *descr, int len );
static int _sde_ntv_name_to_code(const char *name, unsigned int *event_code );

#if defined(SDE_HAVE_OVERFLOW)
static int _sde_set_overflow( EventSetInfo_t *ESI, int EventIndex, int threshold );
static void _sde_dispatch_timer( int n, hwd_siginfo_t *info, void *uc);
static void invoke_user_handler(sde_counter_t *cntr_handle);
static int set_timer_for_overflow( sde_control_state_t *sde_ctl );
#endif // defined(SDE_HAVE_OVERFLOW)

static papi_handle_t do_sde_init(const char *name_of_library);
static int sde_cast_and_store(void *data, long long int previous_value, void *rslt, int type);
static int sde_hardware_read_and_store( sde_counter_t *counter, long long int previous_value, long long int *rslt );
static int sde_read_counter_group( sde_counter_t *counter, long long int *rslt );
static int sde_setup_counter_internals( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter, papi_sde_fptr_t fp_counter, void *param, sde_counter_t **placeholder );
int aggregate_value_in_group(long long int *data, long long int *rslt, int cntr_type, int group_flags);
static inline int sde_do_register( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter, papi_sde_fptr_t fp_counter, void *param );

static sde_counter_t *allocate_and_insert(papisde_library_desc_t* lib_handle, const char *name, unsigned int uniq_id, int cntr_mode, int cntr_type, void *data, papi_sde_fptr_t func_ptr, void *param);
static int delete_counter(papisde_library_desc_t* lib_handle, const char *name);

static inline void free_counter(sde_counter_t *counter);
static unsigned int ht_hash_id(unsigned int uniq_id);
static unsigned long ht_hash_name(const char *str);
static void ht_insert(papisde_list_entry_t *hash_table, int key, sde_counter_t *sde_counter);
static sde_counter_t *ht_delete(papisde_list_entry_t *hash_table, int key, unsigned int uniq_id);
static sde_counter_t *ht_lookup_by_name(papisde_list_entry_t *hash_table, const char *name);
static sde_counter_t *ht_lookup_by_id(papisde_list_entry_t *hash_table, unsigned int uniq_id);
#endif
