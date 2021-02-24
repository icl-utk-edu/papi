#ifndef _PAPI_SDE_COMMON_H
#define _PAPI_SDE_COMMON_H

// Enable the following line if you want to use PAPI_overflow()
#define SDE_HAVE_OVERFLOW

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <stdarg.h>
#if defined(SDE_HAVE_OVERFLOW)
#include <ucontext.h>
#endif //defined(SDE_HAVE_OVERFLOW)
#include "papi_sde_interface.h"

#define EXP_CONTAINER_ENTRIES 52
#define EXP_CONTAINER_MIN_SIZE 2048

#define PAPISDE_HT_SIZE 512

#define is_readonly(_X_)  (PAPI_SDE_RO      == ((_X_)&0x0F))
#define is_readwrite(_X_) (PAPI_SDE_RW      == ((_X_)&0x0F))
#define is_delta(_X_)     (PAPI_SDE_DELTA   == ((_X_)&0xF0))
#define is_instant(_X_)   (PAPI_SDE_INSTANT == ((_X_)&0xF0))

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
extern int _sde_be_verbose;
extern int _sde_debug;
#if defined(DEBUG)
#define SDEDBG(format, args...) { if(_sde_debug){fprintf(stderr,format, ## args);} }
#else // DEBUG
#define SDEDBG(format, args...) { ; }
#endif

static inline void SDE_ERROR( char *format, ... ){
    va_list args;
    if ( _sde_be_verbose ) {
        va_start( args, format );
        fprintf( stderr, "PAPI SDE Error: " );
        vfprintf( stderr, format, args );
        fprintf( stderr, "\n" );
        va_end( args );
    }
}

#endif // _PAPI_SDE_COMMON_H
