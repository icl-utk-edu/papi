#ifndef SDE_H
#define SDE_H

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

#include "interface/papi_sde_interface.h"

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


#endif
