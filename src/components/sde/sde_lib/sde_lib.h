/**
 * @file    sde_lib.h
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is a collection of utility functions that are needed by both the
 *  SDE component in PAPI and third party libraries with SDEs. Everything is
 *  included in this header file to facilitate easy integration into third
 *  party libraries.
 */

#if !defined(PAPI_SDE_LIB_H)
#define PAPI_SDE_LIB_H

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

#define PAPI_SDE_RO       0x00
#define PAPI_SDE_RW       0x01
#define PAPI_SDE_DELTA    0x00
#define PAPI_SDE_INSTANT  0x10

#define PAPI_SDE_long_long 0x0
#define PAPI_SDE_int       0x1
#define PAPI_SDE_double    0x2
#define PAPI_SDE_float     0x3

#define PAPI_SDE_SUM       0x0
#define PAPI_SDE_MAX       0x1
#define PAPI_SDE_MIN       0x2


#define GET_FLOAT_SDE(x) *((float *)&x)
#define GET_DOUBLE_SDE(x) *((double *)&x)
/*
 * GET_SDE_RECORDER_ADDRESS() USAGE EXAMPLE:
 * If SDE recorder logs values of type 'double':
 *     double *ptr = GET_SDE_RECORDER_ADDRESS(papi_event_value[6], double);
 *     for (j=0; j<CNT; j++)
 *        printf("    %d: %.4e\n",j, ptr[j]);
 */
#define GET_SDE_RECORDER_ADDRESS(x,rcrd_type) ((rcrd_type *)x)


typedef long long int (*papi_sde_fptr_t)( void * );
typedef int (*papi_sde_cmpr_fptr_t)( void * );
typedef void * papi_handle_t;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct papi_sde_fptr_struct_s {
    papi_handle_t (*init)(const char *lib_name );
    int (*register_counter)( papi_handle_t handle, const char *event_name, int mode, int type, void *counter );
    int (*register_fp_counter)( papi_handle_t handle, const char *event_name, int mode, int type, papi_sde_fptr_t fp_counter, void *param );
    int (*unregister_counter)( papi_handle_t handle, const char *event_name );
    int (*describe_counter)( papi_handle_t handle, const char *event_name, const char *event_description );
    int (*add_counter_to_group)( papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags );
    int (*create_counter)( papi_handle_t handle, const char *event_name, int cntr_type, void **cntr_handle );
    int (*inc_counter)( papi_handle_t cntr_handle, long long int increment );
    int (*create_recorder)( papi_handle_t handle, const char *event_name, size_t typesize, int (*cmpr_func_ptr)(const void *p1, const void *p2), void **record_handle );
    int (*record)( void *record_handle, size_t typesize, void *value );
    int (*reset_recorder)(void *record_handle );
    int (*reset_counter)( void *cntr_handle );
    void *(*get_counter_handle)(papi_handle_t handle, const char *event_name);
}papi_sde_fptr_struct_t;

papi_handle_t papi_sde_init(const char *name_of_library );
int papi_sde_register_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter );
int papi_sde_register_fp_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t func_ptr, void *param );
int papi_sde_unregister_counter( void *handle, const char *event_name );
int papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description );
int papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags );
int papi_sde_create_counter( papi_handle_t handle, const char *event_name, int cntr_type, void **cntr_handle );
int papi_sde_inc_counter( void *cntr_handle, long long int increment );
int papi_sde_create_recorder( papi_handle_t handle, const char *event_name, size_t typesize, int (*cmpr_func_ptr)(const void *p1, const void *p2), void **record_handle );
int papi_sde_record( void *record_handle, size_t typesize, void *value );
int papi_sde_reset_recorder(void *record_handle );
int papi_sde_reset_counter( void *cntr_handle );
void *papi_sde_get_counter_handle( papi_handle_t handle, const char *event_name);

int papi_sde_compare_long_long(const void *p1, const void *p2);
int papi_sde_compare_int(const void *p1, const void *p2);
int papi_sde_compare_double(const void *p1, const void *p2);
int papi_sde_compare_float(const void *p1, const void *p2);

papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct);
#ifdef __cplusplus
}
#endif

#define POPULATE_SDE_FPTR_STRUCT( _A_ ) do{\
    _A_.init = papi_sde_init;\
    _A_.register_counter = papi_sde_register_counter;\
    _A_.register_fp_counter = papi_sde_register_fp_counter;\
    _A_.unregister_counter = papi_sde_unregister_counter;\
    _A_.describe_counter = papi_sde_describe_counter;\
    _A_.add_counter_to_group = papi_sde_add_counter_to_group;\
    _A_.create_counter = papi_sde_create_counter;\
    _A_.inc_counter = papi_sde_inc_counter;\
    _A_.create_recorder = papi_sde_create_recorder;\
    _A_.record = papi_sde_record;\
    _A_.reset_recorder = papi_sde_reset_recorder;\
    _A_.reset_counter = papi_sde_reset_counter;\
    _A_.get_counter_handle = papi_sde_get_counter_handle;\
}while(0)

////////////////////////////////////////////////////////////////////////////////

//// -- sde_common.h

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

////////////////////////////////////////////////////////////////////////////////

#if !defined(EXCLUDE_SDE_COMMON)
//// --- sde_common.c

__attribute__((visibility("hidden")))
int _sde_be_verbose = 0;

__attribute__((visibility("hidden")))
int _sde_debug = 0;

static papisde_library_desc_t *find_library_by_name(const char *library_name, papisde_control_t *gctl);
static void insert_library_handle(papisde_library_desc_t *lib_handle, papisde_control_t *gctl);

/*************************************************************************/
/* Functions related to internal hashing of events                       */
/*************************************************************************/

__attribute__((visibility("hidden")))
unsigned int ht_hash_id(unsigned int uniq_id){
    return uniq_id%PAPISDE_HT_SIZE;
}

// djb2 hash
__attribute__((visibility("hidden")))
unsigned long ht_hash_name(const char *str)
{
    unsigned long hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash % PAPISDE_HT_SIZE;
}

__attribute__((visibility("hidden")))
void ht_insert(papisde_list_entry_t *hash_table, int ht_key, sde_counter_t *sde_counter)
{
    papisde_list_entry_t *list_head, *new_entry;

    list_head = &hash_table[ht_key];
    // If we have no counter is associated with this key we will put the new
    // counter on the head of the list which has already been allocated.
    if( NULL == list_head->item ){
        list_head->item = sde_counter;
        list_head->next = NULL; // Just for aesthetic reasons.
        return;
    }

    // If we made it here it means that the head was occupied, so we
    // will allocate a new element and put it just after the head.
    new_entry = (papisde_list_entry_t *)calloc(1, sizeof(papisde_list_entry_t));
    new_entry->item = sde_counter;
    new_entry->next = list_head->next;
    list_head->next = new_entry;

    return;
}

__attribute__((visibility("hidden")))
sde_counter_t *ht_delete(papisde_list_entry_t *hash_table, int ht_key, unsigned int uniq_id)
{
    papisde_list_entry_t *list_head, *curr, *prev;
    sde_counter_t *item;

    list_head = &hash_table[ht_key];
    if( NULL == list_head->item ){
        SDE_ERROR("ht_delete(): the entry does not exist.\n");
        return NULL;
    }

    // If the head contains the element to be deleted, free the space of the counter and pull the list up.
    if( list_head->item->glb_uniq_id == uniq_id ){
        item = list_head->item;
        if( NULL != list_head->next)
            *list_head = *(list_head->next);
        return item;
    }

    prev = list_head;
    // Traverse the linked list to find the element.
    for(curr=list_head->next; NULL != curr; curr=curr->next){
        if(NULL == curr->item){ // This is only permitted for the head of the list.
            SDE_ERROR("ht_delete(): the hash table is clobbered.\n");
            return NULL;
        }
        if(curr->item->glb_uniq_id == uniq_id){
            prev->next = curr->next;
            item = curr->item;
            free(curr); // free the hash table entry
            return item;
        }
        prev = curr;
    }

    SDE_ERROR("ht_delete(): the item is not in the list.\n");
    return NULL;
}

__attribute__((visibility("hidden")))
sde_counter_t *ht_lookup_by_name(papisde_list_entry_t *hash_table, const char *name)
{
    papisde_list_entry_t *list_head, *curr;

    list_head = &hash_table[ht_hash_name(name)];
    if( NULL == list_head->item ){
        return NULL;
    }

    for(curr=list_head; NULL != curr; curr=curr->next){
        if(NULL == curr->item){ // This can only legally happen for the head of the list.
            SDE_ERROR("ht_lookup_by_name() the hash table is clobbered\n");
            return NULL;
        }
        if( !strcmp(curr->item->name, name) ){
            return curr->item;
        }
    }

    return NULL;
}

sde_counter_t *ht_lookup_by_id(papisde_list_entry_t *hash_table, unsigned int uniq_id)
{
    papisde_list_entry_t *list_head, *curr;

    list_head = &hash_table[ht_hash_id(uniq_id)];
    if( NULL == list_head->item ){
        return NULL;
    }

    for(curr=list_head; NULL != curr; curr=curr->next){
        if(NULL == curr->item){ // This can only legally happen for the head of the list.
            SDE_ERROR("ht_lookup_by_id() the hash table is clobbered\n");
            return NULL;
        }
        if(curr->item->glb_uniq_id == uniq_id){
            return curr->item;
        }
    }

    return NULL;
}


/*************************************************************************/
/* Utility Functions.                                                    */
/*************************************************************************/

/** This helper function checks to see if a given library has already been initialized and exists
    in the global structure of the component.
  @param[in] a pointer to the global structure.
  @param[in] a string containing the name of the library.
  @return a pointer to the library handle.
  */
papisde_library_desc_t *find_library_by_name(const char *library_name, papisde_control_t *gctl){

    if( (NULL == gctl) || (NULL == library_name) )
        return NULL;

    papisde_library_desc_t *tmp_lib = gctl->lib_list_head;
    // Check to see if this library has already been initialized.
    while(NULL != tmp_lib){
        char *tmp_name = tmp_lib->libraryName;
        SDEDBG("Checking library: '%s' against registered library: '%s'\n",library_name, tmp_lib->libraryName);
        // If we find the same library already registered, we do not create a new entry.
        if( (NULL != tmp_name) && !strcmp(tmp_name, library_name) )
            return tmp_lib;

        tmp_lib = tmp_lib->next;
    }

    return NULL;
}

/** This helper function simply adds a library handle to the beginning of the list of libraries
    in the global structure. It's only reason of existence is to hide the structure of the
    linked list in case we want to change it in the future.
  @param[in] a pointer to the library handle.
  @param[in] a pointer to the global structure.
  */
void insert_library_handle(papisde_library_desc_t *lib_handle, papisde_control_t *gctl){
    SDEDBG("insert_library_handle(): inserting new handle for library: '%s'\n",lib_handle->libraryName);
    lib_handle->next = gctl->lib_list_head;
    gctl->lib_list_head = lib_handle;

    return;
}


// Initialize library handle, or return the existing one if already
// initialized. This function is _not_ thread safe, so it needs to be called
// from within regions protected by papi_sde_lock()/papi_sde_unlock().
papi_handle_t do_sde_init(const char *name_of_library, papisde_control_t *gctl){

    papisde_library_desc_t *tmp_lib;

    SDEDBG("Registering library: '%s'\n",name_of_library);

    // If the library is already initialized, return the handle to it
    tmp_lib = find_library_by_name(name_of_library, gctl);
    if( NULL != tmp_lib ){
        return tmp_lib;
    }

    // If the library is not already initialized, then initialize it.
    tmp_lib = ( papisde_library_desc_t* ) calloc( 1, sizeof( papisde_library_desc_t ) );
    tmp_lib->libraryName = strdup(name_of_library);

    insert_library_handle(tmp_lib, gctl);

    return tmp_lib;
}

sde_counter_t *allocate_and_insert( papisde_control_t *gctl, papisde_library_desc_t* lib_handle, const char* name, unsigned int uniq_id, int cntr_mode, int cntr_type, void* data, papi_sde_fptr_t func_ptr, void *param ){

    // make sure to calloc() the structure, so all the fields which we do not explicitly set remain zero.
    sde_counter_t *item = (sde_counter_t *)calloc(1, sizeof(sde_counter_t));
    item->data = data;
    item->func_ptr = func_ptr;
    item->param = param;
    item->cntr_type = cntr_type;
    item->cntr_mode = cntr_mode;
    item->glb_uniq_id = uniq_id;
    item->name = strdup( name );
    item->description = strdup( name );
    item->which_lib = lib_handle;

    (void)ht_insert(lib_handle->lib_counters, ht_hash_name(name), item);
    (void)ht_insert(gctl->all_reg_counters, ht_hash_id(uniq_id), item);

    return item;
}

void recorder_data_to_contiguous(sde_counter_t *recorder, void *cont_buffer){
    long long current_size, typesize, used_entries, tmp_size = 0;
    void *src, *dst;
    int i;

    typesize = recorder->recorder_data->typesize;
    used_entries = recorder->recorder_data->used_entries;

    for(i=0; i<EXP_CONTAINER_ENTRIES; i++){
       current_size = ((long long)1<<i) * EXP_CONTAINER_MIN_SIZE;
       src = recorder->recorder_data->exp_container[i];
       dst = cont_buffer + tmp_size*typesize;
       if ( (tmp_size+current_size) <= used_entries){
           memcpy(dst, src, current_size*typesize);
           if ( (tmp_size+current_size) == used_entries){
               return;
           }
       }else{
           memcpy(dst, src, (used_entries-tmp_size)*typesize);
           return;
       }
       tmp_size += current_size;
    }
}

#endif // !defined(EXCLUDE_SDE_COMMON)

////////////////////////////////////////////////////////////////////////////////

#if !defined(SDE_COMMON_ONLY)
//// --- sde_lib.c

#if !defined(_GNU_SOURCE)
  #define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <assert.h>

// The following values have been defined such that they match the
// corresponding PAPI values from papi.h
#define SDE_OK          0     /**< No error */
#define SDE_EINVAL     -1     /**< Invalid argument */
#define SDE_ECMP       -4     /**< Not supported by component */

static int sde_setup_counter_internals( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter, papi_sde_fptr_t fp_counter, void *param, sde_counter_t **placeholder );
static inline int sde_do_register( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter, papi_sde_fptr_t fp_counter, void *param );
static int delete_counter(papisde_library_desc_t* lib_handle, const char *name);
static inline void free_counter(sde_counter_t *counter);

static long long _sde_compute_q1(void *param);
static long long _sde_compute_med(void *param);
static long long _sde_compute_q3(void *param);
static long long _sde_compute_min(void *param);
static long long _sde_compute_max(void *param);
static inline long long _sde_compute_quantile(void *param, int percent);
static inline long long _sde_compute_edge(void *param, int which_edge);

int papi_sde_compare_long_long(const void *p1, const void *p2);
int papi_sde_compare_int(const void *p1, const void *p2);
int papi_sde_compare_double(const void *p1, const void *p2);
int papi_sde_compare_float(const void *p1, const void *p2);

/******************************************************************/
/* Prototypes for functions that we expect to access from libpapi */
/******************************************************************/

#if defined(STATIC_SDE)
papisde_control_t *papisde_get_global_struct(void);
int papi_sde_lock(void);
int papi_sde_unlock(void);
#if defined(SDE_HAVE_OVERFLOW)
void papi_sde_check_overflow_status(sde_counter_t *cntr_handle, long long int latest);
int papi_sde_set_timer_for_overflow(void);
#endif // defined(SDE_HAVE_OVERFLOW)
#endif // defined(STATIC_SDE)

/****************************************************************/
/* Pointers for functions that we expect to access from libpapi */
/****************************************************************/

papisde_control_t *(*get_struct_sym)(void);
int (*papi_sde_lock_sym)(void);
int (*papi_sde_unlock_sym)(void);
#if defined(SDE_HAVE_OVERFLOW)
void (*papi_sde_check_overflow_status_sym)(sde_counter_t *hndl, long long int value);
int  (*papi_sde_set_timer_for_overflow_sym)(void);
#endif // SDE_HAVE_OVERFLOW

/***************************************************************************/
/* Encapsulate the difference between dynamic/static build in functions to */
/* keep the rest of the code clean.                                        */
/***************************************************************************/

static inline papisde_control_t *
_get_global_struct(void){
#if defined(STATIC_SDE)
    return papisde_get_global_struct();
#else
    if( NULL == get_struct_sym )
        return NULL;
    return (*get_struct_sym)();
#endif // defined(STATIC_SDE)
}

static inline int
_sde_lock(){
#if defined(STATIC_SDE)
    return papi_sde_lock();
#else
    if( papi_sde_lock_sym )
        return (*papi_sde_lock_sym )();
    return -1;
#endif // defined(STATIC_SDE)
}

static inline int
_sde_unlock(){
#if defined(STATIC_SDE)
    return papi_sde_unlock();
#else
    if( papi_sde_unlock_sym )
        return (*papi_sde_unlock_sym )();
    return -1;
#endif // defined(STATIC_SDE)
}

#if defined(SDE_HAVE_OVERFLOW)
static inline void
_sde_check_overflow_status(sde_counter_t *cntr_handle, long long int latest){
#if defined(STATIC_SDE)
    papi_sde_check_overflow_status(cntr_handle, latest);
#else
    if( NULL != papi_sde_check_overflow_status_sym )
        (*papi_sde_check_overflow_status_sym)(cntr_handle, latest);
#endif // defined(STATIC_SDE)
}

static inline int
_sde_set_timer_for_overflow(void){
#if defined(STATIC_SDE)
    return papi_sde_set_timer_for_overflow();
#else
    if( NULL != papi_sde_set_timer_for_overflow_sym )
        return (*papi_sde_set_timer_for_overflow_sym)();
    return -1;
#endif // defined(STATIC_SDE)
}
#endif //defined(SDE_HAVE_OVERFLOW)

#if !defined(STATIC_SDE)
/*
  If the library is being built statically then there is no need (or ability)
  to access symbols through dlopen/dlsym; applications using the static version
  of this library (libXYZ.a) must also be linked against libpapi, otherwise
  linking will fail. However, if the library is being built into a dynamic
  object (libXYZ.so) then we will look for PAPI's symbols dynamically.
*/
void obtain_papi_symbols(void){
    char *err;
    int dlsym_err = 0;

    (void)dlerror(); // Clear the internal string so we can diagnose errors later on.

    void *handle = dlopen(NULL, RTLD_NOW|RTLD_GLOBAL);
    if( NULL != (err = dlerror()) ){
        SDEDBG("obtain_papi_symbols(): %s\n",err);
        dlsym_err = 1;
        return;
    }

    // This function will give us the global structure that libpapi and libsde
    // will use to store and exchange information about SDEs.
    get_struct_sym = dlsym(handle, "papisde_get_global_struct");
    if( (NULL != (err = dlerror())) || (NULL == get_struct_sym) ){
        SDEDBG("obtain_papi_symbols(): Unable to find symbols from libpapi.so. SDEs will not be accessible by external software. %s\n",err);
        dlsym_err = 1;
        return;
    }

    // We need this function to guarantee thread safety between the threads
    // that change the value of SDEs and the threads calling PAPI to read them.
    papi_sde_lock_sym = dlsym(handle, "papi_sde_lock");
    if( (NULL != (err = dlerror())) || (NULL == papi_sde_lock_sym) ){
        SDEDBG("obtain_papi_symbols(): Unable to find libpapi.so function needed for thread safety. %s\n",err);
        dlsym_err = 1;
    }

    // We need this function to guarantee thread safety between the threads
    // that change the value of SDEs and the threads calling PAPI to read them.
    papi_sde_unlock_sym = dlsym(handle, "papi_sde_unlock");
    if( (NULL != (err = dlerror())) || (NULL == papi_sde_unlock_sym) ){
        SDEDBG("obtain_papi_symbols(): Unable to find libpapi.so function needed for thread safety. %s\n",err);
        dlsym_err = 1;
    }

    // We need this function to inform the SDE component about the value of created counters.
#if defined(SDE_HAVE_OVERFLOW)
    papi_sde_check_overflow_status_sym = dlsym(handle, "papi_sde_check_overflow_status");
    if( (NULL != (err = dlerror())) || (NULL == papi_sde_check_overflow_status_sym) ){
        SDEDBG("obtain_papi_symbols(): Unable to find libpapi.so function needed to support overflowing of SDEs. %s\n",err);
        dlsym_err = 1;
    }

    papi_sde_set_timer_for_overflow_sym = dlsym(handle, "papi_sde_set_timer_for_overflow");
    if( (NULL != (err = dlerror())) || (NULL == papi_sde_set_timer_for_overflow_sym) ){
        SDEDBG("obtain_papi_symbols(): Unable to find libpapi.so function needed to support overflowing of SDEs. %s\n",err);
        dlsym_err = 1;
    }
#endif // SDE_HAVE_OVERFLOW

    if( !dlsym_err ){
        SDEDBG("obtain_papi_symbols(): All symbols from libpapi.so have been successfully acquired.\n");
    }

    return;
}
#endif // !defined(STATIC_SDE)


/*************************************************************************/
/* Functions related to handling memory related to SDE structures.       */
/*************************************************************************/

static inline void free_counter(sde_counter_t *counter)
{
    int i;

    if( NULL == counter )
        return;

    free(counter->name);
    free(counter->description);

    // If we are dealing with a recorder we need to free all the data associated with it.
    if( NULL != counter->recorder_data ){
        if( NULL != counter->recorder_data->sorted_buffer ){
            free( counter->recorder_data->sorted_buffer );
        }
        for(i=0; i<EXP_CONTAINER_ENTRIES; i++){
           if( NULL != counter->recorder_data->exp_container[i] ){
               free( counter->recorder_data->exp_container[i] );
           }
        }
        free(counter->recorder_data);
    }

    // We are dealing with a counter whose 'data' field was
    // allocated by us, not the library, so we need to free it.
    if( counter->is_created ){
        free(counter->data);
    }

    free(counter);
}

static int delete_counter(papisde_library_desc_t* lib_handle, const char* name)
{

    sde_counter_t *tmp_item;
    papisde_control_t *gctl;
    unsigned int item_uniq_id;

    gctl = _get_global_struct();
    if( NULL == gctl ){
        return 1;
    }

    // Look for the counter entry in the hash-table of the library
    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, name);
    if( NULL == tmp_item )
        return 1;

    item_uniq_id = tmp_item->glb_uniq_id;

    // Delete the entry from the library hash-table (which hashes by name)
    tmp_item = ht_delete(lib_handle->lib_counters, ht_hash_name(name), item_uniq_id);
    if( NULL == tmp_item ){
        return 1;
    }

    // Delete the entry from the global hash-table (which hashes by id) and free the memory
    // occupied by the counter (not the hash-table entry 'papisde_list_entry_t', the 'sde_counter_t')
    tmp_item = ht_delete(gctl->all_reg_counters, ht_hash_id(item_uniq_id), item_uniq_id);
    if( NULL == tmp_item ){
        return 1;
    }

    // We free the counter only once, although it is in two hash-tables,
    // because it is the same structure that is pointed to by both hash-tables.
    free_counter(tmp_item);

    // Decrement the number of live events.
    gctl->num_live_events--;

    return 0;
}


/*************************************************************************/
/* API Functions.                                                        */
/*************************************************************************/

/** This function creates the SDE component structure for an individual
  software library and returns a handle to the structure.
  @param[in] name_of_library -- (const char *) library name.
  @param[out] sde_handle -- (papi_handle_t) opaque pointer to sde structure for initialized library.
  */
papi_handle_t
__attribute__((visibility("hidden")))
papi_sde_init(const char *name_of_library)
{
    papisde_library_desc_t *tmp_lib;

    // We have to emulate PAPI's SUBDBG to get the same behavior
    _sde_be_verbose = (NULL != getenv("PAPI_VERBOSE"));
    char *tmp= getenv("PAPI_DEBUG");
    if( (NULL != tmp) && (0 != strlen(tmp)) && strstr(tmp, "SUBSTRATE") )
        _sde_debug = 1;

    SDEDBG("Registering library: '%s'\n",name_of_library);

#if !defined(STATIC_SDE)
    obtain_papi_symbols();
#endif // !defined(STATIC_SDE)

    // This function will give us the global structure that libpapi and libsde
    // will use to store and exchange information about SDEs.
    papisde_control_t *gctl = _get_global_struct();
    if( NULL == gctl ){
        SDEDBG("papi_sde_init(): Unable to find symbols from libpapi.so. SDEs will not be accessible by external software.\n");
        return NULL;
    }

    // Lock before we read and/or modify the global structures.
    _sde_lock();

    // Put the actual work in a different function so we call it from other
    // places in the component.  We have to do this because we cannot call
    // papi_sde_init() from places in the code which already call
    // PAPI_lock()/PAPI_unlock(), or we will end up with deadlocks.
    tmp_lib = do_sde_init(name_of_library, gctl);

    _sde_unlock();

    SDEDBG("Library '%s' has been registered.\n",name_of_library);

    return tmp_lib;
}

/** This function registers an event name and counter within the SDE component
  structure attached to the handle. A default description for an event is
  synthesized from the library name and the event name when they are registered.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event.
  @param[in] cntr_mode -- (int) the mode of the counter (one of: PAPI_SDE_RO, PAPI_SDE_RW and one of: PAPI_SDE_DELTA, PAPI_SDE_INSTANT).
  @param[in] cntr_type -- (int) the type of the counter (PAPI_SDE_long_long, PAPI_SDE_int, PAPI_SDE_double, PAPI_SDE_float).
  @param[in] counter -- pointer to a variable that stores the value for the event.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
__attribute__((visibility("hidden")))
papi_sde_register_counter( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter )
{
    int ret_val;

    // if libpapi.so was not linked in with the application, the handle will be NULL, and that's ok.
    if( !handle ) return SDE_OK;

    _sde_lock();
    ret_val = sde_do_register(handle, event_name, cntr_mode, cntr_type, counter, NULL, NULL);
    _sde_unlock();

    return ret_val;
}

/** This function registers an event name and (caller provided) callback function
  within the SDE component structure attached to the handle.
  A default description for an event is
  synthesized from the library name and the event name when they are registered.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event.
  @param[in] cntr_mode -- (int) the mode of the counter (one of: PAPI_SDE_RO, PAPI_SDE_RW and one of: PAPI_SDE_DELTA, PAPI_SDE_INSTANT).
  @param[in] cntr_type -- (int) the type of the counter (PAPI_SDE_long_long, PAPI_SDE_int, PAPI_SDE_double, PAPI_SDE_float).
  @param[in] fp_counter -- pointer to a callback function that SDE will call when PAPI_read/stop/accum is called.
  @param[in] param -- (void *) opaque parameter that will be passed to the callback function every time it's called.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
__attribute__((visibility("hidden")))
papi_sde_register_fp_counter( void *handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t fp_counter, void *param )
{
    int ret_val;

    // if libpapi.so was not linked in with the application, the handle will be NULL, and that's ok.
    if( !handle ) return SDE_OK;

    _sde_lock();
    ret_val = sde_do_register( handle, event_name, cntr_mode, cntr_type, NULL, fp_counter, param );
    _sde_unlock();

    return ret_val;
}

/** This function unregisters (removes) an event name and counter from the SDE component.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event that is being unregistered.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
__attribute__((visibility("hidden")))
papi_sde_unregister_counter( void *handle, const char *event_name)
{
    papisde_library_desc_t *lib_handle;
    int error;
    char *full_event_name;

    // if libpapi.so was not linked in with the application, the handle will be NULL, and that's ok.
    if( !handle ) return SDE_OK;

    lib_handle = (papisde_library_desc_t *) handle;
    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_unregister_counter(): 'handle' is clobbered. Unable to unregister counter.\n");
        return SDE_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    SDEDBG("papi_sde_unregister_counter(): Preparing to unregister counter: '%s' from SDE library: %s.\n", full_event_name, lib_handle->libraryName);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    _sde_lock();

    error = delete_counter( lib_handle, full_event_name );
    // Check if we found a registered counter, or if it never existed.
    if( error ){
        SDE_ERROR("papi_sde_unregister_counter(): Counter '%s' has not been registered by library '%s'.\n", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        _sde_unlock();
        return SDE_EINVAL;
    }

    // We will not use the name beyond this point
    free(full_event_name);

    _sde_unlock();
    return SDE_OK;
}


/** This function optionally replaces an event's default description with a
  description provided by the library developer within the SDE component
  structure attached to the handle.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event.
  @param[in] event_description -- (const char *) description of the event.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
__attribute__((visibility("hidden")))
papi_sde_describe_counter( void *handle, const char *event_name, const char *event_description )
{
    sde_counter_t *tmp_item;
    papisde_library_desc_t *lib_handle;
    char *full_event_name;

    // if libpapi.so was not linked in with the application, the handle will be NULL, and that's ok.
    if( !handle ) return SDE_OK;

    lib_handle = (papisde_library_desc_t *) handle;
    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_describe_counter(): 'handle' is clobbered. Unable to add description for counter.\n");
        return SDE_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    _sde_lock();

    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if( NULL != tmp_item ){
        tmp_item->description = strdup(event_description);
        free(full_event_name);
        _sde_unlock();
        return SDE_OK;
    }
    SDEDBG("papi_sde_describe_counter() Event: '%s' is not registered in SDE library: '%s'\n", full_event_name, lib_handle->libraryName);
    // We will not use the name beyond this point
    free(full_event_name);
    _sde_unlock();
    return SDE_EINVAL;
}



/** This function adds an event counter to a group. A group is created automatically
    the first time a counter is added to it.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] group_name -- (const char *) name of the group.
  @param[in] group_flags -- (uint32_t) one of PAPI_SDE_SUM, PAPI_SDE_MAX, PAPI_SDE_MIN to define how the members of the group will be used to compute the group's value.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
__attribute__((visibility("hidden")))
papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags)
{
    papisde_library_desc_t *lib_handle;
    sde_counter_t *tmp_item, *tmp_group;
    unsigned int cntr_group_uniq_id;
    char *full_event_name, *full_group_name;

    // if libpapi.so was not linked in with the application, the handle will be NULL, and that's ok.
    if( !handle ) return SDE_OK;

    SDEDBG("papi_sde_add_counter_to_group(): Adding counter: %s into group %s\n",event_name, group_name);

    lib_handle = (papisde_library_desc_t *) handle;
    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_add_counter_to_group(): 'handle' is clobbered. Unable to add counter to group.\n");
        return SDE_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    _sde_lock();

    // Check to make sure that the event is already registered. This is not the place to create a placeholder.
    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if( NULL == tmp_item ){
        _sde_unlock();
        SDE_ERROR("papi_sde_add_counter_to_group(): Unable to find counter: '%s'.\n",full_event_name);
        free(full_event_name);
        return SDE_EINVAL;
    }

    // We will not use the name beyond this point
    free(full_event_name);

    str_len = strlen(lib_handle->libraryName)+strlen(group_name)+2+1; // +2 for "::" and +1 for '\0'
    full_group_name = malloc(str_len*sizeof(char));
    snprintf(full_group_name, str_len, "%s::%s", lib_handle->libraryName, group_name);

    // Check to see if the group exists already. Otherwise we need to create it.
    tmp_group = ht_lookup_by_name(lib_handle->lib_counters, full_group_name);
    if( NULL == tmp_group ){

        papisde_control_t *gctl = _get_global_struct();
        if( NULL == gctl ){
            return SDE_EINVAL;
        }

        // We use the current number of registered events as the uniq id of the counter group, and we
        // increment it because counter groups are treated as real counters by the outside world.
        // They are first class citizens.
        cntr_group_uniq_id = gctl->num_reg_events++;
        gctl->num_live_events++;
//        _sde_vector.cmp_info.num_native_events = gctl->num_live_events;

        SDEDBG("%s line %d: Unique ID for new counter group = %d\n", __FILE__, __LINE__, cntr_group_uniq_id);

        tmp_group = (sde_counter_t *)calloc(1, sizeof(sde_counter_t));
        tmp_group->glb_uniq_id = cntr_group_uniq_id;
        // copy the name because we will free the malloced space further down in this function.
        tmp_group->name = strdup(full_group_name);
        // make a copy here, because we will free() the 'name' and the 'description' separately.
        tmp_group->description = strdup( full_group_name );
        tmp_group->which_lib = lib_handle;
        tmp_group->counter_group_flags = group_flags;
        // Be explicit so that people reading the code can spot the initialization easier.
        tmp_group->data = NULL;
        tmp_group->func_ptr = NULL;
        tmp_group->param = NULL;
        tmp_group->counter_group_head = NULL;

        (void)ht_insert(lib_handle->lib_counters, ht_hash_name(full_group_name), tmp_group);
        (void)ht_insert(gctl->all_reg_counters, ht_hash_id(cntr_group_uniq_id), tmp_group);

    }else{
        // should the following branch ever be true? Why do we already have a group registered if it's empty?
        if( NULL == tmp_group->counter_group_head ){
            SDE_ERROR("papi_sde_add_counter_to_group(): Found an empty counter group: '%s'. This might indicate that a cleanup routine is not doing its job.\n", group_name);
        }

        // make sure the caller is not trying to change the flags of the group after it has been created.
        if( tmp_group->counter_group_flags != group_flags ){
            _sde_unlock();
            SDE_ERROR("papi_sde_add_counter_to_group(): Attempting to add counter '%s' to counter group '%s' with incompatible group flags.\n", event_name, group_name);
            free(full_group_name);
            return SDE_EINVAL;
        }
    }

    // Add the new counter to the group's head.
    papisde_list_entry_t *new_head = calloc(1, sizeof(papisde_list_entry_t));
    new_head->item = tmp_item;
    new_head->next = tmp_group->counter_group_head;
    tmp_group->counter_group_head = new_head;

    _sde_unlock();
    free(full_group_name);
    return SDE_OK;
}

/**

  This function creates a counter whose memory is allocated and managed by libsde,
  in contrast with papi_sde_register_counter(), which works with counters that are managed
  by the user library that is calling this function.
  This counter can only by modified via the functions papi_sde_inc_counter()
  and papi_sde_reset_counter(). This has two benefits over a counter which
  lives inside the user library and is modified directly by that library:
  A) Our counter and the modifying API is guaranteed to be thread safe.
  B) Since libsde knows about each change in the value of the counter,
     overflowing is accurate.
  However, this approach has higher overhead than executing "my_cntr += value" inside
  a user library.

  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event.
  @param[in] cntr_mode -- (int) the mode of the counter (one of: PAPI_SDE_RO, PAPI_SDE_RW and one of: PAPI_SDE_DELTA, PAPI_SDE_INSTANT).
  @param[out] cntr_handle -- address of a pointer in which libsde will store a handle to the newly created counter.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
*/
int
__attribute__((visibility("hidden")))
papi_sde_create_counter( papi_handle_t handle, const char *event_name, int cntr_mode, void **cntr_handle )
{
    int ret_val;
    long long int *counter_data;
    char *full_event_name;
    papisde_library_desc_t *lib_handle;
    sde_counter_t *cntr, *placeholder;

    // if libpapi.so was not linked in with the application, the handle will be NULL, and that's ok.
    if( !handle ) return SDE_OK;

    lib_handle = (papisde_library_desc_t *) handle;
    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_create_counter(): 'handle' is clobbered. Unable to create counter.\n");
        return SDE_EINVAL;
    }

    SDEDBG("Preparing to create counter: '%s' with mode: '%d' in SDE library: %s.\n", event_name, cntr_mode, lib_handle->libraryName);

    counter_data = calloc(1, sizeof(long long int));

    ret_val = sde_setup_counter_internals( lib_handle, event_name, cntr_mode, PAPI_SDE_long_long, counter_data, NULL, NULL, &placeholder );
    if( SDE_OK != ret_val ){
        return ret_val;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    cntr = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if(NULL == cntr) {
        SDEDBG("Logging counter '%s' not properly inserted in SDE library '%s'\n", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        return SDE_ECMP;
    }

    // Signify that this counter is a created counter (as opposed to a registered one).
    // The reason we need to know is so we can free() the 'data' entry which we allocated here, and for
    // correctness checking in papi_sde_inc_counter() and papi_sde_reset_counter().
    cntr->is_created = 1;

    if( NULL != cntr_handle ){
        *(sde_counter_t **)cntr_handle = cntr;
    }

    free(full_event_name);

    return SDE_OK;
}


// The following function works only for counters created using papi_sde_create_counter().
int
__attribute__((visibility("hidden")))
papi_sde_inc_counter( papi_handle_t cntr_handle, long long int increment)
{
    long long int *ptr;
    sde_counter_t *tmp_cntr;

    _sde_lock();

    tmp_cntr = (sde_counter_t *)cntr_handle;
    if( NULL == tmp_cntr ){
        _sde_unlock();
        SDE_ERROR("papi_sde_inc_counter(): 'cntr_handle' is clobbered. Unable to modify value of counter.\n");
        return SDE_EINVAL;
    }

    SDEDBG("Preparing to increment counter: '%s::%s' by %lld.\n", tmp_cntr->which_lib->libraryName, tmp_cntr->name, increment);

    ptr = (long long int *)(tmp_cntr->data);

    if( NULL == ptr ){
        _sde_unlock();
        SDE_ERROR("papi_sde_inc_counter(): Counter structure is clobbered. Unable to modify value of counter.\n");
        return SDE_EINVAL;
    }

    if( !tmp_cntr->is_created ){
        _sde_unlock();
        SDE_ERROR("papi_sde_inc_counter(): Counter is not created by PAPI, cannot be modified using this function.\n");
        return SDE_EINVAL;
    }

    if( PAPI_SDE_long_long != tmp_cntr->cntr_type ){
        _sde_unlock();
        SDE_ERROR("papi_sde_inc_counter(): Counter is not of type \"long long int\" and cannot be modified using this function.\n");
        return SDE_EINVAL;
    }

    *ptr += increment;

#if defined(SDE_HAVE_OVERFLOW)
    _sde_check_overflow_status(tmp_cntr, *ptr);
#endif // SDE_HAVE_OVERFLOW

    _sde_unlock();

    return SDE_OK;
}


int
__attribute__((visibility("hidden")))
papi_sde_create_recorder( papi_handle_t handle, const char *event_name, size_t typesize, int (*cmpr_func_ptr)(const void *p1, const void *p2), void **record_handle )
{

    int ret_val, i;
    sde_counter_t *tmp_rec_handle;
    char *aux_event_name;
    size_t str_len;
    char *full_event_name;
#define _SDE_MODIFIER_COUNT 6
    const char *modifiers[_SDE_MODIFIER_COUNT] = {":CNT",":MIN",":Q1",":MED",":Q3",":MAX"};
    // Add a NULL pointer for symmetry with the 'modifiers' vector, since the modifier ':CNT' does not have a function pointer.
    long long (*func_ptr_vec[_SDE_MODIFIER_COUNT])(void *) = {NULL, _sde_compute_min, _sde_compute_q1, _sde_compute_med, _sde_compute_q3, _sde_compute_max};
    long long total_entries = (long long)EXP_CONTAINER_MIN_SIZE;

    // if libpapi.so was not linked in with the application, the handle will be NULL, and that's ok.
    if( !handle ) return SDE_OK;

    papisde_library_desc_t *lib_handle = handle;

    _sde_lock();

    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        SDE_ERROR("papi_sde_create_recorder(): 'handle' is clobbered. Unable to create recorder.\n");
        _sde_unlock();
        return SDE_EINVAL;
    }

    SDEDBG("Preparing to create recorder: '%s' with typesize: '%d' in SDE library: %s.\n", event_name, (int)typesize, lib_handle->libraryName);

    // We setup the recorder like this, instead of using sde_do_register() because recorders cannot be set to overflow.
    ret_val = sde_setup_counter_internals( lib_handle, event_name, PAPI_SDE_DELTA|PAPI_SDE_RO, PAPI_SDE_long_long, NULL, NULL, NULL, NULL );
    if( SDE_OK != ret_val )
        return ret_val;

    str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    tmp_rec_handle = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if(NULL == tmp_rec_handle) {
        SDEDBG("Recorder '%s' not properly inserted in SDE library '%s'\n", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        _sde_unlock();
        return SDE_ECMP;
    }

    // Allocate the structure for the recorder data and meta-data.
    tmp_rec_handle->recorder_data = calloc(1,sizeof(recorder_data_t));
    // Allocate the first chunk of recorder data.
    tmp_rec_handle->recorder_data->exp_container[0] = malloc(total_entries*typesize);
    tmp_rec_handle->recorder_data->total_entries = total_entries;
    tmp_rec_handle->recorder_data->typesize = typesize;
    tmp_rec_handle->recorder_data->used_entries = 0;

    *(sde_counter_t **)record_handle = tmp_rec_handle;

    // We will not use the name beyond this point
    free(full_event_name);

    // At this point we are done creating the recorder and we will create the additional events which will appear as modifiers of the recorder.
    str_len = 0;
    for(i=0; i<_SDE_MODIFIER_COUNT; i++){
        size_t tmp_len = strlen(modifiers[i]);
        if( tmp_len > str_len )
            str_len = tmp_len;
    }
    str_len += strlen(event_name)+1;
    aux_event_name = calloc(str_len, sizeof(char));

    snprintf(aux_event_name, str_len, "%s%s", event_name, modifiers[0]);
    SDEDBG("papi_sde_create_recorder(): Preparing to register aux counter: '%s' in SDE library: %s.\n", aux_event_name, lib_handle->libraryName);

    // The :CNT aux counter is properly registered so that it can be set to overflow.
    ret_val = sde_do_register( lib_handle, (const char *)aux_event_name, PAPI_SDE_INSTANT|PAPI_SDE_RO, PAPI_SDE_long_long, &(tmp_rec_handle->recorder_data->used_entries), NULL, NULL );
    if( SDE_OK != ret_val ){
        SDEDBG("papi_sde_create_recorder(): Registration of aux counter: '%s' in SDE library: %s FAILED.\n", aux_event_name, lib_handle->libraryName);
        _sde_unlock();
        free(aux_event_name);
        return ret_val;
    }

    // If the caller passed NULL as the function pointer, then they do _not_ want the quantiles. Otherwise, create them.
    if( NULL != cmpr_func_ptr ){
        for(i=1; i<_SDE_MODIFIER_COUNT; i++){
            sde_sorting_params_t *sorting_params;

            sorting_params = malloc(sizeof(sde_sorting_params_t)); // This will be free()-ed by papi_sde_unregister_counter()
            sorting_params->recording = tmp_rec_handle;
            sorting_params->cmpr_func_ptr = cmpr_func_ptr;

            snprintf(aux_event_name, str_len, "%s%s", event_name, modifiers[i]);

            SDEDBG("papi_sde_create_recorder(): Preparing to register aux fp counter: '%s' in SDE library: %s.\n", aux_event_name, lib_handle->libraryName);
            ret_val = sde_do_register(lib_handle, (const char *)aux_event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_long_long, NULL, func_ptr_vec[i], sorting_params );
            if( SDE_OK != ret_val ){
                SDEDBG("papi_sde_create_recorder(): Registration of aux counter: '%s' in SDE library: %s FAILED.\n", aux_event_name, lib_handle->libraryName);
                _sde_unlock();
                free(aux_event_name);
                return ret_val;
            }
        }
    }

    _sde_unlock();
    free(aux_event_name);
    return SDE_OK;
}


// UPDATED for EXP-storage
int
__attribute__((visibility("hidden")))
papi_sde_record( void *record_handle, size_t typesize, void *value)
{
    sde_counter_t *tmp_item;
    long long used_entries, total_entries, prev_entries, offset;
    int i, chunk;
    long long tmp_size;

    SDEDBG("Preparing to record value of size %lu at address: %p\n",typesize, value);

    _sde_lock();

    tmp_item = (sde_counter_t *)record_handle;

    if( NULL == tmp_item ){
        _sde_unlock();
        SDE_ERROR("papi_sde_record(): 'record_handle' is clobbered. Unable to record value.\n");
        return SDE_EINVAL;
    }

    if( NULL == tmp_item->recorder_data || NULL == tmp_item->recorder_data->exp_container[0]){
        _sde_unlock();
        SDE_ERROR("papi_sde_record(): Counter structure is clobbered. Unable to record event.\n");
        return SDE_EINVAL;
    }

    // At this point the recorder exists, but we must check if it has room for more elements

    used_entries = tmp_item->recorder_data->used_entries;
    total_entries = tmp_item->recorder_data->total_entries;
    assert(used_entries <= total_entries);

    // Find how many chunks we have already allocated
    tmp_size = 0;
    for(i=0; i<EXP_CONTAINER_ENTRIES; i++){
       long long factor = (long long)1<<i; // 2^i;
       prev_entries = tmp_size;
       tmp_size += factor * EXP_CONTAINER_MIN_SIZE;
       // At least the first chunk "tmp_item->recorder_data->exp_container[0]"
       // must have been already allocated when creating the recorder, so we can
       // compare the total size after we add the "i-th" size.
       if (total_entries == tmp_size)
           break;
    }
    chunk = i;

    // Find how many entries down the last chunk we are.
    offset = used_entries - prev_entries;

    if( used_entries == total_entries ){
        long long new_segment_size;

        // If we had used all the available entries (and thus we are allocating more), we start from the beginning of the new chunk.
        offset = 0;

        chunk += 1; // we need to allocate the next chunk from the last one we found.
        new_segment_size = ((long long)1<<chunk) * EXP_CONTAINER_MIN_SIZE;
        tmp_item->recorder_data->exp_container[chunk] = malloc(new_segment_size*typesize);
        tmp_item->recorder_data->total_entries += new_segment_size;
    }

    void *dest = tmp_item->recorder_data->exp_container[chunk] + offset*typesize;
    (void)memcpy( dest, value, typesize );
    tmp_item->recorder_data->used_entries++;

    _sde_unlock();
    return SDE_OK;
}



// This function neither frees the allocated, nor does it zero it. It only resets the counter of used entries so that
// the allocated space can be resused (and overwritten) by future calls to record().
int
__attribute__((visibility("hidden")))
papi_sde_reset_recorder( void *record_handle )
{
    sde_counter_t *tmp_rcrdr;

    _sde_lock();
    tmp_rcrdr = (sde_counter_t *)record_handle;

    if( NULL == tmp_rcrdr || NULL == tmp_rcrdr->recorder_data ){
        _sde_unlock();
        SDE_ERROR("papi_sde_record(): 'record_handle' is clobbered. Unable to reset recorder.\n");
        return SDE_EINVAL;
    }

    // NOTE: do _not_ free the chunks and do _not_ reset "recorder_data->total_entries"

    tmp_rcrdr->recorder_data->used_entries = 0;
    free( tmp_rcrdr->recorder_data->sorted_buffer );
    tmp_rcrdr->recorder_data->sorted_buffer = NULL;
    tmp_rcrdr->recorder_data->sorted_entries = 0;

    _sde_unlock();
    return SDE_OK;
}


// The following function works only for counters created using papi_sde_create_counter().
int
__attribute__((visibility("hidden")))
papi_sde_reset_counter( void *cntr_handle )
{
    long long int *ptr;
    sde_counter_t *tmp_cntr;

    _sde_lock();

    tmp_cntr = (sde_counter_t *)cntr_handle;

    if( NULL == tmp_cntr ){
        _sde_unlock();
        SDE_ERROR("papi_sde_reset_counter(): 'cntr_handle' is clobbered. Unable to reset value of counter.\n");
        return SDE_EINVAL;
    }

    ptr = (long long int *)(tmp_cntr->data);

    if( NULL == ptr ){
        _sde_unlock();
        SDE_ERROR("papi_sde_reset_counter(): Counter structure is clobbered. Unable to reset value of counter.\n");
        return SDE_EINVAL;
    }

    if( tmp_cntr->is_created ){
        _sde_unlock();
        SDE_ERROR("papi_sde_reset_counter(): Counter is not created by PAPI, so it cannot be reset.\n");
        return SDE_EINVAL;
    }

    *ptr = 0; // Reset the counter.

    _sde_unlock();

    return SDE_OK;
}


/*************************************************************************/
/* Utility Functions.                                                    */
/*************************************************************************/

static inline int sde_do_register( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter, papi_sde_fptr_t fp_counter, void *param )
{
    sde_counter_t *placeholder;
    int ret;

    SDEDBG("%s: Preparing to register counter: '%s' with mode: '%d' and type: '%d'.\n", __FILE__, event_name, cntr_mode, cntr_type);

    ret = sde_setup_counter_internals( handle, event_name, cntr_mode, cntr_type, counter, fp_counter, param, &placeholder );

    if( SDE_OK != ret )
        return ret;

#if defined(SDE_HAVE_OVERFLOW)
    // Check if we need to worry about overflow (cases r[4-6], or c[4-6]).
    // However the function we are in (sde_do_register()) is only called for
    // registered (and not created) counters, so we know we are in cases r[4-6].
    if( NULL != placeholder && placeholder->overflow ){
        ret = _sde_set_timer_for_overflow();
    }
#endif // defined(SDE_HAVE_OVERFLOW)

    return ret;
}


int
__attribute__((visibility("hidden")))
papi_sde_compare_long_long(const void *p1, const void *p2){
    long long n1, n2;
    n1 = *(long long *)p1;
    n2 = *(long long *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int
__attribute__((visibility("hidden")))
papi_sde_compare_int(const void *p1, const void *p2){
    int n1, n2;
    n1 = *(int *)p1;
    n2 = *(int *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int
__attribute__((visibility("hidden")))
papi_sde_compare_double(const void *p1, const void *p2){
    double n1, n2;
    n1 = *(double *)p1;
    n2 = *(double *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int
__attribute__((visibility("hidden")))
papi_sde_compare_float(const void *p1, const void *p2){
    float n1, n2;
    n1 = *(float *)p1;
    n2 = *(float *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

#define _SDE_CMP_MIN 0
#define _SDE_CMP_MAX 1

// This function returns a "long long" which contains a pointer to the
// data element that corresponds to the edge (min/max), so that it works
// for all types of data, not only integers.
static inline long long _sde_compute_edge(void *param, int which_edge){
	void *edge = NULL, *edge_copy;
    long long elem_cnt;
    long long current_size, cumul_size = 0;
    void *src;
    int i, chunk;
    size_t typesize;
    sde_counter_t *rcrd;
    int (*cmpr_func_ptr)(const void *p1, const void *p2);


    rcrd = ((sde_sorting_params_t *)param)->recording;
    elem_cnt = rcrd->recorder_data->used_entries;
    typesize = rcrd->recorder_data->typesize;

    cmpr_func_ptr = ((sde_sorting_params_t *)param)->cmpr_func_ptr;

    // The return value is supposed to be a pointer to the correct element, therefore zero
    // is a NULL pointer, which should tell the caller that there was a problem.
    if( (0 == elem_cnt) || (NULL == cmpr_func_ptr) )
        return 0;

    // If there is a sorted (contiguous) buffer, but it's stale, we need to free it.
    // The value of elem_cnt (rcrd->recorder_data->used_entries) can
    // only increase, or be reset to zero, but when it is reset to zero
    // (by papi_sde_reset_recorder()) the buffer will be freed (by the same function).
    if( (NULL != rcrd->recorder_data->sorted_buffer) &&
        (rcrd->recorder_data->sorted_entries < elem_cnt) ){

        free( rcrd->recorder_data->sorted_buffer );
        rcrd->recorder_data->sorted_buffer = NULL;
        rcrd->recorder_data->sorted_entries = 0;
    }

    // Check if a sorted contiguous buffer is already there. If there is, return
    // the first or last element (for MIN, or MAX respectively).
    if( NULL != rcrd->recorder_data->sorted_buffer ){
        if( _SDE_CMP_MIN == which_edge )
            edge = rcrd->recorder_data->sorted_buffer;
        if( _SDE_CMP_MAX == which_edge )
            edge = rcrd->recorder_data->sorted_buffer + (elem_cnt-1)*typesize;
    }else{
        // Make "edge" point to the beginning of the first chunk.
        edge = rcrd->recorder_data->exp_container[0];
        if ( NULL == edge )
            return 0;

        cumul_size = 0;
        for(chunk=0; chunk<EXP_CONTAINER_ENTRIES; chunk++){
           current_size = ((long long)1<<chunk) * EXP_CONTAINER_MIN_SIZE;
           src = rcrd->recorder_data->exp_container[chunk];

           for(i=0; (i < (elem_cnt-cumul_size)) && (i < current_size); i++){
               void *next_elem = src + i*typesize;
               int rslt = cmpr_func_ptr(next_elem, edge);

               // If the new element is smaller than the current min and we are looking for the min, then keep it.
               if( (rslt < 0) && (_SDE_CMP_MIN == which_edge) )
                   edge = next_elem;
               // If the new element is larger than the current max and we are looking for the max, then keep it.
               if( (rslt > 0) && (_SDE_CMP_MAX == which_edge) )
                   edge = next_elem;
           }

           cumul_size += current_size;

           if( cumul_size >= elem_cnt )
               break;
        }
    }

    // We might free the sorted_buffer (when it becomes stale), so we can't return "edge".
    // Therefore, we allocate fresh space for the resulting element and copy it there.
    // Since we do not know when the user will use this pointer, we will not be able
    // to free it, so it is the responibility of the user (who calls PAPI_read()) to
    // free this memory.
    edge_copy = malloc( 1 * typesize);
    memcpy(edge_copy, edge, 1 * typesize);

    // A pointer is guaranteed to fit inside a long long, so cast it and return a long long.
    return (long long)edge_copy;
}



// This function returns a "long long" which contains a pointer to the
// data element that corresponds to the edge (min/max), so that it works
// for all types of data, not only integers.

// NOTE: This function allocates memory for one element and returns a pointer
// to this memory. Since we do not know when the user will use this pointer, we
// can not free it anywhere in this component, so it is the responibility of
// the user (who calls PAPI_read()) to free this memory.
static inline long long _sde_compute_quantile(void *param, int percent){
    long long quantile, elem_cnt;
    void *result_data;
    size_t typesize;
    sde_counter_t *rcrd;
    int (*cmpr_func_ptr)(const void *p1, const void *p2);

    rcrd = ((sde_sorting_params_t *)param)->recording;
    elem_cnt = rcrd->recorder_data->used_entries;
    typesize = rcrd->recorder_data->typesize;

    cmpr_func_ptr = ((sde_sorting_params_t *)param)->cmpr_func_ptr;

    // The return value is supposed to be a pointer to the correct element, therefore zero
    // is a NULL pointer, which should tell the caller that there was a problem.
    if( (0 == elem_cnt) || (NULL == cmpr_func_ptr) )
        return 0;

    // If there is a sorted (contiguous) buffer, but it's stale, we need to free it.
    // The value of elem_cnt (rcrd->recorder_data->used_entries) can
    // only increase, or be reset to zero, but when it is reset to zero
    // (by papi_sde_reset_recorder()) the buffer will be freed (by the same function).
    if( (NULL != rcrd->recorder_data->sorted_buffer) &&
        (rcrd->recorder_data->sorted_entries < elem_cnt) ){

        free( rcrd->recorder_data->sorted_buffer );
        rcrd->recorder_data->sorted_buffer = NULL;
        rcrd->recorder_data->sorted_entries = 0;
    }

    // Check if a sorted buffer is already there. If there isn't, allocate one.
    if( NULL == rcrd->recorder_data->sorted_buffer ){
        rcrd->recorder_data->sorted_buffer = malloc(elem_cnt * typesize);
        recorder_data_to_contiguous(rcrd, rcrd->recorder_data->sorted_buffer);
        // We set this field so we can test later to see if the allocated buffer is stale.
        rcrd->recorder_data->sorted_entries = elem_cnt;
    }
    void *sorted_buffer = rcrd->recorder_data->sorted_buffer;

    qsort(sorted_buffer, elem_cnt, typesize, cmpr_func_ptr);
    void *tmp_ptr = sorted_buffer + typesize*((elem_cnt*percent)/100);

    // We might free the sorted_buffer (when it becomes stale), so we can't return "tmp_ptr".
    // Therefore, we allocate fresh space for the resulting element and copy it there.
    // Since we do not know when the user will use this pointer, we will not be able
    // to free it, so it is the responibility of the user (who calls PAPI_read()) to
    // free this memory.
    result_data = malloc(typesize);
    memcpy(result_data, tmp_ptr, typesize);

    // convert the pointer into a long long so we can return it.
    quantile = (long long)result_data;

    return quantile;
}


static long long _sde_compute_q1(void *param){
    return _sde_compute_quantile(param, 25);
}
static long long _sde_compute_med(void *param){
    return _sde_compute_quantile(param, 50);
}
static long long _sde_compute_q3(void *param){
    return _sde_compute_quantile(param, 75);
}
static long long _sde_compute_min(void *param){
    return _sde_compute_edge(param, _SDE_CMP_MIN);
}
static long long _sde_compute_max(void *param){
    return _sde_compute_edge(param, _SDE_CMP_MAX);
}



/** This function finds the handle associated with a created counter, or a recorder,
  given the library handle and the event name.
  @param[in] handle -- (void *) pointer to sde structure for an individual
  library
  @param[in] event_name -- name of the event
  */
void
__attribute__((visibility("hidden")))
*papi_sde_get_counter_handle( void *handle, const char *event_name)
{
    sde_counter_t *counter_handle;
    papisde_library_desc_t *lib_handle;
    char *full_event_name;

    // if libpapi.so was not linked in with the application, the handle will be NULL, and that's ok.
    if( !handle ) return SDE_OK;

    lib_handle = (papisde_library_desc_t *) handle;
    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_get_counter_handle(): 'handle' is clobbered.\n");
        return NULL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be accessing shared data structures, so we need to acquire a lock.
    _sde_lock();
    counter_handle = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    _sde_unlock();

    free(full_event_name);

    return counter_handle;
}



static int sde_setup_counter_internals( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter, papi_sde_fptr_t fp_counter, void *param, sde_counter_t **placeholder )
{
    papisde_library_desc_t *lib_handle;
    sde_counter_t *tmp_item;
    unsigned int counter_uniq_id;
    char *full_event_name;

    if( placeholder )
        *placeholder = NULL;

    lib_handle = (papisde_library_desc_t *) handle;
    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        SDE_ERROR("sde_setup_counter_internals(): 'handle' is clobbered. Unable to register counter.\n");
        return SDE_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    SDEDBG("%s: Counter: '%s' will be added in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);

    if( !is_instant(cntr_mode) && !is_delta(cntr_mode) ){
        SDE_ERROR("Unknown mode %d. SDE counter mode must be either Instant or Delta.\n",cntr_mode);
        free(full_event_name);
        return SDE_ECMP;
    }

    papisde_control_t *gctl = _get_global_struct();
    if( NULL == gctl ){
        SDEDBG("sde_setup_counter_internals(): Unable to find symbols from libpapi.so. SDEs will not be accessible by external software.\n");
        return 1;
    }

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    _sde_lock();

    // Look if the event is already registered.
    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);

    if( NULL != tmp_item ){
        if( NULL != tmp_item->counter_group_head ){
            SDE_ERROR("sde_setup_counter_internals(): Unable to register counter '%s'. There is a counter group with the same name.\n",full_event_name);
            free(full_event_name);
            _sde_unlock();
            return SDE_EINVAL;
        }
        if( (NULL != tmp_item->data) || (NULL != tmp_item->func_ptr) ){
            // If it is registered and it is _not_ a placeholder then ignore it silently.
            SDEDBG("%s: Counter: '%s' was already in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);
            free(full_event_name);
            _sde_unlock();
            return SDE_OK;
        }
        // If it is registered and it _is_ a placeholder then update the mode, the type, and the 'data' pointer or the function pointer.
        SDEDBG("%s: Updating placeholder for counter: '%s' in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);

        // Both "counter" and "fp_counter" can be NULL, if we are creating a recorder.
        if( counter ){
            tmp_item->data = counter;
        }else if( fp_counter ){
            tmp_item->func_ptr = fp_counter;
            tmp_item->param = param;
        }
        tmp_item->cntr_mode = cntr_mode;
        tmp_item->cntr_type = cntr_type;
        free(full_event_name);

        if( placeholder )
            *placeholder = tmp_item;

        _sde_unlock();
        return SDE_OK;
    }

    // If neither the event, nor a placeholder exists, then use the current
    // number of registered events as the index of the new one, and increment it.
    counter_uniq_id = gctl->num_reg_events++;
    gctl->num_live_events++;

    SDEDBG("%s: Counter %s has unique ID = %d\n", __FILE__, full_event_name, counter_uniq_id);

    // allocate_and_insert() does not care if any (or all) of "counter", "fp_counter", or "param" are NULL. It will just assign them to the structure.
    tmp_item = allocate_and_insert( gctl, lib_handle, full_event_name, counter_uniq_id, cntr_mode, cntr_type, counter, fp_counter, param );
    _sde_unlock();
    if(NULL == tmp_item) {
        SDEDBG("%s: Counter not inserted in SDE %s\n", __FILE__, lib_handle->libraryName);
        free(full_event_name);
        return SDE_ECMP;
    }

    free(full_event_name);

    return SDE_OK;
}
#endif // !defined(SDE_COMMON_ONLY)

#endif // !defined(PAPI_SDE_LIB_H)
