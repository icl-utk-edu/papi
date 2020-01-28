/**
 * @file    sde.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is the component for supporting Software Defined Events (SDE).
 *  It provides an interface for libraries, runtimes, and other software
 *  layers to export events to other software layers through PAPI.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <dlfcn.h>
#include <assert.h>
#include "sde_internal.h"


/*************************************************************************/
/* Functions related to internal hashing of events                       */
/*************************************************************************/

static unsigned int ht_hash_id(unsigned int uniq_id){
    return uniq_id%PAPISDE_HT_SIZE;
}

// djb2 hash
static unsigned long ht_hash_name(const char *str)
{
    unsigned long hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash % PAPISDE_HT_SIZE;
}

static void ht_insert(papisde_list_entry_t *hash_table, int ht_key, sde_counter_t *sde_counter)
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

static sde_counter_t *ht_delete(papisde_list_entry_t *hash_table, int ht_key, unsigned int uniq_id)
{
    papisde_list_entry_t *list_head, *curr, *prev;
    sde_counter_t *item;

    list_head = &hash_table[ht_key];
    if( NULL == list_head->item ){
        PAPIERROR("ht_delete(): the entry does not exist.\n");
        fprintf(stderr,"ht_delete(): the entry does not exist.\n");
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
            PAPIERROR("ht_delete(): the hash table is clobbered.\n");
            fprintf(stderr,"ht_delete(): the hash table is clobbered.\n");
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

    fprintf(stderr,"ht_delete(): the item is not in the list.\n");
    return NULL;
}

static sde_counter_t *ht_lookup_by_name(papisde_list_entry_t *hash_table, const char *name)
{
    papisde_list_entry_t *list_head, *curr;

    list_head = &hash_table[ht_hash_name(name)];
    if( NULL == list_head->item ){
        return NULL;
    }

    for(curr=list_head; NULL != curr; curr=curr->next){
        if(NULL == curr->item){ // This can only legally happen for the head of the list.
            PAPIERROR("ht_lookup_by_name() the hash table is clobbered\n");
            return NULL;
        }
        if( !strcmp(curr->item->name, name) ){
            return curr->item;
        }
    }

    return NULL;
}

static sde_counter_t *ht_lookup_by_id(papisde_list_entry_t *hash_table, unsigned int uniq_id)
{
    papisde_list_entry_t *list_head, *curr;

    list_head = &hash_table[ht_hash_id(uniq_id)];
    if( NULL == list_head->item ){
        return NULL;
    }

    for(curr=list_head; NULL != curr; curr=curr->next){
        if(NULL == curr->item){ // This can only legally happen for the head of the list.
            PAPIERROR("ht_lookup_by_id() the hash table is clobbered\n");
            return NULL;
        }
        if(curr->item->glb_uniq_id == uniq_id){
            return curr->item;
        }
    }

    return NULL;
}

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

static void recorder_data_to_contiguous(sde_counter_t *recorder, void *cont_buffer){
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static sde_counter_t *allocate_and_insert(papisde_library_desc_t* lib_handle, const char* name, unsigned int uniq_id, int cntr_mode, int cntr_type, void* data, papi_sde_fptr_t func_ptr, void *param){

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

    papisde_control_t *gctl = _papisde_global_control; 
    (void)ht_insert(gctl->all_reg_counters, ht_hash_id(uniq_id), item);

    return item;
}

int delete_counter(papisde_library_desc_t* lib_handle, const char* name) 
{

    sde_counter_t *tmp_item;
    papisde_control_t *gctl = _papisde_global_control; 
    unsigned int item_uniq_id;

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

    return 0;
}

/*************************************************************************/
/* Below is the actual "hardware implementation" of the sde counters     */
/*************************************************************************/

static int
sde_cast_and_store(void *data, long long int previous_value, void *rslt, int cntr_type){
    void *tmp_ptr;

    switch(cntr_type){
        case PAPI_SDE_long_long:
            *(long long int *)rslt = *((long long int *)data) - previous_value;
            SUBDBG(" value LL=%lld (%lld-%lld)\n", *(long long int *)rslt, *((long long int *)data), previous_value);
            return PAPI_OK;
        case PAPI_SDE_int:
            // We need to cast the result to "long long" so it is expanded to 64bit to take up all the space
            *(long long int *)rslt = (long long int) (*((int *)data) - (int)previous_value);
            SUBDBG(" value LD=%lld (%d-%d)\n", *(long long int *)rslt, *((int *)data), (int)previous_value);
            return PAPI_OK;
        case PAPI_SDE_double:
            tmp_ptr = &previous_value;
            *(double *)rslt = (*((double *)data) - *((double *)tmp_ptr));
            SUBDBG(" value LF=%lf (%lf-%lf)\n", *(double *)rslt, *((double *)data), *((double *)(&previous_value)));
            return PAPI_OK;
        case PAPI_SDE_float:
            // We need to cast the result to "double" so it is expanded to 64bit to take up all the space
            tmp_ptr = &previous_value;
            *(double *)rslt = (double)(*((float *)data) - (float)(*((double *)tmp_ptr)) );
            SUBDBG(" value F=%lf (%f-%f)\n", *(double *)rslt, *((float *)data), (float)(*((double *)(&previous_value))) );
            return PAPI_OK;
        default:
            PAPIERROR("Unsupported counter type: %d\n",cntr_type);
            return -1;
    }

}


/* both "rslt" and "data" are local variables that this component stored after promoting to 64 bits. */
#define _SDE_AGGREGATE( _TYPE, _RSLT_TYPE ) do{\
                switch(group_flags){\
                    case PAPI_SDE_SUM:\
                        *(_RSLT_TYPE *)rslt = (_RSLT_TYPE)  ((_TYPE)(*(_RSLT_TYPE *)rslt) + (_TYPE)(*((_RSLT_TYPE *)data)) );\
                        break;\
                    case PAPI_SDE_MAX:\
                        if( *(_RSLT_TYPE *)rslt < *((_RSLT_TYPE *)data) )\
                            *(_RSLT_TYPE *)rslt = *((_RSLT_TYPE *)data);\
                        break;\
                    case PAPI_SDE_MIN:\
                        if( *(_RSLT_TYPE *)rslt > *((_RSLT_TYPE *)data) )\
                            *(_RSLT_TYPE *)rslt = *((_RSLT_TYPE *)data);\
                        break;\
                    default:\
                        PAPIERROR("Unsupported counter group flag: %d\n",group_flags);\
                        return -1;\
                } \
            }while(0)

int aggregate_value_in_group(long long int *data, long long int *rslt, int cntr_type, int group_flags){

    switch(cntr_type){
        case PAPI_SDE_long_long:
            _SDE_AGGREGATE(long long int, long long int);
            return PAPI_OK;
        case PAPI_SDE_int:
            // We need to cast the result to "long long" so it is expanded to 64bit to take up all the space
            _SDE_AGGREGATE(int, long long int);
            return PAPI_OK;
        case PAPI_SDE_double:
            _SDE_AGGREGATE(double, double);
            return PAPI_OK;
        case PAPI_SDE_float:
            // We need to cast the result to "double" so it is expanded to 64bit to take up all the space
            _SDE_AGGREGATE(float, double);
            return PAPI_OK;
        default:
            PAPIERROR("Unsupported counter type: %d\n",cntr_type);
            return -1;
    }

}

/**
  This function assumes that all counters in a group (including recursive subgroups) have the same type.
  */
static int sde_read_counter_group( sde_counter_t *counter, long long int *rslt ){
    papisde_list_entry_t *curr;
    long long int final_value = 0;

    if( NULL == counter ){
        PAPIERROR("sde_read_counter_group(): Counter parameter is NULL.\n");
        return PAPI_EINVAL;
    }

    curr = counter->counter_group_head;
    if( NULL == curr ){
        PAPIERROR("sde_read_counter_group(): Counter '%s' is not a counter group.\n",counter->name);
        return PAPI_EINVAL;
    }

    do{
        long long int tmp_value = 0;
        int ret_val;

        sde_counter_t *tmp_cntr = curr->item;
        if( NULL == tmp_cntr ){
            PAPIERROR("sde_read_counter_group(): List of counters in counter group '%s' is clobbered.\n",counter->name);
            return PAPI_EINVAL;
        }

        // We can _not_ have a recorder inside a group.
        if( NULL != tmp_cntr->recorder_data ){
            PAPIERROR("sde_read_counter_group(): Recorder found inside counter group: %s.\n",tmp_cntr->name);
        }else{
            // We allow counter groups to contain other counter groups recursively.
            if( NULL != tmp_cntr->counter_group_head ){
                ret_val = sde_read_counter_group( tmp_cntr, &tmp_value );
                if( ret_val != PAPI_OK ){
                    // If something went wrong with one counter group, ignore it silently.
                    continue;
                }
            }else{ // If we are here it means that we are trying to read a real counter.
                if( (NULL == tmp_cntr->data) && (NULL == tmp_cntr->func_ptr) ){
                    PAPIERROR("sde_read_counter_group(): Attempted read on a placeholder: %s.\n",tmp_cntr->name);
                    // If something went wrong with one counter, ignore it silently.
                    continue;
                }

                ret_val = sde_hardware_read_and_store( tmp_cntr, tmp_cntr->previous_data, &tmp_value );
                if( PAPI_OK != ret_val ){
                    PAPIERROR("sde_read_counter_group(): Error occured when reading counter: %s.\n",tmp_cntr->name);
                }
            }

            // There is nothing meaningful we could do with the error code here, so ignore it.
            (void)aggregate_value_in_group(&tmp_value, &final_value, tmp_cntr->cntr_type, counter->counter_group_flags);
        }

        curr = curr->next;
    }while(NULL != curr);

    *rslt = final_value;
    return PAPI_OK;
}

static int
sde_hardware_write( sde_counter_t *counter, long long int new_value )
{
    double tmp_double;
    void *tmp_ptr;

    switch(counter->cntr_type){
        case PAPI_SDE_long_long:
            *((long long int *)(counter->data)) = new_value;
            break;
        case PAPI_SDE_int:
            *((int *)(counter->data)) = (int)new_value;
            break;
        case PAPI_SDE_double:
            tmp_ptr = &new_value;
            tmp_double = *((double *)tmp_ptr);
            *((double *)(counter->data)) = tmp_double;
            break;
        case PAPI_SDE_float:
            // The pointer has to be 64bit. We can cast the variable to safely convert between bit-widths later on.
            tmp_ptr = &new_value;
            tmp_double = *((double *)tmp_ptr);
            *((float *)(counter->data)) = (float)tmp_double;
            break;
        default:
            PAPIERROR("Unsupported counter type: %d\n",counter->cntr_type);
            return -1;
    }

    return PAPI_OK;
}

static int
sde_hardware_read_and_store( sde_counter_t *counter, long long int previous_value, long long int *rslt )
{
    int ret_val;
    long long int tmp_int;
    void *tmp_data;

    char *event_name = counter->name;

    if ( counter->data != NULL ) { 
        SUBDBG("Reading %s by accessing data pointer.\n", event_name);
        tmp_data = counter->data;
    } else if( NULL != counter->func_ptr ){
        SUBDBG("Reading %s by calling registered function pointer.\n", event_name);
        tmp_int = counter->func_ptr(counter->param);
        tmp_data = &tmp_int;
    } else{
        PAPIERROR("sde_hardware_read_and_store(): Event %s has neither a variable nor a function pointer associated with it.\n", event_name);
        return -1;
    }

    if( is_instant(counter->cntr_mode) ){
        /* Instant counter means that we don't subtract the previous value (which we read at PAPI_Start()) */
        previous_value = 0;
    } else if( is_delta(counter->cntr_mode) ){
        /* Do nothing here, this is the default mode */
    } else{
        PAPIERROR("Unsupported mode (%d) for event: %s\n",counter->cntr_mode, event_name);
        return -1;
    }

    ret_val = sde_cast_and_store(tmp_data, previous_value, rslt, counter->cntr_type);
    return ret_val;
}



/** This helper function checks if the global structure has been allocated
    and allocates it if has not.
  @return a pointer to the global structure.
  */
papisde_control_t *get_global_struct(void){
    // Allocate the global control structure, unless it has already been allocated by another library
    // or the application code calling PAPI_name_to_code() for an SDE.
    if ( !_papisde_global_control ) {
        SUBDBG("get_global_struct(): global SDE control struct is being allocated.\n");
        _papisde_global_control = ( papisde_control_t* ) papi_calloc( 1, sizeof( papisde_control_t ) );
    }
    return _papisde_global_control;
}

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
        SUBDBG("Checking library: '%s' against registered library: '%s'\n",library_name, tmp_lib->libraryName);
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
    SUBDBG("insert_library_handle(): inserting new handle for library: '%s'\n",lib_handle->libraryName);
    lib_handle->next = gctl->lib_list_head;
    gctl->lib_list_head = lib_handle;

    return;
}

/** This function creates the SDE component structure for an individual 
  software library and returns a handle to the structure.
  @param[in] name_of_library -- library name
  @param[in] event_count -- number of exposed software defeined events 
  @param[out] sde_handle -- opaque pointer to sde structure for initialized 
  library
  */
papi_handle_t 
__attribute__((visibility("default")))
papi_sde_init(const char *name_of_library)
{
    papisde_library_desc_t *tmp_lib;

    SUBDBG("Registering library: '%s'\n",name_of_library);

    // Lock before we read and/or modify the global structures.
    papi_sde_lock();

    // Put the actual work in a different function so we call it from other
    // places in the component.  We have to do this because we cannot call
    // papi_sde_init() from places in the code which already call
    // papi_sde_lock()/papi_sde_unlock(), or we will end up with deadlocks.
    tmp_lib = do_sde_init(name_of_library);

    papi_sde_unlock();

    SUBDBG("Library '%s' has been registered.\n",name_of_library);

    return tmp_lib;
}


// Initialize library handle, or return the existing one if already
// initialized. This function is _not_ thread safe, so it needs to be called
// from within regions protected by papi_sde_lock()/papi_sde_unlock().
static papi_handle_t 
do_sde_init(const char *name_of_library)
{
    papisde_control_t* gctl;
    papisde_library_desc_t *tmp_lib;

    SUBDBG("Registering library: '%s'\n",name_of_library);

    gctl = get_global_struct();

    // If the library is already initialized, return the handle to it
    tmp_lib = find_library_by_name(name_of_library, gctl);
    if( NULL != tmp_lib ){
        return tmp_lib;
    }

    // If the library is not already initialized, then initialize it.
    tmp_lib = ( papisde_library_desc_t* ) papi_calloc( 1, sizeof( papisde_library_desc_t ) );
    tmp_lib->libraryName = strdup(name_of_library);

    insert_library_handle(tmp_lib, gctl);

    return tmp_lib;
}


int 
__attribute__((visibility("default")))
papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags)
{
    papisde_library_desc_t *lib_handle;
    sde_counter_t *tmp_item, *tmp_group;
    unsigned int cntr_group_uniq_id;
    char *full_event_name, *full_group_name;

    SUBDBG("papi_sde_add_counter_to_group(): Adding counter: %s into group %s\n",event_name, group_name);

    lib_handle = (papisde_library_desc_t *) handle;
    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        PAPIERROR("papi_sde_add_counter_to_group(): 'handle' is clobbered. Unable to add counter to group.\n");
        return PAPI_EINVAL;
    }


    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    papi_sde_lock();

    // Check to make sure that the event is already registered. This is not the place to create a placeholder.
    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if( NULL == tmp_item ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_add_counter_to_group(): Unable to find counter: '%s'.\n",full_event_name);
        free(full_event_name);
        return PAPI_EINVAL;
    }

    // We will not use the name beyond this point
    free(full_event_name);

    str_len = strlen(lib_handle->libraryName)+strlen(group_name)+2+1; // +2 for "::" and +1 for '\0'
    full_group_name = malloc(str_len*sizeof(char));
    snprintf(full_group_name, str_len, "%s::%s", lib_handle->libraryName, group_name);

    // Check to see if the group exists already. Otherwise we need to create it.
    tmp_group = ht_lookup_by_name(lib_handle->lib_counters, full_group_name);
    if( NULL == tmp_group ){

        // We use the current number of registered events as the uniq id of the counter group, and we
        // increment it because counter groups are treated as real counters by the outside world.
        // They are first class citizens.
        papisde_control_t *gctl = _papisde_global_control; 
        cntr_group_uniq_id = gctl->num_reg_events++;
        gctl->num_live_events++;
        _sde_vector.cmp_info.num_native_events = gctl->num_live_events;

        SUBDBG("%s line %d: Unique ID for new counter group = %d\n", __FILE__, __LINE__, cntr_group_uniq_id);

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
            PAPIERROR("papi_sde_add_counter_to_group(): Found an empty counter group: '%s'. This might indicate that a cleanup routine is not doing its job.\n", group_name);
        }

        // make sure the caller is not trying to change the flags of the group after it has been created.
        if( tmp_group->counter_group_flags != group_flags ){
            papi_sde_unlock();
            PAPIERROR("papi_sde_add_counter_to_group(): Attempting to add counter '%s' to counter group '%s' with incompatible group flags.\n", event_name, group_name);
            free(full_group_name);
            return PAPI_EINVAL;
        }
    }

    // Add the new counter to the group's head.
    papisde_list_entry_t *new_head = calloc(1, sizeof(papisde_list_entry_t));
    new_head->item = tmp_item;
    new_head->next = tmp_group->counter_group_head;
    tmp_group->counter_group_head = new_head;

    papi_sde_unlock();
    free(full_group_name);
    return PAPI_OK;
}


// In contrast with papi_sde_register_counter(), the following function creates
// a counter whose memory is allocated and managed by PAPI, not the library.
// This counter can only by modified via the functions papi_sde_inc_counter()
// and papi_sde_reset_counter(). This has two benefits over a counter which
// lives inside a library and is modified directly by that library:
// A) Our counter and the modifying API is guaranteed to be thread safe.
// B) Since we learn about each change in the value of the counter, we can
//    implement accurate overflowing and/or a push mode.
//
// However, this approach has higher overhead than executing "my_cntr += value" inside a library.

int 
__attribute__((visibility("default")))
papi_sde_create_counter( papi_handle_t handle, const char *event_name, int cntr_mode, void **cntr_handle )
{
    int ret_val;
    long long int *counter_data;
    char *full_event_name;
    papisde_library_desc_t *lib_handle;
    sde_counter_t *cntr, *placeholder;

    lib_handle = (papisde_library_desc_t *) handle;
    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        PAPIERROR("papi_sde_create_counter(): 'handle' is clobbered. Unable to create counter.\n");
        return PAPI_EINVAL;
    }

    SUBDBG("Preparing to create counter: '%s' with mode: '%d' in SDE library: %s.\n", event_name, cntr_mode, lib_handle->libraryName);

    counter_data = calloc(1, sizeof(long long int));


    ret_val = sde_setup_counter_internals( lib_handle, event_name, cntr_mode, PAPI_SDE_long_long, counter_data, NULL, NULL, &placeholder );
    if( PAPI_OK != ret_val ){
        return ret_val;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    cntr = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if(NULL == cntr) {
        SUBDBG("Logging counter '%s' not properly inserted in SDE library '%s'\n", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        return PAPI_ECMP;
    }

    // Signify that this counter is a created counter (as opposed to a registered one).
    // The reason we need to know is so we can free() the 'data' entry which we allocated here, and for
    // correctness checking in papi_sde_inc_coutner() and papi_sde_reset_counter().
    cntr->is_created = 1;

    if( NULL != cntr_handle ){
        *(sde_counter_t **)cntr_handle = cntr;
    }

    free(full_event_name);

    return PAPI_OK;
}


// The following function works only for counters created using papi_sde_create_counter().
int 
__attribute__((visibility("default")))
papi_sde_inc_counter( papi_handle_t cntr_handle, long long int increment)
{
    long long int *ptr;
    sde_counter_t *tmp_cntr;
#if defined(SDE_HAVE_OVERFLOW)
    EventSetInfo_t *ESI;
    int cidx, i, index_in_ESI = -1;
    ThreadInfo_t *thread;
    sde_control_state_t *sde_ctl;
#endif //defined(SDE_HAVE_OVERFLOW)

    papi_sde_lock();

    tmp_cntr = (sde_counter_t *)cntr_handle;
    if( NULL == tmp_cntr ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_inc_counter(): 'cntr_handle' is clobbered. Unable to modify value of counter.\n");
        return PAPI_EINVAL;
    }

//    SUBDBG("Preparing to increment counter: '%s::%s' by %lld.\n", tmp_cntr->which_lib->libraryName, tmp_cntr->name, increment);

    ptr = (long long int *)(tmp_cntr->data);

    if( NULL == ptr ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_inc_counter(): Counter structure is clobbered. Unable to modify value of counter.\n");
        return PAPI_EINVAL;
    }

    if( !tmp_cntr->is_created ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_inc_counter(): Counter is not created by PAPI, cannot be modified using this function.\n");
        return PAPI_EINVAL;
    }

    if( PAPI_SDE_long_long != tmp_cntr->cntr_type ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_inc_counter(): Counter is not of type \"long long int\" and cannot be modified using this function.\n");
        return PAPI_EINVAL;
    }

    *ptr += increment;

#if defined(SDE_HAVE_OVERFLOW)
    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );
    if( NULL == thread )
        goto counter_did_not_overflow;

    ESI = thread->running_eventset[cidx];
    // Check if there is a running event set and it has some events set to overflow
    if( (NULL == ESI) || !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) 
        goto counter_did_not_overflow;

    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;
    int event_counter = ESI->overflow.event_counter;

    // Check all the events that are set to overflow
    index_in_ESI = -1;
    for (i = 0; i < event_counter; i++ ) {
        int papi_index = ESI->overflow.EventIndex[i];
        unsigned int counter_uniq_id = sde_ctl->which_counter[papi_index];
        // If the created counter that we are incrementing corresponds to
        // an event that was set to overflow, read the deadline and threshold.
        if( counter_uniq_id == tmp_cntr->glb_uniq_id ){
            index_in_ESI = i;
            break;
        }
    }

    if( index_in_ESI >= 0 ){
        long long deadline, threshold, latest;
        deadline = ESI->overflow.deadline[index_in_ESI];
        threshold = ESI->overflow.threshold[index_in_ESI];

        // If the current value has exceeded the deadline then
        // invoke the user handler and update the deadline.
        latest = *ptr;
        SUBDBG("counter: '%s::%s' has value: %lld and the overflow deadline is at: %lld.\n", tmp_cntr->which_lib->libraryName, tmp_cntr->name, latest, deadline);
        if( latest > deadline ){
            // We adjust the deadline in a way that it remains a multiple of threshold
            // so we don't create an additive error. However, this code path should
            // result in a precise overflow trigger, so this might not be necessary.
            ESI->overflow.deadline[index_in_ESI] = threshold*(latest/threshold) + threshold;
            invoke_user_handler(cntr_handle);
        }
    }

counter_did_not_overflow:
#endif // defined(SDE_HAVE_OVERFLOW)

    papi_sde_unlock();

    return PAPI_OK;
}


int 
__attribute__((visibility("default")))
papi_sde_compare_long_long(const void *p1, const void *p2){
    long long n1, n2;
    n1 = *(long long *)p1;
    n2 = *(long long *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int 
__attribute__((visibility("default")))
papi_sde_compare_int(const void *p1, const void *p2){
    int n1, n2;
    n1 = *(int *)p1;
    n2 = *(int *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int 
__attribute__((visibility("default")))
papi_sde_compare_double(const void *p1, const void *p2){
    double n1, n2;
    n1 = *(double *)p1;
    n2 = *(double *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int 
__attribute__((visibility("default")))
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


long long _sde_compute_q1(void *param){
    return _sde_compute_quantile(param, 25);
}
long long _sde_compute_med(void *param){
    return _sde_compute_quantile(param, 50);
}
long long _sde_compute_q3(void *param){
    return _sde_compute_quantile(param, 75);
}
long long _sde_compute_min(void *param){
    return _sde_compute_edge(param, _SDE_CMP_MIN);
}
long long _sde_compute_max(void *param){
    return _sde_compute_edge(param, _SDE_CMP_MAX);
}


int 
__attribute__((visibility("default")))
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

    papisde_library_desc_t *lib_handle = handle;

    papi_sde_lock();

    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        PAPIERROR("papi_sde_create_recorder(): 'handle' is clobbered. Unable to create recorder.\n");
        papi_sde_unlock();
        return PAPI_EINVAL;
    }

    SUBDBG("Preparing to create recorder: '%s' with typesize: '%d' in SDE library: %s.\n", event_name, (int)typesize, lib_handle->libraryName);

    // We setup the recorder like this, instead of using sde_do_register() because recorders cannot be set to overflow.
    ret_val = sde_setup_counter_internals( lib_handle, event_name, PAPI_SDE_DELTA|PAPI_SDE_RO, PAPI_SDE_long_long, NULL, NULL, NULL, NULL );
    if( PAPI_OK != ret_val )
        return ret_val;

    str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    tmp_rec_handle = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if(NULL == tmp_rec_handle) {
        SUBDBG("Recorder '%s' not properly inserted in SDE library '%s'\n", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        papi_sde_unlock();
        return PAPI_ECMP;
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
    SUBDBG("papi_sde_create_recorder(): Preparing to register aux counter: '%s' in SDE library: %s.\n", aux_event_name, lib_handle->libraryName);

    // The :CNT aux counter is properly registered so that it can be set to overflow.
    ret_val = sde_do_register( lib_handle, (const char *)aux_event_name, PAPI_SDE_INSTANT|PAPI_SDE_RO, PAPI_SDE_long_long, &(tmp_rec_handle->recorder_data->used_entries), NULL, NULL );
    if( PAPI_OK != ret_val ){
        SUBDBG("papi_sde_create_recorder(): Registration of aux counter: '%s' in SDE library: %s FAILED.\n", aux_event_name, lib_handle->libraryName);
        papi_sde_unlock();
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

            SUBDBG("papi_sde_create_recorder(): Preparing to register aux fp counter: '%s' in SDE library: %s.\n", aux_event_name, lib_handle->libraryName);
            ret_val = sde_do_register(lib_handle, (const char *)aux_event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_long_long, NULL, func_ptr_vec[i], sorting_params );
            if( PAPI_OK != ret_val ){
                SUBDBG("papi_sde_create_recorder(): Registration of aux counter: '%s' in SDE library: %s FAILED.\n", aux_event_name, lib_handle->libraryName);
                papi_sde_unlock();
                free(aux_event_name);
                return ret_val;
            }
        }
    }

    papi_sde_unlock();
    free(aux_event_name);
    return PAPI_OK;
}


// UPDATED for EXP-storage
int 
__attribute__((visibility("default")))
papi_sde_record( void *record_handle, size_t typesize, void *value)
{
    sde_counter_t *tmp_item;
    long long used_entries, total_entries, prev_entries, offset;
    int i, chunk;
    long long tmp_size;

    SUBDBG("Preparing to record value of size %lu at address: %p\n",typesize, value);

    papi_sde_lock();

    tmp_item = (sde_counter_t *)record_handle;

    if( NULL == tmp_item ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_record(): 'record_handle' is clobbered. Unable to record value.\n");
        return PAPI_EINVAL;
    }

    if( NULL == tmp_item->recorder_data || NULL == tmp_item->recorder_data->exp_container[0]){
        papi_sde_unlock();
        PAPIERROR("papi_sde_record(): Counter structure is clobbered. Unable to record event.\n");
        return PAPI_EINVAL;
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

    papi_sde_unlock();
    return PAPI_OK;
}



// This function neither frees the allocated, nor does it zero it. It only resets the counter of used entries so that
// the allocated space can be resused (and overwritten) by future calls to record().
int 
__attribute__((visibility("default")))
papi_sde_reset_recorder( void *record_handle )
{
    sde_counter_t *tmp_rcrdr;

    papi_sde_lock();
    tmp_rcrdr = (sde_counter_t *)record_handle;

    if( NULL == tmp_rcrdr || NULL == tmp_rcrdr->recorder_data ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_record(): 'record_handle' is clobbered. Unable to reset recorder.\n");
        return PAPI_EINVAL;
    }

    // NOTE: do _not_ free the chunks and do _not_ reset "recorder_data->total_entries"

    tmp_rcrdr->recorder_data->used_entries = 0;
    free( tmp_rcrdr->recorder_data->sorted_buffer );
    tmp_rcrdr->recorder_data->sorted_buffer = NULL;
    tmp_rcrdr->recorder_data->sorted_entries = 0;

    papi_sde_unlock();
    return PAPI_OK;
}


// The following function works only for counters created using papi_sde_create_counter().
int 
__attribute__((visibility("default")))
papi_sde_reset_counter( void *cntr_handle )
{
    long long int *ptr;
    sde_counter_t *tmp_cntr;

    papi_sde_lock();

    tmp_cntr = (sde_counter_t *)cntr_handle;

    if( NULL == tmp_cntr ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_reset_counter(): 'cntr_handle' is clobbered. Unable to reset value of counter.\n");
        return PAPI_EINVAL;
    }

    ptr = (long long int *)(tmp_cntr->data);

    if( NULL == ptr ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_reset_counter(): Counter structure is clobbered. Unable to reset value of counter.\n");
        return PAPI_EINVAL;
    }

    if( tmp_cntr->is_created ){
        papi_sde_unlock();
        PAPIERROR("papi_sde_reset_counter(): Counter is not created by PAPI, so it cannot be reset.\n");
        return PAPI_EINVAL;
    }

    *ptr = 0; // Reset the counter.

    papi_sde_unlock();

    return PAPI_OK;
}


#if defined(SDE_HAVE_OVERFLOW)
static int
_sde_arm_timer(sde_control_state_t *sde_ctl){
    struct itimerspec its;

    // We will start the timer at 100us because we adjust its period in _sde_dispatch_timer()
    // if the counter is not growing fast enough, or growing too slowly.
    its.it_value.tv_sec = 0;
    its.it_value.tv_nsec = 100*1000; // 100us
    its.it_interval.tv_sec = its.it_value.tv_sec;
    its.it_interval.tv_nsec = its.it_value.tv_nsec;

    SUBDBG( "starting SDE internal timer for emulating HARDWARE overflowing\n");
    if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
        PAPIERROR("timer_settime");
        timer_delete(sde_ctl->timerid);
        sde_ctl->has_timer = 0;

        // If the timer is broken, let the caller know that something internal went wrong.
        return PAPI_ECMP;
    }

    return PAPI_OK;
}
#endif //defined(SDE_HAVE_OVERFLOW)


static int 
sde_setup_counter_internals( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter, papi_sde_fptr_t fp_counter, void *param, sde_counter_t **placeholder )
{   
    papisde_library_desc_t *lib_handle;
    sde_counter_t *tmp_item;
    unsigned int counter_uniq_id;
    char *full_event_name;

    if( placeholder )
        *placeholder = NULL;

    lib_handle = (papisde_library_desc_t *) handle;
    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        PAPIERROR("sde_setup_counter_internals(): 'handle' is clobbered. Unable to register counter.\n");
        return PAPI_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    SUBDBG("%s: Counter: '%s' will be added in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);

    if( !is_instant(cntr_mode) && !is_delta(cntr_mode) ){
        PAPIERROR("Unknown mode %d. SDE counter mode must be either Instant or Delta.\n",cntr_mode);
        free(full_event_name);
        return PAPI_ECMP;
    }

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    papi_sde_lock();

    // Look if the event is already registered.
    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);

    if( NULL != tmp_item ){
        if( NULL != tmp_item->counter_group_head ){
            PAPIERROR("sde_setup_counter_internals(): Unable to register counter '%s'. There is a counter group with the same name.\n",full_event_name);
            free(full_event_name);
            papi_sde_unlock();
            return PAPI_EINVAL;
        }
        if( (NULL != tmp_item->data) || (NULL != tmp_item->func_ptr) ){
            // If it is registered and it is _not_ a placeholder then ignore it silently.
            SUBDBG("%s: Counter: '%s' was already in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);
            free(full_event_name);
            papi_sde_unlock();
            return PAPI_OK;
        }
        // If it is registered and it _is_ a placeholder then update the mode, the type, and the 'data' pointer or the function pointer.
        SUBDBG("%s: Updating placeholder for counter: '%s' in library: %s.\n", __FILE__, full_event_name, lib_handle->libraryName);

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

        papi_sde_unlock();
        return PAPI_OK;
    }

    // If neither the event, nor a placeholder exists, then use the current
    // number of registered events as the index of the new one, and increment it.
    papisde_control_t *gctl = _papisde_global_control; 
    counter_uniq_id = gctl->num_reg_events++;
    gctl->num_live_events++;
    _sde_vector.cmp_info.num_native_events = gctl->num_live_events;

    SUBDBG("%s: Counter %s has unique ID = %d\n", __FILE__, full_event_name, counter_uniq_id);

    // allocate_and_insert() does not care if any (or all) of "counter", "fp_counter", or "param" are NULL. It will just assign them to the structure.
    tmp_item = allocate_and_insert( lib_handle, full_event_name, counter_uniq_id, cntr_mode, cntr_type, counter, fp_counter, param );
    papi_sde_unlock();
    if(NULL == tmp_item) {
        SUBDBG("%s: Counter not inserted in SDE %s\n", __FILE__, lib_handle->libraryName);
        free(full_event_name);
        return PAPI_ECMP;
    }

    free(full_event_name);

    return PAPI_OK;
}



/** This function registers an event name and counter within the SDE component 
  structure attached to the handle. A default description for an event is 
  synthesized from the library name and the event name when they are registered.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library
  @param[in] event_name -- string containing the name of the event 
  @param[in] cntr_type -- the type of the counter (PAPI_SDE_long_long, PAPI_SDE_int, PAPI_SDE_double, PAPI_SDE_float)
  @param[in] cntr_mode -- the mode of the counter (one of: PAPI_SDE_RO, PAPI_SDE_RW and one of: PAPI_SDE_DELTA, PAPI_SDE_INSTANT)
  @param[in] counter -- pointer to a variable that stores the value for the event
  */
int 
__attribute__((visibility("default")))
papi_sde_register_counter( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter )
{
    int ret_val;
    papi_sde_lock();
    ret_val = sde_do_register(handle, event_name, cntr_mode, cntr_type, counter, NULL, NULL);
    papi_sde_unlock();

    return ret_val;
}

/** This function registers an event name and (caller provided) callback function
  within the SDE component structure attached to the handle.
  A default description for an event is 
  synthesized from the library name and the event name when they are registered.
  @param[in] handle -- (void *) pointer to sde structure for an individual library.
  @param[in] event_name -- (char *) name of the event.
  @param[in] cntr_mode -- (int) mode of the event counter.
  @param[in] cntr_type -- (int) type of the event counter.
  @param[in] fp_counter -- pointer to a callback function that SDE will call when PAPI_read/stop/accum is called.
  @param[in] param -- (void *) opaque parameter that will be passed to the callback function every time it's called.
  */
int 
__attribute__((visibility("default")))
papi_sde_register_fp_counter( void *handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t fp_counter, void *param )
{
    int ret_val;
    papi_sde_lock();
    ret_val = sde_do_register( handle, event_name, cntr_mode, cntr_type, NULL, fp_counter, param );
    papi_sde_unlock();

    return ret_val;
}

static inline int
sde_do_register( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter, papi_sde_fptr_t fp_counter, void *param )
{   
    sde_counter_t *placeholder;

    SUBDBG("%s: Preparing to register counter: '%s' with mode: '%d' and type: '%d'.\n", __FILE__, event_name, cntr_mode, cntr_type);

    int ret = sde_setup_counter_internals( handle, event_name, cntr_mode, cntr_type, counter, fp_counter, param, &placeholder );

    if( PAPI_OK != ret )
        return ret;

#if defined(SDE_HAVE_OVERFLOW)
    if( NULL != placeholder ){
        // Check if we need to worry about overflow (cases r[4-6], or c[4-6])
        if( placeholder->overflow ){
            ThreadInfo_t *thread;
            EventSetInfo_t *ESI;
            sde_control_state_t *sde_ctl;

            // Below here means that we are in cases r[4-6]
            thread = _papi_hwi_lookup_thread( 0 );
            if( NULL == thread )
                goto no_new_timer;

            // Get the current running eventset and check if it has some events set to overflow.
            int cidx = _sde_vector.cmp_info.CmpIdx;
            ESI = thread->running_eventset[cidx];
            if( (NULL == ESI) || !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) 
                goto no_new_timer;

            sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;

            // Below this point we know we have a running eventset, so we are in case r5.
            // Since the event is set to overfow, if there is no timer in the eventset, create one and arm it.
            if( !(sde_ctl->has_timer) ){
                int ret = set_timer_for_overflow(sde_ctl);
                if( PAPI_OK != ret ){
                    return ret;
                }
                ret = _sde_arm_timer(sde_ctl);
                return ret;
            }
        }
    }
no_new_timer:
#endif // defined(SDE_HAVE_OVERFLOW)

    return PAPI_OK;
}


int 
__attribute__((visibility("default")))
papi_sde_unregister_counter( void *handle, const char *event_name)
{
    papisde_library_desc_t *lib_handle;
    int error;
    char *full_event_name;

    lib_handle = (papisde_library_desc_t *) handle;
    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        PAPIERROR("papi_sde_unregister_counter(): 'handle' is clobbered. Unable to unregister counter.\n");
        return PAPI_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    SUBDBG("papi_sde_unregister_counter(): Preparing to unregister counter: '%s' from SDE library: %s.\n", full_event_name, lib_handle->libraryName);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    papi_sde_lock();

    error = delete_counter( lib_handle, full_event_name );
    // Check if we found a registered counter, or if it never existed.
    if( error ){
        PAPIERROR("papi_sde_unregister_counter(): Counter '%s' has not been registered by library '%s'.\n", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        papi_sde_unlock();
        return PAPI_EINVAL;
    }

    // We will not use the name beyond this point
    free(full_event_name);

    papisde_control_t *gctl = _papisde_global_control; 
    gctl->num_live_events--;
    _sde_vector.cmp_info.num_native_events = gctl->num_live_events;

    papi_sde_unlock();
    return PAPI_OK;
}




/** This function optionally replaces an event's default description with a 
  description provided by the library developer within the SDE component 
  structure attached to the handle.  
  @param[in] handle -- (void *) pointer to sde structure for an individual 
  library
  @param[in] event_name -- name of the event 
  @param[in] event_description -- description of the event
  */
int 
__attribute__((visibility("default")))
papi_sde_describe_counter( void *handle, const char *event_name, const char *event_description )
{   
    sde_counter_t *tmp_item;
    papisde_library_desc_t *lib_handle;
    char *full_event_name;

    lib_handle = (papisde_library_desc_t *) handle;
    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        PAPIERROR("papi_sde_describe_counter(): 'handle' is clobbered. Unable to add description for counter.\n");
        return PAPI_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    papi_sde_lock();

    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if( NULL != tmp_item ){
        tmp_item->description = strdup(event_description);
        free(full_event_name);
        papi_sde_unlock();
        return PAPI_OK;
    }
    SUBDBG("papi_sde_describe_counter() Event: '%s' is not registered in SDE library: '%s'\n", full_event_name, lib_handle->libraryName);
    // We will not use the name beyond this point
    free(full_event_name);
    papi_sde_unlock();
    return PAPI_EINVAL;
}



/** This function finds the handle associated with a created counter, or a recorder,
  given the library handle and the event name.
  @param[in] handle -- (void *) pointer to sde structure for an individual 
  library
  @param[in] event_name -- name of the event 
  */
void 
__attribute__((visibility("default")))
*papi_sde_get_counter_handle( void *handle, const char *event_name)
{   
    sde_counter_t *counter_handle;
    papisde_library_desc_t *lib_handle;
    char *full_event_name;

    lib_handle = (papisde_library_desc_t *) handle;
    if( (NULL == lib_handle) || (NULL == lib_handle->libraryName) ){
        PAPIERROR("papi_sde_get_counter_handle(): 'handle' is clobbered.\n");
        return NULL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be accessing shared data structures, so we need to acquire a lock.
    papi_sde_lock();
    counter_handle = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    papi_sde_unlock();

    free(full_event_name);

    return counter_handle;
}


/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/


static int
_sde_init_component( int cidx )
{
    SUBDBG("_sde_init_component...\n");

    _sde_vector.cmp_info.num_native_events = 0;
    _sde_vector.cmp_info.CmpIdx = cidx;

    return PAPI_OK;
}



/** This is called whenever a thread is initialized */
static int
_sde_init_thread( hwd_context_t *ctx )
{
    (void)ctx;
    SUBDBG( "_sde_init_thread %p...\n", ctx );
    return PAPI_OK;
}



/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

static int
_sde_init_control_state( hwd_control_state_t * ctl )
{
    SUBDBG( "sde_init_control_state... %p\n", ctl );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;
    memset( sde_ctl, 0, sizeof ( sde_control_state_t ) );

    return PAPI_OK;
}


/** Triggered by eventset operations like add or remove */
static int
_sde_update_control_state( hwd_control_state_t *ctl, 
        NativeInfo_t *native,
        int count, 
        hwd_context_t *ctx )
{

    (void) ctx;
    int i, index;

    SUBDBG( "_sde_update_control_state %p %p...\n", ctl, ctx );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;   

    for( i = 0; i < count; i++ ) {
        index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
        if( index < 0 ){
            PAPIERROR("_sde_update_control_state(): Event at index %d has a negative native event code = %d.\n",i,index);
            return PAPI_EINVAL;
        }
        SUBDBG("_sde_update_control_state: i=%d index=%u\n", i, index );
        sde_ctl->which_counter[i] = (unsigned)index;
        native[i].ni_position = i;
    }

    // If an event for which overflowing was set is being removed from the eventset, then the
    // framework will turn overflowing off (by calling PAPI_overflow() with threshold=0),
    // so we don't need to do anything here.

    sde_ctl->num_events=count;

    return PAPI_OK;
}


/** Triggered by PAPI_start() */
static int
_sde_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    int ret_val;
#if defined(SDE_HAVE_OVERFLOW)
    ThreadInfo_t *thread;
    int cidx;
    struct itimerspec its;
#endif // defined(SDE_HAVE_OVERFLOW)
    ( void ) ctx;
    ( void ) ctl;

    SUBDBG( "%p %p...\n", ctx, ctl );

    ret_val = _sde_reset(ctx, ctl);

#if defined(SDE_HAVE_OVERFLOW)
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    its.it_value.tv_sec = 0;
    // We will start the timer at 100us because we adjust its period in _sde_dispatch_timer()
    // if the counter is not growing fast enough, or growing too slowly.
    its.it_value.tv_nsec = 100*1000; // 100us
    its.it_interval.tv_sec = its.it_value.tv_sec;
    its.it_interval.tv_nsec = its.it_value.tv_nsec;

    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );

    if ( (NULL != thread) && (NULL != thread->running_eventset[cidx]) && (thread->running_eventset[cidx]->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        if( !(sde_ctl->has_timer) ){
            // No registered counters went through r[1-3]
            int i;
            papisde_control_t *gctl = get_global_struct();
            for( i = 0; i < sde_ctl->num_events; i++ ) {
                unsigned int counter_uniq_id = sde_ctl->which_counter[i];
                if( counter_uniq_id >= gctl->num_reg_events ){
                    PAPIERROR("_sde_start(): Event at index %d does not correspond to a registered counter.\n",i);
                    continue;
                }

                sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
                if( NULL == counter ){
                    PAPIERROR("_sde_start(): Event at index %d corresponds to a clobbered counter.\n",i);
                    continue;
                }

                // If the counter that we are checking was set to overflow and it is registered (not created), create the timer.
                if( !(counter->is_created) && counter->overflow ){
                    // Registered counters went through r4
                    int ret = set_timer_for_overflow(sde_ctl);
                    if( PAPI_OK != ret ){
                        papi_sde_unlock();
                    }
                    break;
                }
            }
        }

        // r[1-4]
        if( sde_ctl->has_timer ){
            SUBDBG( "starting SDE internal timer for emulating HARDWARE overflowing\n");
            if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
                PAPIERROR("timer_settime");
                timer_delete(sde_ctl->timerid);
                sde_ctl->has_timer = 0;
                return PAPI_ECMP;
            }
        }
    }
#endif // defined(SDE_HAVE_OVERFLOW)

    return ret_val;
}


/** Triggered by PAPI_stop() */
static int
_sde_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

    (void) ctx;
    (void) ctl;
#if defined(SDE_HAVE_OVERFLOW)
    ThreadInfo_t *thread;
    int cidx;
    struct itimerspec zero_time;
#endif // defined(SDE_HAVE_OVERFLOW)

    SUBDBG( "sde_stop %p %p...\n", ctx, ctl );
    /* anything that would need to be done at counter stop time */

#if defined(SDE_HAVE_OVERFLOW)
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;

    cidx = _sde_vector.cmp_info.CmpIdx;
    thread = _papi_hwi_lookup_thread( 0 );

    if ( (NULL != thread) && (NULL != thread->running_eventset[cidx]) && (thread->running_eventset[cidx]->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        if( sde_ctl->has_timer ){
            SUBDBG( "stopping SDE internal timer\n");
            memset(&zero_time, 0, sizeof(struct itimerspec));
            if (timer_settime(sde_ctl->timerid, 0, &zero_time, NULL) == -1){
                PAPIERROR("timer_settime");
                timer_delete(sde_ctl->timerid);
                sde_ctl->has_timer = 0;
                return PAPI_ECMP;
            }
        }
    }
#endif // defined(SDE_HAVE_OVERFLOW)

    return PAPI_OK;
}

/** Triggered by PAPI_read()     */
/*     flags field is never set? */
static int
_sde_read( hwd_context_t *ctx, hwd_control_state_t *ctl, long long **events, int flags )
{
    int i;
    int ret_val = PAPI_OK;
    (void) flags;
    (void) ctx;

    papisde_control_t *gctl = _papisde_global_control;   

    SUBDBG( "_sde_read... %p %d\n", ctx, flags );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;   

    // Lock before we read num_reg_events and the hash-tables.
    papi_sde_lock();


    for( i = 0; i < sde_ctl->num_events; i++ ) {
        unsigned int counter_uniq_id = sde_ctl->which_counter[i];
        if( counter_uniq_id >= gctl->num_reg_events ){
            PAPIERROR("_sde_read(): Event at index %d does not correspond to a registered counter.\n",i);
            *events[i] = -1;
            continue;
        }

        sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
        if( NULL == counter ){
            PAPIERROR("_sde_read(): Event at index %d corresponds to a clobbered counter.\n",i);
            sde_ctl->counter[i] = -1;
            continue;
        }

        // If the counter represents a counter group then we need to read the values of all the counters in the group.
        if( NULL != counter->counter_group_head ){
            ret_val = sde_read_counter_group( counter, &(sde_ctl->counter[i]) );
            if( PAPI_OK != ret_val ){
                PAPIERROR("_sde_read(): Error occured when reading counter group: '%s'.\n",counter->name);
            }
            // we are done reading this one, move to the next.
            continue;
        }

        // Our convention is that read attempts on a placeholder will set the counter to "-1" to
        // signify semantically that there was an error, but the function will not return an error
        // to avoid breaking existing programs that do something funny when an error is returned.
        if( (NULL == counter->data) && (NULL == counter->func_ptr) && (NULL == counter->recorder_data) ){
            PAPIERROR("_sde_read(): Attempted read on a placeholder: '%s'.\n",counter->name);
            sde_ctl->counter[i] = -1;
            continue;
        }

        // If we are not dealing with a simple counter but with a recorder, we need to allocate
        // a contiguous buffer, copy all the recorded data in it, and return to the user a pointer
        // to this buffer cast as a long long.
        if( NULL != counter->recorder_data ){
            long long used_entries;
            size_t typesize;
            void *out_buffer;

            // At least the first chunk should have been allocated at creation.
            if( NULL == counter->recorder_data->exp_container[0] ){
                SUBDBG( "No space has been allocated for recorder %s\n",counter->name);
                sde_ctl->counter[i] = (long long)-1;
                continue;
            }

            used_entries = counter->recorder_data->used_entries;
            typesize = counter->recorder_data->typesize;

            // NOTE: After returning this buffer we loose track of it, so it's the user's responsibility to free it.
            out_buffer = malloc( used_entries*typesize );
            recorder_data_to_contiguous(counter, out_buffer);
            sde_ctl->counter[i] = (long long)out_buffer;

            continue;
        }

        ret_val = sde_hardware_read_and_store( counter, counter->previous_data, &(sde_ctl->counter[i]) );
     
        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_read(): Error occured when reading counter: '%s'.\n",counter->name);
        }
    }

    papi_sde_unlock();

    *events = sde_ctl->counter;

    return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
static int
_sde_write( hwd_context_t *ctx, hwd_control_state_t *ctl, long long *values )
{
    int i, ret_val = PAPI_OK;
    (void) ctx;
    (void) ctl;

    papisde_control_t *gctl = _papisde_global_control;   

    SUBDBG( "_sde_write... %p\n", ctx );

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;   

    // Lock before we access global data structures.
    papi_sde_lock();

    for( i = 0; i < sde_ctl->num_events; i++ ) {
        unsigned int counter_uniq_id = sde_ctl->which_counter[i];
        if( counter_uniq_id >= gctl->num_reg_events ){
            PAPIERROR("_sde_write(): Event at index %d does not correspond to a registered counter.\n",i);
            continue;
        }

        sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
        if( NULL == counter ){
            PAPIERROR("_sde_read(): Event at index %d corresponds to a clobbered counter.\n",i);
            continue;
        }

        // We currently do not support writing in counter groups.
        if( NULL != counter->counter_group_head ){
            SUBDBG("_sde_write(): Event '%s' corresponds to a counter group, and writing groups is not supported yet.\n",counter->name);
            continue;
        }

        if( NULL == counter->data ){
            if( NULL == counter->func_ptr ){
                // If we are not dealing with a simple counter but with a "recorder", which cannot be written, we have to error.
                if( NULL != counter->recorder_data ){
                    PAPIERROR("_sde_write(): Attempted write on a recorder: '%s'.\n",counter->name);
                }else{
                    PAPIERROR("_sde_write(): Attempted write on a placeholder: '%s'.\n",counter->name);
                }
            }else{
                PAPIERROR("_sde_write(): Attempted write on an event based on a callback function instead of a counter: '%s'.\n",counter->name);
            }
            continue;
        }

        ret_val = sde_hardware_write( counter, values[i] );
        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_write(): Error occured when writing counter: '%s'.\n",counter->name);
        }
    }

    papi_sde_unlock();

    return PAPI_OK;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
static int
_sde_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    int i;
    (void) ctx;

    SUBDBG( "_sde_reset ctx=%p ctrl=%p...\n", ctx, ctl );

    papisde_control_t *gctl = _papisde_global_control;   
    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ctl;   

    // Lock before we read num_reg_events and the hash-tables.
    papi_sde_lock();

    for( i = 0; i < sde_ctl->num_events; i++ ) {
        int ret_val;
        unsigned int counter_uniq_id = sde_ctl->which_counter[i];
        if( counter_uniq_id >= gctl->num_reg_events ){
            PAPIERROR("_sde_reset(): Event at index %d does not correspond to a registered counter.\n",i);
            continue;
        }

        sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
        if( NULL == counter ){
            PAPIERROR("_sde_reset(): Event at index %d corresponds to a clobbered counter.\n",i);
            continue;
        }

        // If the counter represents a counter group then we do not need to record the current value,
        // because when we read the real value we will keep track of all the previous values of the
        // individual counters (if they are DELTA), or not (if they are INSTANT)
        if( NULL != counter->counter_group_head ){
            // we are done with this one, move to the next.
            continue;
        }

        // Our convention is that read attempts on a placeholder will not return an error
        // to avoid breaking existing programs that do something funny when an error is returned.
        if( (NULL == counter->data) && (NULL == counter->func_ptr) ){
            PAPIERROR("_sde_reset(): Attempted read on a placeholder: %s.\n",counter->name);
            continue;
        }

        ret_val = sde_hardware_read_and_store( counter, 0, &(counter->previous_data) );
        if( PAPI_OK != ret_val ){
            PAPIERROR("_sde_reset(): Error occured when resetting counter: %s.\n",counter->name);
        }
    }

    papi_sde_unlock();

    return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
static int
_sde_shutdown_component(void)
{
    papisde_library_desc_t *curr_lib, *next_lib;

    SUBDBG( "sde_shutdown_component...\n" );
    papisde_control_t *gctl = _papisde_global_control; 

    if( NULL == gctl )
        return PAPI_OK;

    /* Free all the meta-data we allocated for libraries that are still active */
    curr_lib = gctl->lib_list_head;
    while(NULL != curr_lib){
        /* save a pointer to the next list element before we free the current */
        next_lib = curr_lib->next;

        if( NULL != curr_lib->libraryName ){
            free( curr_lib->libraryName );
        }
        free(curr_lib);

        curr_lib = next_lib;
    }

    return PAPI_OK;
}

/** Called at thread shutdown */
static int
_sde_shutdown_thread( hwd_context_t *ctx )
{

    (void) ctx;

    SUBDBG( "sde_shutdown_thread... %p\n", ctx );

    /* Last chance to clean up thread */

    return PAPI_OK;
}



/** This function sets various options in the component
  @param[in] ctx -- hardware context
  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, 
  PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param[in] option -- options to be set
  */
static int
_sde_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{

    (void) ctx;
    (void) code;
    (void) option;

    SUBDBG( "sde_ctl...\n" );

    return PAPI_OK;
}

/** This function has to set the bits needed to count different domains
  In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
  By default return PAPI_EINVAL if none of those are specified
  and PAPI_OK with success
  PAPI_DOM_USER is only user context is counted
  PAPI_DOM_KERNEL is only the Kernel/OS context is counted
  PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
  PAPI_DOM_ALL   is all of the domains
  */
static int
_sde_set_domain( hwd_control_state_t * cntrl, int domain )
{
    (void) cntrl;

    int found = 0;
    SUBDBG( "sde_set_domain...\n" );

    if ( PAPI_DOM_USER & domain ) {
        SUBDBG( " PAPI_DOM_USER\n" );
        found = 1;
    }
    if ( PAPI_DOM_KERNEL & domain ) {
        SUBDBG( " PAPI_DOM_KERNEL\n" );
        found = 1;
    }
    if ( PAPI_DOM_OTHER & domain ) {
        SUBDBG( " PAPI_DOM_OTHER\n" );
        found = 1;
    }
    if ( PAPI_DOM_ALL & domain ) {
        SUBDBG( " PAPI_DOM_ALL\n" );
        found = 1;
    }
    if ( !found )
        return ( PAPI_EINVAL );

    return PAPI_OK;
}


/**************************************************************/
/* Naming functions, used to translate event numbers to names */
/**************************************************************/


/** Enumerate Native Events
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 *  If your component has attribute masks then these need to
 *   be handled here as well.
 */
static int
_sde_ntv_enum_events( unsigned int *EventCode, int modifier )
{
    unsigned int curr_code, next_code;

    SUBDBG("_sde_ntv_enum_events begin\n\tEventCode=%u modifier=%d\n", *EventCode, modifier);

    papisde_control_t *gctl = _papisde_global_control; 
    if( NULL == gctl ){
        return PAPI_ENOEVNT;
    }

    switch ( modifier ) {

        /* return EventCode of first event */
        case PAPI_ENUM_FIRST:
            /* return the first event that we support */
            *EventCode = 0;
            return PAPI_OK;

        /* return EventCode of next available event */
        case PAPI_ENUM_EVENTS:
            curr_code = *EventCode & PAPI_NATIVE_AND_MASK;

            // Lock before we read num_reg_events and the hash-tables.
            papi_sde_lock();

            if( curr_code >= gctl->num_reg_events-1 ){
                papi_sde_unlock();
                return PAPI_ENOEVNT;
            }

            /*
             * We have to check the events which follow the current one, because unregistering
             * will create sparcity in the global SDE table, so we can't just return the next
             * index.
             */
            next_code = curr_code;
            do{
                next_code++;
                sde_counter_t *item = ht_lookup_by_id(gctl->all_reg_counters, next_code);
                if( (NULL != item) && (NULL != item->name) ){
                    *EventCode = next_code;
                    SUBDBG("Event name = %s (unique id = %d)\n", item->name, item->glb_uniq_id); 
                    papi_sde_unlock();
                    return PAPI_OK;
                }
            }while(next_code < gctl->num_reg_events);

            papi_sde_unlock();

            break;

        default:
            return PAPI_EINVAL;
    }

    return PAPI_EINVAL;
}

/** Takes a native event code and passes back the name 
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
static int
_sde_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
    papisde_control_t *gctl = _papisde_global_control; 
    unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;

    SUBDBG("_sde_ntv_code_to_name %u\n", code);

    // Lock before we read num_reg_events and the hash-tables.
    papi_sde_lock();

    if( (NULL == gctl) || (code > gctl->num_reg_events) ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, code);
    if( (NULL == counter) || (NULL == counter->name) ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }
    SUBDBG("Event name = %s (unique id = %d)\n", counter->name, counter->glb_uniq_id); 

    (void)strncpy( name, counter->name, len );  

    papi_sde_unlock();
    return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
static int
_sde_ntv_code_to_descr( unsigned int EventCode, char *descr, int len )
{
    unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;
    SUBDBG("_sde_ntv_code_to_descr %u\n", code);

    papisde_control_t *gctl = _papisde_global_control; 

    // Lock before we read num_reg_events and the hash-tables.
    papi_sde_lock();

    if( (NULL == gctl) || (code > gctl->num_reg_events) ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }

    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, code);
    if( (NULL == counter) || (NULL == counter->description) ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }
    SUBDBG("Event (unique id = %d) description: %s\n", counter->glb_uniq_id, counter->description); 

    (void)strncpy( descr, counter->description, len );  
    descr[len] = '\0';

    papi_sde_unlock();
    return PAPI_OK;
}

/** Takes a native event name and passes back the code 
 * @param event_name -- a pointer for the name to be copied to
 * @param event_code -- the native event code
 */
static int
_sde_ntv_name_to_code(const char *event_name, unsigned int *event_code ) 
{
    papisde_library_desc_t *lib_handle;
    char *pos, *tmp_lib_name;
    sde_counter_t *tmp_item = NULL;

    SUBDBG( "%s\n", event_name );

    papi_sde_lock();

    papisde_control_t *gctl = _papisde_global_control;

    // Let's see if the event has the library name as a prefix (as it should). Note that this is
    // the event name as it comes from the framework, so it should contain the library name, although
    // when the library registers an event counter it will not use the library name as part of the event name.
    tmp_lib_name = strdup(event_name);
    pos = strstr(tmp_lib_name, "::");
    if( NULL != pos ){ // Good, it does.
        *pos = '\0';

        if( NULL == gctl ){
            // If no library has initialized the library side of the component, and the application is already inquiring
            // about an event, let's initialize the component pretending to be the library which corresponds to this event.
            lib_handle = do_sde_init(tmp_lib_name);
            if(NULL == lib_handle){
                PAPIERROR("Unable to register library in SDE component.\n");
                papi_sde_unlock();
                return PAPI_ECMP;
            }
            gctl = _papisde_global_control;
        }else{
            int is_library_present = 0;
            // If the library side of the component has been initialized, then look for the library.
            lib_handle = gctl->lib_list_head;
            while(NULL != lib_handle){ // Look for the library.
                if( !strcmp(lib_handle->libraryName, tmp_lib_name) ){
                    // We found the library.
                    is_library_present = 1;
                    // Now, look for the event in the library.
                    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, event_name);
                    break;
                }
                lib_handle = lib_handle->next;
            }

            if( !is_library_present ){
                // If the library side of the component was initialized, but the specific library hasn't called
                // papi_sde_init() then we call it here to allocate the data structures.
                lib_handle = do_sde_init(tmp_lib_name);
                if(NULL == lib_handle){
                    PAPIERROR("Unable to register library in SDE component.\n");
                    papi_sde_unlock();
                    return PAPI_ECMP;
                }
             }
        }
        free(tmp_lib_name); // We don't need the library name any more.

        if( NULL != tmp_item ){
            SUBDBG("Found matching counter with global uniq id: %d in library: %s\n", tmp_item->glb_uniq_id, lib_handle->libraryName );
            *event_code = tmp_item->glb_uniq_id;
            papi_sde_unlock();
            return PAPI_OK;
        } else {
            SUBDBG("Did not find event %s in library %s. Registering a placeholder.\n", event_name, lib_handle->libraryName );  

            // Use the current number of registered events as the index of the new one, and increment it.
            unsigned int counter_uniq_id = gctl->num_reg_events++;
            gctl->num_live_events++;
            _sde_vector.cmp_info.num_native_events = gctl->num_live_events;

            // At this point in the code "lib_handle" contains a pointer to the data structure for this library whether
            // the actual library has been initialized or not.
            tmp_item = allocate_and_insert( lib_handle, event_name, counter_uniq_id, PAPI_SDE_RO, PAPI_SDE_long_long, NULL, NULL, NULL );
            if(NULL == tmp_item) {
                papi_sde_unlock();
                SUBDBG("Event %s does not exist in library %s and placeholder could not be inserted.\n", event_name, lib_handle->libraryName);
                return PAPI_ECMP;
            }
            *event_code = tmp_item->glb_uniq_id;
            papi_sde_unlock();
            return PAPI_OK;
        }
    }else{
        free(tmp_lib_name);
    }

    // If no library has initialized the component and we don't know a library name, then we have to return.
    if( NULL == gctl ){
        papi_sde_unlock();
        return PAPI_ENOEVNT;
    }

    // If the event name does not have the library name as a prefix, then we need to look in all the libraries for the event. However, in this case
    // we can _not_ register a placeholder because we don't know which library the event belongs to.
    lib_handle = gctl->lib_list_head;
    while(NULL != lib_handle){

        tmp_item = ht_lookup_by_name(lib_handle->lib_counters, event_name);
        if( NULL != tmp_item ){
            *event_code = tmp_item->glb_uniq_id;
            papi_sde_unlock();
            SUBDBG("Found matching counter with global uniq id: %d in library: %s\n", tmp_item->glb_uniq_id, lib_handle->libraryName );
            return PAPI_OK;
        } else {
            SUBDBG("Failed to find event %s in library %s. Looking in other libraries.\n", event_name, lib_handle->libraryName );  
        }

        lib_handle = lib_handle->next;
    }
    papi_sde_unlock();

    return PAPI_ENOEVNT;
}

#if defined(SDE_HAVE_OVERFLOW)
static void
_sde_dispatch_timer( int n, hwd_siginfo_t *info, void *uc)
{

    _papi_hwi_context_t hw_context;
    caddr_t address;
    ThreadInfo_t *thread;
    int i, cidx, retval, isHardware, slow_down, speed_up;
    int found_registered_counters, period_has_changed = 0;
    EventSetInfo_t *ESI;
    struct itimerspec its;
    long long overflow_vector = 0;
    sde_control_state_t *sde_ctl;
    papisde_control_t *gctl;

    (void) n;

    SUBDBG("SDE timer expired. Dispatching (papi internal) overflow handler\n");

    thread = _papi_hwi_lookup_thread( 0 );
    cidx = _sde_vector.cmp_info.CmpIdx;

    ESI = thread->running_eventset[cidx];
    // This holds only the number of events in the eventset that are set to overflow.
    int event_counter = ESI->overflow.event_counter;
    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;
    gctl = _papisde_global_control;

    retval = _papi_hwi_read( thread->context[cidx], ESI, ESI->sw_stop );
    if ( retval < PAPI_OK )
        return;

    slow_down = 0;
    speed_up = 0;
    found_registered_counters = 0;
    // Reset the deadline of counters which have exceeded the current deadline
    // and check if we need to slow down the frequency of the timer.
    for ( i = 0; i < event_counter; i++ ) {
        int papi_index = ESI->overflow.EventIndex[i];
        long long deadline, threshold, latest, previous, diff;
        unsigned int counter_uniq_id;

        counter_uniq_id = sde_ctl->which_counter[papi_index];
        sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
        if( (NULL == counter) || counter->is_created )
            continue;

        found_registered_counters = 1;

        latest = ESI->sw_stop[papi_index];
        deadline = ESI->overflow.deadline[i];
        threshold = ESI->overflow.threshold[i];

        // Find the increment from the previous measurement.
        previous = sde_ctl->previous_value[papi_index];

        // NOTE: The following code assumes that the counters are "long long". No other
        // NOTE: type will work correctly.
        diff = latest-previous;

        // If it's too small we need to slow down the timer, it it's
        // too large we need to speed up the timer.
        if( 30*diff < threshold ){
            slow_down = 1; // I.e., grow the sampling period
        }else if( 10*diff > threshold ){
            speed_up = 1; // I.e., shrink the sampling period
        }

        // Update the "previous" measurement to be the latest one.
        sde_ctl->previous_value[papi_index] = latest;

        // If this counter has exceeded the deadline, add it in the vector.
        if ( latest >= deadline ) {
            // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
            // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
            // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
            int pos = ESI->EventInfoArray[papi_index].pos[0];
            double rel_dist = 100.0*(double)(latest-deadline)/(double)threshold;
            SUBDBG ( "Event at index %d (and pos %d) has value %lld which exceeds deadline %lld (threshold %lld, accuracy %.2lf)\n",
                     papi_index, pos, latest, deadline, threshold, rel_dist);

            overflow_vector ^= ( long long ) 1 << pos;
            // We adjust the deadline in a way that it remains a multiple of threshold so we don't create an additive error.
            ESI->overflow.deadline[i] = threshold*(latest/threshold) + threshold;
        }
    }

    if( !found_registered_counters && sde_ctl->has_timer ){
        struct itimerspec zero_time;
        memset(&zero_time, 0, sizeof(struct itimerspec));
        if (timer_settime(sde_ctl->timerid, 0, &zero_time, NULL) == -1){
            PAPIERROR("timer_settime");
            timer_delete(sde_ctl->timerid);
            sde_ctl->has_timer = 0;
            return;
        }
        goto no_change_in_period;
    }

    // Since we potentially check multiple counters in the loop above, both conditions could be true (for different counter).
    // In this case, we give speed_up priority.
    if( speed_up )
        slow_down = 0;

    // If neither was set, there is nothing to do here.
    if( !speed_up && !slow_down )
        goto no_change_in_period;

    if( !sde_ctl->has_timer )
        goto no_change_in_period;

    // Get the current value of the timer.
    if( timer_gettime(sde_ctl->timerid, &its) == -1){
        PAPIERROR("timer_gettime() failed. Timer will not be modified.\n");
        goto no_change_in_period;
    }

    period_has_changed = 0;
    // We only reduce the period if it is above 131.6us, so it never drops below 100us.
    if( speed_up && (its.it_interval.tv_nsec > 131607) ){
        double new_val = (double)its.it_interval.tv_nsec;
        new_val /= 1.31607; // sqrt(sqrt(3)) = 1.316074
        its.it_value.tv_nsec = (int)new_val;
        its.it_interval.tv_nsec = its.it_value.tv_nsec;
        period_has_changed = 1;
        SUBDBG ("Timer will be sped up to %ld ns\n", its.it_value.tv_nsec);
    }

    // We only increase the period if it is below 75.9ms, so it never grows above 100ms.
    if( slow_down && (its.it_interval.tv_nsec < 75983800) ){
        double new_val = (double)its.it_interval.tv_nsec;
        new_val *= 1.31607; // sqrt(sqrt(3)) = 1.316074
        its.it_value.tv_nsec = (int)new_val;
        its.it_interval.tv_nsec = its.it_value.tv_nsec;
        period_has_changed = 1;
        SUBDBG ("Timer will be slowed down to %ld ns\n", its.it_value.tv_nsec);
    }

    if( !period_has_changed )
        goto no_change_in_period;

    if (timer_settime(sde_ctl->timerid, 0, &its, NULL) == -1){
        PAPIERROR("timer_settime() failed when modifying PAPI internal timer. This might have broken overflow support for this eventset.\n");
        goto no_change_in_period;
    }

no_change_in_period:

    // If none of the events exceeded their deadline, there is nothing else to do.
    if( 0 == overflow_vector ){
        return;
    }

    if ( (NULL== thread) || (NULL == thread->running_eventset[cidx]) || (0 == thread->running_eventset[cidx]->overflow.flags) ){
        PAPIERROR( "_sde_dispatch_timer(): 'Can not access overflow flags'");
        return;
    }

    hw_context.si = info;
    hw_context.ucontext = ( hwd_ucontext_t * ) uc;

    address = GET_OVERFLOW_ADDRESS( hw_context );

    int genOverflowBit = 0;

    _papi_hwi_dispatch_overflow_signal( ( void * ) &hw_context, address, &isHardware, overflow_vector, genOverflowBit, &thread, cidx );

   return;
}
#endif // defined(SDE_HAVE_OVERFLOW)

#if defined(SDE_HAVE_OVERFLOW)
static void
invoke_user_handler(sde_counter_t *cntr_handle){
    EventSetInfo_t *ESI;
    int i, cidx;
    ThreadInfo_t *thread;
    sde_control_state_t *sde_ctl;
    _papi_hwi_context_t hw_context;
    ucontext_t uc;
    caddr_t address;
    long long overflow_vector;

    if( NULL == cntr_handle )
        return;

    thread = _papi_hwi_lookup_thread( 0 );
    cidx = _sde_vector.cmp_info.CmpIdx;
    ESI = thread->running_eventset[cidx];

    // checking again, just to be sure.
    if( !(ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE) ) {
        return;
    }

    sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;

    papisde_control_t *gctl = _papisde_global_control; 

    if( NULL == gctl ){
        return;
    }

    // This path comes from papi_sde_inc_counter() which increment _ONLY_ one counter, so we don't
    // need to check if any others have overflown.
    overflow_vector = 0;
    for( i = 0; i < sde_ctl->num_events; i++ ) {
        unsigned int counter_uniq_id = sde_ctl->which_counter[i];

        if( counter_uniq_id == cntr_handle->glb_uniq_id ){
            // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
            // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
            // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
            int pos = ESI->EventInfoArray[i].pos[0];
            if( pos == -1 ){
               PAPIERROR( "The PAPI framework considers this event removed from the eventset, but the component does not\n");
               return;
            }
            overflow_vector = ( long long ) 1 << pos;
        }
    }

    getcontext( &uc );
    hw_context.ucontext = &uc;
    hw_context.si = NULL;
    address = GET_OVERFLOW_ADDRESS( hw_context );

    ESI->overflow.handler( ESI->EventSetIndex, ( void * ) address, overflow_vector, hw_context.ucontext );
    return;
}
#endif // SDE_HAVE_OVERFLOW

#if defined(SDE_HAVE_OVERFLOW)
static int
_sde_set_overflow( EventSetInfo_t *ESI, int EventIndex, int threshold ){

    (void)ESI;
    (void)EventIndex;
    (void)threshold;

    SUBDBG("_sde_set_overflow(%d, %d).\n",EventIndex, threshold);

    sde_control_state_t *sde_ctl = ( sde_control_state_t * ) ESI->ctl_state;
    papisde_control_t *gctl = _papisde_global_control; 

    // pos[0] holds the first among the native events that compose the given event. If it is a derived event,
    // then it might be made up of multiple native events, but this is a CPU component concept. The SDE component
    // does not have derived events (the groups are first class citizens, they don't have multiple pos[] entries).
    int pos = ESI->EventInfoArray[EventIndex].pos[0];
    unsigned int counter_uniq_id = sde_ctl->which_counter[pos];
    sde_counter_t *counter = ht_lookup_by_id(gctl->all_reg_counters, counter_uniq_id);
    // If the counter is created then we will check for overflow every time its value gets updated, we don't need to poll.
    // That is in cases c[1-3]
    if( counter->is_created )
        return PAPI_OK;

    // We do not want to overflow on recorders, because we don't even know what this means (unless we check the number of recorder entries.)
    if( (NULL != counter->recorder_data) && (threshold > 0) ){
        return PAPI_EINVAL;
    }

    // If we still don't know what type the counter is, then we are _not_ in r[1-3] so we can't create a timer here.
    if( (NULL == counter->data) && (NULL == counter->func_ptr) && (threshold > 0) ){
        SUBDBG("Event is a placeholder (it has not been registered by a library yet), so we cannot start overflow, but we can remember it.\n");
        counter->overflow = 1;
        return PAPI_OK;
    }
  
    // A threshold of zero indicates that overflowing is not needed anymore.
    if( 0 == threshold ){
        counter->overflow = 0;
        // If we had a timer (if the counter was created we wouldn't have one) then delete it.
        if( sde_ctl->has_timer )
            timer_delete(sde_ctl->timerid);
        sde_ctl->has_timer = 0;
    }else{
        // If we are here we are in r[1-3] so we can create the timer
        return set_timer_for_overflow(sde_ctl);
    }

    return PAPI_OK;
}

/**
 *  This code assumes that it is called _ONLY_ for registered counters,
 *  and that is why it sets has_timer to REGISTERED_EVENT_MASK
 */
static int 
set_timer_for_overflow( sde_control_state_t *sde_ctl ){
    int signo, sig_offset;
    struct sigevent sigev;
    struct sigaction sa;

    sig_offset = 0;

    // Choose a new real-time signal
    signo = SIGRTMIN+sig_offset;
    if(signo > SIGRTMAX){
        PAPIERROR("_sde_set_overflow(): Unable to create new timer due to large number of existing timers. Overflowing will not be activated for the current event.\n");
        return PAPI_ECMP;
    }

    // setup the signal handler
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = _sde_dispatch_timer;
    sigemptyset(&sa.sa_mask);
    if (sigaction(signo, &sa, NULL) == -1){
        PAPIERROR("sigaction");
        return PAPI_ECMP;
    }

    // create the timer
    sigev.sigev_notify = SIGEV_SIGNAL;
    sigev.sigev_signo = signo;
    sigev.sigev_value.sival_ptr = &(sde_ctl->timerid);
    if (timer_create(CLOCK_REALTIME, &sigev, &(sde_ctl->timerid)) == -1){
        PAPIERROR("timer_create");
        return PAPI_ECMP;
    }
    sde_ctl->has_timer |= REGISTERED_EVENT_MASK;

    return PAPI_OK;
}

#endif // defined(SDE_HAVE_OVERFLOW)


/** Vector that points to entry points for our component */
papi_vector_t _sde_vector = {
    .cmp_info = {
        /* default component information */
        /* (unspecified values are initialized to 0) */
        /* we explicitly set them to zero in this sde */
        /* to show what settings are available            */

        .name = "sde",
        .short_name = "sde",
        .description = "Software Defined Events (SDE) component",
        .version = "1.15",
        .support_version = "n/a",
        .kernel_version = "n/a",
        .num_cntrs =               SDE_MAX_SIMULTANEOUS_COUNTERS, 
        .num_mpx_cntrs =           SDE_MAX_SIMULTANEOUS_COUNTERS,
        .default_domain =          PAPI_DOM_USER,
        .available_domains =       PAPI_DOM_USER,
        .default_granularity =     PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig =       PAPI_INT_SIGNAL,
        .hardware_intr =           1,

        /* component specific cmp_info initializations */
    },

    /* sizes of framework-opaque component-private structures */
    .size = {
        /* once per thread */
        .context = sizeof ( sde_context_t ),
        /* once per eventset */
        .control_state = sizeof ( sde_control_state_t ),
        /* ?? */
        .reg_value = sizeof ( sde_register_t ),
        /* ?? */
        .reg_alloc = sizeof ( sde_reg_alloc_t ),
    },

    /* function pointers */
    /* by default they are set to NULL */

    /* Used for general PAPI interactions */
    .start =                _sde_start,
    .stop =                 _sde_stop,
    .read =                 _sde_read,
    .reset =                _sde_reset, 
    .write =                _sde_write,
    .init_component =       _sde_init_component,    
    .init_thread =          _sde_init_thread,
    .init_control_state =   _sde_init_control_state,
    .update_control_state = _sde_update_control_state,  
    .ctl =                  _sde_ctl,   
    .shutdown_thread =      _sde_shutdown_thread,
    .shutdown_component =   _sde_shutdown_component,
    .set_domain =           _sde_set_domain,
    /* .cleanup_eventset =     NULL, */
    /* called in add_native_events() */
    /* .allocate_registers =   NULL, */

    /* Used for overflow/profiling */
#if defined(SDE_HAVE_OVERFLOW)
    .dispatch_timer =       _sde_dispatch_timer,
    .set_overflow =         _sde_set_overflow,
#endif // defined(SDE_HAVE_OVERFLOW)
    /* .get_overflow_address = NULL, */
    /* .stop_profiling =       NULL, */
    /* .set_profile =          NULL, */

    /* ??? */
    /* .user =                 NULL, */

    /* Name Mapping Functions */
    .ntv_enum_events =   _sde_ntv_enum_events,
    .ntv_code_to_name =  _sde_ntv_code_to_name,
    .ntv_code_to_descr = _sde_ntv_code_to_descr,
    /* if .ntv_name_to_code not available, PAPI emulates  */
    /* it by enumerating all events and looking manually  */
    .ntv_name_to_code  = _sde_ntv_name_to_code,


    /* These are only used by _papi_hwi_get_native_event_info() */
    /* Which currently only uses the info for printing native   */
    /* event info, not for any sort of internal use.            */
    /* .ntv_code_to_bits =  NULL, */

};

