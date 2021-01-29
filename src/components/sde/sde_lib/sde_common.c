/**
 * @file    sde_common.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This is a collection of utility functions that are needed by both the
 *  standalone SDE library and the SDE component in PAPI.
 */

#include <stdarg.h>
#include "sde_common.h"

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
            SDE_ERROR("ht_delete(): the hash table is clobbered.\n");
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

