/**
 * @file    cupti_utils.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <string.h>
#include "papi_memory.h"

#include "cupti_utils.h"
#include "htable.h"
#include "lcuda_debug.h"

#define ADDED_EVENTS_INITIAL_CAPACITY 64

int cuptiu_event_table_create_init_capacity(int capacity, int sizeof_rec, cuptiu_event_table_t **pevt_table)
{
    cuptiu_event_table_t *evt_table = (cuptiu_event_table_t *) papi_malloc(sizeof(cuptiu_event_table_t));
    if (evt_table == NULL) {
        goto fn_fail;
    }
    evt_table->sizeof_rec = sizeof_rec;
    evt_table->capacity = capacity;
    evt_table->count = 0;
    evt_table->evts = papi_calloc (evt_table->capacity, evt_table->sizeof_rec);
    if (evt_table->evts == NULL) {
        cuptiu_event_table_destroy(&evt_table);
        ERRDBG("Error allocating memory for dynamic event table.\n");
        goto fn_fail;
    }
    if (htable_init(&(evt_table->htable)) != HTABLE_SUCCESS) {
        cuptiu_event_table_destroy(&evt_table);
        goto fn_fail;
    }
    *pevt_table = evt_table;
    return PAPI_OK;
fn_fail:
    *pevt_table = NULL;
    return PAPI_ENOMEM;
}

int cuptiu_event_table_create(int sizeof_rec, cuptiu_event_table_t **pevt_table)
{
    return cuptiu_event_table_create_init_capacity(ADDED_EVENTS_INITIAL_CAPACITY, sizeof_rec, pevt_table);
}

int cuptiu_event_table_get_item(cuptiu_event_table_t *evt_table, int evt_idx, cuptiu_event_t **record)
{
    if (evt_idx >= (int) evt_table->count) {
        *record = NULL;
        return PAPI_EINVAL;
    }
    if (evt_idx == -1) {
        evt_idx = evt_table->count - 1;
    }
    *record = evt_table->evts + evt_idx * evt_table->sizeof_rec;
    return PAPI_OK;
}

static int reallocate_array(cuptiu_event_table_t *evt_table)
{
    int papi_errno = PAPI_OK;
    evt_table->capacity *= 2;
    evt_table->evts = papi_realloc(evt_table->evts, evt_table->capacity * evt_table->sizeof_rec);
    if (evt_table == NULL) {
        ERRDBG("Failed to expand event_table array.\n");
        papi_errno = PAPI_ENOMEM;
        goto fn_exit;
    }
    /* Rehash all the table entries */
    unsigned int i;
    cuptiu_event_t *evt_rec;
    for (i=0; i<evt_table->count; i++) {
        papi_errno = cuptiu_event_table_get_item(evt_table, i, &evt_rec);
        if (papi_errno != PAPI_OK) {
            papi_errno = PAPI_EINVAL;
            goto fn_exit;
        }
        if (HTABLE_SUCCESS != htable_insert(evt_table->htable, evt_rec->name, evt_rec)) {
            papi_errno = PAPI_ENOMEM;
            goto fn_exit;
        }
    }
fn_exit:
    return papi_errno;
}

int cuptiu_event_table_insert_record(cuptiu_event_table_t *evt_table, const char *evt_name, unsigned int evt_code, int evt_pos)
{
    int papi_errno = PAPI_OK;

    /* Allocate twice the space if running out */
    if (evt_table->count >= evt_table->capacity) {
        papi_errno = reallocate_array(evt_table);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
    }
    /* Insert record in array */
    cuptiu_event_t *evt_rec = evt_table->evts + evt_table->count * evt_table->sizeof_rec;
    strcpy(evt_rec->name, evt_name);
    evt_rec->evt_code = evt_code;
    evt_rec->evt_pos = evt_pos;
    /* Insert entry in string hash table */
    if (HTABLE_SUCCESS != htable_insert(evt_table->htable, evt_name, evt_rec)) {
        return PAPI_ENOMEM;
    }
    evt_table->count ++;
fn_exit:
    return papi_errno;
}

int cuptiu_event_table_select_by_idx(cuptiu_event_table_t *src, int count, int *idcs, cuptiu_event_table_t **pevt_names)
{
    int papi_errno = PAPI_OK;
    if (count <= 0 || count > (int) src->count) {
        papi_errno = PAPI_EINVAL;
        goto fn_fail;
    }
    cuptiu_event_table_t *target;
    papi_errno = cuptiu_event_table_create_init_capacity(count, src->sizeof_rec, &target);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    int i;
    cuptiu_event_t *evt_rec;
    for (i = 0; i < count; i++) {
        papi_errno = cuptiu_event_table_get_item(src, idcs[i], &evt_rec);
        if (papi_errno != PAPI_OK) {
            cuptiu_event_table_destroy(&target);
            goto fn_fail;
        }
        papi_errno = cuptiu_event_table_insert_record(target, evt_rec->name, evt_rec->evt_code, evt_rec->evt_pos);
        if (papi_errno != PAPI_OK) {
            cuptiu_event_table_destroy(&target);
            goto fn_fail;
        }
    }
    *pevt_names = target;
fn_exit:
    return papi_errno;
fn_fail:
    *pevt_names = NULL;
    goto fn_exit;
}

int cuptiu_event_table_find_name(cuptiu_event_table_t *evt_table, const char *evt_name, cuptiu_event_t **found_rec)
{
    int papi_errno;
    cuptiu_event_t *evt_rec = NULL;
    papi_errno = htable_find(evt_table->htable, evt_name, (void **) &evt_rec);
    if (papi_errno == HTABLE_SUCCESS) {
        *found_rec = evt_rec;
        return PAPI_OK;
    }
    return PAPI_ENOEVNT;
}

void cuptiu_event_table_destroy(cuptiu_event_table_t **pevt_table)
{
    cuptiu_event_table_t *evt_table = *pevt_table;
    if (evt_table == NULL)
        return;
    if (evt_table->evts) {
        papi_free(evt_table->evts);
        evt_table->evts = NULL;
    }
    if (evt_table->htable) {
        htable_shutdown(evt_table->htable);
        evt_table->htable = NULL;
    }
    papi_free(evt_table);
    *pevt_table = NULL;
}

int cuptiu_files_search_in_path(const char *file_name, const char *search_path, char **file_paths)
{
    char path[PATH_MAX];
    char command[PATH_MAX];
    snprintf(command, PATH_MAX, "find %s -name %s", search_path, file_name);

    FILE *fp;
    fp = popen(command, "r");
    if (fp == NULL) {
        ERRDBG("Failed to run system command find using popen.\n");
        return -1;
    }

    int count = 0;
    while (fgets(path, PATH_MAX, fp) != NULL) {
        path[strcspn(path, "\n")] = 0;
        file_paths[count] = strdup(path);
        count++;
        if (count >= CUPTIU_MAX_FILES) {
            break;
        }
    }

    pclose(fp);
    if (count == 0) {
        ERRDBG("%s not found in path PAPI_CUDA_ROOT.\n", file_name);
    }
    return count;
}
