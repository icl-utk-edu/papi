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

int cuptiu_event_table_create_with_size(int size, cuptiu_event_table_t **pevt_table)
{
    cuptiu_event_table_t *evt_table = (cuptiu_event_table_t *) papi_malloc(sizeof(cuptiu_event_table_t));
    if (evt_table == NULL) {
        goto fn_fail;
    }
    evt_table->capacity = size;
    evt_table->count = 0;
    evt_table->evts = (cuptiu_event_t *) papi_calloc (evt_table->capacity, sizeof(cuptiu_event_t));
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

int cuptiu_event_table_create(cuptiu_event_table_t **pevt_table)
{
    return cuptiu_event_table_create_with_size(ADDED_EVENTS_INITIAL_CAPACITY, pevt_table);
}

static int reallocate_array(cuptiu_event_table_t *evt_table)
{
    evt_table->capacity *= 2;
    evt_table->evts = (cuptiu_event_t *) papi_realloc(evt_table->evts, evt_table->capacity * sizeof(cuptiu_event_t));
    if (evt_table == NULL) {
        ERRDBG("Failed to expand event_table array.\n");
        return PAPI_ENOMEM;
    }
    /* Rehash all the table entries */
    unsigned int i;
    for (i=0; i<evt_table->count; i++) {
        if (HTABLE_SUCCESS != htable_insert(evt_table->htable, evt_table->evts[i].name, &(evt_table->evts[i])))
            return PAPI_ENOMEM;
    }
    return PAPI_OK;
}

int cuptiu_event_table_insert_record(cuptiu_event_table_t *evt_table, const char *evt_name, unsigned int evt_code, int evt_pos)
{
    int errno = PAPI_OK;

    /* Allocate twice the space if running out */
    if (evt_table->count >= evt_table->capacity) {
        errno = reallocate_array(evt_table);
        if (errno != PAPI_OK)
            goto fn_exit;
    }
    /* Insert record in array */
    strcpy(evt_table->evts[evt_table->count].name, evt_name);
    evt_table->evts[evt_table->count].desc[0] = '\0';
    evt_table->evts[evt_table->count].evt_code = evt_code;
    evt_table->evts[evt_table->count].evt_pos = evt_pos;
    /* Insert entry in string hash table */
    if (HTABLE_SUCCESS != htable_insert(evt_table->htable, evt_name, &(evt_table->evts[evt_table->count]))) {
        return PAPI_ENOMEM;
    }
    evt_table->count ++;
fn_exit:
    return errno;
}

int cuptiu_event_table_select_by_idx(cuptiu_event_table_t *src, int count, int *idcs, cuptiu_event_table_t **pevt_names)
{
    int papi_errno = PAPI_OK;
    if (count <= 0 || count > (int) src->count) {
        papi_errno = PAPI_EINVAL;
        goto fn_fail;
    }
    cuptiu_event_table_t *target;
    papi_errno = cuptiu_event_table_create_with_size(count, &target);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }
    int i;
    for (i = 0; i < count; i++) {
        if (cuptiu_event_table_insert_record(target, src->evts[idcs[i]].name, src->evts[idcs[i]].evt_code, src->evts[idcs[i]].evt_pos) != PAPI_OK) {
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
    int errno;

    cuptiu_event_t *evt_rec = NULL;
    errno = htable_find(evt_table->htable, evt_name, (void **) &evt_rec);
    if (errno == HTABLE_SUCCESS) {
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

int cuptiu_event_name_tokenize(const char *name, char *nv_name, int *gpuid)
{
    /* Resolve the nvidia name and gpu number from PAPI event name */
    int numchars;

    if (nv_name == NULL) {
        return PAPI_EINVAL;
    }
    const char token[] = ":device=";
    const int tok_len = 8;
    char *rest;

    char *getdevstr = strstr(name, token);
    if (getdevstr == NULL) {
        ERRDBG("Event name does not contain device number.\n");
        return PAPI_EINVAL;
    }
    getdevstr += tok_len;
    *gpuid = strtol(getdevstr, &rest, 10);
    numchars = strlen(name) - strlen(getdevstr) - tok_len;
    memcpy(nv_name, name, numchars);
    nv_name[numchars] = '\0';

    return PAPI_OK;
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
