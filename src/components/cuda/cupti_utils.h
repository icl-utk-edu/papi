/**
 * @file    cupti_utils.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_UTILS_H__
#define __CUPTI_UTILS_H__

#include <papi.h>

typedef struct event_record_s {
    char name[PAPI_2MAX_STR_LEN];
    char desc[PAPI_2MAX_STR_LEN];
    int gpu_id;
    unsigned int evt_code;
    /* index of added event */
    unsigned int evt_pos;
    int num_dep;
    double value;
    /* API specific details */
    void *info;
} cuptiu_event_t;

typedef struct event_table_s {
    cuptiu_event_t *evts;
    unsigned int count;
    unsigned int capacity;
    void *htable;
} cuptiu_event_table_t;

/* These functions form a simple API to handle dynamic list of strings */
int cuptiu_event_table_create(cuptiu_event_table_t **pevt_table);
int cuptiu_event_table_create_with_size(int size, cuptiu_event_table_t **pevt_table);
int cuptiu_event_table_insert_record(cuptiu_event_table_t *evt_table, const char *evt_name, unsigned int evt_code, int evt_pos);
int cuptiu_event_table_select_by_idx(cuptiu_event_table_t *src, int count, int *idcs, cuptiu_event_table_t **pevt_names);
int cuptiu_event_table_find_name(cuptiu_event_table_t *evt_table, const char *evt_name, cuptiu_event_t **found_rec);
void cuptiu_event_table_destroy(cuptiu_event_table_t **pevt_table);
int cuptiu_event_name_tokenize(const char *name, char *nv_name, int *gpuid);

/* Utility to locate a file in a given path */
#define CUPTIU_MAX_FILES 100
int cuptiu_files_search_in_path(const char *file_name, const char *search_path, char **file_paths);

#endif  /* __CUPTI_UTILS_H__ */
