/**
 * @file    lcuda_common.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __LCUDA_COMMON_H__
#define __LCUDA_COMMON_H__

#include <papi.h>

typedef struct event_record_s {
    char name[PAPI_2MAX_STR_LEN];
    char desc[PAPI_2MAX_STR_LEN];
    int gpu_id;
    unsigned int evt_code;
    // index of added event
    unsigned int evt_pos;
    int num_dep;
    double value;
    // API specific details
    void *info;
} ntv_event_t;

typedef struct event_table_s {
    ntv_event_t *evts;
    unsigned int count;
    unsigned int capacity;
    void *htable;
} ntv_event_table_t;

// These functions form a simple API to handle dynamic list of strings
ntv_event_table_t *initialize_dynamic_event_list(void);
ntv_event_table_t *initialize_dynamic_event_list_size(int size);
int insert_event_record(ntv_event_table_t *evt_table, const char *evt_name, unsigned int evt_code, int evt_pos);
ntv_event_table_t *select_by_idx(ntv_event_table_t *src, int count, int *idcs);
int find_event_name(ntv_event_table_t *evt_table, const char *evt_name, ntv_event_t **found_rec);
void free_event_name_list(ntv_event_table_t **pevt_table);
int tokenize_event_name(const char *name, char *nv_name, int *gpuid);

// Functions to track the occupancy of gpu counters in event sets
typedef int64_t gpu_occupancy_t;

int devmask_check_and_acquire(ntv_event_table_t *evt_table);
int devmask_release(ntv_event_table_t *evt_table);

// Utility to locate a file in a given path
#define MAX_FILES 100
int search_files_in_path(const char *file_name, const char *search_path, char **file_paths);

#endif  // __LCUDA_COMMON_H__
