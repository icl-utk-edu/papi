#ifndef _DATA_CACHE_
#define _DATA_CACHE_

#include <stdio.h>
#include <omp.h>
#include "hw_desc.h"
#include "params.h"

int varyBufferSizes(int *values, double **rslts, double **counter, hw_desc_t *hw_desc, int line_size_in_bytes, float pages_per_block, int pattern, int latency_only, int mode, int ONT);
int get_thread_count();
void print_core_affinities(FILE *ofp);
void d_cache_driver(char* papi_event_name, cat_params_t params, hw_desc_t *hw_desc, int latency_only, int mode);
int d_cache_test(int pattern, int max_iter, hw_desc_t *hw_desc, int stride_in_bytes, float pages_per_block, char* papi_event_name, int latency_only, int mode, FILE* ofp);

#endif
