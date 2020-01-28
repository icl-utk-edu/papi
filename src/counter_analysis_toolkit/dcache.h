#ifndef _DATA_CACHE_
#define _DATA_CACHE_

#include <stdio.h>

int varyBufferSizes(int *values, double *rslts, double *counter, int line_size_in_bytes, float pages_per_block, int latency_only, int mode);
void *thread_main(void *arg);
void d_cache_driver(char* papi_event_name, int max_iter, char* outdir, int latency_only, int mode, int show_progress);
void d_cache_test(int pattern, int max_iter, int line_size_in_bytes, float pages_per_block, char* papi_event_name, int latency_only, int mode, FILE* ofp);

#endif
