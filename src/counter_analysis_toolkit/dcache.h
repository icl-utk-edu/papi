#ifndef _DATA_CACHE_
#define _DATA_CACHE_

#include "caches.h"
#include "prepareArray.h"
#include "timing_kernels.h"

int varyBufferSizes(int *values, double *rslts, double *counter, int line_size_in_bytes, float pages_per_block, int detect_size, int readwrite);
void *thread_main(void *arg);
void d_cache_driver(char* papi_event_name, int max_iter, char* outdir, int detect_size, int readwrite);
void d_cache_test(int pattern, int max_iter, int line_size_in_bytes, float pages_per_block, char* papi_event_name, char* papiFileName, int detect_size, int readwrite, FILE* ofp_papi, FILE* ofp);

#endif
