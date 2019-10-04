#ifndef _TIMING_KERNELS_
#define _TIMING_KERNELS_

#include <stdint.h>

#include "caches.h"

#define N_1   p = (uintptr_t *)*p;
#define N_2   N_1 N_1
#define N_16  N_2 N_2 N_2 N_2 N_2 N_2 N_2 N_2
#define N_128 N_16 N_16 N_16 N_16 N_16 N_16 N_16 N_16

#define NW_1   p = (uintptr_t *)*p; *(p+max_size) = 3;
#define NW_2   NW_1 NW_1
#define NW_16  NW_2 NW_2 NW_2 NW_2 NW_2 NW_2 NW_2 NW_2
#define NW_128 NW_16 NW_16 NW_16 NW_16 NW_16 NW_16 NW_16 NW_16

run_output_t probeBufferSize(int l1_size, int line_size, float pageCountPerBlock, uintptr_t *v, uintptr_t *rslt, int detect_size, int readwrite);
void error_handler(int e, int line);

#endif
