#ifndef _PREPARE_ARRAY_
#define _PREPARE_ARRAY_

#include <stdint.h>

#define RANDOM 0x2
#define SECRND 0x3
#define SEQUEN 0x4

int prepareArray(uintptr_t *array, int len, int stride, long secSize);

#endif
