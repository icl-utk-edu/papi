#ifndef _PAPI_DEFINES_H
#define _PAPI_DEFINES_H

#include "papi.h"

/* created to break a circular dependency between papi_internal.h and perfctr-x86.h */
#define NUM_INNER_LOCK  9
#define PAPI_MAX_LOCK   (NUM_INNER_LOCK + PAPI_NUM_LOCK)

#endif
