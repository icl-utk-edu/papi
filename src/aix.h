#ifndef _PAPI_AIX  /* _PAPI_AIX */
#define _PAPI_AIX

#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <libgen.h>
#include <sys/systemcfg.h>
#include <sys/processor.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <procinfo.h>
#include <sys/atomic_op.h>

#define ANY_THREAD_GETS_SIGNAL
#include <dlfcn.h>

#include "pmapi.h"
#define POWER_MAX_COUNTERS MAX_COUNTERS
#define MAX_COUNTER_TERMS MAX_COUNTERS
#define INVALID_EVENT -2
#define POWER_MAX_COUNTERS_MAPPING 8

#include "papi.h"
#include "papi_preset.h"


extern _etext;
extern _edata;
extern _end;
extern _data;

/* globals */
pm_info_t pminfo;

/* prototypes */

#endif  /* _PAPI_AIX */
