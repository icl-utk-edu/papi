/**
 * @file    common.h
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 */

#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"

#define PAPI_ROCM_MAX_COUNTERS (512)

extern unsigned _rocm_lock;

#endif /* End of __COMMON_H__ */
