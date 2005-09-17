#ifndef _PAPI_SOLARIS_H
#define _PAPI_SOLARIS_H

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <libgen.h>
#include <limits.h>
#include <synch.h>
#include <procfs.h>
#include <libcpc.h>
#include <libgen.h>
#include <ctype.h>
#include <errno.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/processor.h>
#include <sys/procset.h>
#include <sys/ucontext.h>
#include <syms.h>
#include <dlfcn.h>
#include <sys/stat.h>

/* Assembler prototypes */

extern void cpu_sync(void);
extern unsigned long_long get_tick(void);
extern caddr_t _start, _end, _etext, _edata;

extern rwlock_t lock[PAPI_MAX_LOCK];

#define _papi_hwd_lock(lck) rw_wrlock(&lock[lck]);

#define _papi_hwd_unlock(lck)   rw_unlock(&lock[lck]);

#define inline_static inline static

#endif

