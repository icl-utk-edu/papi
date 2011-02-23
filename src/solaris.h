#ifndef _PAPI_SOLARIS_H
#define _PAPI_SOLARIS_H

#include "papi_sys_headers.h"

#include <synch.h>
#include <procfs.h>
#include <libcpc.h>
#include <sys/procset.h>
#include <syms.h>

/* Assembler prototypes */

extern void cpu_sync( void );
extern unsigned long long get_tick( void );
extern caddr_t _start, _end, _etext, _edata;

extern rwlock_t lock[PAPI_MAX_LOCK];

#define _papi_hwd_lock(lck) rw_wrlock(&lock[lck]);

#define _papi_hwd_unlock(lck)   rw_unlock(&lock[lck]);

#endif
