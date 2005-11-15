#ifndef _PAPI_IRIX_MIPS_H
#define _PAPI_IRIX_MIPS_H

#include "papi_sys_headers.h"

#include <invent.h>
#include <time.h>
#include <task.h>
#include <rld_interface.h>
#include <sys/cpu.h>
#include <sys/sbd.h>
#include <sys/hwperftypes.h>
#include <sys/hwperfmacros.h>
#include <sys/systeminfo.h>

#define inline_static static

extern int _etext[], _ftext[];
extern int _edata[], _fdata[];
extern int _fbss[], _end[];

extern volatile int lock[PAPI_MAX_LOCK];

#define _papi_hwd_lock(lck)         \
{                                   \
  while (__lock_test_and_set(&lock[lck],1) != 0) { ; } \
}

#define _papi_hwd_unlock(lck) {__lock_release(&lock[lck]);}


#endif
