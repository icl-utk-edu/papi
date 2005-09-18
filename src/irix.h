#ifndef _PAPI_IRIX_MIPS_H
#define _PAPI_IRIX_MIPS_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <invent.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <task.h>
#include <ctype.h>
#include <assert.h>
#include <rld_interface.h>
#include <dlfcn.h>
#include <errno.h>
#include <sys/times.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/procfs.h>
#include <sys/cpu.h>
#include <sys/sysmp.h>
#include <sys/sbd.h>
#include <sys/hwperftypes.h>
#include <sys/hwperfmacros.h>
#include <sys/syscall.h>
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
