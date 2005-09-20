#ifndef _PAPI_UNICOSMP_H
#define _PAPI_UNICOSMP_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/siginfo.h>
#include <sys/ucontext.h>
#include <sys/hwperftypes.h>
#include <sys/hwperfmacros.h>
#include <mutex.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/sysmp.h>
#include <sys/sysinfo.h>
#include <sys/procfs.h>
#include <sys/times.h>
#include <sys/errno.h>
#include <assert.h>
#include <invent.h>
#include <intrinsics.h>
#include <errno.h>


#define inline_static static

#ifdef MSP
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t) (((hwd_ucontext_t *)(ctx.ucontext))[0].uc_mcontext.scontext[CTX_EPC]);
#else
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t) (((hwd_ucontext_t *)(ctx.ucontext))[0].uc_mcontext.scontext[CTX_EPC]);
#endif

extern int _etext[], _ftext[];
extern int _edata[], _fdata[];
extern int _fbss[], _end[];

/* This will always aquire a lock, while acquire_lock is not
 * guaranteed, while spin_lock states:
 * If the lock isnot immediately available, the calling process will either
 * spin (busywait) or be suspended until the lock becomes available.
 * I will try that first and check the performance and load -KSL
 */
extern mutexlock_t x1_lck[PAPI_MAX_LOCK];
/* This needs to be fixed */
#define _papi_hwd_lock(idx) {spin_lock(&x1_lck[idx]);}
#define _papi_hwd_unlock(idx) {release_lock(&x1_lck[idx]);}
#endif
