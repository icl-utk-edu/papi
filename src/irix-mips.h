#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <invent.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <task.h>
#include <assert.h>
#include <unistd.h>
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

#include "papi.h"

#define MAX_COUNTERS HWPERF_EVENTMAX
#define MAX_COUNTER_TERMS 4
#define MAX_NATIVE_EVENT 32
#define PAPI_MAX_NATIVE_EVENTS MAX_NATIVE_EVENT
#include "papi_preset.h"

typedef int hwd_register_t;
typedef int hwd_reg_alloc_t;

typedef struct hwd_control_state {
   /* Generation number of the counters */
   int generation;
   /* Native encoding of the default counting domain */
   int selector;
   /* Buffer to pass to the kernel to control the counters */
   hwperf_profevctrarg_t counter_cmd;
   /* Number on each hwcounter */
   unsigned num_on_counter[2];
   /* Buffer for reading counters */
   hwperf_cntr_t cntrs_read;
} hwd_control_state_t;

typedef int hwd_register_map_t;

typedef struct _Context {
   /* File descriptor controlling the counters; */
   int fd;
} hwd_context_t;


typedef struct {
   unsigned int ri_fill:16, ri_imp:8,   /* implementation id */
    ri_majrev:4,                /* major revision */
    ri_minrev:4;                /* minor revision */
} papi_rev_id_t;


typedef siginfo_t hwd_siginfo_t;
typedef struct sigcontext hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  (void*)ctx->ucontext->sc_pc
#define GET_OVERFLOW_CTR_BITS(context) \
  (((_papi_hwi_context_t *)context)->overflow_vector)

#define HASH_OVERFLOW_CTR_BITS_TO_PAPI_INDEX(bit) \
  (_papi_hwi_event_index_map[bit])

#include "papi_internal.h"

extern int _etext[], _ftext[];
extern int _edata[], _fdata[];
extern int _fbss[], _end[];

#ifdef DEBUG
extern int papi_debug;
#endif

extern volatile int lock[PAPI_MAX_LOCK];

#define _papi_hwd_lock(lck)         \
while (__lock_test_and_set(&lock[lck],1) != 0)  \
{                       \
    usleep(1000);               \
}

#define _papi_hwd_unlock(lck) {__lock_release(&lock[lck]);}
