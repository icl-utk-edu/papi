#ifndef _PAPI_X1                /* _PAPI_X1 */
#define _PAPI_X1
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
#include "papi.h"

#define inline_static static
#define X1_MAX_COUNTERS 64
#define MAX_COUNTERS X1_MAX_COUNTERS
#define MAX_COUNTER_TERMS 8  /* Number of Native events that can be used for a derived event */

#include "papi_preset.h"

typedef struct X1_control {
   /* If the derived event is not associative, this index is the lead operand */
   unsigned int operand_index;

   /* Which counters to use? Bits encode counters to use, may be duplicates */
   unsigned int selector[3];

   /* Is this event derived? */
   u_long_long event_mask[3];

   /* P-Chip Hardware control Information to pass to the ioctl call */
   hwperf_x1_t p_evtctr[NUM_SSP];
   short has_p;

   /* E-Chip Hardware control Information to pass to the ioctl call */
   eperf_x1_t e_evtctr;
   short has_e;

   /* M-Chip Hardware control Information to pass to the ioctl call */
   mperf_x1_t m_evtctr;
   short has_m;

   long_long values[64];
} X1_control_t;

typedef struct X1_reg_alloc {
 int placeholder;
} X1_reg_alloc_t;

typedef struct X1_regmap {
 int placeholder;
} X1_regmap_t;

typedef struct X1_context {
   /* Process File Descriptor for passing to the ioctl call,
    * this should be opened at start time so that reads don't have to
    * reopen the file.
    */
   int fd;
} X1_context_t;

typedef struct X1_register{
 int event;
} X1_register_t;

typedef X1_control_t hwd_control_state_t;
typedef X1_regmap_t hwd_register_map_t;
typedef X1_context_t hwd_context_t;
typedef X1_register_t hwd_register_t;

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

typedef struct hwd_preset {
   /* Is this event derived? */
   unsigned int derived;
   /* If the derived event is not associative, this index is the lead operand */
   unsigned int operand_index;
   /* If the derived event is not associative, this index is the lead operand */
   unsigned int event_code[8];
   /* If it exists, then this is the description of this event */
   char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

typedef struct hwd_native_info {
   /* Description of the resources required by this native event */
   hwd_register_t resources;
   /* If it exists, then this is the name of the event */
   char *event_name;
   /* If it exists, then this is the description of the event */
   char *event_descr;
} hwd_native_info_t;

typedef X1_reg_alloc_t hwd_reg_alloc_t;


/*
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(ctx->ucontext[ctx->si->si_ssp].uc_mcontext.scontext[CTX_EPC])
*/
#ifdef MSP
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(ctx->ucontext[0].uc_mcontext.scontext[CTX_EPC])
#else
#define GET_OVERFLOW_ADDRESS(ctx) (caddr_t)(ctx->ucontext[0].uc_mcontext.scontext[CTX_EPC])
#endif

#include  "x1-native.h"
#include "x1-native-presets.h"
#include "x1-presets.h"
#include "papi_internal.h"

extern int _etext[], _ftext[];
extern int _edata[], _fdata[];
extern int _fbss[], _end[];


/* Prototypes */
extern int set_domain(hwd_context_t * this_state, int domain);
extern int set_granularity(hwd_context_t * this_state, int domain);
void _papi_hwd_lock(int index);
void _papi_hwd_unlock(int index);
#endif                          /* _PAPI_X1 */
