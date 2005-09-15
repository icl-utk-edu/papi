#ifndef _PAPI_SOLARIS_ULTRA_H
#define _PAPI_SOLARIS_ULTRA_H

#include "solaris.h"

#define MAX_COUNTERS 2
#define MAX_COUNTER_TERMS MAX_COUNTERS
#define PAPI_MAX_NATIVE_EVENTS 71
#define MAX_NATIVE_EVENT PAPI_MAX_NATIVE_EVENTS
#define MAX_NATIVE_EVENT_USII  22

typedef int hwd_reg_alloc_t;

typedef struct US_register {
   int event[MAX_COUNTERS];
} hwd_register_t;

typedef struct papi_cpc_event {
   /* Structure to libcpc */
   cpc_event_t cmd;
   /* Flags to kernel */
   int flags;
} papi_cpc_event_t;

typedef struct hwd_control_state {
   /* Buffer to pass to the kernel to control the counters */
   papi_cpc_event_t counter_cmd;
   /* overflow event counter */
   int overflow_num;
} hwd_control_state_t;

typedef int hwd_register_map_t;

typedef struct _native_info {
   /* native name */
   char name[40];
   /* Buffer to pass to the kernel to control the counters */
   int encoding[MAX_COUNTERS];
} native_info_t;

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

#define GET_OVERFLOW_ADDRESS(ctx)  ((caddr_t)(((ucontext_t *)ctx)->uc_mcontext.gregs[REG_PC]))

typedef int hwd_context_t;

#endif
