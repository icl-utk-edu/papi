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
#include <sys/times.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/processor.h>
#include <sys/procset.h>
#include <sys/ucontext.h>
#include <syms.h>

#include "papi.h"

#define MAX_COUNTERS 2
#define MAX_COUNTER_TERMS MAX_COUNTERS
#define PAPI_MAX_NATIVE_EVENTS 71
#define MAX_NATIVE_EVENT PAPI_MAX_NATIVE_EVENTS

typedef int hwd_register_t;

typedef struct papi_cpc_event {
  /* Structure to libcpc */
  cpc_event_t cmd;
  /* Flags to kernel */
  int flags;  
} papi_cpc_event_t;

typedef struct hwd_control_state {
  /* Buffer to pass to the kernel to control the counters */
  papi_cpc_event_t counter_cmd;
  /* Buffer to save the values read from the hardware counter */
  long_long values[MAX_COUNTERS];
} hwd_control_state_t;

typedef int hwd_register_map_t;

typedef struct _native_info {
  /* native name */
  char name[40];
  /* Buffer to pass to the kernel to control the counters */
  int encoding[MAX_COUNTERS];
} native_info_t;

typedef int hwd_context_t;

typedef struct _ThreadInfo {
  unsigned pid;
  unsigned tid;
  hwd_context_t context;
  void * event_set_overflowing;
  void * event_set_profiling;
  int domain;
} ThreadInfo_t;

extern ThreadInfo_t *default_master_thread;

typedef struct _thread_list {
  ThreadInfo_t *master;
  struct _thread_list *next;
} ThreadInfoList_t;

#include "papi_internal.h"

/* Assembler prototypes */

extern void cpu_sync(void);
extern unsigned long long get_tick(void);
extern caddr_t _start, _end, _etext, _edata;

