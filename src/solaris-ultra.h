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

#include "papi.h"

#define US_MAX_COUNTERS 2

typedef struct papi_cpc_event {
  /* Structure to libcpc */
  cpc_event_t cmd;
  /* Flags to kernel */
  int flags;  
} papi_cpc_event_t;

typedef struct hwd_control_state {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Is this event derived? */
  int derived;   
  /* Buffer to pass to the kernel to control the counters */
  papi_cpc_event_t counter_cmd;
} hwd_control_state_t;

#include "papi_internal.h"

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned char selector;  
  /* Is this event derived? */
  unsigned char derived;   
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Buffer to pass to the kernel to control the counters */
  unsigned char counter_cmd[US_MAX_COUNTERS];
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

/* Assembler prototypes */

extern void cpu_sync(void);
extern unsigned long long get_tick(void);
extern _start, _end, _etext, _edata;
