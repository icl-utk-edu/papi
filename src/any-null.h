#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <malloc.h>
#include <assert.h>
#include <limits.h>
#include "papi.h"

#define CNTR1 0x1
#define CNTR2 0x2
#define MAX_COUNTERS 2
#define PERF_USR       0x300
#define PERF_OS        0x400
#define PERF_ENABLE    0x800
#define PERF_EVNT_MASK 0xff

typedef struct _any_command {
  unsigned int cmd[MAX_COUNTERS];
} any_command_t;

typedef struct hwd_control_state {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Is this event derived? */
  int derived;   
  /* Buffer to pass to the kernel to control the counters */
  any_command_t counter_cmd;
} hwd_control_state_t;

#include "papi_internal.h"

typedef struct hwd_search {
  /* PAPI preset code */
  int papi_code;
  /* Is this derived */
  int derived_op;
  /* If so, what is the index of the operand */
  int operand_index;
  /* Events to encode */
  unsigned int findme[MAX_COUNTERS];
} hwd_search_t;

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned char selector;  
  /* Is this event derived? */
  unsigned char derived;   
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Buffer to pass to the kernel to control the counters */
  any_command_t counter_cmd;
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;
