#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <asm/system.h>
#include <asm/perf.h>
#include <linux/unistd.h>	
#include <linux/tasks.h>	
#include <linux/smp.h>
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

#define CNTR1 0x1
#define CNTR2 0x2
#define MAX_COUNTERS PERF_COUNTERS

typedef struct hwd_control_state {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Is this event derived? */
  int derived;   
  /* Buffer to pass to the kernel to control the counters */
  int counter_cmd[MAX_COUNTERS];
} hwd_control_state_t;

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned char selector;  
  /* Is this event derived? */
  unsigned char derived;   
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Buffer to pass to the kernel to control the counters */
  unsigned int counter_cmd[MAX_COUNTERS];
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata;
