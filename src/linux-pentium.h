/* $Id$ */

#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <asm/system.h>
#include <asm/atomic.h> 
#include <asm/perf.h>
#include <linux/unistd.h>	
#include <linux/tasks.h>	
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

typedef struct hwd_control_state {
  int mask;                                      /* Counter select mask */
  unsigned int start_conf[PERF_COUNTERS];        /* This array gets passed to kernel perf(PERF_FASTCONFIG) call */
  int timer_ms;                                  /* Milliseconds between timer interrupts for various things */  
  int domain;                                    /* If different than the default, this takes precedence. */
} hwd_control_state_t;

typedef struct hwd_preset {
  int mask;                
  int counter_code1;
  int counter_code2;
  int sp_code;   /* The per process TSC. */
} hwd_preset_t;


