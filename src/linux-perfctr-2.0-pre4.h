#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <asm/system.h>
#include <linux/unistd.h>	
// #include <linux/smp.h>

#include "libperfctr.h"
#include "x86-events.h"
#include "papi.h"

#ifdef PERFCTR20PRE4
struct papi_perfctr_counter_cmd {
       unsigned int evntsel[4];
} papi_perfctr_counter_cmd ;
#endif

typedef struct hwd_control_state {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Is this event derived? */
  int derived;   
  /* Buffer to pass to the kernel to control the counters */
#ifdef PERFCTR20PRE4
  struct vperfctr_control counter_cmd;
#else
  struct perfctr_control counter_cmd;
#endif
  /* Buffer to control the kernel state of the counters */
  struct vperfctr *self;
} hwd_control_state_t;

#include "papi_internal.h"

#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8

#define ALLCNTRS (CNTR1|CNTR2|CNTR3|CNTR4)

#define PERF_MAX_COUNTERS 4

#define PERF_CTR_MASK          0xFF000000
#define PERF_INV_CTR_MASK      0x00800000
#define PERF_ENABLE            0x00400000
#define PERF_INT_ENABLE        0x00100000
#define PERF_PIN_CONTROL       0x00080000
#define PERF_EDGE_DETECT       0x00040000
#define PERF_OS                0x00020000
#define PERF_USR               0x00010000
#define PERF_UNIT_MASK         0x0000FF00
#define PERF_EVNT_MASK         0x000000FF

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned char selector;  
  /* Is this event derived? */
  unsigned char derived;   
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Buffer to pass to the kernel to control the counters */
#ifdef PERFCTR20PRE4
  struct papi_perfctr_counter_cmd counter_cmd;
#else
  struct perfctr_control counter_cmd;
#endif
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata;
