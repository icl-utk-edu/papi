#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/timers.h>
#include <stropts.h>
#include <unistd.h>
#include <sys/processor.h>
#include <sys/times.h>
#include <sys/sysinfo.h>
#include <sys/procfs.h>
#include <machine/hal_sysinfo.h>
#include <machine/cpuconf.h>
#include <assert.h>
#include <sys/ucontext.h>
/* Below can be removed when we stop using rusuage for PAPI_get_virt_usec -KSL*/
#include <sys/resource.h>

#include "papi.h"
#include "dadd.h"
#include "virtual_counters.h"

#define VC_TOTAL_CYCLES 0
#define VC_BCACHE_MISSES 1
#define VC_TOTAL_DTBMISS 2
#define VC_NYP_EVENTS 3
#define VC_TAKEN_EVENTS 4
#define VC_MISPREDICT_EVENTS 5
#define VC_LD_ST_ORDER_TRAPS 6
#define VC_TOTAL_INSTR_ISSUED 7
#define VC_TOTAL_INSTR_EXECUTED 8
#define VC_INT_INSTR_EXECUTED 9
#define VC_LOAD_INSTR_EXECUTED 10
#define VC_STORE_INSTR_EXECUTED 11
#define VC_TOTAL_LOAD_STORE_EXECUTED 12
#define VC_SYNCH_INSTR_EXECUTED 13
#define VC_NOP_INSTR_EXECUTED 14
#define VC_PREFETCH_INSTR_EXECUTED 15
#define VC_FA_INSTR_EXECUTED 16
#define VC_FM_INSTR_EXECUTED 17
#define VC_FD_INSTR_EXECUTED 18
#define VC_FSQ_INSTR_EXECUTED 19
#define VC_FP_INSTR_EXECUTED 20
#define VC_UNCOND_BR_EXECUTED 21
#define VC_COND_BR_EXECUTED 22
#define VC_COND_BR_TAKEN 23
#define VC_COND_BR_NOT_TAKEN 24
#define VC_COND_BR_MISPREDICTED 25
#define VC_COND_BR_PREDICTED 26

typedef struct hwd_control_state {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;
  /* Is this event derived? */
  int derived;
  /* Pointer to the DADD virtual counter structure */
  virtual_counters *ptr_vc;
  /* Interrupt interval */
  int timer_ms;
} hwd_control_state_t;

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned char selector;
  /* Is this event derived? */
  unsigned char derived;
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Buffer to pass to the kernel to control the counters */
  long counter_cmd;
  /* Footnote to append to the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

typedef struct hwd_search {
  /* PAPI preset code */
  unsigned int papi_code;
  /* DADD event code */
  long dadd_code;
} hwd_search_t;

#include "papi_internal.h"

extern unsigned long _etext, _ftext;

