/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>

#ifdef _WIN32
  #include <errno.h>
  #include "cpuinfo.h"
  #include "pmclib.h"
#else
#include <asm/ucontext.h>    /* sys/ucontext.h  apparently broken */
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <sys/times.h>
#include <sys/time.h>
#include <asm/system.h>
#include <asm/param.h>
#include <asm/bitops.h>
#include <linux/unistd.h>	

#ifndef CONFIG_SMP
/* Assert that CONFIG_SMP is set before including asm/atomic.h to 
 * get bus-locking atomic_* operations when building on UP kernels
 */
#define CONFIG_SMP
#endif
#include "asm/atomic.h"
#include "libperfctr.h"
#endif


#ifdef _WIN32
  #define inline_static static __inline 
#else
  #define inline_static inline static  
#endif






#if 0
typedef struct hwd_control_state {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Is this event derived? */
  int derived;
#ifdef _WIN32
  /* Buffer to pass to the kernel to control the counters */
  struct pmc_control counter_cmd;
  /* Handle to the open kernel driver */
  HANDLE self;
#else
  #ifdef PERFCTR20
    /* Buffer to pass to the kernel to control the counters */
    struct vperfctr_control counter_cmd;
  #else
    /* Buffer to pass to the kernel to control the counters */
    struct perfctr_control counter_cmd;
  #endif
    /* Buffer to control the kernel state of the counters */
    struct vperfctr *self;
#endif
} hwd_control_state_t;
#endif


/* Per event data structure for each event */

typedef struct P3_event {
  unsigned pmc_map;
  unsigned evntsel;
  unsigned evntsel_aux;
  unsigned pebs_enable;
  unsigned pebs_matrix_vert;
  unsigned ireset;
} P3_event_t;

typedef P3_event_t hwd_event_t;

#define P3_MAX_REGS_PER_EVENT 2

typedef struct P3_perfctr_preset {
  P3_event_t data[P3_MAX_REGS_PER_EVENT];
} P3_preset_t;

typedef struct P3_register {
  unsigned selector;               /* Mask for which counters in use */
  unsigned uses_pebs;              /* Binary flag for PEBS */
  unsigned uses_pebs_matrix_vert;  /* Binary flag for PEBS_MATRIX_VERT */
} P3_register_t;

typedef P3_register_t hwd_register_t;

typedef struct P3_regmap {
  unsigned num_hardware_events;
  P3_register_t hardware_event[P3_MAX_REGS_PER_EVENT];
} P3_regmap_t;

typedef P3_regmap_t hwd_register_map_t;

#ifdef _WIN32
/* Per eventset data structure for thread level counters */

typedef struct P3_WinPMC_control {
  P3_register_t allocated_registers;
  /* Buffer to pass to the kernel to control the counters */
  struct pmc_control counter_cmd;
  /* Handle to the open kernel driver */
  HANDLE self;
} P3_WinPMC_control_t;

typedef P3_WinPMC_control_t hwd_control_state_t;

/* Per thread data structure for thread level counters */

typedef struct P3_WinPMC_context {
  /* Handle to the open kernel driver */
  HANDLE self;
  P3_WinPMC_control_t start;
} P3_WinPMC_context_t;

typedef P3_WinPMC_context_t hwd_context_t;
#else
/* Per eventset data structure for thread level counters */

typedef struct P3_perfctr_control {
  P3_register_t allocated_registers;
  struct vperfctr_control control; 
  struct perfctr_sum_ctrs state;
} P3_perfctr_control_t;

typedef P3_perfctr_control_t hwd_control_state_t;

/* Per thread data structure for thread level counters */

typedef struct P3_perfctr_context {
  struct vperfctr *perfctr;
/*  P3_perfctr_control_t start; */
} P3_perfctr_context_t;

typedef P3_perfctr_context_t hwd_context_t;
#endif

typedef struct _ThreadInfo {
  unsigned pid;
  unsigned tid;
  hwd_context_t context;
  void *event_set_overflowing;
  int domain;
} ThreadInfo_t;

extern ThreadInfo_t *default_master_thread;

typedef struct _thread_list {
  ThreadInfo_t *master;
  struct _thread_list *next; 
} ThreadInfoList_t;

#define CNTR1 0x1
#define CNTR2 0x2
#define CNTR3 0x4
#define CNTR4 0x8

#define CNTRS12 (CNTR1|CNTR2)
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

#ifdef PERFCTR20
struct papi_perfctr_counter_cmd {
  unsigned int evntsel[PERF_MAX_COUNTERS];
} papi_perfctr_counter_cmd ;
#endif

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned char selector;  
  /* Is this event derived? */
  unsigned char derived;   
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Number of counters in the following */
  unsigned number;
  /* Buffer to pass to the kernel to control the counters */
#ifdef _WIN32
  struct pmc_control counter_cmd;
#elif defined(PERFCTR20)
  struct papi_perfctr_counter_cmd counter_cmd;
#else
  struct perfctr_control counter_cmd;
#endif
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata;

