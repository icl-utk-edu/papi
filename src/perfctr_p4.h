#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <signal.h>
#include <unistd.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <sys/times.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ucontext.h>

#include <linux/unistd.h>	
#include <asm/atomic.h>

#include "libperfctr.h"

/* Per event data structure for each event */

typedef struct P4_perfctr_event {
  unsigned int pmc_map;
  unsigned int evntsel;
  unsigned int evntsel_aux;
  unsigned int pebs_enable;
  unsigned int pebs_matrix_vert;
  int ireset;
} P4_perfctr_event_t;

#define P4_MAX_REGS_PER_EVENT 4

typedef struct P4_perfctr_codes {
  P4_perfctr_event_t data[P4_MAX_REGS_PER_EVENT];
} P4_perfctr_preset_t;

typedef struct P4_perfctr_avail {
  unsigned cccr_bits;           /* What we write for control, read for data */
  unsigned cccr_num;            /* Only filled when registers are allocated. */
  unsigned escr_low_bits;       /* What we write for control 0-31 */
  unsigned escr_low_num;        /* Only filled when registers are allocated. */
  unsigned escr_high_bits;      /* What we write for control 0-31 */
  unsigned escr_high_num;       /* Only filled when registers are allocated. */
} P4_register_t;

typedef struct P4_regmap {
  unsigned num_hardware_events;
  P4_register_t hardware_event[P4_MAX_REGS_PER_EVENT];
} P4_regmap_t;

/* Per eventset data structure for thread level counters */

typedef struct P4_perfctr_control {
  P4_register_t allocated_registers;
  struct vperfctr_control control; 
  struct perfctr_sum_ctrs state;
} P4_perfctr_control_t;

/* Per thread data structure for thread level counters */

typedef struct P4_perfctr_context {
  struct vperfctr *perfctr;
/*  P4_perfctr_control_t start; */
} P4_perfctr_context_t;

typedef P4_perfctr_control_t hwd_control_state_t;

typedef P4_regmap_t hwd_register_map_t;

typedef P4_register_t hwd_register_t;

typedef P4_perfctr_context_t hwd_context_t;

typedef P4_perfctr_event_t hwd_event_t;

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

#if 0
#include "papi_internal.h"

/* Per thread data structure for global level counters */

typedef struct P4_perfctr_context {
  struct gperfctr *perfctr;
} P4_perfctr_global_context_t;

typedef struct P4_global_control {
  struct gperfctr_control *control; 
  struct gperfctr_state *state;
} P4_perfctr_global_control_t;
#include "papi_protos.h"
#endif


/* Per preset data structure statically defined in dense array in substrate */

typedef struct P4_search {
  unsigned preset;
  char *note;
  unsigned control_selector;
  unsigned read_selector;
  unsigned number;
  P4_perfctr_preset_t info;
} P4_search_t;

/* Per preset data structure dynamically defined in sparse array by substrate
   from array of P4_search_t's. */

typedef struct P4_preset {
  /* Which counters to use? */
  unsigned control_selector;  
  /* Which counters to read from? */
  unsigned read_selector;  
  /* Is this event derived? */
  unsigned derived;   
  /* Number of counters in the following */
  unsigned number;
  /* Where can this register live */
  P4_regmap_t possible_registers;
  /* Information on how to construct the event */
  P4_perfctr_preset_t *info;
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} P4_preset_t;

typedef P4_preset_t hwd_preset_t;

#define AI_ERROR "No support for a-mode counters after adding an i-mode counter"
#define VOPEN_ERROR "vperfctr_open() returned NULL"
#define GOPEN_ERROR "gperfctr_open() returned NULL"
#define VINFO_ERROR "vperfctr_info() returned < 0"
#define VCNTRL_ERROR "vperfctr_control() returned < 0"
#define GCNTRL_ERROR "gperfctr_control() returned < 0"
#define FOPEN_ERROR "fopen(%s) returned NULL"
#define STATE_MAL_ERROR "Error allocating perfctr structures"

#define error_return(retval, format, args...) { fprintf(stderr,"Error in %s, function %s, line %d: ",__FILE__,__FUNCTION__,__LINE__); fprintf(stderr, format , ## args) ; fprintf(stderr, "\n"); return(retval); }
#ifdef DEBUG
#define DEBUGLABEL(a) fprintf(stderr,"%s:%s:%s:%d: ",a,__FILE__,__FUNCTION__,__LINE__)
#define SUBDBG(format, args...) { extern int _papi_hwi_debug; if (_papi_hwi_debug) { DEBUGLABEL("SUBSTRATE"); fprintf (stderr, format , ## args); } }
#else
#define SUBDBG(format, args...) { ; }
#endif

#define PAPI_VENDOR_UNKNOWN -1
#define PAPI_VENDOR_INTEL   1
#define PAPI_VENDOR_AMD     2
#define PAPI_VENDOR_CYRIX   3

#define PAPI_MODEL_UNKNOWN  -1
#define PAPI_MODEL_PENTIUM_4 PERFCTR_X86_INTEL_P4

#define FAST_RDPMC (1 << 31)
#define COUNTER(a) (a)
#define ESCR(a) (a << 13)
#define ENABLE (1 << 12)
#define HYPERTHREAD_ANY (0x3 << 16)
#define NBOGUSNTAG (1 << 9)
#define CPL(a) (a << 2)
#define EVENT(a) (a << 25)
#define EVENTMASK(a) (a << 9)
#define COMPARE (1 << 18)
#define COMPLEMENT (1 << 19)
#define THRESHOLD(a) (a << 20)
#define PEBS_TAG (1 << 24)
#define PEBS_L1_MISS (1)
#define PEBS_L2_MISS (1)
#define PEBS_MV_LOAD (1)
#define PEBS_MV_STORE (1 << 1)

/* Stupid linux basename prototype! */

extern char *basename(char *);
extern int sighold(int);
extern int sigrelse(int);

/* Undefined identifiers in executable */

extern caddr_t _start, _fini, _end, _edata, __data_start, __bss_start;
