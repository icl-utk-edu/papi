#ifndef _PAPI_PENTIUM3
#define _PAPI_PENTIUM3

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
#include <sys/ucontext.h>    /* sys/ucontext.h  apparently broken */
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

#define PERF_MAX_COUNTERS 4
#define MAX_COUNTERS PERF_MAX_COUNTERS
#define PAPI_MAX_NATIVE_EVENTS 140
#define MAX_NATIVE_EVENT 140
#define MAX_COUNTER_TERMS  MAX_COUNTERS
/* Per event data structure for each event */
#define P3_MAX_REGS_PER_EVENT 2

#include "papi.h"
#include "papi_preset.h"

#ifdef _WIN32
  #define inline_static static __inline 
#else
  #define inline_static inline static  
#endif

typedef struct P3_register {
  unsigned int selector;           /* Mask for which counters in use */
  int counter_cmd[MAX_COUNTERS];   /* Mask for which counters in use */
} P3_register_t;

typedef struct P3_reg_alloc {
  P3_register_t ra_bits;    /* Info about this native event mapping */
  unsigned ra_selector;     /* Bit mask showing which counters can carry this metric */
  unsigned ra_rank;         /* How many counters can carry this metric */
} P3_reg_alloc_t;

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

typedef struct hwd_native {
  /* index in the native table, required */
  int index;
  /* Which counters can be used?  */
  unsigned int selector;
  /* Rank determines how many counters carry each metric */
  unsigned char rank;
  /* which counter this native event stays */
  int position;
  int mod;
  int link;
} hwd_native_t;

typedef struct native_event_entry {
  /* If it exists, then this is the name of this event */
  char name[PAPI_MAX_STR_LEN];
  /* If it exists, then this is the description of this event */
  char *description;
  /* description of the resources required by this native event */
  P3_register_t resources;
} native_event_entry_t;

typedef struct P3_perfctr_control {
  hwd_native_t native[MAX_COUNTERS];
  int native_idx;
  unsigned char master_selector;
  P3_register_t allocated_registers;
  struct vperfctr_control control; 
  struct perfctr_sum_ctrs state;
} P3_perfctr_control_t;

/* Per thread data structure for thread level counters */

typedef struct P3_perfctr_context {
  struct vperfctr *perfctr;
/*  P3_perfctr_control_t start; */
} P3_perfctr_context_t;

typedef P3_reg_alloc_t hwd_reg_alloc_t;
typedef P3_perfctr_control_t hwd_control_state_t;
typedef P3_register_t hwd_register_t;
typedef P3_perfctr_context_t hwd_context_t;
#endif

typedef struct _ThreadInfo {
  unsigned pid;
  unsigned tid;
  hwd_context_t context;
  void *event_set_overflowing;
  void *event_set_profiling;
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

#define AI_ERROR "No support for a-mode counters after adding an i-mode counter"
#define VOPEN_ERROR "vperfctr_open() returned NULL"
#define GOPEN_ERROR "gperfctr_open() returned NULL"
#define VINFO_ERROR "vperfctr_info() returned < 0"
#define VCNTRL_ERROR "vperfctr_control() returned < 0"
#define GCNTRL_ERROR "gperfctr_control() returned < 0"
#define FOPEN_ERROR "fopen(%s) returned NULL"
#define STATE_MAL_ERROR "Error allocating perfctr structures"
#define MODEL_ERROR "This is not a Pentium 3"

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

native_event_entry_t *native_table;
preset_search_t *preset_search_map;

extern char *basename(char *);
extern caddr_t _start, _init, _etext, _fini, _end, _edata, __data_start, __bss_start;
extern int get_memory_info( PAPI_mem_info_t * mem_info, int cpu_type );
extern int _papi_hwd_get_system_info(void);

#endif
