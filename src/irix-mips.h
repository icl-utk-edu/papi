#include <stdlib.h>
#include <stdio.h>
#include <invent.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <task.h>
#include <assert.h>
#include <unistd.h>
#include <sys/times.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/procfs.h>
#include <sys/cpu.h>
#include <sys/sysmp.h>
#include <sys/sbd.h>
#include <sys/hwperftypes.h>
#include <sys/hwperfmacros.h>

#include "papi.h"
typedef struct hwd_control_state {
  /* File descriptor controlling the counters; */
  int fd;
  /* Generation number of the counters */
  int generation;
  /* Native encoding of the default counting domain */
  int selector;  
  /* Is this event derived? */
  int derived;   
  /* Buffer to pass to the kernel to control the counters */
  hwperf_profevctrarg_t counter_cmd;
  /* Interrupt interval */
  int timer_ms;
  /* Number on each hwcounter */
  unsigned num_on_counter[2];
  /* Buffer for reading counters */
  hwperf_cntr_t cntrs_read;
} hwd_control_state_t;

/* just to make the compile work */
typedef struct Irix_regmap {
    unsigned selector;
} Irix_regmap_t;

typedef Irix_regmap_t  hwd_register_map_t;

typedef struct _Context {
  int init_flag;
  /* File descriptor controlling the counters; */
  int fd;
}  hwd_context_t;

typedef struct _ThreadInfo {
    unsigned pid;
    unsigned tid;
    hwd_context_t context;
    void *event_set_overflowing;
    void *event_set_profiling;
    int domain;
} ThreadInfo_t;

typedef struct _thread_list  {
    ThreadInfo_t *master;
    struct _thread_list *next;
}  ThreadInfoList_t;

#include "papi_internal.h"

typedef struct {
  unsigned int     ri_fill:16,
    ri_imp:8,		/* implementation id */
    ri_majrev:4,	/* major revision */
    ri_minrev:4;	/* minor revision */
} papi_rev_id_t;

/* Encoding for NON-PAPI events is:

   Low 8 bits indicate which counter number: 0 - 7
   Bits 8-16 indicate which event number: 0 - 50 */

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned int selector;  
  /* Is this event derived? */
  unsigned char derived;   
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Buffer to pass to the kernel to control the counters */
  unsigned char counter_cmd[HWPERF_EVENTMAX];
  /* Number on each hwcounter */
  int num_on_counter[2];
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

typedef struct hwd_search {
  /* PAPI preset code */
  int preset;
  /* Derived code */
  int derived;
  /* Events to encode */
  int findme[2];
} hwd_search_t;

extern ThreadInfo_t *default_master_thread;

int get_memory_info(PAPI_mem_info_t* mem_info);

extern int _etext[], _ftext[];
extern int _edata[], _fdata[];
extern int _fbss[], _end[];

#ifdef DEBUG
extern int papi_debug;
#endif
