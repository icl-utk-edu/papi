#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <libgen.h>
#include <sys/systemcfg.h>
#include <sys/processor.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <procinfo.h>
#include <sys/atomic_op.h>

#define ANY_THREAD_GETS_SIGNAL
#include <dlfcn.h>

#include "pmapi.h"
#define POWER_MAX_COUNTERS MAX_COUNTERS
#define GROUP_INTS 2
#define MAX_GROUPS (GROUP_INTS * 32)
#define INVALID_EVENT -2
#define POWER_MAX_COUNTERS_MAPPING 8

#include "papi.h"

#ifndef _POWER4
typedef struct hwd_native {
  /* Which counters can be used? Bits encode counters available
      Separate selectors for each metric in a derived event;
      Rank determines how many counters carry each metric */
  unsigned char selector;  
  unsigned char rank;
  /* Buffers containing counter cmds for each possible metric */
  unsigned char counter_cmd[POWER_MAX_COUNTERS];
  /* If it exists, then this is the description of this event */
  char name[PAPI_MAX_STR_LEN];
  /* which counter this native event stays */
  int position;
  int mod;
  int link;
} hwd_native_t;
#endif

typedef struct hwd_control_state {
  /* Indices into preset map for event in order of addition */
  /* if !PRESET_MASK then native event and counter # */
  /* only those events will not overlap*/
  int preset[POWER_MAX_COUNTERS];

#ifndef _POWER4
  /* add this array to hold native events info */
  hwd_native_t native[POWER_MAX_COUNTERS];
  
  /* use POWER_MAX_COUNTERS_MAPPING instead of POWER_MAX_COUNTERS */
  /* because there exists overlap of counters used by events 
     keep all added events in order*/
  int allevent[POWER_MAX_COUNTERS_MAPPING];
  
  /*for the derived event, the mapping to those counters may change
     keep this mapping info for evaluator usage: counter number  */ 
  int emap[POWER_MAX_COUNTERS_MAPPING][POWER_MAX_COUNTERS]; 
  int hwd_idx, hwd_idx_a, native_idx;  
#endif
  
  /* bitmap with all counters currently used */
  unsigned char master_selector;  
  /* bitmap with which counters used for which event */
  unsigned char selector[POWER_MAX_COUNTERS];
    
  /* Buffer to pass to the kernel to control the counters */
  pm_prog_t counter_cmd;
  /* Interrupt interval */
  
  int timer_ms;
} hwd_control_state_t;


#include "papi_internal.h"

typedef struct hwd_preset {
  /* Is this event derived? */
  unsigned char derived;
  unsigned char metric_count;
  
#ifdef _POWER4
  /* How many metrics? */
  /*unsigned char metric_count;*/
  /* Bits encode the groups this event lives in */
  unsigned int gps[GROUP_INTS];  
#else
  /* Which counters can be used? Bits encode counters available
      Separate selectors for each metric in a derived event;
      Rank determines how many counters carry each metric */
  unsigned char selector[POWER_MAX_COUNTERS];  
  unsigned char rank[POWER_MAX_COUNTERS];
#endif

  /* Buffers containing counter cmds for each possible metric */
  unsigned char counter_cmd[POWER_MAX_COUNTERS][POWER_MAX_COUNTERS];
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

typedef struct hwd_groups {
  /* group number from the pmapi pm_groups_t struct */
  int group_id;
  /* Buffer containing counter cmds for this group */
  unsigned char counter_cmd[POWER_MAX_COUNTERS];
} hwd_groups_t;

typedef struct pmapi_search {
  /* Preset code */
  int preset;
  /* Derived code */
  int derived;
  /* Strings to look for */
  char *(findme[POWER_MAX_COUNTERS]);
} pmapi_search_t;

extern _text;
extern _etext;
extern _edata;
extern _end;
extern _data;
extern int (*thread_kill_fn)(int, int);
