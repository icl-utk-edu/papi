#ifndef _PAPI_POWER3 /* _PAPI_POWER3 */
#define _PAPI_POWER3

#include "aix.h"

#define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT

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

typedef struct PWR3_pmapi_control {
  /* Indices into preset map for event in order of addition */
  /* if !PRESET_MASK then native event and counter # */
  /* only those events will not overlap*/
  int preset[POWER_MAX_COUNTERS];

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
  
  /* bitmap with all counters currently used */
  unsigned char master_selector;  
  /* bitmap with which counters used for which event */
  unsigned char selector[POWER_MAX_COUNTERS];
    
  /* Buffer to pass to the kernel to control the counters */
  pm_prog_t counter_cmd;
  /* Interrupt interval */
  
  int timer_ms;
} PWR3_pmapi_control_t;


typedef struct PWR3_regmap {
  unsigned int event_code;
  unsigned char selector;  
} PWR3_regmap_t;

typedef struct PWR3_pmapi_context {
  /* this structure is a work in progress */
  PWR3_pmapi_control_t cntrl;
} PWR3_pmapi_context_t;

typedef PWR3_pmapi_control_t hwd_control_state_t;

typedef PWR3_regmap_t hwd_register_map_t;

typedef PWR3_pmapi_context_t hwd_context_t;

/* ... for PAPI3
typedef PWR3_register_t hwd_register_t;

typedef PWR3_pmapi_event_t hwd_event_t;
*/

/* Can these thread structures be moved out of the substrate?
    Or are they platform dependent?
*/
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


typedef struct hwd_preset {
  /* Is this event derived? */
  unsigned char derived;
  unsigned char metric_count;
  
  /* Which counters can be used? Bits encode counters available
      Separate selectors for each metric in a derived event;
      Rank determines how many counters carry each metric */
  unsigned char selector[POWER_MAX_COUNTERS];  
  unsigned char rank[POWER_MAX_COUNTERS];

  /* Buffers containing counter cmds for each possible metric */
  unsigned char counter_cmd[POWER_MAX_COUNTERS][POWER_MAX_COUNTERS];
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

typedef struct pmapi_search {
  /* Preset code */
  int preset;
  /* Derived code */
  int derived;
  /* Strings to look for */
  char *(findme[POWER_MAX_COUNTERS]);
} pmapi_search_t;

/* prototypes */
extern int setup_all_presets(pm_info_t *info);
extern int set_domain(hwd_control_state_t *this_state, int domain);
extern int set_granularity(hwd_control_state_t *this_state, int domain);
extern void init_config(hwd_control_state_t *ptr);
void dump_state(hwd_control_state_t *s);

#endif /* _PAPI_POWER3 */
