
#include "aix.h"

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

typedef struct hwd_control_state {
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
} hwd_control_state_t;


#include "papi_internal.h"

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

#include "allocate.h"
