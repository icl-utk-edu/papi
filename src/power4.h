#include "aix.h"

#define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT|PM_GET_GROUPS

typedef struct PWR4_pmapi_control {
  /* Indices into preset map for event in order of addition */
  /* if !PRESET_MASK then native event and counter # */
  /* only those events will not overlap*/
  int preset[POWER_MAX_COUNTERS];

  /* bitmap with all counters currently used */
  unsigned char master_selector;  
  /* bitmap with which counters used for which event */
  unsigned char selector[POWER_MAX_COUNTERS];
    
  /* Buffer to pass to the kernel to control the counters */
  pm_prog_t counter_cmd;
  /* Interrupt interval */
  
  int timer_ms;
} PWR4_pmapi_control_t;


typedef PWR4_pmapi_control_t hwd_control_state_t;

/* ... for PAPI3
typedef PWR4_regmap_t hwd_register_map_t;

typedef PWR4_register_t hwd_register_t;

typedef PWR4_pmapi_context_t hwd_context_t;

typedef PWR4_pmapi_event_t hwd_event_t;
*/

#include "papi_internal.h"

typedef struct hwd_preset {
  /* Is this event derived? */
  unsigned char derived;
  unsigned char metric_count;
  
  /* How many metrics? */
  /*unsigned char metric_count;*/
  /* Bits encode the groups this event lives in */
  unsigned int gps[GROUP_INTS];  

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

/* prototypes */
extern int setup_p4_presets(pm_info_t *pminfo, pm_groups_info_t *pmgroups);
extern int set_domain(hwd_control_state_t *this_state, int domain);
extern int set_granularity(hwd_control_state_t *this_state, int domain);
extern int update_global_hwcounters(EventSetInfo *global);
extern void init_config(hwd_control_state_t *ptr);

