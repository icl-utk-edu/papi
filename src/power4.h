#ifndef _PAPI_POWER4 /* _PAPI_POWER4 */
#define _PAPI_POWER4

#include "aix.h"

#define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT|PM_GET_GROUPS
#define GROUP_INTS 2

#include "power4_events.h"


typedef struct PWR4_pmapi_control {
  /* Buffer to pass to the kernel to control the counters */
  pm_prog_t counter_cmd;

  int group_id;
  /* Interrupt interval */
  int timer_ms;

} PWR4_pmapi_control_t;

typedef struct PWR4_reg_alloc {
  int ra_position;  
  unsigned int ra_group[GROUP_INTS];
  int ra_counter_cmd[MAX_COUNTERS];
} PWR4_reg_alloc_t;

/*typedef struct PWR4_register {*/
  /* unsigned int event_code; */
  /* register number the corespondent native event in the event lives on */
  /* unsigned char pos[MAX_COUNTERS];  
} PWR4_register_t;*/

typedef struct PWR4_pmapi_context {
  /* this structure is a work in progress */
  PWR4_pmapi_control_t cntrl;
} PWR4_pmapi_context_t;

typedef PWR4_pmapi_control_t hwd_control_state_t;

/*typedef PWR4_register_t hwd_register_t;*/

typedef PWR4_reg_alloc_t hwd_reg_alloc_t;

typedef PWR4_pmapi_context_t hwd_context_t;

/* ... for PAPI3
typedef PWR4_pmapi_event_t hwd_event_t;
*/

/*
typedef struct hwd_groups {
  // group number from the pmapi pm_groups_t struct 
  //int group_id;
  // Buffer containing counter cmds for this group 
  unsigned char counter_cmd[POWER_MAX_COUNTERS];
} hwd_groups_t;
*/
/* prototypes */
extern int set_domain(hwd_control_state_t *this_state, int domain);
extern int set_granularity(hwd_control_state_t *this_state, int domain);
extern int _papi_hwd_init_preset_search_map(pm_info_t *info);

#endif /* _PAPI_POWER4 */
