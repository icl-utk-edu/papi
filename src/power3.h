#ifndef _PAPI_POWER3 /* _PAPI_POWER3 */
#define _PAPI_POWER3

#include "aix.h"

#define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT

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

typedef struct PWR3_pmapi_control {
  /* add this array to hold native events info */
  hwd_native_t native[MAX_COUNTERS];
  
  /* total_events: number of added events
     native_idx:   number of all native events 
	 both are required */
  int native_idx; 
    
  /* bitmap with all counters currently used */
  unsigned char master_selector;  

  /* Buffer to pass to the kernel to control the counters */
  pm_prog_t counter_cmd;
  /* Interrupt interval */
  
  int timer_ms;
} PWR3_pmapi_control_t;


typedef struct PWR3_regmap {
  /* unsigned int event_code; */
  /* register number the corespondent native event in the event lives on */
  unsigned char pos[MAX_COUNTERS];  
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


/* prototypes */
extern int set_domain(hwd_control_state_t *this_state, int domain);
extern int set_granularity(hwd_control_state_t *this_state, int domain);
extern void init_config(hwd_control_state_t *ptr);
/*void dump_state(hwd_control_state_t *s);*/

#endif /* _PAPI_POWER3 */
