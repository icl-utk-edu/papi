#ifndef _PAPI_POWER3_H            /* _PAPI_POWER3 */
#define _PAPI_POWER3_H

#include "aix.h"
#include "power3_events.h"

#define PM_INIT_FLAGS PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT

typedef struct PWR3_pmapi_control {
   /* Buffer to pass to the kernel to control the counters */
   pm_prog_t counter_cmd;

   /* Interrupt interval */
   int timer_ms;
} PWR3_pmapi_control_t;

/* defines the fields needed by _papi_hwd_allocate_registers
   to map the counter set */
typedef struct PWR3_reg_alloc {
   unsigned int ra_selector;    /* Which counters are available? */
   unsigned char ra_rank;       /* How many counters carry each metric */
   /* More generally, which event is most resource restrictive */
   int ra_mod;                  /* don't exactly know what this field does */
} PWR3_reg_alloc_t;

/*typedef struct PWR3_register {*/
  /* unsigned int event_code; */
  /* register number the corespondent native event in the event lives on */
/*  unsigned char pos[MAX_COUNTERS];  
} PWR3_register_t;*/

typedef struct PWR3_pmapi_context {
   /* this structure is a work in progress */
   PWR3_pmapi_control_t cntrl;
} PWR3_pmapi_context_t;

typedef PWR3_pmapi_control_t hwd_control_state_t;

/*typedef PWR3_register_t hwd_register_t;*/

typedef PWR3_reg_alloc_t hwd_reg_alloc_t;

typedef PWR3_pmapi_context_t hwd_context_t;

/* ... for PAPI3
typedef PWR3_pmapi_event_t hwd_event_t;
*/

/* prototypes */
extern int set_domain(hwd_control_state_t * this_state, int domain);
extern int set_granularity(hwd_control_state_t * this_state, int domain);
extern int _papi_hwd_init_preset_search_map(pm_info_t * info);
/*void dump_state(hwd_control_state_t *s);*/

#endif                          /* _PAPI_POWER3 */
