/* $Id$ */

#include <sys/hwperftypes.h>
#include <sys/hwperfmacros.h>

/* A superset of the machine dependent structure passed bewteen us and the kernel */

/* You might ask, why do we keep track of the number of running counters when
everything's multiplexed anyways? Accuracy. If I can select two events that run
of different counters, then the kernel won't multiplex. */

typedef struct hwd_control_state {
  int num_on_counter1;             /* Number of counters running on hardware counter 0 */
  int num_on_counter2;             /* Number of counters running on hardware counter 1 */
  int hwindex[HWPERF_EVENTMAX];    /* Staging area */
  hwperf_profevctrarg_t on; } hwd_control_state_t;

/* Preset structure */

typedef struct hwd_preset {
  int computed;      /* If non-zero, then this preset is a DERIVED quantity. */
                     /* If zero, and multiple counters are coded, that
			means that they can count the same event. */
  int counter_code1; /* 0 through 15 */
  int counter_code2; /* 0 through 15 */ } hwd_preset_t;
