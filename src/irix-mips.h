/* $Id$ */

#include <sys/hwperftypes.h>
#include <sys/hwperfmacros.h>

/* A superset of the machine dependent structure passed between 
   us and the kernel */

/* You might ask, why do we keep track of the number of running counters when
   everything's multiplexed anyways? Accuracy. If I can select two events 
   that run of different counters, then the kernel won't multiplex. */

typedef struct hwd_control_state {
  int mask;            /* Which counters are active */
  int num_on_counter1; /* Number of counters running on hardware counter 0 */
  int num_on_counter2; /* Number of counters running on hardware counter 1 */
  int hwindex[HWPERF_EVENTMAX]; /* Staging area */
  hwperf_profevctrarg_t on; /* Exchange structure with kernel */
} hwd_control_state_t;

typedef struct hwd_preset {
  int mask;          /* Multiple bits mean that they can count same event. */
  int counter_code1; /* 0 through 15 */
  int counter_code2; /* 0 through 15 */
  int pad; } hwd_preset_t;
