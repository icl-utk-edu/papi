/* $Id$ */

#include <sys/hwperftypes.h>
#include <sys/hwperfmacros.h>

/* A superset of the machine dependent structure passed bewteen us and the kernel */

typedef struct hwd_control_state {
  int number_of_events;
  hwperf_profevctrarg_t on; } hwd_control_state_t;

/* Preset structure */

typedef struct hwd_preset {
  int counter_code1; /* 0 through 15 */
  int counter_code2; /* 0 through 15 */ } hwd_preset_t;
