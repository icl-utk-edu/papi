/* $Id$ */

typedef struct hwd_control_state {
  int mask;             /* Counter select mask */
  int mmcr;             /* Counter control register */ } hwd_control_state_t;

/* Preset structure */

#define GROUP_ALL -2

#define UNIT_FXU 0
#define UNIT_ICU 1
#define UNIT_SCU 2
#define UNIT_FPU 3
#define UNIT_ALL 4

typedef struct hwd_preset {
  char group;          /* Group number, GROUP_NONE, GROUP_ALL */
  char number;         /* Counter number in group (0 and 1 are reserved)*/
  char unit;           /* UNIT_FXU,UNIT_SCU,UNIT_ICU,UNIT_DCU,UNIT_ALL */
  char multimask;      /* Multiple counter select mask */ } hwd_preset_t;

