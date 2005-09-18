#ifndef _PAPI_X1                /* _PAPI_X1 */
#define _PAPI_X1

#define X1_MAX_COUNTERS 64
#define MAX_COUNTER_TERMS 8  /* Number of Native events that can be used for a derived event */

#include "papi_preset.h"

typedef struct X1_control {
   /* If the derived event is not associative, this index is the lead operand */
   unsigned int operand_index;

   /* Which counters to use? Bits encode counters to use, may be duplicates */
   unsigned int selector[3];

   /* Is this event derived? */
   u_long_long event_mask[3];

   /* P-Chip Hardware control Information to pass to the ioctl call */
   hwperf_x1_t p_evtctr[NUM_SSP];
   short has_p;

   /* E-Chip Hardware control Information to pass to the ioctl call */
   eperf_x1_t e_evtctr;
   short has_e;

   /* M-Chip Hardware control Information to pass to the ioctl call */
   mperf_x1_t m_evtctr;
   short has_m;

   long_long values[64];
} X1_control_t;

typedef struct X1_reg_alloc {
 int placeholder;
} X1_reg_alloc_t;

typedef struct X1_regmap {
 int placeholder;
} X1_regmap_t;

typedef struct X1_context {
   /* Process File Descriptor for passing to the ioctl call,
    * this should be opened at start time so that reads don't have to
    * reopen the file.
    */
   int fd;
} X1_context_t;

typedef struct X1_register{
 int event;
} X1_register_t;

typedef X1_control_t hwd_control_state_t;
typedef X1_regmap_t hwd_register_map_t;
typedef X1_context_t hwd_context_t;
typedef X1_register_t hwd_register_t;

typedef siginfo_t hwd_siginfo_t;
typedef ucontext_t hwd_ucontext_t;

typedef struct hwd_preset {
   /* Is this event derived? */
   unsigned int derived;
   /* If the derived event is not associative, this index is the lead operand */
   unsigned int operand_index;
   /* If the derived event is not associative, this index is the lead operand */
   unsigned int event_code[8];
   /* If it exists, then this is the description of this event */
   char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

typedef struct hwd_native_info {
   /* Description of the resources required by this native event */
   hwd_register_t resources;
   /* If it exists, then this is the name of the event */
   char *event_name;
   /* If it exists, then this is the description of the event */
   char *event_descr;
} hwd_native_info_t;

typedef X1_reg_alloc_t hwd_reg_alloc_t;


#include  "x1-native.h"
#include "x1-native-presets.h"
#include "x1-presets.h"

#endif                          /* _PAPI_X1 */
