#ifndef _PAPI_X1_H                /* _PAPI_X1 */
#define _PAPI_X1_H

#include "unicosMP.h"
#include  "x1-native.h"

#include "papi_internal.h"
#define inline_static inline static
#define X1_MAX_COUNTERS 64
#define MAX_COUNTERS X1_MAX_COUNTERS

typedef struct X1_control {
   /* If the derived event is not associative, this index is the lead operand */
   unsigned int operand_index;

   /* Which counters to use? Bits encode counters to use, may be duplicates */
   unsigned int selector[3];

   /* Is this event derived? */
   u_long_long event_mask[3];

   /* P-Chip Hardware control Information to pass to the ioctl call */
   hwperf_x1_t p_evtctr;

   /* E-Chip Hardware control Information to pass to the ioctl call */
   eperf_x1_t e_evtctr;

   /* M-Chip Hardware control Information to pass to the ioctl call */
   mperf_x1_t m_evtctr;

   /* Process File Descriptor for passing to the ioctl call,
    * this should be opened at start time so that reads don't have to
    * reopen the file.
    */
   int fd;
} X1_control_t;

typedef struct X1_regmap {
} X1_regmap_t;

typedef struct X1 X1_context {
} X1_context_t;

typedef X1_control_t hwd_control_state_t;

typedef X1_regmap_t hwd_register_map_t;

typedef X1_context_t hwd_context_t;


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


/* Can these thread structures be moved out of the substrate?
 */
typedef struct _ThreadInfo {
   unsigned pid;
   unsigned tid;
   hwd_context_t context;
   void *event_set_overflowing;
   void *event_set_profiling;
   int domain;
} ThreadInfo_t;

typedef struct _thread_list {
   ThreadInfo_t *master;
   struct _thread_list *next;
} ThreadInfoList_t;


/* Prototypes */
extern int set_domain(hwd_control_state_t * this_state, int domain);
extern int set_granularity(hwd_control_state_t * this_state, int domain);

#endif                          /* _PAPI_X1 */
