#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <stropts.h>
#include <unistd.h>
#include <sys/times.h>
#include <sys/sysinfo.h>
#include <sys/procfs.h>

typedef unsigned long ulong_t;

#include <assert.h>
#include <sys/ucontext.h>

/* Below can be removed when we stop using rusuage for PAPI_get_virt_usec -KSL*/
#include <sys/resource.h>

#include "papi.h"

#define EV_MAX_COUNTERS 3
#define EV_MAX_CPUS 32

typedef union {
   long ev6;
   long ev5;
} ev_control_t;

typedef struct hwd_control_state {
   /* File descriptor controlling the counters; */
   int fd;
   /* Which counters to use? Bits encode counters to use, may be duplicates */
   int selector;
   /* Is this event derived? */
   int derived;
   /* Buffer to pass to the kernel to control the counters */
   ev_control_t counter_cmd;
   /* Interrupt interval */
   int timer_ms;
} hwd_control_state_t;

#include "papi_internal.h"

/* Preset structure */

typedef struct hwd_preset {
   /* Which counters to use? Bits encode counters to use, may be duplicates */
   unsigned char selector;
   /* Is this event derived? */
   unsigned char derived;
   /* If the derived event is not associative, this index is the lead operand */
   unsigned char operand_index;
   /* Buffer to pass to the kernel to control the counters */
   long counter_cmd[EV_MAX_COUNTERS];
   /* Footnote to append to the description of this event */
   char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

typedef struct hwd_search {
   /* PAPI preset code */
   unsigned int papi_code;
   /* Events to encode */
   long findme[EV_MAX_COUNTERS];
} hwd_search_t;

extern unsigned long _etext, _ftext;

#define EV3_CPU                 1       /* EV3                  */
#define EV4_CPU                 2       /* EV4 (21064)          */
#define LCA4_CPU                4       /* LCA4 (21066/21068)   */
#define EV5_CPU                 5       /* EV5 (21164)          */
#define EV45_CPU                6       /* EV4.5 (21064/xxx)    */
#define EV56_CPU                7       /* EV5.6 (21164A)       */
#define EV6_CPU                 8       /* EV6 (21264)          */
#define PCA56_CPU               9       /* EV5.6 (21164PC)      */
#define PCA57_CPU               10      /* EV5.7 (21164PC)      */
#define EV67_CPU                11      /* EV6.7 (21264A)       */
#define EV68CB_CPU              12      /* EV6.8CB (21264C)     */
#define EV68AL_CPU              13      /* EV6.8AL (21264B)     */
#define EV68CX_CPU              14      /* EV6.8CX (21264D)     */
