#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/pfcntr.h>
#include <sys/ioctl.h>
#include <sys/timers.h>
#include <stropts.h>
#include <unistd.h>
#include <sys/processor.h>
#include <sys/times.h>
#include <sys/sysinfo.h>
#include <sys/procfs.h>
#include <machine/hal_sysinfo.h>
#include <machine/cpuconf.h>
#include <assert.h>
#include <sys/ucontext.h>
/* Below can be removed when we stop using rusuage for PAPI_get_virt_usec -KSL*/
#include <sys/resource.h>

#include "papi.h"

#define EV_MAX_COUNTERS 3
#define EV_MAX_CPUS 32

typedef union {
   struct pfcntrs_ev6 ev6;
   struct pfcntrs_ev5 ev5;
   struct pfcntrs ev4;
} ev_values_t;

typedef union {
   long ev6;
   long ev5;
   struct iccsr ev4;
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
