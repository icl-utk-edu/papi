#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/processor.h>
#include <sys/procset.h>
#include <procfs.h>
#include <libcpc.h>
#include <libgen.h>

#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

#define MAX_COUNTERS 3

typedef struct hwd_control_state {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  int selector;  
  /* Is this event derived? */
  int derived;   
  /* Buffer to pass to the kernel to control the counters */
  cpc_event_t counter_cmd;
  /* Flags to kernel */
  int flags;  
} hwd_control_state_t;

typedef struct hwd_preset {
  /* Which counters to use? Bits encode counters to use, may be duplicates */
  unsigned char selector;  
  /* Is this event derived? */
  unsigned char derived;   
  /* If the derived event is not associative, this index is the lead operand */
  unsigned char operand_index;
  /* Buffer to pass to the kernel to control the counters */
  unsigned char counter_cmd[MAX_COUNTERS];
  /* If it exists, then this is the description of this event */
  char note[PAPI_MAX_STR_LEN];
} hwd_preset_t;

#define PCR_PRIV_TRACE	    	0x1
#define PCR_SYS_TRACE	    	(0x1 << 1)
#define PCR_USER_TRACE	    	(0x1 << 2)

#define PCR_CODE_MASK(c)        (0xf << pcr_shift[c])
#define PCR_CODE_INV_MASK(c)    (~(PCR_CODE_MASK(c)))
#define PCR_S0_CYCLE_CNT    	(0x0)
#define PCR_S0_INSTR_CNT    	(0x1)
#define PCR_S0_STALL_IC_MISS 	(0x2)
#define PCR_S0_STALL_STORBUF 	(0x3)
#define PCR_S0_IC_REF	    	(0x8)
#define PCR_S0_DC_READ	    	(0x9)
#define PCR_S0_DC_WRITE	    	(0xa)
#define PCR_S0_STALL_LOAD   	(0xb)
#define PCR_S0_EC_REF	    	(0xc)
#define PCR_S0_EC_WRITE_RO  	(0xd)
#define PCR_S0_EC_SNOOP_INV 	(0xe)
#define PCR_S0_EC_READ_HIT  	(0xf)

#define PCR_S1_CYCLE_CNT    	(0x0)
#define PCR_S1_INSTR_CNT    	(0x1)
#define PCR_S1_STALL_MISPRED 	(0x2)
#define PCR_S1_STALL_FPDEP	(0x3)
#define PCR_S1_IC_HIT		(0x8)
#define PCR_S1_DC_READ_HIT	(0x9)
#define PCR_S1_DC_WRITE_HIT	(0xa)
#define PCR_S1_LOAD_STALL_RAW	(0xb)
#define PCR_S1_EC_HIT		(0xc)
#define PCR_S1_EC_WRITEBACK	(0xd)
#define PCR_S1_EC_SNOOP_COPYBCK	(0xe)
#define PCR_S1_EC_IC_HIT	(0xf)
