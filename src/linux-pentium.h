/* $Id$ */

#include <perf.h>
#include <asm/system.h> 
#include <linux/unistd.h>	

typedef struct _hwd_preset {
  int number;                
  int counter_code1;
  int counter_code2;
  int sp_code;   /* possibly needed for setting certain registers to 
                    enable special purpose counters */
} hwd_control_state_t;



