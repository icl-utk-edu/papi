/* took pflops.c example from perf and replaced calls 
	with PAPI substrate, adding a couple of calls
	to use all 3 counters. 
	Note: this is strange, my first test file did not
	have any 'slippage' in values (floating point 
	event counts were exactly 50000000)
	Is this going to be too inefficient?
*/

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include "papiStdEventDefs.h"

typedef struct _hwd_preset {
  int number;
  int counter_code1;
  int counter_code2;
  int sp_code;   
} hwd_control_state;

void main() {
  int r, i;
  double a, b, c;
  unsigned long long  ct[2];
  hwd_control_state test;
 
  test.number = 0;
  test.counter_code1 = test.counter_code2 = test.sp_code = -1;

  _papi_hwd_reset(&test);
  _papi_hwd_add_event(&test, PAPI_FP_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_CYC);
  _papi_hwd_start(&test);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }

  _papi_hwd_stop(&test, ct);
  
  printf("\tFloating point ins.: 	%lld\n", ct[0]);
  printf("\tTotal Instructions : 	%lld\n", ct[1]);
  printf("\tTotal Cycles : 		%lld\n", ct[2]);
}
