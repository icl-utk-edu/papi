
/* added an additional loop to the one in example1.c 
	with reading the counters without stopping
	them- like the other example, a bit of slippage
	in between the end of the loops and the final print
	statements
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
  int r, i, j;
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
  for (j=0; j < 5; j++)
  { for (i=0; i < 10000000; i++) 
    { c = a*b; }
    _papi_hwd_read(&test, ct, test.number);
    printf("\tFloating point ins.:        %lld\n", ct[0]);
    printf("\tTotal Instructions :        %lld\n", ct[1]);
    printf("\tTotal Cycles :              %lld\n\n", ct[2]);
  }

  _papi_hwd_stop(&test, ct);
  
  printf("\tFloating point ins.: 	%lld\n", ct[0]);
  printf("\tTotal Instructions : 	%lld\n", ct[1]);
  printf("\tTotal Cycles : 		%lld\n", ct[2]);
}
