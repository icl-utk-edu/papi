
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
#include "papi.h"

typedef struct _hwd_preset {
  int number;
  int counter_code1;
  int counter_code2;
  int sp_code;  
} hwd_control_state;

typedef struct {
  int eventindex;
  long long deadline;
  int milliseconds;
  papi_overflow_option_t option; } _papi_overflow_info_t;

typedef struct {
  papi_multiplex_option_t option; } _papi_multiplex_info_t;

typedef struct _EventSetInfo {
  int EventSetIndex;
  int NumberOfCounters;
  int *EventCodeArray;
  void *machdep;
  long long *start;
  long long *stop;
  long long *latest;
  int state;
  _papi_overflow_info_t overflow;
  _papi_multiplex_info_t multiplex;
  int granularity;
  int domain;
} EventSetInfo;

void main() {
  int r, i, j;
  double a, b, c;
  unsigned long long  ct[2];
  hwd_control_state test;
  EventSetInfo EventSet;
 
  test.number = 0;
  test.counter_code1 = test.counter_code2 = test.sp_code = -1;

  EventSet.machdep = &test;
  EventSet.domain = 1;		// set to default PAPI_USR

  _papi_hwd_reset(&test);
  _papi_hwd_add_event(&test, PAPI_FP_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_CYC);
  _papi_hwd_start(&EventSet);

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
