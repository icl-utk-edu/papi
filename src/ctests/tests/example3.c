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
  int r, i;
  double a, b, c;
  unsigned long long  ct[3];
  hwd_control_state test;
  EventSetInfo EventSet;
 
  test.number = 0;
  test.counter_code1 = test.counter_code2 = test.sp_code = -1;

  EventSet.machdep = &test;

/* here's where you can set the domain right now:
	1 is default (PAPI_USR)
	2 is PAPI_KERNEL
	3 is PAPI_ALL
	4 is PAPI_OTHER, not supported on this platform
  error checking for invalid values is not going to 
    work when calling the substrate directly in this manner.
    It is implemented when calling from the Low-Level API
*/

  EventSet.domain = 1;   // PAPI_USR

  _papi_hwd_reset(&test);
  _papi_hwd_add_event(&test, PAPI_FP_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_CYC);
  _papi_hwd_start(&EventSet);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }

  _papi_hwd_stop(&test, ct);
  
  puts("\nResults using domain PAPI_USR:\n");
  printf("\tFloating point ins.: 	%lld\n", ct[0]);
  printf("\tTotal Instructions : 	%lld\n", ct[1]);
  printf("\tTotal Cycles : 		%lld\n", ct[2]);

///////////////////////////////////////////////////////
  EventSet.domain = 2;  // PAPI_KERNEL

  _papi_hwd_reset(&test);
  _papi_hwd_add_event(&test, PAPI_FP_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_CYC);
  _papi_hwd_start(&EventSet);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }

  _papi_hwd_stop(&test, ct);

  puts("Results using domain PAPI_KERNEL:\n");
  printf("\tFloating point ins.:        %lld\n", ct[0]);
  printf("\tTotal Instructions :        %lld\n", ct[1]);
  printf("\tTotal Cycles :              %lld\n", ct[2]);

////////////////////////////////////////////////////////
  EventSet.domain = 3;     // PAPI_ALL

  _papi_hwd_reset(&test);
  _papi_hwd_add_event(&test, PAPI_FP_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_INS);
  _papi_hwd_add_event(&test, PAPI_TOT_CYC);
  _papi_hwd_start(&EventSet);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }

  _papi_hwd_stop(&test, ct);

  puts("Results using domain PAPI_ALL:\n");
  printf("\tFloating point ins.:        %lld\n", ct[0]);
  printf("\tTotal Instructions :        %lld\n", ct[1]);
  printf("\tTotal Cycles :              %lld\n", ct[2]);
}
