#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"

/* Header files for the substrates */

#if defined(mips) && defined(unix) && defined(sgi)
#include "irix-mips.h"
#elif defined(i386) && defined(unix) && defined(linux)
#include "linux-pentium.h"
#else
#include "any-null.h"
#endif

void main() {
  int r, i;
  double a, b, c;
  unsigned long long  ct[3];
  hwd_control_state_t test;
  EventSetInfo EventSet;
  EventSetInfo EventSetZero;
 
  memset(&EventSetZero,0x00,sizeof(hwd_control_state_t));
  memset(&test,0x00,sizeof(hwd_control_state_t));
  EventSet.machdep = &test;

/* here's where you can set the domain right now:
	1 is default (PAPI_USER)
	2 is PAPI_KERNEL
	3 is PAPI_ALL
	4 is PAPI_OTHER, not supported on this platform
  error checking for invalid values is not going to 
    work when calling the substrate directly in this manner.
    It is implemented when calling from the Low-Level API
*/

  EventSet.all_options.domain.domain.domain = 1;       /* set to default PAPI_USR */

  _papi_hwd_init(&EventSetZero);
  _papi_hwd_reset(&EventSet);
  _papi_hwd_add_event(&EventSet, PAPI_FP_INS);
  _papi_hwd_add_event(&EventSet, PAPI_TOT_INS);
  _papi_hwd_add_event(&EventSet, PAPI_TOT_CYC);
  _papi_hwd_start(&EventSet);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }

  _papi_hwd_stop(&EventSet, ct);
  
  puts("\nResults using domain PAPI_USER:\n");
  printf("\tFloating point ins.: 	%lld\n", ct[0]);
  printf("\tTotal Instructions : 	%lld\n", ct[1]);
  printf("\tTotal Cycles : 		%lld\n\n", ct[2]);

///////////////////////////////////////////////////////
  EventSet.all_options.domain.domain.domain = 2;       /* set to PAPI_KERNEL */

  _papi_hwd_init(&EventSetZero);
  _papi_hwd_reset(&EventSet);
  _papi_hwd_add_event(&EventSet, PAPI_FP_INS);
  _papi_hwd_add_event(&EventSet, PAPI_TOT_INS);
  _papi_hwd_add_event(&EventSet, PAPI_TOT_CYC);
  _papi_hwd_start(&EventSet);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }

  _papi_hwd_stop(&EventSet, ct);

  puts("Results using domain PAPI_KERNEL:\n");
  printf("\tFloating point ins.:        %lld\n", ct[0]);
  printf("\tTotal Instructions :        %lld\n", ct[1]);
  printf("\tTotal Cycles :              %lld\n\n", ct[2]);

////////////////////////////////////////////////////////
  EventSet.all_options.domain.domain.domain = 3;       /* set to default PAPI_ALL */

  _papi_hwd_init(&EventSetZero);
  _papi_hwd_reset(&EventSet);
  _papi_hwd_add_event(&EventSet, PAPI_FP_INS);
  _papi_hwd_add_event(&EventSet, PAPI_TOT_INS);
  _papi_hwd_add_event(&EventSet, PAPI_TOT_CYC);
  _papi_hwd_start(&EventSet);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }

  _papi_hwd_stop(&EventSet, ct);

  puts("Results using domain PAPI_ALL:\n");
  printf("\tFloating point ins.:        %lld\n", ct[0]);
  printf("\tTotal Instructions :        %lld\n", ct[1]);
  printf("\tTotal Cycles :              %lld\n\n", ct[2]);
}
