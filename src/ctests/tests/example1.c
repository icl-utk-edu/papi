/* took pflops.c example from perf and replaced calls 
	with PAPI substrate, adding a couple of calls
	to use all 3 counters. 
	Note: this is strange, my first test file did not
	have any 'slippage' in values (floating point 
	event counts were exactly 50000000)
	Is this going to be too inefficient?
*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>

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
  unsigned long long ct[3];
  EventSetInfo EventSetZero;
  EventSetInfo EventSet;
  hwd_control_state_t test;

  memset(&EventSetZero,0x00,sizeof(hwd_control_state_t));
  memset(&test,0x00,sizeof(hwd_control_state_t));
  EventSet.machdep = &test;
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
  
  printf("\tFloating point ins.: 	%lld\n", ct[0]);
  printf("\tTotal Instructions : 	%lld\n", ct[1]);
  printf("\tTotal Cycles : 		%lld\n", ct[2]);
}
