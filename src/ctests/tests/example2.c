
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
#include <memory.h>
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
  int i, j;
  double a, b, c;
  unsigned long long  ct[3];
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
  for (j=0; j < 5; j++)
  { for (i=0; i < 10000000; i++) 
    { c = a*b; }
    _papi_hwd_read(&EventSet, ct);
    printf("\tFloating point ins.:        %lld\n", ct[0]);
    printf("\tTotal Instructions :        %lld\n", ct[1]);
    printf("\tTotal Cycles :              %lld\n\n", ct[2]);
  }

  _papi_hwd_stop(&EventSet, ct);
  
  printf("\tFloating point ins.: 	%lld\n", ct[0]);
  printf("\tTotal Instructions : 	%lld\n", ct[1]);
  printf("\tTotal Cycles : 		%lld\n", ct[2]);
}
