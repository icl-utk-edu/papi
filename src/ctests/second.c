/* $Id$ */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"

void main() 
{
  int r, i;
  double a, b, c;
  unsigned long long ct[3];
  int EventSet1, EventSet2, EventSet3;
  PAPI_option_t options;

  EventSet1 = EventSet2 = EventSet3 = PAPI_NULL;

/* this program fills creates 3 EventSets with the same
   three Events, but different DOMAINs, runs them, and
   works!
*/

  r=PAPI_add_event(&EventSet1, PAPI_FP_INS);
  r=PAPI_add_event(&EventSet1, PAPI_TOT_INS);
  r=PAPI_add_event(&EventSet1, PAPI_TOT_CYC);

  r=PAPI_add_event(&EventSet2, PAPI_FP_INS);
  r=PAPI_add_event(&EventSet2, PAPI_TOT_INS);
  r=PAPI_add_event(&EventSet2, PAPI_TOT_CYC);

  r=PAPI_add_event(&EventSet3, PAPI_FP_INS);
  r=PAPI_add_event(&EventSet3, PAPI_TOT_INS);
  r=PAPI_add_event(&EventSet3, PAPI_TOT_CYC);

  options.domain.eventset=1;
  options.domain.domain=PAPI_DOM_DEFAULT;
  r=PAPI_set_opt(PAPI_SET_DOMAIN, &options);

  options.domain.eventset=2;
  options.domain.domain=PAPI_DOM_KERNEL;
  r=PAPI_set_opt(PAPI_SET_DOMAIN, &options);

  options.domain.eventset=3;
  options.domain.domain=PAPI_DOM_ALL;
  r=PAPI_set_opt(PAPI_SET_DOMAIN, &options);

///////////////EventSet1 is started/////////
  r=PAPI_reset(EventSet1);
  r=PAPI_start(EventSet1);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }
  r=PAPI_stop(EventSet1, ct);
///////////////EventSet1 is stopped/////////

///////////////EventSet2 is started/////////
  r=PAPI_reset(EventSet2);
  r=PAPI_start(EventSet2);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }
  r=PAPI_stop(EventSet2, ct);
///////////////EventSet2 is stopped/////////

///////////////EventSet3 is started/////////
  r=PAPI_reset(EventSet3);
  r=PAPI_start(EventSet3);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }
  r=PAPI_stop(EventSet3, ct);
///////////////EventSet3 is stopped/////////
}
