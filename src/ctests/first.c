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
  int EventSet = PAPI_NULL;

  r=PAPI_add_event(&EventSet, PAPI_FP_INS);
  r=PAPI_add_event(&EventSet, PAPI_TOT_INS);
  r=PAPI_add_event(&EventSet, PAPI_TOT_CYC);
  r=PAPI_reset(EventSet);
  r=PAPI_start(EventSet);

  a = 0.5;
  b = 6.2;
  for (i=0; i < 50000000; i++) {
    c = a*b;
  }
  
  r=PAPI_stop(EventSet, ct);
}
