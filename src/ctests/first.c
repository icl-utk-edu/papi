/* $Id$ */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"

#define TESTNUM 10000000

void main() 
{
  int r, i;
  double a, b, c;
  unsigned long long *ct;
  int EventSet = PAPI_NULL;

  i = PAPI_num_events();
  assert(i>=2);

  ct = (unsigned long long *)malloc(i*sizeof(unsigned long long));
  assert(ct!=NULL);
  memset(ct,0x00,i*sizeof(unsigned long long));

  r=PAPI_add_event(&EventSet, PAPI_TOT_CYC);
  assert(r>=PAPI_OK);

  r=PAPI_add_event(&EventSet, PAPI_TOT_INS);
  assert(r>=PAPI_OK); 

  r=PAPI_add_event(&EventSet, PAPI_FP_INS);
  assert(r>=PAPI_OK);

  r=PAPI_start(EventSet);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  
  r=PAPI_stop(EventSet, ct);
  assert(r>=PAPI_OK);

  r=PAPI_rem_event(&EventSet, PAPI_FP_INS);
  assert(r>=PAPI_OK);

  r=PAPI_rem_event(&EventSet, PAPI_TOT_INS);
  assert(r>=PAPI_OK);

  r=PAPI_rem_event(&EventSet, PAPI_TOT_CYC);
  assert(r>=PAPI_OK); 

  PAPI_shutdown();

  printf("%d iterations of c = a*b\n",TESTNUM);
  printf("%lld cycles\n%lld instructions\n%lld floating point instructions\n",ct[0],ct[1],ct[2]);
  printf("%f IPC\n%f FPC\n",(float)ct[1]/(float)ct[0],(float)ct[2]/(float)ct[0]);

  free(ct);
  exit(0);
}
