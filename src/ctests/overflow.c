/* $Id$ */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"

#define TESTNUM 100000000

void handler(int EventSet, int count, int eventcode, unsigned long long value, int *threshold, void *context)
{
  fprintf(stderr,"handler(%d, %d, %x, %llu, %d, %p) YES!!!!!!\n",
	  count,EventSet,eventcode,value,*threshold,context);
}

int main() 
{
  int i, n = 2;
  double a, b, c;
  unsigned long long *ct;
  int EventSet = PAPI_NULL;

  ct = (unsigned long long *)malloc(n*sizeof(unsigned long long));
  assert(ct!=NULL);
  memset(ct,0x00,n*sizeof(unsigned long long));

  assert(PAPI_num_events() >= PAPI_OK);

  assert(PAPI_query_event(PAPI_TOT_CYC) == PAPI_OK);

  assert(PAPI_add_event(&EventSet, PAPI_TOT_CYC) == PAPI_OK);

  assert(PAPI_query_event(PAPI_FP_INS) == PAPI_OK);

  assert(PAPI_add_event(&EventSet, PAPI_FP_INS) == PAPI_OK);

  assert(PAPI_overflow(EventSet, PAPI_FP_INS, 1000000, 0, handler) == PAPI_OK);

  assert(PAPI_start(EventSet) == PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  
  assert(PAPI_stop(EventSet, ct) == PAPI_OK);

  assert(PAPI_rem_event(&EventSet, PAPI_FP_INS) == PAPI_OK);

  assert(PAPI_rem_event(&EventSet, PAPI_TOT_CYC) == PAPI_OK);

  PAPI_shutdown();

  free(ct);
  
  exit(0);
}
