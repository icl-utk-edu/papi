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
  int r, i, n = 0;
  double a, b, c;
  unsigned long long *ct;
  int EventSet = PAPI_NULL;

  r = PAPI_num_events();
  assert(r>=PAPI_OK);

  if (PAPI_query(PAPI_TOT_CYC) == PAPI_OK)
    {
      r=PAPI_add_event(&EventSet, PAPI_TOT_CYC);
      if (r >= PAPI_OK)
	n++;
    }

  if (PAPI_query(PAPI_TOT_INS) == PAPI_OK)
    {
      r=PAPI_add_event(&EventSet, PAPI_TOT_INS);
      if (r >= PAPI_OK)
	n++;
    }

  if (PAPI_query(PAPI_FP_INS) == PAPI_OK)
    {
      r=PAPI_add_event(&EventSet, PAPI_FP_INS);
      if (r >= PAPI_OK)
	n++;
    }
  
  ct = (unsigned long long *)malloc(n*sizeof(unsigned long long));
  cr = (unsigned long long *)malloc(n*sizeof(unsigned long long));
  cs = (unsigned long long *)malloc(n*sizeof(unsigned long long));
  cu = (unsigned long long *)malloc(n*sizeof(unsigned long long));
  assert(ct!=NULL);
  assert(cr!=NULL);
  assert(cs!=NULL);
  assert(cu!=NULL);
  memset(ct,0x00,n*sizeof(unsigned long long));
  memset(cr,0x00,n*sizeof(unsigned long long));
  memset(cs,0x00,n*sizeof(unsigned long long));
  memset(cu,0x00,n*sizeof(unsigned long long));

  r=PAPI_start(EventSet);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  
  r=PAPI_read(EventSet, cr);
  assert(r>=PAPI_OK);

  r=PAPI_reset(EventSet, cr);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }

  r=PAPI_read(EventSet, cs);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }

  r=PAPI_stop(EventSet, ct);
  assert(r>=PAPI_OK);

  r=PAPI_read(EventSet, cu);
  assert(r>=PAPI_OK);

  if (n > 2) 
    {
      r=PAPI_rem_event(&EventSet, PAPI_FP_INS);
      assert(r>=PAPI_OK);
    }

  if (n > 1) 
    {
      r=PAPI_rem_event(&EventSet, PAPI_TOT_INS);
      assert(r>=PAPI_OK);
    }

  r=PAPI_rem_event(&EventSet, PAPI_TOT_CYC);
  assert(r>=PAPI_OK); 

  PAPI_shutdown();

  printf("%d iterations of c = a*b\n",TESTNUM);
  printf("Cycles: %lld %lld %lld %lld\n",cr[0],cs[0],ct[0],cu[0]);
  printf("Instrs: %lld %lld %lld %lld\n",cr[1],cs[1],ct[1],cu[1]);
  printf("Flinst: %lld %lld %lld %lld\n",cr[2],cs[2],ct[2],cu[2]);
  printf("col 1 ~= col 2, col 2 ~= 2 * col 3, col 3 ~= col4\n");

  free(cr);
  free(cs);
  free(ct);
  free(cu);
  
  exit(0);
}
