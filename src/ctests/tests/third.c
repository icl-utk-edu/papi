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
  int r, i, n = 3;
  double a, b, c;
  unsigned long long *cr,*cs,*ct, *cu;
  int EventSet1 = PAPI_NULL;
  int EventSet2 = PAPI_NULL;

  r = PAPI_num_events();
  assert(r>=PAPI_OK);


  r=PAPI_add_event(&EventSet1, PAPI_TOT_CYC);
  r=PAPI_add_event(&EventSet1, PAPI_FP_INS);
  r=PAPI_add_event(&EventSet1, PAPI_TOT_INS);

  r=PAPI_add_event(&EventSet2, PAPI_TOT_CYC);
  r=PAPI_add_event(&EventSet2, PAPI_FP_INS);
  r=PAPI_add_event(&EventSet2, PAPI_TOT_INS);
  
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

  r=PAPI_start(EventSet1);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  
  r=PAPI_read(EventSet1, cr);
  assert(r>=PAPI_OK);

  r=PAPI_start(EventSet2);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }

  r=PAPI_read(EventSet2, cs);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }

  r=PAPI_stop(EventSet1, ct);
  assert(r>=PAPI_OK);

  r=PAPI_stop(EventSet2, cu);
  assert(r>=PAPI_OK);

  PAPI_shutdown();


  printf("%d iterations of c = a*b\n",TESTNUM);
  printf("Instrs: %lld %lld %lld %lld\n",cr[0],cs[0],ct[0],cu[0]);
  printf("Flinst: %lld %lld %lld %lld\n",cr[1],cs[1],ct[1],cu[1]);

  free(cr);
  free(cs);
  free(ct);
  free(cu);
  
  exit(0);
}
