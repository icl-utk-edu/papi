/* $Id$ */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <assert.h>
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"

#define TESTNUM 10000000

int main(int argc, char **argv) 
{
  int r, i, n = 3;
  double a, b, c;
  unsigned long long *cr,*cs,*ct, *cu;
  int EventSet1 = PAPI_NULL;
  int EventSet2 = PAPI_NULL;

  r = PAPI_num_events();
  assert(r>=PAPI_OK);

  r=PAPI_add_event(&EventSet1, PAPI_TOT_CYC);
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
  
  r=PAPI_start(EventSet2);
  assert(r>=PAPI_OK);

  r=PAPI_read(EventSet1, cr);
  assert(r>=PAPI_OK);
  fprintf(stderr,"(flops) read of eventset 1 after eventset 2 started\n");

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }

  r=PAPI_read(EventSet2, cs);
  assert(r>=PAPI_OK);
  fprintf(stderr,"(flops) read of eventset 2 after both started\n");

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }

  r=PAPI_stop(EventSet1, ct);
  assert(r>=PAPI_OK);
  fprintf(stderr,"(flops) stop of eventset 1 after both started\n");

  r=PAPI_stop(EventSet2, cu);
  assert(r>=PAPI_OK);
  fprintf(stderr,"stop of eventset 2 after eventset 1 stopped\n");

  PAPI_shutdown();

  printf("(flops) = %d iterations of c = a*b\n",TESTNUM);
  printf("Cycles: %llu %llu %llu %llu\n",cr[0],cs[0],ct[0],cu[0]);
  printf("Flinst: %llu %llu %llu %llu\n",0ULL,cs[1],0ULL,cu[1]);
  printf("Instrs: %llu %llu %llu %llu\n",cr[1],cs[2],ct[1],cu[2]);

  free(cr);
  free(cs);
  free(ct);
  free(cu);
  
  exit(0);
}
