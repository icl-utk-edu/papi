/* Test for nested start/stop. */

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
  unsigned long long *cr,*cs;
  int EventSet1 = PAPI_NULL;
  int EventSet2 = PAPI_NULL;

  r = PAPI_num_events();
  assert(r>=PAPI_OK);

  r=PAPI_add_event(&EventSet1, PAPI_TOT_CYC);
  assert(r>=PAPI_OK);
  r=PAPI_add_event(&EventSet1, PAPI_TOT_INS);
  assert(r>=PAPI_OK);

  r=PAPI_add_event(&EventSet2, PAPI_FP_INS);
  assert(r>=PAPI_OK);
  r=PAPI_add_event(&EventSet2, PAPI_TOT_CYC);
  assert(r>=PAPI_OK);
  
  cr = (unsigned long long *)malloc(n*sizeof(unsigned long long));
  cs = (unsigned long long *)malloc(n*sizeof(unsigned long long));
  assert(cr!=NULL);
  assert(cs!=NULL);
  memset(cr,0x00,n*sizeof(unsigned long long));
  memset(cs,0x00,n*sizeof(unsigned long long));

  r=PAPI_start(EventSet1);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  
  r=PAPI_start(EventSet2);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }


  r=PAPI_stop(EventSet1, cr);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }

  r=PAPI_stop(EventSet2, cs);
  assert(r>=PAPI_OK);

  PAPI_shutdown();

  printf("%d iterations of c = a*b\n",TESTNUM);
  printf("Cycles: %llu ~= %llu ?\n",cr[0],cs[1]);
  printf("Flinst: %llu %llu\n",0ULL,cs[0]);
  printf("Instrs: %llu %llu\n\n",cr[1],0ULL);

  free(cr);
  free(cs);
  
  exit(0);
}
