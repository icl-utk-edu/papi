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

extern caddr_t _end;
extern caddr_t _fini;
extern caddr_t _start;

int main() 
{
  int i, n = 2;
  double a, b, c;
  unsigned long long *ct;
  int EventSet = PAPI_NULL;
  unsigned short *profbuf;
  unsigned long length = 65536;

  printf("Text start %p, Text end %p, Text finish %p\n",&_start,&_end,&_fini);

  profbuf = (unsigned short *)malloc(length*sizeof(unsigned short));
  assert(profbuf != NULL);
  memset(profbuf,0x00,length*sizeof(unsigned short));

  ct = (unsigned long long *)malloc(n*sizeof(unsigned long long));
  assert(ct!=NULL);
  memset(ct,0x00,n*sizeof(unsigned long long));

  assert(PAPI_num_events() >= PAPI_OK);

  assert(PAPI_query_event(PAPI_TOT_CYC) >= PAPI_OK);

  assert(PAPI_add_event(&EventSet, PAPI_TOT_CYC) >= PAPI_OK);

  assert(PAPI_query_event(PAPI_FP_INS) >= PAPI_OK);

  assert(PAPI_add_event(&EventSet, PAPI_FP_INS) >= PAPI_OK);

  assert(PAPI_profil(profbuf, length, (caddr_t)&_start, 65536, 
		     EventSet, PAPI_FP_INS, 250000, PAPI_PROFIL_POSIX) >= PAPI_OK);

  assert(PAPI_start(EventSet) >= PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  
  assert(PAPI_stop(EventSet, ct) >= PAPI_OK);

  assert(PAPI_rem_event(&EventSet, PAPI_FP_INS) >= PAPI_OK);

  assert(PAPI_rem_event(&EventSet, PAPI_TOT_CYC) >= PAPI_OK);

  PAPI_shutdown();

  printf("PAPI_profil() counts.\n");
  printf("Address\tCount\n");
  for (i=0;i<length;i++)
    {
      if (profbuf[i])
	printf("0x%x\t%d\n",(int)&_start + i,profbuf[i]);
    }

  free(ct);

  free(profbuf);

  exit(0);
}
