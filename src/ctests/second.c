#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <memory.h>
#include <sys/types.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"

#define TESTNUM 10000000

int main(int argc, char **argv) 
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

  memset(&options,0x0,sizeof(options));

  r=PAPI_get_opt(PAPI_GET_DEFDOM, NULL);
  assert(r>=0);
  fprintf(stderr,"Default domain is %x.\n",r);

  options.domain.eventset=EventSet1;
  options.domain.domain=PAPI_DOM_ALL;
  r=PAPI_set_opt(PAPI_SET_DOMAIN, &options);
  assert(r>=PAPI_OK);

  options.domain.eventset=EventSet3;
  options.domain.domain=PAPI_DOM_USER;
  r=PAPI_set_opt(PAPI_SET_DOMAIN, &options);
  assert(r>=PAPI_OK);

  options.domain.eventset=EventSet2;
  options.domain.domain=PAPI_DOM_KERNEL;
  r=PAPI_set_opt(PAPI_SET_DOMAIN, &options);
  assert(r>=PAPI_OK);

  /*  Start EventSet1  */

  fprintf(stderr,"PAPI_DOM_ALL counts\n");
  r=PAPI_start(EventSet1);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  r=PAPI_stop(EventSet1, ct);
  assert(r>=PAPI_OK);

  /*  Stop EventSet1  */

  /*  Start EventSet2  */

  fprintf(stderr,"PAPI_DOM_KERNEL counts\n");
  r=PAPI_start(EventSet2);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  r=PAPI_stop(EventSet2, ct);
  assert(r>=PAPI_OK);

  /*  Start EventSet3  */

  fprintf(stderr,"PAPI_DOM_USER counts\n");
  r=PAPI_start(EventSet3);
  assert(r>=PAPI_OK);

  a = 0.5;
  b = 6.2;
  for (i=0; i < TESTNUM; i++) {
    c = a*b;
  }
  r=PAPI_stop(EventSet3, ct);
  assert(r>=PAPI_OK);

  /*  Stop EventSet3  */

  exit(0);
}
