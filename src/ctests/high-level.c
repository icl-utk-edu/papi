/*  This examples show the essentials in using the PAPI high-level
    interface. The program consists of 4 work-loops. The programer
    intends to count the total events for loop 1, 2 and 4, but not 
    include the number of events in loop 3.

    To accomplish this PAPI_read_counters is used as a counter
    reset function, while PAPI_accum_counters is used to sum
    the contributions of loops 2 and 4 into the total count.
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "test_utils.h"

int handle_error(char *file,int line,char *msg,int errcode)
{
  char errstring[PAPI_MAX_STR_LEN];

  fprintf(stderr,"%s: %d:: %s\n",file,line,msg);
  PAPI_perror(errcode,errstring,PAPI_MAX_STR_LEN);
  fprintf(stderr,"%s: %d:: %s\n",file,line,errstring);
  exit(1);
}
int main(int argc, char **argv) 
{
  int retval;
#define NUM_EVENTS 2
  long long values[NUM_EVENTS], dummyvalues[NUM_EVENTS];
  int Events[NUM_EVENTS]={PAPI_FP_INS,PAPI_TOT_CYC};

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    handle_error(__FILE__,__LINE__,"PAPI_library_init",retval);

  retval = PAPI_start_counters(Events,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_start_counters",retval);

  /* Loop 1*/
  do_flops(NUM_FLOPS);
  
  retval = PAPI_read_counters(values,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_read_counters",retval); 

  printf("%12lld %12lld  (Counters continuing...)\n",values[0],values[1]);

  /* Loop 2*/
  do_flops(NUM_FLOPS);
  
  retval = PAPI_accum_counters(values,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_accum_counters",retval); 

  printf("%12lld %12lld  (Counters being ''held'')\n",values[0],values[1]);

  /* Loop 3*/
  /* Simulated code that should not be counted */
  do_flops(NUM_FLOPS);
  
  retval = PAPI_read_counters(dummyvalues,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_read_counters",retval); 
  printf("%12lld %12lld  (Skipped counts)\n",dummyvalues[0],dummyvalues[1]);

  printf("%12s %12s  (''Continuing'' counting)\n","xxx","xxx");
  /* Loop 4*/
  do_flops(NUM_FLOPS);
  
  retval = PAPI_accum_counters(values,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_accum_counters",retval); 

  printf("%12lld %12lld\n",values[0],values[1]);

  PAPI_shutdown();

  printf("----------------------------------\n");  
  printf("Verification: The last line in each experiment was intended\n");
  printf("to become approximately three the value of the first line.\n");
  
  exit(0);
}
