/*  This examples show the essentials in using the PAPI high-level
    interface. The program consists of 4 work-loops. The programer
    intends to count the total events for loop 1, 2 and 4, but not 
    include the number of events in loop 3.

    To accomplish this PAPI_read_counters is used as a counter
    reset function, while PAPI_accum_counters is used to sum
    the contributions of loops 2 and 4 into the total count.
*/

#include "papi_test.h"
int TESTS_QUIET=0; /* Tests in Verbose mode? */

void handle_error(char *file,int line,char *msg,int errcode)
{
  char errstring[PAPI_MAX_STR_LEN];

  printf("high-level:		FAILED\n");
  if ( errcode == PAPI_ESYS ) {
        sprintf(errstring, "System error in %s:", msg );
        perror(errstring);
  }
  else {
        PAPI_perror(errcode, errstring, PAPI_MAX_STR_LEN );
        printf("Error in %s: %s\n", msg, errstring );
  }
  exit(1);
}

int main(int argc, char **argv) 
{
  int retval;
#define NUM_EVENTS 2
  long_long values[NUM_EVENTS], dummyvalues[NUM_EVENTS];
  long_long myvalues[NUM_EVENTS];
#ifndef NO_FLOPS
  int Events[NUM_EVENTS]={PAPI_FP_INS,PAPI_TOT_CYC};
#else
  int Events[NUM_EVENTS]={PAPI_TOT_INS,PAPI_TOT_CYC};
#endif
  char *mytmp,buf[128];


  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

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

  if ( !TESTS_QUIET )
     printf(TWO12, values[0], values[1], "(Counters continuing...)\n");

  myvalues[0] = values[0];
  myvalues[1] = values[1];
  /* Loop 2*/
  do_flops(NUM_FLOPS);
  
  retval = PAPI_accum_counters(values,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_accum_counters",retval); 

  if ( !TESTS_QUIET )
     printf(TWO12, values[0], values[1], "(Counters being ''held'')\n");

  /* Loop 3*/
  /* Simulated code that should not be counted */
  do_flops(NUM_FLOPS);
  
  retval = PAPI_read_counters(dummyvalues,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_read_counters",retval); 
  if ( !TESTS_QUIET )
     printf(TWO12, dummyvalues[0], dummyvalues[1], "(Skipped counts)\n");

  if ( !TESTS_QUIET )
     printf("%12s %12s  (''Continuing'' counting)\n","xxx","xxx");
  /* Loop 4*/
  do_flops(NUM_FLOPS);
  
  retval = PAPI_accum_counters(values,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_accum_counters",retval); 

  if ( !TESTS_QUIET )
     printf(TWO12, values[0], values[1], "");

  PAPI_shutdown();

  if ( !TESTS_QUIET ) {
  printf("----------------------------------\n");  
  printf("Verification: The last line in each experiment was intended\n");
  printf("to become approximately three the value of the first line.\n");
  }
  
  {
	long_long min, max;
	min = myvalues[0]*.9;
	max = myvalues[0]*1.1;
	if ( values[0]<(3*min)||values[0]>(3*max)){
#ifndef NO_FLOPS
                mytmp = strdup("PAPI_FP_INS");
#else
                mytmp = strdup("PAPI_TOT_INS");
#endif
                retval = 1;
                goto FAILED;
	}
	min = myvalues[1]*.9;
	max = myvalues[1]*1.1;
	if ( values[1]<(3*min)||values[1]>(3*max)){
	        mytmp = strdup("PAPI_TOT_CYC");
                retval = 1;
                goto FAILED;
	}
  }
  printf("high-level:		PASSED\n");
  exit(0);
FAILED:
  printf("high-level:                FAILED\n");
  if ( retval == PAPI_ESYS ) {
        sprintf(buf, "System error in %s:", mytmp );
        perror(buf);
  }
  else if ( retval > 0 ) {
        printf("Error calculating: %s\n", mytmp );
  }
  else {
        char errstring[PAPI_MAX_STR_LEN];
        PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN );
        printf("Error in %s: %s\n", mytmp, errstring );
  }
  free(mytmp);
  exit(1);
}
