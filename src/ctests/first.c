/* This file performs the following test: start, read, stop and again functionality

   - It attempts to use the following three counters. It may use less depending on
     hardware counter resource limitations. These are counted in the default counting
     domain and default granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).
     + PAPI_FP_INS or PAPI_TOT_INS if PAPI_FP_INS doesn't exist
     + PAPI_TOT_CYC
   - Start counters
   - Do flops
   - Read counters
   - Reset counters
   - Do flops
   - Read counters
   - Do flops
   - Read counters
   - Do flops
   - Stop and read counters
   - Read counters
*/

#include "papi_test.h"
int TESTS_QUIET=0; /*Tests is Verbose mode? */

int main(int argc, char **argv) 
{
  int retval, num_tests = 5, num_events, tmp;
  long_long **values;
  int EventSet;
  int i;
#ifndef NO_FLOPS
  int mask = 0x5;
#else
  int mask = 0x3;
#endif
  char *mytmp,buf[128];


  memset( buf, '\0', sizeof(buf) );
  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }

  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        mytmp = strdup("PAPI_library_init");
        goto FAILED;
  }


  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK){
        mytmp = strdup("PAPI_set_debug");
        goto FAILED;
  }

  EventSet = add_test_events(&num_events,&mask);

  values = allocate_test_space(num_tests, num_events);

  if ( (retval = PAPI_start(EventSet) ) != PAPI_OK ) {
            mytmp = strdup("PAPI_start");
            goto FAILED;
  }


  do_flops(NUM_FLOPS);

  if ( (retval = PAPI_read(EventSet, values[0])) != PAPI_OK ) {
	mytmp = strdup("PAPI_read");
	goto FAILED;
  }

  if ( (retval = PAPI_reset(EventSet) ) != PAPI_OK ) {
           mytmp = strdup("PAPI_reset");
           goto FAILED;
  }

  do_flops(NUM_FLOPS);

  if ( (retval = PAPI_read(EventSet, values[1])) != PAPI_OK ) {
	mytmp = strdup("PAPI_read");
	goto FAILED;
  }

  do_flops(NUM_FLOPS);

  if ( (retval = PAPI_read(EventSet, values[2])) != PAPI_OK ) {
	mytmp = strdup("PAPI_read");
	goto FAILED;
  }

  do_flops(NUM_FLOPS);

  if ( (retval = PAPI_stop(EventSet, values[3]) ) != PAPI_OK ) {
           mytmp = strdup("PAPI_stop");
           goto FAILED;
  }

  if ( (retval = PAPI_read(EventSet, values[4])) != PAPI_OK ) {
	mytmp = strdup("PAPI_read");
	goto FAILED;
  }

  remove_test_events(&EventSet, mask);

  if ( !TESTS_QUIET ) {
  printf("Test case 1: Non-overlapping start, stop, read.\n");
  printf("-----------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-------------------------------------------------------------------------\n");

  printf("Test type   : \t1\t\t2\t\t3\t\t4\t\t5\n");
#ifndef NO_FLOPS
  printf(TAB5, "PAPI_FP_INS : ",
#else
  printf(TAB5, "PAPI_TOT_INS : ",
#endif
	 (values[0])[0],(values[1])[0],(values[2])[0],(values[3])[0],(values[4])[0]);
  printf(TAB5, "PAPI_TOT_CYC: ",
	 (values[0])[1],(values[1])[1],(values[2])[1],(values[3])[1],(values[4])[1]);
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Column 1 approximately equals column 2\n");
  printf("Column 3 approximately equals 2 * column 2\n");
  printf("Column 4 approximately equals 3 * column 2\n");
  printf("Column 4 exactly equals column 5\n");
  }

  {
     	long long min, max;
	min = values[1][0]*.9;
	max = values[1][0]*1.1;

	if ( values[0][0] > max || values[0][0] < min || values[2][0]>(2*max)
	|| values[2][0]<(2*min) || values[3][0]>(3*max)||values[3][0]<(3*min)
	|| values[3][0]!=values[4][0])
	{
#ifndef NO_FLOPS
                mytmp = strdup("PAPI_FP_INS");
#else
                mytmp = strdup("PAPI_TOT_INS");
#endif
printf("min: %lld max: %lld  fir: %lld  sec: %lld  thi:  %lld fou: %lld fif: %lld\n", min, max, values[0][0], values[1][0], values[2][0],values[3][0], values[4][0] );
                retval = 1;
                goto FAILED;
        }

	min = values[1][1]*.9;
	max = values[1][1]*1.1;
	if ( values[0][1] > max || values[0][1] < min || values[2][1]>(2*max)
	|| values[2][1]<(2*min) || values[3][1]>(3*max)||values[3][1]<(3*min)
	|| values[3][1]!=values[4][1])
	{
                mytmp = strdup("PAPI_TOT_CYC");
                retval = 1;
                goto FAILED;
        }
  }

  free_test_space(values, num_tests);
  PAPI_shutdown();
  printf("first:		PASSED\n");
  exit(0);
FAILED:
  printf("first:                FAILED\n");
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
