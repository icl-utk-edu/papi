/* This file performs the following test: nested eventsets that share 
   all counter values, but added in reverse order.

   - It attempts to use two eventsets simultaneously. These are counted 
   in the default counting domain and default granularity, depending on 
   the platform. Usually this is the user domain (PAPI_DOM_USER) and 
   thread context (PAPI_GRN_THR).

     Eventset 1 and 2 both have:
     + PAPI_FP_INS or PAPI_TOT_INS if PAPI_FP_INS doesn't exist
     + PAPI_TOT_CYC

   - Start eventset 1
   - Do flops
   - Stop eventset 1
   - Start eventset 1
   - Do flops
   - Start eventset 2
   - Do flops
   - Stop and read eventset 2
   - Do flops
   - Stop and read eventset 1
   - Start eventset 2
   - Do flops
   - Stop eventset 2
*/

#include "papi_test.h"
int TESTS_QUIET=0; /* Tests in Verbose mode? */


int main(int argc, char **argv) 
{
  int retval, num_tests = 4, tmp;
  long_long **values;
  int EventSet1, EventSet2;
  int num_events = 2;
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

  if ( !TESTS_QUIET ) {
  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK){
        mytmp = strdup("PAPI_set_debug");
        goto FAILED;
  }
  }

   if ( (retval = PAPI_create_eventset(&EventSet1) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_create_eventset");
        goto FAILED;
   }
   if ( (retval = PAPI_create_eventset(&EventSet2) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_create_eventset");
        goto FAILED;
   }

  if ((retval = PAPI_add_event(&EventSet1, PAPI_TOT_CYC)) != PAPI_OK){
	mytmp = strdup( "PAPI_add_event[PAPI_TOT_CYC]");
	goto FAILED;
  }
#ifndef NO_FLOPS
  if ((retval = PAPI_add_event(&EventSet1, PAPI_FP_INS)) != PAPI_OK){
	mytmp = strdup( "PAPI_add_event[PAPI_FP_INS]");
#else
  if ((retval = PAPI_add_event(&EventSet1, PAPI_TOT_INS)) != PAPI_OK){
	mytmp = strdup( "PAPI_add_event[PAPI_TOT_INS]");
#endif
	goto FAILED;
  }

  if ((retval = PAPI_add_event(&EventSet2, PAPI_TOT_CYC)) != PAPI_OK){
	mytmp = strdup( "PAPI_add_event[PAPI_TOT_CYC]");
	goto FAILED; 
  }
#ifndef NO_FLOPS
  if ((retval = PAPI_add_event(&EventSet2, PAPI_FP_INS)) != PAPI_OK){
	if( retval != PAPI_ECNFLCT ) {
	   mytmp = strdup( "PAPI_add_event[PAPI_FP_INS]");
#else
  if ((retval = PAPI_add_event(&EventSet1, PAPI_TOT_INS)) != PAPI_OK){
	if( retval != PAPI_ECNFLCT ) {
	   mytmp = strdup( "PAPI_add_event[PAPI_TOT_INS]");
#endif
	   goto FAILED;
	}
	goto PASSED;
  }
  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events);

  if ( (retval = PAPI_start(EventSet1) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_start");
        goto FAILED;
  }
  do_flops(NUM_FLOPS);
  if ( (retval = PAPI_stop(EventSet1, values[0]) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_stop");
        goto FAILED;
  }

  if ( (retval = PAPI_start(EventSet1) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_start");
        goto FAILED;
  }

  do_flops(NUM_FLOPS);
  if ( (retval = PAPI_start(EventSet2) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_start");
        goto FAILED;
  }
  do_flops(NUM_FLOPS);

  if ( (retval = PAPI_stop(EventSet2, values[1]) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_stop");
        goto FAILED;
  }
  do_flops(NUM_FLOPS);

  if ( (retval = PAPI_stop(EventSet1, values[2]) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_stop");
        goto FAILED;
  }
  if ( (retval = PAPI_start(EventSet2) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_start");
        goto FAILED;
  }
  do_flops(NUM_FLOPS);
  if ( (retval = PAPI_stop(EventSet2, values[3]) ) != PAPI_OK ) {
        mytmp = strdup("PAPI_stop");
        goto FAILED;
  }
  
  if ( !TESTS_QUIET ) {
  printf("Test case 4: Overlapping start and stop of 2 eventsets.\n");
  printf("-------------------------------------------------------\n");
  tmp = PAPI_get_opt(PAPI_GET_DEFDOM,NULL);
  printf("Default domain is: %d (%s)\n",tmp,stringify_domain(tmp));
  tmp = PAPI_get_opt(PAPI_GET_DEFGRN,NULL);
  printf("Default granularity is: %d (%s)\n",tmp,stringify_granularity(tmp));
  printf("Using %d iterations of c += a*b\n",NUM_FLOPS);
  printf("-------------------------------------------------------------------------\n");

  printf("Test type   : \t1\t\t2\t\t3\t\t4\n");
#ifndef NO_FLOPS
  printf(TAB4, "PAPI_FP_INS : ",
#else
  printf(TAB4, "PAPI_TOT_INS : ",
#endif
	 (values[0])[1],(values[1])[0],(values[2])[1],(values[3])[0]);
  printf(TAB4, "PAPI_TOT_CYC: ",
	 (values[0])[0],(values[1])[1],(values[2])[0],(values[3])[1]);
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Column 1 approximately equals column 2\n");
  printf("Column 3 approximately equals three times column 2\n");
  printf("Column 4 approximately equals column 2\n");
  }
  {
	long long min, max;
	min = values[1][0]*.9;
	max = values[1][0]*1.1;
	if ( values[0][1] > max || values[0][1]<min || values[2][1]>(3*max)||
	     values[2][1]<(3*min)||values[3][0]<min||values[3][0]>max ){
#ifndef NO_FLOPS
                mytmp = strdup("PAPI_FP_INS");
#else
                mytmp = strdup("PAPI_TOT_INS");
#endif
                retval = 1;
                goto FAILED;
        }
	min = values[1][1]*.9;
	max = values[1][1]*1.1;
	if ( values[0][0] > max || values[0][0]<min || values[2][0]>(3*max)||
	     values[2][0]<(3*min)||values[3][1]<min||values[3][1]>max ){
                mytmp = strdup("PAPI_TOT_CYC");
                retval = 1;
                goto FAILED;
        }
  }
PASSED:
  printf("fourth-inv:		PASSED\n");
  free_test_space(values, num_tests);
  PAPI_shutdown();
  exit(0);
FAILED:
  printf("fourth-inv:                FAILED\n");
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
