/* This file performs the following test: nested eventsets that share 
   all counter values and perform resets.

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
   - Reset eventset 1
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
#ifndef NO_FLOPS
  int mask1 = 0x5, mask2 = 0x5;
#else
  int mask1 = 0x3, mask2 = 0x3;
#endif
  int num_events1, num_events2;
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

  EventSet1 = add_test_events(&num_events1,&mask1);
  EventSet2 = add_test_events(&num_events2,&mask2);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

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

  if ( (retval = PAPI_reset(EventSet1) ) != PAPI_OK ) {
	   mytmp = strdup("PAPI_reset");
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

  remove_test_events(&EventSet1, mask1);
  remove_test_events(&EventSet2, mask2);

  if ( !TESTS_QUIET ) {
  printf("Test case 5: Overlapping start and stop of 2 eventsets with reset.\n");
  printf("------------------------------------------------------------------\n");
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
	 (values[0])[0],(values[1])[0],(values[2])[0],(values[3])[0]);
  printf(TAB4, "PAPI_TOT_CYC: ",
	 (values[0])[1],(values[1])[1],(values[2])[1],(values[3])[1]);
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Column 1 approximately equals column 2\n");
  printf("Column 3 approximately equals two times column 2\n");
  printf("Column 4 approximately equals column 2\n");
  }

  {
 	long long min, max;
	min = values[1][0]*.9;
 	max = values[1][0]*1.1;
	if ( values[0][0] > max || values[0][0] < min || values[2][0]>(2*max)
	  || values[2][0] < (min*2) || values[3][0] < min || values[3][0]>max){
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
	if ( values[0][1] > max || values[0][1] < min || values[2][1]>(2*max)
	  || values[2][1] < (min*2) || values[3][1] < min || values[3][1]>max){
	 	mytmp = strdup("PAPI_TOT_CYC");
		retval = 1;
		goto FAILED;
	}
  }
  printf("fifth:		PASSED\n");
  free_test_space(values, num_tests);
  PAPI_shutdown();
  exit(0);
FAILED:
  printf("fifth:                FAILED\n");
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
