#include "papi_test.h"

int main()
{
  int retval;
  int preset;
  char *tmp,buf[128];


  retval = PAPI_event_name_to_code( "PAPI_FP_INS", &preset );
  if (retval != PAPI_OK){
	tmp = strdup("PAPI_event_name_to_code[PAPI_FP_INS]");
	goto FAILED;
  }
  if (preset != PAPI_FP_INS){
	printf("eventname:		FAILED\n");
	printf("*preset returned did not equal PAPI_FP_INS\n");
        exit(1);
  }

  retval = PAPI_event_name_to_code( "PAPI_TOT_CYC", &preset );
  if (retval != PAPI_OK){
	tmp = strdup("PAPI_event_name_to_code[PAPI_TOT_CYC]");
	goto FAILED;
  }
  if (preset != PAPI_TOT_CYC){
	printf("eventname:		FAILED\n");
	printf("*preset returned did not equal PAPI_TOT_CYC\n");
        exit(1);
  }

  printf("eventname:		PASSED\n");
  exit(0);
FAILED:
  printf("eventname:                FAILED\n");
  if ( retval == PAPI_ESYS ) {
        sprintf(buf, "System error in %s:", tmp );
        perror(buf);
  }
  else {
        char errstring[PAPI_MAX_STR_LEN];
        PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN );
        printf("Error in %s: %s\n", tmp, errstring );
  }
  free(tmp);
  exit(1);
}



