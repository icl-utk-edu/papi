/* From Dave McNamara at PSRV. Thanks! */

/* If you try to add an event that doesn't exist, you get the correct error
message, yet you get subsequent Seg. Faults when you try to do PAPI_start and
PAPI_stop. I would expect some bizarre behavior if I had no events added to the
event set and then tried to PAPI_start but if I had successfully added one
event, then the 2nd one get an error when I tried to add it, is it possible for
PAPI_start to work but just count the first event?
*/

#include "papi_test.h"

int TESTS_QUIET=0; /* Tests in Verbose mode? */

int main(int argc, char **argv)
{
   double c,a = 0.999,b = 1.001;
   int n = 1000;
   int EventSet;
   int retval;
   int i, j = 0;
   long_long g1[2];
   char *tmp,buf[128];

   memset( buf, '\0', sizeof(buf) );
   if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
   }


  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        tmp = strdup("PAPI_library_init");
        goto FAILED;
  }

   
   if ( (retval = PAPI_create_eventset(&EventSet) ) != PAPI_OK ) {
	tmp = strdup("PAPI_create_eventset");
	goto FAILED;
   }

   if (PAPI_query_event(PAPI_L2_TCM) == PAPI_OK)
     j++;

  if(j==1&&(retval = PAPI_add_event(&EventSet, PAPI_L2_TCM)) != PAPI_OK) {
        if ( retval != PAPI_ECNFLCT ){
	  tmp = strdup("PAPI_add_event[PAPI_L2_TCM]");
	  goto FAILED;
	}
   }

   i = j;
   if (PAPI_query_event(PAPI_L2_DCM) == PAPI_OK)
     j++;

   if (j==(i+1)&&(retval = PAPI_add_event(&EventSet, PAPI_L2_DCM)) != PAPI_OK){
        if ( retval != PAPI_ECNFLCT ){
	  tmp = strdup("PAPI_add_event[PAPI_L2_DCM]");
	  goto FAILED;
	}
   }

   if (j)
     {
       if ( (retval = PAPI_start(EventSet) ) != PAPI_OK ) {
	    tmp = strdup("PAPI_start");
	    goto FAILED;
	}
       for ( i = 0; i < n; i++ )
	 {
	   c = a * b;
	 }
       if ( (retval = PAPI_stop(EventSet, g1) ) != PAPI_OK ) {
	    tmp = strdup("PAPI_stop");
	    goto FAILED;
	}
     }

   printf("case1:		PASSED\n");
   exit(0);
FAILED:
  printf("case1:                FAILED\n");
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
