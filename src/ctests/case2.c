/* From Dave McNamara at PSRV. Thanks! */

/* If an event is countable but you've exhausted the counter resources
and you try to add an event, it seems subsequent PAPI_start and/or
PAPI_stop will causes a Seg. Violation.

   I got around this by calling PAPI to get the # of countable events,
then making sure that I didn't try to add more than these number of
events. I still have a problem if someone adds Level 2 cache misses
and then adds FLOPS 'cause I didn't count FLOPS as actually requiring
2 counters. */

#include "papi_test.h"

int TESTS_QUIET=0; /* Tests in Verbose mode? */

int main(int argc, char **argv)
{
   double c,a = 0.999,b = 1.001;
   int n = 1000;
   int EventSet;
   int retval;
   int j = 0,i;
   long_long g1[3];
   char *tmp,buf[128];


  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        tmp = strdup("PAPI_library_init");
        goto FAILED;
  }

   if ( (retval = PAPI_create_eventset(&EventSet) ) != PAPI_OK ) {
        tmp = strdup("PAPI_create_eventset");
        goto FAILED;
   }

   if (PAPI_query_event(PAPI_BR_CN) == PAPI_OK)
     j++;

  if(j==1&&(retval = PAPI_add_event(&EventSet, PAPI_BR_CN)) != PAPI_OK) {
        if ( retval != PAPI_ECNFLCT ){
          tmp = strdup("PAPI_add_event[PAPI_BR_CN]");
          goto FAILED;
	}
   }

   i = j;
   if (PAPI_query_event(PAPI_TOT_CYC) == PAPI_OK)
     j++;

   if (j==(i+1)&&(retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC)) != PAPI_OK){
        if ( retval != PAPI_ECNFLCT ){
        tmp = strdup("PAPI_add_event[PAPI_TOT_CYC]");
        goto FAILED;
	}
   }

   i = j;
   if (PAPI_query_event(PAPI_TOT_INS) == PAPI_OK)           
     j++;

   if (j==(i+1)&&(retval = PAPI_add_event(&EventSet, PAPI_TOT_INS)) != PAPI_OK){
        if ( retval != PAPI_ECNFLCT ){
           tmp = strdup("PAPI_add_event[PAPI_TOT_INS]");
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
  printf("case2:                PASSED\n");
  exit(0);
FAILED:
  printf("case2:                FAILED\n");
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
