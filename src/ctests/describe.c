/* From Paul Drongowski at HP. Thanks.

/*  I have not been able to call PAPI_describe_event without
    incurring a segv, including the sample code on the man page.
    I noticed that PAPI_describe_event is not exercised by the
    PAPI test programs, so I haven't been able to check the
    function call using known good code. (Or steal your code
    for that matter. :-)
*/

#include "papi_test.h"

extern int TESTS_QUIET; /* Declared in test_utils.c */

int main(int argc, char **argv)
{
   double c,a = 0.999,b = 1.001;
   int n = 1000;
   int EventSet;
   int retval;
   int i, j = 0;
   long_long g1[2];
   int eventcode = PAPI_TOT_INS;
   char eventname[PAPI_MAX_STR_LEN];
   char eventdesc[PAPI_MAX_STR_LEN];

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__, "PAPI_library_init", retval );

   
   if ( (retval = PAPI_create_eventset(&EventSet) ) != PAPI_OK ) 
	test_fail(__FILE__,__LINE__,"PAPI_create_eventset", retval );

   if ( ( retval = PAPI_query_event(eventcode) ) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_query_event(PAPI_TOT_INS)", retval );

   if ( (retval = PAPI_add_event(&EventSet, eventcode) ) != PAPI_OK) 
     test_fail(__FILE__,__LINE__,"PAPI_add_event(PAPI_TOT_INS)",retval);

   if ( (retval = PAPI_start(EventSet) ) != PAPI_OK ) 
     test_fail(__FILE__,__LINE__,"PAPI_start",retval);	   
     
   if ( (retval = PAPI_stop(EventSet, g1) ) != PAPI_OK ) 
     test_fail(__FILE__,__LINE__,"PAPI_stop",retval);	   

   /* Case 0, no info, should fail */
   eventname[0] = '\0';
   eventcode = 0;
   if ( ( retval = PAPI_describe_event(eventname,&eventcode,eventdesc) ) == PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event",retval);	   

   /* Case 1, fill in name field. */
   eventcode = PAPI_TOT_INS;
   eventname[0] = '\0';
   if ( ( retval = PAPI_describe_event(eventname,&eventcode,eventdesc) ) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event",retval);	   

   if (strcmp(eventname,"PAPI_TOT_INS") != 0)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event name value is bogus",retval);	   
   if (strlen(eventdesc) == 0)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event descr value is bogus",retval);	   

   eventcode = 0;

   /* Case 2, fill in code field. */
   if ( ( retval = PAPI_describe_event(eventname,&eventcode,eventdesc) ) != PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event",retval);	   

   if (eventcode != PAPI_TOT_INS)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event code value is bogus",retval);	   

   if (strlen(eventdesc) == 0)
     test_fail(__FILE__,__LINE__,"PAPI_describe_event descr value is bogus",retval);	   

   test_pass(__FILE__, NULL, 0 );
   exit(1);
}
