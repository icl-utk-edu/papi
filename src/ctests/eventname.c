#include "papi_test.h"

int main()
{
  int retval;
  int preset;


  retval = PAPI_event_name_to_code( "PAPI_FP_INS", &preset );
  if (retval != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_event_name_to_code",retval);
  if (preset != PAPI_FP_INS)
	test_fail(__FILE__,__LINE__,"Wrong preset returned",retval);

  retval = PAPI_event_name_to_code( "PAPI_TOT_CYC", &preset );
  if (retval != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_event_name_to_code",retval);
  if (preset != PAPI_TOT_CYC){
	test_fail(__FILE__,__LINE__,"*preset returned did not equal PAPI_TOT_CYC",retval);

  test_pass(__FILE__,0,0);
}



