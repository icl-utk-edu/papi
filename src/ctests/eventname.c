#include "papi_test.h"

int main()
{
  int retval;
  int preset;

  retval = PAPI_event_name_to_code( "PAPI_FP_INS", &preset );
  if (retval != PAPI_OK)
    exit(1);
  if (preset != PAPI_FP_INS)
    exit(1);

  retval = PAPI_event_name_to_code( "PAPI_TOT_CYC", &preset );
  if (retval != PAPI_OK)
    exit(1);
  if (preset != PAPI_TOT_CYC)
    exit(1);

  exit(0);
}



