#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"

int main()
{
  int retval;
  int preset;

  retval = PAPI_event_name_to_code( "PAPI_FP_INS", &preset );
  assert(retval == PAPI_OK);
  assert(preset == PAPI_FP_INS);

  exit(0);
}



