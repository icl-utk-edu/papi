#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
  #include <unistd.h>
#endif
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#undef NDEBUG
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "WinPAPI_protos.h"

#ifndef _WIN32
  int main(int argc, char **argv)
  {
    exit(PAPI_Errors());
  }
#endif

int PAPI_Errors(void) 
{
  int i;
  char *errorStr;

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    return(1);


	printf("International Error String Test\n");
	printf("-------------------------------------------------------------------------\n");
	
	for (i=1; i < PAPI_NUM_ERRORS; i++)
	{
		errorStr = PAPI_strerror(-i);
		printf("%s\n", errorStr);
	}
	printf("\n");
	
	return(0);
}
