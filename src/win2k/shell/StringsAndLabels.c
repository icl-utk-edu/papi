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
#include "papi_internal.h"
//#include "test_utils.h"
#include "WinPAPI_protos.h"

#ifndef _WIN32
  extern int PAPI_StringsAndLabels(void);

  int main(int argc, char **argv)
  {
    exit(PAPI_StringsAndLabels());
  }
#endif

int PAPI_StringsAndLabels(void) 
{
  int i, EventCode, errorCode;
  const PAPI_preset_info_t *info = NULL;
  const PAPI_hw_info_t *hwinfo = NULL;
  char name[PAPI_MAX_STR_LEN];
  char descr[PAPI_MAX_STR_LEN];
  char label[PAPI_MAX_STR_LEN];
  char *errorStr;

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    return(1);


	printf("International Event String and Label Test\n");
	printf("-------------------------------------------------------------------------\n");
	
	for (i=0, EventCode = PAPI_L1_DCM; EventCode <= PAPI_FNV_INS; i++, EventCode++)
	{
		name[0] = 0;
		errorCode = PAPI_describe_event(name, &EventCode, descr);
		if (errorCode != PAPI_OK) {
			errorStr = PAPI_strerror(errorCode);
			printf("%s\n", errorStr);
		}
		errorCode = PAPI_label_event(EventCode, label);
		printf("%-20s %-20s %s\n", name, label, descr);
		if (i && i%20 == 0) waitConsole();
	}
	printf("\n");
	
	return(0);
}
