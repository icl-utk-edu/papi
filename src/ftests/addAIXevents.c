/*
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "test_utils.h"
*/

#include "papiStdEventDefs.h"
#include "papi.h"

void addaixevents(int *EventSet, int *retval)
{
#if defined(_AIX)
  int native;

  if (*EventSet == PAPI_NULL) {
	  *retval = PAPI_ENOEVST;
	  return;
  }

      native = 0 | 5 << 8  | 0; /* ICM */
      if ( (*retval = PAPI_add_event(EventSet, native))!=PAPI_OK) return;
      native = 0 | 35 << 8 | 1; /* FPU1CMPL */
      if ( (*retval = PAPI_add_event(EventSet, native))!=PAPI_OK) return;
      native = 0 | 5 << 8  | 2; /* LDCM */
      if ( (*retval = PAPI_add_event(EventSet, native))!=PAPI_OK) return;
      native = 0 | 5 << 8  | 3; /* LDCMPL */
      if ( (*retval = PAPI_add_event(EventSet, native))!=PAPI_OK) return;
      native = 0 | 5 << 8  | 4; /* FPU0CMPL */
      if ( (*retval = PAPI_add_event(EventSet, native))!=PAPI_OK) return;
      native = 0 | 12 << 8 | 5; /* CYC */
      if ( (*retval = PAPI_add_event(EventSet, native))!=PAPI_OK) return;
      native = 0 | 9 << 8  | 6; /* FMA */
      if ( (*retval = PAPI_add_event(EventSet, native))!=PAPI_OK) return;
      native = 0 | 0 << 8  | 7; /* TLB */
      if ( (*retval = PAPI_add_event(EventSet, native))!=PAPI_OK) return;
#endif
}

void ADDAIXEVENTS(int *EventSet, int *retval)
{
	addaixevents(EventSet, retval);
}

void addaixevents_(int *EventSet, int *retval)
{
	addaixevents(EventSet, retval);
}

void addaixevents__(int *EventSet, int *retval)
{
	addaixevents(EventSet, retval);
}
