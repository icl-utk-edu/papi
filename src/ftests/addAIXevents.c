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

   #if defined(_POWER4)
      /* defined in Makefile.aix.power4 */
      /* arbitrarily code events from group 28: pm_fpu3 - Floating point events by unit */
      native = 0 | 10 << 8  | 0; /* PM_FPU0_DIV */
      if((*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;  /* JT */
      native = 0 | 19 << 8 | 1; /* PM_FPU1_DIV */
      if((*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;  /* JT */
      native = 0 | 25 << 8  | 2; /* PM_FPU0_FRSP_FCONV */
      if((*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;  /* JT */
      native = 0 | 29 << 8  | 3; /* PM_FPU1_FRSP_FCONV */
      if((*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;  /* JT */
      native = 0 | 11 << 8  | 4; /* PM_FPU0_FMA */
      if((*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;  /* JT */
      native = 0 | 20 << 8 | 5; /* PM_FPU1_FMA */
      if((*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;  /* JT */
      native = 0 | 78 << 8  | 6; /* PM_INST_CMPL */
      if((*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;  /* JT */
      native = 0 | 74 << 8  | 7; /* PM_CYC */
      if((*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;  /* JT */
  #else
      native = 0 | 5 << 8  | 0; /* ICM */
      if ( (*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;
      native = 0 | 35 << 8 | 1; /* FPU1CMPL */
      if ( (*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;
      native = 0 | 5 << 8  | 2; /* LDCM */
      if ( (*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;
      native = 0 | 5 << 8  | 3; /* LDCMPL */
      if ( (*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;
      native = 0 | 5 << 8  | 4; /* FPU0CMPL */
      if ( (*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;
      native = 0 | 12 << 8 | 5; /* CYC */
      if ( (*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;
      native = 0 | 9 << 8  | 6; /* FMA */
      if ( (*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;
      native = 0 | 0 << 8  | 7; /* TLB */
      if ( (*retval = PAPI_add_event(*EventSet, native))!=PAPI_OK) return;
  #endif
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
