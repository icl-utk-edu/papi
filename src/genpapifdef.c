/* This file performs the following test: hardware info and which events are available */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#undef NDEBUG
#include <assert.h>
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"

int main(int argc, char **argv) 
{
  int i;
  const PAPI_preset_info_t *info = NULL;

  assert(info = PAPI_query_all_events_verbose());

  printf("#define %s %d\n","PAPI_OK", PAPI_OK);
  printf("#define %s %d\n","PAPI_VER_CURRENT",PAPI_VER_CURRENT);
  printf("#define %s %d\n","PAPI_NUM_ERRORS",PAPI_NUM_ERRORS);
  printf("#define %s %d\n","PAPI_EINVAL",PAPI_EINVAL);
  printf("#define %s %d\n","PAPI_ENOMEM",PAPI_ENOMEM);
  printf("#define %s %d\n","PAPI_ESYS",PAPI_ESYS);
  printf("#define %s %d\n","PAPI_ESBSTR",PAPI_ESBSTR);
  printf("#define %s %d\n","PAPI_ECLOST",PAPI_ECLOST);
  printf("#define %s %d\n","PAPI_EBUG",PAPI_EBUG);
  printf("#define %s %d\n","PAPI_ENOEVNT",PAPI_ENOEVNT);
  printf("#define %s %d\n","PAPI_ECNFLCT",PAPI_ECNFLCT);
  printf("#define %s %d\n","PAPI_ENOTRUN",PAPI_ENOTRUN);
  printf("#define %s %d\n","PAPI_EISRUN",PAPI_EISRUN);
  printf("#define %s %d\n","PAPI_ENOEVST",PAPI_ENOEVST);
  printf("#define %s %d\n","PAPI_ENOTPRESET",PAPI_ENOTPRESET);
  printf("#define %s %d\n","PAPI_ENOCNTR",PAPI_ENOCNTR);
  printf("#define %s %d\n","PAPI_EMISC",PAPI_EMISC);
  printf("#define %s %d\n","PAPI_NULL",PAPI_NULL);
  printf("#define %s %d\n","PAPI_DOM_USER",PAPI_DOM_USER);
  printf("#define %s %d\n","PAPI_DOM_KERNEL",PAPI_DOM_KERNEL);
  printf("#define %s %d\n","PAPI_DOM_OTHER",PAPI_DOM_OTHER);
  printf("#define %s %d\n","PAPI_DOM_ALL",PAPI_DOM_ALL);
  printf("#define %s %d\n","PAPI_STOPPED",PAPI_STOPPED);
  printf("#define %s %d\n","PAPI_RUNNING",PAPI_RUNNING);
  printf("#define %s %d\n","PAPI_OVERFLOWING",PAPI_OVERFLOWING);
  printf("#define %s %d\n","PAPI_PROFILING",PAPI_PROFILING);
  printf("#define %s %d\n","PAPI_QUIET",PAPI_QUIET);
  printf("#define %s %d\n","PAPI_VERB_ECONT",PAPI_VERB_ECONT);
  printf("#define %s %d\n","PAPI_VERB_ESTOP",PAPI_VERB_ESTOP);
  printf("#define %s %d\n","PAPI_MAX_STR_LEN",PAPI_MAX_STR_LEN);

  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    if (info[i].event_name)
      printf("#define %s %d\n",info[i].event_name,info[i].event_code);

  exit(0);
}
