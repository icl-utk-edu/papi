/* This file performs the following test: hardware info and which events are available */

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

int main(int argc, char **argv) 
{
  int i;
  const PAPI_preset_info_t *info = NULL;
  const PAPI_hw_info_t *hwinfo = NULL;

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  if ((info = PAPI_query_all_events_verbose()) == NULL)
    exit(1);

  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
    exit(1);

  printf("Test case 8: Available events and hardware information.\n");
  printf("-------------------------------------------------------------------------\n");
  printf("Vendor string and code   : %s (%d)\n",hwinfo->vendor_string,hwinfo->vendor);
  printf("Model string and code    : %s (%d)\n",hwinfo->model_string,hwinfo->model);
  printf("CPU revision             : %f\n",hwinfo->revision);
  printf("CPU Megahertz            : %f\n",hwinfo->mhz);
  printf("CPU's in an SMP node     : %d\n",hwinfo->ncpu);
  printf("Nodes in the system      : %d\n",hwinfo->nnodes);
  printf("Total CPU's in the system: %d\n",hwinfo->totalcpus);
  printf("-------------------------------------------------------------------------\n");
  printf("Name\t\tCode\t\tAvail\tDeriv\tDescription (Note)\n");
  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    if (info[i].event_name)
      printf("%s\t0x%x\t%s\t%s\t%s (%s)\n",
	     info[i].event_name,
	     info[i].event_code,
	     (info[i].avail ? "Yes" : "No"),
	     (info[i].flags & PAPI_DERIVED ? "Yes" : "No"),
	     info[i].event_descr,
	     (info[i].event_note ? info[i].event_note : ""));
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Check your architecture and substrate file\n");
  
  exit(0);
}
