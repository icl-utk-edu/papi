/* This file performs the following test: hardware info and which events are available */

#include "papi_test.h"
extern int TESTS_QUIET; /* Declared in test_utils.c */


int main(int argc, char **argv) 
{
  int i;
  int retval;
  const PAPI_preset_info_t *info = NULL;
  const PAPI_hw_info_t *hwinfo = NULL;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }


  if ((info = PAPI_query_all_events_verbose()) == NULL)
	test_fail(__FILE__, __LINE__, "PAPI_query_all_events_verbose", 1);

  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
	test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

  if ( !TESTS_QUIET ) {
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
  }
  test_pass(__FILE__, NULL, 0);
  exit(1);
}
