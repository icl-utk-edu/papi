/* This file performs the following test: hardware info and which events are available */

#include "papi_test.h"
extern int TESTS_QUIET; /* Declared in test_utils.c */


int main(int argc, char **argv) 
{
  int i;
  int retval;
  int print_avail_only = 0;
  int use_enum = 0;
  PAPI_preset_info_t enum_info;
  const PAPI_preset_info_t *info = NULL;
  const PAPI_hw_info_t *hwinfo = NULL;
  
  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */
  for(i=0;i<argc;i++)
    if (argv[i])
      {
	if (strstr(argv[i],"-a"))
	  print_avail_only = 1;
	if (strstr(argv[i],"-e"))
	  use_enum = 1;
      }

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT)  test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
	test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

  if ( !TESTS_QUIET ) {
	printf("Test case 8: Available events and hardware information.\n");
	printf("-------------------------------------------------------------------------\n");
	printf("Vendor string and code   : %s (%d)\n",hwinfo->vendor_string,hwinfo->vendor);
	printf("Model string and code    : %s (%d)\n",hwinfo->model_string,hwinfo->model);
	printf("CPU Revision             : %f\n",hwinfo->revision);
	printf("CPU Megahertz            : %f\n",hwinfo->mhz);
	printf("CPU's in this Node       : %d\n",hwinfo->ncpu);
	printf("Nodes in this System     : %d\n",hwinfo->nnodes);
	printf("Total CPU's              : %d\n",hwinfo->totalcpus);
	printf("Number Hardware Counters : %d\n",PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL));
	printf("Max Multiplex Counters   : %d\n",PAPI_MPX_DEF_DEG);
	printf("-------------------------------------------------------------------------\n");

    if (use_enum && print_avail_only) {
      printf("Name\t\tDerived\tDescription (Mgr. Note)\n");
      i = PRESET_MASK;
      do {
	if (PAPI_query_event_verbose(i, &enum_info) == PAPI_OK) {
 	   printf("%s\t%s\t%s (%s)\n",
	       enum_info.event_name,
	       (enum_info.flags & PAPI_DERIVED ? "Yes" : "No"),
	       enum_info.event_descr,
	       (enum_info.event_note ? enum_info.event_note : ""));
	} else test_fail(__FILE__, __LINE__, "PAPI_query_event_verbose", 1);
      } while (PAPI_enum_event(&i, print_avail_only) == PAPI_OK);
      printf("-------------------------------------------------------------------------\n");
    } else {
      if ((info = PAPI_query_all_events_verbose()) == NULL)
	    test_fail(__FILE__, __LINE__, "PAPI_query_all_events_verbose", 1);

      if (print_avail_only == 0)
	    {
	  printf("Name\t\tCode\t\tAvail\tDeriv\tDescription (Note)\n");

	  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
	    if (info[i].event_name)
	      {
		printf("%s\t0x%x\t%s\t%s\t%s (%s)\n",
		       info[i].event_name,
		       info[i].event_code,
		       (info[i].avail ? "Yes" : "No"),
		       (info[i].flags & PAPI_DERIVED ? "Yes" : "No"),
		       info[i].event_descr,
		       (info[i].event_note ? info[i].event_note : ""));
	      }
	  printf("-------------------------------------------------------------------------\n");
	    }
	  else
	    {
	  printf("Name\t\tDerived\tDescription (Mgr. Note)\n");

	  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
	    if ((info[i].event_name) && (info[i].avail))
	      {
		printf("%s\t%s\t%s (%s)\n",
		       info[i].event_name,
		       (info[i].flags & PAPI_DERIVED ? "Yes" : "No"),
		       info[i].event_descr,
		       (info[i].event_note ? info[i].event_note : ""));
	      }
	  exit(0);
	    }
      }
    }
  test_pass(__FILE__, NULL, 0);
  exit(1);
}
