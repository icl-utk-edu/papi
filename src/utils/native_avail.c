/* This file performs the following test: hardware info and which native events are available */

#include "papi_test.h"
extern int TESTS_QUIET; /* Declared in test_utils.c */

#ifdef PENTIUM4
enum {
  PAPI_P4_ENUM_ALL = 0,	// all 83,000+ native events
  PAPI_P4_ENUM_GROUPS,	// 45 groups + custom + user
  PAPI_P4_ENUM_COMBOS,	// all combinations of mask bits for given group
  PAPI_P4_ENUM_BITS	// all individual bits for given group
};
#endif

int main(int argc, char **argv) 
{
  int i,j,k,l;
  int retval;
  int print_by_name = 0;
  int print_describe = 0;
  char name[PAPI_MAX_STR_LEN] = {0}, descr[1024] = {0};
  PAPI_preset_info_t info;
  const PAPI_hw_info_t *hwinfo = NULL;
  
  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */
  for(i=0;i<argc;i++)
    if (argv[i])
      {
	if (strstr(argv[i],"-n"))
	  print_by_name = 1;
	if (strstr(argv[i],"-d"))
	  print_describe = 1;
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
	printf("Test case NATIVE_AVAIL: Available native events and hardware information.\n");
	printf("-------------------------------------------------------------------------\n");
	printf("Vendor string and code   : %s (%d)\n",hwinfo->vendor_string,hwinfo->vendor);
	printf("Model string and code    : %s (%d)\n",hwinfo->model_string,hwinfo->model);
	printf("CPU Revision             : %f\n",hwinfo->revision);
	printf("CPU Megahertz            : %f\n",hwinfo->mhz);
	printf("CPU's in this Node       : %d\n",hwinfo->ncpu);
	printf("Nodes in this System     : %d\n",hwinfo->nnodes);
	printf("Total CPU's              : %d\n",hwinfo->totalcpus);
	printf("Number Hardware Counters : %d\n",PAPI_get_opt(PAPI_MAX_HWCTRS,NULL));
	printf("Max Multiplex Counters   : %d\n",PAPI_MPX_DEF_DEG);
	printf("-------------------------------------------------------------------------\n");

	printf("Name\t\t\t       Code\t   Description\n");
  }	
  i = 0 | NATIVE_MASK;
  j = 0;
  name[0] = 0;
  descr[0] = 0;
  info.event_name = name;
  info.event_descr = descr;
  info.event_note = NULL;
  do {
    j++;
    if (print_by_name) {
      info.event_code = i;
      retval = PAPI_event_code_to_name(info.event_code, name);
      if (name) {
	descr[0] = 0;
	retval = PAPI_describe_event(name, (int *)&info.event_code, descr);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_describe_event", 1);
      }
    }
    else if (print_describe) {
      info.event_code =  i;
      name[0] = 0;
      retval = PAPI_describe_event(name, (int *)&info.event_code, descr);
    }
    else {
      retval = PAPI_query_event_verbose(i, &info);
    }
    if ( !TESTS_QUIET && retval == PAPI_OK) {
		printf("%-30s 0x%-10x\n%s\n", \
	       info.event_name, info.event_code, info.event_descr);
    }
#ifdef PENTIUM4
    k = i;
    if (PAPI_enum_event(&k, PAPI_P4_ENUM_BITS) == PAPI_OK) {
      l = strlen(info.event_descr);
      do {
	j++;
	if (print_by_name) {
	  info.event_code = k;
	  retval = PAPI_event_code_to_name(info.event_code, name);
	  if (name) {
	    descr[0] = 0;
	    retval = PAPI_describe_event(name, (int *)&info.event_code, descr);
	    if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_describe_event", 1);
	  }
	}
	else if (print_describe) {
	  info.event_code =  k;
	  name[0] = 0;
	  retval = PAPI_describe_event(name, (int *)&info.event_code, descr);
	}
	else {
	  retval = PAPI_query_event_verbose(k, &info);
	}
	if ( !TESTS_QUIET && retval == PAPI_OK) {
		    printf("    %-26s 0x%-10x    %s\n", \
		   info.event_name, info.event_code, info.event_descr + l);
	}
      } while (PAPI_enum_event(&k, PAPI_P4_ENUM_BITS) == PAPI_OK);
    }
    if ( !TESTS_QUIET && retval == PAPI_OK) printf("\n");
  } while (PAPI_enum_event(&i, PAPI_P4_ENUM_GROUPS) == PAPI_OK);
#else
  } while (PAPI_enum_event(&i, 0) == PAPI_OK);
#endif

  if ( !TESTS_QUIET )
    printf("-------------------------------------------------------------------------\n");
    printf("Total events reported: %d\n",j);
  test_pass(__FILE__, NULL, 0);
  exit(1);
}
