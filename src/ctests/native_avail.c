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
  PAPI_event_info_t info;
  const PAPI_hw_info_t *hwinfo = NULL;
#ifdef _POWER4
  int group=0;
#endif
  
  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */
  for(i=0;i<argc;i++)

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
  do {
#ifdef _POWER4
    group=(i&0x00FF0000)>>16;
    if(group)
    	printf("%10d", group-1);
    else{
    printf("\n\n");
#endif
    j++;
    retval = PAPI_get_event_info(i, &info);
#ifndef _POWER4
    if ( !TESTS_QUIET && retval == PAPI_OK) {
		printf("%-30s 0x%-10x\n%s\n", \
	       info.symbol, info.event_code, info.long_descr);
    }
#else
    if ( !TESTS_QUIET && retval == PAPI_OK) {
		printf("%-30s 0x%-10x\n%s", \
	       info.symbol, info.event_code, info.long_descr);
    }
    printf("Groups: ");
    }
#endif
#ifdef PENTIUM4
    k = i;
    if (PAPI_enum_event(&k, PAPI_P4_ENUM_BITS) == PAPI_OK) {
      l = strlen(info.long_descr);
      do {
	j++;
	retval = PAPI_get_event_info(k, &info);
	if ( !TESTS_QUIET && retval == PAPI_OK) {
		    printf("    %-26s 0x%-10x    %s\n", \
		   info.symbol, info.event_code, info.long_descr + l);
	}
      } while (PAPI_enum_event(&k, PAPI_P4_ENUM_BITS) == PAPI_OK);
    }
    if ( !TESTS_QUIET && retval == PAPI_OK) printf("\n");
  } while (PAPI_enum_event(&i, PAPI_P4_ENUM_GROUPS) == PAPI_OK);
#elif defined(_POWER4)
/* this function would return the next native event code.
    modifer=0    it simply returns next native event code
    modifer=1    it would return information of groups this native event lives
                 0x400000ed is the native code of PM_FXLS_FULL_CYC,
		 before it returns 0x400000ee which is the next native event's
		 code, it would return *EventCode=0x400400ed, the digits 16-23
		 indicate group number
   function return value:
     PAPI_OK successful, next event is valid
     PAPI_ENOEVNT  fail, next event is invalid
*/
  } while (PAPI_enum_event(&i, 1) == PAPI_OK);
#else
  } while (PAPI_enum_event(&i, 0) == PAPI_OK);
#endif

  if ( !TESTS_QUIET )
    printf("-------------------------------------------------------------------------\n");
    printf("Total events reported: %d\n",j);
  test_pass(__FILE__, NULL, 0);
  exit(1);
}
