/* This file performs the following test: event_code_to_name */

#include "papi_test.h"

static void test_continue(char *file, int line, char *call, int retval)
{
      char errstring[PAPI_MAX_STR_LEN];
      PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN);
      printf("Expected error in %s: %s\n", call, errstring);
}

int main(int argc, char **argv)
{
  int retval;
  int code = PAPI_TOT_CYC;
  char event_name[PAPI_MAX_STR_LEN];
   const PAPI_hw_info_t *hwinfo = NULL;
  const PAPI_component_info_t *cmp_info;

  tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);
  
   if ((hwinfo = PAPI_get_hardware_info()) == NULL)
      test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

  printf("Test case code2name.c: Check limits and indexing of event tables.\n");
  printf ("-------------------------------------------------------------------------\n");
  printf("Vendor string and code   : %s (%d)\n", hwinfo->vendor_string,
         hwinfo->vendor);
  printf("Model string and code    : %s (%d)\n", hwinfo->model_string, hwinfo->model);
  printf("CPU Revision             : %f\n", hwinfo->revision);
  printf("CPU Megahertz            : %f\n", hwinfo->mhz);
  printf("CPU's in this Node       : %d\n", hwinfo->ncpu);
  printf("Nodes in this System     : %d\n", hwinfo->nnodes);
  printf("Total CPU's              : %d\n", hwinfo->totalcpus);
  printf("Number Hardware Counters : %d\n", PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
  printf("Max Multiplex Counters   : %d\n", PAPI_get_opt(PAPI_MAX_MPX_CTRS, NULL));
  printf ("-------------------------------------------------------------------------\n\n");

  printf("Looking for PAPI_TOT_CYC...\n");
  retval = PAPI_event_code_to_name(code, event_name);
  if (retval != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  printf("Found |%s|\n", event_name);
  
  code = PAPI_FP_OPS;
  printf("Looking for highest defined preset event (PAPI_FP_OPS): 0x%x...\n", code);
  retval = PAPI_event_code_to_name(code, event_name);
  if (retval != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  printf("Found |%s|\n", event_name);
  
  code = PAPI_PRESET_MASK | (PAPI_MAX_PRESET_EVENTS - 1);
  printf("Looking for highest allocated preset event: 0x%x...\n", code);
  retval = PAPI_event_code_to_name(code, event_name);
  if (retval != PAPI_OK)
    test_continue(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  else printf("Found |%s|\n", event_name);
  
  code = PAPI_PRESET_MASK | PAPI_NATIVE_AND_MASK;
  printf("Looking for highest possible preset event: 0x%x...\n", code);
  retval = PAPI_event_code_to_name(code, event_name);
  if (retval != PAPI_OK)
    test_continue(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  else printf("Found |%s|\n", event_name);

  /* Find the first defined native event */
  code = PAPI_NATIVE_MASK;
  printf("Looking for first native event: 0x%x...\n", code);
  retval = PAPI_event_code_to_name(code, event_name);
  if (retval != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  printf("Found |%s|\n", event_name);

  /* Find the last defined native event from the cpu component*/
  cmp_info = PAPI_get_component_info(0);
  if (cmp_info == NULL)
    test_fail(__FILE__, __LINE__, "PAPI_get_component_info", PAPI_ESBSTR);
#ifdef PENTIUM4
  /* this is a hack to accomodate the fact that P4 events are encoded
    in bits 16 - 23. Other x86 platforms have the events in the bottom byte.
    Non- x86 platforms may behave differently and fail this test...
  */
  code = ((cmp_info->num_native_events - 2)<<16) | PAPI_NATIVE_MASK;
#else
  code = (cmp_info->num_native_events - 1) | PAPI_NATIVE_MASK;
#endif
  printf("Looking for last native event: 0x%x...\n", code);
  retval = PAPI_event_code_to_name(code, event_name);
  if (retval != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  printf("Found |%s|\n", event_name);

  /* Highly doubtful we have this many natives */
  code = PAPI_NATIVE_MASK | (PAPI_NATIVE_MASK-1);
  printf("Looking for highest definable native event: 0x%x...\n", code);
  retval = PAPI_event_code_to_name(code, event_name);
  if (retval != PAPI_OK)
    test_continue(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);
  else printf("Found |%s|\n", event_name);
  if ((retval == PAPI_ENOEVNT) || (retval == PAPI_OK))
   test_pass(__FILE__, 0, 0);
   
   test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", PAPI_EBUG);

   exit(1);
}

