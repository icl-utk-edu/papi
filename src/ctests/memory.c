#define ITERS 100

/* This file performs the following test: start, stop and 
   timer functionality for L1 related events

   - They are counted in the default counting domain and default
     granularity, depending on the platform. Usually this is 
     the user domain (PAPI_DOM_USER) and thread context (PAPI_GRN_THR).

   - Start counters
   - Do iterations
   - Stop and read counters
*/

#include "papi_test.h"

#define NUMTESTS 41
#define TEST_NAME "memory"
#define PRERRORANDEXIT(code,exitval) {\
  fprintf(stderr,"Fatal error: %s at %d: PAPI call failed\nError:\t%s\n",\
          __FILE__,__LINE__,PAPI_strerror(code));\
  exit(exitval); }
int main() 
{
  int retval, num_tests = NUMTESTS, i;
  int EventSet;
  long_long **values;
  const PAPI_hw_info_t *hwinfo = NULL;

  char descr[PAPI_MAX_STR_LEN];

  const int eventlist[NUMTESTS]={
    PAPI_CSR_TOT ,
    PAPI_MEM_SCY ,
    PAPI_MEM_RCY ,
    PAPI_MEM_WCY ,
    PAPI_LD_INS  ,
    PAPI_SR_INS  ,
    PAPI_LST_INS ,
    PAPI_L1_DCM  ,
    PAPI_L1_ICM  ,
    PAPI_L1_TCM  ,
    PAPI_L1_LDM  ,
    PAPI_L1_STM  ,
    PAPI_L1_DCH  ,
    PAPI_L1_DCA  ,
    PAPI_L1_DCR  ,
    PAPI_L1_DCW  ,
    PAPI_L1_ICH  ,
    PAPI_L1_ICA  ,
    PAPI_L1_ICR  ,
    PAPI_L1_ICW  ,
    PAPI_L1_TCH  ,
    PAPI_L1_TCA  ,
    PAPI_L1_TCR  ,
    PAPI_L1_TCW  ,
    PAPI_L2_DCM  ,
    PAPI_L2_ICM  ,
    PAPI_L2_TCM  ,
    PAPI_L2_LDM  ,
    PAPI_L2_STM  ,
    PAPI_L2_DCH  ,
    PAPI_L2_DCA  ,
    PAPI_L2_DCR  ,
    PAPI_L2_DCW  ,
    PAPI_L2_ICH  ,
    PAPI_L2_ICA  ,
    PAPI_L2_ICR  ,
    PAPI_L2_ICW  ,
    PAPI_L2_TCH  ,
    PAPI_L2_TCA  ,
    PAPI_L2_TCR  ,
    PAPI_L2_TCW
};

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    test_fail(__FILE__,__LINE__,"library init",retval);

  values = allocate_test_space(num_tests, 1);
  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
    exit(1);

  printf("Available hardware information.\n");
  printf("-------------------------------------------------------------\n");
  printf("Vendor string and code   : %s (%d)\n",
	 hwinfo->vendor_string,hwinfo->vendor);
  printf("Model string and code    : %s (%d)\n",
	 hwinfo->model_string,hwinfo->model);
  printf("CPU revision             : %f\n",hwinfo->revision);
  printf("CPU Megahertz            : %f\n",hwinfo->mhz);
  printf("CPU's in an SMP node     : %d\n",hwinfo->ncpu);
  printf("Nodes in the system      : %d\n",hwinfo->nnodes);
  printf("Total CPU's in the system: %d\n",hwinfo->totalcpus);
  printf("-------------------------------------------------------------\n");


  retval=PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"Create event set",retval);

  for(i=0;i<num_tests;i++) {
    values[i][0]=-1ll;

    PAPI_event_code_to_name(eventlist[i],descr);
    if(PAPI_add_event(&EventSet, eventlist[i]) != PAPI_OK)
      continue;  /* All events may not be available */

    /* Warm me up */
    do_l1misses(ITERS);

    retval = PAPI_start(EventSet);
    if (retval != PAPI_OK)
      test_fail(__FILE__,__LINE__,"Start counter",retval);

    do_l1misses(ITERS);

    retval = PAPI_stop(EventSet, values[i]);
    if (retval != PAPI_OK)
      test_fail(__FILE__,__LINE__,"Stop counter",retval);

    printf("%3d: Test 0x%08x %s %12lld\n",i,eventlist[i],descr,values[i][0]);
    retval = PAPI_rem_event(&EventSet, eventlist[i]);
    if (retval != PAPI_OK)
      test_fail(__FILE__,__LINE__,"Remove event",retval);
  }

  retval=PAPI_destroy_eventset(&EventSet);
  if (retval != PAPI_OK)
    test_fail(__FILE__,__LINE__,"Destroy event set",retval);


  test_pass(TEST_NAME,values,num_tests);
  exit(0);
}
