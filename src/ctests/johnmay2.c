#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <wait.h>
#include "papi.h"
#include "test_utils.h"

int main()
{
   int FPEventSet;
   long long int values;

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
     exit(1);

   if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
     exit(1);

   if (PAPI_query_event(PAPI_FP_INS) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&FPEventSet, PAPI_FP_INS) != PAPI_OK)
     exit(1);

   if (PAPI_start(FPEventSet) != PAPI_OK)
     exit(1);

   if (PAPI_cleanup_eventset(&FPEventSet) != PAPI_EINVAL)
     exit(1);

   if (PAPI_destroy_eventset(&FPEventSet) != PAPI_EINVAL)
     exit(1);

   do_flops(1000000);

   if (PAPI_stop(FPEventSet, &values) != PAPI_OK)
     exit(1);

   if (PAPI_destroy_eventset(&FPEventSet) != PAPI_EINVAL)
     exit(1);

   if (PAPI_cleanup_eventset(&FPEventSet) != PAPI_OK)
     exit(1);

   if (PAPI_destroy_eventset(&FPEventSet) != PAPI_OK)
     exit(1);

   if (FPEventSet != PAPI_NULL)
     exit(1);

  printf("Test case John May 2: cleanup / destroy eventset.\n");
  printf("-------------------------------------------------\n");
  printf("Test run    : \t1\n");
  printf("PAPI_FP_INS : \t%lld\n", values);
  printf("-------------------------------------------------\n");

  printf("Verification:\n");
  printf("Row 1 approximately equals %d\n",1000000);
  printf("This error message 3 times: PAPI Error Code -1: Invalid argument\n");

  exit(0);
}
