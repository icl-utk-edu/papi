#include <stdio.h>
#include <unistd.h>
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

   if (PAPI_create_eventset(&FPEventSet) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&FPEventSet, PAPI_FP_INS) != PAPI_OK)
     exit(1);

   if (PAPI_start(FPEventSet) != PAPI_OK)
     exit(1);

   if (PAPI_cleanup_eventset(&FPEventSet) != PAPI_EISRUN)
     exit(1);

   if (PAPI_destroy_eventset(&FPEventSet) != PAPI_EISRUN)
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
  printf("These error messages:\n");
  printf("PAPI Error Code -10: PAPI_EISRUN: EventSet is currently counting\n");
  printf("PAPI Error Code -10: PAPI_EISRUN: EventSet is currently counting\n");
  printf("PAPI Error Code -1: PAPI_EINVAL: Invalid argument\n");
  exit(0);
}
