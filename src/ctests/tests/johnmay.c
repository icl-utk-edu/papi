#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "papi.h"
#include "test_utils.h"

int main()
{
   int CycEventSet, FPEventSet;
   long long int values, values2, values3, values4, values5;

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
     exit(1);

   if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
     exit(1);

   if (PAPI_query_event(PAPI_TOT_CYC) != PAPI_OK)
     exit(1);

   if (PAPI_create_eventset(&CycEventSet) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&CycEventSet, PAPI_TOT_CYC) != PAPI_OK)
     exit(1);

   if (PAPI_query_event(PAPI_FP_INS) != PAPI_OK)
     exit(1);

   if (PAPI_create_eventset(&FPEventSet) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&FPEventSet, PAPI_FP_INS) != PAPI_OK)
     exit(1);

   if (PAPI_start(FPEventSet) != PAPI_OK)
     exit(1);
   do_flops(1000000);
   if (PAPI_stop(FPEventSet, &values) != PAPI_OK)
     exit(1);

   if (PAPI_start(FPEventSet) != PAPI_OK)
     exit(1);
   do_flops(1000000);
   if (PAPI_stop(FPEventSet, &values2) != PAPI_OK)
     exit(1);

   if (PAPI_start(FPEventSet) != PAPI_OK)
     exit(1);
   if (PAPI_start(CycEventSet) != PAPI_OK)
     exit(1);
   do_flops(1000000);
   if (PAPI_stop(FPEventSet, &values3) != PAPI_OK)
     exit(1);
  
   if (PAPI_start(FPEventSet) != PAPI_OK)
     exit(1);
   do_flops(1000000);
   if (PAPI_stop(FPEventSet, &values4) != PAPI_OK)
     exit(1);
   if (PAPI_stop(CycEventSet, &values5) != PAPI_OK)
     exit(1);
  
  printf("Test case John May: start & stop of overlapped unshared counters.\n");
  printf("-----------------------------------------------------------------\n");

  printf("Test run    : \t1\t2\t3\t4\n");
  printf("PAPI_FP_INS : \t%lld\t%lld\t%lld\t%lld\n", values, values2, values3, values4);
  printf("PAPI_TOT_CYC: \t\t\t\t%lld\n", values5);
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");

  exit(0);
}
