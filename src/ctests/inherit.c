#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#if defined(_AIX)
#include <sys/wait.h> /* ARGH! */
#else
#include <wait.h>
#endif
#include "papi.h"
#include "test_utils.h"

int main()
{
   int pid, status, EventSet;
   long long int values[2];
   PAPI_option_t opt;

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
     exit(1);

   if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
     exit(1);

   if (PAPI_query_event(PAPI_TOT_CYC) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&EventSet, PAPI_TOT_CYC) != PAPI_OK)
     exit(1);

   if (PAPI_query_event(PAPI_FP_INS) != PAPI_OK)
     exit(1);

   if (PAPI_add_event(&EventSet, PAPI_FP_INS) != PAPI_OK)
     exit(1);

   memset(&opt,0x0,sizeof(PAPI_option_t));
   opt.inherit.inherit = PAPI_INHERIT_ALL;
   if (PAPI_set_opt(PAPI_SET_INHERIT, &opt) != PAPI_OK)
     exit(1);

   if (PAPI_start(EventSet) != PAPI_OK)
     exit(1);

   pid = fork();
   if (pid == 0)
     {
       do_flops(1000000);
       exit(0);
     }
   waitpid(pid,&status,0);

   if (PAPI_stop(EventSet, values) != PAPI_OK)
     exit(1);
  
  printf("Test case inherit: parent starts, child works, parent stops.\n");
  printf("------------------------------------------------------------\n");

  printf("Test run    : \t1\n");
  printf("PAPI_FP_INS : \t%lld\n", values[1]);
  printf("PAPI_TOT_CYC: \t%lld\n", values[0]);
  printf("------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Row 1 approximately equals %d\n",1000000);
  
  exit(0);
}
