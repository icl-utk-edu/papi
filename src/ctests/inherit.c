#include <stdio.h>
#include <unistd.h>
#if defined(_AIX)
#include <sys/wait.h>           /* ARGH! */
#else
#include <wait.h>
#endif
#include "papi_test.h"

int err_exit(int code, char *str)
{
   char out[PAPI_MAX_STR_LEN];
   PAPI_perror(code, out, PAPI_MAX_STR_LEN);
   printf("Error in %s: %s\n", str, out);
   exit(1);
}

int main()
{
   int retval, pid, status, EventSet=PAPI_NULL;
   long long int values[2];
   PAPI_option_t opt;

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      err_exit(retval, "PAPI_library_init(PAPI_VER_CURRENT)");

   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      err_exit(retval, "PAPI_create_eventset(&EventSet)");

   if ((retval = PAPI_query_event(PAPI_TOT_CYC)) != PAPI_OK)
      err_exit(retval, "PAPI_query_event(PAPI_TOT_CYC)");

   if ((retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
      err_exit(retval, "PAPI_add_event(EventSet, PAPI_TOT_CYC)");

   if ((retval = PAPI_query_event(PAPI_FP_INS)) != PAPI_OK)
      err_exit(retval, "PAPI_query_event(PAPI_FP_INS)");

   if ((retval = PAPI_add_event(EventSet, PAPI_FP_INS)) != PAPI_OK)
      err_exit(retval, "PAPI_add_event(EventSet, PAPI_FP_INS)");

   memset(&opt, 0x0, sizeof(PAPI_option_t));
   opt.inherit.inherit = PAPI_INHERIT_ALL;
   if ((retval = PAPI_set_opt(PAPI_INHERIT, &opt)) != PAPI_OK)
      err_exit(retval, "PAPI_set_opt(PAPI_INHERIT, &opt)");

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      err_exit(retval, "PAPI_start(EventSet)");

   pid = fork();
   if (pid == 0) {
      do_flops(1000000);
      exit(0);
   }
   waitpid(pid, &status, 0);

   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      err_exit(retval, "PAPI_stop(EventSet, values)");

   printf("Test case inherit: parent starts, child works, parent stops.\n");
   printf("------------------------------------------------------------\n");

   printf("Test run    : \t1\n");
   printf("PAPI_FP_INS : \t%lld\n", values[1]);
   printf("PAPI_TOT_CYC: \t%lld\n", values[0]);
   printf("------------------------------------------------------------\n");

   printf("Verification:\n");
   printf("Row 1 approximately equals %d\n", 1000000);

   exit(1);
}
