#include "papi_test.h"

int main(int argc, char **argv)
{
   int retval;
   long_long elapsed_us, elapsed_cyc;
   const PAPI_hw_info_t *hw_info;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET) {
      retval = PAPI_set_debug(PAPI_VERB_ECONT);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   }

   hw_info = PAPI_get_hardware_info();
   if (hw_info == NULL)
     test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 2);

   elapsed_us = PAPI_get_real_usec();

   elapsed_cyc = PAPI_get_real_cyc();

   printf("Testing real time clock: %f MHz.\n",hw_info->mhz);
   printf("Sleeping for 10 seconds.\n");

   sleep(10);

   elapsed_us = PAPI_get_real_usec() - elapsed_us;

   elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

   printf("%lld us. %lld cyc.\n",elapsed_us,elapsed_cyc);
   printf("%f Computed MHz.\n",(float)elapsed_cyc/(float)elapsed_us);

   if ((elapsed_us / (long_long)1000000) < 10)
     test_fail(__FILE__, __LINE__, "Real time less than 10 seconds!", PAPI_EMISC);

   /* We'll accept 1 part per thousand error here (to allow Pentium 4 to pass) */
   if ((10.0 * hw_info->mhz * 1000000.0) > (((float)elapsed_cyc) + ((float)elapsed_cyc)/1000))
     test_fail(__FILE__, __LINE__, "Real cycles less than 10*MHz*1000000.0!", PAPI_EMISC);
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
