/* 
* File:    zero_fork.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email address>
*/

/* This file performs the following test: start, stop and timer
functionality for a parent and a forked child. */

#include "papi_test.h"
#include <sys/wait.h>

int EventSet1;
int PAPI_event, mask1;
int num_events1 = 2;
long_long elapsed_us, elapsed_cyc;
long_long **values;
char event_name[PAPI_MAX_STR_LEN];
int retval, num_tests = 1;

void process_init(void)
{
   printf("Process %d \n", (int) getpid());
   
   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

   if (!TESTS_QUIET) {
      retval = PAPI_set_debug(PAPI_VERB_ECONT);
      if (retval != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   }

   /* query and set up the right instruction to monitor */
   if (PAPI_query_event(PAPI_FP_INS) == PAPI_OK) {
      PAPI_event = PAPI_FP_INS;
      mask1 = MASK_FP_INS | MASK_TOT_CYC;
   } else {
      PAPI_event = PAPI_TOT_INS;
      mask1 = MASK_TOT_INS | MASK_TOT_CYC;
   }

   retval = PAPI_event_code_to_name(PAPI_event, event_name);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", retval);

   EventSet1 = add_test_events(&num_events1, &mask1);

   /* num_events1 is greater than num_events2 so don't worry. */

   values = allocate_test_space(num_tests, num_events1);

   elapsed_us = PAPI_get_real_usec();

   elapsed_cyc = PAPI_get_real_cyc();

   retval = PAPI_start(EventSet1);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
}

void process_fini(void)
{
   retval = PAPI_stop(EventSet1, values[0]);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   elapsed_us = PAPI_get_real_usec() - elapsed_us;

   elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

   remove_test_events(&EventSet1, mask1);

      printf("Process %d %-12s : \t%lld\n", (int)getpid(), event_name,
             (values[0])[0]);
      printf("Process %d PAPI_TOT_CYC : \t%lld\n", (int)getpid(), (values[0])[1]);
      printf("Process %d Real usec    : \t%lld\n", (int)getpid(), elapsed_us);
      printf("Process %d Real cycles  : \t%lld\n", (int)getpid(), elapsed_cyc);

   free_test_space(values, num_tests);

}

int main(int argc, char **argv)
{
   int flops1;
   int retval;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   printf("This tests if PAPI_library_init(),2*fork(),PAPI_library_init() works.\n");
   /* Initialize PAPI for this process */
   process_init();
   flops1 = 1000000;
   if (fork() == 0)
     {
       /* Initialize PAPI for the child process */
       process_init();
       /* Let the child process do work */
       do_flops(flops1);
       /* Measure the child process */
       process_fini();
       exit(0);
     }
   flops1 = 2000000;
   if (fork() == 0)
     {
       /* Initialize PAPI for the child process */
       process_init();
       /* Let the child process do work */
       do_flops(flops1);
       /* Measure the child process */
       process_fini();
       exit(0);
     }
   /* Let this process do work */
   flops1 = 4000000;
   do_flops(flops1);

   /* Wait for child to finish */
   wait(&retval);
   /* Wait for child to finish */
   wait(&retval);

   /* Measure this process */
   process_fini();

   test_pass(__FILE__, NULL, 0);
   exit(1);
}
