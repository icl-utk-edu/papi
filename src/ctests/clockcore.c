#include "papi_test.h"

#define ITERS 100000

extern int TESTS_QUIET;         /* Declared in test_utils.c */

void clockcore(void)
{
   long_long *elapsed_usec, *elapsed_cyc,
       total_usec = 0, uniq_usec = 0, diff_usec = 0,
       total_cyc = 0, uniq_cyc = 0, diff_cyc = 0;
   int i;

   elapsed_usec = (long_long *) malloc(ITERS * sizeof(long_long));
   elapsed_cyc = (long_long *) malloc(ITERS * sizeof(long_long));

   /* Real */

   for (i = 0; i < ITERS; i++)
      elapsed_cyc[i] = (long_long) PAPI_get_real_cyc();

   for (i = 1; i < ITERS; i++) {
      if (elapsed_cyc[i] - elapsed_cyc[i - 1] < 0){
	 fprintf(stderr,"Negative elapsed time, bailing\n");
         abort();
      }
      diff_cyc = elapsed_cyc[i] - elapsed_cyc[i - 1];
      if (diff_cyc != 0)
         uniq_cyc++;
      total_cyc += diff_cyc;
   }
   if (!TESTS_QUIET) {
      if (uniq_cyc == ITERS - 1)
         printf("PAPI_get_real_cyc : %7.3f   <%7.3f\n",
                (double) total_cyc / (double) (ITERS),
                (double) total_cyc / (double) uniq_cyc);
      else if (uniq_cyc)
         printf("PAPI_get_real_cyc : %7.3f    %7.3f\n",
                (double) total_cyc / (double) (ITERS),
                (double) total_cyc / (double) uniq_cyc);
      else
         printf("PAPI_get_real_cyc : %7.3f   >%7.3f\n",
                (double) total_cyc / (double) (ITERS), (double) total_cyc);
   }

   for (i = 0; i < ITERS; i++)
      elapsed_usec[i] = (long_long) PAPI_get_real_usec();

   for (i = 1; i < ITERS; i++) {
      if (elapsed_usec[i] - elapsed_usec[i - 1] < 0){
	 fprintf(stderr,"Negative elapsed time, bailing\n");
         abort();
      }
      diff_usec = elapsed_usec[i] - elapsed_usec[i - 1];
      if (diff_usec != 0)
         uniq_usec++;
      total_usec += diff_usec;
   }
   if (!TESTS_QUIET) {
      if (uniq_usec == ITERS - 1)
         printf("PAPI_get_real_usec: %7.3f   <%7.3f\n",
                (double) total_usec / (double) (ITERS),
                (double) total_usec / (double) uniq_usec);
      else if (uniq_usec)
         printf("PAPI_get_real_usec: %7.3f    %7.3f\n",
                (double) total_usec / (double) (ITERS),
                (double) total_usec / (double) uniq_usec);
      else
         printf("PAPI_get_real_usec: %7.3f   >%7.3f\n",
                (double) total_usec / (double) (ITERS), (double) total_usec);
   }

   /* Virtual */

   total_cyc = 0;
   uniq_cyc = 0;

   if (PAPI_get_virt_cyc() != -1) {
      for (i = 0; i < ITERS; i++)
         elapsed_cyc[i] = PAPI_get_virt_cyc();

      for (i = 1; i < ITERS; i++) {
         if (elapsed_cyc[i] - elapsed_cyc[i - 1] < 0){
	    fprintf(stderr,"Negative elapsed time, bailing.\n");
            abort();
	 }
         diff_cyc = elapsed_cyc[i] - elapsed_cyc[i - 1];
         if (diff_cyc != 0)
            uniq_cyc++;
         total_cyc += diff_cyc;
      }
      if (!TESTS_QUIET) {
         if (uniq_cyc == ITERS - 1)
            printf("PAPI_get_virt_cyc : %7.3f   <%7.3f\n",
                   (double) total_cyc / (double) (ITERS),
                   (double) total_cyc / (double) uniq_cyc);
         else if (uniq_cyc)
            printf("PAPI_get_virt_cyc : %7.3f    %7.3f\n",
                   (double) total_cyc / (double) (ITERS),
                   (double) total_cyc / (double) uniq_cyc);
         else
            printf("PAPI_get_virt_cyc : %7.3f   >%7.3f\n",
                   (double) total_cyc / (double) (ITERS), (double) total_cyc);
      }
   } else
      test_fail(__FILE__, __LINE__, "PAPI_get_virt_cyc", -1);
   total_usec = 0;
   uniq_usec = 0;

   if (PAPI_get_virt_usec() != -1) {
      for (i = 0; i < ITERS; i++)
         elapsed_usec[i] = (long_long) PAPI_get_virt_usec();

      for (i = 1; i < ITERS; i++) {
         if (elapsed_usec[i] - elapsed_usec[i - 1] < 0){
	    fprintf(stderr,"Negative elapsed time, bailing\n");
            abort();
	 }
         diff_usec = elapsed_usec[i] - elapsed_usec[i - 1];
         if (diff_usec != 0)
            uniq_usec++;
         total_usec += diff_usec;
      }
      if (!TESTS_QUIET) {
         if (uniq_usec == ITERS - 1)
            printf("PAPI_get_virt_usec: %7.3f   <%7.3f\n",
                   (double) total_usec / (double) (ITERS),
                   (double) total_usec / (double) uniq_usec);
         else if (uniq_usec)
            printf("PAPI_get_virt_usec: %7.3f    %7.3f\n",
                   (double) total_usec / (double) (ITERS),
                   (double) total_usec / (double) uniq_usec);
         else
            printf("PAPI_get_virt_usec: %7.3f   >%7.3f\n",
                   (double) total_usec / (double) (ITERS), (double) total_usec);
      }
   } else
      test_fail(__FILE__, __LINE__, "PAPI_get_virt_usec", -1);
}
