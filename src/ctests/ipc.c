/*
 * A simple example for the use of PAPI, using PAPI_ipc
 * -Kevin London
 */

#include "papi_test.h"

#define INDEX 100

#ifdef _WIN32
char format_string[] = { "Real_time: %f Proc_time: %f Total ins: %I64d IPC: %f\n" };
#else
char format_string[] = { "Real_time: %f Proc_time: %f Total ins: %lld IPC: %f\n" };
#endif
extern int TESTS_QUIET;         /* Declared in test_utils.c */


int main(int argc, char **argv)
{
   extern void dummy(void *);
   float matrixa[INDEX][INDEX], matrixb[INDEX][INDEX], mresult[INDEX][INDEX];
   float real_time, proc_time, ipc;
   long_long ins;
   int retval;
   int i, j, k;

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */


   /* Initialize the Matrix arrays */
   for (i = 0; i < INDEX * INDEX; i++) {
      mresult[0][i] = 0.0;
      matrixa[0][i] = matrixb[0][i] = rand() * (float) 1.1;
   }

   /* Setup PAPI library and begin collecting data from the counters */
   if ((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_ipc", retval);

   /* Matrix-Matrix multiply */
   for (i = 0; i < INDEX; i++)
      for (j = 0; j < INDEX; j++)
         for (k = 0; k < INDEX; k++)
            mresult[i][j] = mresult[i][j] + matrixa[i][k] * matrixb[k][j];

   /* Collect the data into the variables passed in */
   if ((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_ipc", retval);
   dummy((void *) mresult);

   if (!TESTS_QUIET)
      printf(format_string, real_time, proc_time, ins, ipc);
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
