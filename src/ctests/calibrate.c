/*
   Calibrate.c
        A program to perform one or all of three tests to count flops.
        Test 1. Inner Product:                          2*n operations
                for i = 1:n; a = a + x(i)*y(i); end
        Test 2. Matrix Vector Product:          2*n^2 operations
                for i = 1:n; for j = 1:n; x(i) = x(i) + a(i,j)*y(j); end; end;
        Test 3. Matrix Matrix Multiply:         2*n^3 operations
                for i = 1:n; for j = 1:n; for k = 1:n; c(i,j) = c(i,j) + a(i,k)*b(k,j); end; end; end;

  Supply a command line argument of 1, 2, or 3 to perform each test, or
  no argument to perform all three.

  Each test initializes PAPI and presents a header with processor information.
  Then it performs 500 iterations, printing result lines containing:
  n, measured counts, theoretical counts, (measured - theory), % error
 */

#include "papi_test.h"

static void resultline(int i, int j, int TESTS_QUIET);
static void headerlines(char *title, int TESTS_QUIET);

#define INDEX1 100
#define INDEX5 500

/* #define DONT_FAIL */

extern int TESTS_QUIET;

int main(int argc, char *argv[])
{
   extern void dummy(void *);
   float real_time, proc_time, mflops;
   long_long flpins;

   float *a, *b, *c, *x, *y, *z;
   float aa = 0.0;
   int i, j, k, n, t;
   int retval = PAPI_OK;

   /*
      Check for inputs of 1, 2, or 3. If TRUE, do that test only.
      Otherwise, do all three tests.
    */
   t = 0;
   if (argc > 1) {
      if (!strcmp(argv[1], "1"))
         t = 1;
      if (!strcmp(argv[1], "2"))
         t = 2;
      if (!strcmp(argv[1], "3"))
         t = 3;
   }

   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   if (!TESTS_QUIET)
      printf("Initializing...");

   /* Initialize PAPI */
   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

/* If this platform doesn't support floating point, skip the test */
   if (PAPI_query_event(PAPI_FP_OPS) != PAPI_OK) {
      test_skip(__FILE__, __LINE__, "PAPI_query_event", PAPI_ENOEVNT);
   }

   /* Initialize memory pointers */
   a = b = c = 0;
   x = y = z = 0;

   /* Inner Product test */
   if (t == 1 || t == 0) {
      /* Allocate the linear arrays */
      x = malloc(INDEX5 * sizeof(float));
      y = malloc(INDEX5 * sizeof(float));

      if (!(x && y))
         retval = PAPI_ENOMEM;
      else {

         headerlines("Inner Product Test", TESTS_QUIET);

         /* step through the different array sizes */
         for (n = 0; n < INDEX5; n++) {
            if (n < INDEX1 || ((n + 1) % 50) == 0) {

               /* Initialize the needed arrays at this size */
               for (i = 0; i <= n; i++) {
                  x[i] = rand() * (float) 1.1;
                  y[i] = rand() * (float) 1.1;
               }

               /* reset PAPI flops count */
               retval = PAPI_stop_counters(NULL, 0);
               if (!(retval == PAPI_OK || retval == PAPI_ENOTRUN))
                  test_fail(__FILE__, __LINE__, "Inner Product Test: PAPI_stop_counters", retval);
               retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
               if (retval != PAPI_OK)
                  test_fail(__FILE__, __LINE__, "Inner Product Test: PAPI_flops", retval);

               /* do the multiplication */
               for (i = 0; i <= n; i++)
                  aa = aa + x[i] * y[i];
               resultline(n, 1, TESTS_QUIET);
               dummy((void *) &aa);
            }
         }
      }
   }

   /* Matrix Vector test */
   if ((t == 2 || t == 0) && retval != PAPI_ENOMEM) {
      /* Allocate the needed arrays */
      a = malloc(INDEX5 * INDEX5 * sizeof(float));
      if (!x)
         x = malloc(INDEX5 * sizeof(float));
      if (!y)
         y = malloc(INDEX5 * sizeof(float));
      if (!(a && x && y))
         retval = PAPI_ENOMEM;
      else {

         headerlines("Matrix Vector Test", TESTS_QUIET);

         /* step through the different array sizes */
         for (n = 0; n < INDEX5; n++) {
            if (n < INDEX1 || ((n + 1) % 50) == 0) {

               /* Initialize the needed arrays at this size */
               for (i = 0; i <= n; i++) {
                  y[i] = 0.0;
                  x[i] = rand() * (float) 1.1;
                  for (j = 0; j <= n; j++)
                     a[i * n + j] = rand() * (float) 1.1;
               }

               /* reset PAPI flops count */
               retval = PAPI_stop_counters(NULL, 0);
               if (!(retval == PAPI_OK || retval == PAPI_ENOTRUN))
                  test_fail(__FILE__, __LINE__, "Matrix Vector Test: PAPI_stop_counters", retval);
               retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
               if (retval != PAPI_OK)
                  test_fail(__FILE__, __LINE__, "Matrix Vector Test: PAPI_flops", retval);

               /* compute the resultant vector */
               for (i = 0; i <= n; i++)
                  for (j = 0; j <= n; j++)
                     y[i] = y[i] + a[i * n + j] * x[i];
               resultline(n, 2, TESTS_QUIET);
               dummy((void *) y);
            }
         }
      }
   }

   /* Matrix Multiply test */
   if ((t == 3 || t == 0) && retval != PAPI_ENOMEM) {
      /* Allocate the needed arrays */
      if (!a)
         a = malloc(INDEX5 * INDEX5 * sizeof(float));
      b = malloc(INDEX5 * INDEX5 * sizeof(float));
      c = malloc(INDEX5 * INDEX5 * sizeof(float));
      if (!(a && b && c))
         retval = PAPI_ENOMEM;
      else {

         headerlines("Matrix Multiply Test", TESTS_QUIET);

         /* step through the different array sizes */
         for (n = 0; n < INDEX5; n++) {
            if (n < INDEX1 || ((n + 1) % 50) == 0) {

               /* Initialize the needed arrays at this size */
               for (i = 0; i <= n * n + n; i++) {
                  c[i] = 0.0;
                  a[i] = rand() * (float) 1.1;
                  b[i] = rand() * (float) 1.1;
               }

               /* reset PAPI flops count */
               retval = PAPI_stop_counters(NULL, 0);
               if (!(retval == PAPI_OK || retval == PAPI_ENOTRUN))
                  test_fail(__FILE__, __LINE__, "Matrix Multiply Test: PAPI_stop_counters", retval);
               retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
               if (retval != PAPI_OK)
                  test_fail(__FILE__, __LINE__, "Matrix Multiply Test: PAPI_flops",
                            retval);

               /* compute the resultant matrix */
               for (i = 0; i <= n; i++)
                  for (j = 0; j <= n; j++)
                     for (k = 0; k <= n; k++)
                        c[i * n + j] = c[i * n + j] + a[i * n + k] * b[k * n + j];
               resultline(n, 3, TESTS_QUIET);
               dummy((void *) c);
            }
         }
      }
   }

   /* Use results so they don't get optimized away */
   dummy((void *) &aa);
   dummy((void *) c);
   dummy((void *) y);

   /* free allocated memory */
   if (a)
      free(a);
   if (b)
      free(b);
   if (c)
      free(c);
   if (x)
      free(x);
   if (y)
      free(y);

   /* exit with status code */
   if (retval == PAPI_ENOMEM)
      test_fail(__FILE__, __LINE__, "malloc", retval);
   else
      test_pass(__FILE__, NULL, 0);
   exit(1);
}

/*
        Extract and display hardware information for this processor.
        (Re)Initialize PAPI_flops() and begin counting floating ops.
*/
static void headerlines(char *title, int TESTS_QUIET)
{
   const PAPI_hw_info_t *hwinfo = NULL;

   if (!TESTS_QUIET) {
      if ((hwinfo = PAPI_get_hardware_info()) == NULL)
         test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 1);

      printf
          ("\n-------------------------------------------------------------------------\n");
      printf("Vendor string and code   : %s (%d)\n", hwinfo->vendor_string,
             hwinfo->vendor);
      printf("Model string and code    : %s (%d)\n", hwinfo->model_string, hwinfo->model);
      printf("CPU revision             : %f\n", hwinfo->revision);
      printf("CPU Megahertz            : %f\n", hwinfo->mhz);
      printf("CPU's in an SMP node     : %d\n", hwinfo->ncpu);
      printf("Nodes in the system      : %d\n", hwinfo->nnodes);
      printf("Total CPU's in the system: %d\n", hwinfo->totalcpus);
      printf
          ("-------------------------------------------------------------------------\n");
      printf("\n%s:\n%8s %12s %12s %8s %8s\n", title, "i", "papi", "theory", "diff",
             "%error");
      printf
          ("-------------------------------------------------------------------------\n");
   }
}

/*
  Read PAPI_flops.
  Format and display results.
  Compute error without using floating ops.
*/
#if defined(mips)
#define SLOPE 9
#define FMA 1
#elif (defined(sparc) && defined(sun))
#define SLOPE 0
#define FMA 1
#else
#define SLOPE 0
#define FMA 0
#endif

static void resultline(int i, int j, int TESTS_QUIET)
{
   float real_time, proc_time, mflops, ferror = 0;
   long_long flpins = 0;
   int papi, theory, diff = 0;
   int retval;

   retval = PAPI_flops(&real_time, &proc_time, &flpins, &mflops);
   if (retval != PAPI_OK)
      test_fail(__FILE__, __LINE__, "resultline: PAPI_flops", retval);

   i++;                         /* convert to 1s base  */
   theory = 2;
   while (j--)
      theory *= i;              /* theoretical ops   */
   papi = (int) (flpins) << FMA;

   diff = papi - theory;

   ferror = ((float) abs(diff)) / ((float) theory) * 100;

   if (!TESTS_QUIET)
      printf("%8d %12d %12d %8d %10.4f\n", i, papi, theory, diff, ferror);

#ifndef DONT_FAIL
   if (ferror > 10 && diff > 8)
#if defined(__ALPHA) && defined(__osf__)
      if (!TESTS_QUIET)
         fprintf(stderr, "Calibrate: error exceeds 10 percent\n");
#else
      test_fail(__FILE__, __LINE__, "Calibrate: error exceeds 10%", PAPI_EMISC);
#endif
#endif
}
