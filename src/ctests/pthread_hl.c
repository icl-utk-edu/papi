#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include "papi.h"
#include "papi_test.h"

#define NUM_THREADS 4

typedef struct papi_args
{
   long tid;
   int quiet;
} papi_args_t;

int matmult()
{
   int m, n, p, q, c, d, k, sum = 0, cksum = 0;
   int first[10][10], second[10][10], multiply[10][10];

   m = 10;
   n = 10;

   srand(time(NULL));

   for (c = 0; c < m; c++)
   for (d = 0; d < n; d++)
   first[c][d] = rand() % 20;

   p = 10;
   q = 10;

   if (n != p) {
   printf("Matrices with entered orders can't be multiplied with each other.\n"); return -1; }

   for (c = 0; c < p; c++)
      for (d = 0; d < q; d++)
         second[c][d] = rand() % 20;

   for (c = 0; c < m; c++) {
      for (d = 0; d < q; d++) {
         for (k = 0; k < p; k++) {
            sum = sum + first[c][k]*second[k][d];
         }
         multiply[c][d] = sum;
         cksum += multiply[c][d];
         sum = 0;
      }
   }
   return(cksum);
}

void *CallMatMul(void *args)
{
   long tid;
   int retval, quiet;
   char region_name[10];

   papi_args_t* papi_args = (papi_args_t*)args;
   tid = (*papi_args).tid;
   quiet = (*papi_args).quiet;

   sprintf(region_name, "matmul_%ld", tid);

   retval = PAPI_hl_region_begin(region_name);
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_hl_region_begin", retval );
   }
   if ( !quiet ) {
      printf("Sum matmul thread %ld: 0x%x\n", tid, matmult());
   }
   retval = PAPI_hl_region_end(region_name);
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_hl_region_end", retval );
   }

   pthread_exit(NULL);
}

int main( int argc, char **argv )
{
   pthread_t threads[NUM_THREADS];
   papi_args_t args[NUM_THREADS];
   int rc;
   long t;
   int quiet = 0;

   /* Set TESTS_QUIET variable */
   quiet = tests_quiet( argc, argv );

   for( t = 0; t < NUM_THREADS; t++) {
      args[t].tid = t;
      args[t].quiet = quiet;
      rc = pthread_create(&threads[t], NULL, CallMatMul, (void *)&args[t]);
      if (rc) {
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
      }
   }

   for( t = 0; t < NUM_THREADS; t++) {
      pthread_join(threads[t], NULL);
   }

   PAPI_hl_print_output();
   test_pass( __FILE__ );

   return 0;
}