#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include "papi.h"

#define NUM_THREADS 4

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

void *CallMatMul(void *threadid)
{
   long tid;
   tid = (long)threadid;
   char region_name[10];

   sprintf(region_name, "matmul_%ld", tid);
   PAPI_hl_region_begin(region_name);
   printf("Sum matmul thread %ld: 0x%x\n", tid, matmult());
   PAPI_hl_region_end(region_name);

   pthread_exit(NULL);
}


int main()
{
   pthread_t threads[NUM_THREADS];
   int rc;
   long t;

   for( t = 0; t < NUM_THREADS; t++) {
      rc = pthread_create(&threads[t], NULL, CallMatMul, (void *)t);
      if (rc) {
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
      }
   }

   pthread_exit(NULL);

   return 0;
}