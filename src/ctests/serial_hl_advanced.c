#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "papi.h"

int matmult(const char* region_name)
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

   PAPI_hl_read(region_name);

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


int main() 
{
   int i;

   PAPI_hl_init();

   PAPI_hl_set_events("PAPI_TOT_INS, PAPI_TOT_CYC");
   
   for ( i = 1; i < 5; ++i ) {
      char region_name[10];
      sprintf(region_name, "matmul_%d", i);
      PAPI_hl_region_begin(region_name);
      printf("Sum matmul round %d: 0x%x\n", i, matmult(region_name));
      PAPI_hl_region_end(region_name);
   }

   PAPI_hl_print_output();

   PAPI_hl_finalize();
   return 0;
}