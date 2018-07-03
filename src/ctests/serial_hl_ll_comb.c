#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "papi.h"
#include "papi_test.h"

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


int main( int argc, char **argv )
{
   int retval, i;
   int quiet = 0;

   /* Set TESTS_QUIET variable */
   quiet = tests_quiet( argc, argv );

   /* three iterations with high-level API */
   if ( !quiet ) {
      printf("Testing high-level API...\n");
   }

   for ( i = 1; i < 4; ++i ) {
      char region_name[10];
      sprintf(region_name, "matmul_%d", i);

      retval = PAPI_hl_region_begin(region_name);
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_hl_region_begin", retval );
      }

      if ( !quiet ) {
         printf("Sum matmul round %d: 0x%x\n", i, matmult());
      }

      retval = PAPI_hl_region_end(region_name);
      if ( retval != PAPI_OK ) {
         test_fail( __FILE__, __LINE__, "PAPI_hl_region_end", retval );
      }
   }
   PAPI_hl_print_output();
   PAPI_hl_finalize();

   /* one iteration with low-level API */
   if ( !quiet ) {
      printf("\nTesting low-level API...\n");
   }

   long long values[2];
   int EventSet = PAPI_NULL;
   char event_name1[]="PAPI_TOT_CYC";
   char event_name2[]="PAPI_TOT_INS";

   /* create the eventset */
   retval = PAPI_create_eventset( &EventSet );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_create_eventset", retval );
   }

   retval = PAPI_add_named_event( EventSet, event_name1);
   if ( retval != PAPI_OK ) {
      if (!quiet) printf("Couldn't add %s\n",event_name1);
      test_skip(__FILE__,__LINE__,"Couldn't add PAPI_TOT_CYC",0);
   }

   retval = PAPI_add_named_event( EventSet, event_name2);
   if ( retval != PAPI_OK ) {
      if (!quiet) printf("Couldn't add %s\n",event_name2);
      test_skip(__FILE__,__LINE__,"Couldn't add PAPI_TOT_INS",0);
   }

   /* Start PAPI */
   retval = PAPI_start( EventSet );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_start", retval );
   }

   if ( !quiet ) {
      printf("Sum matmul round 4: 0x%x\n", matmult());
   }

   /* Read results */
   retval = PAPI_stop( EventSet, values );
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
   }

   if ( !quiet ) {
      printf("%s: %lld\n", event_name1, values[0]);
      printf("%s: %lld\n", event_name2, values[1]);
   }

   /* remove results. */
   PAPI_remove_named_event(EventSet,event_name1);
   PAPI_remove_named_event(EventSet,event_name2);

   test_pass( __FILE__ );

   return 0;
}