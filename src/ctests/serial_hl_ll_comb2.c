#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "papi.h"
#include "papi_test.h"
#include "do_loops.h"

struct threeNum
{
   int n1, n2, n3;
};

void do_io()
{
   int n;
   struct threeNum num;
   FILE *fptr;

   if ((fptr = fopen("storage.bin","wb")) == NULL){
      printf("Error! opening file");
      /* File exits if the file pointer returns NULL. */
      exit(1);
   }

   for (n = 1; n < 5; ++n) {
      num.n1 = n;
      num.n2 = 5*n;
      num.n3 = 5*n + 1;
      fwrite(&num, sizeof(struct threeNum), 1, fptr); 
   }

   for(n = 1; n < 5; ++n)
   {
      fread(&num, sizeof(struct threeNum), 1, fptr); 
   }
   fclose(fptr);
}


int main( int argc, char **argv )
{
   int retval;
   int quiet = 0;
   char* region_name;

   /* Set TESTS_QUIET variable */
   quiet = tests_quiet( argc, argv );

   region_name = "do_flops";

   if ( !quiet ) {
      printf("\nTesting high-level and low-level API in parallel: do_flops\n");
   }

   PAPI_hl_init();
   PAPI_hl_set_events("appio:::READ_BYTES, appio:::WRITE_BYTES, coretemp:::hwmon0:temp2_input=instant");

   retval = PAPI_hl_region_begin(region_name);
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_hl_region_begin", retval );
   }

   do_flops( NUM_FLOPS );

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

   do_flops( NUM_FLOPS );
   /* do some IO */
   do_io();

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

   retval = PAPI_hl_region_end(region_name);
   if ( retval != PAPI_OK ) {
      test_fail( __FILE__, __LINE__, "PAPI_hl_region_end", retval );
   }

   PAPI_hl_print_output();
   PAPI_hl_finalize();

   test_pass( __FILE__ );

   return 0;
}