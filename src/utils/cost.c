#include "papi_test.h"

static int num_iters = NUM_ITERS;

static void print_help(void)
{
   printf("This is the PAPI cost program.\n");
   printf("It computes min / max / mean / std. deviation for PAPI start/stop pairs; for PAPI reads, and for PAPI_accums.  Usage:\n\n");
   printf("    cost [options] [parameters]\n");
   printf("    cost TESTS_QUIET\n\n");
   printf("Options:\n\n");
   printf("  -b BINS       set the number of bins for the graphical distribution of costs. Default: 100\n");
   printf("  -d            show a graphical distribution of costs\n");
   printf("  -h            print this help message\n");
   printf("  -s            show number of iterations above the first 10 std deviations\n");
   printf("  -t THRESHOLD  set the threshold for the number of iterations. Default: 100,000\n");
   printf("\n");
}

/* computes min, max, and mean for an array; returns std deviation */
static double do_stats(long long *array, long long *min, long long *max, double *average) {
   int i;
   double std, tmp;

   *min = *max = array[0]; 
   *average = 0;
   for(i=0; i < num_iters; i++ ) {
     *average += (double)array[i]; 
      if (*min > array[i]) *min = array[i];
      if (*max < array[i]) *max = array[i];
   }
   *average = *average/(double)num_iters; 
   std=0;
   for(i=0; i < num_iters; i++ ) {
     tmp = (double)array[i]-(*average); 
      std += tmp * tmp;
   }
   std = sqrt(std/(num_iters-1));
   return(std);
}

static void do_std_dev(long long *a, int *s, double std, double ave) {
   int i,j;
   double dev[10];

   for(i=0;i<10;i++) {
      dev[i] = std*(i+1);
      s[i] = 0;
   }

   for(i=0; i < num_iters; i++ ) {
      for(j=0;j<10;j++) {
	if (((double)a[i] - dev[j]) > ave) 
          s[j]++;
      }
   }
}

static void do_dist(long long *a, long long min, long long max, int bins, int *d) {
   int i, j;
   int dmax = 0;
   int range = (int)(max - min + 1); /* avoid edge conditions */

   /* clear the distribution array */
   for(i=0;i<bins;i++) {
      d[i] = 0;
   }

   /* scan the array to distribute cost per bin */
   for(i=0; i < num_iters; i++ ) {
      j = ((int)(a[i] - min)*bins)/range;
      d[j]++;
      if (j && (dmax < d[j])) dmax = d[j];
   }

   /* scale each bin to a max of 100 */
   for(i=1;i<bins;i++) {
      d[i] = (d[i] * 100)/dmax;
   }
}

static void print_dist(long long min, long long max, int bins, int *d) {
   int i,j;
   int step = (int)(max - min) / bins;

   printf("\nCost distribution profile\n\n");
   for (i=0;i<bins;i++) {
      printf("%8d:", (int)min + (step*i));
      if (d[i] > 100) {
         printf("**************************** %d counts ****************************",d[i]);
      }else {
         for (j=0;j<d[i];j++) printf("*");
      }
      printf("\n");
   }
}

static void print_stats(int i, long long min, long long max, double average, double std) {
   char *test[] = {"loop latency", "PAPI_start/stop (2 counters)",
	   "PAPI_read (2 counters)", "PAPI_read_ts (2 counters)", "PAPI_accum (2 counters)", "PAPI_reset (2 counters)"};
   printf("\nTotal cost for %s over %d iterations\n", test[i], num_iters);
   printf("min cycles   : %lld\nmax cycles   : %lld\nmean cycles  : %lf\nstd deviation: %lf\n ",
      min, max, average, std);
}

static void print_std_dev(int *s) {
   int i;

   printf("\n");
   printf("              --------# Standard Deviations Above the Mean--------\n");
   printf("0-------1-------2-------3-------4-------5-------6-------7-------8-------9-----10\n");
   for (i=0;i<10;i++) printf("  %d\t", s[i]); 
   printf("\n\n");
}
static void do_output(int test_type, long long *array, int bins, int show_std_dev, int show_dist) {
   int s[10];
   long long min, max;
   double average, std;

   std = do_stats(array, &min, &max, &average);

   print_stats(test_type, min, max, average, std);

   if (show_std_dev) {
      do_std_dev(array, s, std, average);
      print_std_dev(s);
   }

   if (show_dist) {
      int *d;
      d = malloc((size_t)bins*sizeof(int));
      do_dist(array, min, max, bins, d);
      print_dist(min, max, bins, d);
      free(d);
   }
}


int main(int argc, char **argv)
{
   int i, retval, EventSet = PAPI_NULL;
   int bins = 100;
   int show_dist = 0, show_std_dev = 0;
   long long totcyc, values[2];
   long long *array;


   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   for (i = 0; i < argc; i++) {
      if (argv[i]) {
         if (strstr(argv[i], "-b")) {
            bins = atoi(argv[i+1]);
            if (bins) i++;
            else {
               printf ("-b requires a bin count!\n");
               exit(1);
            }
         }
         if (strstr(argv[i], "-d"))
            show_dist = 1;
         if (strstr(argv[i], "-h")) {
            print_help();
            exit(1);
         }
         if (strstr(argv[i], "-s"))
            show_std_dev = 1;
         if (strstr(argv[i], "-t")) {
	   num_iters = (int)atol(argv[i+1]);
            if (num_iters) i++;
            else {
               printf ("-t requires a threshold value!\n");
               exit(1);
            }
         }
      }
   }

   printf("Cost of execution for PAPI start/stop, read and accum.\n");
   printf("This test takes a while. Please be patient...\n");

   if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
      test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);
   if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
   if ((retval = PAPI_query_event(PAPI_TOT_CYC)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_query_event", retval);
   if ((retval = PAPI_query_event(PAPI_TOT_INS)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_query_event", retval);
   if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

   if ((retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

   if ((retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)
      if ((retval = PAPI_add_event(EventSet, PAPI_TOT_IIS)) != PAPI_OK)
         test_fail(__FILE__, __LINE__, "PAPI_add_event", retval);

   /* Make sure no errors and warm up */

   totcyc = PAPI_get_real_cyc();
   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   if ((retval = PAPI_stop(EventSet, NULL)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   array = (long long *)malloc((size_t)num_iters*sizeof(long long));
   if (array == NULL ) 
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   /* Determine clock latency */

   printf("\nPerforming loop latency test...\n");

   for (i = 0; i < num_iters; i++) {
      totcyc = PAPI_get_real_cyc();
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i] = totcyc;
   }

   do_output(0, array, bins, show_std_dev, show_dist);

   /* Start the start/stop eval */

   printf("\nPerforming start/stop test...\n");

   for (i = 0; i < num_iters; i++) {
      totcyc = PAPI_get_real_cyc();
      PAPI_start(EventSet);
      PAPI_stop(EventSet, values);
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i] = totcyc;
   }

   do_output(1, array, bins, show_std_dev, show_dist);

   /* Start the read eval */
   printf("\nPerforming read test...\n");

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   PAPI_read(EventSet, values);

   for (i = 0; i < num_iters; i++) {
      totcyc = PAPI_get_real_cyc();
      PAPI_read(EventSet, values);
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i] = totcyc;
   }
   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   do_output(2, array, bins, show_std_dev, show_dist);

   /* Start the read with timestamp eval */
   printf("\nPerforming read with timestamp test...\n");

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   PAPI_read_ts(EventSet, values, &totcyc);

   for (i = 0; i < num_iters; i++) {
      PAPI_read_ts(EventSet, values, &array[i]);
   }
   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   /* post-process the timing array */
   for (i = num_iters - 1; i > 0 ; i--) {
      array[i] -= array[i-1];
   }
   array[0] -= totcyc;

   do_output(3, array, bins, show_std_dev, show_dist);

   /* Start the accum eval */
   printf("\nPerforming accum test...\n");

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   PAPI_accum(EventSet, values);

   for (i = 0; i < num_iters; i++) {
      totcyc = PAPI_get_real_cyc();
      PAPI_accum(EventSet, values);
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i] = totcyc;
   }
   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   do_output(4, array, bins, show_std_dev, show_dist);

   /* Start the reset eval */
   printf("\nPerforming reset test...\n");

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   for (i = 0; i < num_iters; i++) {
      totcyc = PAPI_get_real_cyc();
      PAPI_reset(EventSet);
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i] = totcyc;
   }
   if ((retval = PAPI_stop(EventSet, values)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   do_output(5, array, bins, show_std_dev, show_dist);

   free(array);
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
