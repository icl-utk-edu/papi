#include "papi_test.h"

static int num_iters = NUM_ITERS;

static void print_help(void)
{
   printf("This is the PAPI cost program.\n");
   printf("It computes min / max / mean / std. deviation for PAPI start/stop pairs and for PAPI reads.  Usage:\n\n");
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
static double do_stats(long_long *array, long_long *min, long_long *max, double *average) {
   int i;
   double std, tmp;

   *min = *max = array[0]; 
   *average = 0;
   for(i=0; i < num_iters; i++ ) {
      *average += array[i]; 
      if (*min > array[i]) *min = array[i];
      if (*max < array[i]) *max = array[i];
   }
   *average = (long) (*average/num_iters); 
   std=0;
   for(i=0; i < num_iters; i++ ) {
      tmp = array[i]-(*average); 
      std += tmp * tmp;
   }
   std = sqrt(std/(num_iters-1));
   return(std);
}

static void do_std_dev(long_long *a, int *s, double std, double ave) {
   int i,j;
   double dev[10];

   for(i=0;i<10;i++) {
      dev[i] = std*(i+1);
      s[i] = 0;
   }

   for(i=0; i < num_iters; i++ ) {
      for(j=0;j<10;j++) {
         if ((a[i] - dev[j]) > ave) s[j]++;
      }
   }
}

static void do_dist(long_long *a, long_long min, long_long max, int bins, int *d) {
   int i, j;
   int dmax = 0;
   int range = max - min;

   /* clear the distribution array */
   for(i=0;i<bins;i++) {
      d[i] = 0;
   }

   /* scan the array to distribute cost per bin */
   for(i=0; i < num_iters; i++ ) {
      j = ((a[i] - min)*bins)/range;
      d[j]++;
      if (j && (dmax < d[j])) dmax = d[j];
   }

   /* scale each bin to a max of 100 */
   for(i=1;i<bins;i++) {
      d[i] = (d[i] * 100)/dmax;
   }
}

static void print_dist(long_long min, long_long max, int bins, int *d) {
   int i,j;
   int step = (max - min) / bins;

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

static void print_stats(int i, long_long min, long_long max, double average, double std) {
   char *test[] = {"PAPI_start/stop","PAPI_read"};
   printf("\nTotal cost for %s(2 counters) over %d iterations\n", test[i], num_iters);
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

int main(int argc, char **argv)
{
   int i, retval, EventSet = PAPI_NULL;
   int bins = 100;
   int show_dist = 0, show_std_dev = 0;
   int s[10];
   long_long totcyc, values[2];
   long_long *array;
   long_long min, max;
   double  average, std;


   tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */

   for (i = 0; i < argc; i++) {
      if (argv[i]) {
         if (strstr(argv[i], "-b")) {
            bins = atoi(argv[i+1]);
            if (bins) i++;
            else {
               printf ("-b requires a bin count!\n");
               exit(0);
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
            num_iters = atol(argv[i+1]);
            if (num_iters) i++;
            else {
               printf ("-t requires a threshold value!\n");
               exit(0);
            }
         }
      }
   }

   printf("Cost of execution for PAPI start/stop and PAPI read.\n");
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


   /* Make sure no errors */

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);
   if ((retval = PAPI_stop(EventSet, NULL)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   array = (long_long *)malloc(num_iters*sizeof(long_long));

   /* Start the start/stop eval */

   printf("Performing start/stop test...\n");
   if (array == NULL ) 
      test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

   for (i = 0; i < num_iters; i++) {
      totcyc = PAPI_get_real_cyc();
      PAPI_start(EventSet);
      PAPI_stop(EventSet, values);
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i]=totcyc;
   }

   std = do_stats(array, &min, &max, &average);

   print_stats(0, min, max, average, std);

   if (show_std_dev) {
      do_std_dev(array, s, std, average);
      print_std_dev(s);
   }

   if (show_dist) {
      int *d;
      d = malloc(bins*sizeof(int));
      do_dist(array, min, max, bins, d);
      print_dist(min, max, bins, d);
      free(d);
   }


   /* Start the read eval */
   printf("\n\nPerforming read test...\n");

   if ((retval = PAPI_start(EventSet)) != PAPI_OK)
      test_fail(__FILE__, __LINE__, "PAPI_start", retval);

   for (i = 0; i < num_iters; i++) {
      totcyc = PAPI_get_real_cyc();
      PAPI_read(EventSet, values);
      totcyc = PAPI_get_real_cyc() - totcyc;
      array[i]=totcyc;
   }

   std = do_stats(array, &min, &max, &average);
   
   print_stats(1, min, max, average, std);

   if (show_std_dev) {
      do_std_dev(array, s, std, average);
      print_std_dev(s);
   }

   if (show_dist) {
      int *d;
      d = malloc(bins*sizeof(int));
      do_dist(array, min, max, bins, d);
      print_dist(min, max, bins, d);
      free(d);
   }

   free(array);
   test_pass(__FILE__, NULL, 0);
   exit(1);
}
