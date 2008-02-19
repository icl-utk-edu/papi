/*
*  Test PAPI_overflow() running in multiple threads.
*
*  This program launches some threads, starts PAPI_overflow() in one
*  or all threads, churns cycles for a few seconds, counts the number
*  of interrupts and loop iterations on a per-second basis, and
*  prints the results.
*
*  When running PAPI_overflow() in all threads (signal_thead = 99),
*  often something happens and some of the threads stop receiving
*  interrupts.
*/

#include <sys/time.h>
#include <sys/types.h>
#include <err.h>
#include <errno.h>
#include <papi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_THREADS   40
#define MAX_SECS     200

#define DEF_PROGRAM_TIME  10
#define DEF_THRESHOLD      20000000
#define DEF_NUM_THREADS    3
#define DEF_SIGNAL_THREAD  0
#define EVENT  PAPI_TOT_CYC

int program_time = DEF_PROGRAM_TIME;
int threshold = DEF_THRESHOLD;
int num_threads = DEF_NUM_THREADS;
int signal_thread = DEF_SIGNAL_THREAD;

pthread_t tid[MAX_THREADS];
long count[MAX_THREADS][MAX_SECS];
long iter[MAX_THREADS][MAX_SECS];

struct timeval start, end;
volatile char at_end = 0;

void
my_handler(int EventSet, void *pc, long long ovec, void *context)
{
   struct timeval now;
   pthread_t self;
   int th;

   gettimeofday(&now, NULL);
   self = pthread_self();
   for (th = 0; th <= num_threads; th++) {
       if (pthread_equal(tid[th], self)) {
           count[th][now.tv_sec - start.tv_sec]++;
           return;
       }
   }
   errx(1, "unable to find my pthread id");
}

void
print_results(void)
{
   int th, n;
   long den;

   for (th = 0; th <= num_threads; th++) {
	printf("\nthread %d\n", th);
	for (n = 0; n <= program_time; n++) {
	    den = (iter[th][n] == 0) ? 1 : iter[th][n];
	    printf("count[%d] = %ld, iter[%d] = %ld, KCYC/iter= %ld\n",
		   n, count[th][n], n, iter[th][n],
		   (count[th][n] * threshold) / (1000 * den));
	}
   }
}

void
launch_timer(void)
{
   int EventSet = PAPI_NULL;

   if (PAPI_register_thread() != PAPI_OK)
       errx(1, "PAPI_register_thread failed");

   if (PAPI_create_eventset(&EventSet) != PAPI_OK)
       errx(1, "PAPI_create_eventset failed");

   if (PAPI_add_event(EventSet, EVENT) != PAPI_OK)
       errx(1, "PAPI_add_event failed");

   if (PAPI_overflow(EventSet, EVENT, threshold, PAPI_OVERFLOW_FORCE_SW, my_handler) != PAPI_OK)
       errx(1, "PAPI_overflow failed");

   if (PAPI_start(EventSet) != PAPI_OK)
       errx(1, "PAPI_start failed");
}

void
do_cycles(int th)
{
   struct timeval now;
   double x, sum;

   for (;;) {
       sum = 1.0;
       for (x = 1.0; x < 250000.0; x += 1.0)
           sum += x;
       if (sum < 0.0)
           printf("==>>  SUM IS NEGATIVE !!  <<==\n");

	gettimeofday(&now, NULL);
	iter[th][now.tv_sec - start.tv_sec]++;
	if (now.tv_sec > start.tv_sec + program_time)
	    break;
   }

   if (at_end) {
       for (;;)
           usleep(1);
   }
   at_end = 1;
   print_results();
   exit(0);
}

void *
my_thread(void *v)
{
   long num = (long)v;

   if (num == signal_thread || signal_thread == 99) {
       launch_timer();
       printf("launched timer from thread %ld\n", num);
   }
   do_cycles(num);

   return (NULL);
}

/*
* Main program args:
* program_time, threshold, num_threads, signal_thread.
*/
int
main(int argc, char **argv)
{
   int i, j;

   if (argc < 2 || sscanf(argv[1], "%d", &program_time) < 1)
       program_time = DEF_PROGRAM_TIME;
   if (argc < 3 || sscanf(argv[2], "%d", &threshold) < 1)
       threshold = DEF_THRESHOLD;
   if (argc < 4 || sscanf(argv[3], "%d", &num_threads) < 1)
       num_threads = DEF_NUM_THREADS;
   if (argc < 5 || sscanf(argv[4], "%d", &signal_thread) < 1)
       signal_thread = DEF_SIGNAL_THREAD;

   printf("program_time = %d, threshold = %d, "
	   "num_threads = %d, signal_thread = %d\n",
	   program_time, threshold, num_threads, signal_thread);

   for (i = 0; i <= num_threads; i++) {
	for (j = 0; j < program_time + 2; j++) {
	    count[i][j] = 0;
	    iter[i][j] = 0;
	}
   }

   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
       errx(1, "PAPI_library_init failed");

   if (PAPI_thread_init(pthread_self) != PAPI_OK)
       errx(1, "PAPI_thread_init failed");

   /*
    * Thread 0 is main, create 1, ..., num_threads.
    */
   tid[0] = pthread_self();
   gettimeofday(&start, NULL);
   for (i = 1; i <= num_threads; i++) {
       if (pthread_create(&tid[i], NULL, my_thread, (void *)(long)i))
           errx(1, "pthread_create failed");
   }

   my_thread((void *)0);

   return (0);
}
