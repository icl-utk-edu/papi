/* Compile me with -O0 or else you'll get none. */

#include "papi_test.h"

int buf[CACHE_FLUSH_BUFFER_SIZE_INTS];
int buf_dummy = 0;
volatile int *flush = NULL;
volatile int flush_dummy = 0;
volatile double a = 0.5, b = 2.2;

void do_reads(int n)
{
#if !defined(_WIN32)
   int i, retval;
   static int fd = -1;
   char buf;

   if (fd == -1) {
      fd = open("/dev/zero", O_RDONLY);
      if (fd == -1) {
         perror("open(/dev/zero)");
         exit(1);
      }
   }

   for (i = 0; i < n; i++) {
      retval = read(fd, &buf, sizeof(buf));
      if (retval != sizeof(buf))
	{
	  if (retval < 0)
	    perror("/dev/zero cannot be read");
	  else
	    fprintf(stderr,"/dev/zero cannot be read: only got %d bytes.\n",retval);
	  exit(1);
	}
   }
#endif                          /* _WIN32 */
}

void do_reads_(int *n)
{
  do_reads(*n);
}
void do_reads__(int *n)
{
  do_reads(*n);
}

void do_flops(int n)
{
   int i;
   double c = 0.11;

   for (i = 0; i < n; i++) {
      c += a * b;
#ifndef _CRAYT3E
      dummy((void *) &c);
#endif
   }
}

void do_flops_(int *n)
{
  do_flops(*n);
}

void do_flops__(int *n)
{
  do_flops(*n);
}

void do_misses(int n, int size)
{
  register int i, j, tmp = buf_dummy, len = size / sizeof(int);
  dummy(buf);
  for (j = 0; j < n; j++)
    {
      for (i = 0; i < len; i++)
	{
	  buf[i] = tmp;
	}
      tmp += len;
    }
  buf_dummy = tmp;
  dummy(buf);
  dummy(&buf_dummy);
}

void do_misses_(int *n, int *size)
{
  do_misses(*n,*size);
}

void do_misses__(int *n, int *size)
{
  do_misses(*n,*size);
}

void do_both(int n)
{
   int i;
   const int flops = NUM_FLOPS / n;
   const int reads = NUM_READS / n;

   for (i = 0; i < n; i++) {
      do_flops(flops);
      do_reads(reads);
   }
}

void do_both_(int *n)
{
  do_both(*n);
}

void do_both__(int *n)
{
  do_both(*n);
}

void do_flush(void)
{
  register int i;
  if (flush == NULL)
    flush = (int *)malloc((1024*1024*16)*sizeof(int));
  dummy(flush);
  for (i=0;i<(1024*1024*16);i++)
    {
      flush[i] += flush_dummy;
    }
  flush_dummy++;
  dummy(flush);
  dummy(&flush_dummy);
}

void do_flush_(void)
{
  do_flush();
}

void do_flush__(void)
{
  do_flush();
}

void do_l1misses(int n)
{
  do_misses(n,L1_MISS_BUFFER_SIZE_INTS);
}

void do_l1misses_(int *n)
{
  do_l1misses(*n);
}
void do_l1misses__(int *n)
{
  do_l1misses(*n);
}
