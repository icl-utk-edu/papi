/* Compile me with -O0 or else you'll get none. */

#include "papi_test.h"

int buf[CACHE_FLUSH_BUFFER_SIZE_INTS];
int buf_dummy = 0;
int *flush = NULL;
int flush_dummy = 0;

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

void do_flops(int n)
{
   extern void dummy(void *);
   int i;
   double c = 0.11;

   for (i = 0; i < n; i++) {
      c += a * b;
#ifndef _CRAYT3E
      dummy((void *) &c);
#endif
   }
}

void do_misses(int n, int size)
{
  int j;
  register int i, len = size / sizeof(int), dummy = buf_dummy;
  for (j = 0; j < n; j++)
    {
      for (i = 0; i < len; i++)
	{
	  buf[i] = dummy;
	}
      dummy += j;
    }
  buf_dummy = dummy;
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

void do_flush(void)
{
  register int i;
  if (flush == NULL)
    flush = (int *)malloc((1024*1024*16)*sizeof(int));
  for (i=0;i<(1024*1024*16);i++)
    {
      flush[i] += flush_dummy;
    }
  flush_dummy++;
}

void do_l1misses(int n)
{
  do_misses(n,L1_MISS_BUFFER_SIZE_INTS);
}
