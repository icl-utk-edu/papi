/* Compile me with -O0 or else you'll get none. */

#include "papi_test.h"

static int buf[L1_MISS_BUFFER_SIZE_INTS];
volatile double a = 0.5, b = 2.2;

void do_reads(int n)
{
#if !defined(_WIN32)
   int i, j, retval;
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


void do_l1misses(int n)
{
   int i, j;

   for (j = 0; j < n; j++)
      for (i = 0; i < L1_MISS_BUFFER_SIZE_INTS; i++)
         buf[i] = buf[L1_MISS_BUFFER_SIZE_INTS - i - 1] + 1;
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
