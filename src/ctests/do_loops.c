/* Compile me with -O0 or else you'll get none. */

#include "papi_test.h"
#include <fcntl.h>

#define L1_MISS_BUFFER_SIZE_INTS 128*1024
static int buf[L1_MISS_BUFFER_SIZE_INTS];

volatile double a = 0.5, b = 6.2;
volatile long_long z = -101010101;

void do_flops(int n)
{
  extern void dummy(void *);
  int i;
  double c = 0.11;

  for (i=0; i < n; i++) {
    c += a * b;
#ifndef _CRAYT3E
    dummy((void*) &c);
#endif
  }
}

void do_reads(int n)
{
#if !defined(_WIN32)
  int i, j;
  static int fd = -1;
  char buf;
  long_long c = (long_long)0.11;

  if (fd == -1)
    {
      fd = open("/dev/null",O_RDONLY);
      if (fd == -1)
        {
          perror("open(/dev/null)");
          exit(1);
        }
    }

  for (i=0; i < n; i++) {
    for (j=0; j < n; j++) {
      c += z;
    }
    read(fd,&buf,sizeof(buf));
  }
#endif /* _WIN32 */
}

void do_l1misses(int n)
{
  int i, j;

  for (j=0; j < n; j++)
    for (i=0; i < L1_MISS_BUFFER_SIZE_INTS; i++)
      buf[i] = buf[L1_MISS_BUFFER_SIZE_INTS-i-1] + 1;
}

void do_both(int n)
{
  int i,j;
  double c = 0.11;
  const int flops = NUM_FLOPS / n;
  const int reads = NUM_READS / n;

  for (i=0;i<n;i++)
    {
    	do_flops(flops);
	do_reads(reads);
    }
}
