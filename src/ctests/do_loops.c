/* Compile me with -O0 or else you'll get none. */

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include "test_utils.h"

#define L1_MISS_BUFFER_SIZE_INTS 128*1024
static int buf[L1_MISS_BUFFER_SIZE_INTS];

volatile double a = 0.5, b = 6.2;
volatile long long z = -101010101;

void do_flops(int n)
{
  int i;
  double c = 0.11;

  for (i=0; i < n; i++) {
    c += a * b;
  }
}

void do_reads(int n)
{
  int i, j;
  static int fd = -1;
  char buf;
  long long c = 0.11;

  if (fd == -1)
    {
      fd = open("/dev/zero",O_RDONLY);
      if (fd == -1)
	{
	  perror("open(/dev/zero)");
	  exit(1);
	}
    }
  
  for (i=0; i < n; i++) {
    for (j=0; j < n; j++) {
      c += z;
    }
    read(fd,&buf,sizeof(buf));
  }
}

void do_l1misses(int n)
{
  int i, j;

  for (j=0; j < n; j++) 
    for (i=0; i < L1_MISS_BUFFER_SIZE_INTS; i++) 
      buf[i] = buf[L1_MISS_BUFFER_SIZE_INTS-i] + 1;
}	

void do_both(int n)
{
  int i,j;
  double c = 0.11;

  for (i=0;i<n;i++)
    {
      for (j=0; j < n; j++) 
	c += a*b;
      for (j=0;j<L1_MISS_BUFFER_SIZE_INTS;j++)
	buf[j] = buf[L1_MISS_BUFFER_SIZE_INTS-j] + 1;
    }
}

