/* Compile me with -O0 or else you'll get none. */

#include "test_utils.h"

#define L1_MISS_BUFFER_SIZE_INTS 128*1024
static int buf[L1_MISS_BUFFER_SIZE_INTS];

void do_flops(int n)
{
  int i;
  double a = 0.5, b = 6.2, c;

  for (i=0; i < n; i++) {
    c = a*b;
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
  double a = 0.5, b = 6.2, c;

  for (i=0;i<n;i++)
    {
      for (j=0; j < n; j++) 
	c += a*b;
      for (j=0;j<L1_MISS_BUFFER_SIZE_INTS;j++)
	buf[j] = buf[L1_MISS_BUFFER_SIZE_INTS-j] + 1;
    }
}

