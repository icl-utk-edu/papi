#define L1_MISS_BUFFER_SIZE_INTS 77777
#include <assert.h>
#include <time.h>
#include "test_utils.h"

/* Compile me with -O0 or else you'll get none. */

void do_flops(int n)
{
  int i;
  double a = 0.5, b = 6.2, c;

  for (i=0; i < n; i++) {
    c = a*b;
  }
}

/* This barely works */

static int buf[L1_MISS_BUFFER_SIZE_INTS];

void do_l1misses(int n, int m, int s)
{
  int i, j;

  for (j=0; j < n; j++) 
    for (i=0; i < m; i++) 
      {
	buf[i] = buf[m-i] + s;
      }
}	

void do_both(int n)
{
  int i,j;
  double a = 0.5, b = 6.2, c;

  for (i=0;i<n;i++)
    {
      for (j=0; j < n*10; j++) 
	c += a*b;
      for (j=0;j<L1_MISS_BUFFER_SIZE_INTS;j++)
	buf[L1_MISS_BUFFER_SIZE_INTS-j-1] = buf[j] + 1;
      for (j=0; j < n*10; j++) 
	c += a*b;
    }
}

