/* Compile me with -O0 or else you'll get none. */

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#if defined ( _CRAYT3E ) 
#include  <fortran.h>  
#endif
#include "papi.h"

volatile long_long z = -101010101;

void do_reads(int *n)
{
#ifndef _WIN32
  int i, j;
  static int fd = -1;
  char buf;
  long_long c = (long_long)0.11;

  if (fd == -1)
    {
      fd = open("/dev/zero",O_RDONLY);
      if (fd == -1)
	{
	  perror("open(/dev/zero)");
	  exit(1);
	}
    }
  
  for (i=0; i < *n; i++) {
    for (j=0; j < *n; j++) {
      c += z;
    }
    read(fd,&buf,sizeof(buf));
  }
#endif /* _WIN32 */
}

void DO_READS(int *n)
{
  do_reads(n);
}

void do_reads_(int *n)
{
  do_reads(n);
}

void do_reads__(int *n)
{
  do_reads(n);
}
