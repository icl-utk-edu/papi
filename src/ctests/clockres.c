#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include "papi.h"
#define ITERS 10000

int main() 
{
  long long elapsed_usec[10000], elapsed_cyc[10000];
  long long total_usec = 0, total_cyc = 0;
  int i;

  if (PAPI_init() < PAPI_OK)
    abort();

  printf("Test case: Clock resolution.\n");
  printf("-----------------------------------------------\n");

  for (i=0;i<10000;i++)
    elapsed_cyc[i] = PAPI_get_real_cyc();

  for (i=1;i<10000;i++)
    {
      if (elapsed_cyc[i] - elapsed_cyc[i-1] < 0)
	abort();
      total_cyc += elapsed_cyc[i] - elapsed_cyc[i-1];
    }

  for (i=0;i<10000;i++)
    elapsed_usec[i] = PAPI_get_real_usec();

  for (i=1;i<10000;i++)
    {
      if (elapsed_usec[i] - elapsed_usec[i-1] < 0)
	abort();
      total_usec += elapsed_usec[i] - elapsed_usec[i-1];
    }
  
  printf("PAPI_get_real_cyc : %f\n",(double)total_cyc/(double)(ITERS-1));
  printf("PAPI_get_real_usec: %f\n",(double)total_usec/(double)(ITERS-1));

  PAPI_shutdown();
  
  exit(0);
}
