#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi.h"
#define ITERS 100000

int main() 
{
  long long elapsed_usec[ITERS], elapsed_cyc[ITERS];
  long long total_usec = 0, total_cyc = 0, nodup_usec = 0, nodup_cyc = 0, diff_usec = 0, diff_cyc = 0;
  int i;

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    exit(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  printf("Test case: Clock latency and resolution.\n");
  printf("-----------------------------------------------\n");

  /* Real */

  for (i=0;i<ITERS;i++)
    elapsed_cyc[i] = PAPI_get_real_cyc();

  for (i=1;i<ITERS;i++)
    {
      if (elapsed_cyc[i] - elapsed_cyc[i-1] < 0)
	abort();
      diff_cyc = elapsed_cyc[i] - elapsed_cyc[i-1];
      if (diff_cyc != 0)
	nodup_cyc++;
      total_cyc += diff_cyc;
    }

  for (i=0;i<ITERS;i++)
    elapsed_usec[i] = PAPI_get_real_usec();

  for (i=1;i<ITERS;i++)
    {
      if (elapsed_usec[i] - elapsed_usec[i-1] < 0)
	abort();
      diff_usec = elapsed_usec[i] - elapsed_usec[i-1];
      if (diff_usec != 0)
	nodup_usec++;
      total_usec += diff_usec;
    }
  printf("PAPI_get_real_cyc : %f %f\n",(double)total_cyc/(double)(ITERS),(double)total_cyc/(double)nodup_cyc);
  printf("PAPI_get_real_usec: %f %f\n",(double)total_usec/(double)(ITERS),(double)total_usec/(double)nodup_usec);

  /* Virtual */

  nodup_cyc = 0;
  if (PAPI_get_virt_cyc() != -1)
    {
      for (i=0;i<ITERS;i++)
	elapsed_cyc[i] = PAPI_get_virt_cyc();

      for (i=1;i<ITERS;i++)
	{
	  if (elapsed_cyc[i] - elapsed_cyc[i-1] < 0)
	    abort();
	  diff_cyc = elapsed_cyc[i] - elapsed_cyc[i-1];
	  if (diff_cyc != 0)
	    nodup_cyc++;
	  total_cyc += diff_cyc;
	}
      printf("PAPI_get_virt_cyc : %f %f\n",(double)total_cyc/(double)(ITERS),(double)total_cyc/(double)nodup_cyc);
    }
  else
    printf("PAPI_get_virt_cyc : Not supported\n");

  nodup_usec = 0;
  if (PAPI_get_virt_usec() != -1)
    {
      for (i=0;i<ITERS;i++)
	elapsed_usec[i] = PAPI_get_virt_usec();

      for (i=1;i<ITERS;i++)
	{
	  if (elapsed_usec[i] - elapsed_usec[i-1] < 0)
	    abort();
	  diff_usec = elapsed_usec[i] - elapsed_usec[i-1];
	  if (diff_usec != 0)
	    nodup_usec++;
	  total_usec += diff_usec;
	}
      printf("PAPI_get_virt_usec: %f %f\n",(double)total_usec/(double)(ITERS),(double)total_usec/(double)nodup_usec);
    }
  else
    {
      printf("PAPI_get_virt_usec: Not supported\n");
    }

  PAPI_shutdown();
  
  exit(0);
}
