#include "papi_test.h"

#ifndef _WIN32
  #define ITERS 100000
#else
  #define ITERS 10000
#endif

extern int TESTS_QUIET; /* Declared in test_utils.c */

int main(int argc, char **argv)
{
  long_long elapsed_usec[ITERS], elapsed_cyc[ITERS];
  long_long total_usec = 0, uniq_usec = 0, diff_usec = 0, 
    total_cyc = 0, uniq_cyc = 0, diff_cyc = 0;
  int i,retval;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ((retval=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  if ((retval = PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);

  if ( !TESTS_QUIET ) {
  printf("Test case: Clock latency and resolution.\n");
  printf("-----------------------------------------------\n");
  }

  /* Real */

 for (i=0;i<ITERS;i++)
    elapsed_cyc[i] = PAPI_get_real_cyc();

  for (i=1;i<ITERS;i++)
    {
      if (elapsed_cyc[i] - elapsed_cyc[i-1] < 0)
	abort();
      diff_cyc = elapsed_cyc[i] - elapsed_cyc[i-1];
      if (diff_cyc != 0)
	uniq_cyc++;
      total_cyc += diff_cyc;
    }
  if ( !TESTS_QUIET ) {
  if(uniq_cyc==ITERS-1)
    printf("PAPI_get_real_cyc : %7.3f   <%7.3f\n",
	   (double)total_cyc/(double)(ITERS),
	   (double)total_cyc/(double)uniq_cyc);
  else if(uniq_cyc)
    printf("PAPI_get_real_cyc : %7.3f    %7.3f\n",
	   (double)total_cyc/(double)(ITERS),
	   (double)total_cyc/(double)uniq_cyc);
  else
    printf("PAPI_get_real_cyc : %7.3f   >%7.3f\n",
	   (double)total_cyc/(double)(ITERS),
	   (double)total_cyc);
  }

  for (i=0;i<ITERS;i++)
    elapsed_usec[i] = PAPI_get_real_usec();

  for (i=1;i<ITERS;i++)
    {
      if (elapsed_usec[i] - elapsed_usec[i-1] < 0)
	abort();
      diff_usec = elapsed_usec[i] - elapsed_usec[i-1];
      if (diff_usec != 0)
	uniq_usec++;
      total_usec += diff_usec;
    }
  if ( !TESTS_QUIET ) {
  if(uniq_usec==ITERS-1)
    printf("PAPI_get_real_usec: %7.3f   <%7.3f\n",
	   (double)total_usec/(double)(ITERS),
	   (double)total_usec/(double)uniq_usec);
  else if(uniq_usec)
    printf("PAPI_get_real_usec: %7.3f    %7.3f\n",
	   (double)total_usec/(double)(ITERS),
	   (double)total_usec/(double)uniq_usec);
  else
    printf("PAPI_get_real_usec: %7.3f   >%7.3f\n",
	   (double)total_usec/(double)(ITERS),
	   (double)total_usec);
  }

  /* Virtual */

  total_cyc=0;
  uniq_cyc = 0;
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
	    uniq_cyc++;
	  total_cyc += diff_cyc;
	}
      if ( !TESTS_QUIET ) {
      if(uniq_cyc==ITERS-1)
	printf("PAPI_get_real_cyc : %7.3f   <%7.3f\n",
	       (double)total_cyc/(double)(ITERS),
	       (double)total_cyc/(double)uniq_cyc);
      else if(uniq_cyc)
	printf("PAPI_get_virt_cyc : %7.3f    %7.3f\n",
	       (double)total_cyc/(double)(ITERS),
	       (double)total_cyc/(double)uniq_cyc);
      else
	printf("PAPI_get_virt_cyc : %7.3f   >%7.3f\n",
	       (double)total_cyc/(double)(ITERS),
	       (double)total_cyc);
      }
    }
  else if ( !TESTS_QUIET ) 
    printf("PAPI_get_virt_cyc : Not supported\n");

  total_usec=0;
  uniq_usec = 0;
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
	    uniq_usec++;
	  total_usec += diff_usec;
	}
      if ( !TESTS_QUIET ) {
      if(uniq_usec==ITERS-1)
	printf("PAPI_get_virt_usec: %7.3f   <%7.3f\n",
	       (double)total_usec/(double)(ITERS),
	       (double)total_usec/(double)uniq_usec);
      else if(uniq_usec)
	printf("PAPI_get_virt_usec: %7.3f    %7.3f\n",
	       (double)total_usec/(double)(ITERS),
	       (double)total_usec/(double)uniq_usec);
      else
	printf("PAPI_get_virt_usec: %7.3f   >%7.3f\n",
	       (double)total_usec/(double)(ITERS),
	       (double)total_usec);
      }
    }
  else if ( !TESTS_QUIET )
    {
      printf("PAPI_get_virt_usec: Not supported\n");
    }

  test_pass(__FILE__,NULL,0);
}
