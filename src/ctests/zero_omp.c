/* 
* File:    zero_omp.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Nils Smeds
*          smeds@pdc.kth.se
*          Anders Nilsson
*          anni@pdc.kth.se
*/  

/* This file performs the following test: start, stop and timer
functionality for 2 slave OMP threads

   - It attempts to use the following two counters. It may use less
depending on hardware counter resource limitations. These are counted
in the default counting domain and default granularity, depending on
the platform. Usually this is the user domain (PAPI_DOM_USER) and
thread context (PAPI_GRN_THR).

     + PAPI_FP_INS
     + PAPI_TOT_CYC

Each thread inside the Thread routine:
   - Get cyc.
   - Get us.
   - Start counters
   - Do flops
   - Stop and read counters
   - Get us.
   - Get cyc.

Master serial thread:
   - Get us.
   - Get cyc.
   - Run parallel for loop
   - Get us.
   - Get cyc.
*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>

#include "papi.h"
#include "test_utils.h"
#ifdef _OPENMP
#include <omp.h>
#else
#error "This compiler does not understand OPENMP"
#endif

void Thread(int n)
{
  int retval, num_tests = 1, tmp;
  int EventSet1;
  int mask1 = 0x5;
  int num_events1;
  long long **values;
  long long elapsed_us, elapsed_cyc;
  char errstring[PAPI_MAX_STR_LEN];
  
  EventSet1 = add_test_events(&num_events1,&mask1);

  /* num_events1 is greater than num_events2 so don't worry. */

  values = allocate_test_space(num_tests, num_events1);

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  retval = PAPI_start(EventSet1);
  if (retval != PAPI_OK) {
    tmp=PAPI_MAX_STR_LEN;
    PAPI_perror(retval,errstring,tmp);
    printf("PAPI_start failed. %d %s\n",retval,errstring);
    exit(1);
    }

  do_flops(n);
  
  retval = PAPI_stop(EventSet1, values[0]);
  if (retval != PAPI_OK){
    tmp=PAPI_MAX_STR_LEN;
    PAPI_perror(retval,errstring,tmp);
    printf("PAPI_stop failed. %d %s\n",retval,errstring);
    exit(1);
    }

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  remove_test_events(&EventSet1, mask1);

  printf("Thread 0x%x PAPI_FP_INS : \t%lld\n",omp_get_thread_num(),
	 (values[0])[0]);
  printf("Thread 0x%x PAPI_TOT_CYC: \t%lld\n",omp_get_thread_num(),
	 (values[0])[1]);
  printf("Thread 0x%x Real usec   : \t%lld\n",omp_get_thread_num(),
	 elapsed_us);
  printf("Thread 0x%x Real cycles : \t%lld\n",omp_get_thread_num(),
	 elapsed_cyc);

  free_test_space(values, num_tests);
}

int main()
{
  int i, maxthr, retval;
  long long elapsed_us, elapsed_cyc;
  char errstring[PAPI_MAX_STR_LEN];

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT){
    PAPI_perror(retval,errstring,PAPI_MAX_STR_LEN);
    printf("%s:%d::PAPI_library_init failed. %d %s\n",__FILE__,__LINE__,retval,errstring);
    exit(1);
  }

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    exit(1);

  retval = PAPI_thread_init((unsigned long (*)(void))(omp_get_thread_num),0);
  if (retval != PAPI_OK){
    PAPI_perror(retval,errstring,PAPI_MAX_STR_LEN);
    printf("%s:%d::PAPI_thread_init failed. %d %s\n",__FILE__,__LINE__,retval,errstring);
    exit(1);
  }

  elapsed_us = PAPI_get_real_usec();

  elapsed_cyc = PAPI_get_real_cyc();

  maxthr = omp_get_num_procs();

#if defined(sgi)
#pragma omp parallel for private(i) schedule(static)
  for (i=1;i<3;i++)
#else
#pragma omp parallel for
  for (i=1;i<maxthr+1;i++)
#endif
    Thread(1000000*i);

  elapsed_cyc = PAPI_get_real_cyc() - elapsed_cyc;

  elapsed_us = PAPI_get_real_usec() - elapsed_us;

  printf("Master real usec   : \t%lld\n",
	 elapsed_us);
  printf("Master real cycles : \t%lld\n",
	 elapsed_cyc);

  exit(0);
}
