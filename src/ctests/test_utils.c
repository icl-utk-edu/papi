#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <assert.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

long long **allocate_test_space(int num_tests, int num_events)
{
  long long **values;
  int i;

  values = (long long **)malloc(num_tests*sizeof(long long *));
  assert(values!=NULL);
  memset(values,0x0,num_tests*sizeof(long long *));
    
  for (i=0;i<num_tests;i++)
    {
      values[i] = (long long *)malloc(num_events*sizeof(long long));
      assert(values[i]!=NULL);
      memset(values[i],0x00,num_events*sizeof(long long));
    }
  return(values);
}

void free_test_space(long long **values, int num_tests)
{
  int i;

  for (i=0;i<num_tests;i++)
    free(values[i]);
  free(values);
}

/* Mask tells us what to select. 
   0x40 is PAPI_L1_DCM.
   0x20 is PAPI_L1_ICM.
   0x10 is PAPI_L1_TCM.
   0x8 is PAPI_FLOPS.
   0x4 is PAPI_FP_INS.
   0x2 is PAPI_TOT_INS. 
   0x1 is PAPI_TOT_CYC. */


int add_test_events(int *number, int *mask)
{
  int retval;
  int EventSet = PAPI_NULL;

  *number = 0;

  retval = PAPI_num_events();
  assert(retval >= PAPI_OK);
 
  if ((*mask & 0x40) && PAPI_query_event(PAPI_L1_DCM) == PAPI_OK)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_DCM);
      if (retval >= PAPI_OK)
	(*number)++;
      else
	*mask = *mask ^ 0x4;
    }

  if ((*mask & 0x20) && PAPI_query_event(PAPI_L1_ICM) == PAPI_OK)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_ICM);
      if (retval >= PAPI_OK)
	(*number)++;
      else
	*mask = *mask ^ 0x4;
    }

  if ((*mask & 0x10) && PAPI_query_event(PAPI_L1_TCM) == PAPI_OK)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_TCM);
      if (retval >= PAPI_OK)
	(*number)++;
      else
	*mask = *mask ^ 0x4;
    }

  if ((*mask & 0x8) && PAPI_query_event(PAPI_FLOPS) == PAPI_OK)
    {
      retval = PAPI_add_event(&EventSet, PAPI_FLOPS);
      if (retval >= PAPI_OK)
	(*number)++;
      else
	*mask = *mask ^ 0x4;
    }

  if ((*mask & 0x4) && PAPI_query_event(PAPI_FP_INS) == PAPI_OK)
    {
      retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
      if (retval >= PAPI_OK)
	(*number)++;
      else
	*mask = *mask ^ 0x4;
    }

  if ((*mask & 0x2) && PAPI_query_event(PAPI_TOT_INS) == PAPI_OK)
    {
      retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
      if (retval >= PAPI_OK)
	(*number)++;
      else
	*mask = *mask ^ 0x2;
    }

  if ((*mask & 0x1) && PAPI_query_event(PAPI_TOT_CYC) == PAPI_OK)
    {
      retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC);
      if (retval >= PAPI_OK)
	(*number)++;
      else
	*mask = *mask ^ 0x1;
    }

  return(EventSet);
}

void remove_test_events(int *EventSet, int mask)
{
  int retval;

  if (mask & 0x40) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_DCM);
      assert(retval >= PAPI_OK);
    }

  if (mask & 0x20) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_ICM);
      assert(retval >= PAPI_OK);
    }

  if (mask & 0x10) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_TCM);
      assert(retval >= PAPI_OK);
    }

  if (mask & 0x8) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_FLOPS);
      assert(retval >= PAPI_OK);
    }

  if (mask & 0x4) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_FP_INS);
      assert(retval >= PAPI_OK);
    }

  if (mask & 0x2) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_TOT_INS);
      assert(retval >= PAPI_OK);
    }
 
  if (mask & 0x1) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_TOT_CYC);
      assert(retval >= PAPI_OK); 
    }

  retval = PAPI_destroy_eventset(EventSet);
  assert(retval >= PAPI_OK); 
  assert(*EventSet == PAPI_NULL);
}

char *stringify_domain(int domain)
{
  switch(domain)
    {
    case PAPI_DOM_USER:
      return("PAPI_DOM_USER");
    case PAPI_DOM_KERNEL:
      return("PAPI_DOM_KERNEL");
    case PAPI_DOM_OTHER:
      return("PAPI_DOM_OTHER");
    case PAPI_DOM_ALL:
      return("PAPI_DOM_ALL");
    default:
      abort();
    }
  return(NULL);
}

char *stringify_granularity(int granularity)
{
  switch(granularity)
    {
    case PAPI_GRN_THR:
      return("PAPI_GRN_THR");
    case PAPI_GRN_PROC:
      return("PAPI_GRN_PROC");
    case PAPI_GRN_PROCG:
      return("PAPI_GRN_PROCG");
    case PAPI_GRN_SYS_CPU:
      return("PAPI_GRN_SYS_CPU");
    case PAPI_GRN_SYS:
      return("PAPI_GRN_SYS");
    default:
      abort();
    }
  return(NULL);
}
