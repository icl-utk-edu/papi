#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#undef NDEBUG
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
#include "test_utils.h"

long long **allocate_test_space(int num_tests, int num_events)
{
  long long **values;
  int i;

  values = (long long **)malloc(num_tests*sizeof(long long *));
  if (values==NULL)
    exit(1);
  memset(values,0x0,num_tests*sizeof(long long *));
    
  for (i=0;i<num_tests;i++)
    {
      values[i] = (long long *)malloc(num_events*sizeof(long long));
      if (values[i]==NULL)
	exit(1);
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
   0x400 is PAPI_L2_TCH.
   0x200 is PAPI_L2_TCA.
   0x100 is PAPI_L2_TCM.
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

  retval = PAPI_get_opt(PAPI_GET_MAX_HWCTRS,NULL);
  if (retval < 1)
    exit(1);
 
  if (PAPI_create_eventset(&EventSet) != PAPI_OK)
    exit(1);

  if (*mask & 0x400)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L2_TCH);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L2_TCH is not available.\n");
	  *mask = *mask ^ 0x400;
	}
    }

  if (*mask & 0x200)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L2_TCA);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L2_TCA is not available.\n");
	  *mask = *mask ^ 0x200;
	}
    }

  if (*mask & 0x100)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L2_TCM);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L2_TCM is not available.\n");
	  *mask = *mask ^ 0x100;
	}
    }

  if (*mask & 0x40)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_DCM);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L1_DCM is not available.\n");
	  *mask = *mask ^ 0x40;
	}
    }

  if (*mask & 0x20)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_ICM);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L1_ICM is not available.\n");
	  *mask = *mask ^ 0x20;
	}
    }

  if (*mask & 0x10)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_TCM);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L1_TCM is not available.\n");
	  *mask = *mask ^ 0x10;
	}
    }

  if (*mask & 0x8)
    {
      retval = PAPI_add_event(&EventSet, PAPI_FLOPS);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_FLOPS is not available.\n");
	  *mask = *mask ^ 0x8;
	}
    }

  if (*mask & 0x4)
    {
#if defined(__digital__)
      fprintf(stderr,"Using PAPI_TOT_INS instead of PAPI_FP_INS.\n");
      retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
#else
      retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
#endif
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_FP_INS is not available.\n");
	  *mask = *mask ^ 0x4;
	}
    }

  if (*mask & 0x2)
    {
      retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_TOT_INS is not available.\n");
	  *mask = *mask ^ 0x2;
	}
    }

  if (*mask & 0x1)
    {
      retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_TOT_CYC is not available.\n");
	  *mask = *mask ^ 0x1;
	}
    }

  return(EventSet);
}

int remove_test_events(int *EventSet, int mask)
{
  int retval = PAPI_OK;
  
  if (mask & 0x400) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L2_TCH);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & 0x200) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L2_TCA);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & 0x100) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L2_TCM);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & 0x40) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_DCM);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & 0x20) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_ICM);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & 0x10) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_TCM);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & 0x8) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_FLOPS);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & 0x4) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_FP_INS);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & 0x2) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_TOT_INS);
      if (retval < PAPI_OK) return(retval);
    }
 
  if (mask & 0x1) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_TOT_CYC);
      if (retval < PAPI_OK) return(retval); 
    }
  
  return(PAPI_destroy_eventset(EventSet));
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
