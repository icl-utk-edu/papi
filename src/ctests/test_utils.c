#include "papi_test.h"

long_long **allocate_test_space(int num_tests, int num_events)
{
  long_long **values;
  int i;

  values = (long_long **)malloc(num_tests*sizeof(long_long *));
  if (values==NULL)
    exit(1);
  memset(values,0x0,num_tests*sizeof(long_long *));
    
  for (i=0;i<num_tests;i++)
    {
      values[i] = (long_long *)malloc(num_events*sizeof(long_long));
      if (values[i]==NULL)
	exit(1);
      memset(values[i],0x00,num_events*sizeof(long_long));
    }
  return(values);
}

void free_test_space(long_long **values, int num_tests)
{
  int i;

  for (i=0;i<num_tests;i++)
    free(values[i]);
  free(values);
}

/* Mask tells us what to select. 
	See test_utils.h for mask definitions
*/

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

  if (*mask & MASK_L2_TCH)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L2_TCH);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L2_TCH is not available.\n");
	  *mask = *mask ^ MASK_L2_TCH;
	}
    }

  if (*mask & MASK_L2_TCA)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L2_TCA);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L2_TCA is not available.\n");
	  *mask = *mask ^ MASK_L2_TCA;
	}
    }

  if (*mask & MASK_L2_TCM)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L2_TCM);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L2_TCM is not available.\n");
	  *mask = *mask ^ MASK_L2_TCM;
	}
    }

  if (*mask & MASK_L1_DCM)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_DCM);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L1_DCM is not available.\n");
	  *mask = *mask ^ MASK_L1_DCM;
	}
    }

  if (*mask & MASK_L1_ICM)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_ICM);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L1_ICM is not available.\n");
	  *mask = *mask ^ MASK_L1_ICM;
	}
    }

  if (*mask & MASK_L1_TCM)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L1_TCM);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_L1_TCM is not available.\n");
	  *mask = *mask ^ MASK_L1_TCM;
	}
    }

  if (*mask & MASK_FLOPS)
    {
      retval = PAPI_add_event(&EventSet, PAPI_FLOPS);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_FLOPS is not available.\n");
	  *mask = *mask ^ MASK_FLOPS;
	}
    }

  if (*mask & MASK_FP_INS)
    {
#if defined(__digital__)
      fprintf(stderr,"Using PAPI_TOT_INS instead of PAPI_FP_INS.\n");
      retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
	  *mask = *mask ^ MASK_FP_INS;
	  *mask = *mask & MASK_TOT_INS;
#else
      retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
#endif
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_FP_INS is not available.\n");
	  *mask = *mask ^ MASK_FP_INS;
	}
    }

  if (*mask & MASK_TOT_INS)
    {
      retval = PAPI_add_event(&EventSet, PAPI_TOT_INS);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_TOT_INS is not available.\n");
	  *mask = *mask ^ MASK_TOT_INS;
	}
    }

  if (*mask & MASK_TOT_CYC)
    {
      retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  fprintf(stderr,"PAPI_TOT_CYC is not available.\n");
	  *mask = *mask ^ MASK_TOT_CYC;
	}
    }

  return(EventSet);
}

int remove_test_events(int *EventSet, int mask)
{
  int retval = PAPI_OK;
  
  if (mask & MASK_L2_TCH) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L2_TCH);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & MASK_L2_TCA) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L2_TCA);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & MASK_L2_TCM) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L2_TCM);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & MASK_L1_DCM) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_DCM);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & MASK_L1_ICM) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_ICM);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & MASK_L1_TCM) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_L1_TCM);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & MASK_FLOPS) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_FLOPS);
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & MASK_FP_INS) 
    {
#if defined(__digital__)
      retval = PAPI_rem_event(EventSet, PAPI_TOT_INS);
#else
      retval = PAPI_rem_event(EventSet, PAPI_FP_INS);
#endif
      if (retval < PAPI_OK) return(retval);
    }

  if (mask & MASK_TOT_INS) 
    {
      retval = PAPI_rem_event(EventSet, PAPI_TOT_INS);
      if (retval < PAPI_OK) return(retval);
    }
 
  if (mask & MASK_TOT_CYC) 
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

void test_pass(char *file, long_long **values, int num_tests)
{
	printf("\n%s:  PASSED\n\n", file);
	if (values) free_test_space(values, num_tests);
	PAPI_shutdown();
	exit(0);
}

void test_fail(char *file, int line, char *call, int retval)
{
	char buf[128];

	memset( buf, '\0', sizeof(buf) );
	printf("%s:  FAILED\nLine # %d\n", file, line);
	if ( retval == PAPI_ESYS ) {
		sprintf(buf, "System error in %s:", call );
		perror(buf);
	}
	else if ( retval > 0 ) {
		printf("Error calculating: %s\n", call );
	}
	else {
		char errstring[PAPI_MAX_STR_LEN];
		PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN );
		printf("Error in %s: %s\n", call, errstring );
		/* if the error is because it's not supported or doesn't exist
			log the error message, but return with no error condition
		*/
		if ( retval == PAPI_ESBSTR || retval == PAPI_ENOEVNT ) exit(0);
	}
	exit(1);
}

#ifdef _WIN32
#undef exit
	int wait_exit(int retval)
	{
		HANDLE hStdIn;
		BOOL bSuccess;
		INPUT_RECORD inputBuffer;
		DWORD dwInputEvents; /* number of events actually read */

		printf("Press any key to continue...\n");
		hStdIn = GetStdHandle(STD_INPUT_HANDLE);
		do { bSuccess = ReadConsoleInput(hStdIn, &inputBuffer, 
			1, &dwInputEvents);
		} while (!(inputBuffer.EventType == KEY_EVENT &&
 			inputBuffer.Event.KeyEvent.bKeyDown));
		exit(retval);
	}
#define exit wait_exit
#endif

