#include "papi_test.h"

/*  Variable to hold reporting status
	if TRUE, output is suppressed
	if FALSE output is sent to stdout
	initialized to FALSE
	declared here so it can be available globally
*/
int TESTS_QUIET=0;

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
  if (retval < 1) test_fail(__FILE__, __LINE__, "PAPI_get_opt", retval);
 
  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset", retval);

  if (*mask & MASK_L2_TCH)
    {
      retval = PAPI_add_event(&EventSet, PAPI_L2_TCH);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  if ( !TESTS_QUIET )
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
	  if ( !TESTS_QUIET )
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
	  if ( !TESTS_QUIET )
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
	  if ( !TESTS_QUIET )
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
	  if ( !TESTS_QUIET )
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
	  if ( !TESTS_QUIET )
	     fprintf(stderr,"PAPI_L1_TCM is not available.\n");
	  *mask = *mask ^ MASK_L1_TCM;
	}
    }

  if (*mask & MASK_BR_CN)
    {
      retval = PAPI_add_event(&EventSet, PAPI_BR_CN);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{

                char errstring[PAPI_MAX_STR_LEN];
                PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN );
	  if ( !TESTS_QUIET ){
                printf("Error: %s\n", errstring );
	        fprintf(stderr,"PAPI_BR_CN is not available.\n");
          }
	  *mask = *mask ^ MASK_BR_CN;
	}
    }
  if (*mask & MASK_BR_MSP)
    {
      retval = PAPI_add_event(&EventSet, PAPI_BR_MSP);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  if ( !TESTS_QUIET )
	     fprintf(stderr,"PAPI_BR_MSP is not available.\n");
	  *mask = *mask ^ MASK_BR_MSP;
	}
    }
  if (*mask & MASK_BR_PRC)
    {
      retval = PAPI_add_event(&EventSet, PAPI_BR_PRC);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  if ( !TESTS_QUIET )
	     fprintf(stderr,"PAPI_BR_PRC is not available.\n");
	  *mask = *mask ^ MASK_BR_PRC;
	}
    }
  if (*mask & MASK_FLOPS)
    {
      retval = PAPI_add_event(&EventSet, PAPI_FLOPS);
      if (retval == PAPI_OK)
	(*number)++;
      else
	{
	  if ( !TESTS_QUIET )
	     fprintf(stderr,"PAPI_FLOPS is not available.\n");
	  *mask = *mask ^ MASK_FLOPS;
	}
    }

  if (*mask & MASK_FP_INS)
    {
#if defined(__digital__)
	  if ( !TESTS_QUIET )
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
	  if ( !TESTS_QUIET )
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
	  if ( !TESTS_QUIET )
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
	  if ( !TESTS_QUIET )
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

void tests_quiet(int argc, char **argv)
{
  if ( argc > 1 ) {
		if ( !strcmp( argv[1], "TESTS_QUIET" ) )
		   TESTS_QUIET=1;
  }
}

void test_pass(char *file, long_long **values, int num_tests)
{
	printf("%-40s PASSED\n", file);
	if (values) free_test_space(values, num_tests);
	PAPI_set_debug( PAPI_QUIET ); /* Prevent error messages on Alpha */
	PAPI_shutdown();
	exit(0);
}

void test_fail(char *file, int line, char *call, int retval)
{
	char buf[128];

	if ( retval == PAPI_ESBSTR || retval == PAPI_ENOEVNT ) 
            test_skip(file,line,call,retval);
	memset( buf, '\0', sizeof(buf) );
	if ( retval != 0 )
	   printf("%-40s FAILED\nLine # %d\n", file, line);
	else{
	       printf("%-40s SKIPPED\n",file );
           if ( !TESTS_QUIET ) printf("Line # %d\n",line );
        }
	if ( retval == PAPI_ESYS ) {
		sprintf(buf, "System error in %s:", call );
		perror(buf);
	}
	else if ( retval > 0 ) {
		printf("Error calculating: %s\n", call );
	}
	else if ( retval == 0 ) {
		printf("SGI requires root permissions for this test\n");
	}
	else {
		char errstring[PAPI_MAX_STR_LEN];
		PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN );
		printf("Error in %s: %s\n", call, errstring );
	}
        printf("\n");
	exit(1);
}

void test_skip(char *file, int line, char *call, int retval)
{
	char buf[128];

	memset( buf, '\0', sizeof(buf) );
	printf("%-40s SKIPPED\n",file );
    if ( !TESTS_QUIET ) {
      printf("Line # %d\n",line );
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
	  }
          printf("\n");
         }
	exit(0);
}

#ifdef _WIN32
#undef exit
	int wait_exit(int retval)
	{
		HANDLE hStdIn;
		BOOL bSuccess;
		INPUT_RECORD inputBuffer;
		DWORD dwInputEvents; /* number of events actually read */
		
		if (!TESTS_QUIET) {
			printf("Press any key to continue...\n");
			hStdIn = GetStdHandle(STD_INPUT_HANDLE);
			do { bSuccess = ReadConsoleInput(hStdIn, &inputBuffer, 
				1, &dwInputEvents);
			} while (!(inputBuffer.EventType == KEY_EVENT &&
 				inputBuffer.Event.KeyEvent.bKeyDown));
		}
		exit(retval);
	}
#define exit wait_exit
#endif

