#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "papiStdEventDefs.h"
#include "papi.h"

#ifdef SETMAX
#define MAX SETMAX
#else
#define MAX 10000
#endif
#define TIMES 1000

#define PAPI_MAX_EVENTS 2
long_long PAPI_values1[PAPI_MAX_EVENTS];
long_long PAPI_values2[PAPI_MAX_EVENTS];
long_long PAPI_values3[PAPI_MAX_EVENTS];
static int EventSet = PAPI_NULL;

#include "papi_test.h"

extern int TESTS_QUIET; /* Declared in test_utils.c */

int main(argc, argv)
     int argc;
     char *argv[];
{
    int      i, retval;
    double a[MAX],b[MAX];
    void funcX(),funcA();

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

    for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
      PAPI_values1[i] = PAPI_values2[i] = 0;

    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

  if ( !TESTS_QUIET ) {
	retval = PAPI_set_debug(PAPI_VERB_ECONT);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  }

#ifdef MULTIPLEX
  if (!TESTS_QUIET) { printf("Activating PAPI Multiplex\n"); }
  retval = PAPI_multiplex_init();
  if ( retval != PAPI_OK ) 
    test_fail(__FILE__,__LINE__,"PAPI multiplex init fail\n",retval);
#endif
    
    retval = PAPI_create_eventset(&EventSet);
    if ( retval != PAPI_OK )
      test_fail(__FILE__,__LINE__,"PAPI set event fail\n",retval);

#ifdef MULTIPLEX
    retval = PAPI_set_multiplex( &EventSet );
    if ( retval != PAPI_OK ) 
      test_fail(__FILE__,__LINE__, "PAPI_set_multiplex fails \n",retval);
#endif

    retval = PAPI_add_event( &EventSet, PAPI_FP_INS );
    if (retval < PAPI_OK )
      {
	retval = PAPI_add_event( &EventSet, PAPI_TOT_INS );
	if (retval < PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI add PAPI_FP_INS or PAPI_TOT_INS fail\n",retval); 
	else
	  if (!TESTS_QUIET) { printf("PAPI_TOT_INS\n"); }
      }
    else
      if (!TESTS_QUIET) { printf("PAPI_FP_INS\n"); }

    retval = PAPI_add_event( &EventSet, PAPI_TOT_CYC );
    if (retval < PAPI_OK )
      test_fail(__FILE__,__LINE__,"PAPI add PAPI_TOT_CYC  fail\n",retval); 
    if (!TESTS_QUIET) { printf("PAPI_TOT_CYC\n"); }

    retval = PAPI_start( EventSet );
    if ( retval != PAPI_OK )
      test_fail(__FILE__,__LINE__,"PAPI start fail\n",retval);

    funcX(a,b,MAX);

    retval = PAPI_read(EventSet, PAPI_values1);
    if ( retval != PAPI_OK )
      test_fail(__FILE__,__LINE__,"PAPI read fail \n",retval);

#ifdef RESET
    retval = PAPI_reset(EventSet);
    if ( retval != PAPI_OK )
      test_fail(__FILE__,__LINE__,"PAPI read fail \n",retval);
#endif

    funcA(a,b,MAX);

    retval = PAPI_read(EventSet, PAPI_values2);
    if ( retval != PAPI_OK )
      test_fail(__FILE__,__LINE__,"PAPI read fail \n",retval);

    if (!TESTS_QUIET) { printf("values1 is:\n"); 
    for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
      printf(" %15lld", PAPI_values1[i] );

    printf("\nvalues2 is:\n");
    for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
      printf(" %15lld", PAPI_values2[i] );

#ifndef RESET
    printf("\nPAPI value is : \n" );
    for ( i = 0; i < PAPI_MAX_EVENTS; i++ )
      printf(" %15lld", PAPI_values2[i] - PAPI_values1[i] );
#endif

    printf("\n"); }
    test_pass(__FILE__, NULL, 0);
    exit(1);
}

void funcX(a,b,n)
double a[MAX],b[MAX];
int n;
{
    int i,k;
    for (k=0; k<TIMES; k++)
        for (i=0; i<n; i++) a[i] = a[i]*b[i] + 1.;
}

void funcA(a,b,n)
double a[MAX],b[MAX];
int n;
{
  int i,k;
  double t[MAX];
  for (k=0; k<TIMES; k++)
    for (i=0; i<n; i++) {
      t[i] = b[n-i];
      b[i] = a[n-i];
      a[i] = t[i];
    }
}
