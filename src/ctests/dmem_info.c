/*
 * This file perfoms the following test:  dynamic memory info
 * The pages used should increase steadily.
 *
 * Author: Kevin London
 *	   london@cs.utk.edu
 */
#include "papi_test.h"
extern int TESTS_QUIET; /*Declared in test_utils.c */
#define ALLOCMEM 200000

int main(int argc, char **argv ) {
  long value[7];
  int retval,i;
  double *a,*b,*c,*d,*e,*f;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval!=PAPI_VER_CURRENT) 
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);

  if ( (value[0]=PAPI_get_dmem_info(PAPI_GET_RESSIZE)) < 0 )
	test_fail(__FILE__,__LINE__,"PAPI_get_dmem_info",value[0]);

  a= (double*) malloc(ALLOCMEM*sizeof(double));
  touch_dummy(a, ALLOCMEM);
  if ( (value[1]=PAPI_get_dmem_info(PAPI_GET_RESSIZE)) < 0 )
	test_fail(__FILE__,__LINE__,"PAPI_get_dmem_info",value[1]);

  b= (double*) malloc(ALLOCMEM*sizeof(double));
  touch_dummy(b, ALLOCMEM);
  if ( (value[2]=PAPI_get_dmem_info(PAPI_GET_RESSIZE)) < 0 )
	test_fail(__FILE__,__LINE__,"PAPI_get_dmem_info",value[2]);

  c= (double*) malloc(ALLOCMEM*sizeof(double));
  touch_dummy(c, ALLOCMEM);
  if ( (value[3]=PAPI_get_dmem_info(PAPI_GET_RESSIZE)) < 0 )
	test_fail(__FILE__,__LINE__,"PAPI_get_dmem_info",value[3]);

  d= (double*) malloc(ALLOCMEM*sizeof(double));
  touch_dummy(d, ALLOCMEM);
  if ( (value[4]=PAPI_get_dmem_info(PAPI_GET_RESSIZE)) < 0 )
	test_fail(__FILE__,__LINE__,"PAPI_get_dmem_info",value[4]);

  e= (double*) malloc(ALLOCMEM*sizeof(double));
  touch_dummy(e, ALLOCMEM);
  if ( (value[5]=PAPI_get_dmem_info(PAPI_GET_RESSIZE)) < 0 )
	test_fail(__FILE__,__LINE__,"PAPI_get_dmem_info",value[5]);

  f= (double*) malloc(ALLOCMEM*sizeof(double));
  touch_dummy(f, ALLOCMEM);
  if ( (value[6]=PAPI_get_dmem_info(PAPI_GET_RESSIZE)) < 0 )
	test_fail(__FILE__,__LINE__,"PAPI_get_dmem_info",value[6]);


  if ( !TESTS_QUIET ) {
     printf("Test case:  Dynamic Memory Information.\n");
     printf("------------------------------------------------------------------------\n");
     for ( i=0;i<7;i++)
        printf("Malloc additional: %d KB  Resident Size in Pages: %ld\n",
	    ((sizeof(double)*ALLOCMEM)/1024),value[i] );
     printf("------------------------------------------------------------------------\n");
     printf("Resident Size in Pages: %ld  Size in Pages: %ld Total Malloc size: %d KB\n", value[6], PAPI_get_dmem_info(PAPI_GET_SIZE), ((6*sizeof(double)*ALLOCMEM)/1024));
	printf("Pagesize in bytes: %ld  Total Resident Memory: %ld KB\n",  PAPI_get_dmem_info(PAPI_GET_PAGESIZE), (PAPI_get_dmem_info(PAPI_GET_PAGESIZE)*value[6]/1024));
  }
  if ( value[6]>value[5]&&value[5]>value[4]&&value[4]>value[3]&&value[3]>value[2]&&value[2]>value[1]&&value[1]>value[0] )
  	test_pass(__FILE__,NULL,0);
  else
	test_fail(__FILE__,__LINE__,"Calculating Resident Memory",value[6]);
  exit(1);
}
