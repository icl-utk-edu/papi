/*
 * A simple example for the use of PAPI, the number of flops you should
 * get is about INDEX^3  on machines that consider add and multiply one flop
 * such as SGI, and 2*(INDEX^3) that don't consider it 1 flop such as INTEL
 * -Kevin London
 */

#include "papi_test.h"

#define INDEX 100

#ifdef _WIN32
	char format_string[] = {"Real_time: %f Proc_time: %f Total flpins: %I64d MFLOPS: %f\n"};
#else
	char format_string[] = {"Real_time: %f Proc_time: %f Total flpins: %lld MFLOPS: %f\n"};
#endif
int TESTS_QUIET=0;	/*Tests in Verbose mode? */


int main(int argc, char **argv) {
  extern void dummy(void *);
  float matrixa[INDEX][INDEX], matrixb[INDEX][INDEX], mresult[INDEX][INDEX];
  float real_time, proc_time, mflops;
  long_long flpins;
  int retval;
  int i,j,k;

  if ( argc > 1 ) {
        if ( !strcmp( argv[1], "TESTS_QUIET" ) )
           TESTS_QUIET=1;
  }


  /* Initialize the Matrix arrays */
  for ( i=0; i<INDEX*INDEX; i++ ){
	mresult[0][i] = 0.0;
	matrixa[0][i] = matrixb[0][i] = rand()*(float)1.1; }

  /* Setup PAPI library and begin collecting data from the counters */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_flops", retval);

  /* Matrix-Matrix multiply */
  for (i=0;i<INDEX;i++)
   for(j=0;j<INDEX;j++)
    for(k=0;k<INDEX;k++)
	mresult[i][j]=mresult[i][j] + matrixa[i][k]*matrixb[k][j];

  /* Collect the data into the variables passed in */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
  dummy((void*) mresult);
 
  if ( !TESTS_QUIET )
    printf(format_string, real_time, proc_time, flpins, mflops);
  test_pass(__FILE__,NULL,0);
}


