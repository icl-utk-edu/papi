/*
   Calibrate.c
	A program to perform one or all of three tests to count flops.
	Test 1. Inner Product:				2*n operations
		for i = 1:n; a = a + x(i)*y(i); end
	Test 2. Matrix Vector Product:		2*n^2 operations
		for i = 1:n; for j = 1:n; x(i) = x(i) + a(i,j)*y(j); end; end;
	Test 3. Matrix Matrix Multiply:		2*n^3 operations
		for i = 1:n; for j = 1:n; for k = 1:n; c(i,j) = c(i,j) + a(i,k)*b(k,j); end; end; end;

  Supply a command line argument of 1, 2, or 3 to perform each test, or
  no argument to perform all three.

  Each test initializes PAPI and presents a header with processor information.
  Then it performs 500 iterations, printing result lines containing:
  n, measured counts, theoretical counts, (measured - theory), % error
 */

#include "papi_test.h"

static void resultline(int i, int j);
static void headerlines(char * title);

#define INDEX1 100
#define INDEX2 250	/* Microsoft can't handle x[500][500]. */
#define INDEX3 500

int TESTS_QUIET=0;

int main(int argc, char *argv[]) {
  extern void dummy(void *);
  float x[INDEX3], y[INDEX3], z[INDEX3];
  float a[INDEX2][INDEX2], b[INDEX2][INDEX2], c[INDEX2][INDEX2];

  float aa = 0.0;
  int i,j,k,t;
  int retval;


  /*
  Check for inputs of 1, 2, or 3. If TRUE, do that test only.
  Otherwise, do all three tests.
  */
  t = 0;
  if (argc > 1) {
	if(!strcmp(argv[1],"1")) t = 1;
	if(!strcmp(argv[1],"2")) t = 2;
	if(!strcmp(argv[1],"3")) t = 3;
	if(!strcmp(argv[1],"TESTS_QUIET")) TESTS_QUIET=1;
  }

  if ( !TESTS_QUIET ) 
  	printf("Initializing...");

  /* Initialize the linear arrays */
  for ( i=0; i<INDEX3; i++ ){
	z[i] = 0.0;
	x[i] = y[i] = rand()*(float)1.1; }

  /* Initialize the Matrix arrays */
  for ( i=0; i<INDEX2*INDEX2; i++ ){
	c[0][i] = 0.0;
	a[0][i] = b[0][i] = rand()*(float)1.1; }

  /* Initialize PAPI */
  retval = PAPI_library_init( PAPI_VER_CURRENT );
  if ( retval != PAPI_VER_CURRENT) test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);

#ifdef NO_FLOPS
   test_pass(__FILE__,0,0);
#endif
  /* Inner Product test */
  if (t == 1 || t == 0) {
     if ( !TESTS_QUIET ) 
	headerlines("Inner Product Test");

	for (i=0;i<INDEX3;i++) {
	 aa = aa + x[i]*y[i];
	 if (i < INDEX1 || ((i+1) % 50) == 0)
             if ( !TESTS_QUIET ) 
		 resultline(i, 1);
	}
  }

  /* Matrix Vector test */
  if (t == 2 || t == 0) {
     if ( !TESTS_QUIET ) 
	headerlines("Matrix Vector Test");

	for (i=0;i<INDEX3;i++) {
		for(j=0;j<INDEX3;j++)
			z[i] = z[i] + a[i%INDEX2][j%INDEX2]*y[j];
		if (i < INDEX1 || ((i+1) % 50) == 0)
                      if ( !TESTS_QUIET ) 
			resultline(i, INDEX3);
	}
  }

  /* Matrix Multiply test */
  if (t == 3 || t == 0) {
     if ( !TESTS_QUIET ) 
	headerlines("Matrix Multiply Test");

	for (i=0;i<INDEX3;i++) {
		for(j=0;j<INDEX3;j++)
			for(k=0;k<INDEX3;k++)
				c[i%INDEX2][j%INDEX2] = c[i%INDEX2][j%INDEX2] + a[i%INDEX2][k%INDEX2]*b[k%INDEX2][j%INDEX2];
		if (i < INDEX1 || ((i+1) % 50) == 0)
                     if ( !TESTS_QUIET ) 
			resultline(i, INDEX3 * INDEX3);
	}
  }

  /* Use results so they don't get optimized away */
  c[0][0] = aa;
  dummy((void*) c);
  dummy((void*) z);
  test_pass(__FILE__,NULL,0);
}

/*
	Extract and display hardware information for this processor.
	(Re)Initialize PAPI_flops() and begin counting floating ops.
*/
static void headerlines(char * title)
{
  const PAPI_hw_info_t *hwinfo = NULL;
  float real_time, proc_time, mflops;
  long_long flpins;
  int retval;

  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
	test_fail(__FILE__, __LINE__, "PAPI_get_hardware_info", 1);

  printf("\n-------------------------------------------------------------------------\n");
  printf("Vendor string and code   : %s (%d)\n",hwinfo->vendor_string,hwinfo->vendor);
  printf("Model string and code    : %s (%d)\n",hwinfo->model_string,hwinfo->model);
  printf("CPU revision             : %f\n",hwinfo->revision);
  printf("CPU Megahertz            : %f\n",hwinfo->mhz);
  printf("CPU's in an SMP node     : %d\n",hwinfo->ncpu);
  printf("Nodes in the system      : %d\n",hwinfo->nnodes);
  printf("Total CPU's in the system: %d\n",hwinfo->totalcpus);
  printf("-------------------------------------------------------------------------\n");
  printf("\n%s:\n%8s %12s %12s %8s %8s\n", title, "i", "papi", "theory", "diff", "%error");
  printf("-------------------------------------------------------------------------\n");

  /* Setup PAPI library and begin collecting data from the counters */
  flpins = -1;
  retval = PAPI_flops( &real_time, &proc_time, &flpins, &mflops); 
  if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "headerlines: PAPI_flops", retval);
}

/*
  Read PAPI_flops.
  Format and display results.
  Compute error without using floating ops.
*/
static void resultline(int i, int j)
{
	float real_time, proc_time, mflops;
	long_long flpins = 0;
	int papi, theory, diff, adiff, error, errord;
	int retval;
		
	retval = PAPI_flops( &real_time, &proc_time, &flpins, &mflops);
	if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "resultline: PAPI_flops", retval);

	i++;						/* convert to 1s base  */
	theory = 2 * i * j;			/* theoretical ops   */
	papi = (int)(flpins);
	diff = papi - theory;
	adiff = abs(diff);
	if (adiff < 2000)
		errord = ((adiff * 1000000) / theory) % 10000;
	else errord = ((adiff * 1000) / (theory / 1000)) % 10000;
	if (adiff < 1000000) error = (100 * adiff) / theory;
	else error = 100 * (adiff / theory);
	printf("%8d %12d %12d %8d %5d.%.4d\n", i, papi, theory, diff, error,errord);
}



