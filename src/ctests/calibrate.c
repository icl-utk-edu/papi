#define INNER_TEST  1
#define VECTOR_TEST 1
#define MATRIX_TEST 1

/*
 * Calibrate.c
Kevin,
I want PAPI's  flops op count calibrated on every system. I understand about IBM;
the scientific community counts the FMA instruction as 2 ops, even if the IBM
counts doesn't. I want PAPI to count a FMA as 2 ops. What I propose is putting
together a calibrator program so we can adjust PAPI to account for any overhead in
PAPI to produce the right thing on order n ops (vector ops like inner product),
order n^2 ops (like matrix vector multiply) and order n^3 ops like matrix multiply.

Let's write a simple test and run it on all the machines that PAPI works on and see
the results. (This is more important than getting Fortran on my laptop.)

Let's meet on August 3rd to review the results from these calibration tests. What I
would like you to do is generate a program to do the 3 ops, in a loop that goes
from 1-100 in steps of 1 and from 100 to 500 in steps of 50. The output should be a
table of op counts from PAPI, from what it should be and the difference.
For inner product it should be 2*n
For matrix vector product it should be 2*n^2
For matrix multiple it should be 2^n3.

What I would like to see is output that looks like this:

Intel Pentium III 850 MHz
inner product test:
n   papi count     theoretical count     difference
1
2
...
matrix vector test
n   papi count     theoretical count     difference
1
2
...
matrix multiply test
n   papi count     theoretical count     difference
1
2
...

For the tests do simple loops, not ATLAS.
for i = 1:n; a = a + x(i)*y(i); end
for i = 1:n; for j = 1:n; x(i) = x(i) + a(i,j)*y(j); end; end;
for i = 1:n; for j = 1:n; for k = 1:n; c(i,j) = c(i,j) + a(i,k)*b(k,j); end; end; end;

- Jack
 */

#include "papi_test.h"

static int l_PAPI_flops(float *real_time, float *proc_time, long_long *flpins, float *mflops);

static void resultline(int i, int j)
{
	float real_time, proc_time, mflops;
	long_long flpins = 0;
	int papi, theory, diff, error, errord;
	
	i++;						/* convert to 1s base  */
	theory = 2 * i * j;			/* thoretical ops   */
	
	l_PAPI_flops( &real_time, &proc_time, &flpins, &mflops);
	papi = (int)(flpins);
	diff = papi - theory;
	errord = abs(1000000 * diff / theory)%100;
	if (diff < 1000000) error = (100 * diff) / theory;
	else error = 100 * (diff / theory);
	printf("%8d %12d %12d %8d %5d.%.4d\n", i, papi, theory, diff, error,errord);
}

static void headerlines(char * title)
{
  const PAPI_hw_info_t *hwinfo = NULL;

  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
    exit(-1);

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
}


#define INDEX1 100
#define INDEX2 250	/* Microsoft can't handle x[500][500]. Sad but true... */
#define INDEX3 500

int main(int argc, char *argv[]) {
  extern void dummy(void *);
  float x[INDEX3], y[INDEX3], z[INDEX3];
  float a[INDEX2][INDEX2], b[INDEX2][INDEX2], c[INDEX2][INDEX2];

  float real_time, proc_time, mflops, aa;
  long_long flpins;
  int i,j,k,t;

  t = 0;
  if (argc > 1) {
	if(!strcmp(argv[1],"1")) t = 1;
	if(!strcmp(argv[1],"2")) t = 2;
	if(!strcmp(argv[1],"3")) t = 3;
  }

  printf("Initializing...");

  /* Initialize the linear arrays */
  for ( i=0; i<INDEX3; i++ ){
	z[i] = 0.0;
	x[i] = y[i] = rand()*(float)1.1; }

  /* Initialize the Matrix arrays */
  for ( i=0; i<INDEX2*INDEX2; i++ ){
	c[0][i] = 0.0;
	a[0][i] = b[0][i] = rand()*(float)1.1; }

  /* Setup PAPI library and begin collecting data from the counters */
  if(l_PAPI_flops( &real_time, &proc_time, &flpins, &mflops)<PAPI_OK){ 
	 printf("Error starting the counters, aborting.\n"); 
	 exit(-1); } 

  if (t == 1 || (INNER_TEST && t == 0)) {
	/* Inner Product test */
	headerlines("Inner Product Test");
	for (i=0;i<INDEX3;i++) {
	 aa = aa + x[i]*y[i];
	 if (i < INDEX1 || ((i+1) % 50) == 0)
		resultline(i, 1);
	}
	flpins = -1;
	l_PAPI_flops( &real_time, &proc_time, &flpins, &mflops);
  }

  if (t == 2 || (VECTOR_TEST && t == 0)) {
	/* Matrix Vector test */
	headerlines("Matrix Vector Test");
	for (i=0;i<INDEX3;i++) {
		for(j=0;j<INDEX3;j++)
			z[i] = z[i] + a[i%INDEX2][j%INDEX2]*y[j];
		if (i < INDEX1 || ((i+1) % 50) == 0)
			resultline(i, INDEX3);
	}

	flpins = -1;
	l_PAPI_flops( &real_time, &proc_time, &flpins, &mflops);
  }

  if (t == 3 || (MATRIX_TEST && t == 0)) {
	/* Matrix Multiply test */
	headerlines("Matrix Multiply Test");
	for (i=0;i<INDEX3;i++) {
		for(j=0;j<INDEX3;j++)
			for(k=0;k<INDEX3;k++)
				c[i%INDEX2][j%INDEX2] = c[i%INDEX2][j%INDEX2] + a[i%INDEX2][k%INDEX2]*b[k%INDEX2][j%INDEX2];
		if (i < INDEX1 || ((i+1) % 50) == 0)
			resultline(i, INDEX3 * INDEX3);
	}
  }

  /* Use results so they don't get optimized away */
  c[0][0] = aa;
  dummy((void*) c);
  dummy((void*) z);
 
  exit(0);
}

/* Use this local version to test potential changes for various platforms
*/
static int initialized = 0;
static int l_PAPI_flops(float *real_time, float *proc_time, long_long *flpins, float *mflops)
{
   static float total_proc_time=0.0; 
   static int EventSet = PAPI_NULL;
   static float mhz;
   static long_long start_us = 0;
   static long_long total_flpins = 0;
   const PAPI_hw_info_t *hwinfo = NULL;
   long_long values[2] = {0,0};
   char buf[500];
   int retval;

   if ( !initialized ) {
	mhz = 0.0;
	*mflops = 0.0;
 	*real_time = 0.0;
 	*proc_time = 0.0;
	*flpins = 0;
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT )
	   return(retval);
	if ( (hwinfo = PAPI_get_hardware_info()) == NULL ) {
	   printf("Error getting hw_info\n");
	   return -1;
        } 
	mhz = hwinfo->mhz;
	PAPI_create_eventset( &EventSet );
	retval = PAPI_add_event(&EventSet, PAPI_FP_INS);
	PAPI_perror( retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     return retval;
	}
	retval = PAPI_add_event(&EventSet, PAPI_TOT_CYC);
	PAPI_perror(retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     return retval;
	}
	initialized = 1;
	start_us = PAPI_get_real_usec();
	retval = PAPI_start(EventSet);
	PAPI_perror(retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     return retval;
	}
   }
   else {
	retval = PAPI_stop( EventSet, values );
	/* If fp instuction count is negative, re-initialize */
	if ( *flpins < 0 ) {
		total_flpins = 0;
		PAPI_reset(EventSet);
		start_us = PAPI_get_real_usec();
	}
	*real_time = (float)((PAPI_get_real_usec()-start_us)/1000000.0);
	PAPI_perror( retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     initialized = 0;
	     return retval;
	}
	*proc_time = (float)(values[1]/(mhz*1000000.0));
	*mflops = (float)(values[0]/(*proc_time*1000000.0));
	total_proc_time += *proc_time;
	total_flpins += values[0];
	*proc_time = total_proc_time;
	*flpins = total_flpins;
 	retval = PAPI_start(EventSet);
	PAPI_perror(retval, buf, 500);
	if ( retval < PAPI_OK ) {
	     PAPI_shutdown();
	     return retval;
	}
   }
   return PAPI_OK;
}



