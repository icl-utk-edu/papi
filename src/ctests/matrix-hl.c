/****************************************************************************
 *C     
 *C     matrix-hl.f
 *C     An example of matrix-matrix multiplication and using PAPI high level to 
 *C     look at the performance. written by Kevin London
 *C     March 2000
 *C 	Added to c tests to check stop
 *C****************************************************************************
 */
#include "papi.h"

main(int argc, char **argv) {

      int nrows1=175,ncols1=225,nrows2=ncols1,ncols2=150;
      int i,j,k,num_events,retval;
      /*     PAPI standardized event to be monitored */
      int event[2];
      /*     PAPI values of the counters */
      long_long values[2];
      double p[nrows1,ncols1],q[nrows2,ncols2], r[nrows1,ncols2],tmp;
      extern int TESTS_QUIET;

      tests_quiet(argc,argv);

      /*     Setup default values */
      num_events=0;

      /*     See how many hardware events at one time are supported
       *     This also initializes the PAPI library */
      num_events=PAPI_num_counters();
      if ( num_events < 2 ) {
        printf("This example program requries the architecture to "
	       "support 2 simultaneous hardware events...shutting down.\n");
        test_skip(__FILE__,retval);
      }

      if (!TESTS_QUIET) 
	printf("Number of hardware counters supported: %d\n", num_events);

      if (PAPI_query_event(PAPI_FP_INS)!= PAPI_OK)
	event[0] = PAPI_TOT_INS;
      else
        event[0] = PAPI_FP_INS;

      /*     Time used */
      event[1] = PAPI_TOT_CYC;

      /*     matrix 1: read in the matrix values */
      for(i=0;i<nrows1;i++)
	for(j=0;j<nrows1;j++)
          p[i,j] = i*j*1.0;

      for(i=0;i<nrows2;i++)
	for(j=0;j<ncols2;j++)
          q[i,j] = i*j*1.0;

      for(i=0;i<nrows1;i++)
	for(j=0;j<ncols2;j++)
          r[i,j] = i*j*1.0;

      /*     Set up the counters */
      num_events = 2;
      retval=PAPI_start_counters( event, num_events);
      if ( retval != PAPI_OK ) 
        test_fail(__FILE__, __LINE__, 
		  "PAPI_start_counters", retval);

      /*     Clear the counter values */
      retval=PAPI_read_counters(values, num_events);
      if ( retval !=PAPI_OK ) 
        test_fail(__FILE__, __LINE__, 
		  "PAPI_read_counters", retval);

      /*     Compute the matrix-matrix multiplication  */
      for(i=0;i<nrows1;i++)
	for(j=0;j<ncols2;j++)
	  for(k=0;k<ncols1;k++)
            r[i,j]=r[i,j] + p[i,k]*q[k,j];

      /*     Stop the counters and put the results in the array values  */
      retval=PAPI_stop_counters(values,num_events);    
      if ( retval != PAPI_OK ) 
        test_fail(__FILE__, __LINE__, 
		  "PAPI_stop_counters", retval);

      /*     Make sure the compiler does not optimize away the multiplication
            call dummy(r) */

      if (!TESTS_QUIET) {

        if (event[0] == PAPI_TOT_INS) 
          printf("TOT Instructions:  %d\n",values[0]);
        else
          printf("FP Instructions:   %d\n",values[0]);

	printf("Cycles:              %d\n",values[1]);

        if (event[0] == PAPI_FP_INS) {
	  /*     Compare measured FLOPS to expected value */
          tmp=2.0*(double)(nrows1)*(double)(ncols2)*(double)(ncols1);
          if(abs(values[0]-tmp) > tmp*0.05){
	    /*     Maybe we are counting FMAs? */
	    tmp=tmp/2.0;
	    if(abs(values[0]-tmp)>tmp*0.05) {
              printf("Expected operation count: %f\n",2.0*tmp);
              printf("Or possibly (using FMA):  %f\n",tmp);
              printf("Instead I got:            %d\n",values[0]);
              test_fail(__FILE__, __LINE__, 
			"Unexpected FLOP count (check vector operations)", 1);
		}
	  }
	}
      }
      test_pass(__FILE__,0,0);
}
