#include "mex.h"
#include "matrix.h"
#include "papi.h"

/*
  Compile in Matlab with the following:
	>> cd c;papi/src/win2k/matlab
	>> mex -I..\.. -I..\substrate PAPI_start_flops.c ..\substrate\release\winpapi.lib
    -- Make sure you are using the Microsoft C++ compiler.
*/

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  float real_time, proc_time, mflops;
  long_long flpins = -1;

  /* Check for proper number of arguments. */
  if(nrhs != 0) {
    mexErrMsgTxt("This function requires no inputs.");
  } else if(nlhs != 0) {
    mexErrMsgTxt("This function produces no outputs.");
  }

  /* resets the counters by passing -1 in flpins */
  if(PAPI_flops( &real_time, &proc_time, &flpins, &mflops)<PAPI_OK) { 
    mexErrMsgTxt("Error getting flops.");
  }
}

