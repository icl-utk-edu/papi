#include "mex.h"
#include "matrix.h"
#include "papi.h"

/*
  Compile in Matlab with the following:
	>> cd c;papi/src/win2k
	>> mex -I..\.. -I..\substrate PAPI_count_flops.c ..\substrate\release\winpapi.lib
    -- Make sure you are using the Microsoft C++ compiler.
*/

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  float real_time, proc_time, mflops;
  long_long flpins = 0;

  /* Check for proper number of arguments. */
  if(nrhs != 0) {
    mexErrMsgTxt("This function expects no inputs.");
  } else if(nlhs != 2) {
    mexErrMsgTxt("This function produces 2 outputs: [ops, mflops].");
  }

  if(PAPI_flops( &real_time, &proc_time, &flpins, &mflops)<PAPI_OK) { 
    mexErrMsgTxt("Error getting flops.");
  }

  plhs[0] = mxCreateScalarDouble((double)flpins);
  plhs[1] = mxCreateScalarDouble((double)mflops);
}

