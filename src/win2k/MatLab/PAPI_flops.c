#include "mex.h"
#include "matrix.h"
#include "papi.h"

/*
  Compile in Matlab with the following:
	>> cd c:papi/src/win2k
	>> mex -I..\.. -I..\substrate PAPI_flops.c ..\substrate\release\winpapi.lib
    -- Make sure you are using the Microsoft C++ compiler.

  Or use the Microsoft Visual Studio environment.
*/

/* accumulates the error overhead of the MATLAB conversion calls */
static long_long accum_error = 0;

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  float real_time, proc_time, mflops;
  double *x;
  int mrows,ncols;
  long_long flpins = 0; // this MUST be initialized to a positive #

  /* Check for proper number of arguments. */
  if(nrhs > 1) {
    mexErrMsgTxt("This function expects one optional input.");
  } else if(nlhs > 2) {
    mexErrMsgTxt("This function produces 1 or 2 outputs: [ops, mflops].");
  }


 /* The input must be a noncomplex scalar double.*/
  if (nrhs == 1) {
	mrows = mxGetM(prhs[0]);
	ncols = mxGetN(prhs[0]);
	if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || !(mrows == 1 && ncols == 1) ) {
		mexErrMsgTxt("Input must be a noncomplex scalar double.");
	}

	/* Assign a pointer to the input. */
	x = mxGetPr(prhs[0]);

	/* if input is 0, reset the counters by passing -1 in flpins */
	if (*x == 0) {
		flpins = -1;
		accum_error = 0; // also reset the accumulated error
	}
  }

  if(PAPI_flops( &real_time, &proc_time, &flpins, &mflops)<PAPI_OK) { 
    mexErrMsgTxt("Error getting flops.");
  }

  if (nlhs > 0) {
	  plhs[0] = mxCreateScalarDouble((double)(flpins - accum_error));
	  // this call adds 7 fp instructions to the total
	  accum_error += 7;
	  if (nlhs == 2) {
		plhs[1] = mxCreateScalarDouble((double)mflops);
		// the second call adds 4 fp instructions to the total
		accum_error += 4;
	  }
  }
}

