#include "mex.h"
#include "matrix.h"
#include "papi.h"

/*
  Compile in Matlab with the following:
        >> cd c:papi/src/win2k
        >> mex -I..\.. -I..\substrate PAPI_num_counters.c ..\substrate\release\winpapi.lib
    -- Make sure you are using the Microsoft C++ compiler.

  Or use the Microsoft Visual Studio environment.
*/

/*========================================================================*/
/* int PAPI_num_counters(void)                                            */
/*                                                                        */
/* This function returns the optimal length of the values array           */
/* for the high level functions. This value corresponds to the            */
/* number of hardware counters supported by the current substrate.        */
/*========================================================================*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
   int counters;
   /* Check for proper number of arguments. */
   if(nrhs != 0) {
      mexErrMsgTxt("This function expects no input.");
   }
   else if(nlhs > 1) {
      mexErrMsgTxt("This function produces only 1 output: counters.");
   }
   counters = PAPI_num_counters();
   if(counters < PAPI_OK) {
      mexErrMsgTxt("Error reading counters.");
   }
   if(nlhs) {
      plhs[0] = mxCreateScalarDouble((double)counters);
   }
}
