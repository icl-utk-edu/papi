#include "mex.h"
#include "matrix.h"
#include "papi.h"

/*
  Compile in Matlab with the following:
        >> cd c:papi/src/win2k
        >> mex -I..\.. -I..\substrate PAPI_read_counters.c ..\substrate\release\winpapi.lib
    -- Make sure you are using the Microsoft C++ compiler.

  Or use the Microsoft Visual Studio environment.
*/

/*========================================================================*/
/* int PAPI_accum_counters(long_long *values, int array_len)              */
/* Add the running counter values to the values in the values array. This */
/* call implicitly re-initializes the counters to zero and lets them      */
/* continue to run upon return. Returns PAPI_OK if successful, and an     */
/* appropriate error code otherwise.                                      */
/*========================================================================*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
   int  i;
   long_long *x;

   if(nrhs != 0) {
      mexErrMsgTxt("This function expects no input.");
   }
   if(nlhs > PAPI_num_counters()) {
      mexErrMsgTxt("This function produces one output per running counter.");
   }
   x = (long_long *)mxCalloc(nlhs, sizeof(long_long) + 1);
   if(PAPI_read_counters(x, nlhs) < PAPI_OK) {
      mexErrMsgTxt("Error adding to running counters.");
   }
   for(i = 0; i < nlhs; i++) {
      plhs[i] = mxCreateScalarDouble((double)x[i]);
   }
   mxFree(x);
}
