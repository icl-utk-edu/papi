#include "mex.h"
#include "matrix.h"
#include "papi.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
   int  i, result;
   long_long *x;

   if(nrhs != 0) {
      mexErrMsgTxt("This function expects no input.");
   }
   if(nlhs > PAPI_num_counters()) {
      mexErrMsgTxt("This function produces one output per running counter.");
   }
   x = (long_long *)mxCalloc(nlhs, sizeof(long_long) + 1);
   if((result = PAPI_stop_counters(x, nlhs)) < PAPI_OK) {
      mexPrintf("%d\n", result);
      mexErrMsgTxt("Error stopping the running counters.");
   }
   for(i = 0; i < nlhs; i++) {
      plhs[i] = mxCreateScalarDouble((double)x[i]);
   }
   mxFree(x);
}
