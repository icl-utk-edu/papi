#include "mex.h"
#include "matrix.h"
#include "papi.h"

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
      mexErrMsgTxt("Error reading the running counters.");
   }
   for(i = 0; i < nlhs; i++) {
      plhs[i] = mxCreateScalarDouble((double)x[i]);
   }
   mxFree(x);
}
