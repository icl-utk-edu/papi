#include "mex.h"
#include "papi.h"

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
