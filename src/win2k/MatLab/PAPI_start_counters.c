#include "mex.h"
#include "matrix.h"
#include "papi.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
   unsigned int mrows, nchars, result, *events;
   char *x;
   int i;

   if(nlhs != 0) {
      mexErrMsgTxt("This function produces no output.");
   }
   mrows = mxGetM(prhs[0]);
   events = (unsigned int *)mxCalloc(nrhs, sizeof(unsigned int) + 1);
   for(i = 0; i < nrhs; i++) {
      if(!mxIsChar(prhs[i]) || mxIsComplex(prhs[i]) || !(mrows == 1) ) {
         mexErrMsgTxt("Input must be a list of strings.");
      }
      nchars = mxGetNumberOfElements(prhs[i]);
      x = (char *)mxCalloc(nchars, sizeof(char) + 1);
      x = mxArrayToString(prhs[i]);
      if(PAPI_event_name_to_code(x, &(events[i])) < PAPI_OK) {
         mxFree(x);
         mexErrMsgTxt("Incorrect PAPI code given.");
      }
      mxFree(x);
   }
   if((result = PAPI_start_counters(events, nrhs)) < PAPI_OK) {
      mxFree(events);
      mexPrintf("Error code: %u\n", result);
      mexErrMsgTxt("Error initializing counters.");
   }
   mxFree(events);
}
