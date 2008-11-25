#include "mex.h"
#include "matrix.h"
#include "papi.h"

static long long accum_error = 0;
static long long start_time = 0;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
  float real_time, proc_time, rate;
  double *x;
  unsigned int mrows, ncols;
  int result;
  unsigned int flop_events[2];
  long long ins = 0, flop_values[2];
  long long elapsed_time;

  /* Check for proper number of arguments. */
    if(nrhs > 1) {
      mexErrMsgTxt("This function expects one optional input.");
    } else if(nlhs > 2) {
	mexErrMsgTxt("This function produces 1 or 2 outputs: [ops, mflops].");
    }
    /* The input must be a noncomplex scalar double.*/
    if(nrhs == 1) {
      mrows = mxGetM(prhs[0]);
      ncols = mxGetN(prhs[0]);
      if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || !(mrows == 1 && ncols == 1)) {
        mexErrMsgTxt("Input must be a noncomplex scalar double.");
      }
      /* Assign a pointer to the input. */
      x = mxGetPr(prhs[0]);

      /* if input is 0, reset the counters by calling PAPI_stop_counters with 0 values */
      if(*x == 0) {
         if (start_time == 0) {
            PAPI_stop_counters(NULL, 0);
            accum_error = 0;
         } else {
            start_time = 0;
            PAPI_stop_counters(flop_values, 2);
         }
      }
    }
   if(result = PAPI_event_name_to_code("EMON_SSE_SSE2_COMP_INST_RETIRED_PACKED_DOUBLE", &(flop_events[0])) < PAPI_OK) {
      if(result = PAPI_flops( &real_time, &proc_time, &ins, &rate)<PAPI_OK) {
         mexPrintf("Error code: %d\n", result);
	      mexErrMsgTxt("Error getting flops.");
      }
   } else {
      if(start_time == 0) {
         flop_events[1] = PAPI_FP_OPS;
         start_time = PAPI_get_real_usec();
         if((result = PAPI_start_counters(flop_events, 2)) < PAPI_OK) {
            mexPrintf("Error code: %d\n", result);
            mexErrMsgTxt("Error getting flops.");
         } else {
            ins = 0;
            rate = 0;
         }
      } else {
         if((result = PAPI_read_counters(flop_values, 2)) < PAPI_OK) {
            mexPrintf("%d\n", result);
            mexErrMsgTxt("Error reading the running counters.");
         } else {
            elapsed_time = PAPI_get_real_usec() - start_time;
            ins = (2*flop_values[0])+flop_values[1];
            rate = ((float)ins)/((float)elapsed_time);
         }
      }
   }


/*    mexPrintf("real: %f, proc: %f, rate: %f, ins: %lld\n", real_time, proc_time, rate, ins); */

    if(nlhs > 0) {
      plhs[0] = mxCreateScalarDouble((double)(ins - accum_error));
      /* this call adds 7 fp instructions to the total */
      /* but apparently not on Pentium M with Matlab 7.0.4 */
//      accum_error += 7;
      if(nlhs == 2) {
        plhs[1] = mxCreateScalarDouble((double)rate);
      /* the second call adds 4 fp instructions to the total */
      /* but apparently not on Pentium M with Matlab 7.0.4 */
//        accum_error += 4;
      }
    }
}
