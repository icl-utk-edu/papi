#include "mex.h"
#include "matrix.h"
#include "papi.h"

static long_long accum_error = 0;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
  float real_time, proc_time, rate;
  double *x;
  unsigned int i, mrows, ncols, nchars;
  unsigned int *events;
  long_long ins = 0, *values;
  int result;
  char *input, *temp;

  /* Check for proper number of arguments. */
  if(nrhs < 1) {
    mexErrMsgTxt("This function expects input.");
  }
  nchars = mxGetNumberOfElements(prhs[0]);
  input = (char *)mxCalloc(nchars, sizeof(char) + 1);
  input = mxArrayToString(prhs[0]);
  if(!strcmp(input, "num")) {
    if(nrhs != 1) {
      mexErrMsgTxt("This function expects no input.");
    }
    else if(nlhs != 1) {
      mexErrMsgTxt("This function produces one and only one output: counters.");
    }
    result = PAPI_num_counters();
    if(result < PAPI_OK) {
      mexErrMsgTxt("Error reading counters.");
    }
    plhs[0] = mxCreateScalarDouble((double)result);
  }

  else if(!strcmp(input, "flips")) {
    if(nrhs > 2) {
      mexErrMsgTxt("This function expects one optional input.");
    } else if(nlhs > 2) {
      mexErrMsgTxt("This function produces 1 or 2 outputs: [ops, mflips].");
    }
    /* The input must be a noncomplex scalar double.*/
    if(nrhs == 2) {
      mrows = mxGetM(prhs[1]);
      ncols = mxGetN(prhs[1]);
      if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !(mrows == 1 && ncols == 1)) {
        mexErrMsgTxt("Input must be a noncomplex scalar double.");
      }
      /* Assign a pointer to the input. */
      x = mxGetPr(prhs[1]);

      /* if input is 0, reset the counters by passing -1 in ins */
      if(*x == 0) {
        ins = -1;
        accum_error = 0;
      }
    }
    if(PAPI_flips( &real_time, &proc_time, &ins, &rate)<PAPI_OK) {
      mexErrMsgTxt("Error getting flops.");
    }
    if(nlhs > 0) {
      plhs[0] = mxCreateScalarDouble((double)(ins - accum_error));
      /* this call adds 7 fp instructions to the total */
      accum_error += 7;
      if(nlhs == 2) {
        plhs[1] = mxCreateScalarDouble((double)rate);
        /* the second call adds 4 fp instructions to the total */
        accum_error += 4;
      }
    }
  }

  else if(!strcmp(input, "start")) {
    if(nlhs != 0) {
      mexErrMsgTxt("This function produces no output.");
    }
    if(nrhs > (PAPI_num_counters() + 1)) {
      mexErrMsgTxt("This function produces one output per running counter.");
    }
    mrows = mxGetM(prhs[1]);
    events = (unsigned int *)mxCalloc(nrhs - 1, sizeof(int) + 1);
    for(i = 1; i < nrhs; i++) {
      if(mxIsComplex(prhs[i]) || !(mrows == 1) ) {
        mexErrMsgTxt("Input must be a list of strings.");
      }
      if(mxIsChar(prhs[i])) {
        nchars = mxGetNumberOfElements(prhs[i]);
        temp = (char *)mxCalloc(nchars, sizeof(char) + 1);
        temp = mxArrayToString(prhs[i]);
        if(PAPI_event_name_to_code(temp, &(events[i - 1])) < PAPI_OK) {
          mxFree(temp);
          mexErrMsgTxt("Incorrect PAPI code given.");
        }
        mxFree(temp);
      }
      else {
        events[i - 1] = (unsigned int)mxGetScalar(prhs[i]);
      }
    }
    if((result = PAPI_start_counters(events, nrhs - 1)) < PAPI_OK) {
      mxFree(events);
      mexPrintf("Error code: %u\n", result);
      mexErrMsgTxt("Error initializing counters.");
    }
    mxFree(events);
  }

  else if(!strcmp(input, "stop")) {
    if(nrhs != 1) {
      mexErrMsgTxt("This function expects no input.");
    }
    if(nlhs > PAPI_num_counters()) {
      mexErrMsgTxt("This function produces one output per running counter.");
    }
    values = (long_long *)mxCalloc(nlhs, sizeof(long_long) + 1);
    if((result = PAPI_stop_counters(values, nlhs)) < PAPI_OK) {
      mexPrintf("%d\n", result);
      mexErrMsgTxt("Error stopping the running counters.");
    }
    for(i = 0; i < nlhs; i++) {
      plhs[i] = mxCreateScalarDouble((double)values[i]);
    }
    mxFree(values);
  }

  else if(!strcmp(input, "read")) {
    if(nrhs != 1) {
      mexErrMsgTxt("This function expects no input.");
    }
    if(nlhs > PAPI_num_counters()) {
      mexErrMsgTxt("This function produces one output per running counter.");
    }
    values = (long_long *)mxCalloc(nlhs, sizeof(long_long) + 1);
    if((result = PAPI_read_counters(values, nlhs)) < PAPI_OK) {
      mexPrintf("%d\n", result);
      mexErrMsgTxt("Error reading the running counters.");
    }
    for(i = 0; i < nlhs; i++) {
      plhs[i] = mxCreateScalarDouble((double)values[i]);
    }
    mxFree(values);
  }

  else if(!strcmp(input, "accum")) {
    if(nrhs > PAPI_num_counters() + 1) {
      mexErrMsgTxt("This function expects no input.");
    }
    if(nlhs > PAPI_num_counters()) {
      mexErrMsgTxt("This function produces one output per running counter.");
    }
    values = (long_long *)mxCalloc(nlhs, sizeof(long_long) + 1);
    for(i = 0; i < nrhs - 1; i++) {
      values[i] = *(mxGetPr(prhs[i + 1]));
    }
    if(PAPI_accum_counters(values, nlhs) < PAPI_OK) {
      mexPrintf("%d\n", result);
      mexErrMsgTxt("Error reading the running counters.");
    }
    for(i = 0; i < nlhs; i++) {
      plhs[i] = mxCreateScalarDouble((double)values[i]);
    }
    mxFree(values);
  }

  else if(!strcmp(input, "ipc")) {
    if(nrhs > 2) {
      mexErrMsgTxt("This function expects one optional input.");
    } else if(nlhs > 2) {
      mexErrMsgTxt("This function produces 1 or 2 outputs: [ops, ipc].");
    }
    /* The input must be a noncomplex scalar double.*/
    if(nrhs == 2) {
      mrows = mxGetM(prhs[1]);
      ncols = mxGetN(prhs[1]);
      if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !(mrows == 1 && ncols == 1)) {
        mexErrMsgTxt("Input must be a noncomplex scalar double.");
      }
      /* Assign a pointer to the input. */
      x = mxGetPr(prhs[1]);

      /* if input is 0, reset the counters by passing -1 in ins */
      if(*x == 0) {
        ins = -1;
        accum_error = 0;
      }
    }
    if(PAPI_ipc(&real_time, &proc_time, &ins, &rate)<PAPI_OK) {
      mexErrMsgTxt("Error getting flops.");
    }
    if(nlhs > 0) {
      plhs[0] = mxCreateScalarDouble((double)(ins - accum_error));
      /* this call adds 7 fp instructions to the total */
      accum_error += 7;
      if(nlhs == 2) {
        plhs[1] = mxCreateScalarDouble((double)rate);
        /* the second call adds 4 fp instructions to the total */
        accum_error += 4;
      }
    }
  }

  else {
    mexPrintf("Cannot find the command you specified.\n");
    mexErrMsgTxt("See the included readme file.");
  }
}
