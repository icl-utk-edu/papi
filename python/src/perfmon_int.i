/*

 * Copyright (c) 2008 Google, Inc.
 * Contributed by Arun Sharma <arun.sharma@google.com>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * Python Bindings for perfmon.
 */
%module perfmon_int
%{
#include <perfmon/pfmlib_perf_event.h>
#include <perfmon/pfmlib.h>

static PyObject *libpfm_err;
%}
%include "carrays.i"
%include "cstring.i"
%include <stdint.i>

/* Convert libpfm errors into exceptions */
%typemap(out) os_err_t {  
  if (result == -1) {
    PyErr_SetFromErrno(PyExc_OSError);
    SWIG_fail;
  } 
  resultobj = SWIG_From_int((int)(result));
};

%typemap(out) pfm_err_t {  
  if (result != PFM_SUCCESS) {
    PyObject *obj = Py_BuildValue("(i,s)", result, 
                                  pfm_strerror(result));
    PyErr_SetObject(libpfm_err, obj);
    SWIG_fail;
  } else {
    PyErr_Clear();
  }
  resultobj = SWIG_From_int((int)(result));
}

/* Kernel interface */
%include <perfmon/pfmlib_perf_event.h>
/* Library interface */
%include <perfmon/pfmlib.h>
