/*
 *
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
#include <unistd.h>

#define SWIG
#include <perfmon/pfmlib.h>
#include <perfmon/perf_event.h>
#include <perfmon/pfmlib_perf_event.h>

static PyObject *libpfm_err;
typedef struct perf_event_attr perf_event_attr_t;
%}
%include "typemaps.i"
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

/* Generic return structures via pointer output arguments */
%define ptr_argout(T)
%typemap(argout) T* output {
    if (!PyTuple_Check($result)) {
      PyObject *x = $result;
      $result = PyTuple_New(1);
      PyTuple_SET_ITEM($result, 0, x);
    }
    PyObject *o = SWIG_NewPointerObj((void *)$1, $descriptor, 0);
    $result = SWIG_AppendOutput($result, o);
}

%typemap(in, numinputs=0) T* output (T temp) {
    $1 = (T*) malloc(sizeof(T));
    memset($1, 0, sizeof(T));
}

%extend T{
    ~T(){
        free(self);
    }
}
%enddef

ptr_argout(pfm_pmu_info_t);
ptr_argout(pfm_event_info_t);
ptr_argout(pfm_event_attr_info_t);
ptr_argout(perf_event_attr_t);

/* Kernel interface */
%include <perfmon/perf_event.h>
/* Library interface */
%include <perfmon/pfmlib.h>

extern pfm_err_t
pfm_get_perf_event_encoding(const char *str, int dfl_plm,
                            perf_event_attr_t *output, char **fstr, int *idx);

%typedef int pid_t;

extern os_err_t
perf_event_open(
    perf_event_attr_t        *hw_event_uptr,
    pid_t                    pid,
    int                      cpu,
    int                      group_fd,
    unsigned long            flags);

%init %{
  libpfm_err = PyErr_NewException("perfmon.libpfmError", NULL, NULL);
  PyDict_SetItemString(d, "libpfmError", libpfm_err);
%}
