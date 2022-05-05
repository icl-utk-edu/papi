# CUDA Component

The CUDA component exposes counters and controls for NVIDIA GPUs.

* [Enabling the CUDA Component](#markdown-header-enabling-the-cuda-component)
* [Environment Variables](#markdown-header-environment-variables)
* [Known Limitations](#markdown-header-known-limitations)
* [FAQ](#markdown-header-faq)
***
## Enabling the CUDA Component

To enable reading or writing of CUDA counters the user needs to link against a
PAPI library that was configured with the CUDA component enabled. As an
example the following command: `./configure --with-components="cuda"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For CUDA, PAPI requires one environment variable: `PAPI_CUDA_ROOT`. This is
required for both compiling and runtime. 

Typically in Linux one would export this (examples are show below) variable but
some systems have software to manage environment variables (such as `module` or
`spack`), so consult with your sysadmin if you have such management software.
An example (this works on the ICL Saturn system):

    export PAPI_CUDA_ROOT=/usr/local/cuda-10.1

Within PAPI_CUDA_ROOT, we expect the following standard directories:

    PAPI_CUDA_ROOT/include
    PAPI_CUDA_ROOT/lib64
    PAPI_CUDA_ROOT/extras/CUPTI/include
    PAPI_CUDA_ROOT/extras/CUPTI/lib64

As of this writing (07/2021) Nvidia has overhauled performance reporting;
divided now into "Legacy CUpti" and "CUpti_11", the new approach. Legacy
Cupti works on devices up to Compute Capability 7.0; while only CUpti_11
works on devices with Compute Capability >=7.5. Both work on CC==7.0.

This component automatically distinguishes between the two; but it cannot
handle a "mix", one device that can only work with Legacy and another that
can only work with CUpti_11.

For the CUDA component to be operational, both versions require
the following dynamic libraries be found at runtime:

    libcuda.so
    libcudart.so
    libcupti.so

CUpti\_11 also requires:

    libnvperf_host.so

If those libraries cannot be found or some of those are stub libraries in the
standard `PAPI_CUDA_ROOT` subdirectories, you must add the correct paths,
e.g. `/usr/lib64` or `/usr/lib` to `LD_LIBRARY_PATH`, separated by colons `:`.
This can be set using export; e.g. 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WhereLib1CanBeFound:/WhereLib2CanBeFound

## Known Limitations
* In CUpti\_11, the number of possible events is vastly expanded; e.g. from
  some hundreds of events per device to over 110,000 events per device. this can
  make the utility papi/src/utils/papi\_native\_events run for several minutes;
  as much as 2 to 4 minutes per GPU. If the output is redirected to a file, this 
  may appear to "hang up". Give it time.

Note CUcontexts are device specific, a context can only apply to one GPU. So
each GPU must have its own context.

If PAPI events are being used, the CUcontexts to be used for each kernel+device must 
already exist and must be the most recent Current CUcontext on that device before 
PAPI\_add\_event() is invoked.

An example is given in papi/src/components/cuda/tests/simpleMultiGPU.cu.

First it executes cuCtxCreate() for each device and stores it in an array; e.g.
cuCtxCreate(&sessionCtx[deviceNum], 0, deviceNum); 

There are three main ways to change to a different context:
cuCtxSetCurrent(), cuCtxPushCurrent(), and cuCtxPopCurrent(). 

cuCtxPushCurrent() and cuCtxPopCurrent() are often used; these manage the
Nvidia driver context stack (not the application's code stack). There is an
example in simpleMultiGPU.cu when it sets up Nvidia Streams.

The CUcontext at the top of the Nvidia driver context stack is the "current"
context. So Push makes the context pushed the current context (and device if
the context was created on a different device), and Pop makes the context 
that WAS the top before the Push the new current context (and device, if that
changes for the new context).

However, we do not used Push and Pop right before PAPI_add_event. It is possible,
but we use cuCtxSetCurrent() because it is more efficient. cuCtxSetCurrent()
does NOT remember the previous context on the top of the stack, it just
replaces it with the context being set. However, we can use cuCtxGetCurrent()
to remember what context we are about to replace, and that is what we do, and
then restore it later.
 
Note that "cudaSetDevice(deviceNum)" will change the device number and the
context to that device's 'Primary' context ('Primary' is what Nvidia
documentation calls it).  Initially if no context has been created for that
device, a default context is created by cudaSetDevice(); but in our experience
these default contexts do not allow all the Profiler functionality of a context
explicitly created with cuCtxCreate().  Thus we recommend always using
cuCtxCreate() for each device; and ensuring the context used to run a kernel is
the most recent context active on that device when PAPI\_add\_event() is invoked.

Code details are in simpleMultiGpu.cu just before the PAPI_add_event invocation.

***

## FAQ

1. [Unusual installations](#markdown-header-unusual-installations)

## Unusual installations
Three libraries are required for the PAPI CUDA component. `libcuda.so`,
`libcudart.so' (The CUDA run-time library), and `libcupti.so`. For CUpti_11,
`libnvperf_host.so` is also necessary. 

For the CUDA component to be operational, it must find the dynamic libraries
mentioned above. If they are not found in the standard `PAPI_CUDA_ROOT`
subdirectories mentioned above, the component looks in the Linux default
directories listed by `/etc/ld.so.conf`, usually `/usr/lib64`, `/lib64`,
`/usr/lib` and `/lib`. 

The system will also search the directories listed in `LD_LIBRARY_PATH`,
separated by colons `:`. This can be set using export; e.g. 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WhereLib1CanBeFound:/WhereLib2CanBeFound

Finally, for very problematic installations, the `Rules.cuda` is invoked as
part of the `make` process and has an explanation of how to specify an explicit
arbitrary path for each of these libraries, including an alternative library
name. For example, if you wish to test a previous version of a library or a
private version, to test for a bug. See:

    papi/src/components/cuda/Rules.cuda
