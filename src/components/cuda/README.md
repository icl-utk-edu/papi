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

For the CUDA component to be operational at runtime, it must find the following dynamic libraries:

    libcuda.so
    libcudart.so
    libcupti.so

If those libraries cannot be found or some of those are stub libraries in the standard `PAPI_CUDA_ROOT` subdirectories, you have to add the correct paths, e.g. `/usr/lib64` or `/usr/lib` to `LD_LIBRARY_PATH`, separated by colons `:`. This can be set using export; e.g. 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WhereLib1CanBeFound:/WhereLib2CanBeFound

## Known Limitations

* NVIDIA libraries now regard Compute Capability (CC) < 7.0 as 'legacy', and
going forward CC>=7.0 is 'cupti 11' code. These are completely different
interfaces; the cupti 11 uses the PerfWorks profiler, the legacy code does not.
This component automatically distinguishes between the two.

HOWEVER, with Cupti 11, users calling PAPI must have created a valid cuda context
for each device (GPU), and they must make those cuda contexts active by 
pushing them on the GPU stack when they are adding PAPI_events for that GPU 
to the PAPI eventset.

An example is given in papi/src/components/cuda/tests/simpleMultiGPU.cu.
Note that Cuda Contexts are device specific. The relevant calls are:

CUcontext ctx[maxDevices]; // Space to create a context for each GPU.

Execute cuCtxCreate for each device, note it is pushed on the internal Nvidia stack.
cuCtxCreate(&sessionCtx[deviceNum], 0, deviceNum); 

Use cuCtxPushCurrent to switch to a created context, this is not pushed on the 
software stack but an internal Nvidia driver stack. Note it will automatically
also make the relevant device for that context the current device.
cuCtxPushCurrent(ctx[deviceNum]);

use cuCtxPopcurrent to undo a Push, it pops it off the internal Nvidia stack; and
restores whatever the previous context may be, also making that context's device 
the current device.
cuCtxPopCurrent(&(ctx[deviceNum]);

Note that "cudaSetDevice(deviceNum)" will change the device number and the context
to that device's 'Primary' context ('Primary' is what Nvidia documentation calls it).
Basically a default context. But in our experience Primary contexts do not allow all
the Profiler functionality of a created ("non-Primary") context. We recommend always
using a created context; and in particular if you are collecting performance events
for a kernel, then when adding PAPI events use the same created context you will
subsequently use to execute the kernel.

***

## FAQ

1. [Unusual installations](#markdown-header-unusual-installations)

## Unusual installations
Three libraries are required for the PAPI CUDA component. `libcuda.so`,
`libcudart.so' (The CUDA run-time library), and `libcupti.so`.  

For the CUDA component to be operational, it must find the dynamic libraries
mentioned above. If they are not found in the standard `PAPI_CUDA_ROOT`
subdirectories mentioned above, the component looks in the Linux default
directories listed by `/etc/ld.so.conf`, usually `/usr/lib64`, `/lib64`,
`/usr/lib` and `/lib`. 

The system will also search the directories listed in `LD_LIBRARY_PATH`,
separated by colons `:`. This can be set using export; e.g. 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WhereLib1CanBeFound:/WhereLib2CanBeFound
