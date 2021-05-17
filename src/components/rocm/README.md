# ROCM Component

The ROCM component exposes numerous performance events on AMD GPUs.
The component is an adapter to the ROCm profiling library (ROC-profiler) which is included in a standard ROCM release.


* [Enabling the ROCM Component](#markdown-header-enabling-the-rocm-component)
* [Environment Variables](#markdown-header-environment-variables)
* [Known Limitations](#markdown-header-known-limitations)
* [FAQ](#markdown-header-faq)
***
## Enabling the ROCM Component

To enable reading ROCM events the user needs to link against a PAPI library that was configured with the ROCM component enabled. As an example the following command: `./configure --with-components="rocm"` is sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in `papi/src/utils/papi_components_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For ROCM, PAPI requires the location of the ROCM install directory. This can be
specified by one environment variable: **PAPI\_ROCM\_ROOT**.

If it is NOT specified, PAPI will attempt to find the ROCM install directory by
examining\ $LD_LIBRARY\_PATH, if there are 'rocm' library paths in it, PAPI may
automatically find the path. (LD\_LIBRARY\_PATH is typically modified by
systems that use the 'module load' commands to set up). 

PAPI will also look for some common environment variables; including
"ROCM\_PATH", "ROCM\_DIR", and "ROCMDIR".

However, if PAPI cannot automatically find the rocm include and library
directories, the user needs to explictly export PAPI\_ROCM\_ROOT.

Access to the rocm main directory is required at both compile (for include
files) and at runtime (for libraries).

Example:

    export PAPI_ROCM_ROOT=/opt/rocm

Within PAPI\_ROCM\_ROOT, we expect the following standard directories:

    PAPI_ROCM_ROOT/include
    PAPI_ROCM_ROOT/include/hsa
    PAPI_ROCM_ROOT/lib
    PAPI_ROCM_ROOT/rocprofiler/lib
    PAPI_ROCM_ROOT/rocprofiler/include

Besides the PAPI\_ROCM\_ROOT environment variable, five more environment
variables are required at runtime. These are not needed by PAPI, but by the AMD
ROCPROFILER software we interface with. 

If these are not set at runtime, the PAPI will automatically export them (they 
will vanish when PAPI exits). 

These added environment variables are typically set as follows, after
PAPI\_ROCM\_ROOT has been exported. An example is provided below:

    export ROCP_METRICS=$PAPI_ROCM_ROOT/rocprofiler/lib/metrics.xml
    export ROCPROFILER_LOG=1
    export HSA_VEN_AMD_AQLPROFILE_LOG=1
    export AQLPROFILE_READ_API=1
    export HSA_TOOLS_LIB=librocprofiler64.so

The first of these, ROCP\_METRICS, must point at a file containing the
descriptions of metrics. The standard location is shown above, the final four
exports are fixed settings.
    
If the user relies on PAPI to export these, then it is important to execute the
function PAPI\_library\_init() in the user code BEFORE any HIP functions are
executed. Because these values are read once by AMD with the first HIP function
call, and if HIP sets up without them, PAPI may not read counters correctly.

If the user code cannot be changed, then export the variables explicitly before
execution.

## Known Limitations

* PAPI may read zeros for many events if rocprofiler environment variables are
  not exported and HIP functions are executed by the user before the user
  executes PAPI\_library\_init(). 

* Only sets of metrics and events that can be gathered in a single pass are supported.

* Although AMD metrics may be floating point, all values are recast and returned as long long integers.

    The binary image of a `double` is intact; but users must recast to `double` for display purposes.

***
## FAQ

1. [Unusual installations](#markdown-header-unusual-installations)

## Unusual installations
For the ROCM component to be operational, it must find the dynamic libraries `libhsa-runtime64.so` and `librocprofiler64.so`. These are normally found in the above standard directories, or one of the Linux default directories listed by `/etc/ld.so.conf`, usually `/usr/lib64`, `/lib64`, `/usr/lib` and `/lib`. If these libraries are not found (or are not functional) then the component will be listed as "disabled" with a reason explaining the problem. If libraries were not found, then they are not in the expected places. 

The system will search the directories listed in **LD\_LIBRARY\_PATH**. You can add an additional path with a colon e.g. 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WhereALibraryCanBeFound
