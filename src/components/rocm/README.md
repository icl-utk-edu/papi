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

For ROCM, PAPI requires one environment variable: **PAPI\_ROCM\_ROOT**.

This is required for both compiling, and at runtime. 

Example:

    export PAPI_ROCM_ROOT=/opt/rocm

Within PAPI\_ROCM\_ROOT, we expect the following standard directories:

    PAPI_ROCM_ROOT/include
    PAPI_ROCM_ROOT/include/hsa
    PAPI_ROCM_ROOT/lib
    PAPI_ROCM_ROOT/rocprofiler/lib
    PAPI_ROCM_ROOT/rocprofiler/include

Besides the PAPI\_ROCM\_ROOT environment variable, four more environment variables are required at runtime. These are not needed by PAPI, but by the AMD ROCPROFILER software we interface with. These added environment variables are typically set as follows, after PAPI\_ROCM\_ROOT has been exported. An example is provided below:

    export ROCP_METRICS=$PAPI_ROCM_ROOT/rocprofiler/lib/metrics.xml
    export ROCPROFILER_LOG=1
    export HSA_VEN_AMD_AQLPROFILE_LOG=1
    export AQLPROFILE_READ_API=1
    export HSA_TOOLS_LIB=librocprofiler64.60

The first of these, ROCP\_METRICS, must point at a file containing the descriptions of metrics. The standard location is shown above, the final three exports are fixed settings.
    

## Known Limitations

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
