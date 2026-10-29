# SYSDETECT Component

The `sysdetect` component allows PAPI users to query comprehensive system
information. The information is gathered at `PAPI_library_init()` time and
presented to the user through appropriate APIs. The component works
similarly to other components, which means that hardware information for
a specific device might not be available at runtime if, e.g., the device
runtime software is not installed.


* [Enabling the SYSDETECT Component](#enabling-the-sysdetect-component)
* [Supported Environment Variables](#supported-environment-variables)
* [Optionally Disabling the SYSDETECT Component](#optionally-disabling-the-sysdetect-component)
***

## Enabling the SYSDETECT Component

The `sysdetect` component is enabled by default, however, support for various
devices accessed by the component has to be enabled by exporting appropriate
environment variables at configure time (see [Supported Environment Variables](#supported-environment-variables)). This is required so that the component can `dlopen()` the needed libraries and access hardware information.

> [!NOTE]
>
> Recompilation of PAPI is required if the appropriate environment variable is not set
> before configure.

Typically, the utility `papi_component_avail` (available in
`papi/src/utils/papi_component_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

To verify the `sysdetect` component is functioning, the PAPI utility 
`papi/src/utils/papi_hardware_avail` can be run. Reported output includes
installed and configured hardware information.

## Supported Environment Variables
The table below shows the current set of supported environment variables for the
`sysdetect` component. Each of the listed environment variables should be exported
to the root of the ROCm or Cuda Toolkit installations in use as necessary headers
must be found.

| Environment Variable | Targeted Devices | Necessary Header | Notes | 
| :------------- | :------------- | :------------- | :------------- |
| `PAPI_CUDA_ROOT` | NVIDIA GPUs | `cuda.h` | To query NVIDIA device attributes  |
| `PAPI_NVML_ROOT` | NVIDIA GPUs | `nvml.h` | To query NVIDIA device affinity  |
| `PAPI_ROCP_SDK_ROOT` <br> (Supersedes `PAPI_ROCM_ROOT`) | AMD GPUs | `hsa.h` | To query AMD device attributes |
| `PAPI_AMDSMI_ROOT` <br> (Supersedes `PAPI_ROCMSMI_ROOT`) | AMD GPUs | `amdsmi.h` (`rocm_smi.h`) | To query AMD device affinity |

> [!NOTE]
>
> If a necessary header is missing, the ROCm or Cuda Toolkit installation in use has likely deprecated and removed it.
> To rememdy this, attempt to export the environment variable to a different installation and recompile PAPI.

## Optionally Disabling the SYSDETECT Component

The `sysdetect` component can be optionally disabled at configure time, to do so:
```sh
./configure --disable-sysdetect
```

Furthermore, if `--disable-cpu` or `--disable-perf-event` are provided at configure
time then the `sysdetect` component will be disabled as a consequence.