# CUDA_RANGE Component
 The `cuda_range` component takes advantage of the CUPTI Profiler Host API to perform various host-side
 tasks necessary for collecting profiling data such as enumeration, configuration, and evaluation.
 Additionally, the Range Profiler API is utilized to gather the profiling data.

* [Enabling the CUDA_RANGE Component](#enabling-the-cuda_range-component)
* [Supported Architectures](#supported-architectures)
* [Multiple Pass Events](#multiple-pass-events)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)

## Enabling the CUDA_RANGE Component
To enable reading `cuda_range` native events the user needs to link against a PAPI library
that was configured with the `cuda_range` component i.e.,
```
./configure --with-components="cuda_range"
```
Furthermore, the `cuda_range` component must be **ACTIVE**. To ensure this, PAPI requires a single environment variable
to be set: `PAPI_CUDA_RANGE_ROOT`. As an example:
```
PAPI_CUDA_RANGE_ROOT=/packages/cuda/#.#.#
```

Within `PAPI_CUDA_RANGE_ROOT`, we expect the following standard directories for building:
```
PAPI_CUDA_RANGE_ROOT/include
PAPI_CUDA_RANGE_ROOT/extras/CUPTI/include
```

and for runtime:
```
PAPI_CUDA_RANGE_ROOT/lib64
PAPI_CUDA_RANGE_ROOT/extras/CUPTI/lib64
```

To verify the `cuda_range` component is active, run the PAPI utility
`utils/papi_component_avail` and parse the components under the "Active components" banner
for `cuda_range`:
```
Active components:
Name:   cuda_range              Profiling of NVIDIA GPU's via CUPTI Profiler Host and Cupti Range Profiling API's.
```

If the `cuda_range` component does not appear under the "Active components" banner then
it will appear under the "Compiled-in components" banner with a disabled reason:
```
Name:   cuda_range              Profiling of NVIDIA GPU's via CUPTI Profiler Host and Cupti Range Profiling API's.
   \-> Disabled: Unable to load the CUPTI API's. Try setting PAPI_CUDA_RANGE_ROOT or PAPI_CUDA_RANGE_CUPTI.
```
Remedy the disabled reason and the `cuda_range` component will then become active.

## Supported Architectures

To see the `cuda_range` component's latest supported hardware and software please visit the [Supported Architectures](https://github.com/icl-utk-edu/papi/wiki/Supported-Architectures#nvidia) GitHub Wiki page.

## Multiple Pass Events 
The `cuda_range` component supports event's that require multiple passes to be submitted to the GPU for collection. Although the `cuda_range` component supports multiple pass events, they are not supported by default and attempting to add one to a PAPI eventset results in the error code `PAPI_EMULPASS` being returned. Therefore, to signal the desire to profile with multiple pass events and to avoid `PAPI_EMULPASS` being returned the environment variable `PAPI_CUDA_RANGE_ENABLE_MULTIPASSES` must be set:
```
export PAPI_CUDA_RANGE_ENABLE_MULTIPASSES=1
setenv("PAPI_CUDA_RANGE_ENABLE_MULTIPASSES", 1, 1)
```

To see the available `cuda_range` multiple pass events simply run `utils/papi_native_avail` and parse the output descriptions for `PAPI_CUDA_RANGE_ENABLE_MULTIPASSES`:
```
--------------------------------------------------------------------------------
| cuda_range:::l1tex__throughput
|            L1TEX Throughput. The number of passes required for collection is 3.
|            To profile, set the environment variable PAPI_CUDA_RANGE_ENABLE_MULTIPASSES
|            equal to 1.
```

Furthermore, in application code you can query the required number of passes for collection for a `cuda_range` native event through `PAPI_get_event_info`:
```c
int component_index = PAPI_get_component_index("cuda_range");
int modifier = PAPI_ENUM_FIRST;
int eventcode = 0 | PAPI_NATIVE_MASK;
PAPI_enum_cmp_event(&eventcode, modifier, component_index);

PAPI_event_info_t event_info;
PAPI_get_event_info(eventcode, &event_info);
printf("Number of passes: %d\n", event_info.num_passes_for_collection);
```

To see an example of how to profile `cuda_range` multiple pass events, please see `components/cuda_range/tests/test_cuda_range_multiple_pass_events.cu`.

Lastly, if a PAPI eventset is created with both a single pass and a mutiple pass `cuda_range` native event then the eventset will require multiple passes for collection.

## Known Limitations
* Cuda Toolkit 13.0.0 removed support for offline compilation of the NVIDIA GPU architectures with compute capabilities <= 7.5 (i.e. P100 and V100).

* The CUPTI Profiler Host API exposes 1,000s of CUPTI metrics. Due to this, the CUPTI Profiler Host workflow to enumerate metrics and obtain metric descriptions will take several minutes to complete and affect the runtime of the PAPI utility `utils/papi_native_avail`. For instance, on a system with a single NVIDIA GH100 the runtime of `utils/papi_native_avail` is roughly 3 minutes to completion.

# FAQ

## Unusual Installations
If the dynamic shared objects `libcupti` and `libcudart` cannot be found by setting `PAPI_CUDA_RANGE_ROOT` then two other options remain to find them:

1. Setting the dynamic shared objects corresponding environment variable:
   ```
   export PAPI_CUDA_RANGE_CUPTI=/your/path/to/libcupti.so
   export PAPI_CUDA_RANGE_RUNTIME=/your/path/to/libcudart.so
   ```

   Note, that if using this option:
     * You must set the enviornment variable directly to the dynamic shared object as shown above.
     * If the set path fails to open a dynamic shared object, the `cuda_range` component will be disabled.

2. Using `dlopen` and following the search logic used by the dynamic linker. For this option, it is advised to set `LD_LIBRARY_PATH` to the directories containing your `libcupti` and `libcudart` dynamic shared objects, i.e.
   ```
   export LD_LIBRARY_PATH=/your/path/to/WhereLib1CanBeFound:/your/path/to/WhereLib2CanBeFound:$LD_LIBRARY_PATH
   ```

   Note, that if using this option:
     * Make sure to separate the dynamic shared objects by a colon (`:`).
     * This option serves as a final fallback if both `PAPI_CUDA_RANGE_ROOT` or the dynamic shared objects corresponding environment variable are unable to load the respective dynamic shared objects (`libcupti`, `libcudart`).

For the dynamic shared object `libcuda`, it is commonly found in `/lib` or `/usr/lib` rather than the Cuda Toolkit installation. Therefore, `dlopen` is used to search `/lib` and `/usr/lib` by default. If not found, the logic for option 1 and option 2 listed above still hold. In the case of option 1, set the corresponding environment variable:
```
export PAPI_CUDA_RANGE_DRIVER=/your/path/to/libcuda.so
```

## CUDA Contexts
For each `cuda_range` native event added to a PAPI EventSet a CUDA context is required.
One can be created in the application code using `cuCtxCreate` or `cudaSetDevice`; however, if a context is not present on the calling CPU thread then the `cuda_range` component will create one based on the `#` from the appended device qualifier (`:device=#`). See `cuda_range/tests` for examples.


## CUDA Toolkit Versions
Once your binaries are compiled, it is possible to swap to other Cuda Toolkit versions without needing to recompile the source. Simply update `PAPI_CUDA_RANGE_ROOT` to the location of the newly desired Cuda Toolkit version. As a note, `LD_LIBRARY_PATH` may need to be updated as well.
