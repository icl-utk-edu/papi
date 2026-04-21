#!/bin/bash
absolute_path_to_script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$absolute_path_to_script_dir"

# Add the libcupti shared object to LD_LIBRARY_PATH for the
# two concurrent_profiling tests
if [ -n "$PAPI_CUDA_CUPTI" ]; then
    directory_of_libcupti="$(dirname "$PAPI_CUDA_CUPTI")"
    export LD_LIBRARY_PATH="${directory_of_libcupti}:${LD_LIBRARY_PATH}"
elif [ -n "$PAPI_CUDA_ROOT" ]; then
    export LD_LIBRARY_PATH="${PAPI_CUDA_ROOT}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
fi

# Add -verbose to show output, i.e. sh run_sde_tests.sh -verbose.
if [ "$1" = "--suppress-output" ]; then
    export PAPI_CUDA_TEST_QUIET=1
fi

make_cuda_test_targets=(
    "test_multi_read_and_reset"
    "concurrent_profiling"
    "concurrent_profiling_noCuCtx"
    "pthreads"
    "pthreads_noCuCtx"
    "cudaOpenMP"
    "cudaOpenMP_noCuCtx"
    "test_multipass_event_fail"
    "test_2thr_1gpu_not_allowed"
    "HelloWorld"
    "HelloWorld_noCuCtx"
    "simpleMultiGPU"
    "simpleMultiGPU_noCuCtx"
)

for cuda_test in ${make_cuda_test_targets[@]}; do
    echo "make $cuda_test:"
    make $cuda_test

    printf "\n"

    echo "Running $cuda_test:"
    ./$cuda_test

    echo "-------------------------------------"
done
