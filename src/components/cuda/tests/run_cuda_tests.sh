#!/bin/bash

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
