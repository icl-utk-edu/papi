#!/bin/bash
absolute_path_to_script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$absolute_path_to_script_dir"

if [ -n "$PAPI_CUDA_RANGE_ROOT" ]; then
    export LD_LIBRARY_PATH="${PAPI_CUDA_RANGE_ROOT}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
fi

make_cuda_range_test_targets=(
    "test_cuda_range_floating_point_operations"
    "test_cuda_range_hello_world"
    "test_cuda_range_2thr_1gpu_not_allowed"
    "test_cuda_range_no_user_context"
    "test_cuda_range_pthreads"
    "test_cuda_range_multiple_pass_events_succeed"
    "test_cuda_range_multiple_pass_events_fail"
)

for cuda_range_test in ${make_cuda_range_test_targets[@]}; do
    echo "make $cuda_range_test:"
    make $cuda_range_test

    printf "\n"

    echo "Running $cuda_range_test:"
    ./$cuda_range_test

    echo "-------------------------------------"
done
