#include "kernel.h"

extern "C" int launch_kernel(int device_id);

__global__ void
gemm(double *A, double *B, double *C, int N)
{
    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        for(int k = 0; k < N; ++k) {
            C[rowIdx*N + colIdx] += A[rowIdx*N + k] * B[k*N + colIdx];
        }
    }
}

__global__ void
kernelA(int x, int y)
{
    volatile int i, t;
    for(i=0; i<1000000; i++){
        t = 173/x;
        x = t + y;
    }
}

template <typename T>
__global__ void
kernelC(T* C_d, const T* A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = offset; i < N; i += stride)
    {
        C_d[i] = A_d[i] * A_d[i];
    }
}

int launch_kernel(int device_id) {
    const int NUM_LAUNCH = 1;

    HIP_CALL(hipSetDevice(device_id));

    for(int i = 0; i < NUM_LAUNCH; i++)
    {
        hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0, 1, 2);
    }

    HIP_CALL(hipDeviceSynchronize());

    return 0;
}
