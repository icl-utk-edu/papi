#include <papi.h>
#include <papi_test.h>
#include <hip/hip_runtime.h>

#define PAPI_CALL(call)                                                        \
do {                                                                           \
    int _status = call;                                                        \
    if (_status != PAPI_OK) {                                                  \
        test_fail(__FILE__, __LINE__, #call, _status);                         \
    }                                                                          \
} while (0)

#define HIP_CALL(call)                                                         \
do {                                                                           \
    hipError_t err = call;                                                     \
    if(err != hipSuccess) {                                                    \
        test_fail(__FILE__, __LINE__, hipGetErrorString(err), PAPI_EMISC);     \
    }                                                                          \
} while(0)

#define ROCTX_CALL(call)                                                       \
do {                                                                           \
    int _status = call;                                                        \
    if(_status != 0) {                                                         \
        test_fail(__FILE__, __LINE__, #call, _status);                         \
    }                                                                          \
} while(0)

__global__ void gemm(double *A, double *B, double *C, int N);
