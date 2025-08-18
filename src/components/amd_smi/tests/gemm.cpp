#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "hip/hip_runtime.h"
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

#define M_DIM 7296
#define K_DIM 14592
#define N_DIM 7296

#define NUM_STREAMS 1
#define ITERATIONS_PER_STREAM 1

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "Failed: HIP error %s:%d '%s' (code: %d)\n", __FILE__, __LINE__, hipGetErrorString(e), e); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIP_CHECK_CLEANUP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "Warning: HIP cleanup error %s:%d '%s' (code: %d)\n", __FILE__, __LINE__, hipGetErrorString(e), e); \
    } \
} while(0)

volatile int stop_monitor = 0;

struct monitor_params {
    int EventSet;
    FILE *csvFile;
    struct timeval start_time;
};

void *monitor_events(void *args) {
    struct monitor_params *params = (struct monitor_params *)args;
    int statusFlag;
    long long values[5];

    while (!stop_monitor) {
        statusFlag = PAPI_read(params->EventSet, values);
        if (statusFlag != PAPI_OK) {
            fprintf(stderr, "PAPI read failed in monitor: %s\n", PAPI_strerror(statusFlag));
            break;
        }

        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - params->start_time.tv_sec) +
                         (current_time.tv_usec - params->start_time.tv_usec) / 1e6;

        fprintf(params->csvFile, "%.6f,%lld,%lld,%lld,%lld,%lld\n",
                elapsed, values[0], values[1], values[2], values[3], values[4]);
        fflush(params->csvFile);

        fprintf(stdout,
                "Time: %.6f sec -> event1: %lld, event2: %lld, event3: %lld, event4: %lld, event5: %lld\n",
                elapsed, values[0], values[1], values[2], values[3], values[4]);

        usleep(300000);
    }
    return NULL;
}

__global__ void dgemm_kernel(const double *A, const double *B, double *C,
                             int M, int N, int K, double alpha, double beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

int main(int argc, char *argv[]) {
    int statusFlag;
    int EventSet = PAPI_NULL;

    statusFlag = PAPI_library_init(PAPI_VER_CURRENT);
    if (statusFlag != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI shared library version error: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    statusFlag = PAPI_create_eventset(&EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    const char *event1 = "amd_smi:::mem_total_VRAM:device=0";
    const char *event2 = "amd_smi:::temp_current:device=0:sensor=1";
    const char *event3 = "amd_smi:::temp_current:device=0:sensor=2";
    const char *event4 = "amd_smi:::clk_freq_current:device=0";
    const char *event5 = "amd_smi:::power_average:device=0";

    statusFlag = PAPI_add_named_event(EventSet, event1);
    if (statusFlag != PAPI_OK) return -1;
    statusFlag = PAPI_add_named_event(EventSet, event2);
    if (statusFlag != PAPI_OK) return -1;
    statusFlag = PAPI_add_named_event(EventSet, event3);
    if (statusFlag != PAPI_OK) return -1;
    statusFlag = PAPI_add_named_event(EventSet, event4);
    if (statusFlag != PAPI_OK) return -1;
    statusFlag = PAPI_add_named_event(EventSet, event5);
    if (statusFlag != PAPI_OK) return -1;

    HIP_CHECK(hipSetDevice(1));
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, 1));
    printf("Device Name: %s\n", deviceProp.name);
    printf("Compute Units: %d\n", deviceProp.multiProcessorCount);
    printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);

    size_t size_A = ((size_t)M_DIM * K_DIM * sizeof(double));
    size_t size_B = ((size_t)K_DIM * N_DIM * sizeof(double));
    size_t size_C = ((size_t)M_DIM * N_DIM * sizeof(double));

    double *h_A, *h_B, *h_C;
    HIP_CHECK(hipHostMalloc(&h_A, size_A, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_B, size_B, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_C, size_C, hipHostMallocDefault));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        if (h_A) HIP_CHECK_CLEANUP(hipHostFree(h_A));
        if (h_B) HIP_CHECK_CLEANUP(hipHostFree(h_B));
        if (h_C) HIP_CHECK_CLEANUP(hipHostFree(h_C));
        return -1;
    }

    for (int i = 0; i < M_DIM * K_DIM; i++) h_A[i] = (double)(i % 100);
    for (int i = 0; i < K_DIM * N_DIM; i++) h_B[i] = (double)(i % 100);
    for (int i = 0; i < M_DIM * N_DIM; i++) h_C[i] = 0.0;

    double *d_A[NUM_STREAMS], *d_B[NUM_STREAMS], *d_C[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipMalloc((void**)&d_A[s], size_A));
        HIP_CHECK(hipMalloc((void**)&d_B[s], size_B));
        HIP_CHECK(hipMalloc((void**)&d_C[s], size_C));
    }

    hipStream_t streams[NUM_STREAMS];
    hipEvent_t events[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking));
        HIP_CHECK(hipEventCreate(&events[s]));
    }

    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipMemcpyAsync(d_A[s], h_A, size_A, hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_B[s], h_B, size_B, hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_C[s], h_C, size_C, hipMemcpyHostToDevice, streams[s]));
    }

    FILE *csvFile = fopen("test.csv", "w");
    if (!csvFile) {
        fprintf(stderr, "Failed to open CSV file for writing.\n");
        return -1;
    }
    fprintf(csvFile, "timestamp,%s,%s,%s,%s,%s\n",
        event1, event2, event3, event4, event5);

    statusFlag = PAPI_start(EventSet);
    if (statusFlag != PAPI_OK) return -1;

    pthread_t monitor_thread;
    struct monitor_params params;
    params.EventSet = EventSet;
    params.csvFile = csvFile;
    gettimeofday(&params.start_time, NULL);
    statusFlag = pthread_create(&monitor_thread, NULL, monitor_events, &params);
    if (statusFlag != 0) return -1;

    for (int s = 0; s < NUM_STREAMS; s++) HIP_CHECK(hipStreamSynchronize(streams[s]));

    double alpha = 0.75;
    double beta  = 0.5;

    dim3 blockDim(32, 32);
    dim3 gridDim((N_DIM + blockDim.x - 1) / blockDim.x,
                 (M_DIM + blockDim.y - 1) / blockDim.y);

    for (int iter = 0; iter < ITERATIONS_PER_STREAM; iter++) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            hipLaunchKernelGGL(dgemm_kernel, gridDim, blockDim, 0, streams[s],
                               d_A[s], d_B[s], d_C[s],
                               M_DIM, N_DIM, K_DIM, alpha, beta);
            HIP_CHECK(hipEventRecord(events[s], streams[s]));
            HIP_CHECK(hipStreamSynchronize(streams[s]));
            usleep(3000000);
        }
    }

    stop_monitor = 1;
    pthread_join(monitor_thread, NULL);

    fclose(csvFile);
    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK_CLEANUP(hipEventDestroy(events[s]));
        HIP_CHECK_CLEANUP(hipStreamDestroy(streams[s]));
        HIP_CHECK_CLEANUP(hipFree(d_A[s]));
        HIP_CHECK_CLEANUP(hipFree(d_B[s]));
        HIP_CHECK_CLEANUP(hipFree(d_C[s]));
    }

    HIP_CHECK_CLEANUP(hipHostFree(h_A));
    HIP_CHECK_CLEANUP(hipHostFree(h_B));
    HIP_CHECK_CLEANUP(hipHostFree(h_C));

    statusFlag = PAPI_stop(EventSet, NULL);
    if (statusFlag != PAPI_OK) return -1;
    statusFlag = PAPI_cleanup_eventset(EventSet);
    if (statusFlag != PAPI_OK) return -1;
    statusFlag = PAPI_destroy_eventset(&EventSet);
    if (statusFlag != PAPI_OK) return -1;

    return 0;
}
