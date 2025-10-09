/**
 * @file    amdsmi_gemm.c
 * @author  Dong Jun Woun
 *          djwoun@gmail.com
 * @brief   Launches a large HIP DGEMM workload (on device 1) while sampling a
 *          small set of AMD SMI counters via PAPI.
 *
 * The monitor thread polls the PAPI EventSet ~3 times per second while the kernel runs.
 * This is intended for simple integration/soak testing rather than performance tuning.
 *
 * If no events are specified, we enumerate the AMD-SMI component and add the
 * first five usable native events (skipping ones that cannot be added).
 */

#include "test_harness.h"

#include "papi.h"
#include "hip/hip_runtime.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/* ----------------------------- Configuration ----------------------------- */

#define M_DIM 7296
#define K_DIM 14592
#define N_DIM 7296

#define NUM_STREAMS 1
#define ITERATIONS_PER_STREAM 1
#define MAX_EVENTS 5

/* --------------------------- HIP error helpers --------------------------- */

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "Failed: HIP error %s:%d '%s' (code: %d)\n", \
                __FILE__, __LINE__, hipGetErrorString(e), e); \
        return 1; \
    } \
} while(0)

#define HIP_CHECK_CLEANUP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "Warning: HIP cleanup error %s:%d '%s' (code: %d)\n", \
                __FILE__, __LINE__, hipGetErrorString(e), e); \
    } \
} while(0)

/* --------------------------- Monitoring thread --------------------------- */

/**
 * @brief Background poller for PAPI EventSet values.
 *
 * If params->print is 1, it writes one line per sample to stdout with a timestamp.
 */
static volatile int stop_monitor = 0;

struct monitor_params {
    int EventSet;
    struct timeval start_time;
    int print; // 0/1: whether to print readings (controls stdout chatter)
};

static void *monitor_events(void *args) {
    struct monitor_params *params = (struct monitor_params *)args;
    int statusFlag;
    long long values[MAX_EVENTS] = {0};

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

        if (params->print) {
            fprintf(stdout,
                    "Time: %.6f sec -> e1: %lld, e2: %lld, e3: %lld, e4: %lld, e5: %lld\n",
                    elapsed, values[0], values[1], values[2], values[3], values[4]);
            fflush(stdout);
        }

        usleep(300000); // ~3 Hz
    }
    return NULL;
}

/* ------------------------------- Workload -------------------------------- */

/**
 * @brief Naive DGEMM: C = alpha * A * B + beta * C
 *        A: MxK, B: KxN, C: MxN (row-major)
 */
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

/* ------------------------------- Test body -------------------------------- */

static int real_main(const HarnessOpts *opts) {
    /* Gracefully skip if the PAPI AMD SMI component isn't available. */
    const char* root = getenv("PAPI_AMDSMI_ROOT");
    if (!root || !*root) {
        SKIP("PAPI_AMDSMI_ROOT not set");
    }

    /* Initialize PAPI */
    int statusFlag = PAPI_library_init(PAPI_VER_CURRENT);
    if (statusFlag != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI shared library version error: %s\n", PAPI_strerror(statusFlag));
        return 1;
    }

    /* Create EventSet */
    int EventSet = PAPI_NULL;
    statusFlag = PAPI_create_eventset(&EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset: %s\n", PAPI_strerror(statusFlag));
        return 1;
    }

    /* ------------------ Enumerate first five usable AMD-SMI events ------------------ */
    int cid = -1;
    const int ncomps = PAPI_num_components();
    for (int i = 0; i < ncomps && cid < 0; ++i) {
        const PAPI_component_info_t *cinfo = PAPI_get_component_info(i);
        if (cinfo && strcmp(cinfo->name, "amd_smi") == 0) cid = i;
    }
    if (cid < 0) {
        SKIP("Unable to locate the amd_smi component (PAPI built without ROCm?)");
    }

    char chosen_names[MAX_EVENTS][PAPI_MAX_STR_LEN] = {{0}};
    int added = 0;
    int code = PAPI_NATIVE_MASK;

    if (PAPI_enum_cmp_event(&code, PAPI_ENUM_FIRST, cid) != PAPI_OK) {
        SKIP("No native events found for AMD-SMI component");
    }

    do {
        char base_name[PAPI_MAX_STR_LEN] = {0};
        if (PAPI_event_code_to_name(code, base_name) != PAPI_OK || base_name[0] == '\0') {
            continue;
        }

        int qualified_code = 0;
        if (PAPI_event_name_to_code(base_name, &qualified_code) != PAPI_OK) {
            continue;
        }

        char name[PAPI_MAX_STR_LEN] = {0};
        if (PAPI_event_code_to_name(qualified_code, name) != PAPI_OK || name[0] == '\0') {
            continue; /* couldn't resolve name; try next */
        }

        /* Skip process* events as in amdsmi_basics (not testable in this harness). */
        if (strncmp(name, "amd_smi:::process", 17) == 0 || strncmp(name, "process", 7) == 0) {
            continue;
        }

        statusFlag = PAPI_add_event(EventSet, qualified_code);
        if (statusFlag == PAPI_OK) {
            strncpy(chosen_names[added], name, PAPI_MAX_STR_LEN - 1);
            ++added;
        } else if (statusFlag == PAPI_ENOEVNT || statusFlag == PAPI_ECNFLCT || statusFlag == PAPI_EPERM) {
            /* Not usable (missing or HW/resource-limited) — keep enumerating. */
            continue;
        } else {
            /* Unexpected error; keep going to try to fill five. */
            continue;
        }
    } while (added < MAX_EVENTS && PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, cid) == PAPI_OK);

    if (added < MAX_EVENTS) {
        SKIP("Fewer than five usable AMD-SMI events available");
    }

    if (opts->print) {
        printf("Monitoring events:\n");
        for (int i = 0; i < MAX_EVENTS; ++i) {
            printf("  %d) %s\n", i + 1, chosen_names[i]);
        }
    }

    /* HIP runtime preflight so HIP_CHECK won't hard-exit. */
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 1) {
        SKIP("HIP device 1 not available");
    }

    /* Use device 1 and (optionally) print basic properties. */
    HIP_CHECK(hipSetDevice(1));
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, 1));
    if (opts->print) {
        printf("Device Name: %s\n", deviceProp.name);
        printf("Compute Units: %d\n", deviceProp.multiProcessorCount);
        printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
    }

    /* Host buffers (pinned) */
    size_t size_A = ((size_t)M_DIM * K_DIM * sizeof(double));
    size_t size_B = ((size_t)K_DIM * N_DIM * sizeof(double));
    size_t size_C = ((size_t)M_DIM * N_DIM * sizeof(double));

    double *h_A = NULL, *h_B = NULL, *h_C = NULL;
    HIP_CHECK(hipHostMalloc(&h_A, size_A, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_B, size_B, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_C, size_C, hipHostMallocDefault));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        if (h_A) HIP_CHECK_CLEANUP(hipHostFree(h_A));
        if (h_B) HIP_CHECK_CLEANUP(hipHostFree(h_B));
        if (h_C) HIP_CHECK_CLEANUP(hipHostFree(h_C));
        return 1;
    }

    for (int i = 0; i < M_DIM * K_DIM; i++) h_A[i] = (double)(i % 100);
    for (int i = 0; i < K_DIM * N_DIM; i++) h_B[i] = (double)(i % 100);
    for (int i = 0; i < M_DIM * N_DIM; i++) h_C[i] = 0.0;

    /* Device buffers per stream */
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

    /* H2D copies */
    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipMemcpyAsync(d_A[s], h_A, size_A, hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_B[s], h_B, size_B, hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_C[s], h_C, size_C, hipMemcpyHostToDevice, streams[s]));
    }

    /* Start counters */
    statusFlag = PAPI_start(EventSet);
    if (statusFlag == PAPI_ECNFLCT || statusFlag == PAPI_EPERM) {
        SKIP("Cannot start counters due to HW/resource limits");
    } else if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI_start: %s\n", PAPI_strerror(statusFlag));
        return 1;
    }

    /* Launch monitor thread (prints unless suppressed) */
    pthread_t monitor_thread;
    struct monitor_params params;
    params.EventSet = EventSet;
    params.print    = opts->print ? 1 : 0;
    gettimeofday(&params.start_time, NULL);
    statusFlag = pthread_create(&monitor_thread, NULL, monitor_events, &params);
    if (statusFlag != 0) {
        fprintf(stderr, "pthread_create failed\n");
        return 1;
    }

    /* Ensure copies are done */
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
            usleep(3000000); // Allow the monitor to capture a few samples
        }
    }

    /* Stop the monitor and clean up */
    stop_monitor = 1;
    pthread_join(monitor_thread, NULL);

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

    long long stop_values[MAX_EVENTS] = {0};  // five events were added
    statusFlag = PAPI_stop(EventSet, stop_values);
    if (statusFlag != PAPI_OK) { fprintf(stderr, "PAPI_stop: %s\n", PAPI_strerror(statusFlag)); return 1; }
    statusFlag = PAPI_cleanup_eventset(EventSet);
    if (statusFlag != PAPI_OK) { fprintf(stderr, "PAPI_cleanup_eventset: %s\n", PAPI_strerror(statusFlag)); return 1; }
    statusFlag = PAPI_destroy_eventset(&EventSet);
    if (statusFlag != PAPI_OK) { fprintf(stderr, "PAPI_destroy_eventset: %s\n", PAPI_strerror(statusFlag)); return 1; }

    HIP_CHECK_CLEANUP(hipDeviceReset());   // Optional; reduces "still reachable" reports from HIP in leak checkers
    PAPI_shutdown();                       // Triggers component cleanup and AMD SMI shutdown
    return 0;
}

/* --------------------------- Test harness glue --------------------------- */

int main(int argc, char *argv[]) {
    harness_accept_tests_quiet(&argc, argv);
    HarnessOpts opts = parse_harness_cli(argc, argv);
    int papi_errno = real_main(&opts);
    return eval_result(opts, papi_errno);
}
