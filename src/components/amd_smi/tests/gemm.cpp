#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "hip/hip_runtime.h"
//#include <rocblas.h>   // No longer needed since we use a custom kernel
#include <unistd.h>      // For usleep()
#include <sys/time.h>    // For gettimeofday()
#include <pthread.h>     // For pthreads

#define M_DIM 7296 //14592  
#define K_DIM 14592//65536  
#define N_DIM 7296 //14592  //14592

// Number of streams to use for concurrent execution
#define NUM_STREAMS 1

// Number of iterations to run in each stream
#define ITERATIONS_PER_STREAM 1

// Global flag to signal the monitor thread to stop.
volatile int stop_monitor = 0;

// Structure to pass parameters to the monitoring thread.
struct monitor_params {
    int EventSet;
    FILE *csvFile;
    struct timeval start_time;
};

// Monitor thread that periodically reads PAPI counters and logs them.
void *monitor_events(void *args) {
    struct monitor_params *params = (struct monitor_params *)args;
    int statusFlag;
    long long values[5];

    // Continue monitoring until stop_monitor is set.
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


        int gpu1_power = -1; // Default to -1 (error/unavailable)
        FILE *fp = popen("amd-smi metric -g 1 -p --csv", "r"); // Use specific command
        if (fp != NULL) {
            char buffer[128]; // Sufficient buffer for the expected output
            int header_skipped = 0;
            int data_parsed = 0;

            while (fgets(buffer, sizeof(buffer), fp) != NULL) {
                // Skip the header line (contains "gpu")
                if (!header_skipped && strstr(buffer, "gpu")) {
                    header_skipped = 1;
                    continue;
                }
                // Parse the data line (after header)
                if (header_skipped) {
                    int gpu_id_read;
                    // Expect format like "1,83,..." - parse first two ints
                    if (sscanf(buffer, "%d,%d", &gpu_id_read, &gpu1_power) == 2) {
                        data_parsed = 1; // Flag success
                        break;          // Got the data, no need to read further
                    } else {
                        // Failed to parse data line, treat as error for this sample
                        gpu1_power = -1;
                        break;
                    }
                }
            }

            // Check if data was actually parsed after skipping header
            if (header_skipped && !data_parsed) {
                gpu1_power = -1; // Header found, but data parsing failed/missing
            } else if (!header_skipped) {
                 gpu1_power = -1; // Header wasn't even found
            }


            int status = pclose(fp);
            // If command failed execution, ensure power is marked as error
            if (status == -1 || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
                 if (gpu1_power != -1) { // Only print warning if we previously thought we succeeded
                      // Optional: fprintf(stderr, "Warning: amd-smi command failed, but power value was parsed earlier.\n");
                 }
                 gpu1_power = -1;
            }
        } else {
             perror("Failed to run amd-smi"); // popen failed itself
             // gpu1_power remains -1
        }

        // Write the PAPI values and the GPU 1 power value to the CSV file.
        fprintf(params->csvFile, "%.6f,%lld,%lld,%lld,%lld,%lld,%d\n",
                elapsed, values[0], values[1], values[2], values[3], values[4], gpu1_power);
        fflush(params->csvFile);

        // Also print to stdout.
        fprintf(stdout,
                "Time: %.6f sec -> event1: %lld, event2: %lld, event3: %lld, event4: %lld, event5: %lld, GPU1_POWER: %d\n",
                elapsed, values[0], values[1], values[2], values[3], values[4], gpu1_power);

        usleep(300000);  // Sleep for 0.5 seconds.
    }
    return NULL; 
}


// Custom DGEMM kernel using a simple row-major implementation.
__global__ void dgemm_kernel(const double *A, const double *B, double *C,
                             int M, int N, int K, double alpha, double beta) {
    // Compute the row and column index of the C element.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        // Compute the dot product of row of A and column of B.
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
            //sum += sin(A[row * K + k] * B[k * N + col]) + cos(A[row * K + k] * B[k * N + col]);
            
        }
        // Scale the result and add the scaled C element.
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

int main(int argc, char *argv[]) {
    int statusFlag;
    int EventSet = PAPI_NULL;

    /* Initialize PAPI. */
    statusFlag = PAPI_library_init(PAPI_VER_CURRENT);
    if (statusFlag != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI shared library version error: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Create event set. */
    statusFlag = PAPI_create_eventset(&EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset: %s\n", PAPI_strerror(statusFlag));
        return -1;
    } 

    /* Add GPU events to the event set. */
    const char *event1 = "amd_smi:::temp_current:device=0:sensor=0";
    const char *event2 = "amd_smi:::temp_current:device=0:sensor=1";
    const char *event3 = "amd_smi:::temp_current:device=0:sensor=2";
    const char *event4 = "amd_smi:::temp_current:device=0:sensor=3";
    const char *event5 = "amd_smi:::power_average:device=0";  // New event

    statusFlag = PAPI_add_named_event(EventSet, event1);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 1: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_add_named_event(EventSet, event2);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 2: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_add_named_event(EventSet, event3);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 3: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_add_named_event(EventSet, event4);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 4: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_add_named_event(EventSet, event5);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 5: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Set HIP device properties to optimize for MI300 */
    hipSetDevice(1);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, 1);
    printf("Device Name: %s\n", deviceProp.name);
    printf("Compute Units: %d\n", deviceProp.multiProcessorCount);
    printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
    
    /* Allocate host memory for matrices A, B, and C with page-locked memory for faster transfers */
    size_t size_A = ((size_t)M_DIM * K_DIM * sizeof(double));
    size_t size_B = ((size_t)K_DIM * N_DIM * sizeof(double));
    size_t size_C = ((size_t)M_DIM * N_DIM * sizeof(double));

    double *h_A, *h_B, *h_C;
    hipHostMalloc(&h_A, size_A, hipHostMallocDefault);
    hipHostMalloc(&h_B, size_B, hipHostMallocDefault);
    hipHostMalloc(&h_C, size_C, hipHostMallocDefault);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return -1;
    }

    /* Initialize host matrices. */
    for (int i = 0; i < M_DIM * K_DIM; i++) {
        h_A[i] = (double)(i % 100);
    }
    for (int i = 0; i < K_DIM * N_DIM; i++) {
        h_B[i] = (double)(i % 100);
    }
    for (int i = 0; i < M_DIM * N_DIM; i++) {
        h_C[i] = 0.0;
    }

    /* Allocate device memory. */
    double *d_A[NUM_STREAMS], *d_B[NUM_STREAMS], *d_C[NUM_STREAMS];
    hipError_t hipStatus;
    
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStatus = hipMalloc((void**)&d_A[s], size_A);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMalloc d_A[%d] failed.\n", s);
            return -1;
        }
        hipStatus = hipMalloc((void**)&d_B[s], size_B);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMalloc d_B[%d] failed.\n", s);
            return -1;
        }
        hipStatus = hipMalloc((void**)&d_C[s], size_C);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMalloc d_C[%d] failed.\n", s);
            return -1;
        }
    }

    /* Create multiple streams for concurrent execution */
    hipStream_t streams[NUM_STREAMS];
    hipEvent_t events[NUM_STREAMS];
    
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStatus = hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipStreamCreate failed for stream %d.\n", s);
            return -1;
        }
        
        hipStatus = hipEventCreate(&events[s]);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipEventCreate failed for event %d.\n", s);
            return -1;
        }
    }

    /* Copy host matrices to device memory in parallel across streams */
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStatus = hipMemcpyAsync(d_A[s], h_A, size_A, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMemcpyAsync d_A[%d] failed.\n", s);
            return -1;
        }
        hipStatus = hipMemcpyAsync(d_B[s], h_B, size_B, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMemcpyAsync d_B[%d] failed.\n", s);
            return -1;
        }
        hipStatus = hipMemcpyAsync(d_C[s], h_C, size_C, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMemcpyAsync d_C[%d] failed.\n", s);
            return -1;
        }
    }

    /* Open CSV file for recording data and write header. */
    FILE *csvFile = fopen("test.csv", "w");
    if (!csvFile) {
        fprintf(stderr, "Failed to open CSV file for writing.\n");
        return -1;
    }
    fprintf(csvFile, "timestamp,%s,%s,%s,%s,%s,%s\n",
        event1, event2, event3, event4, event5, "power");

    /* Start PAPI counters to monitor GPU metrics. */
    statusFlag = PAPI_start(EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI start: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Start the monitoring thread immediately after PAPI_start */
    pthread_t monitor_thread;
    struct monitor_params params;
    params.EventSet = EventSet;
    params.csvFile = csvFile;
    gettimeofday(&params.start_time, NULL);  // Record the start time

    statusFlag = pthread_create(&monitor_thread, NULL, monitor_events, &params);
    if (statusFlag != 0) {
        fprintf(stderr, "pthread_create failed.\n");
        return -1;
    }

    /* Wait for initial copies to complete */
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStreamSynchronize(streams[s]);
    }

    /* GEMM parameters */
    double alpha = 0.75;
    double beta  = 0.5;
    
    // Define grid and block dimensions for the kernel launch.
    dim3 blockDim(32, 32);
    dim3 gridDim((N_DIM + blockDim.x - 1) / (blockDim.x),
                 (M_DIM + blockDim.y - 1) / (blockDim.y));

    /* Kernel execution loop to keep the GPU busy */
    for (int iter = 0; iter < ITERATIONS_PER_STREAM; iter++) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            // Launch the custom DGEMM kernel on stream 's'
            hipLaunchKernelGGL(dgemm_kernel, gridDim, blockDim, 0, streams[s],
                               d_A[s], d_B[s], d_C[s],
                               M_DIM, N_DIM, K_DIM, alpha, beta);

            // Record event but don't synchronize.
            hipEventRecord(events[s], streams[s]);
            
            hipStreamSynchronize(streams[s]);
            usleep(3000000);
        }
    }
    
    /* Wait for all streams to complete */
    /*for (int s = 0; s < NUM_STREAMS; s++) {
        
    }*/

    //usleep(3000000);
    
    /*
    hipStatus = hipMemcpyAsync(h_C, d_C[0], size_C, hipMemcpyDeviceToHost, streams[0]);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy h_C failed.\n");
        return -1;
    }*/
    
    hipStreamSynchronize(streams[0]);

    /* Signal the monitor thread to stop and wait for it to finish. */
    stop_monitor = 1;
    pthread_join(monitor_thread, NULL);

    /* Cleanup resources. */
    fclose(csvFile);
    
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipEventDestroy(events[s]);
        hipStreamDestroy(streams[s]);
        hipFree(d_A[s]);
        hipFree(d_B[s]);
        hipFree(d_C[s]);
    }
    
    hipHostFree(h_A);
    hipHostFree(h_B);
    hipHostFree(h_C);

    statusFlag = PAPI_stop(EventSet, NULL);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI stop: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    
    statusFlag = PAPI_cleanup_eventset(EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI cleanup eventset: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_destroy_eventset(&EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI destroy eventset: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
 
    return 0;
}
