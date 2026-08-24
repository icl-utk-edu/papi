/**
* @file  cuda_range_floating_point_operations.
* @brief This test verifies the counters collected for cuda_range native events that deal with
*        floating point operations.
*/

// C++ STL headers.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// CTK headers.
#include <cuda.h>

// Internal headers.
#include "cuda_range_tests_helper.hpp"
#include "papi.h"
#include "papi_test.h"

#define MAX_ITERATION_ENTRIES 3

/** 
  * @brief An enum containing the available operations.
*/
typedef enum {
    ADD = 0,
    MULTIPLY,
    FUSED_MULTIPLY_ADD,
} operation_e;

/** 
  * @brief An enum containing the available precisions.
*/
typedef enum {
    SINGLE = 0,
    DOUBLE,
} precision_e;

/** 
  * @brief An addition kernel.
  *
  * @param number_of_iterations
  *   The number of iterations to perform.
  * @param *x
  *   An array for floating point operations.
  * @param *y
  *   An array for floating point operations.
  * @param precision
  *   The precision to be performed.
*/
__global__ void add(int number_of_iterations, void *x, void *y, precision_e precision) {
    int i;
    for (i = 0; i < number_of_iterations; i++) {
        if (precision == SINGLE) {
            ((float *) y)[i] = ((float *) x)[i] + ((float *) y)[i];
        }
        else {
            ((double *) y)[i] = ((double *) x)[i] + ((double *) y)[i];
        }
    }

    return;
}

/** 
  * @brief A multiply kernel.
  *
  * @param number_of_iterations
  *   The number of iterations to perform.
  * @param *x
  *   An array for floating point operations.
  * @param *y
  *   An array for floating point operations.
  * @param precision
  *   The precision to be performed.
*/
__global__ void multiply(int number_of_iterations, void *x, void *y, precision_e precision)
{
    int i;
    for (i = 0; i < number_of_iterations; i++) {
        if (precision == SINGLE) {
            ((float *) y)[i] = ((float *) x)[i] * ((float *) y)[i];
        }
        else {
            ((double *) y)[i] = ((double *) x)[i] * ((double *) y)[i];
        }
    }

    return;
}

/** 
  * @brief A fused multiply add kernel.
  *
  * @param number_of_iterations
  *   The number of iterations to perform.
  * @param *x
  *   An array for floating point operations.
  * @param *y
  *   An array for floating point operations.
  * @param precision
  *   The precision to be performed.
*/
__global__ void fused_multiply_add(int number_of_iterations, void *x, void *y, precision_e precision)
{
    int i;
    for (i = 0; i < number_of_iterations; i++) {
        if (precision == SINGLE) {
            ((float *) y)[i] = ((float *) x)[i] * ((float *) y)[i] + 1.0f;
        }
        else {
            ((double *) y)[i] = ((double *) x)[i] * ((double *) y)[i] + 1.0;
        }
    }

    return;
}

/** 
  * @brief A wrapper function to aid in launching the correct operation.
  *
  * @param number_of_iterations
  *   The number of iterations to perform.
  * @param *x
  *   An array for floating point operations.
  * @param *y
  *   An array for floating point operations.
  * @param precision
  *   The precision to be performed.
  * @param operation
  *   The operation to be performed.
*/
void launch_kernel(int number_of_iterations, void *x, void *y, precision_e precision, operation_e operation)
{
    switch(operation) {
        case ADD:
            add<<<1, 1>>>(number_of_iterations, x, y, precision);
            break;
        case MULTIPLY:
            multiply<<<1,1>>>(number_of_iterations, x, y, precision);
            break;
        case FUSED_MULTIPLY_ADD:
            fused_multiply_add<<<1, 1>>>(number_of_iterations, x, y, precision);
            break;
        default:
            break;
    }
    CHECK_CUDA_RUNTIME_API_CALL( cudaGetLastError() );
    CHECK_CUDA_RUNTIME_API_CALL( cudaDeviceSynchronize() );

    return;
}

/** 
  * @brief A function wrapper to print out the help message for test_cuda_range_floating_point_operations.cu.
*/
static void print_help_message(void)
{
    std::cout << "./cuda_floating_point_operations --device [NVIDIA device index] --number-of-iterations [list of iterations to perform (must be exactly 3) separated by a comma]"
              << " --precision [options include single (default) or double]"
              << " --operation [options include add (default), multiply, fused_add_multiply]." << std::endl
              << "Notes:" << std::endl
              << "1. The default precision is single and the default operation is add." << std::endl
              << "2. If the number of iterations listed is greater than 3 then the iteration value will not be stored, BUT the test will proceed." << std::endl;

    return;
}

/** 
  * @brief Parse the command line arguments provided by the user.
  *
  * @param argc
  *   Number of user passed arguments on the command line.
  * @param *argv
  *   Argument vector.
  * @param &device_index
  *   Stores the device index passed by the user to --device.
  * @param &precision
  *   Stores the precision passed by the user to --precision.
  * @param &operation
  *   Stores the operation passed by the user to --operation.
  * @param &number_of_iterations
  *   Stores the number of iterations passed by the user to --number-of-iterations.
*/
static void parse_and_assign_args(int argc, char *argv[], int &device_index, precision_e &precision,
                                  operation_e &operation, std::vector<int> &number_of_iterations)
{
    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_help_message();
            exit(EXIT_SUCCESS);
        }
        else if (strcmp(arg, "--device") == 0) {
            if (!argv[i + 1]) {
                std::cout << "ERROR!! Add a NVIDIA device index." << std::endl;
                exit(EXIT_FAILURE);
            }
            device_index = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(arg, "--number-of-iterations") == 0) {
            if (!argv[i + 1]) {
                std::cout << "ERROR!! --number-of-iterations given, but no numbers listed. Exiting." << std::endl;
                exit(EXIT_FAILURE);
            }

            // Clear the current default iteration values.
            number_of_iterations.clear();

            std::stringstream ss(argv[i + 1]);
            std::string iteration_number;
            while (std::getline(ss, iteration_number, ',')) {
                // Under the allowed number of iterations.
                if (number_of_iterations.size() + 1 <= MAX_ITERATION_ENTRIES) {
                    number_of_iterations.push_back(std::stoi(iteration_number));
                }
                // Exceeded the allowed number of iterations.
                else {
                    print_help_message();
                    exit(EXIT_FAILURE);
                }
            }   
            i++;
        }
        else if (strcmp(arg, "--precision") == 0) {
            if (!argv[i + 1]) {
                std::cout << "ERROR!! --precision given, but no precision listed. Exiting." << std::endl;
                exit(EXIT_FAILURE);
            }

            if (strcmp(argv[i + 1], "single") == 0) {
                precision = SINGLE;
            }
            else if (strcmp(argv[i + 1], "double") == 0) {
                precision = DOUBLE;
            }
            else {
                std::cout << "ERROR!! Provided precision is not valid. Exiting." << std::endl;
                print_help_message();
                exit(EXIT_FAILURE);
            }
            i++;
        }
        else if (strcmp(arg, "--operation") == 0) {
            if (!argv[i + 1]) {
                std::cout << "ERROR!! --operation given, but no operation listed. Exiting." << std::endl;
                exit(EXIT_FAILURE);
            }

            if (strcmp(argv[i + 1], "add") == 0) {
                operation = ADD;
            }
            else if (strcmp(argv[i + 1], "multiply") == 0) {
                operation = MULTIPLY;
            }
            else if (strcmp(argv[i + 1], "fused_multiply_add") == 0) {
                operation = FUSED_MULTIPLY_ADD;
            }
            else {
                std::cout << "ERROR!! Provided operation is not valid. Exiting." << std::endl;
                print_help_message();
                exit(EXIT_FAILURE);
            }
            i++;
        }
        else {
            print_help_message();
            exit(EXIT_FAILURE);
        }
    }

    return;
}

int main(int argc, char **argv)
{
    std::cout << "Running the cuda_range component test -- test_cuda_range_floating_point_operations.cu." << std::endl;
    CHECK_CUDA_DRIVER_API_CALL( cuInit(0) );

    int device_index = 0;
    precision_e precision = SINGLE;
    operation_e operation = ADD;
    std::vector<int> number_of_iterations {2, 4, 16};
    // If a user provided command line arguments then parse them.
    if (argc > 1) {
        parse_and_assign_args(argc, argv, device_index, precision, operation, number_of_iterations);
    }

    // Determine the number of compute-capable devices.
    int number_of_devices_on_system = 0;
    CHECK_CUDA_RUNTIME_API_CALL( cudaGetDeviceCount(&number_of_devices_on_system) );
    // No compute-capable devices on the machine. Exiting.
    if (number_of_devices_on_system < 1) {
        std::cout << "No compute-capable devices found on the machine. This is required for the test to run." << std::endl;
        exit(EXIT_FAILURE);
    }

    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }   
    std::cout << "The PAPI version being used for this test is: "
              << PAPI_VERSION_MAJOR(PAPI_VERSION) << "." 
              << PAPI_VERSION_MINOR(PAPI_VERSION) << "." 
              << PAPI_VERSION_REVISION(PAPI_VERSION) << "." 
              << std::endl;

    // Confirm the cuda_range component exists e.g. compiled in.
    int cuda_range_cmp_index = PAPI_get_component_index("cuda_range");
    if (cuda_range_cmp_index < 0) {
        std::cout << "The cuda_range component does not exist. This is often due to cuda_range not being passed to --with-components at configure." << std::endl;
        exit(EXIT_FAILURE);
    } 

    std::string cuda_range_native_event_name {};
    cuda_range_native_event_name.reserve(PAPI_2MAX_STR_LEN);
    // Native event name which corresponds to single precision and an operation of addition.
    if (precision == SINGLE && operation == ADD) {
        cuda_range_native_event_name = "cuda_range:::smsp__sass_thread_inst_executed_op_fadd_pred_on:stat=sum";
    }
    // Native event name which corresponds to single precision and an operation of multiply.
    else if (precision == SINGLE && operation == MULTIPLY) {
        cuda_range_native_event_name = "cuda_range:::smsp__sass_thread_inst_executed_op_fmul_pred_on:stat=sum";
    }
    // Native event name which corresponds to single precision and an operation of multiply + add.
    else if (precision == SINGLE && operation == FUSED_MULTIPLY_ADD) {
        cuda_range_native_event_name = "cuda_range:::smsp__sass_thread_inst_executed_op_ffma_pred_on:stat=sum";
    }
    // Native event name which corresponds to double precision and an operation of add.
    else if (precision == DOUBLE && operation == ADD) {
        cuda_range_native_event_name = "cuda_range:::smsp__sass_thread_inst_executed_op_dadd_pred_on:stat=sum";
    }
    // Native event name which corresponds to double precision and an operation of multiply.
    else if (precision == DOUBLE && operation == MULTIPLY) {
        cuda_range_native_event_name = "cuda_range:::smsp__sass_thread_inst_executed_op_dmul_pred_on:stat=sum";
    }
    // Native event name which corresponds to double precision and an operation of multiply + add.
    else if (precision == DOUBLE && operation == FUSED_MULTIPLY_ADD) {
        cuda_range_native_event_name = "cuda_range:::smsp__sass_thread_inst_executed_op_dfma_pred_on:stat=sum";
    }
    // Combination of precision and operation does not exist.
    else {
        std::cout << "The combination of precision and operation does not correspond to a cuda_range native event. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Confirm the cuda_range native event corresponding to a precision and operation exists on the system.
    papi_errno = PAPI_query_named_event(cuda_range_native_event_name.c_str());
    if (papi_errno != PAPI_OK) {
        std::cout << "The cuda_range native event (" << cuda_range_native_event_name << ") does not exist on the machine. Skipping." << std::endl;
        test_skip(__FILE__, __LINE__, "", 0); 
    } 

    // Set the device to be used for GPU execution.
    CHECK_CUDA_RUNTIME_API_CALL( cudaSetDevice(device_index) );

    // To properly allocate the below arrays with enough space determine the max iteration number.
    long long maximum_iteration = 0;
    for (size_t i = 0; i < number_of_iterations.size(); i++) {
        if (maximum_iteration < number_of_iterations[i]) {
            maximum_iteration = number_of_iterations[i];
        }
    }

    // Allocate memory for arrays.
    float *x, *y;
    unsigned int flags = cudaMemAttachGlobal; // Allows memory to be accessible from any stream on any device
    CHECK_CUDA_RUNTIME_API_CALL( cudaMallocManaged(&x, maximum_iteration * sizeof(float), flags ) );
    CHECK_CUDA_RUNTIME_API_CALL( cudaMallocManaged(&y, maximum_iteration * sizeof(float), flags ) );

    // Initialize values for arrays on the host.
    for (size_t i = 0; i < maximum_iteration; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Create a PAPI event set.
    int event_set = PAPI_NULL;
    CHECK_PAPI_API_CALL( PAPI_create_eventset(&event_set) );

    // Add the cuda_range native event to the event set.
    CHECK_PAPI_API_CALL( PAPI_add_named_event(event_set, cuda_range_native_event_name.c_str()) );

    // Start profiling.
    CHECK_PAPI_API_CALL( PAPI_start(event_set) );

    // 1st PAPI_read.
    long long expected_cuda_range_cuda_range_counter_value = number_of_iterations[0];
    launch_kernel(number_of_iterations[0], x, y, precision, operation);
    long long cuda_range_counter_value = 0;
    CHECK_PAPI_API_CALL( PAPI_read(event_set, &cuda_range_counter_value) );
    if (cuda_range_counter_value == expected_cuda_range_cuda_range_counter_value) {
        std::cout << "1st PAPI_read: Correct count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
    }
    else {
        std::cout << "1st PAPI_read: Incorrect count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
        exit(EXIT_FAILURE);
    }

    // 2nd PAPI_read.
    expected_cuda_range_cuda_range_counter_value += number_of_iterations[1];
    launch_kernel(number_of_iterations[1], x, y, precision, operation);
    CHECK_PAPI_API_CALL( PAPI_read(event_set, &cuda_range_counter_value) );
    if (cuda_range_counter_value == static_cast<long long>(expected_cuda_range_cuda_range_counter_value)) {
        std::cout << "2nd PAPI_read: Correct count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
    }
    else {
        std::cout << "2nd PAPI_read: Incorrect count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
        exit(EXIT_FAILURE);
    }

    // 3rd PAPI_read.
    expected_cuda_range_cuda_range_counter_value += number_of_iterations[2];
    launch_kernel(number_of_iterations[2], x, y, precision, operation);
    CHECK_PAPI_API_CALL( PAPI_read(event_set, &cuda_range_counter_value) );
    if (cuda_range_counter_value == static_cast<long long>(expected_cuda_range_cuda_range_counter_value)) {
        std::cout << "3rd PAPI_read: Correct count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
    }
    else {
        std::cout << "3rd PAPI_read: Incorrect count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
        fprintf(stderr, "\033[0;31m3rd PAPI_read: Correct count -- expected was  %lld and actual is %lld.\n\033[0m", expected_cuda_range_cuda_range_counter_value, cuda_range_counter_value);
        exit(EXIT_FAILURE);
    }

    // Final PAPI_read.
    // No work is occurring; therefore, PAPI_read here SHOULD give back the counter value obtained in the 3rd PAPI_read
    CHECK_PAPI_API_CALL( PAPI_read(event_set, &cuda_range_counter_value) );
    if (cuda_range_counter_value == static_cast<long long>(expected_cuda_range_cuda_range_counter_value)) {
        std::cout << "Final PAPI_read: Correct count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
    }   
    else {
        std::cout << "Final PAPI_read: Incorrect count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
        exit(EXIT_FAILURE);
    }  

    // 1st PAPI_stop.
    // No work is occurring; therefore, PAPI_stop here SHOULD give back the counter value obtained in the 3rd PAPI_read
    CHECK_PAPI_API_CALL( PAPI_stop(event_set, &cuda_range_counter_value) );
    if (cuda_range_counter_value == static_cast<long long>(expected_cuda_range_cuda_range_counter_value)) {
        std::cout << "PAPI_stop: Correct count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
    }
    else {
        std::cout << "PAPI_stop: Incorrect count -- expected was " << expected_cuda_range_cuda_range_counter_value << " and actual is " << cuda_range_counter_value << std::endl;
        exit(EXIT_FAILURE);
    }

    // Free the allocated memory on the device. 
    CHECK_CUDA_RUNTIME_API_CALL( cudaFree(x) );
    CHECK_CUDA_RUNTIME_API_CALL( cudaFree(y) );

    // Shutdown the PAPI library.
    PAPI_shutdown();

    test_pass(__FILE__);

    return 0;
}
