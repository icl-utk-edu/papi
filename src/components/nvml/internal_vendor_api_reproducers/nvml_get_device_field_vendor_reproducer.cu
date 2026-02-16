/**
 * @file nvml_get_device_field_vendor_reproducer.cu
 *
 * @brief This vendor sample code will get the values for the fieldId NVML_FI_DEV_POWER_INSTANT
 *        and the scopeId NVML_POWER_SCOPE_GPU for each GPU detected on the machine.
 * 
 *        This file also can serves as a starting point as a reproducer if a bug report needs to be
 *        sent to NVIDIA for the nvml component.
 *
 *        To compile: nvcc -I$CUDA_HOME/include -L$CUDA_HOME/lib64 nvml_get_device_field_vendor_reproducer.cu -o nvml_get_device_field_vendor_reproducer -lnvidia-ml
 *
 *        Tested on:
 *        - Methane at ICL - 1 * A100 
 *        - Athena at Oregon - 4 * A100
 */

// C library headers
#include <stdio.h>

// Cuda Toolkit headers
#include <nvml.h>

int main()
{
    nvmlReturn_t status = nvmlInit();
    if (status != NVML_SUCCESS) {
        printf("nvmlInit error %d: %s\n", status, nvmlErrorString(status));
        return status;
    }

    unsigned int device_count = 0;
    status = nvmlDeviceGetCount(&device_count);
    if (status != NVML_SUCCESS) {
        printf("nvmlDeviceGetCount error %d: %s\n", status, nvmlErrorString(status));
        return status;
    }

    for (unsigned int device_index = 0; device_index < device_count; device_index++) {

        nvmlDevice_t device;
        status = nvmlDeviceGetHandleByIndex(device_index, &device);
        if (status != NVML_SUCCESS) {
            printf("For device index %u nvmlDeviceGetHandleByIndex error %d: %s - continuing to next device if applicable\n", device_index, status, nvmlErrorString(status));
            continue;
        }

        int value_count = 1;
        nvmlFieldValue_t field_value[value_count];
        field_value->fieldId = NVML_FI_DEV_POWER_INSTANT;
        field_value->scopeId = NVML_POWER_SCOPE_GPU;

        status = nvmlDeviceGetFieldValues(device, value_count, field_value);
        if (status != NVML_SUCCESS) {
            printf("For device index %u nvmlDeviceGetFieldValues error %d: %s - continuing to next device if applicable\n", device_index, status, nvmlErrorString(status));
            continue;
        }

        if (field_value->nvmlReturn != NVML_SUCCESS) {
            printf("For device index %u failed to obtain value for fieldId %d: %s - continuing to next device if applicable\n",
                   device_index, field_value->fieldId, nvmlErrorString(field_value->nvmlReturn));
            continue;
        }

        switch(field_value->valueType) {
            case NVML_VALUE_TYPE_DOUBLE:
                printf("Power for device index %d is %f\n", device_index, field_value->value.dVal);
                break;
            case NVML_VALUE_TYPE_UNSIGNED_INT:
                printf("Power for device index %d is %u\n", device_index, field_value->value.uiVal);
                break;
            case NVML_VALUE_TYPE_UNSIGNED_LONG:
                printf("Power for device index %d is %lu\n", device_index, field_value->value.ulVal);
                break;
            case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
                printf("Power for device index %d is %llu\n", device_index, field_value->value.ullVal);
                break;
            case NVML_VALUE_TYPE_SIGNED_LONG_LONG:
                printf("Power for device index %d is %lld\n", device_index, field_value->value.sllVal);
                break;
            case NVML_VALUE_TYPE_SIGNED_INT:
                printf("Power for device index %d is %d\n", device_index, field_value->value.siVal);
                break;
            case NVML_VALUE_TYPE_UNSIGNED_SHORT:
                printf("Power for device index %d is %hu\n", device_index, field_value->value.usVal);
                break;
            default:
                printf("valueType is not recognized.\n");
                break;
        }
    }

    status = nvmlShutdown();
    if (status != NVML_SUCCESS) {
        printf("nvmlShutdown error %d: %s\n", status, nvmlErrorString(status));
        return status;
    }

    return 0;
}
