/**
 * @file nvml_get_and_set_power_management_reproducer.c
 *
 * @brief This vendor sample code will get the values for a device's power usage, power limit,
 *        and minimum/maximum power. Furthermore, a new power limit will attempt to be set 
 *        (requires elevated privileges). 
 * 
 *        This file also serves as a starting point for a reproducer if a bug report needs to be
 *        sent to NVIDIA for the nvml component.
 *
 *        Tested on:
 *        - Hexane at ICL - 1 * H100 and 1 * V100 
 */

// C library headers.
#include <stdio.h>

// Cuda Toolkit headers.
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
        nvmlDevice_t device_handle;
        status = nvmlDeviceGetHandleByIndex(device_index, &device_handle);
        if (status != NVML_SUCCESS) {
            printf("For device index %u nvmlDeviceGetHandleByIndex error %d: %s - continuing to next device if applicable.\n", device_index, status, nvmlErrorString(status));
            continue;
        }

        char device_name[128] = { 0 };
        status = nvmlDeviceGetName(device_handle, device_name, sizeof(device_name)); //Should I subtract by 1 here?
        if (status != NVML_SUCCESS) {
            printf("nvmlDeviceGetName error %d: %s\n", status, nvmlErrorString(status));
            return status;
        }
        printf("Power mangement for device %d (%s)\n", device_index, device_name);
        printf("---------------------------------------------------------\n");

        printf("Handling nvmlDeviceGet functions:\n"); 
        unsigned int current_power_usage = 0;
        status = nvmlDeviceGetPowerUsage(device_handle, &current_power_usage);
        if (status != NVML_SUCCESS) {
            printf("nvmlDeviceGetPowerUsage error %d: %s\n", status, nvmlErrorString(status));
            return status;
        }
        printf("- Current power usage: %u\n", current_power_usage);

        unsigned int current_power_limit = 0;
        status = nvmlDeviceGetPowerManagementLimit(device_handle, &current_power_limit);
        if (status != NVML_SUCCESS) {
            printf("nvmlDeviceGetPowerManagementLimit error %d: %s\n", status, nvmlErrorString(status));
            return status;
        }
        printf("- Current power limit: %u\n", current_power_limit);

        unsigned int current_min_power = 0, current_max_power = 0;
        status = nvmlDeviceGetPowerManagementLimitConstraints(device_handle, &current_min_power, &current_max_power);
        if (status != NVML_SUCCESS) {
            printf("nvmlDeviceGetPowerManagementLimit error %d: %s\n", status, nvmlErrorString(status));
            return status;
        }
        printf("- Current power minimum: %u\n", current_min_power);
        printf("- Current power maximum: %u\n", current_max_power);

        printf("\nHandling nvmlDeviceSet functions:\n");
        unsigned int power_reduction = 100;
        unsigned int new_power_limit = current_max_power - power_reduction;
        // NVML_SUCCESS for nvmlDeviceSetPowerManagementLimit signifies the limit has been set.
        status = nvmlDeviceSetPowerManagementLimit(device_handle, new_power_limit);
        if (status != NVML_SUCCESS) {
            printf("nvmlDeviceSetPowerManagementLimit error %d: %s\n", status, nvmlErrorString(status));
            return status;
        }
        printf("- Power limit set from %u to %u\n", current_power_limit, new_power_limit);

        // NVML_SUCCESS for nvmlDeviceSetPowerManagementLimit signifies the limit has been set.
        status = nvmlDeviceSetPowerManagementLimit(device_handle, current_power_limit);
        if (status != NVML_SUCCESS) {
            printf("nvmlDeviceSetPowerManagementLimit error %d: %s\n", status, nvmlErrorString(status));
            return status;
        }
        printf("- Power limit set from %u to %u\n", new_power_limit, current_power_limit);

        // Formatting.
        if (device_index + 1 < device_count) {
            printf("\n");
        }
    } 

    status = nvmlShutdown();
    if (status != NVML_SUCCESS) {
        printf("nvmlShutdown error %d: %s\n", status, nvmlErrorString(status));
        return status;
    }

   return 0;
}
