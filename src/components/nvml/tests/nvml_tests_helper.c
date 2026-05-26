// Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal headers
#include "papi.h"
#include "papi_test.h"
#include "nvml_tests_helper.h" 

/** @class enumerate_and_store_nvml_native_events
  * @brief For the case users do not add an event on the command line, enumerate through
  *        the available nvml native events and store one to be used for profiling.
  *
  * @param ***nvml_native_event_names
  *   Stores the enumerated event name to be used for profiling.
  * @param *total_event_count_arg
  *   Number of events that were stored.
  * @param *nvidia_device_index_arg
  *  Device index that will be used with cudaSetDevice.
*/
void enumerate_and_store_nvml_native_events(char ***nvml_native_event_names_arg, int *total_event_count_arg, int *nvidia_device_index_arg)
{
    int nvml_cmp_idx = PAPI_get_component_index("nvml");
    if (nvml_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index()", nvml_cmp_idx);
    }
 
    int modifier = PAPI_ENUM_FIRST;
    int nvml_eventcode = 0 | PAPI_NATIVE_MASK;
    int papi_errno = PAPI_enum_cmp_event(&nvml_eventcode, modifier, nvml_cmp_idx);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event()", papi_errno);
    }

    int num_spaces_to_allocate = 1;
    char **enumerated_nvml_native_event_name = (char **) malloc(num_spaces_to_allocate * sizeof(char *));
    check_memory_allocation_call(enumerated_nvml_native_event_name);

    enumerated_nvml_native_event_name[(*total_event_count_arg)] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
    check_memory_allocation_call(enumerated_nvml_native_event_name[(*total_event_count_arg)]);

    // Convert the first nvml native event code to a name
    papi_errno = PAPI_event_code_to_name(nvml_eventcode, enumerated_nvml_native_event_name[(*total_event_count_arg)]);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name()", papi_errno);
    }
    *nvml_native_event_names_arg = enumerated_nvml_native_event_name;

    // Get the device + state substring, for the nvml component it is always present
    const char *needle = ":device_";
    char *device_and_state_substring = strstr(enumerated_nvml_native_event_name[(*total_event_count_arg)], needle);
    // Move past needle
    device_and_state_substring += strlen(needle);

    // Count the number of decimals after needle
    int c, num_decimals = 0;
    for (c = 0; device_and_state_substring[c] != '\0'; c++) {
        // We have hit state 
        if (device_and_state_substring[c] == ':') {
            break;
        }   
    
        num_decimals++; 
    }   

    char device_index[PAPI_MAX_STR_LEN] = { 0 };
    int strLen = snprintf(device_index, sizeof(device_index), "%.*s", num_decimals, device_and_state_substring);
    if (strLen < 0 || (size_t) strLen >= sizeof(device_index)) {
        fprintf(stderr, "Failed to fully write decimals into buffer.\n");
        exit(EXIT_FAILURE);
    }
    *nvidia_device_index_arg = atoi(device_index);

    (*total_event_count_arg)++;

    return;
}
