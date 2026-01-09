// Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Cuda Toolkit headers
#include <cuda_runtime_api.h>

// Internal headers
#include "papi.h"
#include "papi_test.h"

/** @class add_cuda_native_events
  * @brief Try and add each event provided on the command line by the user.
  *
  * @param EventSet
  *   A PAPI eventset.
  * @param *cuda_native_event_name
  *   Event to add to the EventSet.
  * @param *num_events_successfully_added
  *   Total number of successfully added events.
  * @param **events_successfully_added
  *   Events that we are able to add to the EventSet.
  * @param *numMultipassEvents
  *   Counter to see if a multiple pass event was provided on the command line.
*/
void add_cuda_native_events(int EventSet, const char *cuda_native_event_name, int *num_events_successfully_added, char **events_successfully_added, int *numMultipassEvents)
{
   int papi_errno = PAPI_add_named_event(EventSet, cuda_native_event_name);
   if (papi_errno != PAPI_OK) {
       if (papi_errno != PAPI_EMULPASS) {
           fprintf(stderr, "Unable to add event %s to the EventSet with error code %d.\n", cuda_native_event_name, papi_errno);
           exit(EXIT_FAILURE);
       }   
       // Handle multiple pass events
       (*numMultipassEvents)++;
       return;
   }   

   // Handle successfully added events
   int strLen = snprintf(events_successfully_added[(*num_events_successfully_added)], PAPI_MAX_STR_LEN, "%s", cuda_native_event_name);
   if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
       fprintf(stderr, "Failed to fully write successfully added event.\n");
       exit(EXIT_FAILURE);
   }   
   (*num_events_successfully_added)++;

    return;
}

/** @class determine_if_device_is_enabled
  * @brief If a machine has mixed compute capabilites determine which devices
  *        are available to be used.
  *
  * @param device_idx
  *   Index of the device on the machine.
*/
int determine_if_device_is_enabled(int device_idx) 
{
    cudaDeviceProp device_prop;
    cudaError_t cudaError = cudaGetDeviceProperties(&device_prop, device_idx); 
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "Call to cudaGetDeviceProperties failed with error code: %d.\n", cudaError);
        exit(EXIT_FAILURE);
    }   

    int device_enabled = 1;
    char *cudaApi = getenv("PAPI_CUDA_API");
    // Perfworks API is enabled
    if (cudaApi == NULL) {
        // Perfworks Metrics API supports CC's >= 7
        if (device_prop.major < 7) {
            device_enabled = 0;
        }   
    }   
    // Legacy API is enabled
    else {
        // Legacy API supports CC's <= 7
        if (device_prop.major > 7) {
            device_enabled = 0;
        }   
    }   

    return device_enabled;

}

/** @class enumerate_and_store_cuda_native_events
  * @brief For the case users do not add an event on the command line, enumerate through
  *        the available cuda native events and store one to be used for profiling.
  *
  * @param **cuda_native_event_names
  *   Stores the enumerated event name to be used for profiling.
  * @param *total_event_count
  *   Number of events that were stored.
  * @param *cuda_device_index
  *  Device index that will be used to create a cuda context. 
*/
void enumerate_and_store_cuda_native_events(char **cuda_native_event_names, int *total_event_count, int *cuda_device_index)
{
    // Get the first cuda native event on the architecture.
    int cuda_cmp_idx = PAPI_get_component_index("cuda");
    if (cuda_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index", cuda_cmp_idx);
    } 
 
    int modifier = PAPI_ENUM_FIRST;
    int cuda_eventcode = 0 | PAPI_NATIVE_MASK;
    int papi_errno = PAPI_enum_cmp_event(&cuda_eventcode, modifier, cuda_cmp_idx);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event", papi_errno);
    } 

    // Convert the first cuda native event code to a name, the name will
    // be in the format of cuda:::basename with no qualifiers appended.
    char basename[PAPI_MAX_STR_LEN];
    papi_errno = PAPI_event_code_to_name(cuda_eventcode, basename);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", papi_errno);
    }

    // Begin reconstructing the Cuda native event name with qualifiers
    cuda_native_event_names[(*total_event_count)] = (char *) calloc((*total_event_count) + 1, PAPI_MAX_STR_LEN * sizeof(char));
    if (cuda_native_event_names[(*total_event_count)] == NULL) {
        fprintf(stderr, "Failed to allocate memory for index %d in metric names.\n", total_event_count);
        exit(EXIT_FAILURE);
    }

    int strLen = snprintf(cuda_native_event_names[(*total_event_count)], PAPI_MAX_STR_LEN, "%s", basename);
    if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
        fprintf(stderr, "Failed to fully write event name.");
        exit(EXIT_FAILURE);
    } 

    // Enumerate through the available default qualifiers.
    // The Legacy API only has the device qualifiers
    // while the Perfworks Metrics API has a stat and device
    // qualifier.
    modifier = PAPI_NTV_ENUM_DEFAULT_QUALIFIERS;
    papi_errno = PAPI_enum_cmp_event(&cuda_eventcode, modifier, cuda_cmp_idx);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event", papi_errno);
    }   

    do {
        PAPI_event_info_t info;
        papi_errno = PAPI_get_event_info(cuda_eventcode, &info);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_get_event_info", papi_errno);
        }   

        char *qualifier = strstr(info.symbol + strlen("cuda:::"), ":");
        if (strncmp(qualifier, ":device=", 8) == 0) {
            (*cuda_device_index) = strtol(qualifier + strlen(":device="), NULL, 10);
        }   

        int strLen = snprintf(cuda_native_event_names[(*total_event_count)] + strlen(cuda_native_event_names[(*total_event_count)]), PAPI_MAX_STR_LEN - strlen(cuda_native_event_names[(*total_event_count)]), "%s", qualifier);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Unable to construct cuda native event name.\n");
             exit(EXIT_FAILURE);
        }   

    } while (PAPI_enum_cmp_event(&cuda_eventcode, modifier, cuda_cmp_idx) == PAPI_OK);

    // Safety net, this should never be triggered
    if ((*cuda_device_index) == -1) {
        fprintf(stderr, "A device qualifier is needed to continue or a device index must be provided on the command line.\n");
        exit(EXIT_FAILURE);
    }

    (*total_event_count)++;

    return;
}
