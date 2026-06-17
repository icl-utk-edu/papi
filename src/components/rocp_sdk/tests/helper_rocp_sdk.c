// Standard library headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal headers
#include <papi.h>
#include <papi_test.h>

/** @class add_rocp_sdk_native_events
  * @brief Wrapper function to simply add the provided native events to an event set by name.
  *
  * @param EventSet
  *   The EventSet we want to add the events to.
  * @param total_event_count
  *   The maximum number of events we will attempt to add.
  * @param **rocp_sdk_native_event_names
  *   Names of the rocp_sdk native events we want to add.
*/
void add_rocp_sdk_native_events(int EventSet, int total_event_count, char **rocp_sdk_native_event_names)
{
    int i;
    for (i = 0; i < total_event_count; i++) {
        int papi_errno = PAPI_add_named_event(EventSet, rocp_sdk_native_event_names[i]);
        if (papi_errno != PAPI_OK) {
            fprintf(stderr, "Unable to add event %s to the EventSet with error code %d.\n", rocp_sdk_native_event_names[i], papi_errno);
            exit(EXIT_FAILURE);
        }
    }

    return;
}

/** @class enumerate_and_store_rocp_sdk_native_events
  * @brief For the case users do not add an event on the command line, enumerate through
  *        the available rocp_sdk native events and store one to be used for profiling.
  *
  * @param ***rocp_sdk_native_event_names
  *   Stores the enumerated event name to be used for profiling.
  * @param *total_event_count
  *   Number of events that were stored.
*/
void enumerate_and_store_rocp_sdk_native_events(char ***rocp_sdk_native_event_names, int *total_event_count)
{
    int rocp_sdk_cmp_idx = PAPI_get_component_index("rocp_sdk");
    if (rocp_sdk_cmp_idx < 0) {
        test_fail(__FILE__, __LINE__, "PAPI_get_component_index", rocp_sdk_cmp_idx);
    }

    // Get the first rocp_sdk native event on the architecture
    int modifier = PAPI_ENUM_FIRST;
    int rocp_sdk_eventcode = 0 | PAPI_NATIVE_MASK;
    int papi_errno = PAPI_enum_cmp_event(&rocp_sdk_eventcode, modifier, rocp_sdk_cmp_idx);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_enum_cmp_event", papi_errno);
    }

    // Note: We are only going to get a single event here as stands, but if you want to enumerate more events
    // then update num_spaces_to_allocate to the desired number.
    int num_spaces_to_allocate = 1;
    char **enumerated_rocp_sdk_native_event_names = (char **) malloc(num_spaces_to_allocate * sizeof(char *));
    if (enumerated_rocp_sdk_native_event_names == NULL) {
        exit(EXIT_FAILURE);
    }

    // Enumerate and store rocp_sdk native events on the architecture
    modifier = PAPI_ENUM_EVENTS;
    do {
        char rocp_sdk_eventname[PAPI_MAX_STR_LEN];
        papi_errno = PAPI_event_code_to_name(rocp_sdk_eventcode, rocp_sdk_eventname);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_event_code_to_name", papi_errno);
        }

        enumerated_rocp_sdk_native_event_names[(*total_event_count)] = (char *) malloc(PAPI_MAX_STR_LEN * sizeof(char));
        if (enumerated_rocp_sdk_native_event_names[(*total_event_count)] == NULL) {
            exit(EXIT_FAILURE);
        }

        int strLen = snprintf(enumerated_rocp_sdk_native_event_names[(*total_event_count)], PAPI_MAX_STR_LEN, "%s", rocp_sdk_eventname);
        if (strLen < 0 || strLen >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "Unable to copy the event %s.\n", rocp_sdk_eventname);
            exit(EXIT_FAILURE);
        }

        (*total_event_count)++;

    } while(PAPI_enum_cmp_event(&rocp_sdk_eventcode, modifier, rocp_sdk_cmp_idx) == PAPI_OK && (*total_event_count) != num_spaces_to_allocate);
    *rocp_sdk_native_event_names = enumerated_rocp_sdk_native_event_names;

    return;
}
