#include <stdio.h>
#include <stdlib.h>

#include "papi.h" // PAPI header file

// Basic PAPI error handling function
static void handle_papi_error(int retval, const char *function_name, int current_event_set, int papi_initialized_flag) {
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI error in function %s(): %s (Error Code: %d)\n",
                function_name, PAPI_strerror(retval), retval);
        if (papi_initialized_flag) {
            if (current_event_set != PAPI_NULL) {
                // Attempt to clean up the event set if it exists
                // Note: Depending on when the error occurred, events might still be in the set.
                // PAPI_cleanup_eventset or PAPI_destroy_eventset handles this.
                char event_name_buf[PAPI_MAX_STR_LEN];
                if (PAPI_list_events(current_event_set, NULL, NULL) > 0) { // Check if events are in set
                    // If specific event name was stored, it could be removed here.
                    // For simplicity, PAPI_destroy_eventset will handle cleanup.
                }
                PAPI_destroy_eventset(&current_event_set);
            }
            PAPI_shutdown();
        }
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    (void)argc; // Unused
    (void)argv; // Unused

    int retval=0;
    int EventSet = PAPI_NULL;
    long_long event_values[1]; // PAPI counters are long_long
    int papi_is_initialized = 0;
    
    
    
    // --- Step 1: Initialize the PAPI library ---
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    printf("AAAAAAAAAAAAAAAAAAAA\n");
    printf("%d\n", retval);
    if (retval != PAPI_VER_CURRENT && retval > 0) {
        // PAPI_VER_CURRENT is the version, > 0 are error codes
        fprintf(stderr, "PAPI library version mismatch! Expected %d, got %d\n",
                PAPI_VER_CURRENT, retval);
        exit(EXIT_FAILURE);
    }
    
    printf("BBBBBBBBBBBBBBBBBBBB\n");
    
    
    printf("%d\n", retval);
    
    
    // For other errors from PAPI_library_init (negative values)
    handle_papi_error(retval, "PAPI_library_init", EventSet, papi_is_initialized);
    papi_is_initialized = 1;
    printf("PAPI library initialized successfully.\n");

    // --- Step 2: Create an EventSet ---
    retval = PAPI_create_eventset(&EventSet);
    handle_papi_error(retval, "PAPI_create_eventset", EventSet, papi_is_initialized);
    printf("PAPI EventSet created.\n");

    // --- Step 3: Add a PAPI event to the EventSet ---
    // You can change "PAPI_TOT_INS" to any available PAPI event.
    // For CPU events: "PAPI_TOT_CYC" (Total Cycles), "PAPI_L1_DCM" (L1 Data Cache Misses), etc.
    // For rocm_smi events (if PAPI is configured for it and you have a ROCm GPU):
    // e.g., "rocm_smi:::power_average:device=0" (Average GPU Power)
    // e.g., "rocm_smi:::temp_rx_soc:device=0" (GPU Temperature)
    // Check availability with `papi_avail` or `papi_native_avail` utilities.
    const char *eventName = "PAPI_TOT_INS"; // Example: Total Instructions Executed

    printf("Attempting to add event: %s\n", eventName);
    retval = PAPI_add_named_event(EventSet, eventName);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Failed to add PAPI event '%s'. Error: %s (Code: %d)\n",
                eventName, PAPI_strerror(retval), retval);
        if (retval == PAPI_ECMP) { // Component not available error
            fprintf(stderr, "This error often means the PAPI component required for '%s' (e.g., 'rocm_smi' for GPU events) is not available, not configured, or the event name is incorrect for your hardware.\n", eventName);
            fprintf(stderr, "Please use PAPI utilities like 'papi_components_avail' and 'papi_native_avail' to check available components and events.\n");
        }
        handle_papi_error(retval, "PAPI_add_named_event", EventSet, papi_is_initialized); // Will cleanup and exit
    }
    printf("Successfully added event '%s' to EventSet.\n", eventName);

    // --- Step 4: Start counting events in the EventSet ---
    retval = PAPI_start(EventSet);
    handle_papi_error(retval, "PAPI_start", EventSet, papi_is_initialized);
    printf("PAPI event counting started.\n");

    // --- Optional: Perform some work ---
    // The PAPI counters will measure events that occur between PAPI_start() and PAPI_stop().
    // For a simple check, even a small amount of computation will do.
    printf("Performing some dummy computation...\n");
    volatile double work_dummy = 0.0;
    for (long i = 0; i < 20000000; ++i) { // Increased loop iterations
        work_dummy += (double)i / (double)(i + 1);
    }
    printf("Dummy computation finished. (Result: %f to prevent optimization)\n", work_dummy);

    // --- Step 5: Stop counting events and retrieve the values ---
    // PAPI_stop() will read the current values of the events in EventSet into event_values.
    retval = PAPI_stop(EventSet, event_values);
    handle_papi_error(retval, "PAPI_stop", EventSet, papi_is_initialized);
    printf("PAPI event counting stopped.\n");

    // --- Step 6: Print the event value ---
    printf("\n--- PAPI Event Value ---\n");
    printf("Event: %s\n", eventName);
    printf("Value: %lld\n\n", event_values[0]);

    // --- Step 7: Clean up PAPI resources ---
    // Remove the event (optional if PAPI_destroy_eventset is called, but good practice)
    // PAPI_cleanup_eventset also works and is sometimes preferred as it can clear, stop, and remove events.
    retval = PAPI_remove_named_event(EventSet, eventName);
    // Don't treat "event not in set" or "set not running" as fatal after stop for simple cleanup
    if (retval != PAPI_OK && retval != PAPI_ENOEVNT && retval != PAPI_ENOTRUN ) {
         handle_papi_error(retval, "PAPI_remove_named_event", EventSet, papi_is_initialized);
    } else if (retval == PAPI_OK) {
        printf("PAPI event '%s' removed from EventSet.\n", eventName);
    }


    // Destroy the EventSet
    retval = PAPI_destroy_eventset(&EventSet);
    handle_papi_error(retval, "PAPI_destroy_eventset", EventSet, papi_is_initialized); // EventSet becomes PAPI_NULL internally
    printf("PAPI EventSet destroyed.\n");

    // Shutdown the PAPI library
    PAPI_shutdown();
    printf("PAPI library shut down successfully.\n");

    return EXIT_SUCCESS;
}