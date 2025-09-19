/**
 * @file    amdsmi_hello.c
 * @author  Dong Jun Woun <djwoun@gmail.com>
 * @brief   Minimal example that reads a single AMD-SMI event via PAPI's AMD-SMI component.
 * @details Selects the event from argv[1] if provided; otherwise defaults to
 *          "amd_smi:::temp_current:device=0:sensor=1". Requires PAPI_AMDSMI_ROOT
 *          so the component can dlopen the AMD-SMI library. Uses the test harness
 *          (test_harness.h) for consistent output and skip handling.
 */

#include "test_harness.h"

#include "papi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv) {
    // Disable stdout buffering so the harness status line appears immediately.
    setvbuf(stdout, NULL, _IONBF, 0);

    harness_accept_tests_quiet(&argc, argv);
    HarnessOpts opts = parse_harness_cli(argc, argv);

    // Event to measure (override with argv[1], e.g.:
    //   ./amdsmi_hello amd_smi:::power_average:device=0
    // )
    const char* ev = "amd_smi:::temp_current:device=0:sensor=1";
    if (argc > 1 && strncmp(argv[1], "--", 2) != 0) ev = argv[1];

    // Check AMD-SMI root so the component can dlopen the library.
    const char* root = getenv("PAPI_AMDSMI_ROOT");
    if (!root || !*root) {
        SKIP("PAPI_AMDSMI_ROOT not set");
    }

    // Initialize PAPI.
    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        NOTE("PAPI_library_init failed: %s", PAPI_strerror(papi_errno));
        return eval_result(opts, 1);
    }

    // Create an EventSet.
    int EventSet = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) {
        NOTE("PAPI_create_eventset: %s", PAPI_strerror(papi_errno));
        return eval_result(opts, 1);
    }

    // Add the selected event.
    papi_errno = PAPI_add_named_event(EventSet, ev);
    if (papi_errno == PAPI_ENOEVNT || papi_errno == PAPI_ECNFLCT ||
        papi_errno == PAPI_EPERM) {
        NOTE("Event unavailable or HW/resource-limited: %s (%s)", ev,
             PAPI_strerror(papi_errno));
        SKIP("Event unavailable or HW/resource-limited");
    } else if (papi_errno != PAPI_OK) {
        NOTE("PAPI_add_named_event(%s): %s", ev, PAPI_strerror(papi_errno));
        PAPI_destroy_eventset(&EventSet);
        return eval_result(opts, 1);
    }

    // Start counters -> short wait -> stop/read.
    papi_errno = PAPI_start(EventSet);
    if (papi_errno == PAPI_ECNFLCT || papi_errno == PAPI_EPERM) {
        NOTE("Cannot start counters: %s", PAPI_strerror(papi_errno));
        SKIP("Cannot start counters");
    } else if (papi_errno != PAPI_OK) {
        NOTE("PAPI_start: %s", PAPI_strerror(papi_errno));
        PAPI_destroy_eventset(&EventSet);
        return eval_result(opts, 1);
    }

    usleep(100000); // ~100 ms sampling interval for this simple demo.

    long long val = 0;
    papi_errno = PAPI_stop(EventSet, &val);
    if (papi_errno != PAPI_OK) {
        NOTE("PAPI_stop: %s", PAPI_strerror(papi_errno));
        PAPI_destroy_eventset(&EventSet);
        return eval_result(opts, 1);
    }

    (void)PAPI_cleanup_eventset(EventSet);
    (void)PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    // If --print was requested via the harness, emit the event name and value.
    if (opts.print) {
        printf("Event: %s\nValue: %lld\n", ev, val);
    }

    return eval_result(opts, 0);
}
