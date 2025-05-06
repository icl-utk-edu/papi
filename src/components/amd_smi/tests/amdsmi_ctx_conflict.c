/**
 * @file    amdsmi_ctx_conflict.c
 * @author  Dong Jun Woun
 *          djwoun@gmail.com
 * @brief   Validates that an AMD-SMI native event exposed via PAPI is context-exclusive
 *          by attempting to start the same event in two threads. Expected result:
 *          thread 1 starts successfully; thread 2 fails with PAPI_ECNFLCT.
 *
 * Usage:
 *   ./amdsmi_ctx_conflict [<amd_smi native event string>] [harness options]
 *   If no event is provided, defaults to "amd_smi:::temp_current:device=0:sensor=1".
 */

#include "test_harness.h"
#include "papi.h"

#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/** PAPI thread-id callback. */
static unsigned long get_tid(void) { return (unsigned long)pthread_self(); }

struct ThreadState {
    int start_papi_errno;
};

static _Atomic bool t1_started = false;

/* Default event; can be overridden by argv[1] (any AMD-SMI native event string). */
static const char* g_event = "amd_smi:::temp_current:device=0:sensor=1";

/**
 * Thread 1:
 * - Creates an EventSet, adds the selected event, and starts it.
 * - Keeps it running briefly so thread 2 collides on start.
 * Expected: PAPI_start succeeds.
 */
static void* thread_fn1(void* arg) {
    PAPI_register_thread();
    struct ThreadState* st = (struct ThreadState*)arg;

    int EventSet = PAPI_NULL;
    int papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) { NOTE("t1 create: %s", PAPI_strerror(papi_errno)); st->start_papi_errno = papi_errno; PAPI_unregister_thread(); return NULL; }

    papi_errno = PAPI_add_named_event(EventSet, g_event);
    if (papi_errno == PAPI_ENOEVNT) { SKIP("Event not supported on this platform"); }
    if (papi_errno == PAPI_ECNFLCT || papi_errno == PAPI_EPERM) { SKIP("Cannot add event due to HW/resource limits"); }
    if (papi_errno != PAPI_OK) { NOTE("t1 add: %s", PAPI_strerror(papi_errno)); st->start_papi_errno = papi_errno; PAPI_destroy_eventset(&EventSet); PAPI_unregister_thread(); return NULL; }

    papi_errno = PAPI_start(EventSet);
    st->start_papi_errno = papi_errno;
    if (papi_errno == PAPI_OK) {
        /* Publish that t1 is actively running the event so t2 can attempt to collide. */
        atomic_store_explicit(&t1_started, true, memory_order_release);
        long long v = 0; (void)PAPI_read(EventSet, &v);
        usleep(100000); /* run long enough for thread 2 to attempt start */
        (void)PAPI_stop(EventSet, &v);
    } else {
        /* If t1 cannot start, the test cannot be executed cleanly: skip due to HW/resource limits. */
        SKIP("Cannot start thread1 due to HW/resource limits");
    }

    (void)PAPI_cleanup_eventset(EventSet);
    (void)PAPI_destroy_eventset(&EventSet);
    PAPI_unregister_thread();
    return NULL;
}

/**
 * Thread 2:
 * - Waits until t1 is running, then attempts to start the same event.
 * Expected: PAPI_start fails with PAPI_ECNFLCT (resource conflict).
 */
static void* thread_fn2(void* arg) {
    PAPI_register_thread();
    struct ThreadState* st = (struct ThreadState*)arg;

    int EventSet = PAPI_NULL;
    int papi_errno = PAPI_create_eventset(&EventSet);
    if (papi_errno != PAPI_OK) { NOTE("t2 create: %s", PAPI_strerror(papi_errno)); st->start_papi_errno = papi_errno; PAPI_unregister_thread(); return NULL; }

    papi_errno = PAPI_add_named_event(EventSet, g_event);
    if (papi_errno == PAPI_ENOEVNT) { SKIP("Event not supported on this platform"); }
    if (papi_errno == PAPI_ECNFLCT || papi_errno == PAPI_EPERM) { SKIP("Cannot add event due to HW/resource limits"); }
    if (papi_errno != PAPI_OK) { NOTE("t2 add: %s", PAPI_strerror(papi_errno)); st->start_papi_errno = papi_errno; (void)PAPI_destroy_eventset(&EventSet); PAPI_unregister_thread(); return NULL; }

    /* Busy-wait until t1 has started the event (adequate for a short test). */
    while (!atomic_load_explicit(&t1_started, memory_order_acquire)) { /* spin */ }

    papi_errno = PAPI_start(EventSet);
    st->start_papi_errno = papi_errno;
    if (papi_errno != PAPI_OK) {
        NOTE("t2 start expected fail: %s", PAPI_strerror(papi_errno));
    } else {
        NOTE("t2 start unexpectedly succeeded");
        long long v = 0; (void)PAPI_stop(EventSet, &v);
    }

    (void)PAPI_cleanup_eventset(EventSet);
    (void)PAPI_destroy_eventset(&EventSet);
    PAPI_unregister_thread();
    return NULL;
}

/**
 * Program entry:
 * - Parses harness options and optional event override.
 * - Ensures PAPI_AMDSMI_ROOT is set and PAPI is initialized for threading.
 * - Runs the two-thread contention test and evaluates pass/fail:
 *   PASS  => t1 start == PAPI_OK and t2 start == PAPI_ECNFLCT
 *   FAIL  => any other combination.
 */
int main(int argc, char** argv) {
    /* Unbuffer stdout so the final status line always shows promptly. */
    setvbuf(stdout, NULL, _IONBF, 0);

    harness_accept_tests_quiet(&argc, argv);
    HarnessOpts opts = parse_harness_cli(argc, argv);

    /* Optional override of the event: ./amdsmi_ctx_conflict "<event>" */
    if (argc > 1 && strncmp(argv[1], "--", 2) != 0) g_event = argv[1];

    const char* root = getenv("PAPI_AMDSMI_ROOT");
    if (!root || !*root) SKIP("PAPI_AMDSMI_ROOT not set");

    int papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) { NOTE("PAPI_library_init failed: %s", PAPI_strerror(papi_errno)); int e = eval_result(opts, 1); fflush(stdout); return e; }

    if (PAPI_thread_init(&get_tid) != PAPI_OK) { NOTE("PAPI_thread_init failed"); int e = eval_result(opts, 1); fflush(stdout); return e; }

    atomic_store_explicit(&t1_started, false, memory_order_relaxed);

    struct ThreadState s1;
    struct ThreadState s2;
    s1.start_papi_errno = PAPI_OK;
    s2.start_papi_errno = PAPI_OK;

    pthread_t th1, th2;
    pthread_create(&th1, NULL, thread_fn1, &s1);
    pthread_create(&th2, NULL, thread_fn2, &s2);
    pthread_join(th1, NULL);
    pthread_join(th2, NULL);

    if (opts.print) {
        printf("event: %s\n", g_event);
        printf("t1 start papi_errno: %d (%s)\n", s1.start_papi_errno, PAPI_strerror(s1.start_papi_errno));
        printf("t2 start papi_errno: %d (%s)\n", s2.start_papi_errno, PAPI_strerror(s2.start_papi_errno));
    }

    /* PASS when expected contention occurred; else FAIL. */
    int final_status = (s1.start_papi_errno == PAPI_OK && s2.start_papi_errno == PAPI_ECNFLCT) ? 0 : 1;
    if (final_status != 0) NOTE("Unexpected results (wanted t1 OK, t2 PAPI_ECNFLCT).");

    int exit_code = eval_result(opts, final_status);
    fflush(stdout);
    return exit_code;
}
