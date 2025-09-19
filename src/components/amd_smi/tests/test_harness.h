/**
 * @file    test_harness.h
 * @author  Dong Jun Woun
 *          djwoun@gmail.com
 * @brief   Minimal test-harness utilities for PAPI AMD-SMI tests:
 *          CLI parsing, quiet-mode handling, warnings accounting,
 *          and pass/fail reporting.
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"  /* for PAPI_* error codes used by helper macros */

/** Options controlling harness behavior. */
typedef struct HarnessOpts {
    bool print;         /**< Whether to print normal output. */
    bool expect_fail;   /**< If true, a nonzero return is considered PASS. */
    int  had_warning;   /**< Set to 1 if ENOEVNT/ECNFLCT/EPERM or any warning occurred. */
} HarnessOpts;

/** Global harness state used by macros. */
static HarnessOpts harness_opts;

/**
 * @brief Accept and normalize the positional quiet token.
 *
 * Recognizes the literal tokens "TESTS_QUIET" or "QUIET" on the command line,
 * removes them from @p argv so they aren't misinterpreted as positional args,
 * and sets TESTS_QUIET=1. If the TESTS_QUIET environment variable is set to a
 * non-literal value, that value is filtered out of @p argv and the variable is
 * unset so tests do not treat it as an argument.
 *
 * @param[in,out] argc Argument count.
 * @param[in,out] argv Argument vector.
 */
static inline void harness_accept_tests_quiet(int *argc, char **argv) {
    /* The PAPI test harness invokes each test with a single positional token
       holding the value of the TESTS_QUIET environment variable. Only the
       literal string "TESTS_QUIET" should trigger quiet mode. Any other value
       is dropped from argv and the environment variable is ignored. */

    char *badarg = NULL;
    const char *tq_env = getenv("TESTS_QUIET");
    if (tq_env && strcmp(tq_env, "TESTS_QUIET") != 0) {
        badarg = strdup(tq_env);  /* remember stray value to filter from argv */
        unsetenv("TESTS_QUIET");  /* ignore non-literal TESTS_QUIET */
    }

    int w = 1;
    int saw_quiet = 0;
    for (int r = 1; r < *argc; ++r) {
        const char *a = argv[r];
        if (a && (!strcmp(a, "TESTS_QUIET") || !strcmp(a, "QUIET"))) {
            saw_quiet = 1;
            continue;
        }
        if (badarg && a && strcmp(a, badarg) == 0) {
            /* discard unexpected TESTS_QUIET value */
            continue;
        }
        argv[w++] = argv[r];
    }
    argv[w] = NULL;
    *argc = w;
    if (saw_quiet) setenv("TESTS_QUIET", "1", 1);
    if (badarg) free(badarg);
}

/**
 * @brief Parse common harness CLI/environment options.
 *
 * Defaults to printing unless TESTS_QUIET is present. Mirrors src/run_tests.sh
 * behavior where invoking with -v unsets TESTS_QUIET (tests should emit output).
 *
 * Also sets/clears PAPI_AMDSMI_TEST_QUIET so individual tests can key off it.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Populated HarnessOpts (also stored in @ref harness_opts).
 */
static inline HarnessOpts parse_harness_cli(int argc, char **argv) {
    /* Default to printing unless the TESTS_QUIET token is present.
       This mirrors src/run_tests.sh where invoking with -v unsets
       TESTS_QUIET, signalling that tests should emit output. */
    harness_opts.print = true;
    harness_opts.expect_fail = false;
    harness_opts.had_warning = 0;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--expect=", 9) == 0) {
            const char *v = argv[i] + 9;
            harness_opts.expect_fail = (strcmp(v, "fail") == 0);
        }
    }

    /* Suppress output only if TESTS_QUIET is explicitly set. When
       run_tests.sh is invoked without -v it passes the literal token
       "TESTS_QUIET", which harness_accept_tests_quiet converts into
       this environment variable. */
    const char *tq = getenv("TESTS_QUIET");
    if (tq && *tq) harness_opts.print = false;

    if (!harness_opts.print) {
        const char* q = getenv("PAPI_AMDSMI_TEST_QUIET");
        if (!q || q[0] != '1') setenv("PAPI_AMDSMI_TEST_QUIET", "1", 1);
    } else {
        unsetenv("PAPI_AMDSMI_TEST_QUIET");
    }
    return harness_opts;
}

/**
 * @brief Evaluate the test result and print a final status line.
 *
 * A zero @p result_code is PASS unless @ref HarnessOpts::expect_fail is true,
 * in which case nonzero indicates PASS. If any warnings were recorded, output
 * "PASSED with WARNING".
 *
 * @param opts         The harness options in effect (warning flag may be
 *                     updated from the global state).
 * @param result_code  The test's return code.
 * @return 0 on PASS (per @p opts), 1 on FAIL.
 */
static inline int eval_result(HarnessOpts opts, int result_code) {
    if (harness_opts.had_warning) {
        opts.had_warning = harness_opts.had_warning;
    }

    bool passed = opts.expect_fail ? (result_code != 0) : (result_code == 0);
    if (passed) {
        if (opts.had_warning) printf("PASSED with WARNING\n");
        else                  printf("PASSED\n");
    } else {
        printf("FAILED!!!\n");
    }
    return passed ? 0 : 1;
}

/* ---------- Output helpers ---------- */

/** Print a note only when normal output is enabled. */
#define NOTE(...) do { \
    if (harness_opts.print) { fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n"); } \
} while (0)

/** Mark a warning (does not exit). */
#define WARNF(...) do { \
    harness_opts.had_warning = 1; \
    if (harness_opts.print) { fprintf(stdout, "WARNING: "); fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n"); } \
} while (0)

/* ---------- Cannot-conduct helpers ---------- */
/* Treat certain hardware/resource limitations as success-with-warning. */

/**
 * @brief Exit immediately as "PASSED with WARNING".
 * Prints an optional formatted warning message when output is enabled.
 */
#define EXIT_WARNING(...) do { \
    harness_opts.had_warning = 1; \
    if (harness_opts.print && *#__VA_ARGS__) { fprintf(stdout, "WARNING: "); fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n"); } \
    printf("PASSED with WARNING\n"); fflush(stdout); exit(0); \
} while (0)

/**
 * @brief If adding the event set fails due to unsupported or hardware/resource
 *        limits, exit as "PASSED with WARNING".
 *
 * Recognizes PAPI_ENOEVNT, PAPI_ECNFLCT, and PAPI_EPERM.
 */
#define EXIT_WARNING_ON_ADD(rc, evname) do { \
    if ((rc) == PAPI_ENOEVNT || (rc) == PAPI_ECNFLCT || (rc) == PAPI_EPERM) { \
        EXIT_WARNING("Event unavailable (%s): %s", \
            ((rc) == PAPI_ENOEVNT ? "ENOEVNT" : (rc) == PAPI_ECNFLCT ? "ECNFLCT" : "EPERM"), (evname)); \
    } \
} while (0)

/**
 * @brief If starting counters fails due to hardware/resource limits,
 *        exit as "PASSED with WARNING".
 *
 * Recognizes PAPI_ECNFLCT and PAPI_EPERM.
 */
#define EXIT_WARNING_ON_START(rc, ctx) do { \
    if ((rc) == PAPI_ECNFLCT || (rc) == PAPI_EPERM) { \
        EXIT_WARNING("Cannot start counters (%s): %s", (ctx), PAPI_strerror(rc)); \
    } \
} while (0)

/** Keep SKIP as a cannot-conduct success-with-warning. */
#define SKIP(reason) EXIT_WARNING("%s", (reason))

#endif /* TEST_HARNESS_H */
