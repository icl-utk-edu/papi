/*
 * (C) Copyright IBM Corp. 2006
 * Contributed by Kevin Corry <kevcorry@us.ibm.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sellcopies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *
 * pfmsetup
 *
 * Very simple command-line tool to drive the perfmon2 kernel API. Inspired
 * by the dmsetup tool from device-mapper.
 *
 * Compile with:
 *   gcc -Wall -o pfmsetup pfmsetup.c -lpfm
 *
 * Run with:
 *   pfmsetup <command_file>
 *
 * Available commands for the command_file:
 *
 *   create_context [options] <context_id>
 *      Create a new context for accessing the performance counters. Each new
 *      context automatically gets one event-set with an ID of 0.
 *        - options: --system
 *                   --no-overflow-msg
 *                   --block-on-notify
 *                   --sampler <sampler_name>
 *        - <context_id>: specify an integer that you want to associate with
 *                        the new context for use in other commands.
 *
 *   load_context <context_id> <event_set_id> <program_id|cpu_id>
 *      Attach the specified context and event-set to the specified program.
 *        - <context_id>: ID that you specified when creating the context.
 *        - <event_set_id>: ID that you specified when creating an event-set
 *                          within the given context. All contexts automatically
 *                          have an event-set with ID of 0.
 *        - <program_id|cpu_id>: ID that you specified when starting a program
 *                               with the run_program command, or the number of
 *                               the CPU to attach to for system-wide mode.
 *
 *   unload_context <context_id>
 *      Detach the specified context from the program that it's currently
 *      attached to.
 *        - <context_id>: ID that you specified when creating the context.
 *
 *   close_context <context_id>
 *      Clean up the specified context. After this call, the context_id will no
 *      longer be valid.
 *        - <context_id>: ID that you specified when creating the context.
 *
 *   write_pmc <context_id> <event_set_id> <<pmc_id> <pmc_value>>+
 *      Write one or more control register values.
 *        - <context_id>: ID that you specified when creating the context.
 *        - <event_set_id>: ID that you specified when creating an event-set
 *                          within the given context. All contexts automatically
 *                          have an event-set with ID of 0.
 *        - <pmc_id>: ID of the desired control register. See the register
 *                    mappings in the Perfmon kernel code to determine which
 *                    PMC represents the control register you're interested in.
 *        - <pmc_value>: Value to write into the specified PMC. You need to know
 *                       the exact numeric value - no translations are done from
 *                       event names or masks. Multiple PMC id/value pairs can
 *                       be given in one write_pmc command.
 *
 *   write_pmd <context_id> <event_set_id> <<pmd_id> <pmd_value>>+
 *      Write one or more data register values.
 *        - <context_id>: ID that you specified when creating the context.
 *        - <event_set_id>: ID that you specified when creating an event-set
 *                          within the given context. All contexts automatically
 *                          have an event-set with ID of 0.
 *        - <pmd_id>: ID of the desired data register. See the register
 *                    mappings in the Perfmon kernel code to determine which
 *                    PMD represents the control register you're interested in.
 *        - <pmd_value>: Value to write into the specified PMD. Multiple PMD
 *                       id/value pairs can be given in one write_pmd command.
 *
 *   read_pmd <context_id> <event_set_id> <pmd_id>+
 *      Read one or more data register values.
 *        - <context_id>: ID that you specified when creating the context.
 *        - <event_set_id>: ID that you specified when creating an event-set
 *                          within the given context. All contexts automatically
 *                          have an event-set with ID of 0.
 *        - <pmd_id>: ID of the desired data register. See the register
 *                    mappings in the Perfmon kernel code to determine which
 *                    PMD represents the control register you're interested in.
 *                    Multiple PMD IDs can be given in one read_pmd command.
 *
 *   start_counting <context_id> <event_set_id>
 *      Start counting using the specified context and event-set.
 *        - <context_id>: ID that you specified when creating the context.
 *        - <event_set_id>: ID that you specified when creating an event-set
 *                          within the given context. All contexts automatically
 *                          have an event-set with ID of 0.
 *
 *   stop_counting <context_id>
 *      Stop counting on the specified context.
 *        - <context_id>: ID that you specified when creating the context.
 *
 *   restart_counting <context_id>
 *      Restart counting on the specified context.
 *        - <context_id>: ID that you specified when creating the context.
 *
 *   create_eventset [options] <context_id> <event_set_id>
 *      Create a new event-set for an existing context.
 *        - options: --next-set <next_event_set_id>
 *                   --timeout <nanoseconds>
 *                   --switch-on-overflow
 *                   --exclude-idle
 *        - <context_id>: ID that you specified when creating the context.
 *        - <event_set_id>: specify an integer that you want to associate with
 *                          the new event-set for use in other commands.
 *
 *   delete_eventset <context_id> <event_set_id>
 *      Delete an existing event-set from an existing context.
 *        - <context_id>: ID that you specified when creating the context.
 *        - <event_set_id>: ID that you specified when creating the event-set.
 *
 *   getinfo_eventset <context_id> <event_set_id>
 *      Display information about an event-set.
 *        - <context_id>: ID that you specified when creating the context.
 *        - <event_set_id>: ID that you specified when creating the event-set.
 *
 *   run_program <program_id> <program name and arguments>
 *      First step in starting a program to monitor. In order to allow time to
 *      set up the counters to monitor the program, this command only forks a
 *      child process. It then suspends itself using ptrace. You must call the
 *      resume_program command to wake up the new child process and exec the
 *      desired program.
 *        - <program_id>: Specify an integer that you want to associate with
 *                        the program for use in other commands.
 *        - <program name and arguments>: Specify the program and its arguments
 *                                        exactly as you would on the command
 *                                        line.
 *
 *   resume_program <program_id>
 *      When a program is 'run', a child process is forked, but the child is
 *      ptrace'd before exec'ing the specified program. This gives you time to
 *      do any necessary setup to monitor the program. This resume_program
 *      command wakes up the child process and finishes exec'ing the desired
 *      program. If a context has been loaded and started for this program,
 *      then the counters will have actually started following this command.
 *        - <program_id>: ID that you specified when starting the program.
 *
 *   wait_on_program <program_id>
 *      Wait for a program to complete and exit. After this call, the program_id
 *      will no longer be valid.
 *        - <program_id>: ID that you specified when starting the program.
 *
 *   sleep <time_in_seconds)
 *      Sleep for the specified number of seconds. This could be used if you
 *      want to take measurements while a program is running, or if you're
 *      running a system-wide context.
 *
 * Blank lines in the command file and lines starting with '#' are ignored.
 *
 * Example command-file for use on an Intel P4/EM64T. This command-file creates
 * one context, starts 'dd' to read data from /dev/sda, loads the context onto
 * the 'dd' program, writes values into two PMCs (MSR_CRU_ESCR0 and
 * MSR_IQ_CCCR0) in order to set up for counting retired instructions, clears
 * one PMD (MSR_IQ_COUNTER0), starts the counters, resumes the 'dd' program,
 * waits for it to complete, and reads the number of instructions retired from
 * the PMD.
 *
 *   create_context 1
 *   run_program 1 dd if=/dev/sda of=/dev/null bs=1M count=1024
 *   load_context 1 0 1
 *   write_pmc 1 0 20 0x0400020c 29 0x04039000
 *   write_pmd 1 0 6 0
 *   start_counting 1 0
 *   resume_program 1
 *   wait_on_program 1
 *   read_pmd 1 0 6
 *   close_context 1
 *
 * The output will look like this:
 *
 *   pfmsetup: Created context 1 with file-descriptor 4.
 *   pfmsetup: Started program 1: 'dd'.
 *   pfmsetup: Loaded context 1, event-set 0 onto program 1.
 *   pfmsetup: Wrote to PMC 20: 0x400020c
 *   pfmsetup: Wrote to PMC 29: 0x4039000
 *   pfmsetup: Wrote to PMD 6: 0
 *   pfmsetup: Started counting for context 1, event-set 0.
 *   pfmsetup: Resumed program 1.
 *   1024+0 records in
 *   1024+0 records out
 *   pfmsetup: Waited for program 1 to complete.
 *   pfmsetup: Read from PMD 6: 415218111
 *   pfmsetup: Closed and freed context 1.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <sched.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <perfmon/perfmon.h>
#include <perfmon/perfmon_dfl_smpl.h>

#define FALSE 0
#define TRUE  1

#define WHITESPACE " \t\n"
#define MAX_TOKENS 32
#define PFMSETUP_NAME "pfmsetup"

#define USAGE(f, x...)     printf(PFMSETUP_NAME ": USAGE: " f "\n" , ## x)
#define LOG_ERROR(f, x...) printf(PFMSETUP_NAME ": Error: %s: " f "\n", __FUNCTION__ , ## x)
#define LOG_INFO(f, x...)  printf(PFMSETUP_NAME ": " f "\n" , ## x)

typedef int (*command_fn)(int argc, char **argv);

struct command {
	const char *full_name;
	const char *short_name;
	const char *help;
	command_fn fn;
	int min_args;
};

struct context {
	int id;
	int fd;
	int cpu;
	uint32_t ctx_flags;
	pfm_dfl_smpl_arg_t smpl_arg;
	struct event_set *event_sets;
	struct context *next;
};

struct event_set {
	int id;
	struct event_set *next;
};

struct program {
	int id;
	pid_t pid;
	struct program *next;
};


/* Global list of all contexts that have been created. List is ordered by
 * context id. Each context contains a list of event-sets belonging to that
 * context, which is ordered by event-set id.
 */
static struct context *contexts = NULL;


/* Global list of all programs that have been started.
 * List is ordered by program id.
 */
static struct program *programs = NULL;


/*
 * Routines to manipulate the context, event-set, and program lists.
 */

static struct context *find_context(int ctx_id)
{
	struct context *ctx;

	for (ctx = contexts; ctx; ctx = ctx->next) {
		if (ctx->id == ctx_id) {
			break;
		}
	}

	return ctx;
}

static void insert_context(struct context *ctx)
{
	struct context **next_ctx;

	for (next_ctx = &contexts;
	     *next_ctx && (*next_ctx)->id < ctx->id;
	     next_ctx = &((*next_ctx)->next)) {
		;
	}

	ctx->next = *next_ctx;
	*next_ctx = ctx;
}

static void remove_context(struct context *ctx)
{
	struct context **next_ctx;

	for (next_ctx = &contexts; *next_ctx; next_ctx = &((*next_ctx)->next)) {
		if (*next_ctx == ctx) {
			*next_ctx = ctx->next;
			break;
		}
	}
}

static struct event_set *find_event_set(struct context *ctx, int event_set_id)
{
	struct event_set *evt;

	for (evt = ctx->event_sets; evt; evt = evt->next) {
		if (evt->id == event_set_id) {
			break;
		}
	}

	return evt;
}

static void insert_event_set(struct context *ctx, struct event_set *evt)
{
	struct event_set **next_evt;

	for (next_evt = &ctx->event_sets;
	     *next_evt && (*next_evt)->id < evt->id;
	     next_evt = &((*next_evt)->next)) {
		;
	}

	evt->next = *next_evt;
	*next_evt = evt;
}

static struct program *find_program(int program_id)
{
	struct program *prog;

	for (prog = programs; prog; prog = prog->next) {
		if (prog->id == program_id) {
			break;
		}
	}

	return prog;
}

static void insert_program(struct program *prog)
{
	struct program **next_prog;

	for (next_prog = &programs;
	     *next_prog && (*next_prog)->id < prog->id;
	     next_prog = &((*next_prog)->next)) {
		;
	}

	prog->next = *next_prog;
	*next_prog = prog;
}

static void remove_program(struct program *prog)
{
	struct program **next_prog;

	for (next_prog = &programs;
	     *next_prog;
	     next_prog = &((*next_prog)->next)) {
		if (*next_prog == prog) {
			*next_prog = prog->next;
			break;
		}
	}
}

/**
 * set_affinity
 *
 * When loading or unloading a system-wide context, we must pin the pfmsetup
 * process to that CPU before making the system call. Also, get the current
 * affinity and return it to the caller so we can change it back later.
 **/
static int set_affinity(int cpu, cpu_set_t *old_cpu_set)
{
	cpu_set_t new_cpu_set;
	int rc;

	rc = sched_getaffinity(0, sizeof(*old_cpu_set), old_cpu_set);
	if (rc) {
		rc = errno;
		LOG_ERROR("Can't get current process affinity mask: %d\n", rc);
		return rc;
	}

	CPU_ZERO(&new_cpu_set);
	CPU_SET(cpu, &new_cpu_set);
	rc = sched_setaffinity(0, sizeof(new_cpu_set), &new_cpu_set);
	if (rc) {
		rc = errno;
		LOG_ERROR("Can't set process affinity to CPU %d: %d\n", cpu, rc);
		return rc;
	}

	return 0;
}

/**
 * revert_affinity
 *
 * Reset the process affinity to the specified mask.
 **/
static void revert_affinity(cpu_set_t *old_cpu_set)
{
	int rc;

	rc = sched_setaffinity(0, sizeof(*old_cpu_set), old_cpu_set);
	if (rc) {
		/* Not a fatal error if we can't reset the affinity. */
		LOG_INFO("Can't revert process affinity to original value.\n");
	}
}

/**
 * create_context
 *
 * Arguments: [options] <context_id>
 * Options: --system
 *          --no-overflow-msg
 *          --block-on-notify
 *          --sampler <sampler_name>
 *
 * Call the pfm_create_context system-call to create a new perfmon context.
 * Add a new entry to the global 'contexts' list.
 **/
static int create_context(int argc, char **argv)
{
	pfm_dfl_smpl_arg_t smpl_arg;
	struct context *new_ctx = NULL;
	char *sampler_name = NULL;
	void *smpl_p;
	int no_overflow_msg = FALSE;
	int block_on_notify = FALSE;
	int system_wide = FALSE;
	int c, ctx_id = 0;
	int rc;
	uint32_t ctx_flags;
	size_t sz;

	struct option long_opts[] = {
		{"sampler",         required_argument, NULL, 1},
		{"system",          no_argument,       NULL, 2},
		{"no-overflow-msg", no_argument,       NULL, 3},
		{"block-on-notify", no_argument,       NULL, 4},
		{NULL,              0,                 NULL, 0} };

	ctx_flags = 0;

	opterr = 0;
	optind = 0;
	while ((c = getopt_long_only(argc, argv, "",
				     long_opts, NULL)) != EOF) {
		switch (c) {
		case 1:
			sampler_name = optarg;
			break;
		case 2:
			system_wide = TRUE;
			break;
		case 3:
			no_overflow_msg = TRUE;
			break;
		case 4:
			block_on_notify = TRUE;
			break;
		default:
			LOG_ERROR("invalid option: %c", optopt);
			rc = EINVAL;
			goto error;
		}
	}

	if (argc < optind + 1) {
		USAGE("create_context [options] <context_id>");
		rc = EINVAL;
		goto error;
	}

	ctx_id = strtoul(argv[optind], NULL, 0);
	if (ctx_id <= 0) {
		LOG_ERROR("Invalid context ID (%s). Must be a positive "
			  "integer.", argv[optind]);
		rc = EINVAL;
		goto error;
	}

	/* Make sure we don't already have a context with this ID. */
	new_ctx = find_context(ctx_id);
	if (new_ctx) {
		LOG_ERROR("Context with ID %d already exists.", ctx_id);
		rc = EINVAL;
		goto error;
	}

	if (sampler_name) {
		smpl_arg.buf_size = getpagesize();
		smpl_p = &smpl_arg;
		sz = sizeof(smpl_arg);
	} else {
		smpl_p = NULL;
		sz = 0;
	}

	ctx_flags = (system_wide     ? PFM_FL_SYSTEM_WIDE  : 0) |
		    (no_overflow_msg ? PFM_FL_OVFL_NO_MSG  : 0) |
		    (block_on_notify ? PFM_FL_NOTIFY_BLOCK : 0);

	if (sampler_name)
		ctx_flags |= PFM_FL_SMPL_FMT;

	rc = pfm_create(ctx_flags, NULL, sampler_name, smpl_p, sz);
	if (rc == -1) {
		rc = errno;
		LOG_ERROR("pfm_create_context system call returned "
			  "an error: %d.", rc);
		goto error;
	}

	/* Allocate and initialize a new context structure and add it to the
	 * global list. Every new context automatically gets one event_set
	 * with an event ID of 0.
	 */
	new_ctx = calloc(1, sizeof(*new_ctx));
	if (!new_ctx) {
		LOG_ERROR("Can't allocate structure for new context %d.",
			  ctx_id);
		rc = ENOMEM;
		goto error;
	}

	new_ctx->event_sets = calloc(1, sizeof(*(new_ctx->event_sets)));
	if (!new_ctx->event_sets) {
		LOG_ERROR("Can't allocate event-set structure for new "
			  "context %d.", ctx_id);
		rc = ENOMEM;
		goto error;
	}

	new_ctx->id = ctx_id;
	new_ctx->fd = rc;
	new_ctx->cpu = -1;
	new_ctx->ctx_flags = ctx_flags;
	new_ctx->smpl_arg = smpl_arg;

	insert_context(new_ctx);

	LOG_INFO("Created context %d with file-descriptor %d.",
		 new_ctx->id, new_ctx->fd);

	return 0;

error:
	if (new_ctx) {
		close(new_ctx->fd);
		free(new_ctx->event_sets);
		free(new_ctx);
	}
	return rc;
}

/**
 * load_context
 *
 * Arguments: <context_id> <event_set_id> <program_id|cpu_id>
 *
 * Call the pfm_load_context system-call to load a perfmon context into the
 * system's performance monitoring unit.
 **/
static int load_context(int argc, char **argv)
{
	struct context *ctx;
	struct event_set *evt;
	struct program *prog;
	cpu_set_t old_cpu_set;
	int ctx_id, event_set_id, program_id;
	int system_wide, rc;
	int load_pid =  0;

	ctx_id = strtoul(argv[1], NULL, 0);
	event_set_id = strtoul(argv[2], NULL, 0);
	program_id = strtoul(argv[3], NULL, 0);

	if (ctx_id <= 0 || event_set_id < 0 || program_id < 0) {
		LOG_ERROR("context ID, event-set ID, and program/CPU ID must "
			  "be positive integers.");
		return EINVAL;
	}

	/* Find the context, event_set, and program in the global lists. */
	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	evt = find_event_set(ctx, event_set_id);
	if (!evt) {
		LOG_ERROR("Can't find event-set with ID %d in context %d.",
			  event_set_id, ctx_id);
		return EINVAL;
	}
	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide) {
		if (ctx->cpu >= 0) {
			LOG_ERROR("Trying to load context %d which is already "
				  "loaded on CPU %d.\n", ctx_id, ctx->cpu);
			return EBUSY;
		}

		rc = set_affinity(program_id, &old_cpu_set);
		if (rc) {
			return rc;
		}

		/* Specify the CPU as the PID. */
		load_pid = program_id;
	} else {
		prog = find_program(program_id);
		if (!prog) {
			LOG_ERROR("Can't find program with ID %d.", program_id);
			return EINVAL;
		}
		load_pid = prog->pid;
	}

	rc = pfm_attach(ctx->fd, 0, load_pid);
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_attach  system call returned "
			  "an error: %d.", rc);
		return rc;
	}

	if (system_wide) {
		/* Keep track of which CPU this context is loaded on. */
		ctx->cpu = program_id;

		revert_affinity(&old_cpu_set);
	}

	LOG_INFO("Loaded context %d, event-set %d onto %s %d.",
		 ctx_id, event_set_id, system_wide ? "cpu" : "program",
		 program_id);

	return 0;
}

/**
 * unload_context
 *
 * Arguments: <context_id>
 *
 * Call the pfm_unload_context system-call to unload a perfmon context from
 * the system's performance monitoring unit.
 **/
static int unload_context(int argc, char **argv)
{
	struct context *ctx;
	cpu_set_t old_cpu_set;
	int system_wide;
	int ctx_id;
	int rc;

	ctx_id = strtoul(argv[1], NULL, 0);
	if (ctx_id <= 0) {
		LOG_ERROR("context ID must be a positive integer.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide) {
		if (ctx->cpu < 0) {
			/* This context isn't loaded on any CPU. */
			LOG_ERROR("Trying to unload context %d that isn't "
				  "loaded.\n", ctx_id);
			return EINVAL;
		}

		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			return rc;
		}
	}

	rc = pfm_attach(ctx->fd, 0, PFM_NO_TARGET);
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_attach(detach) system call returned "
			  "an error: %d.", rc);
		return rc;
	}

	if (system_wide) {
		ctx->cpu = -1;
		revert_affinity(&old_cpu_set);
	}

	LOG_INFO("Unloaded context %d.", ctx_id);

	return 0;
}

/**
 * close_context
 *
 * Arguments: <context_id>
 *
 * Close the context's file descriptor, remove it from the global list, and
 * free the context data structures.
 **/
static int close_context(int argc, char **argv)
{
	struct context *ctx;
	struct event_set *evt, *next_evt;
	int ctx_id;

	ctx_id = strtoul(argv[1], NULL, 0);
	if (ctx_id <= 0) {
		LOG_ERROR("context ID must be a positive integer.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	/* There's no perfmon system-call to delete a context. We simply call
	 * close on the file handle.
	 */
	close(ctx->fd);
	remove_context(ctx);

	for (evt = ctx->event_sets; evt; evt = next_evt) {
		next_evt = evt->next;
		free(evt);
	}
	free(ctx);

	LOG_INFO("Closed and freed context %d.", ctx_id);

	return 0;
}

/**
 * write_pmc
 *
 * Arguments: <context_id> <event_set_id> <<pmc_id> <pmc_value>>+
 *
 * Write values to one or more control registers.
 **/
static int write_pmc(int argc, char **argv)
{
	struct context *ctx;
	struct event_set *evt;
	pfarg_pmr_t *pmc_args = NULL;
	cpu_set_t old_cpu_set;
	int ctx_id, event_set_id;
	int pmc_id, num_pmcs;
	unsigned long long pmc_value;
	int system_wide, i, rc;

	ctx_id = strtoul(argv[1], NULL, 0);
	event_set_id = strtoul(argv[2], NULL, 0);

	if (ctx_id <= 0 || event_set_id < 0) {
		LOG_ERROR("context ID and event-set ID must be "
			  "positive integers.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	evt = find_event_set(ctx, event_set_id);
	if (!evt) {
		LOG_ERROR("Can't find event-set with ID %d in context %d.",
			  event_set_id, ctx_id);
		return EINVAL;
	}

	/* Allocate an array of PMC structures. */
	num_pmcs = (argc - 3) / 2;
	pmc_args = calloc(num_pmcs, sizeof(*pmc_args));
	if (!pmc_args) {
		LOG_ERROR("Can't allocate PMC argument array.");
		return ENOMEM;
	}

	for (i = 0; i < num_pmcs; i++) {
		pmc_id = strtoul(argv[3 + i*2], NULL, 0);
		pmc_value = strtoull(argv[4 + i*2], NULL, 0);

		if (pmc_id < 0) {
			LOG_ERROR("PMC ID must be a positive integer.");
			rc = EINVAL;
			goto out;
		}

		pmc_args[i].reg_num = pmc_id;
		pmc_args[i].reg_set = evt->id;
		pmc_args[i].reg_value = pmc_value;
	}

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide && ctx->cpu >= 0) {
		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			goto out;
		}
	}

	rc = pfm_write(ctx->fd, 0, PFM_RW_PMC, pmc_args, num_pmcs * sizeof(*pmc_args));
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_write system call returned "
			  "an error: %d.", rc);
		goto out;
	}

	if (system_wide && ctx->cpu >= 0) {
		revert_affinity(&old_cpu_set);
	}
out:
	free(pmc_args);
	return rc;
}

/**
 * write_pmd
 *
 * Arguments: <context_id> <event_set_id> <<pmd_id> <pmd_value>>+
 *
 * FIXME: Add options for other fields in pfarg_pmd_t.
 **/
static int write_pmd(int argc, char **argv)
{
	struct context *ctx;
	struct event_set *evt;
	pfarg_pmr_t *pmd_args = NULL;
	cpu_set_t old_cpu_set;
	int ctx_id, event_set_id;
	int pmd_id, num_pmds;
	unsigned long long pmd_value;
	int system_wide, i, rc;

	ctx_id = strtoul(argv[1], NULL, 0);
	event_set_id = strtoul(argv[2], NULL, 0);

	if (ctx_id <= 0 || event_set_id < 0) {
		LOG_ERROR("context ID and event-set ID must be "
			  "positive integers.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	evt = find_event_set(ctx, event_set_id);
	if (!evt) {
		LOG_ERROR("Can't find event-set with ID %d in context %d.",
			  event_set_id, ctx_id);
		return EINVAL;
	}

	/* Allocate an array of PMD structures. */
	num_pmds = (argc - 3) / 2;
	pmd_args = calloc(num_pmds, sizeof(*pmd_args));
	if (!pmd_args) {
		LOG_ERROR("Can't allocate PMD argument array.");
		return ENOMEM;
	}

	for (i = 0; i < num_pmds; i++) {
		pmd_id = strtoul(argv[3 + i*2], NULL, 0);
		pmd_value = strtoull(argv[4 + i*2], NULL, 0);

		if (pmd_id < 0) {
			LOG_ERROR("PMD ID must be a positive integer.");
			rc = EINVAL;
			goto out;
		}

		pmd_args[i].reg_num = pmd_id;
		pmd_args[i].reg_set = evt->id;
		pmd_args[i].reg_value = pmd_value;
	}

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide && ctx->cpu >= 0) {
		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			goto out;
		}
	}

	rc = pfm_write(ctx->fd, 0, PFM_RW_PMD, pmd_args, num_pmds * sizeof(*pmd_args));
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_write system call returned "
			  "an error: %d.", rc);
		goto out;
	}

	if (system_wide && ctx->cpu >= 0) {
		revert_affinity(&old_cpu_set);
	}
out:
	free(pmd_args);
	return rc;
}

/**
 * read_pmd
 *
 * Arguments: <context_id> <event_set_id> <pmd_id>+
 *
 * FIXME: Add options for other fields in pfarg_pmd_t.
 **/
static int read_pmd(int argc, char **argv)
{
	struct context *ctx;
	struct event_set *evt;
	pfarg_pmr_t *pmd_args = NULL;
	cpu_set_t old_cpu_set;
	int ctx_id, event_set_id;
	int pmd_id, num_pmds;
	int system_wide, i, rc;

	ctx_id = strtoul(argv[1], NULL, 0);
	event_set_id = strtoul(argv[2], NULL, 0);

	if (ctx_id <= 0 || event_set_id < 0) {
		LOG_ERROR("context ID and event-set ID must be "
			  "positive integers.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	evt = find_event_set(ctx, event_set_id);
	if (!evt) {
		LOG_ERROR("Can't find event-set with ID %d in context %d.",
			  event_set_id, ctx_id);
		return EINVAL;
	}

	/* Allocate an array of PMD structures. */
	num_pmds = argc - 3;
	pmd_args = calloc(num_pmds, sizeof(*pmd_args));
	if (!pmd_args) {
		LOG_ERROR("Can't allocate PMD argument array.");
		return ENOMEM;
	}

	for (i = 0; i < num_pmds; i++) {
		pmd_id = strtoul(argv[3 + i], NULL, 0);
		if (pmd_id < 0) {
			LOG_ERROR("PMD ID must be a positive integer.");
			rc = EINVAL;
			goto out;
		}

		pmd_args[i].reg_num = pmd_id;
		pmd_args[i].reg_set = evt->id;
	}

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide && ctx->cpu >= 0) {
		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			goto out;
		}
	}

	rc = pfm_read(ctx->fd, 0, PFM_RW_PMD, pmd_args, num_pmds * sizeof(*pmd_args));
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_read system call returned "
			  "an error: %d.", rc);
		goto out;
	}

	if (system_wide && ctx->cpu >= 0) {
		revert_affinity(&old_cpu_set);
	}
out:
	free(pmd_args);
	return rc;
}

/**
 * start_counting
 *
 * Arguments: <context_id> <event_set_id>
 *
 * Call the pfm_start system-call to start counting for a perfmon context
 * that was previously stopped.
 **/
static int start_counting(int argc, char **argv)
{
	struct context *ctx;
	struct event_set *evt;
	cpu_set_t old_cpu_set;
	int ctx_id, event_set_id;
	int system_wide, rc;

	ctx_id = strtoul(argv[1], NULL, 0);
	event_set_id = strtoul(argv[2], NULL, 0);

	if (ctx_id <= 0 || event_set_id < 0) {
		LOG_ERROR("context ID and event-set ID must be "
			  "positive integers.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	evt = find_event_set(ctx, event_set_id);
	if (!evt) {
		LOG_ERROR("Can't find event-set with ID %d in context %d.",
			  event_set_id, ctx_id);
		return EINVAL;
	}

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide && ctx->cpu >= 0) {
		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			return rc;
		}
	}

	rc = pfm_set_state(ctx->fd, 0, PFM_ST_START);
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_set_state system call returned an error: %d.", rc);
		return rc;
	}

	if (system_wide && ctx->cpu >= 0) {
		revert_affinity(&old_cpu_set);
	}

	LOG_INFO("Started counting for context %d, event-set %d.",
		 ctx_id, event_set_id);

	return 0;
}

/**
 * stop_counting
 *
 * Arguments: <context_id>
 *
 * Call the pfm_stop system-call to stop counting for a perfmon context that
 * was previously loaded.
 **/
static int stop_counting(int argc, char **argv)
{
	struct context *ctx;
	cpu_set_t old_cpu_set;
	int system_wide;
	int ctx_id;
	int rc;

	ctx_id = strtoul(argv[1], NULL, 0);

	if (ctx_id <= 0) {
		LOG_ERROR("context ID must be a positive integer.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide && ctx->cpu >= 0) {
		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			return rc;
		}
	}

	rc = pfm_set_state(ctx->fd, 0, PFM_ST_STOP);
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_set_state(stop) system call returned an error: %d.", rc);
		return rc;
	}

	if (system_wide && ctx->cpu >= 0) {
		revert_affinity(&old_cpu_set);
	}

	LOG_INFO("Stopped counting for context %d.", ctx_id);

	return 0;
}

/**
 * restart_counting
 *
 * Arguments: <context_id>
 *
 * Call the pfm_restart system-call to clear the data counters and start
 * counting from zero for a perfmon context that was previously loaded.
 **/
static int restart_counting(int argc, char **argv)
{
	struct context *ctx;
	cpu_set_t old_cpu_set;
	int system_wide;
	int ctx_id;
	int rc;

	ctx_id = strtoul(argv[1], NULL, 0);

	if (ctx_id <= 0) {
		LOG_ERROR("context ID must be a positive integer.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide && ctx->cpu >= 0) {
		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			return rc;
		}
	}

	rc = pfm_set_state(ctx->fd, 0, PFM_ST_RESTART);
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_set_state(restart) system call returned an error: %d.", rc);
		return rc;
	}

	if (system_wide && ctx->cpu >= 0) {
		revert_affinity(&old_cpu_set);
	}

	LOG_INFO("Restarted counting for context %d.", ctx_id);

	return 0;
}

/**
 * create_eventset
 *
 * Arguments: [options] <context_id> <event_set_id>
 * Options: --timeout <nanoseconds>
 *          --switch-on-overflow
 *          --exclude-idle
 **/
static int create_eventset(int argc, char **argv)
{
	pfarg_set_desc_t set_arg;
	struct context *ctx;
	struct event_set *evt;
	cpu_set_t old_cpu_set;
	int ctx_id, event_set_id;
	unsigned long timeout = 0;
	int switch_on_overflow = FALSE;
	int switch_on_timeout = FALSE;
	int exclude_idle = FALSE;
	int new_set = FALSE;
	int system_wide,c, rc;
	struct option long_opts[] = {
		{"next-set",           required_argument, NULL, 1},
		{"timeout",            required_argument, NULL, 2},
		{"switch-on-overflow", no_argument,       NULL, 3},
		{"exclude-idle",       no_argument,       NULL, 4},
		{NULL,                 0,                 NULL, 0} };

	memset(&set_arg, 0, sizeof(set_arg));

	opterr = 0;
	optind = 0;
	while ((c = getopt_long_only(argc, argv, "",
				     long_opts, NULL)) != EOF) {
		switch (c) {
		case 1:
			timeout = strtoul(optarg, NULL, 0);
			if (!timeout) {
				LOG_ERROR("timeout must be a "
					  "non-zero integer.");
				return EINVAL;
			}
			switch_on_timeout = TRUE;
			break;
		case 2:
			switch_on_overflow = TRUE;
			break;
		case 3:
			exclude_idle = TRUE;
			break;
		default:
			LOG_ERROR("invalid option: %c", optopt);
			return EINVAL;
		}
	}
	(void)exclude_idle;

	if (argc < optind + 2) {
		USAGE("create_eventset [options] <context_id> <event_set_id>");
		return EINVAL;
	}

	ctx_id = strtoul(argv[optind], NULL, 0);
	event_set_id = strtoul(argv[optind+1], NULL, 0);

	if (ctx_id <= 0 || event_set_id < 0) {
		LOG_ERROR("context ID and event-set ID must be "
			  "positive integers.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	if (switch_on_timeout && switch_on_overflow) {
		LOG_ERROR("Cannot switch set %d (context %d) on both "
			  "timeout and overflow.", event_set_id, ctx_id);
		return EINVAL;
	}

	evt = find_event_set(ctx, event_set_id);
	if (!evt) {
		evt = calloc(1, sizeof(*evt));
		if (!evt) {
			LOG_ERROR("Can't allocate structure for new event-set "
				  "%d in context %d.", event_set_id, ctx_id);
			return ENOMEM;
		}
		evt->id = event_set_id;
		new_set = TRUE;
	}

	set_arg.set_id = event_set_id;
	set_arg.set_timeout = timeout; /* in nanseconds */
	set_arg.set_flags = (switch_on_overflow ? PFM_SETFL_OVFL_SWITCH : 0) |
			    (switch_on_timeout  ? PFM_SETFL_TIME_SWITCH : 0);

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide && ctx->cpu >= 0) {
		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			free(evt);
			return rc;
		}
	}

	rc = pfm_create_sets(ctx->fd, 0, &set_arg, 1);
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_create_sets system call returned "
			  "an error: %d.", rc);
		free(evt);
		return rc;
	}

	if (system_wide && ctx->cpu >= 0) {
		revert_affinity(&old_cpu_set);
	}

	if (new_set) {
		insert_event_set(ctx, evt);
	}

	LOG_INFO("%s event-set %d in context %d.",
		 new_set ? "Created" : "Modified", event_set_id, ctx_id);
	if (switch_on_timeout) {
		LOG_INFO("   Actual timeout set to %llu ns.",
			 (unsigned long long)set_arg.set_timeout);
	}

	return 0;
}

/**
 * delete_eventset
 *
 * Arguments: <context_id> <event_set_id>
 **/
static int delete_eventset(int argc, char **argv)
{
	LOG_ERROR("pfm_delete_evtsets not supported in v3.x");
	return EINVAL;
}

/**
 * getinfo_eventset
 *
 * Arguments: <context_id> <event_set_id>
 **/
static int getinfo_eventset(int argc, char **argv)
{
	pfarg_set_info_t set_arg;
	struct context *ctx;
	struct event_set *evt;
	cpu_set_t old_cpu_set;
	int ctx_id, event_set_id;
	int system_wide, rc;

	memset(&set_arg, 0, sizeof(set_arg));

	ctx_id = strtoul(argv[1], NULL, 0);
	event_set_id = strtoul(argv[2], NULL, 0);

	if (ctx_id <= 0 || event_set_id < 0) {
		LOG_ERROR("context ID and event-set ID must be "
			  "positive integers.");
		return EINVAL;
	}

	ctx = find_context(ctx_id);
	if (!ctx) {
		LOG_ERROR("Can't find context with ID %d.", ctx_id);
		return EINVAL;
	}

	evt = find_event_set(ctx, event_set_id);
	if (!evt) {
		LOG_ERROR("Can't find event-set with ID %d in context %d.",
			  event_set_id, ctx_id);
		return EINVAL;
	}

	set_arg.set_id = evt->id;

	system_wide = ctx->ctx_flags & PFM_FL_SYSTEM_WIDE;
	if (system_wide && ctx->cpu >= 0) {
		rc = set_affinity(ctx->cpu, &old_cpu_set);
		if (rc) {
			return rc;
		}
	}

	rc = pfm_getinfo_sets(ctx->fd, 0, &set_arg, 1);
	if (rc) {
		rc = errno;
		LOG_ERROR("pfm_getinfo_evtsets system call returned "
			  "an error: %d.", rc);
		return rc;
	}

	if (system_wide && ctx->cpu >= 0) {
		revert_affinity(&old_cpu_set);
	}

	LOG_INFO("Got info for event-set %d in context %d.", event_set_id, ctx_id);
	LOG_INFO("   Runs: %llu", (unsigned long long)set_arg.set_runs);
	LOG_INFO("   Timeout: %"PRIu64, set_arg.set_timeout);

	return 0;
}

/**
 * run_program
 *
 * Arguments: <program_id> <program name and arguments>
 *
 * Start the specified program. After fork'ing but before exec'ing, ptrace
 * the child so it will remain suspended until a corresponding resume_program
 * command. We do this so we can load a context for the program before it
 * actually starts running. This logic is taken from the task.c example in
 * the libpfm source code tree.
 **/
static int run_program(int argc, char **argv)
{
	struct program *prog;
	int program_id;
	pid_t pid;
	int rc;

	program_id = strtoul(argv[1], NULL, 0);
	if (program_id <= 0) {
		LOG_ERROR("program ID must be a positive integer.");
		return EINVAL;
	}

	/* Make sure we haven't already started a program with this ID. */
	prog = find_program(program_id);
	if (prog) {
		LOG_ERROR("Program with ID %d already exists.", program_id);
		return EINVAL;
	}

	prog = calloc(1, sizeof(*prog));
	if (!prog) {
		LOG_ERROR("Can't allocate new program structure to run '%s'.",
			  argv[2]);
		return ENOMEM;
	}

	prog->id = program_id;

	pid = fork();
	if (pid == -1) {
		/* Error fork'ing. */
		LOG_ERROR("Unable to fork child process.");
		return EINVAL;

	} else if (!pid) {
		/* Child */

		/* This will cause the program to stop before executing the
		 * first user level instruction. We can only load a context
		 * if the program is in the STOPPED state. This child
		 * process will sit here until we've process a resume_program
		 * command.
		 */
		rc = ptrace(PTRACE_TRACEME, 0, NULL, NULL);
		if (rc) {
			rc = errno;
			LOG_ERROR("Error ptrace'ing '%s': %d", argv[2], rc);
			exit(rc);
		}

		execvp(argv[2], argv + 2);

		rc = errno;
		LOG_ERROR("Error exec'ing '%s': %d", argv[2], rc);
		exit(rc);
	}

	/* Parent */
	prog->pid = pid;
	insert_program(prog);

	/* Wait for the child to exec. */
	waitpid(pid, &rc, WUNTRACED);

	/* Check if process exited early. */
	if (WIFEXITED(rc)) {
		LOG_ERROR("Program '%s' exited too early with status "
			  "%d", argv[2], WEXITSTATUS(rc));
		return WEXITSTATUS(rc);
	}

	LOG_INFO("Started program %d: '%s'.", program_id, argv[2]);

	return 0;
}

/**
 * resume_program
 *
 * Arguments: <program_id>
 *
 * A program started with run_program must be 'resumed' before it actually
 * begins running. This allows us to load a context to the process and
 * start the counters before the program executes any code.
 **/
static int resume_program(int argc, char **argv)
{
	struct program *prog;
	int program_id;
	int rc;

	program_id = strtoul(argv[1], NULL, 0);
	if (program_id <= 0) {
		LOG_ERROR("program ID must be a positive integer.");
		return EINVAL;
	}

	prog = find_program(program_id);
	if (!prog) {
		LOG_ERROR("Can't find program with ID %d.", program_id);
		return EINVAL;
	}

	/* Call ptrace to resume execution of the process. If a context has
	 * been loaded and the counters started, this is where monitoring
	 * is effectively activated.
	 */
	rc = ptrace(PTRACE_DETACH, prog->pid, NULL, 0);
	if (rc) {
		rc = errno;
		LOG_ERROR("Error detaching program %d.\n", prog->id);
		return rc;
	}

	LOG_INFO("Resumed program %d.", program_id);

	return 0;
}

/**
 * wait_on_program
 *
 * Arguments: <program_id>
 *
 * Wait for the specified program to complete and exit.
 **/
static int wait_on_program(int argc, char **argv)
{
	struct program *prog;
	int program_id;
	int rc;

	program_id = strtoul(argv[1], NULL, 0);
	if (program_id <= 0) {
		LOG_ERROR("program ID must be a positive integer.");
		return EINVAL;
	}

	prog = find_program(program_id);
	if (!prog) {
		LOG_ERROR("Can't find program with ID %d.", program_id);
		return EINVAL;
	}

	waitpid(prog->pid, &rc, 0);

	/* The program has exitted, but if there was a context loaded on that
	 * process, it will still have the latest counts available to read.
	 */

	remove_program(prog);
	free(prog);

	LOG_INFO("Waited for program %d to complete.", program_id);

	return 0;
}

/**
 * _sleep
 *
 * Arguments: <time in seconds>
 *
 * Wait for the specified number of seconds.
 **/
static int _sleep(int argc, char **argv)
{
	int seconds;

	seconds = strtoul(argv[1], NULL, 0);
	if (seconds < 0) {
		LOG_ERROR("time in seconds must be a positive integer.");
		return EINVAL;
	}

	LOG_INFO("Sleeping for %d seconds.", seconds);

	while (seconds > 0)
		seconds = sleep(seconds);

	LOG_INFO("Done sleeping.");

	return 0;
}


/**
 * _commands
 *
 * Array to describe all the available commands, their options, and the
 * routines that will process the commands.
 *
 * The concept for this array and the code to search it comes from the dmsetup
 * program in the device-mapper project.
 **/
static struct command _commands[] = {

	{ "create_context", "cc",
	  "<context_id> [--system] [--no-overflow-msg] "
	    "[--block-on-notify] [--sampler <sampler_name>]",
	  create_context, 1 },

	{ "load_context", "load",
	  "<context_id> <event_set_id> <program_id|cpu_id>",
	  load_context, 3 },

	{ "unload_context", "unload",
	  "<context_id>",
	  unload_context, 1 },

	{ "close_context", "close",
	  "<context_id>",
	  close_context, 1 },

	{ "write_pmc", "wpmc",
	  "<context_id> <event_set_id> <<pmc_id> <pmc_value>>+",
	  write_pmc, 4 },

	{ "write_pmd", "wpmd",
	  "<context_id> <event_set_id> <<pmd_id> <pmd_value>>+",
	  write_pmd, 4 },

	{ "read_pmd", "rpmd",
	  "<context_id> <event_set_id> <pmd_id>+",
	  read_pmd, 3 },

	{ "start_counting", "start",
	  "<context_id> <event_set_id>",
	  start_counting, 2 },

	{ "stop_counting", "stop",
	  "<context_id>",
	  stop_counting, 1 },

	{ "restart_counting", "restart",
	  "<context_id>",
	  restart_counting, 1 },

	{ "create_eventset", "ce",
	  "<context_id> <event_set_id> [--next-set <next_event_set_id>] "
	    "[--timeout <nanoseconds>] [--switch-on-overflow] [--exclude-idle]",
	  create_eventset, 2 },


	{ "delete_eventset", "de",
	  "<context_id> <event_set_id>",
	  delete_eventset, 2 },

	{ "getinfo_eventset", "ge",
	  "<context_id> <event_set_id>",
	  getinfo_eventset, 2 },

	{ "run_program", "run",
	  "<program_id> <program command line and arguments>",
	  run_program, 2 },

	{ "resume_program", "resume",
	  "<program_id>",
	  resume_program, 1 },

	{ "wait_on_program", "wait",
	  "<program_id>",
	  wait_on_program, 1 },

	{ "sleep", "sleep",
	  "<time in seconds>",
	  _sleep, 1 },

	{NULL, NULL, NULL, NULL, 0},
};

/**
 * find_command
 *
 * Search for the specified command in the _commands array. The command
 * can be specified using the full name or the short name.
 **/
static struct command *find_command(const char *command)
{
	int i;

	for (i = 0; _commands[i].full_name; i++) {
		if (!strcasecmp(command, _commands[i].full_name) ||
		    !strcasecmp(command, _commands[i].short_name)) {
			return _commands + i;
		}
	}

	return NULL;
}

static void print_help(const char *prog_name)
{
	int i;

	LOG_INFO("USAGE: %s <command_file>", prog_name);
	LOG_INFO("");
	LOG_INFO("Available commands and arguments for command-file:");

	for (i = 0; _commands[i].full_name; i++) {
		LOG_INFO("\t%s (%s)", _commands[i].full_name,
			 _commands[i].short_name);
		LOG_INFO("\t\t%s", _commands[i].help);
	}
}

/**
 * free_lines
 *
 * Free all the strings that were read from the command file.
 **/
void free_lines(char **lines)
{
	int i;
	if (lines) {
		for (i = 0; lines[i]; i++) {
			free(lines[i]);
		}
		free(lines);
	}
}

/**
 * read_file
 *
 * Read in the command-file. Create an array of strings, with one string
 * for each line in the file. The last entry in the array will be NULL
 * to indicate the end of the file.
 **/
static int read_file(FILE *fp, char ***lines)
{
	char one_line[1024], *str, **strings = NULL;
	int num_lines = 1;
	int i = 0;

	while (1) {
		str = fgets(one_line, 1024, fp);
		if (!str) {
			break;
		}

		if (i == num_lines || !strings) {
			num_lines *= 2;
			strings = realloc(strings, num_lines * sizeof(*strings));
			if (!strings) {
				return ENOMEM;
			}
		}

		strings[i] = strdup(one_line);
		if (!strings[i]) {
			free_lines(strings);
			return ENOMEM;
		}

		i++;
		strings[i] = NULL;
	}

	*lines = strings;
	return 0;
}

/**
 * tokenize
 *
 * Break up the specified line into whitespace-seperated tokens. Fill in
 * the 'tokens' array with pointers to each token.
 **/
static void tokenize(char *line, int *num_tokens, char **tokens)
{
	char *saved_line, *token;

	*num_tokens = 0;

	while (1) {
		token = strtok_r(line, WHITESPACE, &saved_line);
		if (!token) {
			break;
		}

		tokens[*num_tokens] = token;
		(*num_tokens)++;
		if (*num_tokens >= MAX_TOKENS) {
			break;
		}
		line = NULL;
	}

	tokens[*num_tokens] = NULL;
}

int main(int argc, char **argv)
{
	FILE *fp;
	struct command *cmd;
	char *filename;
	char **lines;
	char *tokens[MAX_TOKENS + 1] = {NULL};
	int num_tokens;
	int rc, i;

	if (argc < 2 ||
	    !strcmp(argv[1], "-?") ||
	    !strcasecmp(argv[1], "-h") ||
	    !strcasecmp(argv[1], "--help")) {
		print_help(argv[0]);
		return EINVAL;
	}
	filename = argv[1];

	/* Open the command file and read the entire
	 * contents into the 'lines' array.
	 */
	fp = fopen(filename, "r");
	if (!fp) {
		rc = errno;
		LOG_ERROR("Can't open file %s.\n", filename);
		return rc;
	}

	rc = read_file(fp, &lines);
	if (rc) {
		LOG_ERROR("Can't read file %s.\n", filename);
		return rc;
	}

	if (!lines) {
		LOG_ERROR("File %s is empty.\n", filename);
		rc = EINVAL;
		return rc;
	}

	/* Process each line from the command file. */
	for (i = 0; lines[i]; i++) {
		tokenize(lines[i], &num_tokens, tokens);
		if (!num_tokens) {
			/* Skip empty lines. */
			continue;
		}

		if (tokens[0][0] == '#') {
			/* Skip lines that start with '#'. */
			continue;
		}

		/* The first token specifies the command to run. Find this
		 * command in the array, check that we have enough arguments,
		 * and then run the command. If anything goes wrong with a
		 * command, we skip all remaining commands.
		 */
		cmd = find_command(tokens[0]);
		if (!cmd) {
			LOG_ERROR("Invalid command '%s' (line %d).\n",
				  tokens[0], i+1);
			rc = EINVAL;
			break;
		}

		if (num_tokens - 1 < cmd->min_args) {
			LOG_ERROR("Incorrect number of arguments for command "
				  "\'%s\' (line %d)", tokens[0], i+1);
			USAGE("%s %s", cmd->full_name, cmd->help);
			rc = EINVAL;
			break;
		}

		rc = cmd->fn(num_tokens, tokens);
		if (rc) {
			LOG_ERROR("command '%s' (line %d) returned an error: "
				  "%d.", tokens[0], i+1, rc);
			break;
		}
	}

	free_lines(lines);
	return rc;
}
