/*
 * task_inherit.c - example of a task counting event in a tree of child processes
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <signal.h>
#include <sys/wait.h>
#include <err.h>

#include "perf_util.h"

typedef struct {
	const char *events;
	int inherit;
	int group;
	int print;
	int pin;
	pid_t pid;
} options_t;

static options_t options;
static volatile int quit;

int
child(char **arg)
{
	/*
	 * execute the requested command
	 */
	execvp(arg[0], arg);
	errx(1, "cannot exec: %s\n", arg[0]);
	/* not reached */
}

static void
print_counts(perf_event_desc_t *fds, int num)
{
	uint64_t values[3];
	int i, ret;
	/*
	 * now simply read the results.
	 */
	for(i=0; i < num; i++) {
		uint64_t val;
		double ratio;

		ret = read(fds[i].fd, values, sizeof(values));
		if (ret != sizeof(values)) { /* unsigned */
			if (ret == -1)
				err(1, "cannot read values event %s", fds[i].name);
			else	/* likely pinned and could not be loaded */
				warnx("could not read event%d", i);
		}

		/*
		 * scaling because we may be sharing the PMU and
		 * thus may be multiplexed
		 */
		fds[i].prev_value = fds[i].value;
		fds[i].value = val = perf_scale(values);
		ratio = perf_scale_ratio(values);

		val = val - fds[i].prev_value;
		if (ratio == 1.0)
			printf("%20"PRIu64" %s\n", val, fds[i].name);
		else
			if (ratio == 0.0)
				printf("%20"PRIu64" %s (did not run: incompatible events, too many events in a group, competing session)\n", val, fds[i].name);
			else
				printf("%20"PRIu64" %s (scaled from %.2f%% of time)\n", val, fds[i].name, ratio*100.0);

	}
}

static void sig_handler(int n)
{
	quit = 1;
}

int
parent(char **arg)
{
	perf_event_desc_t *fds;
	int status, ret, i, num;
	int ready[2], go[2];
	char buf;
	pid_t pid;

	if (pfm_initialize() != PFM_SUCCESS)
		errx(1, "libpfm initialization failed");

	num = perf_setup_list_events(options.events, &fds);
	if (num < 1)
		exit(1);

	pid = options.pid;
	if (!pid) {
		ret = pipe(ready);
		if (ret)
			err(1, "cannot create pipe ready");

		ret = pipe(go);
		if (ret)
			err(1, "cannot create pipe go");


		/*
		 * Create the child task
		 */
		if ((pid=fork()) == -1)
			err(1, "Cannot fork process");

		/*
		 * and launch the child code
		 */
		if (pid == 0) {
			close(ready[0]);
			close(go[1]);

			/*
			 * let the parent know we exist
			 */
			close(ready[1]);
			if (read(go[0], &buf, 1) == -1)
				err(1, "unable to read go_pipe");


			exit(child(arg));
		}

		close(ready[1]);
		close(go[0]);

		if (read(ready[0], &buf, 1) == -1)
			err(1, "unable to read child_ready_pipe");

		close(ready[0]);
	}

	fds[0].fd = -1;
	for(i=0; i < num; i++) {
		/*
		 * create leader disabled with enable_on-exec
		 */
		if (!options.pid) {
			if (options.group) {
				fds[i].hw.disabled = !i;
				fds[i].hw.enable_on_exec = !i;
			} else {
				fds[i].hw.disabled = 1;
				fds[i].hw.enable_on_exec = 1;
			}
		}

		/* request timing information necessary for scaling counts */
		fds[i].hw.read_format = PERF_FORMAT_SCALE;

		if (options.inherit)
			fds[i].hw.inherit = 1;

		if (options.pin && ((options.group && i== 0) || (!options.group)))
			fds[i].hw.pinned = 1;

		fds[i].fd = perf_event_open(&fds[i].hw, pid, -1, options.group ? fds[0].fd : -1, 0);
		if (fds[i].fd == -1) {
			warn("cannot attach event%d %s", i, fds[i].name);
			goto error;
		}
	}	

	if (!options.pid)
		close(go[1]);

	if (options.print) {
		if (!options.pid) {
			while(waitpid(pid, &status, WNOHANG) == 0) {
				sleep(1);
				print_counts(fds, num);
			}
		} else {
			while(quit == 0) {
				sleep(1);
				print_counts(fds, num);
			}
		}
	} else {
		if (!options.pid)
			waitpid(pid, &status, 0);
		else
			pause();
		print_counts(fds, num);
	}

	for(i=0; i < num; i++)
		close(fds[i].fd);

	free(fds);
	return 0;
error:
	free(fds);
	if (!options.pid)
		kill(SIGKILL, pid);
	return -1;
}

static void
usage(void)
{
	printf("usage: task [-h] [-i] [-g] [-p] [-P] [-t pid] [-e event1,event2,...] cmd\n"
		"-h\t\tget help\n"
		"-i\t\tinherit across fork\n"
		"-g\t\tgroup events\n"
		"-p\t\tprint counts every second\n"
		"-P\t\tpin events\n"
		"-t pid\tmeasure existing pid\n"
		"-e ev,ev\tlist of events to measure\n"
		);
}

int
main(int argc, char **argv)
{
	int c;

	while ((c=getopt(argc, argv,"he:igpPt:")) != -1) {
		switch(c) {
			case 'e':
				options.events = optarg;
				break;
			case 'g':
				options.group = 1;
				break;
			case 'p':
				options.print = 1;
				break;
			case 'P':
				options.pin = 1;
				break;
			case 'i':
				options.inherit = 1;
				break;
			case 't':
				options.pid = atoi(optarg);
				break;
			case 'h':
				usage();
				exit(0);
			default:
				errx(1, "unknown error");
		}
	}
	if (!options.events)
		options.events = "PERF_COUNT_HW_CPU_CYCLES,PERF_COUNT_HW_INSTRUCTIONS";

	if (!argv[optind] && !options.pid)
		errx(1, "you must specify a command to execute or a thread to attach to\n");
	
	signal(SIGINT, sig_handler);

	return parent(argv+optind);
}
