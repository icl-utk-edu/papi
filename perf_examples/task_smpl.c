/*
 * task_smpl.c - example of a task sampling another one using a randomized sampling period
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on:
 * Copyright (c) 2003-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
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
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <signal.h>
#include <getopt.h>
#include <setjmp.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/poll.h>
#include <sys/mman.h>
#include <err.h>

#include "perf_util.h"

#define SMPL_PERIOD	240000000ULL

typedef struct {
	int opt_no_show;
	int opt_inherit;
	int opt_freq;
	int mmap_pages;
	char *events;
	uint64_t period;
} options_t;

static jmp_buf jbuf;
static uint64_t collected_samples, lost_samples;
static perf_event_desc_t *fds;
static int num_events;
static options_t options;

static struct option the_options[]={
	{ "help", 0, 0,  1},
	{ "no-show", 0, &options.opt_no_show, 1},
	{ 0, 0, 0, 0}
};

static char *gen_events = "PERF_COUNT_HW_CPU_CYCLES,PERF_COUNT_HW_INSTRUCTIONS";

static void
cld_handler(int n)
{
	longjmp(jbuf, 1);
}

int
child(char **arg)
{
	/*
	 * force the task to stop before executing the first
	 * user level instruction
	 */
	ptrace(PTRACE_TRACEME, 0, NULL, NULL);

	execvp(arg[0], arg);
	/* not reached */
	return -1;
}

struct timeval last_read, this_read;

/*
 * sz = sample payload size
 */
static void
display_sample(perf_event_desc_t *hw, size_t sz)
{
	struct { uint32_t pid, tid; } pid;
	struct { uint64_t value, id; } grp;
	uint64_t time_enabled, time_running;
	uint64_t type;
	uint64_t val64;
	char *str;
	int ret, e;

	type = hw->hw.sample_type;

	collected_samples++;
	printf("%4"PRIu64" ", collected_samples);
	/*
	 * information laid out by increasing index in the
	 * perf_event.record_format enum therefore we MUST
	 * check in the same order.
	 */
	if (type & PERF_SAMPLE_IP) {
		ret = perf_read_buffer_64(hw->buf, hw->pgmsk, &val64);
		if (ret)
			errx(1, "cannot read IP");

		printf("IIP:0x%016"PRIx64" ", val64);
		sz -= sizeof(val64);
	}

	if (type & PERF_SAMPLE_TID) {
		ret = perf_read_buffer(hw->buf, hw->pgmsk, &pid, sizeof(pid));
		if (ret)
			errx(1, "cannot read PID");

		printf("PID:%d TID:%d ", pid.pid, pid.tid);
		sz -= sizeof(pid);
	}

	if (type & PERF_SAMPLE_TIME) {
		ret = perf_read_buffer_64(hw->buf, hw->pgmsk, &val64);
		if (ret)
			errx(1, "cannot read time");

		printf("TIME:%"PRIu64" ", val64);
		sz -= sizeof(val64);
	}

	/*
	 *      { u64           nr;
	 *        { u64         time_enabled; } && PERF_FORMAT_ENABLED
	 *        { u64         time_running; } && PERF_FORMAT_RUNNING
	 *        { u64         value;
	 *          { u64       id;           } && PERF_FORMAT_ID
	 *        }             cntr[nr];
	 */ 
	if (type & PERF_SAMPLE_READ) {
		uint64_t nr;

		ret = perf_read_buffer_64(hw->buf, hw->pgmsk, &nr);
		if (ret)
			errx(1, "cannot read nr");

		sz -= sizeof(nr);

		time_enabled = time_running = 1;

		ret = perf_read_buffer_64(hw->buf, hw->pgmsk, &time_enabled);
		if (ret)
			errx(1, "cannot read timing info");

		sz -= sizeof(time_enabled);

		ret = perf_read_buffer_64(hw->buf, hw->pgmsk, &time_running);
		if (ret)
			errx(1, "cannot read timing info");

		sz -= sizeof(time_running);

		printf("ENA=%"PRIu64" RUN=%"PRIu64" NR=%"PRIu64"\n", time_enabled, time_running, nr);

		while(nr--) {
			ret = perf_read_buffer(hw->buf, hw->pgmsk, &grp, sizeof(grp));
			if (ret)
				errx(1, "cannot read grp");

			sz -= sizeof(grp);

			e = perf_id2event(fds, num_events, grp.id);
			if (e == -1)
				str = "unknown sample event";
			else
				str = fds[e].name;

			if (time_running)
				grp.value = grp.value * time_enabled / time_running;

			printf("\t%"PRIu64" %s (%"PRIu64"%s)\n",
				grp.value, str,
				grp.id,
				time_running != time_enabled ? ", scaled":"");

		}
	}

	/*
	 * if we have some data left, it is because there is more
	 * than what we know about. In fact, it is more complicated
	 * because we may have the right size but wrong layout. But
	 * that's the best we can do.
	 */
	if (sz)
		err(1, "did not correctly parse sample");

	putchar('\n');
}

static void
display_lost(perf_event_desc_t *hw)
{
	struct { uint64_t id, lost; } lost;
	char *str;
	int e, ret;

	ret = perf_read_buffer(hw->buf, hw->pgmsk, &lost, sizeof(lost));
	if (ret)
		errx(1, "cannot read lost info");

	e = perf_id2event(fds, num_events, lost.id);
	if (e == -1)
		str = "unknown lost event";
	else
		str = fds[e].name;

	printf("<<<LOST %"PRIu64" SAMPLES FOR EVENT %s>>>\n", lost.lost, str);
	lost_samples += lost.lost;
}

static void
display_exit(perf_event_desc_t *hw)
{
	struct { pid_t pid, ppid, tid, ptid; } grp;
	int ret;

	ret = perf_read_buffer(hw->buf, hw->pgmsk, &grp, sizeof(grp));
	if (ret)
		errx(1, "cannot read exit info");

	printf("[%d] exited\n", grp.pid);
}

static void
display_freq(int mode, perf_event_desc_t *hw)
{
	struct { uint64_t time, id, stream_id; } thr;
	int ret;

	ret = perf_read_buffer(hw->buf, hw->pgmsk, &thr, sizeof(thr));
	if (ret)
		errx(1, "cannot read throttling info");

	printf("%s value=%"PRIu64" event ID=%"PRIu64"\n", mode ? "Throttled" : "Unthrottled", thr.id, thr.stream_id);
}

static void
process_smpl_buf(perf_event_desc_t *hw)
{
	struct perf_event_header ehdr;
	int ret;

	for(;;) {
		ret = perf_read_buffer(hw->buf, hw->pgmsk, &ehdr, sizeof(ehdr));
		if (ret)
			return; /* nothing to read */

		switch(ehdr.type) {
			case PERF_RECORD_SAMPLE:
				display_sample(hw, ehdr.size - sizeof(ehdr));
				break;
			case PERF_RECORD_EXIT:
				display_exit(hw);
				break;
			case PERF_RECORD_LOST:
				display_lost(hw);
				break;
			case PERF_RECORD_THROTTLE:
				display_freq(1, hw);
				break;
			case PERF_RECORD_UNTHROTTLE:
				display_freq(0, hw);
				break;
			default:
				printf("unknown sample type %d\n", ehdr.type);
				perf_skip_buffer(hw->buf, ehdr.size);
		}
	}
}

int
mainloop(char **arg)
{
	static uint64_t ovfl_count; /* static to avoid setjmp issue */
	struct pollfd pollfds[1];
	uint64_t *val;
	size_t sz, pgsz;
	size_t map_size = 0;
	pid_t pid;
	int status, ret;
	int i;

	if (pfm_initialize() != PFM_SUCCESS)
		errx(1, "libpfm initialization failed\n");

	pgsz = getpagesize();
	map_size = (options.mmap_pages+1)*pgsz;

	/*
	 * does allocate fds
	 */
	num_events = perf_setup_list_events(options.events, &fds);
	if (num_events == -1)
		errx(1, "cannot setup event list");

	memset(pollfds, 0, sizeof(pollfds));

	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1)
		err(1, "cannot fork process\n");

	if (pid == 0)
		exit(child(arg));

	/*
	 * wait for the child to exec
	 */
	ret = waitpid(pid, &status, WUNTRACED);
	if (ret == -1)
		err(1, "waitpid failed");

	if (WIFEXITED(status))
		errx(1, "task %s [%d] exited already status %d\n", arg[0], pid, WEXITSTATUS(status));

	fds[0].fd = -1;
	for(i=0; i < num_events; i++) {

		fds[i].hw.disabled = 0; /* start immediately */

		/*
		 * set notification threshold to be halfway through the buffer
		 * (header page removed)
		 */
		fds[i].hw.wakeup_watermark = (options.mmap_pages*pgsz) / 2; 
		fds[i].hw.watermark = 1;

		if (options.opt_inherit)
			fds[i].hw.inherit = 1;

		if (!i) {
			fds[i].hw.sample_type = PERF_SAMPLE_IP|PERF_SAMPLE_TID|PERF_SAMPLE_READ|PERF_SAMPLE_TIME;

			if (options.opt_freq)
				fds[i].hw.freq = 1;

			fds[i].hw.sample_period = options.period;
			printf("period=%"PRIu64" freq=%d\n", options.period, options.opt_freq);

			/* must get event id for SAMPLE_GROUP */
			fds[i].hw.read_format = PERF_FORMAT_GROUP|PERF_FORMAT_ID|PERF_FORMAT_SCALE;
		}

		fds[i].fd = perf_event_open(&fds[i].hw, pid, -1, fds[0].fd, 0);
		if (fds[i].fd == -1)
			err(1, "cannot attach event %s", fds[i].name);
	}


	fds[0].buf = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fds[0].fd, 0);
	if (fds[0].buf == MAP_FAILED)
		err(1, "cannot mmap buffer");

	/* does not include header page */
	fds[0].pgmsk = (options.mmap_pages*getpagesize())-1;

	/*
	 * we are using PERF_FORMAT_GROUP, therefore the structure
	 * of val is as follows:
	 *
	 *      { u64           nr;
	 *        { u64         time_enabled; } && PERF_FORMAT_ENABLED
	 *        { u64         time_running; } && PERF_FORMAT_RUNNING
	 *        { u64         value;
	 *          { u64       id;           } && PERF_FORMAT_ID
	 *        }             cntr[nr];
	 * We are skipping the first 3 values (nr, time_enabled, time_running)
	 * and then for each event we get a pair of values.
	 */
	sz = (3+2*num_events)*sizeof(uint64_t);
	val = malloc(sz);
	if (!val)
		err(1, "cannot allocated memory");

	ret = read(fds[0].fd, val, sz);
	if (ret == -1)
		err(1, "cannot read id %zu", sizeof(val));


	for(i=0; i < num_events; i++) {
		fds[i].id = val[2*i+1+3];
		printf("%"PRIu64"  %s\n", fds[i].id, fds[i].name);
	}

	pollfds[0].fd = fds[0].fd;
	pollfds[0].events = POLLIN;
	
	/*
	 * effectively activate monitoring
	 */
	ptrace(PTRACE_DETACH, pid, NULL, 0);

	signal(SIGCHLD, cld_handler);

	if (setjmp(jbuf) == 1)
		goto terminate_session;

	/*
	 * core loop
	 */
	for(;;) {
		ret = poll(pollfds, 1, -1);
		if (ret < 0 && errno == EINTR)
			break;
		ovfl_count++;
		process_smpl_buf(&fds[0]);
	}
terminate_session:
	/*
	 * cleanup child
	 */
	wait4(pid, &status, 0, NULL);

	for(i=0; i < num_events; i++)
		close(fds[i].fd);

	/* check for partial event buffer */
	process_smpl_buf(&fds[0]);
	munmap(fds[0].buf, map_size);

	free(fds);
	free(val);

	printf("%"PRIu64" samples collected in %"PRIu64" poll events, %"PRIu64" lost samples\n",
		collected_samples,
		ovfl_count, lost_samples);
	return 0;
}

static void
usage(void)
{
	printf("usage: task_smpl [-h] [--help] [-i] [-m mmap_pages] [-f] [-e event1,...,eventn] [-p period] cmd\n");
}

int
main(int argc, char **argv)
{
	int c;

	while ((c=getopt_long(argc, argv,"he:m:p:if", the_options, 0)) != -1) {
		switch(c) {
			case 0: continue;
			case 'e':
				if (options.events)
					errx(1, "events specified twice\n");
				options.events = optarg;
				break;
			case 'i':
				options.opt_inherit = 1;
				break;
			case 'f':
				options.opt_freq = 1;
				break;
			case 'm':
				if (options.mmap_pages)
					errx(1, "mmap pages already set\n");
				options.mmap_pages = atoi(optarg);
				break;
			case 'p':
				options.period = strtoull(optarg, NULL, 0);
				break;
			case 'h':
				usage();
				exit(0);
			default:
				errx(1, "unknown option");
		}
	}

	if (argv[optind] == NULL)
		errx(1, "you must specify a command to execute\n");
	if (!options.events)
		options.events = strdup(gen_events);

	if (!options.period)
		options.period = options.opt_freq ? 1 : SMPL_PERIOD;

	if (!options.mmap_pages)
		options.mmap_pages = 1;
	
	return mainloop(argv+optind);
}
