/*
 * freesmpl.c - example of how to get continuous sampling without notification
 *
 * Copyright (C) 2002 Hewlett-Packard Co
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux/ia64.
 */
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <stdarg.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/time.h>


#include <perfmon/pfmlib.h>

#define SAMPLING_INTERVAL	10000UL /* in microseconds */
#define SMPL_PERIOD		100000UL
#define SMPL_BUF_ENTRIES	8192UL


#define NUM_PMCS PMU_MAX_PMCS
#define NUM_PMDS PMU_MAX_PMDS

static jmp_buf jbuf;	/* setjmp buffer */
static pid_t child_pid;	/* process id of signaling child */

static void *smpl_vaddr;

static char *event_list[]={
	"cpu_cycles",
	"IA64_INST_RETIRED",
	NULL
};


static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
fatal_error(char *fmt, ...) 
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}

static unsigned long ret_max;
static unsigned long ret_min = ~0UL;
static unsigned long ret_sum;
static unsigned long sample_count;
static int time_to_quit;

static void
alarm_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	perfmon_smpl_hdr_t *hdr = (perfmon_smpl_hdr_t *)smpl_vaddr;
	perfmon_smpl_entry_t *ent;
	unsigned long *reg, off, base;

	if (time_to_quit) return;

	if (hdr->hdr_count < 2) return;
	
	base = (unsigned long)(hdr+1);
	off  = rand() % hdr->hdr_count;
	ent = (perfmon_smpl_entry_t *)(base + off*hdr->hdr_entry_size);
	reg = (unsigned long *)(ent+1);

	if (*reg > ret_max) ret_max = *reg;
	if (*reg < ret_min) ret_min = *reg;

	ret_sum += *reg;
	sample_count++;

	//printf("IPC=%4.2f\r", ((double)*reg)/SMPL_PERIOD);
	//fflush(stdout);
}


static void
child_handler(int n, struct siginfo *info, struct sigcontext *sc)
{
	time_to_quit = 1;
	/*
	 * we need to record the child pid here because we need to avoid
	 * a race condition with the parent returning from fork().
	 * In some cases, the pid=fork() instruction is not completed before
	 * we come to the SIGCHILD handler. the pid variable still has its
	 * default (zero) value. That's because the signal was received on
	 * return from fork() by the parent.
	 * So here we keept track of who just died and use a global variable
	 * to pass it back to the parent.
	 */
	child_pid = info->si_pid;

	/*
	 * That's not very pretty but that's one way of avoiding a race
	 * condition with the pause() system call. You may deadlock if the 
	 * signal is delivered before the parent reaches the pause() call.
	 * Using a variable and test reduces the window but it still exists
	 * (see pause() below). The longjmp/setjmp mechanism avoids it 
	 * completely.
	 */
	longjmp(jbuf,1);
}

int
child(char **arg, pfmlib_param_t *evt)
{
	int i;
	pid_t pid = getpid();
	pfarg_reg_t pd[NUM_PMDS];

	/* 
	 * Must be done before any PMD/PMD calls (unfreeze PMU). Initialize
	 * PMC/PMD to safe values. psr.up is cleared.
	 */
	if (perfmonctl(pid, PFM_ENABLE, NULL, 0) == -1) {
		fatal_error("child: perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	/*
	 * extract the PMD registers from their PMC counterpart.
	 * we just have to fill in the register numbers from the pc[] array.
	 */
	for (i=0; i < evt->pfp_event_count; i++) {
		pd[i].reg_num = evt->pfp_pc[i].reg_num;
	}
	/*
	 * Now prepare the argument to initialize the PMD and the sampling period
	 */
	pd[0].reg_value       = (~0UL) - SMPL_PERIOD +1;
	pd[0].reg_long_reset  = (~0UL) - SMPL_PERIOD +1;
	pd[0].reg_short_reset = (~0UL) - SMPL_PERIOD +1;

	/*
	 * Now program the registers
	 *
	 * We don't use the save variable to indicate the number of elements passed to
	 * the kernel because, as we said earlier, pc may contain more elements than
	 * the number of events we specified, i.e., contains more thann coutning monitors.
	 */
	if (perfmonctl(pid, PFM_WRITE_PMCS, evt->pfp_pc, evt->pfp_pc_count) == -1) {
		fatal_error("child: perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}

	/*
	 * initialize the PMDs
	 */
	if (perfmonctl(pid, PFM_WRITE_PMDS, pd, evt->pfp_event_count) == -1) {
		fatal_error("child: perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	/* 
	 * use the lightweight version. This must be done before we protect the context
	 *
	 * You can look at the code of pfmon (in pfmon_task.c) for an example of a way
	 * to hide the execution overhead until we execute the new code (from exec).
	 * The scheme is based on ptrace(2).
	 */
	pfm_start();

	/*
	 * This call is required to make sure that the monitored task cannot at random
	 * access its context and modify the session. After this call, only the creator
	 * of the context, i.e., our parent in this case, can access it. Similarly,
	 * if the monitored program toggles psr.up, this will have no effect on monitoring.
	 */
	if (perfmonctl(pid, PFM_PROTECT_CONTEXT, NULL, 0) == -1) {
		fatal_error("child: perfmonctl error PFM_PROTECT errno %d\n",errno);
	}
	execvp(arg[0], arg);

	//do_our_work();

	exit(1);
}

int 
monitor(char **arg)
{
	char **p;
	pfmlib_param_t evt;
	pfarg_context_t ctx[1];
	int i, status, ret;
	unsigned long smpl_regs;
	pid_t pid;

	memset(ctx, 0, sizeof(ctx));
	memset(&evt,0, sizeof(evt));

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&evt,0, sizeof(evt));

	/*
	 * be nice to user!
	 */
	//p = argc > 1 ? argv+1 : event_list;
	p = event_list;

	for (i=0; *p; i++, p++) {
		if (pfm_find_event(*p, &evt.pfp_events[i].event) != PFMLIB_SUCCESS) {
			fatal_error("Cannot find %s event\n", *p);
		}
	}

	/*
	 * set the privilege mode:
	 * 	PFM_PLM3 : user level
	 */
	evt.pfp_dfl_plm   = PFM_PLM3 |PFM_PLM0; 
	/*
	 * how many counters we use
	 */
	evt.pfp_event_count = i;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&evt)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}

	/*
	 * figure out which PMD to record in  every sample.
	 * we skip the first event as it is used for the sampling period
	 */
	smpl_regs = 0;
	for(i=1; i < evt.pfp_event_count; i++) {
		smpl_regs |= 1UL << evt.pfp_pc[i].reg_num;
	}
	evt.pfp_pc[0].reg_reset_pmds[0] = smpl_regs;

	/*
	 * setup perfmon context:
	 * 	- inherit to our child
	 * 	- no notification
	 * 	- size of sampling buffer
	 * 	- which register to record in every sample
	 */
	ctx[0].ctx_flags        = PFM_FL_INHERIT_ONCE;
	ctx[0].ctx_notify_pid   = 0; /* no notification: buffer will wrap around */
	ctx[0].ctx_smpl_entries = SMPL_BUF_ENTRIES;
	ctx[0].ctx_smpl_regs[0] = smpl_regs;

	/*
	 * now create a context in our task. We don't use it in this task but we use
	 * it for the child task. It will be inherited during the clone2() call (used by fork()). 
	 *
	 * The use of a context just for inheritance may seem clumsy and unecessary in the
	 * case where we are just counting but it is required when doing sampling to access
	 * the sampling buffer.
	 */
	if (perfmonctl(getpid(), PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	printf("Sampling buffer mapped at %p\n", ctx[0].ctx_smpl_vaddr);

	smpl_vaddr = ctx[0].ctx_smpl_vaddr;


	/*
	 * when we get called from the SIGCHILD handler we go straight to the result
	 * extraction code
	 */
	if (setjmp(jbuf) == 1) goto extract_results;
	
	/*
	 * Create the child task
	 */
	if ((pid=fork()) == -1) fatal_error("Cannot fork worker process: %s\n", strerror(errno));

	/*
	 * and launch the child code
	 */
	if (pid == 0) child(arg, &evt);

	/*
	 * while the parent is waiting for any signal.
	 * The SIGCHILD handler will take us out with the longjmp().
	 *
	 * It is not possible to solve the race condition by just having the SIGCHILD
	 * handler set a flag that would be check in the following loop. Signal are
	 * delivered on the kernel exit path, so this happens when (a) returning from
	 * a system call, or (b) interruption. The use of a flag would cover (a) because
	 * of pause() but not (b). The worst case is if we have interrupted after the test
	 * in the for loop but before the pause(): for(;flag == 0; ) pause().
	 */
	for(;;) pause();

extract_results:
	/* 
	 * Now that we have etxracted the results from the zombie child task,
	 * we can clean it up child. We can't do this call before we read the PMDS
	 */
	waitpid(child_pid, &status, 0);

	/* 
	 * We don't need our context anymore. Note that because we are not using the
	 * sampling buffer for this example, we could have destroyed our context much 
	 * earlier, e.g., just after the fork().
	 */
	if (perfmonctl(getpid(), PFM_DESTROY_CONTEXT, NULL, 0) == -1) {
		fatal_error("perfmonctl error PFM_DESTROY errno %d\n",errno);
	}

	if (sample_count < 2) fatal_error("Did not capture any sample\n");
	printf("sampling period %lu cycles\n", SMPL_PERIOD);
	printf("%lu samples captured\n", sample_count);
	printf("minIPC=%4.2f avgIPC=%4.2f maxIPC=%4.2f\n", 
			(double)ret_min/SMPL_PERIOD, 
			(double)ret_sum/(sample_count*SMPL_PERIOD), 
			(double)ret_max/SMPL_PERIOD);
	return 0;
}

int 
main(int argc, char **argv)
{
	pfmlib_options_t pfmlib_options;
	struct sigaction act;
	struct itimerval itimer;
	int r;

	if (argc < 2) {
		fatal_error("You must specify a command to execute\n");
	}
	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		printf("Can't initialize library\n");
		exit(1);
	}

	srand(time(NULL));
	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */
	pfm_set_options(&pfmlib_options);

	/* 
	 * install SIGCHLD handler 
	 */
	memset(&act,0,sizeof(act));

	act.sa_handler = (sig_t)child_handler;
	sigaction (SIGCHLD, &act, 0);

	act.sa_handler = (sig_t)alarm_handler;
	sigaction (SIGALRM, &act, 0);

	itimer.it_interval.tv_sec  = 0;
	itimer.it_interval.tv_usec = SAMPLING_INTERVAL;
	itimer.it_value.tv_sec  = 1;
	itimer.it_value.tv_usec = 0;

	r = setitimer(ITIMER_REAL, &itimer, NULL);
	if (r == -1) fatal_error("itimer: %s\n", strerror(errno));

	return monitor(argv+1);
}
