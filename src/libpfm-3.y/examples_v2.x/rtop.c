/* rtop.c - a simple PMU-based CPU utilization tool
 *
 * Copyright (c) 2004-2006 Hewlett-Packard Development Company, L.P.
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
#ifndef _GNU_SOURCE
  #define _GNU_SOURCE /* for getline */
#endif
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <syscall.h>
#include <getopt.h>
#include <curses.h>
#include <termios.h>
#include <signal.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/poll.h>

#include <perfmon/pfmlib.h>
#include <perfmon/perfmon.h>

#include "detect_pmcs.h"

#define SWITCH_TIMEOUT	1000000000 /* in nanoseconds */

#define RTOP_VERSION "0.1"

#define RTOP_MAX_CPUS		1024	/* maximum number of CPU supported */

#define MAX_EVT_NAME_LEN	128
#define RTOP_NUM_PMCS 		4
#define RTOP_NUM_PMDS 		4
/* 
 * max number of cpus (threads) supported
 */
#define RTOP_MAX_CPUS		1024 /* MUST BE power of 2 */
#define RTOP_CPUMASK_BITS	(sizeof(unsigned long)<<3)
#define RTOP_CPUMASK_COUNT	(RTOP_MAX_CPUS/RTOP_CPUMASK_BITS)

#define RTOP_CPUMASK_SET(m, g)		((m)[(g)/RTOP_CPUMASK_BITS] |=  (1UL << ((g) % RTOP_CPUMASK_BITS)))
#define RTOP_CPUMASK_CLEAR(m, g)	((m)[(g)/RTOP_CPUMASK_BITS] &= ~(1UL << ((g) % RTOP_CPUMASK_BITS)))
#define RTOP_CPUMASK_ISSET(m, g)	((m)[(g)/RTOP_CPUMASK_BITS] &   (1UL << ((g) % RTOP_CPUMASK_BITS)))

typedef unsigned long rtop_cpumask_t[RTOP_CPUMASK_COUNT];

typedef struct {
	struct {
		int	opt_verbose;
		int	opt_delay;	/* refresh delay in second */
		int	opt_delay_set;
	} program_opt_flags;
	rtop_cpumask_t	cpu_mask;	  /* which CPUs to use in system wide mode */
	long		online_cpus;
	long		selected_cpus;
	unsigned long	cpu_mhz;
	char		*outfile;
} program_options_t;

#define opt_verbose program_opt_flags.opt_verbose
#define opt_delay program_opt_flags.opt_delay
#define opt_delay_set program_opt_flags.opt_delay_set


typedef struct {
	char *name;
	unsigned int plm;
} eventdesc_t;

typedef enum {
	THREAD_STARTED,
	THREAD_RUN,
	THREAD_DONE,
	THREAD_ERROR
} thread_state_t;


typedef struct {
	pthread_t	tid;		/* logical thread identification */
	long		cpuid;
	unsigned int	id;
	thread_state_t	state;
	int		is_last;
	sem_t		his_sem;
	sem_t		my_sem;
	FILE		*fp;
	uint64_t	nsamples;
	int		has_msg;
} thread_desc_t;

typedef struct _setdesc_t {
	pfarg_pmc_t		pc[RTOP_NUM_PMDS];
	pfarg_pmd_t		pd[RTOP_NUM_PMCS];
	pfmlib_input_param_t	inp;
	pfmlib_output_param_t	outp;
	uint16_t		set_id;
	uint32_t		set_flags;
	uint32_t		set_timeout; /* actual timeout */
	int			(*handler)(int fd, FILE *fp, thread_desc_t *td, struct _setdesc_t *my_sdesc);
	void 			*data;
	eventdesc_t		*evt_desc;
} setdesc_t;

typedef struct _barrier {
	pthread_mutex_t mutex;
	pthread_cond_t	cond;
	unsigned long	counter;
	unsigned long	max;
	unsigned long   generation; /* avoid race condition on wake-up */
} barrier_t;

typedef enum {
	SESSION_INIT,
	SESSION_RUN,
	SESSION_STOP,
	SESSION_ABORTED
} session_state_t;

typedef struct {
	uint64_t prev_k_cycles;
	uint64_t prev_u_cycles;
} set0_data_t;

static barrier_t 		barrier;
static session_state_t		session_state;
static program_options_t 	options;
static pfarg_ctx_t 		master_ctx;
static thread_desc_t		*thread_info;
static struct termios		saved_tty;
static int			time_to_quit;
static int			term_rows, term_cols;

static eventdesc_t	set0_evt[]={
	{ .name  = "*", .plm = PFM_PLM0 },
	{ .name  = "*", .plm = PFM_PLM3 },
	{ .name  = NULL}
 };

static int handler_set0(int fd, FILE *fp, thread_desc_t *td, setdesc_t *my_sdesc);

static setdesc_t setdesc_tab[]={
	{ .set_id    = 0,
	  .evt_desc  = set0_evt,
	  .handler   = handler_set0
	}
};
#define RTOP_NUM_SDESC (sizeof(setdesc_tab)/sizeof(setdesc_t))

static int
barrier_init(barrier_t *b, unsigned long count)
{
	int r;

	r = pthread_mutex_init(&b->mutex, NULL);
	if (r == -1) return -1;
	r = pthread_cond_init(&b->cond, NULL);
	if (r == -1) return -1;
	b->max = b->counter = count;

	b->generation = 0;

	return 0;
}

static void
cleanup_barrier(void *arg)
{
	int r;
	barrier_t *b = (barrier_t *)arg;
	r = pthread_mutex_unlock(&b->mutex);
	(void)r;
}


static int
barrier_wait(barrier_t *b)
{
	unsigned long generation;
	int oldstate;

	pthread_cleanup_push(cleanup_barrier, b);

	pthread_mutex_lock(&b->mutex);

	pthread_testcancel();

	if (--b->counter == 0) {

		/* reset barrier */
		b->counter = b->max;
		/*
		 * bump generation number, this avoids thread getting stuck in the
		 * wake up loop below in case a thread just out of the barrier goes
		 * back in right away before all the thread from the previous "round"
		 * have "escaped".
		 */
		b->generation++;

		pthread_cond_broadcast(&b->cond);
	} else {

		generation = b->generation;

		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &oldstate);

		while (b->counter != b->max && generation == b->generation) {
			pthread_cond_wait(&b->cond, &b->mutex);
		}

		pthread_setcancelstate(oldstate, NULL);
	}
	pthread_mutex_unlock(&b->mutex);

	pthread_cleanup_pop(0);

	return 0;
}


static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
warning(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}


static void
fatal_error(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}

int
gettid(void)
{
	int tmp;
	tmp = syscall(__NR_gettid);
	return tmp;
}

#ifndef __NR_sched_setaffinity
#error "you need to define __NR_sched_setaffinity"
#endif

/*
 * Hack to get this to work without libc support
 */
int
pin_self_cpu(unsigned int cpu)
{
	unsigned long my_mask;

	my_mask = 1UL << cpu;

	return syscall(__NR_sched_setaffinity, gettid(), sizeof(my_mask), &my_mask);
}

static void
sigint_handler(int n)
{
	time_to_quit = 1;
}

static unsigned long
find_cpu_speed(void)
{
	FILE *fp1;	
	unsigned long f1 = 0, f2 = 0;
	char buffer[128], *p, *value;

	memset(buffer, 0, sizeof(buffer));

	fp1 = fopen("/proc/cpuinfo", "r");
	if (fp1 == NULL) return 0;

	for (;;) {
		buffer[0] = '\0';

		p  = fgets(buffer, 127, fp1);
		if (p == NULL)
			break;

		/* skip  blank lines */
		if (*p == '\n') continue;

		p = strchr(buffer, ':');
		if (p == NULL)
			break;

		/*
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0';
		value = p+2;

		value[strlen(value)-1] = '\0';

		if (!strncasecmp("cpu MHz", buffer, 7)) {
			float fl;
			sscanf(value, "%f", &fl);
			f1 = lroundf(fl);
			break;
		}
		if (!strncasecmp("BogoMIPS", buffer, 8)) {
			float fl;
			sscanf(value, "%f", &fl);
			f2 = lroundf(fl);
		}
	}
	fclose(fp1);
	return f1 == 0 ? f2 : f1;
}

static void
get_term_size(void)
{
	int ret;
        struct winsize ws;

	ret = ioctl(1, TIOCGWINSZ, &ws);
	if (ret == -1) 
		fatal_error("cannot determine screen size\n");

	if (ws.ws_row > 10) {
                term_cols = ws.ws_col;
                term_rows = ws.ws_row;
        } else {
                term_cols = 80;
                term_rows = 24;
        }

	if (term_rows < options.selected_cpus)
		fatal_error("you need at least %ld rows on your terminal to display all CPUs\n", options.selected_cpus);
}

static void
sigwinch_handler(int n)
{
	get_term_size();
}

static void
setup_screen(void)
{
	int ret;

	ret = tcgetattr(0, &saved_tty);
	if (ret == -1)
		fatal_error("cannot save tty settings\n");

	get_term_size();

	initscr();
	nocbreak();
	resizeterm(term_rows, term_cols);
}

static void
close_screen(void)
{
	int ret;

	endwin();

	ret = tcsetattr(0, TCSAFLUSH, &saved_tty);
	if (ret == -1)
		warning("cannot restore tty settings\n");
}


static void
setup_signals(void)
{
	struct sigaction act;
	sigset_t my_set;

	/*
	 * SIGINT is a asynchronous signal
	 * sent to the process (not a specific thread). POSIX states
	 * that one and only one thread will execute the handler. This
	 * could be any thread that does not have the signal blocked.
	 */

	/*
	 * install SIGINT handler
	 */
	memset(&act,0,sizeof(act));
	sigemptyset(&my_set);
	act.sa_handler = (__sighandler_t)sigint_handler;
	sigaction (SIGINT, &act, 0);

	/*
	 * install SIGWINCH handler
	 */
	memset(&act,0,sizeof(act));
	sigemptyset(&my_set);
	act.sa_handler = (__sighandler_t)sigwinch_handler;
	sigaction (SIGWINCH, &act, 0);
}

static void
setup_worker_signals(void)
{
	struct sigaction act;
	sigset_t my_set;

	/*
	 * SIGINT is a asynchronous signal
	 * sent to the process (not a specific thread). POSIX states
	 * that one and only one thread will execute the handler. This
	 * could be any thread that does not have the signal blocked.
	 */

	/*
	 * block SIGINT, forcing it to master thread only
	 */
	memset(&act,0,sizeof(act));

	sigemptyset(&my_set);
        sigaddset(&my_set, SIGINT);
        sigaddset(&my_set, SIGWINCH);

	pthread_sigmask(SIG_BLOCK, &my_set, NULL);
}


static struct option rtop_cmd_options[]={
	{ "help", 0, 0, 1 },
	{ "version", 0, 0, 2 },
	{ "delay", 0, 0, 3 },
	{ "cpu-list", 1, 0, 4 },
	{ "outfile", 1, 0, 5 },

	{ "verbose", 0, &options.opt_verbose, 1 },
	{ 0, 0, 0, 0}
};



int
handler_set0(int fd, FILE *fp, thread_desc_t *td, setdesc_t *my_sdesc)
{
	double k_cycles, u_cycles, i_cycles;
	set0_data_t *sd1;
	uint64_t itc_delta;
	long mycpu;

	mycpu = td->cpuid;

	if (my_sdesc->data == NULL) {
		my_sdesc->data = sd1 = calloc(1, sizeof(set0_data_t));
		if (sd1 == NULL)
			return -1;

	}
	sd1 = my_sdesc->data;

	/*
	 * now read the results
	 */
	if (pfm_read_pmds(fd, my_sdesc->pd, my_sdesc->inp.pfp_event_count) == -1) {
		warning( "CPU%ld pfm_read_pmds error errno %d\n", mycpu, errno);
		return -1;
	}

	/*
	 * expected maximum duration with monitoring active for this set
	 * set_timeout is in nanoseconds, we need to divide mhz by 1000
	 * to get cycles.
	 */
	itc_delta =  (my_sdesc->set_timeout*(uint64_t)options.cpu_mhz)/1000;

	k_cycles   = (double)(my_sdesc->pd[0].reg_value - sd1->prev_k_cycles)*100.0/ (double)itc_delta;
	u_cycles   = (double)(my_sdesc->pd[1].reg_value - sd1->prev_u_cycles)*100.0/ (double)itc_delta;
	i_cycles   = 100.0 - (k_cycles + u_cycles);

	/*
	 * adjust for rounding errors
	 */
	if (i_cycles < 0.0) i_cycles = 0.0;
	if (i_cycles > 100.0) i_cycles = 100.0;
	if (k_cycles > 100.0) k_cycles = 100.0;
	if (u_cycles > 100.0) u_cycles = 100.0;

	printw("CPU%-2ld %6.2f%% usr %6.2f%% sys %6.2f%% idle\n",
		mycpu,
		u_cycles,
		k_cycles,
		i_cycles);

	sd1->prev_k_cycles      = my_sdesc->pd[0].reg_value;
	sd1->prev_u_cycles      = my_sdesc->pd[1].reg_value;

	if (fp)
		fprintf(fp, "%"PRIu64" %6.2f %6.2f %6.2f\n",
			td->nsamples,
			u_cycles,
			k_cycles,
			i_cycles);
	td->nsamples++;
	return 0;
}

static void
do_measure_one_cpu(void *data)
{
	thread_desc_t	*arg= (thread_desc_t *)data;
	pfarg_ctx_t	ctx;
	pfarg_load_t	load_args;
	pfarg_setdesc_t	setd;
	setdesc_t 	*my_sdesc, *my_sdesc_tab = NULL;
	long		mycpu;
	sem_t		*his_sem;
	unsigned int	id;
	int		fd, ret, j;
	int		old_rows;
	char		cpu_str[16];
	char		*fn;
	FILE		*fp = NULL;

	setup_worker_signals();

	mycpu    = arg->cpuid;
	id       = arg->id;
	his_sem    = &arg->his_sem;
	old_rows = term_rows;


	if (options.outfile) {
		sprintf(cpu_str,".cpu%ld", mycpu);

		fn = malloc(strlen(options.outfile)+1+strlen(cpu_str));
		if (fn == NULL) goto error;

		strcpy(fn, options.outfile);
		strcat(fn, cpu_str);

		fp = fopen(fn, "w");
		if (fp == NULL) {
			warning("cannot open %s\n", fn);
			free(fn);
			goto error;
		}
		free(fn);
		fprintf(fp, "# Results for CPU%ld\n"
				"# sample delay %d seconds\n"
				"# Column1 : sample number\n"
				"# Column2 : %% user time\n"
				"# Column3 : %% system time\n"
				"# Column4 : %% idle\n"
				"# Column5 : kernel entry-exit\n",
				mycpu,
				options.opt_delay);

	}

	memset(&load_args, 0, sizeof(load_args));
	memset(&setd, 0, sizeof(setd));

	ret = pin_self_cpu(mycpu);
	if (ret) {
		warning("CPU%ld cannot pin\n");
	}

	ctx = master_ctx;

	my_sdesc_tab = malloc(sizeof(setdesc_t)*RTOP_NUM_SDESC);
	if (my_sdesc_tab == NULL) {
		warning("CPU%ld cannot allocate sdesc\n", mycpu);
		goto error;
	}

	memcpy(my_sdesc_tab, setdesc_tab, sizeof(setdesc_t)*RTOP_NUM_SDESC);

	fd = pfm_create_context(&ctx, NULL, NULL, 0);
	if (fd == -1) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support!\n");
		}
		warning("CPU%ld cannot create context: %d\n", mycpu, errno);
		goto error;
	}

	/*
	 * WARNING: on processors where the idle loop goes into some power-saving
	 * state, the results of this program may be incorrect
	 */
	for(j=0; j < RTOP_NUM_SDESC; j++) {

		my_sdesc       = my_sdesc_tab+j; 

		setd.set_id    = my_sdesc->set_id;
		setd.set_flags = my_sdesc->set_flags;
		setd.set_timeout =  SWITCH_TIMEOUT; /* in nsecs */

		/*
 		 * do not bother if we have only one set
 		 */
		if (RTOP_NUM_SDESC > 1 && pfm_create_evtsets(fd, &setd, 1) == -1) {
			warning("CPU%ld cannot create set%u: %d\n", mycpu, j, errno);
			goto error;
		}
		my_sdesc->set_timeout = setd.set_timeout;

		if (pfm_write_pmcs(fd, my_sdesc->pc, my_sdesc->outp.pfp_pmc_count) == -1) {
			warning("CPU%ld pfm_write_pmcs error errno %d\n", mycpu, errno);
			goto error;
		}

		/*
	 	 * To be read, each PMD must be either written or declared
	 	 * as being part of a sample (reg_smpl_pmds)
	 	 */
		if (pfm_write_pmds(fd, my_sdesc->pd, my_sdesc->inp.pfp_event_count) == -1) {
			warning("CPU%ld pfm_write_pmds error errno %d\n", mycpu, errno);
			goto error;
		}
	}

	/*
	 * in system-wide mode, this field must provide the CPU the caller wants
	 * to monitor. The kernel checks and if calling from the wrong CPU, the
	 * call fails. The affinity is not affected.
	 */
	load_args.load_pid = mycpu;

	if (pfm_load_context(fd, &load_args) == -1) {
		warning("CPU%ld pfm_load_context error errno %d\n", mycpu, errno);
		goto error;
	}

	thread_info[id].state = THREAD_RUN;

	barrier_wait(&barrier);

	/*
	 * must wait until we are sure the master is out of its thread_create loop
	 */

	barrier_wait(&barrier);

	for(;session_state == SESSION_RUN;) {

		if (pfm_start(fd, NULL) == -1) {
			warning("CPU%ld pfm_start error errno %d\n", mycpu, errno);
			goto error;
		}

		/*
		 * wait for order from master
		 */
		sem_wait(his_sem);

		if (pfm_stop(fd) == -1) {
			warning("CPU%ld pfm_stop error %d\n", mycpu, errno);
			goto error;
		}

		if (old_rows != term_rows) {
			resizeterm(term_rows, term_cols);
			old_rows = term_rows;
		}

		for(j=0; j < RTOP_NUM_SDESC; j++) {

			move(id*RTOP_NUM_SDESC+j, 0);

			if (my_sdesc_tab[j].handler)
				(*my_sdesc_tab[j].handler)(fd, fp, arg, my_sdesc_tab+j);
		}
		if (session_state == SESSION_RUN) {
			sem_post(&arg->my_sem);
			barrier_wait(&barrier);
		}
	}
	if (fp) fclose(fp);

	close(fd);

	thread_info[id].state = THREAD_DONE;

	pthread_exit((void *)(0));
error:
	thread_info[id].state = THREAD_ERROR;
	barrier_wait(&barrier);
	if (fp) fclose(fp);
	if (my_sdesc_tab) free(my_sdesc_tab);
	pthread_exit((void *)(~0));
}

static void
mainloop(void)
{
	long i, j, ncpus = 0;
	int ret;
	void *retval;
	struct pollfd	fds;

	ncpus = options.selected_cpus;

	barrier_init(&barrier, ncpus+1);

	thread_info = malloc(sizeof(thread_desc_t)*ncpus);
	if (thread_info == NULL) {
		fatal_error("cannot allocate thread_desc for %ld CPUs\n", ncpus);
	}

	for(i=0, j = 0; ncpus; i++) {

		if (RTOP_CPUMASK_ISSET(options.cpu_mask, i) == 0) continue;

		thread_info[j].id    = j;
		thread_info[j].cpuid = i;

		sem_init(&thread_info[j].his_sem, 0, 0);
		sem_init(&thread_info[j].my_sem, 0, 0);

		ret = pthread_create(&thread_info[j].tid, 
				     NULL, 
				     (void *(*)(void *))do_measure_one_cpu, 
				     (void *)(thread_info+j));
		if (ret != 0) goto abort;

		ncpus--;
		j++;
	}

	/* set last marker */
	thread_info[j-1].is_last = 1;

	ncpus = j;

	barrier_wait(&barrier);
	/*
	 * check if some threads got problems
	 */
	for(i=0; i < ncpus ; i++) {
		if (thread_info[i].state == THREAD_ERROR) {
			printw("aborting\n"); refresh();
			goto abort;
		}
	}

	session_state = SESSION_RUN;

	barrier_wait(&barrier);

	fds.fd      = 0;
	fds.events  = POLLIN;
	fds.revents = 0;

	for(;time_to_quit == 0;) {
		ret = poll(&fds, 1, options.opt_delay*1000);
		switch(ret) {
			case 0:
				for(i=0; i < ncpus ; i++) {
					/* give order to print */
					sem_post(&thread_info[i].his_sem);
					/* wait for thread to be done */
					sem_wait(&thread_info[i].my_sem);
				}
				/* give order to start measuring again */
				refresh();
				barrier_wait(&barrier);
				break;
			case -1:
				/* restart in case of signal */
				if (errno == EINTR)
					continue;
				warning("polling error: %s\n", strerror(errno));
				 /* fall through */
			default: time_to_quit = 1;
		}
	}
	session_state = SESSION_STOP;

	/*
	 * get worker thread out of their mainloop
	 */
	for (i=0; i < ncpus; i++)
		sem_post(&thread_info[i].his_sem);
join_all:
	for(i=0; i< ncpus; i++) {
		ret = pthread_join(thread_info[i].tid, &retval);
		if (ret !=0) fatal_error("cannot join thread %ld\n", i);

	}
	free(thread_info);
	return;
abort:
	session_state = SESSION_ABORTED;

	for(i=0; i < ncpus; i++) {
		pthread_cancel(thread_info[i].tid);
	}
	goto join_all;
}

static void
setup_measurement(void)
{

	pfmlib_options_t pfmlib_options;
	eventdesc_t *evt;
	setdesc_t *sdesc;
	pfmlib_event_t trigger_event;
	unsigned int i, j;
	int ret;

	/*
	 * pass options to library (optional)
	 */
	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	pfmlib_options.pfm_debug = 0;
	pfmlib_options.pfm_verbose = 0;
	pfm_set_options(&pfmlib_options);

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	ret = pfm_initialize();
	if (ret != PFMLIB_SUCCESS)
		fatal_error("Cannot initialize library: %s\n", pfm_strerror(ret));

	/*
	 * In system wide mode, the perfmon context cannot be inherited.
	 * Also in this mode, we cannot use the blocking form of user level notification.
	 */
	master_ctx.ctx_flags = PFM_FL_SYSTEM_WIDE;

	if (pfm_get_cycle_event(&trigger_event) != PFMLIB_SUCCESS)
		fatal_error("cannot find cycle event for trigger\n");

	for(i=0; i < RTOP_NUM_SDESC; i++) {

		sdesc = setdesc_tab+i;

		sdesc->inp.pfp_dfl_plm   = PFM_PLM3|PFM_PLM0;
		/*
		 * indicate we are using the monitors for a system-wide session.
		 * This may impact the way the library sets up the PMC values.
		 */
		sdesc->inp.pfp_flags = PFMLIB_PFP_SYSTEMWIDE;
		evt = sdesc->evt_desc;

		for(j=0; evt[j].name ; j++) {
			if (*evt[j].name == '*')
				sdesc->inp.pfp_events[j] = trigger_event;
			else if (pfm_find_full_event(evt[j].name, &sdesc->inp.pfp_events[j]) != PFMLIB_SUCCESS)
				fatal_error("cannot find %s event\n", evt[j].name);
			sdesc->inp.pfp_events[j].plm   = evt[j].plm;
		}

		/*
		 * how many counters we use in this set (add the overflow trigger)
		 */
		sdesc->inp.pfp_event_count = j;
		/*
	 	 * build the pfp_unavail_pmcs bitmask by looking
	 	 * at what perfmon has available. It is not always
	 	 * the case that all PMU registers are actually available
	 	 * to applications. For instance, on IA-32 platforms, some
	 	 * registers may be reserved for the NMI watchdog timer.
	 	 *
	 	 * With this bitmap, the library knows which registers NOT to
	 	 * use. Of source, it is possible that no valid assignement may
	 	 * be possible if certina PMU registers  are not available.
	 	 */
		detect_unavail_pmcs(-1, &sdesc->inp.pfp_unavail_pmcs);

		/*
		 * let the library figure out the values for the PMCS
		 */
		if ((ret=pfm_dispatch_events(&sdesc->inp, NULL, &sdesc->outp, NULL)) != PFMLIB_SUCCESS)
			fatal_error("cannot configure events: %s\n", pfm_strerror(ret));

		for (j=0; j < sdesc->outp.pfp_pmc_count; j++) {
			sdesc->pc[j].reg_set   = i;
			sdesc->pc[j].reg_num   = sdesc->outp.pfp_pmcs[j].reg_num;
			sdesc->pc[j].reg_value = sdesc->outp.pfp_pmcs[j].reg_value;
		}
		for (j=0; j < sdesc->outp.pfp_pmd_count; j++) {
			sdesc->pd[j].reg_num = sdesc->outp.pfp_pmds[j].reg_num;
			sdesc->pd[j].reg_set = i;
		}
	}
}

void
populate_cpumask(char *cpu_list)
{
	char *p;
	unsigned long start_cpu, end_cpu = 0;
	unsigned long i, count = 0;

	options.online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
	if (options.online_cpus == -1) 
		fatal_error("cannot figure out the number of online processors\n");

	if (cpu_list == NULL)  {
		/*
		 * The limit is mostly driven by the affinity support in NPTL and glibc __CPU_SETSIZE.
		 * the kernel interface does not expose any limitation.
	         */
		if (options.online_cpus >= RTOP_MAX_CPUS)
			fatal_error("rtop can only handle to %u CPUs\n", RTOP_MAX_CPUS);

		for(i=0; i < options.online_cpus; i++) {
			RTOP_CPUMASK_SET(options.cpu_mask, i);
		}
		options.selected_cpus = options.online_cpus;

		return;
	} 

	while(isdigit(*cpu_list)) { 
		p = NULL;
		start_cpu = strtoul(cpu_list, &p, 0); /* auto-detect base */

		if (start_cpu == ULONG_MAX || (*p != '\0' && *p != ',' && *p != '-')) goto invalid;

		if (p && *p == '-') {
			cpu_list = ++p;
			p = NULL;

			end_cpu = strtoul(cpu_list, &p, 0); /* auto-detect base */
			
			if (end_cpu == ULONG_MAX || (*p != '\0' && *p != ',')) goto invalid;
			if (end_cpu < start_cpu) goto invalid_range; 
		} else {
			end_cpu = start_cpu;
		}

		if (start_cpu >= RTOP_MAX_CPUS || end_cpu >= RTOP_MAX_CPUS) goto too_big;

		for (; start_cpu <= end_cpu; start_cpu++) {

			if (start_cpu >= options.online_cpus) goto not_online; /* XXX: assume contiguous range of CPUs */

			if (RTOP_CPUMASK_ISSET(options.cpu_mask, start_cpu)) continue;

			RTOP_CPUMASK_SET(options.cpu_mask, start_cpu);

			count++;
		}

		if (*p) ++p;

		cpu_list = p;
	}

	options.selected_cpus = count;

	return;
invalid:
	fatal_error("invalid cpu list argument: %s\n", cpu_list);
	/* no return */
not_online:
	fatal_error("cpu %lu is not online\n", start_cpu);
	/* no return */
invalid_range:
	fatal_error("cpu range %lu - %lu is invalid\n", start_cpu, end_cpu);
	/* no return */
too_big:
	fatal_error("rtop is limited to %u CPUs\n", RTOP_MAX_CPUS);
	/* no return */
}


static void
usage(void)
{
	printf(	"usage: rtop [options]:\n"
		"-h, --help\t\t\tdisplay this help and exit\n"
		"-v, --verbose\t\t\tverbose output\n"
		"-V, --version\t\t\tshow version and exit\n"
		"-d nsec, --delay=nsec\t\tnumber of seconds between refresh (default=1s)\n"
		"--cpu-list=cpu1,cpu2\t\tlist of CPUs to monitor(default=all)\n"
	);

}

int
main(int argc, char **argv)
{
	int c;
	char *cpu_list = NULL;

	while ((c=getopt_long(argc, argv,"+vhVd:", rtop_cmd_options, 0)) != -1) {
		switch(c) {
			case   0: continue; /* fast path for options */
			case 'v': options.opt_verbose = 1;
				  break;
			case 1:
			case 'h':
				usage();
				exit(0);
			case 2:
			case 'V':
				printf("rtop version " RTOP_VERSION " Date: " __DATE__ "\n"
					"Copyright (C) 2004 Hewlett-Packard Company\n");
				exit(0);
			case 3:
			case 'd':
				if (options.opt_delay_set) fatal_error("cannot set delay twice\n");
				options.opt_delay     = atoi(optarg);
				if (options.opt_delay < 0) {
					fatal_error("invalid delay, must be >= 0\n");
				}
				options.opt_delay_set = 1;
				break;
			case 4:
				if (cpu_list) fatal_error("cannot specify --cpu-list more than once\n");
				if (*optarg == '\0') fatal_error("--cpu-list needs an argument\n");
				cpu_list = optarg;
				break;
			case 5:
				if (options.outfile) fatal_error("cannot specify --outfile more than once\n");
				if (*optarg == '\0') fatal_error("--outfile needs an argument\n");
				options.outfile = optarg;
				break;
			default:
				fatal_error("unknown option\n");
		}
	}
	/*
	 * default refresh delay
	 */
	if (options.opt_delay_set == 0) options.opt_delay = 1;

	options.cpu_mhz = find_cpu_speed();

	populate_cpumask(cpu_list);

	setup_measurement();
	setup_signals();
	setup_screen();
	mainloop();
	close_screen();

	return 0;
}
