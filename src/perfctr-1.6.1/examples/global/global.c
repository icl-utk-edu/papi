/* $Id$
 * global.c
 *
 * usage: ./global [sampling_interval_usec [sleep_interval_sec]]
 *
 * This test program illustrates how a process may use the
 * Linux x86 Performance-Monitoring Counters interface to
 * do system-wide performance monitoring.
 *
 * Copyright (C) 2000  Mikael Pettersson
 */
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "libperfctr.h"

struct perfctr_dev *dev;
unsigned cpu_type;
unsigned nrcpus;
struct gperfctr_control *control;
struct gperfctr_state *state;
unsigned long sampling_interval = 1000000;
unsigned int sleep_interval = 5;
unsigned int sample_num = 0;

jmp_buf main_buf;

void onint(int sig)	/* ^C handler */
{
    longjmp(main_buf, 1);
}

void do_init(void)
{
    struct perfctr_info info;
    struct sigaction act;
    size_t nbytes;

    dev = perfctr_dev_open();
    if( !dev ) {
	perror("perfctr_dev_open");
	exit(1);
    }
    if( perfctr_info(dev, &info) < 0 ) {
	perror("perfctr_info");
	exit(1);
    }
    nrcpus = info.nrcpus;
    cpu_type = info.cpu_type;
    printf("\nPerfCtr Info:\n");
    printf("driver_version\t\t%u.%u", info.version_major, info.version_minor);
    if( info.version_micro )
	printf(".%u", info.version_micro);
    printf("\n");
    printf("nrcpus\t\t\t%u\n", nrcpus);
    printf("cpu_type\t\t%u (%s)\n", info.cpu_type, perfctr_cpu_name(dev));
    printf("cpu_features\t\t0x%x\n", info.cpu_features);
    printf("cpu_khz\t\t\t%lu\n", info.cpu_khz);
    printf("nrctrs\t\t\t%u\n", perfctr_cpu_nrctrs(dev));

    memset(&act, 0, sizeof act);
    act.sa_handler = onint;
    if( sigaction(SIGINT, &act, NULL) < 0 ) {
	perror("unable to catch SIGINT");
	exit(1);
    }

    /* now alloc control and state memory based on nrcpus */

    nbytes = offsetof(struct gperfctr_control, cpu_control[0])
	+ nrcpus * sizeof(control->cpu_control[0]);
    control = malloc(nbytes);
    if( !control ) {
	perror("malloc");
	exit(1);
    }
    memset(control, 0, nbytes);

    nbytes = offsetof(struct gperfctr_state, cpu_state[0])
	+ nrcpus * sizeof(state->cpu_state[0]);
    state = malloc(nbytes);
    if( !state ) {
	perror("malloc");
	exit(1);
    }
    memset(state, 0, nbytes);
}

int do_read(void)
{
    int i, j, nactive;

    state->nrcpus = nrcpus;
    if( (nactive = perfctr_global_read(dev, state)) < 0 ) {
	perror("perfctr_global_read");
	return -1;
    }
    printf("\nSample #%u\n", ++sample_num);
    for(i = 0; i < nactive; ++i) {
	printf("\nCPU %d:\n", i);
	printf("\ttsc\t0x%016llX\n", state->cpu_state[i].sum.ctr[0]);
	for(j = 1; j < state->cpu_state[i].nrctrs; ++j)
	    printf("\tpmc[%d]\t0x%016llX\n", j-1, state->cpu_state[i].sum.ctr[j]);
    }
    return 0;
}

void setup_control(struct perfctr_control *control)
{
    unsigned evntsel0, evntsel1, i, n;

    switch( cpu_type ) {
      case PERFCTR_X86_GENERIC:
	return;
      case PERFCTR_X86_INTEL_P5:
      case PERFCTR_X86_INTEL_P5MMX:
      case PERFCTR_X86_CYRIX_MII:
	evntsel0 = 0x12 | (1 << 6);	/* BRANCHES, CPL 0-2 */
	evntsel1 = 0x13 | (1 << 6);	/* BTB_HITS, CPL 0-2 */
	control->evntsel[0] = evntsel0 | (evntsel1 << 16);
	n = 1;
	break;
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
	evntsel0 = 0xC4 | (1 << 17);	/* BR_INST_RETIRED, CPL 0 */
	evntsel1 = 0xC5 | (1 << 17);	/* BR_MISS_PRED_RETIRED, CPL 0 */
	control->evntsel[0] = evntsel0 | (1 << 22);	/* ENable */
	control->evntsel[1] = evntsel1;
	n = 2;
	break;
      case PERFCTR_X86_AMD_K7:
	evntsel0 = 0xC2 | (1 << 17);	/* RETIRED_BRANCHES, CPL 0 */
	evntsel1 = 0xC3 | (1 << 17);	/* RETIRED_BRANCHES_MISPREDICTED, CPL 0 */
	control->evntsel[0] = evntsel0 | (1 << 22);	/* ENable */
	control->evntsel[1] = evntsel1 | (1 << 22);	/* ENable */
	n = 2;
	break;
      default:
	fprintf(stderr, "cpu_type %u (%s) not supported\n",
		cpu_type, perfctr_cpu_name(dev));
	exit(1);
    }
    printf("\nControl used:\n");
    for(i = 0; i < n; ++i)
	printf("\tevntsel[%u]\t0x%08X\n", i, control->evntsel[i]);
}

void do_enable(unsigned long sampling_interval)
{
    int i;

    setup_control(&control->cpu_control[0]);

    for(i = 1; i < nrcpus; ++i)
	control->cpu_control[i] = control->cpu_control[0];
    control->interval_usec = sampling_interval;
    control->nrcpus = nrcpus;		/* use all available cpus */

    if( perfctr_global_control(dev, control) < 0 ) {
	perror("perfctr_global_control");
	exit(1);
    }
}

int main(int argc, const char **argv)
{
    if( argc >= 2 ) {
	sampling_interval = strtoul(argv[1], NULL, 0);
	if( argc >= 3 )
	    sleep_interval = strtoul(argv[2], NULL, 0);
    }

    do_init();
    do_enable(sampling_interval);

    printf("\nSampling interval:\t%lu usec\n", sampling_interval);
    printf("Sleep interval:\t\t%u sec\n", sleep_interval);

    if( setjmp(main_buf) == 0 ) {
	do {
	    sleep(sleep_interval);
	} while( do_read() == 0 );
    }
    printf("shutting down..\n");
    perfctr_global_stop(dev);
    return 0;
}
