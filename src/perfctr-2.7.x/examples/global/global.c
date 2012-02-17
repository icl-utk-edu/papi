/* $Id: global.c,v 1.37 2004/01/12 14:25:40 mikpe Exp $
 *
 * usage: ./global [sampling_interval_usec [sleep_interval_sec]]
 *
 * This test program illustrates how a process may use the
 * Linux x86 Performance-Monitoring Counters interface to
 * do system-wide performance monitoring.
 *
 * Copyright (C) 2000-2004  Mikael Pettersson
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
#include "arch.h"

static struct gperfctr *gperfctr;
static struct perfctr_info info;
static unsigned int nrcpus;
static unsigned short *cpu_logical_map;
struct gperfctr_state {	/* no longer defined in or used by the kernel */
    unsigned int nrcpus;
    struct gperfctr_cpu_state cpu_state[1]; /* actually 'nrcpus' */
};
static struct gperfctr_state *state;
static struct gperfctr_state *prev_state;
static unsigned int sample_num;
int counting_mips;	/* for CPUs that cannot FLOPS */
static unsigned long sampling_interval = 1000000; /* XXX: reduce for >4GHz CPUs */
static unsigned int sleep_interval = 5;

static jmp_buf main_buf;

static void onint(int sig)	/* ^C handler */
{
    longjmp(main_buf, 1);
}

static void catch_sigint(void)
{
    struct sigaction act;

    memset(&act, 0, sizeof act);
    act.sa_handler = onint;
    if( sigaction(SIGINT, &act, NULL) < 0 ) {
	perror("unable to catch SIGINT");
	exit(1);
    }
}

static unsigned int hweight32(unsigned int w)
{
    unsigned int res = (w & 0x55555555) + ((w >> 1) & 0x55555555);
    res = (res & 0x33333333) + ((res >> 2) & 0x33333333);
    res = (res & 0x0F0F0F0F) + ((res >> 4) & 0x0F0F0F0F);
    res = (res & 0x00FF00FF) + ((res >> 8) & 0x00FF00FF);
    return (res & 0x0000FFFF) + ((res >> 16) & 0x0000FFFF);
}

static void setup_cpu_logical_map_and_nrcpus(const struct perfctr_cpus_info *cpus_info)
{
    const unsigned int *cpus, *cpus_forbidden;
    unsigned int nrwords, i, cpumask, bitmask;
    unsigned int logical_cpu_nr, kernel_cpu_nr;

    cpus = cpus_info->cpus->mask;
    cpus_forbidden = cpus_info->cpus_forbidden->mask;
    nrwords = cpus_info->cpus->nrwords;

    nrcpus = 0;
    for(i = 0; i < nrwords; ++i)
	nrcpus += hweight32(cpus[i] & ~cpus_forbidden[i]);

    cpu_logical_map = malloc(nrcpus*sizeof(cpu_logical_map[0]));
    if( !cpu_logical_map ) {
	perror("malloc");
	exit(1);
    }

    logical_cpu_nr = 0;
    for(i = 0; i < nrwords; ++i) {
	cpumask = cpus[i] & ~cpus_forbidden[i];
	kernel_cpu_nr = i * 8 * sizeof(int);
	for(bitmask = 1; cpumask != 0; ++kernel_cpu_nr, bitmask <<= 1) {
	    if( cpumask & bitmask ) {
		cpumask &= ~bitmask;
		cpu_logical_map[logical_cpu_nr] = kernel_cpu_nr;
		++logical_cpu_nr;
	    }
	}
    }

    if( logical_cpu_nr != nrcpus )
	abort();
}

static void do_init(void)
{
    struct perfctr_cpus_info *cpus_info;
    size_t nbytes;
    unsigned int i;

    gperfctr = gperfctr_open();
    if( !gperfctr ) {
	perror("gperfctr_open");
	exit(1);
    }
    if( gperfctr_info(gperfctr, &info) < 0 ) {
	perror("gperfctr_info");
	exit(1);
    }
    cpus_info = gperfctr_cpus_info(gperfctr);
    if( !cpus_info ) {
	perror("gperfctr_info");
	exit(1);
    }
    printf("\nPerfCtr Info:\n");
    perfctr_info_print(&info);
    perfctr_cpus_info_print(cpus_info);

    /* use all non-forbidden CPUs */

    setup_cpu_logical_map_and_nrcpus(cpus_info);
    free(cpus_info);

    /* now alloc state memory based on nrcpus */

    nbytes = offsetof(struct gperfctr_state, cpu_state[0])
	+ nrcpus * sizeof(state->cpu_state[0]);
    state = malloc(nbytes);
    prev_state = malloc(nbytes);
    if( !state || !prev_state ) {
	perror("malloc");
	exit(1);
    }
    memset(state, 0, nbytes);
    memset(prev_state, 0, nbytes);

    /* format state to indicate which CPUs we want to sample */

    for(i = 0; i < nrcpus; ++i)
	state->cpu_state[i].cpu = cpu_logical_map[i];
    state->nrcpus = nrcpus;
}

static int do_read(unsigned int sleep_interval)
{
    unsigned int i, cpu, ctr;

    for(i = 0; i < state->nrcpus; ++i) {
	if( gperfctr_read(gperfctr, &state->cpu_state[i]) < 0 ) {
	    perror("gperfctr_read");
	    return -1;
	}
    }
    printf("\nSample #%u\n", ++sample_num);
    for(i = 0; i < state->nrcpus; ++i) {
	cpu = state->cpu_state[i].cpu;
	printf("\nCPU %d:\n", cpu);
	if( state->cpu_state[i].cpu_control.tsc_on )
	    printf("\ttsc\t%lld\n", state->cpu_state[i].sum.tsc);
	for(ctr = 0; ctr < state->cpu_state[i].cpu_control.nractrs; ++ctr)
	    printf("\tpmc[%d]\t%lld\n",
		   ctr, state->cpu_state[i].sum.pmc[ctr]);
	if( ctr >= 1 ) {	/* compute and display MFLOP/s or MIP/s */
	    unsigned long long tsc = state->cpu_state[i].sum.tsc;
	    unsigned long long prev_tsc = prev_state->cpu_state[i].sum.tsc;
	    unsigned long long ticks = tsc - prev_tsc;
	    unsigned long long pmc0 = state->cpu_state[i].sum.pmc[0];
	    unsigned long long prev_pmc0 = prev_state->cpu_state[i].sum.pmc[0];
	    unsigned long long ops = pmc0 - prev_pmc0;
	    double seconds = state->cpu_state[i].cpu_control.tsc_on
		? ((double)ticks * (double)(info.tsc_to_cpu_mult ? : 1) / (double)info.cpu_khz) / 1000.0
		: (double)sleep_interval; /* don't div-by-0 on WinChip ... */
	    printf("\tSince previous sample:\n");
	    printf("\tSECONDS\t%.15g\n", seconds);
	    printf("\t%s\t%llu\n", counting_mips ? "INSNS" : "FLOPS", ops);
	    printf("\t%s/s\t%.15g\n",
		   counting_mips ? "MIP" : "MFLOP",
		   ((double)ops / seconds) / 1e6);
	    prev_state->cpu_state[i].sum.tsc = tsc;
	    prev_state->cpu_state[i].sum.pmc[0] = pmc0;
	}
    }
    return 0;
}

static void print_control(const struct perfctr_cpu_control *control)
{
    printf("\nControl used:\n");
    perfctr_cpu_control_print(control);
}

static void do_enable(unsigned long sampling_interval)
{
    struct perfctr_cpu_control cpu_control;
    unsigned int i;

    setup_control(&info, &cpu_control);
    print_control(&cpu_control);

    for(i = 0; i < nrcpus; ++i) {
	struct gperfctr_cpu_control control;
	control.cpu = cpu_logical_map[i];
	control.cpu_control = cpu_control;
	if( gperfctr_control(gperfctr, &control) < 0 ) {
	    perror("gperfctr_control");
	    exit(1);
	}
    }
    if( gperfctr_start(gperfctr, sampling_interval) < 0 ) {
	perror("gperfctr_start");
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

    if( setjmp(main_buf) == 0 ) {
	catch_sigint();
	do_init();
	do_enable(sampling_interval);
	printf("\nSampling interval:\t%lu usec\n", sampling_interval);
	printf("Sleep interval:\t\t%u sec\n", sleep_interval);
	do {
	    sleep(sleep_interval);
	} while( do_read(sleep_interval) == 0 );
    }
    if( gperfctr ) {
	printf("shutting down..\n");
	gperfctr_stop(gperfctr);
    }
    return 0;
}
