/* $Id$
 * global.c
 *
 * usage: ./global [sampling_interval_usec [sleep_interval_sec]]
 *
 * This test program illustrates how a process may use the
 * Linux x86 Performance-Monitoring Counters interface to
 * do system-wide performance monitoring.
 *
 * Copyright (C) 2000-2001  Mikael Pettersson
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

struct gperfctr *gperfctr;
struct perfctr_info info;
#define have_rdtsc (info.cpu_features & PERFCTR_FEATURE_RDTSC)
struct gperfctr_control *control;
struct gperfctr_state *state;
unsigned long sampling_interval = 1000000;
unsigned int sleep_interval = 5;
unsigned int sample_num = 0;
struct gperfctr_state *prev_state;
int counting_mips = 0;	/* default is to count MFLOP per second */

jmp_buf main_buf;

void onint(int sig)	/* ^C handler */
{
    longjmp(main_buf, 1);
}

void do_init(void)
{
    struct sigaction act;
    size_t nbytes;

    gperfctr = gperfctr_open();
    if( !gperfctr ) {
	perror("gperfctr_open");
	exit(1);
    }
    if( gperfctr_info(gperfctr, &info) < 0 ) {
	perror("gperfctr_info");
	exit(1);
    }
    printf("\nPerfCtr Info:\n");
    perfctr_print_info(&info);

    memset(&act, 0, sizeof act);
    act.sa_handler = onint;
    if( sigaction(SIGINT, &act, NULL) < 0 ) {
	perror("unable to catch SIGINT");
	exit(1);
    }

    /* now alloc control and state memory based on nrcpus */

    nbytes = offsetof(struct gperfctr_control, cpu_control[0])
	+ info.nrcpus * sizeof(control->cpu_control[0]);
    control = malloc(nbytes);
    if( !control ) {
	perror("malloc");
	exit(1);
    }
    memset(control, 0, nbytes);

    nbytes = offsetof(struct gperfctr_state, cpu_state[0])
	+ info.nrcpus * sizeof(state->cpu_state[0]);
    state = malloc(nbytes);
    prev_state = malloc(nbytes);
    if( !state || !prev_state ) {
	perror("malloc");
	exit(1);
    }
    memset(state, 0, nbytes);
    memset(prev_state, 0, nbytes);
}

int do_read(void)
{
    int cpu, ctr, nactive;

    state->nrcpus = info.nrcpus;
    if( (nactive = gperfctr_read(gperfctr, state)) < 0 ) {
	perror("gperfctr_read");
	return -1;
    }
    printf("\nSample #%u\n", ++sample_num);
    for(cpu = 0; cpu < nactive; ++cpu) {
	printf("\nCPU %d:\n", cpu);
	if( have_rdtsc )	/* don't print TSC unless it's real */
	    printf("\ttsc\t%lld\n", state->cpu_state[cpu].sum.tsc);
	for(ctr = 0; ctr < state->cpu_state[cpu].cpu_control.nractrs; ++ctr)
	    printf("\tpmc[%d]\t%lld\n",
		   ctr, state->cpu_state[cpu].sum.pmc[ctr]);
	if( ctr == 1 ) {	/* compute and display MFLOP/s or MIP/s */
	    unsigned int tsc = state->cpu_state[cpu].sum.tsc;
	    unsigned int prev_tsc = prev_state->cpu_state[cpu].sum.tsc;
	    unsigned int ticks = tsc - prev_tsc;
	    unsigned int pmc0 = state->cpu_state[cpu].sum.pmc[0];
	    unsigned int prev_pmc0 = prev_state->cpu_state[cpu].sum.pmc[0];
	    unsigned int ops = pmc0 - prev_pmc0;
	    double seconds = have_rdtsc
		? ((double)ticks / (double)info.cpu_khz) / 1000.0
		: (double)sleep_interval; /* don't div-by-0 on WinChip ... */
	    printf("\tSince previous sample:\n");
	    printf("\tSECONDS\t%.15g\n", seconds);
	    printf("\t%s\t%u\n", counting_mips ? "INSNS" : "FLOPS", ops);
	    printf("\t%s/s\t%.15g\n",
		   counting_mips ? "MIP" : "MFLOP",
		   ((double)ops / seconds) / 1e6);
	    prev_state->cpu_state[cpu].sum.tsc = tsc;
	    prev_state->cpu_state[cpu].sum.pmc[0] = pmc0;
	}
    }
    return 0;
}

void setup_control(struct perfctr_cpu_control *control)
{
    unsigned evntsel0;

    control->tsc_on = 1;
    switch( info.cpu_type ) {
      case PERFCTR_X86_GENERIC:
	return;
      case PERFCTR_X86_INTEL_P5:
      case PERFCTR_X86_INTEL_P5MMX:
      case PERFCTR_X86_CYRIX_MII:
	/* FLOPS, any CPL */
	evntsel0 = 0x22 | (3 << 6);
	break;
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
	/* FLOPS, any CPL, ENable */
	evntsel0 = 0xC1 | (3 << 16) | (1 << 22);
	break;
      case PERFCTR_X86_AMD_K7:
	/* "AMD Athlon Processor x86 Code Optimization Guide, Rev. H".
	   doesn't list any event for FLOPS. Amazing. */
	counting_mips = 1;
	/* RETIRED_INSTRUCTIONS, any CPL, ENable */	
	evntsel0 = 0xC0 | (3 << 16) | (1 << 22);
	break;
      case PERFCTR_X86_WINCHIP_C6:
	/* Can't count FLOPS :-( */
	counting_mips = 1;
	evntsel0 = 0x02;		/* X86_INSTRUCTIONS */
	break;
      case PERFCTR_X86_WINCHIP_2:
	/* Can't count FLOPS :-( */
	counting_mips = 1;
	evntsel0 = 0x16;		/* INSTRUCTIONS_EXECUTED */
	break;
      case PERFCTR_X86_VIA_C3:
	/* Can't count FLOPS :-( */
	counting_mips = 1;
	evntsel0 = 0xC0;		/* INSTRUCTIONS_EXECUTED */
	break;
      default:
	fprintf(stderr, "cpu_type %u (%s) not supported\n",
		info.cpu_type, perfctr_cpu_name(&info));
	exit(1);
    }
    control->nractrs = 1;
    control->evntsel[0] = evntsel0;
    /* XXX: pmc_map[] fixup here */
    printf("\nControl used:\n");
    printf("\tevntsel[%u]\t0x%08X\n", 0, control->evntsel[0]);
}

void do_enable(unsigned long sampling_interval)
{
    int cpu;

    setup_control(&control->cpu_control[0]);

    for(cpu = 1; cpu < info.nrcpus; ++cpu)
	control->cpu_control[cpu] = control->cpu_control[0];
    control->interval_usec = sampling_interval;
    control->nrcpus = info.nrcpus;		/* use all available cpus */

    if( gperfctr_control(gperfctr, control) < 0 ) {
	perror("gperfctr_control");
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
    gperfctr_stop(gperfctr);
    return 0;
}
