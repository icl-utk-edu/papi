/* $Id$
 * self.c
 *
 * This test program illustrates how a process may use the
 * Linux x86 Performance-Monitoring Counters interface to
 * monitor its own execution.
 *
 * The library uses mmap() to map the kernel's accumulated counter
 * state into the process' address space.
 * When perfctr_read_state() is called, it uses the RDPMC and RDTSC
 * instructions to get the current register values, then adds
 * these to the base values found in the mapped-in kernel state.
 * The resulting counts are then delivered to the application.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "libperfctr.h"

static struct vperfctr *self;
static unsigned nrctrs; /* > 1 if we have PMCs in addition to a TSC */
static unsigned evntsel0;

void do_init(void)
{
    struct perfctr_info info;

    self = vperfctr_open();
    if( self ) {			/* /proc/self/perfctr interface */
	if( vperfctr_info(self, &info) < 0 ) {
	    perror("vperfctr_info");
	    exit(1);
	}
    } else {				/* /dev/perfctr interface */
	struct perfctr_dev *dev = perfctr_dev_open();
	if( !dev ) {
	    perror("perfctr_dev_open");
	    exit(1);
	}
	if( perfctr_info(dev, &info) < 0 ) {
	    perror("perfctr_info");
	    exit(1);
	}
	if( (self = vperfctr_attach(dev)) == NULL ) {
	    perror("vperfctr_attach");
	    exit(1);
	}
	perfctr_dev_close(dev);
    }
    nrctrs = perfctr_cpu_nrctrs(&info);
    if( nrctrs > 1 )
	evntsel0 = perfctr_evntsel_num_insns(&info);
    printf("\nPerfCtr Info:\n");
    printf("driver_version\t\t%u.%u", info.version_major, info.version_minor);
    if( info.version_micro )
	printf(".%u", info.version_micro);
    printf("\n");
    printf("nrcpus\t\t\t%u\n", info.nrcpus);
    printf("cpu_type\t\t%u (%s)\n", info.cpu_type, perfctr_cpu_name(&info));
    printf("cpu_features\t\t0x%x\n", info.cpu_features);
    printf("cpu_khz\t\t\t%lu\n", info.cpu_khz);
    printf("nrctrs\t\t\t%u\n", nrctrs);
}

void do_read(void)
{
    struct vperfctr_state state;

    if( vperfctr_read_state(self, &state) < 0 ) {
	 perror("vperfctr_read_state");
	 exit(1);
    }
    printf("\nCurrent Sample:\n");
    printf("status\t\t\t%d\n", state.status);
    printf("control_id\t\t%u\n", state.control_id);
    if( nrctrs > 1 )
	printf("control.evntsel[0]\t0x%08X\n", state.control.evntsel[0]);
    printf("tsc\t\t\t0x%016llX\n", state.sum.ctr[0]);
    if( nrctrs > 1 )
	printf("pmc[0]\t\t\t0x%016llX\n", state.sum.ctr[1]);
}

void do_enable(void)
{
    struct perfctr_control control;

    memset(&control, 0, sizeof control);
    if( nrctrs > 1 )
	control.evntsel[0] = evntsel0;
    if( vperfctr_control(self, &control) < 0 ) {
	perror("vperfctr_control");
	exit(1);
    }
}

unsigned fac(unsigned n)
{
    return (n < 2) ? 1 : n * fac(n-1);
}

void do_fac(unsigned n)
{
    printf("\nfac(%u) == %u\n", n, fac(n));
}

int main(void)
{
    do_init();
    do_read();
    do_enable();
    do_fac(15);
    do_read();
    return 0;
}
