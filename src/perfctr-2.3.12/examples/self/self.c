/* $Id$
 * self.c
 *
 * This test program illustrates how a process may use the
 * Linux x86 Performance-Monitoring Counters interface to
 * monitor its own execution.
 *
 * The library uses mmap() to map the kernel's accumulated counter
 * state into the process' address space.
 * When perfctr_read_ctrs() is called, it uses the RDPMC and RDTSC
 * instructions to get the current register values, and combines
 * these with (sum,start) values found in the mapped-in kernel state.
 * The resulting counts are then delivered to the application.
 *
 * Copyright (C) 1999-2001  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "libperfctr.h"

static struct vperfctr *self;
static unsigned nrctrs;	/* # of PMCs, does not include TSC */
static unsigned evntsel0;

void do_init(void)
{
    struct perfctr_info info;

    self = vperfctr_open();
    if( !self ) {
	perror("vperfctr_open");
	exit(1);
    }
    if( vperfctr_info(self, &info) < 0 ) {
	perror("vperfctr_info");
	exit(1);
    }
    nrctrs = perfctr_cpu_nrctrs(&info);
    if( nrctrs )
	evntsel0 = perfctr_evntsel_num_insns(&info);
    printf("\nPerfCtr Info:\n");
    perfctr_print_info(&info);
}

void do_read(struct perfctr_sum_ctrs *sum)
{
    /*
     * This is the preferred method for sampling all enabled counters.
     * It doesn't return control data or current kernel-level state though.
     * The control data can be retrieved using vperfctr_read_state().
     *
     * Alternatively you may call vperfctr_read_tsc() or vperfctr_read_pmc()
     * to sample a single counter's value.
     */
    vperfctr_read_ctrs(self, sum);
}

void do_enable(void)
{
    struct vperfctr_control control;

    memset(&control, 0, sizeof control);
    control.cpu_control.tsc_on = 1;
    if( nrctrs ) {
	control.cpu_control.nractrs = 1;
	control.cpu_control.evntsel[0] = evntsel0;
	/* XXX: pmc_map[] fixup here */
    }
    if( vperfctr_control(self, &control) < 0 ) {
	perror("vperfctr_control");
	exit(1);
    }
}

void do_print(const struct perfctr_sum_ctrs *before,
	      const struct perfctr_sum_ctrs *after)
{
    printf("\nFinal Sample:\n");
    if( nrctrs )
	printf("evntsel[0]\t\t0x%08X\n", evntsel0);
    printf("tsc\t\t\t%lld\n", after->tsc - before->tsc);
    if( nrctrs )
	printf("pmc[0]\t\t\t%lld\n", after->pmc[0] - before->pmc[0]);
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
    struct perfctr_sum_ctrs before, after;

    do_init();
    do_enable();
    do_read(&before);
    do_fac(15);
    do_read(&after);
    do_print(&before, &after);
    return 0;
}
