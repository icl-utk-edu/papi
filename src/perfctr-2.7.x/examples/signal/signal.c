/* $Id: signal.c,v 1.20 2005/04/08 14:37:55 mikpe Exp $
 *
 * This test program illustrates how performance counter overflow
 * can be caught and sent to the process as a user-specified signal.
 *
 * Limitations:
 * - Requires a 2.4 or newer kernel with local APIC support.
 * - Requires a CPU with a local APIC (P4, P6, K8, K7).
 *
 * Copyright (C) 2001-2004  Mikael Pettersson
 */
#define __USE_GNU /* enable symbolic names for gregset_t[] indices */
#include <sys/ucontext.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libperfctr.h"
#include "arch.h"

static struct vperfctr *vperfctr;
static struct perfctr_info info;

static void do_open(void)
{
    vperfctr = vperfctr_open();
    if( !vperfctr ) {
	perror("vperfctr_open");
	exit(1);
    }
    if( vperfctr_info(vperfctr, &info) < 0 ) {
	perror("vperfctr_info");
	exit(1);
    }
    if( !(info.cpu_features & PERFCTR_FEATURE_PCINT) )
	printf("PCINT not supported -- expect failure\n");
}

static void on_sigio(int sig, siginfo_t *si, void *puc)
{
    struct ucontext *uc;
    unsigned long pc;
    unsigned int pmc_mask;

    if( sig != SIGIO ) {
	printf("%s: unexpected signal %d\n", __FUNCTION__, sig);
	return;
    }
    if( si->si_code != SI_PMC_OVF ) {
	printf("%s: unexpected si_code #%x\n", __FUNCTION__, si->si_code);
	return;
    }
    if( (pmc_mask = si->si_pmc_ovf_mask) == 0 ) {
	printf("%s: overflow PMCs not identified\n", __FUNCTION__);
	return;
    }
    uc = puc;
    pc = ucontext_pc(uc);
    if( !vperfctr_is_running(vperfctr) ) {
	/*
	 * My theory is that this happens if a perfctr overflowed
	 * at the very instruction for the VPERFCTR_STOP call.
	 * Signal delivery is delayed until the kernel returns to
	 * user-space, at which time VPERFCTR_STOP will already
	 * have marked the vperfctr as stopped. In this case, we
	 * cannot and must not attempt to IRESUME it.
	 * This can be triggered by counting e.g. BRANCHES and setting
	 * the overflow limit ridiculously low.
	 */
	printf("%s: unexpected overflow from PMC set %#x at pc %#lx\n",
	       __FUNCTION__, pmc_mask, pc);
	return;
    }
    printf("%s: PMC overflow set %#x at pc %#lx\n", __FUNCTION__, pmc_mask, pc);
    if( vperfctr_iresume(vperfctr) < 0 ) {
	perror("vperfctr_iresume");
	abort();
    }
}

static void do_sigaction(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_sigio;
    sa.sa_flags = SA_SIGINFO;
    if( sigaction(SIGIO, &sa, NULL) < 0 ) {
	perror("sigaction");
	exit(1);
    }
}

static void do_control(void)
{
    struct vperfctr_control control;

    memset(&control, 0, sizeof control);
    do_setup(&info, &control.cpu_control);
    control.si_signo = SIGIO;

    printf("Control used:\n");
    perfctr_cpu_control_print(&control.cpu_control);
    printf("\n");

    if( vperfctr_control(vperfctr, &control) < 0 ) {
	perror("vperfctr_control");
	exit(1);
    }
}

static void do_stop(void)
{
    struct sigaction sa;

    if( vperfctr_stop(vperfctr) )
	perror("vperfctr_stop");
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = SIG_DFL;
    if( sigaction(SIGIO, &sa, NULL) < 0 ) {
	perror("sigaction");
	exit(1);
    }
}

#define N 150
static double v[N], w[N];
static double it;

static void do_dotprod(void)
{
    int i;
    double sum;

    sum = 0.0;
    for(i = 0; i < N; ++i)
	sum += v[i] * w[i];
    it = sum;
}

int main(void)
{
    do_sigaction();
    do_open();
    do_control();
    do_dotprod();
    do_stop();
    return 0;
}
