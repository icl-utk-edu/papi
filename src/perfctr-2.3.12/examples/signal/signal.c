/* $Id$
 * signal.c
 *
 * This test program illustrates how performance counter overflow
 * can be caught and sent to the process as a user-specified signal.
 *
 * Limitations:
 * - Requires a 2.4 kernel with UP-APIC support.
 * - Requires an Intel P4, Intel P6, or AMD K7 CPU.
 *
 * Copyright (C) 2001-2002  Mikael Pettersson
 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <asm/sigcontext.h>
#include <asm/ucontext.h>	/* _not_ the broken <sys/ucontext.h> */
#include "libperfctr.h"

static const struct vperfctr *vperfctr;
static struct perfctr_info info;

static void do_open(void)
{
    vperfctr = vperfctr_open();
    if( !vperfctr ) {
	perror("vperfctr_open");
	exit(1);
    }
    if( vperfctr_info(vperfctr, &info) != 0 ) {
	perror("vperfctr_info");
	exit(1);
    }
    if( !(info.cpu_features & PERFCTR_FEATURE_PCINT) )
	printf("PCINT not supported -- expect failure\n");
}

static void on_sigio(int sig, siginfo_t *si, void *puc)
{
    struct ucontext *uc;
    struct sigcontext *mc;
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
    mc = &uc->uc_mcontext;
    pc = mc->eip;	/* clearly more readable than glibc's mc->gregs[14] */
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
    unsigned int nractrs = 0;
    unsigned int pmc_map0 = 0, pmc_map1 = 1;
    unsigned int evntsel0, evntsel1;
    unsigned int evntsel0_aux = 0, evntsel1_aux = 0;
    struct vperfctr_control control;

    memset(&control, 0, sizeof control);

    switch( info.cpu_type ) {
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
	/* FLOPS, USR, ENable, INT */
	evntsel0 = 0xC1 | (1 << 16) | (1 << 22) | (1 << 20);
	/* BR_TAKEN_RETIRED, USR, INT */
	evntsel1 = 0xC9 | (1 << 16) | (1 << 20);
	break;
      case PERFCTR_X86_AMD_K7:
	/* K7 can't count FLOPS. Count RETIRED_OPS instead. */
	evntsel0 = 0xC1 | (1 << 16) | (1 << 22) | (1 << 20);
	/* RETIRED_TAKEN_BRANCHES, USR, INT */
	evntsel1 = 0xC4 | (1 << 16) | (1 << 22) | (1 << 20);
	break;
      case PERFCTR_X86_INTEL_P4:
	nractrs = 1;
	/* PMC(0) produces tagged x87_FP_uop:s (FLAME_CCCR0, FIRM_ESCR0) */
	control.cpu_control.pmc_map[0] = 0x8 | (1 << 31);
	control.cpu_control.evntsel[0] = (0x3 << 16) | (1 << 13) | (1 << 12);
	control.cpu_control.evntsel_aux[0] = (4 << 25) | (1 << 24) | (1 << 5) | (1 << 4) | (1 << 2);
	/* PMC(1) counts execution_event(X87_FP_retired) (IQ_CCCR0, CRU_ESCR2) */
	pmc_map0 = 0xC | (1 << 31);
	evntsel0 = (1 << 26) | (0x3 << 16) | (5 << 13) | (1 << 12);
	evntsel0_aux = (0xC << 25) | (1 << 9) | (1 << 2);
	/* PMC(2) counts branch_retired(TP,TM) (IQ_CCCR2, CRU_ESCR3) */
	pmc_map1 = 0xE | (1 << 31);
	evntsel1 = (1 << 26) | (0x3 << 16) | (5 << 13) | (1 << 12);
	evntsel1_aux = (6 << 25) | (((1 << 3)|(1 << 2)) << 9) | (1 << 2);
	break;
      default:
	printf("%s: unsupported cpu type %u\n", __FUNCTION__, info.cpu_type);
	exit(1);
    }	
    control.cpu_control.tsc_on = 1;
    control.cpu_control.nractrs = nractrs;
    control.cpu_control.nrictrs = 2;
    control.cpu_control.pmc_map[nractrs+0] = pmc_map0;
    control.cpu_control.evntsel[nractrs+0] = evntsel0;
    control.cpu_control.evntsel_aux[nractrs+0] = evntsel0_aux;
    control.cpu_control.ireset[nractrs+0] = -25;
    control.cpu_control.pmc_map[nractrs+1] = pmc_map1;
    control.cpu_control.evntsel[nractrs+1] = evntsel1;
    control.cpu_control.evntsel_aux[nractrs+1] = evntsel1_aux;
    control.cpu_control.ireset[nractrs+1] = -25;
    control.si_signo = SIGIO;
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
