/* $Id$
 * signal.c
 *
 * This test program illustrates how performance counter overflow
 * can be caught and sent to the process as a user-specified signal.
 *
 * Limitations:
 * - Requires a 2.4 kernel with UP-APIC support.
 * - Requires an Intel P6 or AMD K7 CPU.
 *
 * Copyright (C) 2001  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <signal.h>
#include <asm/sigcontext.h>
#include <asm/ucontext.h>	/* _not_ the broken <sys/ucontext.h> */
#include "linux/perfctr.h"
#define PAGE_SIZE	4096

static int fd;
static volatile const struct vperfctr_state *kstate;
static struct vperfctr_control control;
static struct perfctr_info info;

static void do_open(void)
{
    fd = open("/proc/self/perfctr", O_RDONLY|O_CREAT);
    if( fd < 0 ) {
	perror("open");
	exit(1);
    }
    if( ioctl(fd, PERFCTR_INFO, &info) != 0 ) {
	perror("perfctr_info");
	exit(1);
    }
    if( !(info.cpu_features & PERFCTR_FEATURE_PCINT) )
	printf("PCINT not supported -- expect failure\n");
    kstate = mmap(NULL, PAGE_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if( kstate == MAP_FAILED ) {
	perror("mmap");
	exit(1);
    }
}

static void on_sigio(int sig, siginfo_t *si, void *puc)
{
    struct ucontext *uc;
    struct sigcontext *mc;
    unsigned long pc;
    unsigned int pmc_mask;

    if( sig != SIGIO ) {
	printf(__FUNCTION__ ": unexpected signal %d\n", sig);
	return;
    }
    if( si->si_code != SI_PMC_OVF ) {
	printf(__FUNCTION__ ": unexpected si_code #%x\n", si->si_code);
	return;
    }
    if( (pmc_mask = si->si_pmc_ovf_mask) == 0 ) {
	printf(__FUNCTION__ ": overflow PMCs not identified\n");
	return;
    }
    uc = puc;
    mc = &uc->uc_mcontext;
    pc = mc->eip;	/* clearly more readable than glibc's mc->gregs[14] */
    if( !kstate->cpu_state.cstatus ) {
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
	printf(__FUNCTION__ ": unexpected overflow from PMC set %#x at pc %#lx (cstatus %#x)\n",
	       pmc_mask, pc, kstate->cpu_state.cstatus);
	return;
    }
    printf(__FUNCTION__ ": PMC overflow set %#x at pc %#lx\n", pmc_mask, pc);
    if( ioctl(fd, VPERFCTR_IRESUME, 0) < 0 ) {
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
    unsigned evntsel0, evntsel1;

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
      default:
	printf(__FUNCTION__ ": unsupported cpu type %u\n", info.cpu_type);
	exit(1);
    }	
    control.cpu_control.tsc_on = 1;
    control.cpu_control.nractrs = 0;
    control.cpu_control.nrictrs = 2;
    control.cpu_control.pmc_map[0] = 0;
    control.cpu_control.evntsel[0] = evntsel0;
    control.cpu_control.ireset[0] = -25; /* interrupt after 25 occurrences */
    control.cpu_control.pmc_map[1] = 1;
    control.cpu_control.evntsel[1] = evntsel1;
    control.cpu_control.ireset[1] = -25;
    control.si_signo = SIGIO;
    if( ioctl(fd, VPERFCTR_CONTROL, &control) < 0 ) {
	perror("vperfctr_control");
	exit(1);
    }
}

static void do_stop(void)
{
    struct sigaction sa;

    if( ioctl(fd, VPERFCTR_STOP, NULL) )
	perror("stop");
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
