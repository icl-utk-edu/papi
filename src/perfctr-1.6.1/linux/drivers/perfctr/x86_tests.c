/* $Id$
 * Performance-monitoring counters driver.
 * Optional x86-specific init-time tests.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/perfctr.h>
#include <asm/msr.h>
#include "compat.h"
#include "x86_tests.h"

#define MSR_P5_CESR		0x11
#define MSR_P5_CTR0		0x12
#define MSR_P6_PERFCTR0		0xC1
#define MSR_P6_EVNTSEL0		0x186
#define MSR_K7_EVNTSEL0		0xC0010000
#define MSR_K7_PERFCTR0		0xC0010004

static __inline__ unsigned int get_cr4(void)
{
	unsigned int cr4;
	__asm__ __volatile__("movl %%cr4,%0" : "=r"(cr4));
	return cr4;
}

#define NITER	64
#define X2(S)	S";"S
#define X8(S)	X2(X2(X2(S)))

static void __init do_rdpmc(unsigned unused1, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("rdpmc") : : "c"(0) : "eax", "edx");
}

static void __init do_rdmsr(unsigned msr, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("rdmsr") : : "c"(msr) : "eax", "edx");
}

static void __init do_wrmsr(unsigned msr, unsigned data)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("wrmsr") : : "c"(msr), "a"(data), "d"(0));
}

static void __init do_rdcr4(unsigned unused1, unsigned unused2)
{
	unsigned i;
	unsigned dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("movl %%cr4,%0") : "=r"(dummy));
}

static void __init do_wrcr4(unsigned cr4, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("movl %0,%%cr4") : : "r"(cr4));
}

static void __init do_empty_loop(unsigned unused1, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__("" : : "c"(0));
}

static unsigned __init run(void (*doit)(unsigned, unsigned),
			   unsigned arg1, unsigned arg2)
{
	unsigned start, dummy, stop;
	rdtsc(start, dummy);
	(*doit)(arg1, arg2);	/* should take < 2^32 cycles to complete */
	rdtsc(stop, dummy);
	return stop - start;
}

static void __init init_tests_message(void)
{
	printk(KERN_INFO "Please email the following PERFCTR INIT lines "
	       "to mikpe@csd.uu.se\n"
	       KERN_INFO "To remove this message, rebuild the kernel "
	       "with CONFIG_PERFCTR_INIT_TESTS=n\n");
	printk(KERN_INFO "PERFCTR INIT: vendor %u, family %u, model %u\n",
	       boot_cpu_data.x86_vendor,
	       boot_cpu_data.x86,
	       boot_cpu_data.x86_model);
}

static void __init
measure_overheads(unsigned msr_evntsel0, unsigned evntsel0, unsigned msr_perfctr0)
{
	int i;
	unsigned int ticks[8];
	char *name[8];

	name[0] = "rdpmc";
	ticks[0] = (perfctr_cpu_features & PERFCTR_FEATURE_RDPMC)
		? run(do_rdpmc,0,0)
		: 0;
	name[1] = "rdmsr (counter)";
	ticks[1] = run(do_rdmsr, msr_perfctr0, 0);
	name[2] = "rdmsr (evntsel)";
	ticks[2] = run(do_rdmsr, msr_evntsel0, 0);
	name[3] = "wrmsr (counter)";
	ticks[3] = run(do_wrmsr, msr_perfctr0, 0);
	name[4] = "wrmsr (evntsel)";
	ticks[4] = run(do_wrmsr, msr_evntsel0, evntsel0);
	name[5] = "read %cr4";
	ticks[5] = run(do_rdcr4, 0, 0);
	name[6] = "write %cr4";
	ticks[6] = run(do_wrcr4, get_cr4(), 0);
	name[7] = "loop overhead";
	ticks[7] = run(do_empty_loop, 0, 0);
	wrmsr(msr_evntsel0, 0, 0);

	init_tests_message();
	printk(KERN_INFO "PERFCTR INIT: NITER == %u\n", NITER);
	for(i = 0; i < ARRAY_SIZE(ticks); ++i)
		printk(KERN_INFO "PERFCTR INIT: %s ticks == %u\n",
		       name[i], ticks[i]);
}

void __init perfctr_p5_init_tests(void)
{
	unsigned evnt = 0x16 | (3 << 6);
	measure_overheads(MSR_P5_CESR, evnt, MSR_P5_CTR0);
}

void __init perfctr_p6_init_tests(void)
{
	unsigned evnt = 0xC0 | (3 << 16) | (1 << 22);
	measure_overheads(MSR_P6_EVNTSEL0, evnt, MSR_P6_PERFCTR0);
}

static void __init k7_stop_and_clear(void)
{
	wrmsr(MSR_K7_EVNTSEL0+0, 0, 0);
	wrmsr(MSR_K7_EVNTSEL0+1, 0, 0);
	wrmsr(MSR_K7_PERFCTR0+0, 0, 0);
	wrmsr(MSR_K7_PERFCTR0+1, 0, 0);
}

static unsigned __init k7_test(unsigned ctr0_on, unsigned ctr1_on)
{
	unsigned ctr1_evnt = 0xC0 | (3 << 16);	/* num insns */
	unsigned ctr0_evnt = 0xDF | (3 << 16);	/* dr3 match */
	const unsigned ENable = (1<<22);
	unsigned dummy, ctr1_result;

	if( ctr1_on ) ctr1_evnt |= ENable;
	if( ctr0_on ) ctr0_evnt |= ENable;
	wrmsr(MSR_K7_EVNTSEL0+1, ctr1_evnt, 0);
	wrmsr(MSR_K7_EVNTSEL0+0, ctr0_evnt, 0);
	do_empty_loop(0, 0);
	rdmsr(MSR_K7_PERFCTR0+1, ctr1_result, dummy);
	k7_stop_and_clear();
	return ctr1_result;
}

static void __init k7_check_how_ENable_works(void)
{
	unsigned test0, test1, test2, test3;

	k7_stop_and_clear();

	test0 = k7_test(0, 0); /* should be == 0 */
	test1 = k7_test(1, 1); /* should be > 0 */
	test2 = k7_test(0, 1); /* if > 0, EvntSel0 does not override */
	test3 = k7_test(1, 0); /* if > 0, only EvntSel0 has ENable */

	printk(KERN_INFO "PERFCTR INIT: Athlon test0 == %u (%s)\n",
	       test0, test0 == 0 ? "ok" : "unexpected");
	printk(KERN_INFO "PERFCTR INIT: Athlon test1 == %u (%s)\n",
	       test1, test1 > 0 ? "ok" : "unexpected");
	printk(KERN_INFO "PERFCTR INIT: Athlon test2 == %u (%s)\n",
	       test2, test2 > 0 ? "EvntSel0 does not override" : "ok");
	printk(KERN_INFO "PERFCTR INIT: Athlon test3 == %u (%s)\n",
	       test3, test3 > 0 ? "only EvntSel0 has ENable" : "ok");
}

void __init perfctr_k7_init_tests(void)
{
	unsigned evnt = 0xC0 | (3 << 16) | (1 << 22);
	measure_overheads(MSR_K7_EVNTSEL0, evnt, MSR_K7_PERFCTR0);
	k7_check_how_ENable_works();
}

#if defined(CONFIG_PERFCTR_WINCHIP)
void __init perfctr_c6_init_tests(void)
{
	unsigned int cesr, dummy;

	rdmsr(MSR_P5_CESR, cesr, dummy);
	init_tests_message();
	printk(KERN_INFO "PERFCTR INIT: boot CESR == %#08x\n", cesr);
}
#endif
