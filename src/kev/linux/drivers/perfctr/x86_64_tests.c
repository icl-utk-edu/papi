/* $Id$
 * Performance-monitoring counters driver.
 * Optional x86_64-specific init-time tests.
 *
 * Copyright (C) 2003  Mikael Pettersson
 */
#include <linux/config.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/perfctr.h>
#include <asm/msr.h>
#include "x86_64_compat.h"
#include "x86_64_tests.h"

#define MSR_K8_EVNTSEL0		0xC0010000
#define MSR_K8_PERFCTR0		0xC0010004
#define K8_EVNTSEL0_VAL		(0xC0 | (3<<16) | (1<<22))

#define NITER	64
#define X2(S)	S";"S
#define X8(S)	X2(X2(X2(S)))

static void __init do_rdpmc(unsigned pmc, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("rdpmc") : : "c"(pmc) : "eax", "edx");
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
	unsigned long dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("movq %%cr4,%0") : "=r"(dummy));
}

static void __init do_wrcr4(unsigned cr4, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("movq %0,%%cr4") : : "r"((long)cr4));
}

static void __init do_rdtsc(unsigned unused1, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("rdtsc") : : : "eax", "edx");
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
	       KERN_INFO "To remove this message, rebuild the driver "
	       "with CONFIG_PERFCTR_INIT_TESTS=n\n");
	printk(KERN_INFO "PERFCTR INIT: vendor %u, family %u, model %u, stepping %u, clock %u kHz\n",
	       current_cpu_data.x86_vendor,
	       current_cpu_data.x86,
	       current_cpu_data.x86_model,
	       current_cpu_data.x86_mask,
	       perfctr_cpu_khz());
}

static void __init
measure_overheads(unsigned msr_evntsel0, unsigned evntsel0, unsigned msr_perfctr0)
{
	int i;
	unsigned int loop, ticks[8];
	const char *name[8];

	if( msr_evntsel0 )
		wrmsr(msr_evntsel0, 0, 0);

	name[0] = "rdtsc";
	ticks[0] = run(do_rdtsc, 0, 0);
	name[1] = "rdpmc";
	ticks[1] = (perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC)
		? run(do_rdpmc,1,0) : 0;
	name[2] = "rdmsr (counter)";
	ticks[2] = msr_perfctr0 ? run(do_rdmsr, msr_perfctr0, 0) : 0;
	name[3] = "rdmsr (evntsel)";
	ticks[3] = msr_evntsel0 ? run(do_rdmsr, msr_evntsel0, 0) : 0;
	name[4] = "wrmsr (counter)";
	ticks[4] = msr_perfctr0 ? run(do_wrmsr, msr_perfctr0, 0) : 0;
	name[5] = "wrmsr (evntsel)";
	ticks[5] = msr_evntsel0 ? run(do_wrmsr, msr_evntsel0, evntsel0) : 0;
	name[6] = "read cr4";
	ticks[6] = run(do_rdcr4, 0, 0);
	name[7] = "write cr4";
	ticks[7] = run(do_wrcr4, read_cr4(), 0);

	loop = run(do_empty_loop, 0, 0);

	if( msr_evntsel0 )
		wrmsr(msr_evntsel0, 0, 0);

	init_tests_message();
	printk(KERN_INFO "PERFCTR INIT: NITER == %u\n", NITER);
	printk(KERN_INFO "PERFCTR INIT: loop overhead is %u cycles\n", loop);
	for(i = 0; i < ARRAY_SIZE(ticks); ++i) {
		unsigned int x;
		if( !ticks[i] )
			continue;
		x = ((ticks[i] - loop) * 10) / NITER;
		printk(KERN_INFO "PERFCTR INIT: %s cost is %u.%u cycles (%u total)\n",
		       name[i], x/10, x%10, ticks[i]);
	}
}

void __init perfctr_k8_init_tests(void)
{
	measure_overheads(MSR_K8_EVNTSEL0, K8_EVNTSEL0_VAL, MSR_K8_PERFCTR0);
}

void __init perfctr_generic_init_tests(void)
{
	measure_overheads(0, 0, 0);
}
