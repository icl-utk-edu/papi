/* $Id: ppc_tests.c,v 1.8 2007/10/06 13:02:07 mikpe Exp $
 * Performance-monitoring counters driver.
 * Optional PPC32-specific init-time tests.
 *
 * Copyright (C) 2004, 2007  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/perfctr.h>
#include <asm/processor.h>
#include <asm/time.h>	/* for tb_ticks_per_jiffy */
#include "ppc_tests.h"

#define NITER	256
#define X2(S)	S"; "S
#define X8(S)	X2(X2(X2(S)))

static void __init do_read_tbl(unsigned int unused)
{
	unsigned int i, dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mftbl %0") : "=r"(dummy));
}

static void __init do_read_pmc1(unsigned int unused)
{
	unsigned int i, dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mfspr %0," __stringify(SPRN_PMC1)) : "=r"(dummy));
}

static void __init do_read_pmc2(unsigned int unused)
{
	unsigned int i, dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mfspr %0," __stringify(SPRN_PMC2)) : "=r"(dummy));
}

static void __init do_read_pmc3(unsigned int unused)
{
	unsigned int i, dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mfspr %0," __stringify(SPRN_PMC3)) : "=r"(dummy));
}

static void __init do_read_pmc4(unsigned int unused)
{
	unsigned int i, dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mfspr %0," __stringify(SPRN_PMC4)) : "=r"(dummy));
}

static void __init do_read_mmcr0(unsigned int unused)
{
	unsigned int i, dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mfspr %0," __stringify(SPRN_MMCR0)) : "=r"(dummy));
}

static void __init do_read_mmcr1(unsigned int unused)
{
	unsigned int i, dummy;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mfspr %0," __stringify(SPRN_MMCR1)) : "=r"(dummy));
}

static void __init do_write_pmc2(unsigned int arg)
{
	unsigned int i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mtspr " __stringify(SPRN_PMC2) ",%0") : : "r"(arg));
}

static void __init do_write_pmc3(unsigned int arg)
{
	unsigned int i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mtspr " __stringify(SPRN_PMC3) ",%0") : : "r"(arg));
}

static void __init do_write_pmc4(unsigned int arg)
{
	unsigned int i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mtspr " __stringify(SPRN_PMC4) ",%0") : : "r"(arg));
}

static void __init do_write_mmcr1(unsigned int arg)
{
	unsigned int i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mtspr " __stringify(SPRN_MMCR1) ",%0") : : "r"(arg));
}

static void __init do_write_mmcr0(unsigned int arg)
{
	unsigned int i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("mtspr " __stringify(SPRN_MMCR0) ",%0") : : "r"(arg));
}

static void __init do_empty_loop(unsigned int unused)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__("" : : );
}

static unsigned __init run(void (*doit)(unsigned int), unsigned int arg)
{
	unsigned int start, stop;
	start = mfspr(SPRN_PMC1);
	(*doit)(arg);	/* should take < 2^32 cycles to complete */
	stop = mfspr(SPRN_PMC1);
	return stop - start;
}

static void __init init_tests_message(void)
{
	unsigned int pvr = mfspr(SPRN_PVR);
	printk(KERN_INFO "Please email the following PERFCTR INIT lines "
	       "to mikpe@it.uu.se\n"
	       KERN_INFO "To remove this message, rebuild the driver "
	       "with CONFIG_PERFCTR_INIT_TESTS=n\n");
	printk(KERN_INFO "PERFCTR INIT: PVR 0x%08x, CPU clock %u kHz, TB clock %u kHz\n",
	       pvr,
	       perfctr_info.cpu_khz,
	       tb_ticks_per_jiffy*(HZ/10)/(1000/10));
}

static void __init clear(int have_mmcr1)
{
	mtspr(SPRN_MMCR0, 0);
	mtspr(SPRN_PMC1, 0);
	mtspr(SPRN_PMC2, 0);
	if (have_mmcr1) {
		mtspr(SPRN_MMCR1, 0);
		mtspr(SPRN_PMC3, 0);
		mtspr(SPRN_PMC4, 0);
	}
}

static void __init check_fcece(unsigned int pmc1ce)
{
	unsigned int mmcr0;

	/*
	 * This test checks if MMCR0[FC] is set after PMC1 overflows
	 * when MMCR0[FCECE] is set.
	 * 74xx documentation states this behaviour, while documentation
	 * for 604/750 processors doesn't mention this at all.
	 *
	 * Also output the value of PMC1 shortly after the overflow.
	 * This tells us if PMC1 really was frozen. On 604/750, it may not
	 * freeze since we don't enable PMIs. [No freeze confirmed on 750.]
	 *
	 * When pmc1ce == 0, MMCR0[PMC1CE] is zero. It's unclear whether
	 * this masks all PMC1 overflow events or just PMC1 PMIs.
	 *
	 * PMC1 counts processor cycles, with 100 to go before overflowing.
	 * FCECE is set.
	 * PMC1CE is clear if !pmc1ce, otherwise set.
	 */
	mtspr(SPRN_PMC1, 0x80000000-100);
	mmcr0 = (1<<(31-6)) | (0x01 << 6);
	if (pmc1ce)
		mmcr0 |= (1<<(31-16));
	mtspr(SPRN_MMCR0, mmcr0);
	do {
		do_empty_loop(0);
	} while (!(mfspr(SPRN_PMC1) & 0x80000000));
	do_empty_loop(0);
	printk(KERN_INFO "PERFCTR INIT: %s(%u): MMCR0[FC] is %u, PMC1 is %#x\n",
	       __FUNCTION__, pmc1ce,
	       !!(mfspr(SPRN_MMCR0) & (1<<(31-0))), mfspr(SPRN_PMC1));
	mtspr(SPRN_MMCR0, 0);
	mtspr(SPRN_PMC1, 0);
}

static void __init check_trigger(unsigned int pmc1ce)
{
	unsigned int mmcr0;

	/*
	 * This test checks if MMCR0[TRIGGER] is reset after PMC1 overflows.
	 * 74xx documentation states this behaviour, while documentation
	 * for 604/750 processors doesn't mention this at all.
	 * [No reset confirmed on 750.]
	 *
	 * Also output the values of PMC1 and PMC2 shortly after the overflow.
	 * PMC2 should be equal to PMC1-0x80000000.
	 *
	 * When pmc1ce == 0, MMCR0[PMC1CE] is zero. It's unclear whether
	 * this masks all PMC1 overflow events or just PMC1 PMIs.
	 *
	 * PMC1 counts processor cycles, with 100 to go before overflowing.
	 * PMC2 counts processor cycles, starting from 0.
	 * TRIGGER is set, so PMC2 doesn't start until PMC1 overflows.
	 * PMC1CE is clear if !pmc1ce, otherwise set.
	 */
	mtspr(SPRN_PMC2, 0);
	mtspr(SPRN_PMC1, 0x80000000-100);
	mmcr0 = (1<<(31-18)) | (0x01 << 6) | (0x01 << 0);
	if (pmc1ce)
		mmcr0 |= (1<<(31-16));
	mtspr(SPRN_MMCR0, mmcr0);
	do {
		do_empty_loop(0);
	} while (!(mfspr(SPRN_PMC1) & 0x80000000));
	do_empty_loop(0);
	printk(KERN_INFO "PERFCTR INIT: %s(%u): MMCR0[TRIGGER] is %u, PMC1 is %#x, PMC2 is %#x\n",
	       __FUNCTION__, pmc1ce,
	       !!(mfspr(SPRN_MMCR0) & (1<<(31-18))), mfspr(SPRN_PMC1), mfspr(SPRN_PMC2));
	mtspr(SPRN_MMCR0, 0);
	mtspr(SPRN_PMC1, 0);
	mtspr(SPRN_PMC2, 0);
}

static void __init
measure_overheads(int have_mmcr1)
{
	int i;
	unsigned int mmcr0, loop, ticks[12];
	const char *name[12];

	clear(have_mmcr1);

	/* PMC1 = "processor cycles",
	   PMC2 = "completed instructions",
	   not disabled in any mode,
	   no interrupts */
	mmcr0 = (0x01 << 6) | (0x02 << 0);
	mtspr(SPRN_MMCR0, mmcr0);

	name[0] = "mftbl";
	ticks[0] = run(do_read_tbl, 0);
	name[1] = "mfspr (pmc1)";
	ticks[1] = run(do_read_pmc1, 0);
	name[2] = "mfspr (pmc2)";
	ticks[2] = run(do_read_pmc2, 0);
	name[3] = "mfspr (pmc3)";
	ticks[3] = have_mmcr1 ? run(do_read_pmc3, 0) : 0;
	name[4] = "mfspr (pmc4)";
	ticks[4] = have_mmcr1 ? run(do_read_pmc4, 0) : 0;
	name[5] = "mfspr (mmcr0)";
	ticks[5] = run(do_read_mmcr0, 0);
	name[6] = "mfspr (mmcr1)";
	ticks[6] = have_mmcr1 ? run(do_read_mmcr1, 0) : 0;
	name[7] = "mtspr (pmc2)";
	ticks[7] = run(do_write_pmc2, 0);
	name[8] = "mtspr (pmc3)";
	ticks[8] = have_mmcr1 ? run(do_write_pmc3, 0) : 0;
	name[9] = "mtspr (pmc4)";
	ticks[9] = have_mmcr1 ? run(do_write_pmc4, 0) : 0;
	name[10] = "mtspr (mmcr1)";
	ticks[10] = have_mmcr1 ? run(do_write_mmcr1, 0) : 0;
	name[11] = "mtspr (mmcr0)";
	ticks[11] = run(do_write_mmcr0, mmcr0);

	loop = run(do_empty_loop, 0);

	clear(have_mmcr1);

	init_tests_message();
	printk(KERN_INFO "PERFCTR INIT: NITER == %u\n", NITER);
	printk(KERN_INFO "PERFCTR INIT: loop overhead is %u cycles\n", loop);
	for(i = 0; i < ARRAY_SIZE(ticks); ++i) {
		unsigned int x;
		if (!ticks[i])
			continue;
		x = ((ticks[i] - loop) * 10) / NITER;
		printk(KERN_INFO "PERFCTR INIT: %s cost is %u.%u cycles (%u total)\n",
		       name[i], x/10, x%10, ticks[i]);
	}
	check_fcece(0);
	check_fcece(1);
	check_trigger(0);
	check_trigger(1);
}

void __init perfctr_ppc_init_tests(int have_mmcr1)
{
	preempt_disable();
	measure_overheads(have_mmcr1);
	preempt_enable();
}
