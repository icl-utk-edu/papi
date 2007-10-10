/*
 * Performance-monitoring counters driver.
 * Optional PPC64-specific init-time tests.
 *
 * Copyright (C) 2004  David Gibson, IBM Corporation.
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
#include "ppc64_tests.h"

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
#if 0
	printk(KERN_INFO "Please email the following PERFCTR INIT lines "
	       "to mikpe@it.uu.se\n"
	       KERN_INFO "To remove this message, rebuild the driver "
	       "with CONFIG_PERFCTR_INIT_TESTS=n\n");
	printk(KERN_INFO "PERFCTR INIT: PVR 0x%08x, CPU clock %u kHz, TB clock %lu kHz\n",
	       pvr,
	       perfctr_info.cpu_khz,
	       tb_ticks_per_jiffy*(HZ/10)/(1000/10));
#endif
}

static void __init clear(void)
{
	mtspr(SPRN_MMCR0, 0);
	mtspr(SPRN_MMCR1, 0);
	mtspr(SPRN_MMCRA, 0);
	mtspr(SPRN_PMC1, 0);
	mtspr(SPRN_PMC2, 0);
	mtspr(SPRN_PMC3, 0);
	mtspr(SPRN_PMC4, 0);
	mtspr(SPRN_PMC5, 0);
	mtspr(SPRN_PMC6, 0);
	mtspr(SPRN_PMC7, 0);
	mtspr(SPRN_PMC8, 0);
}

static void __init check_fcece(unsigned int pmc1ce)
{
	unsigned int mmcr0;
	unsigned int pmc1;
	int x = 0;

	/* JHE check out section 1.6.6.2 of the POWER5 pdf */

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
	pmc1 = mfspr(SPRN_PMC1);

	mtspr(SPRN_PMC1, 0x80000000-100);
	mmcr0 = MMCR0_FCECE | MMCR0_SHRFC;

	if (pmc1ce)
		mmcr0 |= MMCR0_PMC1CE;

	mtspr(SPRN_MMCR0, mmcr0);

	pmc1 = mfspr(SPRN_PMC1);

	do {
		do_empty_loop(0);

		pmc1 = mfspr(SPRN_PMC1);
		if (x++ > 20000000) {
			break;
		}
	} while (!(mfspr(SPRN_PMC1) & 0x80000000));
	do_empty_loop(0);

	printk(KERN_INFO "PERFCTR INIT: %s(%u): MMCR0[FC] is %u, PMC1 is %#lx\n",
	       __FUNCTION__, pmc1ce,
	       !!(mfspr(SPRN_MMCR0) & MMCR0_FC), mfspr(SPRN_PMC1));
	mtspr(SPRN_MMCR0, 0);
	mtspr(SPRN_PMC1, 0);
}

static void __init check_trigger(unsigned int pmc1ce)
{
	unsigned int mmcr0;
	unsigned int pmc1;
	int x = 0;

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
	mmcr0 = MMCR0_TRIGGER | MMCR0_SHRFC | MMCR0_FCHV;

	if (pmc1ce)
		mmcr0 |= MMCR0_PMC1CE;

	mtspr(SPRN_MMCR0, mmcr0);
	do {
		do_empty_loop(0);
		pmc1 = mfspr(SPRN_PMC1);
		if (x++ > 20000000) {
			break;
		}

	} while (!(mfspr(SPRN_PMC1) & 0x80000000));
	do_empty_loop(0);
	printk(KERN_INFO "PERFCTR INIT: %s(%u): MMCR0[TRIGGER] is %u, PMC1 is %#lx, PMC2 is %#lx\n",
	       __FUNCTION__, pmc1ce,
	       !!(mfspr(SPRN_MMCR0) & MMCR0_TRIGGER), mfspr(SPRN_PMC1), mfspr(SPRN_PMC2));
	mtspr(SPRN_MMCR0, 0);
	mtspr(SPRN_PMC1, 0);
	mtspr(SPRN_PMC2, 0);
}

static void __init measure_overheads(void)
{
	int i;
	unsigned int mmcr0, loop, ticks[12];
	const char *name[12];

	clear();

	/* PMC1 = "processor cycles",
	   PMC2 = "completed instructions",
	   not disabled in any mode,
	   no interrupts */
	/* mmcr0 = (0x01 << 6) | (0x02 << 0); */
	mmcr0 = MMCR0_SHRFC | MMCR0_FCWAIT;
	mtspr(SPRN_MMCR0, mmcr0);

	name[0] = "mftbl";
	ticks[0] = run(do_read_tbl, 0);
	name[1] = "mfspr (pmc1)";
	ticks[1] = run(do_read_pmc1, 0);
	name[2] = "mfspr (pmc2)";
	ticks[2] = run(do_read_pmc2, 0);
	name[3] = "mfspr (pmc3)";
	ticks[3] = run(do_read_pmc3, 0);
	name[4] = "mfspr (pmc4)";
	ticks[4] = run(do_read_pmc4, 0);
	name[5] = "mfspr (mmcr0)";
	ticks[5] = run(do_read_mmcr0, 0);
	name[6] = "mfspr (mmcr1)";
	ticks[6] = run(do_read_mmcr1, 0);
	name[7] = "mtspr (pmc2)";
	ticks[7] = run(do_write_pmc2, 0);
	name[8] = "mtspr (pmc3)";
	ticks[8] = run(do_write_pmc3, 0);
	name[9] = "mtspr (pmc4)";
	ticks[9] = run(do_write_pmc4, 0);
	name[10] = "mtspr (mmcr1)";
	ticks[10] = run(do_write_mmcr1, 0);
	name[11] = "mtspr (mmcr0)";
	ticks[11] = run(do_write_mmcr0, mmcr0);

	loop = run(do_empty_loop, 0);

	clear();

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
#if 0
	check_fcece(1);
	check_trigger(0);
	check_trigger(1);
#endif
}

void __init perfctr_ppc64_init_tests(void)
{
	preempt_disable();
	measure_overheads();
	preempt_enable();
}
