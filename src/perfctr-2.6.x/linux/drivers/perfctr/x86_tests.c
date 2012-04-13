/* $Id: x86_tests.c,v 1.23.2.14 2007/10/07 17:18:32 mikpe Exp $
 * Performance-monitoring counters driver.
 * Optional x86/x86_64-specific init-time tests.
 *
 * Copyright (C) 1999-2007  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/perfctr.h>
#include <asm/msr.h>
#undef MSR_P6_PERFCTR0
#undef MSR_P6_EVNTSEL0
#undef MSR_K7_PERFCTR0
#undef MSR_K7_EVNTSEL0
#undef MSR_CORE_PERF_FIXED_CTR_CTRL
#undef MSR_P4_IQ_CCCR0
#undef MSR_P4_CRU_ESCR0
#include <asm/fixmap.h>
#include <asm/apic.h>
#include "x86_compat.h"
#include "x86_tests.h"

#define MSR_P5_CESR		0x11
#define MSR_P5_CTR0		0x12
#define P5_CESR_VAL		(0x16 | (3<<6))
#define MSR_P6_PERFCTR0		0xC1
#define MSR_P6_EVNTSEL0		0x186
#define P6_EVNTSEL0_VAL		(0xC0 | (3<<16) | (1<<22))
#define MSR_K7_EVNTSEL0		0xC0010000
#define MSR_K7_PERFCTR0		0xC0010004
#define K7_EVNTSEL0_VAL		(0xC0 | (3<<16) | (1<<22))
#define VC3_EVNTSEL1_VAL	0xC0
#define MSR_CORE_PERF_FIXED_CTR_CTRL	0x38D
#define CORE2_PMC_FIXED_CTR0	((1<<30) | 0)
#define MSR_P4_IQ_COUNTER0	0x30C
#define MSR_P4_IQ_CCCR0		0x36C
#define MSR_P4_CRU_ESCR0	0x3B8
#define P4_CRU_ESCR0_VAL	((2<<25) | (1<<9) | (0x3<<2))
#define P4_IQ_CCCR0_VAL		((0x3<<16) | (4<<13) | (1<<12))

#define NITER	64
#define X2(S)	S";"S
#define X8(S)	X2(X2(X2(S)))

#ifdef __x86_64__
#define CR4MOV	"movq"
#else
#define CR4MOV	"movl"
#endif

#ifndef CONFIG_X86_LOCAL_APIC
#undef apic_write
#define apic_write(reg,vector)			do{}while(0)
#endif

#define rdtsc_low(low) \
	__asm__ __volatile__("rdtsc" : "=a"(low) : : "edx")

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
		__asm__ __volatile__(X8(CR4MOV" %%cr4,%0") : "=r"(dummy));
}

static void __init do_wrcr4(unsigned cr4, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8(CR4MOV" %0,%%cr4") : : "r"((long)cr4));
}

static void __init do_rdtsc(unsigned unused1, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i)
		__asm__ __volatile__(X8("rdtsc") : : : "eax", "edx");
}

static void __init do_wrlvtpc(unsigned val, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i) {
		apic_write(APIC_LVTPC, val);
		apic_write(APIC_LVTPC, val);
		apic_write(APIC_LVTPC, val);
		apic_write(APIC_LVTPC, val);
		apic_write(APIC_LVTPC, val);
		apic_write(APIC_LVTPC, val);
		apic_write(APIC_LVTPC, val);
		apic_write(APIC_LVTPC, val);
	}
}

static void __init do_sync_core(unsigned unused1, unsigned unused2)
{
	unsigned i;
	for(i = 0; i < NITER/8; ++i) {
		sync_core();
		sync_core();
		sync_core();
		sync_core();
		sync_core();
		sync_core();
		sync_core();
		sync_core();
	}
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
	unsigned start, stop;
	sync_core();
	rdtsc_low(start);
	(*doit)(arg1, arg2);	/* should take < 2^32 cycles to complete */
	sync_core();
	rdtsc_low(stop);
	return stop - start;
}

static void __init init_tests_message(void)
{
	printk(KERN_INFO "Please email the following PERFCTR INIT lines "
	       "to mikpe@it.uu.se\n"
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
measure_overheads(unsigned msr_evntsel0, unsigned evntsel0, unsigned msr_perfctr0,
		  unsigned msr_cccr, unsigned cccr_val, unsigned is_core2)
{
	int i;
	unsigned int loop, ticks[15];
	const char *name[15];

	if (msr_evntsel0)
		wrmsr(msr_evntsel0, 0, 0);
	if (msr_cccr)
		wrmsr(msr_cccr, 0, 0);

	name[0] = "rdtsc";
	ticks[0] = run(do_rdtsc, 0, 0);
	name[1] = "rdpmc";
	ticks[1] = (perfctr_info.cpu_features & PERFCTR_FEATURE_RDPMC)
		? run(do_rdpmc,1,0) : 0;
	name[2] = "rdmsr (counter)";
	ticks[2] = msr_perfctr0 ? run(do_rdmsr, msr_perfctr0, 0) : 0;
	name[3] = msr_cccr ? "rdmsr (escr)" : "rdmsr (evntsel)";
	ticks[3] = msr_evntsel0 ? run(do_rdmsr, msr_evntsel0, 0) : 0;
	name[4] = "wrmsr (counter)";
	ticks[4] = msr_perfctr0 ? run(do_wrmsr, msr_perfctr0, 0) : 0;
	name[5] = msr_cccr ? "wrmsr (escr)" : "wrmsr (evntsel)";
	ticks[5] = msr_evntsel0 ? run(do_wrmsr, msr_evntsel0, evntsel0) : 0;
	name[6] = "read cr4";
	ticks[6] = run(do_rdcr4, 0, 0);
	name[7] = "write cr4";
	ticks[7] = run(do_wrcr4, read_cr4(), 0);
	name[8] = "rdpmc (fast)";
	ticks[8] = msr_cccr ? run(do_rdpmc, 0x80000001, 0) : 0;
	name[9] = "rdmsr (cccr)";
	ticks[9] = msr_cccr ? run(do_rdmsr, msr_cccr, 0) : 0;
	name[10] = "wrmsr (cccr)";
	ticks[10] = msr_cccr ? run(do_wrmsr, msr_cccr, cccr_val) : 0;
	name[11] = "write LVTPC";
	ticks[11] = (perfctr_info.cpu_features & PERFCTR_FEATURE_PCINT)
		? run(do_wrlvtpc, APIC_DM_NMI|APIC_LVT_MASKED, 0) : 0;
	name[12] = "sync_core";
	ticks[12] = run(do_sync_core, 0, 0);
	name[13] = "read fixed_ctr0";
	ticks[13] = is_core2 ? run(do_rdpmc, CORE2_PMC_FIXED_CTR0, 0) : 0;
	name[14] = "wrmsr fixed_ctr_ctrl";
	ticks[14] = is_core2 ? run(do_wrmsr, MSR_CORE_PERF_FIXED_CTR_CTRL, 0) : 0;

	loop = run(do_empty_loop, 0, 0);

	if (msr_evntsel0)
		wrmsr(msr_evntsel0, 0, 0);
	if (msr_cccr)
		wrmsr(msr_cccr, 0, 0);

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
}

#ifndef __x86_64__
static inline void perfctr_p5_init_tests(void)
{
	measure_overheads(MSR_P5_CESR, P5_CESR_VAL, MSR_P5_CTR0, 0, 0, 0);
}

#if !defined(CONFIG_X86_TSC)
static inline void perfctr_c6_init_tests(void)
{
	unsigned int cesr, dummy;

	rdmsr(MSR_P5_CESR, cesr, dummy);
	init_tests_message();
	printk(KERN_INFO "PERFCTR INIT: boot CESR == %#08x\n", cesr);
}
#endif

static inline void perfctr_vc3_init_tests(void)
{
	measure_overheads(MSR_P6_EVNTSEL0+1, VC3_EVNTSEL1_VAL, MSR_P6_PERFCTR0+1, 0, 0, 0);
}
#endif /* !__x86_64__ */

static inline void perfctr_p6_init_tests(void)
{
	measure_overheads(MSR_P6_EVNTSEL0, P6_EVNTSEL0_VAL, MSR_P6_PERFCTR0, 0, 0, 0);
}

static inline void perfctr_core2_init_tests(void)
{
	measure_overheads(MSR_P6_EVNTSEL0, P6_EVNTSEL0_VAL, MSR_P6_PERFCTR0, 0, 0, 1);
}

static inline void perfctr_p4_init_tests(void)
{
	measure_overheads(MSR_P4_CRU_ESCR0, P4_CRU_ESCR0_VAL, MSR_P4_IQ_COUNTER0,
			  MSR_P4_IQ_CCCR0, P4_IQ_CCCR0_VAL, 0);
}

static inline void perfctr_k7_init_tests(void)
{
	measure_overheads(MSR_K7_EVNTSEL0, K7_EVNTSEL0_VAL, MSR_K7_PERFCTR0, 0, 0, 0);
}

static inline void perfctr_generic_init_tests(void)
{
	measure_overheads(0, 0, 0, 0, 0, 0);
}

enum perfctr_x86_tests_type perfctr_x86_tests_type __initdata = PTT_UNKNOWN;

void __init perfctr_x86_init_tests(void)
{
	switch (perfctr_x86_tests_type) {
#ifndef __x86_64__
	case PTT_P5: /* Intel P5, P5MMX; Cyrix 6x86MX, MII, III */
		perfctr_p5_init_tests();
		break;
#if !defined(CONFIG_X86_TSC)
	case PTT_WINCHIP: /* WinChip C6, 2, 3 */
		perfctr_c6_init_tests();
		break;
#endif
	case PTT_VC3: /* VIA C3 */
		perfctr_vc3_init_tests();
		break;
#endif /* !__x86_64__ */
	case PTT_P6: /* Intel PPro, PII, PIII, PENTM, CORE */
		perfctr_p6_init_tests();
		break;
	case PTT_CORE2: /* Intel Core 2 */
		perfctr_core2_init_tests();
		break;
	case PTT_P4: /* Intel P4 */
		perfctr_p4_init_tests();
		break;
	case PTT_AMD: /* AMD K7, K8 */
		perfctr_k7_init_tests();
		break;
	case PTT_GENERIC:
		perfctr_generic_init_tests();
		break;
	default:
		printk(KERN_INFO "%s: unknown CPU type %u\n",
		       __FUNCTION__, perfctr_x86_tests_type);
		break;
	}
}
