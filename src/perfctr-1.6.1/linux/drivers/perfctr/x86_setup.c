/* $Id$
 * Performance-monitoring counters driver.
 * x86-specific kernel-resident code.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include <linux/config.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <asm/processor.h>
#include <asm/perfctr.h>
#include "compat.h"
#include "x86_compat.h"

void __init perfctr_dodgy_tsc(void)
{
#if defined(CONFIG_PERFCTR_WINCHIP)
	/* Check for Centaur WinChip C6/2/2A/3 with broken TSC */
	if( boot_cpu_data.x86_vendor != X86_VENDOR_CENTAUR )
		return;
	if( boot_cpu_data.x86 != 5 )
		return;
	switch( boot_cpu_data.x86_model ) {
	case 4:	break;		/* WinChip C6 */
	case 8:	break;		/* WinChip 2 or 2A */
	case 9: break;		/* WinChip 3 */
	default: return;
	}
	/*
	 * WinChip processors implement the TSC by using the
	 * low 32 bits of the two performance counters.
	 * If the performance counters are reprogrammed, the
	 * RDTSC instruction will yield incorrect results.
	 * (On the C6, a read of the TSC using RDTSC or RDMSR
	 * may even cause the processor to hang in this case.
	 * See erratum I-13 in the C6 data sheet for details.)
	 *
	 * Since the user explicitly requested the performance
	 * counters, we have to disable all uses of the TSC.
	 */
	set_cr4_tsd_local();
	boot_cpu_data.x86_capability &= ~X86_FEATURE_TSC;
	printk(KERN_INFO "WinChip Time-Stamp Counter disabled\n");
#endif
}

#ifdef CONFIG_PERFCTR_MODULE
MAYBE_EXPORT_mmu_cr4_features;
EXPORT_cpu_khz;
#endif
