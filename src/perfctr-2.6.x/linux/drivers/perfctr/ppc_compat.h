/* $Id: ppc_compat.h,v 1.1.2.8 2009/01/23 17:01:02 mikpe Exp $
 * Performance-monitoring counters driver.
 * PPC32-specific compatibility definitions for 2.6 kernels.
 *
 * Copyright (C) 2004-2007, 2009  Mikael Pettersson
 */
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,16)
#include <asm/pmc.h>
#else
static inline int reserve_pmc_hardware(void (*new_perf_irq)(struct pt_regs*)) { return 0; }
static inline void release_pmc_hardware(void) { }
#endif
extern int perfctr_reserve_pmc_hardware(void);
static inline void perfctr_release_pmc_hardware(void) { release_pmc_hardware(); }

#undef MMCR0_FC
#undef MMCR0_FCECE
#undef MMCR0_FCM0
#undef MMCR0_FCM1
#undef MMCR0_FCP
#undef MMCR0_FCS
#undef MMCR0_PMC1CE
#undef MMCR0_PMC1SEL
#undef MMCR0_PMC2SEL
#undef MMCR0_PMXE
#undef MMCR0_TBEE
#undef MMCR0_TRIGGER
#undef MMCR1_PMC3SEL
#undef MMCR1_PMC4SEL
#undef MMCR1_PMC5SEL
#undef MMCR1_PMC6SEL
#undef SPRN_MMCR0
#undef SPRN_MMCR1
#undef SPRN_MMCR2
#undef SPRN_PMC1
#undef SPRN_PMC2
#undef SPRN_PMC3
#undef SPRN_PMC4
#undef SPRN_PMC5
#undef SPRN_PMC6

#define SPRN_MMCR0	0x3B8	/* 604 and up */
#define SPRN_PMC1	0x3B9	/* 604 and up */
#define SPRN_PMC2	0x3BA	/* 604 and up */
#define SPRN_SIA	0x3BB	/* 604 and up */
#define SPRN_MMCR1	0x3BC	/* 604e and up */
#define SPRN_PMC3	0x3BD	/* 604e and up */
#define SPRN_PMC4	0x3BE	/* 604e and up */
#define SPRN_MMCR2	0x3B0	/* 7400 and up */
#define SPRN_BAMR	0x3B7	/* 7400 and up */
#define SPRN_PMC5	0x3B1	/* 7450 and up */
#define SPRN_PMC6	0x3B2	/* 7450 and up */

/* MMCR0 layout (74xx terminology) */
#define MMCR0_FC		0x80000000 /* Freeze counters unconditionally. */
#define MMCR0_FCS		0x40000000 /* Freeze counters while MSR[PR]=0 (supervisor mode). */
#define MMCR0_FCP		0x20000000 /* Freeze counters while MSR[PR]=1 (user mode). */
#define MMCR0_FCM1		0x10000000 /* Freeze counters while MSR[PM]=1. */
#define MMCR0_FCM0		0x08000000 /* Freeze counters while MSR[PM]=0. */
#define MMCR0_PMXE		0x04000000 /* Enable performance monitor exceptions.
					    * Cleared by hardware when a PM exception occurs.
					    * 604: PMXE is not cleared by hardware.
					    */
#define MMCR0_FCECE		0x02000000 /* Freeze counters on enabled condition or event.
					    * FCECE is treated as 0 if TRIGGER is 1.
					    * 74xx: FC is set when the event occurs.
					    * 604/750: ineffective when PMXE=0.
					    */
#define MMCR0_TBSEL		0x01800000 /* Time base lower (TBL) bit selector.
					    * 00: bit 31, 01: bit 23, 10: bit 19, 11: bit 15.
					    */
#define MMCR0_TBEE		0x00400000 /* Enable event on TBL bit transition from 0 to 1. */
#define MMCR0_THRESHOLD		0x003F0000 /* Threshold value for certain events. */
#define MMCR0_PMC1CE		0x00008000 /* Enable event on PMC1 overflow. */
#define MMCR0_PMCjCE		0x00004000 /* Enable event on PMC2-PMC6 overflow.
					    * 604/750: Overrides FCECE (DISCOUNT).
					    */
#define MMCR0_TRIGGER		0x00002000 /* Disable PMC2-PMC6 until PMC1 overflow or other event.
					    * 74xx: cleared by hardware when the event occurs.
					    */
#define MMCR0_PMC1SEL		0x00001FC0 /* PMC1 event selector, 7 bits. */
#define MMCR0_PMC2SEL		0x0000003F /* PMC2 event selector, 6 bits. */

/* MMCR1 layout (604e-7457) */
#define MMCR1_PMC3SEL		0xF8000000 /* PMC3 event selector, 5 bits. */
#define MMCR1_PMC4SEL		0x07C00000 /* PMC4 event selector, 5 bits. */
#define MMCR1_PMC5SEL		0x003E0000 /* PMC5 event selector, 5 bits. (745x only) */
#define MMCR1_PMC6SEL		0x0001F800 /* PMC6 event selector, 6 bits. (745x only) */
#define MMCR1__RESERVED		0x000007FF /* should be zero */

/* MMCR2 layout (7400-7457) */
#define MMCR2_THRESHMULT	0x80000000 /* MMCR0[THRESHOLD] multiplier. */
#define MMCR2_SMCNTEN		0x40000000 /* 7400/7410 only, should be zero. */
#define MMCR2_SMINTEN		0x20000000 /* 7400/7410 only, should be zero. */
#define MMCR2__RESERVED		0x1FFFFFFF /* should be zero */
#define MMCR2_RESERVED		(MMCR2_SMCNTEN | MMCR2_SMINTEN | MMCR2__RESERVED)
