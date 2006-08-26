/*
 * Intel Core Duo/Core Solo PMU specific types and definitions
 *
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PFMLIB_COREDUO_H__
#define __PFMLIB_COREDUO_H__

#include <perfmon/pfmlib.h>
/*
 * privilege level mask usage for Intel Core Duo/Core Solo:
 *
 * PFM_PLM0 = OS (kernel, hypervisor, ..)
 * PFM_PLM1 = unused (ignored)
 * PFM_PLM2 = unused (ignored)
 * PFM_PLM3 = USR (user level)
 */

#ifdef __cplusplus
extern "C" {
#endif

#define PMU_COREDUO_NUM_COUNTERS	2	/* total numbers of EvtSel/EvtCtr */
#define PMU_COREDUO_NUM_PERFSEL		2	/* total number of EvtSel defined */
#define PMU_COREDUO_NUM_PERFCTR		2	/* total number of EvtCtr defined */
#define PMU_COREDUO_COUNTER_WIDTH	32	/* hardware counter bit width   */

#define PMU_COREDUO_CNT_MASK_MAX	4 	/* max cnt_mask value */
/*
 * This structure provides a detailed way to setup a PMC register.
 * Note about sel_unit_mask:
 * 	- bits [0:3] are used for MESI encoding (bit[0]=I, bit[1]=S, bit[2]=I, bit[3]=M)
 *	- bit [12:13]: for some events: Hardware Prefeth Qualification
 *	- bit [13]: for some events: bus agent encoding
 *
 *	Given the overlap condition for bit[13] and the fact that all 3 fields depend on the
 *	event, we do not define a bitfields for those.
 */
typedef union {
	unsigned long long val;				/* complete register value */
	struct {
		unsigned long long sel_event_mask:8;	/* event mask */
		unsigned long long sel_unit_mask:8;	/* unit mask */
		unsigned long long sel_usr:1;		/* user level */
		unsigned long long sel_os:1;		/* system level */
		unsigned long long sel_edge:1;		/* edge detec */
		unsigned long long sel_pc:1;		/* pin control */
		unsigned long long sel_int:1;		/* enable APIC intr */
		unsigned long long sel_res1:1;		/* reserved */
		unsigned long long sel_en:1;		/* enable */
		unsigned long long sel_inv:1;		/* invert counter mask */
		unsigned long long sel_cnt_mask:8;	/* counter mask */
		unsigned long long sel_res2:32;		/* reserved */
	} perfsel;
} pfm_coreduo_perfevtsel_reg_t;

typedef union {
	unsigned long long val;	/* counter value */
	/* counting perfctr register */
	struct {
		unsigned long long ctr_count:32;	/* 32-bit hardware counter  */
		unsigned long long ctr_res1:32;		/* reserved */
	} perfctr;
} pfm_coreduo_ctr_reg_t;

typedef struct {
	unsigned int	cnt_mask;	/* threshold ([4-255] are reserved) */
	unsigned int	flags;		/* counter specific flag */
} pfmlib_coreduo_counter_t;

/*
 * possible flags for pfmlib_coreduo_counter_t
 */
#define PFM_COREDUO_SEL_INV		0x001	/* inverse */
#define PFM_COREDUO_SEL_EDGE		0x002	/* edge detect */

/*
 * specific parameters for the library
 */
typedef struct {
	pfmlib_coreduo_counter_t	pfp_coreduo_counters[PMU_COREDUO_NUM_COUNTERS];	/* extended counter features */
	uint64_t			reserved[4];		/* for future use */
} pfmlib_coreduo_input_param_t;

typedef struct {
	uint64_t	reserved[8];		/* for future use */
} pfmlib_coreduo_output_param_t;

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_COREDUO_H__ */
