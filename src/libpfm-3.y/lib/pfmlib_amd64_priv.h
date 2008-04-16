/*
 * Copyright (c) 2004-2006 Hewlett-Packard Development Company, L.P.
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */
#ifndef __PFMLIB_AMD64_PRIV_H__
#define __PFMLIB_AMD64_PRIV_H__

/* PERFSEL/PERFCTR include IBS registers of family 10h */
#define PMU_AMD64_NUM_PERFSEL	6	/* total number of PMCs defined */
#define PMU_AMD64_NUM_PERFCTR	14	/* total number of PMDs defined */
#define PMU_AMD64_NUM_COUNTERS	4	/* total numbers of EvtSel/EvtCtr */
#define PMU_AMD64_COUNTER_WIDTH	48	/* hardware counter bit width */
#define PMU_AMD64_CNT_MASK_MAX	4 	/* max cnt_mask value */
#define PMU_AMD64_IBSFETCHCTL_PMC 4	/* IBS: fetch PMC base */
#define PMU_AMD64_IBSFETCHCTL_PMD 4	/* IBS: fetch PMD base */
#define PMU_AMD64_IBSOPCTL_PMC 5	/* IBS: op PMC base */
#define PMU_AMD64_IBSOPCTL_PMD 7	/* IBS: op PMD base */

#define PFMLIB_AMD64_MAX_UMASK	9

typedef struct {
	char			*pme_uname; /* unit mask name */
	char			*pme_udesc; /* event/umask description */
	unsigned int		pme_ucode;  /* unit mask code */
	unsigned int		pme_uflags; /* unit mask flags */
} pme_amd64_umask_t;

typedef struct {
	char			*pme_name;	/* event name */
	char			*pme_desc;	/* event description */
	pme_amd64_umask_t	pme_umasks[PFMLIB_AMD64_MAX_UMASK]; /* umask desc */
	unsigned int		pme_code; 	/* event code */
	unsigned int		pme_numasks;	/* number of umasks */
	unsigned int		pme_flags;	/* flags */
} pme_amd64_entry_t;

/* 
 * pme_flags values
 */
#define PFMLIB_AMD64_UMASK_COMBO	0x1 /* unit mask can be combined */
#define PFMLIB_AMD64_K8_REV_D		0x2 /* event requires at least rev D */
#define PFMLIB_AMD64_K8_REV_E		0x4 /* event requires at least rev E */
#define PFMLIB_AMD64_K8_REV_F		0x8 /* event requires at least rev F */

#endif /* __PFMLIB_AMD64_PRIV_H__ */
