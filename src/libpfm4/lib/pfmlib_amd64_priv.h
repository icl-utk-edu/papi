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

#define PFMLIB_AMD64_MAX_UMASK	12

typedef struct {
	const char		*uname; /* unit mask name */
	const char		*udesc; /* event/umask description */
	unsigned int		ucode;  /* unit mask code */
	unsigned int		uflags; /* unit mask flags */
} amd64_umask_t;

typedef struct {
	const char		*name;	/* event name */
	const char		*desc;	/* event description */
	amd64_umask_t		umasks[PFMLIB_AMD64_MAX_UMASK]; /* umask desc */
	unsigned int		code; 	/* event code */
	unsigned int		numasks;/* number of umasks */
	unsigned int		flags;	/* flags */
	unsigned int		modmsk;	/* modifiers bitmask */
} amd64_entry_t;

#define AMD64_FAM10H AMD64_FAM10H_REV_B
typedef enum {
        AMD64_CPU_UN = 0,
        AMD64_K7,
        AMD64_K8_REV_B,
        AMD64_K8_REV_C,
        AMD64_K8_REV_D,
        AMD64_K8_REV_E,
        AMD64_K8_REV_F,
        AMD64_K8_REV_G,
        AMD64_FAM10H_REV_B,
        AMD64_FAM10H_REV_C,
        AMD64_FAM10H_REV_D,
} amd64_rev_t;

/* 
 * flags values (bottom 8 bits only)
 * bits 00-07: flags
 * bits 08-15: from revision
 * bits 16-23: till revision
 */
#define AMD64_FROM_REV(rev)	((rev)<<8)
#define AMD64_TILL_REV(rev)	((rev)<<16)
#define AMD64_NOT_SUPP		0x1ff00

#define AMD64_FL_NCOMBO        	0x1 /* unit mask can be combined */
#define AMD64_FL_IBSFE		0x2 /* IBS fetch */
#define AMD64_FL_IBSOP		0x4 /* IBS op */
#define AMD64_FL_DFL		0x8 /* unit mask is default choice */

#define AMD64_FL_TILL_K8_REV_C		AMD64_TILL_REV(AMD64_K8_REV_C)
#define AMD64_FL_K8_REV_D		AMD64_FROM_REV(AMD64_K8_REV_D)
#define AMD64_FL_K8_REV_E		AMD64_FROM_REV(AMD64_K8_REV_E)
#define AMD64_FL_TILL_K8_REV_E		AMD64_TILL_REV(AMD64_K8_REV_E)
#define AMD64_FL_K8_REV_F		AMD64_FROM_REV(AMD64_K8_REV_F)
#define AMD64_FL_TILL_FAM10H_REV_B	AMD64_TILL_REV(AMD64_FAM10H_REV_B)
#define AMD64_FL_FAM10H_REV_C		AMD64_FROM_REV(AMD64_FAM10H_REV_C)
#define AMD64_FL_TILL_FAM10H_REV_C	AMD64_TILL_REV(AMD64_FAM10H_REV_C)
#define AMD64_FL_FAM10H_REV_D		AMD64_FROM_REV(AMD64_FAM10H_REV_D)

#define AMD64_ATTR_U	0
#define AMD64_ATTR_K	1
#define AMD64_ATTR_H	2
#define AMD64_ATTR_G	3
#define AMD64_ATTR_I	4
#define AMD64_ATTR_E	5
#define AMD64_ATTR_C	6
#define AMD64_ATTR_R	7
#define AMD64_ATTR_P	8

#define _AMD64_ATTR_U  (1 << AMD64_ATTR_U)
#define _AMD64_ATTR_K  (1 << AMD64_ATTR_K)
#define _AMD64_ATTR_I  (1 << AMD64_ATTR_I)
#define _AMD64_ATTR_E  (1 << AMD64_ATTR_E)
#define _AMD64_ATTR_C  (1 << AMD64_ATTR_C)
#define _AMD64_ATTR_H  (1 << AMD64_ATTR_H)
#define _AMD64_ATTR_G  (1 << AMD64_ATTR_G)
#define _AMD64_ATTR_R  (1 << AMD64_ATTR_R)
#define _AMD64_ATTR_P  (1 << AMD64_ATTR_P)

#define AMD64_BASIC_ATTRS \
	(_AMD64_ATTR_I|_AMD64_ATTR_E|_AMD64_ATTR_C|_AMD64_ATTR_U|_AMD64_ATTR_K)

#define AMD64_K8_ATTRS			(AMD64_BASIC_ATTRS)
#define AMD64_FAM10H_ATTRS		(AMD64_BASIC_ATTRS|_AMD64_ATTR_H|_AMD64_ATTR_G)
#define AMD64_FAM10H_ATTRS_IBSFE	(_AMD64_ATTR_R)
#define AMD64_FAM10H_ATTRS_IBSOP	(0)

/*
 * AMD64 MSR definitions
 */
typedef union {
	uint64_t val;				/* complete register value */
	struct {
		uint64_t sel_event_mask:8;	/* event mask */
		uint64_t sel_unit_mask:8;	/* unit mask */
		uint64_t sel_usr:1;		/* user level */
		uint64_t sel_os:1;		/* system level */
		uint64_t sel_edge:1;		/* edge detec */
		uint64_t sel_pc:1;		/* pin control */
		uint64_t sel_int:1;		/* enable APIC intr */
		uint64_t sel_res1:1;		/* reserved */
		uint64_t sel_en:1;		/* enable */
		uint64_t sel_inv:1;		/* invert counter mask */
		uint64_t sel_cnt_mask:8;	/* counter mask */
		uint64_t sel_event_mask2:4;	/* 10h only: event mask [11:8] */
		uint64_t sel_res2:4;		/* reserved */
		uint64_t sel_guest:1;		/* 10h only: guest only counter */
		uint64_t sel_host:1;		/* 10h only: host only counter */
		uint64_t sel_res3:22;		/* reserved */
	} perfsel;

	struct {
		uint64_t maxcnt:16;
		uint64_t cnt:16;
		uint64_t lat:16;
		uint64_t en:1;
		uint64_t val:1;
		uint64_t comp:1;
		uint64_t icmiss:1;
		uint64_t phyaddrvalid:1;
		uint64_t l1tlbpgsz:2;
		uint64_t l1tlbmiss:1;
		uint64_t l2tlbmiss:1;
		uint64_t randen:1;
		uint64_t reserved:6;
	} ibsfetch;
	struct {
		uint64_t maxcnt:16;
		uint64_t reserved1:1;
		uint64_t en:1;
		uint64_t val:1;
		uint64_t reserved2:45;
	} ibsop;
} pfm_amd64_reg_t; /* MSR 0xc001000-0xc001003 */

/* let's define some handy shortcuts! */
#define sel_event_mask	perfsel.sel_event_mask
#define sel_unit_mask	perfsel.sel_unit_mask
#define sel_usr		perfsel.sel_usr
#define sel_os		perfsel.sel_os
#define sel_edge	perfsel.sel_edge
#define sel_pc		perfsel.sel_pc
#define sel_int		perfsel.sel_int
#define sel_en		perfsel.sel_en
#define sel_inv		perfsel.sel_inv
#define sel_cnt_mask	perfsel.sel_cnt_mask
#define sel_event_mask2 perfsel.sel_event_mask2
#define sel_guest	perfsel.sel_guest
#define sel_host	perfsel.sel_host

static struct {
        amd64_rev_t     	revision;
        char            	*name;
        int             	family;
        int             	model;
        int             	stepping;
	int			num_events; /* total number of events in table */
        const amd64_entry_t	*events;
} amd64_pmu;


#define amd64_revision    amd64_pmu.revision
#define amd64_num_events  amd64_pmu.num_events
#define amd64_events      amd64_pmu.events
#define amd64_family      amd64_pmu.family
#define amd64_model       amd64_pmu.model
#define amd64_stepping    amd64_pmu.stepping

static inline int
amd64_eflag(int idx, int flag)
{
	return !!(amd64_events[idx].flags & flag);
}

static inline int
amd64_uflag(int idx, int attr, int flag)
{
	return !!(amd64_events[idx].umasks[attr].uflags & flag);
}

#endif /* __PFMLIB_AMD64_PRIV_H__ */
