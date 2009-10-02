/*
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
#ifndef __PFMLIB_INTEL_X86_PRIV_H__
#define __PFMLIB_INTEL_X86_PRIV_H__

/*
 * This file contains the definitions used for all Intel X86 processors
 */


/*
 * maximum number of unit masks/event
 */
#define INTEL_X86_NUM_UMASKS 32

/*
 * unit mask description
 */
typedef struct {
	char			*uname; /* unit mask name */
	char			*udesc; /* unit umask description */
	unsigned int		ucode;  /* unit mask code */
	unsigned int		uflags;	/* unit mask flags */
} intel_x86_umask_t;

/*
 * event-specific encoder (optional)
 */
typedef int (*intel_x86_encoder_t)(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs);

/*
 * event description
 */
typedef struct {
	char				*name;	/* event name */
	char				*desc;	/* event description */
	uint64_t			cntmsk;	/* supported counters */
	unsigned int			code; 	/* event code */
	unsigned int			numasks;/* number of umasks */
	unsigned int			flags;	/* flags */
	unsigned int			modmsk;/* bitmask of attr for this event */
	intel_x86_encoder_t		encoder; /* event-specific encoder (optional) */
	intel_x86_umask_t		umasks[INTEL_X86_NUM_UMASKS]; /* umask desc */
} intel_x86_entry_t;

/*
 * pme_flags value (event and unit mask)
 */
#define INTEL_X86_UMASK_NCOMBO	0x01	/* unit mask cannot be combined (default: combination ok) */
#define INTEL_X86_FALLBACK_GEN	0x02	/* fallback from fixed to generic counter possible */
#define INTEL_X86_CSPEC		0x04 	/* requires a intel_x86 core specification */
#define INTEL_X86_PEBS		0x08 	/* event support PEBS */
#define INTEL_X86_ENCODER	0x10 	/* event requires model-specific encoding */
#define INTEL_X86_MESI		0x20 	/* requires MESI bits to be set */

typedef union pfm_intel_x86_reg {
	unsigned long long val;			/* complete register value */
	struct {
		unsigned long sel_event_select:8;	/* event mask */
		unsigned long sel_unit_mask:8;		/* unit mask */
		unsigned long sel_usr:1;		/* user level */
		unsigned long sel_os:1;			/* system level */
		unsigned long sel_edge:1;		/* edge detec */
		unsigned long sel_pc:1;			/* pin control */
		unsigned long sel_int:1;		/* enable APIC intr */
		unsigned long sel_anythr:1;		/* measure any thread */
		unsigned long sel_en:1;			/* enable */
		unsigned long sel_inv:1;		/* invert counter mask */
		unsigned long sel_cnt_mask:8;		/* counter mask */
		unsigned long sel_res2:32;
	} perfevtsel;

	struct {
		unsigned long usel_event:8;	/* event select */
		unsigned long usel_umask:8;	/* event unit mask */
		unsigned long usel_res1:1;	/* reserved */
		unsigned long usel_occ:1;	/* occupancy reset */
		unsigned long usel_edge:1;	/* edge detection */
		unsigned long usel_res2:1;	/* reserved */
		unsigned long usel_int:1;	/* PMI enable */
		unsigned long usel_res3:1;	/* reserved */
		unsigned long usel_en:1;	/* enable */
		unsigned long usel_inv:1;	/* invert */
		unsigned long usel_cnt_mask:8;	/* counter mask */
		unsigned long usel_res4:32;	/* reserved */
	} nhm_unc;

	struct {
		unsigned long cpl_eq0:1;	/* filter out branches at pl0 */
		unsigned long cpl_neq0:1;	/* filter out branches at pl1-pl3 */
		unsigned long jcc:1;		/* filter out condition branches */
		unsigned long near_rel_call:1;	/* filter out near relative calls */
		unsigned long near_ind_call:1;	/* filter out near indirect calls */
		unsigned long near_ret:1;	/* filter out near returns */
		unsigned long near_ind_jmp:1;	/* filter out near unconditional jmp/calls */
		unsigned long near_rel_jmp:1;	/* filter out near uncoditional relative jmp */
		unsigned long far_branch:1;	/* filter out far branches */ 
		unsigned long reserved1:23;	/* reserved */
		unsigned long reserved2:32;	/* reserved */
	} nhm_lbr_select;
} pfm_intel_x86_reg_t;

#define INTEL_X86_ATTR_U	0 /* user (1, 2, 3) */
#define INTEL_X86_ATTR_K	1 /* kernel (0) */
#define INTEL_X86_ATTR_I	2 /* invert */
#define INTEL_X86_ATTR_E	3 /* edge */
#define INTEL_X86_ATTR_C	4 /* counter mask */
#define INTEL_X86_ATTR_T	5 /* any thread */

#define _INTEL_X86_ATTR_U  (1 << INTEL_X86_ATTR_U)
#define _INTEL_X86_ATTR_K  (1 << INTEL_X86_ATTR_K)
#define _INTEL_X86_ATTR_I  (1 << INTEL_X86_ATTR_I)
#define _INTEL_X86_ATTR_E  (1 << INTEL_X86_ATTR_E)
#define _INTEL_X86_ATTR_C  (1 << INTEL_X86_ATTR_C)
#define _INTEL_X86_ATTR_T  (1 << INTEL_X86_ATTR_T)

#define INTEL_X86_ATTRS \
	(_INTEL_X86_ATTR_I|_INTEL_X86_ATTR_E|_INTEL_X86_ATTR_C|_INTEL_X86_ATTR_U|_INTEL_X86_ATTR_K)

#define INTEL_V1_ATTRS 		INTEL_X86_ATTRS
#define INTEL_V2_ATTRS 		INTEL_X86_ATTRS
#define INTEL_FIXED2_ATTRS	(_INTEL_X86_ATTR_U|_INTEL_X86_ATTR_K)
#define INTEL_FIXED3_ATTRS	(INTEL_FIXED2_ATTRS|_INTEL_X86_ATTR_T)
#define INTEL_V3_ATTRS 		(INTEL_V2_ATTRS|_INTEL_X86_ATTR_T)

/* let's define some handy shortcuts! */
#define sel_event_select perfevtsel.sel_event_select
#define sel_unit_mask	 perfevtsel.sel_unit_mask
#define sel_usr		 perfevtsel.sel_usr
#define sel_os		 perfevtsel.sel_os
#define sel_edge	 perfevtsel.sel_edge
#define sel_pc		 perfevtsel.sel_pc
#define sel_int		 perfevtsel.sel_int
#define sel_en		 perfevtsel.sel_en
#define sel_inv		 perfevtsel.sel_inv
#define sel_cnt_mask	 perfevtsel.sel_cnt_mask
#define sel_anythr	 perfevtsel.sel_anythr

typedef struct {
	unsigned int version:8;
	unsigned int num_cnt:8;
	unsigned int cnt_width:8;
	unsigned int ebx_length:8;
} intel_x86_pmu_eax_t;

typedef struct {
	unsigned int num_cnt:6;
	unsigned int cnt_width:6;
	unsigned int reserved:20;
} intel_x86_pmu_edx_t;

typedef struct {
	unsigned int no_core_cycle:1;
	unsigned int no_inst_retired:1;
	unsigned int no_ref_cycle:1;
	unsigned int no_llc_ref:1;
	unsigned int no_llc_miss:1;
	unsigned int no_br_retired:1;
	unsigned int no_br_mispred_retired:1;
	unsigned int reserved:25;
} intel_x86_pmu_ebx_t;

static inline int
intel_x86_eflag(void *this, pfmlib_event_desc_t *e, int flag)
{
	const intel_x86_entry_t *pe = this_pe(this);
	return !!(pe[e->event].flags & flag);
}

static inline int
intel_x86_uflag(void *this, pfmlib_event_desc_t *e, int attr, int flag)
{
	const intel_x86_entry_t *pe = this_pe(this);
	return !!(pe[e->event].umasks[attr].uflags & flag);
}

extern const pfmlib_attr_desc_t intel_x86_mods[];

extern int intel_x86_detect(int *family, int *model);
extern int intel_x86_encode_gen(void *this, pfmlib_event_desc_t *e, pfm_intel_x86_reg_t *reg);
extern void intel_x86_display_reg(void *this, pfm_intel_x86_reg_t reg, int c, int event);

extern int pfm_intel_x86_event_is_valid(void *this, int pidx);
extern int pfm_intel_x86_get_event_code(void *this, int i, uint64_t *code);
extern const char *pfm_intel_x86_get_event_name(void *this, int i);
extern const char *pfm_intel_x86_get_event_umask_name(void *this, int e, int attr);
extern const char *pfm_intel_x86_get_event_desc(void *this, int ev);
extern const char *pfm_intel_x86_get_event_umask_name(void *this, int e, int attr);
extern const char *pfm_intel_x86_get_event_umask_desc(void *this, int e, int attr);
extern int pfm_intel_x86_get_event_umask_code(void *this, int e, int attr, uint64_t *code);
extern const char *pfm_intel_x86_get_cycle_event(void *this);
extern const char *pfm_intel_x86_get_inst_retired(void *this);
extern int pfm_intel_x86_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs);
extern int pfm_intel_x86_get_event_numasks(void *this, int idx);
extern int pfm_intel_x86_get_event_first(void *this);
extern int pfm_intel_x86_get_event_next(void *this, int idx);
extern int pfm_intel_x86_get_event_umask_first(void *this, int idx);
extern int pfm_intel_x86_get_event_umask_next(void *this, int idx, int attr);
extern int pfm_intel_x86_get_event_perf_type(void *this, int pidx);
extern int pfm_intel_x86_get_event_modifiers(void *this, int pidx);
extern int pfm_intel_x86_validate_table(void *this, FILE *fp);
#endif /* __PFMLIB_INTEL_X86_PRIV_H__ */
