/*
 * Generic MIPS64 PMU specific types and definitions
 *
 * Contributed by Philip Mucci <mucci@cs.utk.edu> based on code from
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
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

#ifndef __PFMLIB_GEN_MIPS64_H__
#define __PFMLIB_GEN_MIPS64_H__

#include <endian.h> /* MIPS are bi-endian */

#include <perfmon/pfmlib.h>
/*
 * privilege level mask usage for MIPS:
 *
 * PFM_PLM0 = KERNEL
 * PFM_PLM1 = SUPERVISOR
 * PFM_PLM2 = INTERRUPT
 * PFM_PLM3 = USER
 */

#ifdef __cplusplus
extern "C" {
#endif

#define PMU_GEN_MIPS64_NUM_COUNTERS		4	/* total numbers of EvtSel/EvtCtr */
#define PMU_GEN_MIPS64_COUNTER_WIDTH		32	/* hardware counter bit width   */
/*
 * This structure provides a detailed way to setup a PMC register.
 * Once value is loaded, it must be copied (via pmu_reg) to the
 * perfmon_req_t and passed to the kernel via perfmonctl().
 *
 * It needs to be adjusted based on endianess
 */
#if __BYTE_ORDER == __LITTLE_ENDIAN

typedef union {
	uint64_t	val;				/* complete register value */
	struct {
	        unsigned long sel_exl:1;		/* int level */
		unsigned long sel_os:1;			/* system level */
		unsigned long sel_sup:1;		/* supervisor level */
		unsigned long sel_usr:1;		/* user level */
	        unsigned long sel_int:1;		/* enable intr */
	  	unsigned long sel_event_mask:4;		/* event mask */
		unsigned long sel_res1:23;		/* reserved */
		unsigned long sel_res2:32;		/* reserved */
	} perfsel;
} pfm_gen_mips64_sel_reg_t;

#elif __BYTE_ORDER == __BIG_ENDIAN

typedef union {
	uint64_t	val;				/* complete register value */
	struct {
		unsigned long sel_res2:32;		/* reserved */
		unsigned long sel_res1:23;		/* reserved */
	  	unsigned long sel_event_mask:4;		/* event mask */
	        unsigned long sel_int:1;		/* enable intr */
		unsigned long sel_usr:1;		/* user level */
		unsigned long sel_sup:1;		/* supervisor level */
		unsigned long sel_os:1;			/* system level */
	        unsigned long sel_exl:1;		/* int level */
	} perfsel;
} pfm_gen_mips64_sel_reg_t;

#else
#error "cannot determine endianess"
#endif

typedef union {
	uint64_t val;	/* counter value */
	/* counting perfctr register */
	struct {
		unsigned long ctr_count:32;	/* 32-bit hardware counter  */
	} perfctr;
} pfm_gen_mips64_ctr_reg_t;

typedef struct {
	unsigned int	cnt_mask;	/* threshold ([4-255] are reserved) */
	unsigned int	flags;		/* counter specific flag */
} pfmlib_gen_mips64_counter_t;

/*
 * MIPS64 specific parameters for the library
 */
typedef struct {
	pfmlib_gen_mips64_counter_t	pfp_gen_mips64_counters[PMU_GEN_MIPS64_NUM_COUNTERS];	/* extended counter features */
	uint64_t			reserved[4];		/* for future use */
} pfmlib_gen_mips64_input_param_t;

typedef struct {
	uint64_t	reserved[8];		/* for future use */
} pfmlib_gen_mips64_output_param_t;

/*
 * SiCortex specific
 */

typedef union {
	uint64_t	val;				/* complete register value */
	struct {
	        unsigned long sel_exl:1;		/* int level */
		unsigned long sel_os:1;			/* system level */
		unsigned long sel_sup:1;		/* supervisor level */
		unsigned long sel_usr:1;		/* user level */
	        unsigned long sel_int:1;		/* enable intr */
	  	unsigned long sel_event_mask:6;		/* event mask */
		unsigned long sel_res1:23;		/* reserved */
		unsigned long sel_res2:32;		/* reserved */
	} perfsel;
} pfm_gen_ice9_sel_reg_t;

#define PMU_ICE9_SCB_NUM_COUNTERS 256

typedef union {
	uint64_t val;
	struct {
		unsigned long Interval:4;
		unsigned long IntBit:5;
		unsigned long NoInc:1;
		unsigned long AddrAssert:1;
		unsigned long MagicEvent:2;
		unsigned long Reserved:19;
	} ice9_ScbPerfCtl_reg; 
	struct {
		unsigned long HistGte:20;
		unsigned long Reserved:12;
	} ice9_ScbPerfHist_reg; 
	struct {
		unsigned long Bucket:8;
		unsigned long Reserved:24;
	} ice9_ScbPerfBuckNum_reg;
	struct {
		unsigned long ena:1;
		unsigned long Reserved:31;
	} ice9_ScbPerfEna_reg; 
	struct {
		unsigned long event:15;
		unsigned long hist:1;
		unsigned long ifOther:2;
		unsigned long Reserved:15;
	} ice9_ScbPerfBucket_reg; 
} pmc_ice9_scb_reg_t;

typedef union {
	uint64_t val;
	struct {
		unsigned long Reserved:2;
		uint64_t VPCL:38;
		unsigned long VPCH:2;
	} ice9_CpuPerfVPC_reg; 
	struct {
		unsigned long Reserved:5;
		unsigned long PEA:31;
		unsigned long Reserved2:12;
		unsigned long ASID:8;
		unsigned long L2STOP:4;
		unsigned long L2STATE:3;
		unsigned long L2HIT:1;
	} ice9_CpuPerfPEA_reg; 
} pmd_ice9_cpu_reg_t;
  
typedef struct {
	unsigned long NoInc:1;
	unsigned long Interval:4;
	unsigned long HistGte:20;
	unsigned long Bucket:8;
} pfmlib_ice9_scb_t;

typedef struct {
	unsigned long ifOther:2;
	unsigned long hist:1;
} pfmlib_ice9_scb_counter_t;
  
#define PFMLIB_INPUT_SCB_NONE (unsigned long)0x0
#define PFMLIB_INPUT_SCB_INTERVAL (unsigned long)0x1
#define PFMLIB_INPUT_SCB_NOINC (unsigned long)0x2
#define PFMLIB_INPUT_SCB_HISTGTE (unsigned long)0x4
#define PFMLIB_INPUT_SCB_BUCKET (unsigned long)0x8
#define PFMLIB_INPUT_SCB_COUNTER (unsigned long)0x10

typedef struct {
	unsigned long flags; 
	pfmlib_ice9_scb_counter_t pfp_ice9_scb_counters[PMU_ICE9_SCB_NUM_COUNTERS];
	pfmlib_ice9_scb_t pfp_ice9_scb_global;
} pfmlib_ice9_input_param_t;

typedef struct {
	unsigned long reserved;
} pfmlib_ice9_output_param_t;

/*
 * function exported to applications
 */

/* CPU counter */
int pfm_ice9_is_cpu(unsigned int i);

/* SCB counter */
int pfm_ice9_is_scb(unsigned int i);

/* Reg 25 domain support */
int pfm_ice9_support_domain(unsigned int i);

/* VPC/PEA sampling support */
int pfm_ice9_support_vpc_pea(unsigned int i);

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_GEN_MIPS64_H__ */
