/*
 * perfmon.h
 *
 * Copyright (C) 2001 Hewlett-Packard Co
 * Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>
 */

/*
 * this header file defines the kernel API
 */
#ifndef __PERFMON_H__
#define __PERFMON_H__

/*
 * This structure provides a detailed way to setup a PMC register.
 * Once value is loaded, it must be copied (via pmu_reg) to the
 * perfmon_req_t and passed to the kernel via perfmonctl().
 */
typedef union {
	unsigned long pmu_reg;			/* generic PMD register */

	/* This is the Itanium-specific PMC layout for counter config */
	struct {
		unsigned long pmc_plm:4;	/* privilege level mask */
		unsigned long pmc_ev:1;		/* external visibility */
		unsigned long pmc_oi:1;		/* overflow interrupt */
		unsigned long pmc_pm:1;		/* privileged monitor */
		unsigned long pmc_ig1:1;	/* reserved */
		unsigned long pmc_es:7;		/* event select */
		unsigned long pmc_ig2:1;	/* reserved */
		unsigned long pmc_umask:4;	/* unit mask */
		unsigned long pmc_thres:3;	/* threshold */
		unsigned long pmc_ig3:1;	/* reserved (missing from table on p6-17) */
		unsigned long pmc_ism:2;	/* instruction set mask */
		unsigned long pmc_ig4:38;	/* reserved */
	} pmc_count_reg;

	/* Instruction Event Address Registers */
	struct {
		unsigned long iear_plm:4;	/* privilege level mask */
		unsigned long iear_ig1:2;	/* reserved */
		unsigned long iear_pm:1;	/* privileged monitor */
		unsigned long iear_tlb:1;	/* cache/tlb mode */
		unsigned long iear_ig2:8;	/* reserved */
		unsigned long iear_umask:4;	/* unit mask */
		unsigned long iear_ig3:4;	/* reserved */
		unsigned long iear_ism:2;	/* instruction set */
		unsigned long iear_ig4:38;	/* reserved */
	} pmc10_reg;

	/* Data Event Address Registers */
	struct {
		unsigned long dear_plm:4;	/* privilege level mask */
		unsigned long dear_ig1:2;	/* reserved */
		unsigned long dear_pm:1;	/* privileged monitor */
		unsigned long dear_tlb:1;	/* cache/tlb mode */
		unsigned long dear_ig2:8;	/* reserved */
		unsigned long dear_umask:4;	/* unit mask */
		unsigned long dear_ig3:4;	/* reserved */
		unsigned long dear_ism:2;	/* instruction set */
		unsigned long dear_ig4:2;	/* reserved */
		unsigned long dear_pt:1;	/* pass tags */
		unsigned long dear_ig5:35;	/* reserved */
	} pmc11_reg;

	/* Opcode matcher */
	struct {
		unsigned long ignored1:3;
		unsigned long mask:27;		/* mask encoding bits {40:27}{12:0} */
		unsigned long ignored2:3;	
		unsigned long match:27;		/* match encoding bits {40:27}{12:0} */
		unsigned long b:1;		/* B-syllable */
		unsigned long f:1;		/* F-syllable */
		unsigned long i:1;		/* I-syllable */
		unsigned long m:1;		/* M-syllable */
	} pmc8_9_reg;

	struct {
		unsigned long iear_v:1;		/* valid bit */
		unsigned long iear_tlb:1;	/* tlb miss bit */
		unsigned long iear_ig1:3;	/* reserved */
		unsigned long iear_icla:59;	/* instruction cache line address {60:51} sxt {50}*/
	} pmd0_reg;

	struct {
		unsigned long iear_lat:12;	/* latency */
		unsigned long iear_ig1:52;	/* reserved */
	} pmd1_reg;
	struct {
		unsigned long dear_v:1;		/* valid bit */
		unsigned long dear_ig1:1;	/* reserved */
		unsigned long dear_slot:2;	/* slot number */
		unsigned long dear_iaddr:60;	/* instruction address */
	} pmd17_reg;

	struct {
		unsigned long dear_lat:12;	/* latency */
		unsigned long dear_ig1:50;	/* reserved */
		unsigned long dear_level:2;	/* level */
	} pmd3_reg;

	struct {
		unsigned long dear_daddr;	/* data address */
	} pmd2_reg;

	/* BRanch Trace Buffer registers */
	struct {
		unsigned long btbc_plm:4;	/* privilege level */
		unsigned long btbc_ig1:2;
		unsigned long btbc_pm:1;	/* privileged monitor */
		unsigned long btbc_tar:1;	/* target address register */
		unsigned long btbc_tm:2;	/* taken mask */
		unsigned long btbc_ptm:2;	/* predicted taken address mask */
		unsigned long btbc_ppm:2;	/* predicted predicate mask */
		unsigned long btbc_bpt:1;	/* branch prediction table */
		unsigned long btbc_bac:1;	/* branch address calculator */
		unsigned long btbc_ig2:48;
	} pmc12_reg;

	struct {
		unsigned long btb_b:1;		/* branch bit */
		unsigned long btb_mp:1;		/* mispredict bit */
		unsigned long btb_slot:2;	/* which slot, 3=not taken branch */
		unsigned long btb_addr:60;	/* b=1, bundle address, b=0 target address */
	} pmd8_15_reg;

	struct {
		unsigned long btbi_bbi:3;	/* branch buffer index */
		unsigned long btbi_full:1;	/* full bit (sticky) */
		unsigned long btbi_ignored:60;
	} pmd16_reg;

} perfmon_reg_t; 

/*
 * This header is at the beginning of the sampling buffer returned to the user.
 */
typedef struct {
	int		hdr_version;		/* could be used to differentiate formats */
	int		hdr_reserved;
	unsigned long	hdr_entry_size;		/* size of one entry in bytes */
	unsigned long	hdr_count;		/* how many valid entries */
	unsigned long   hdr_pmds;		/* which pmds get recorded */
} perfmon_smpl_hdr_t;


/*
 * Header entry in the buffer as a header as follows.
 * The header is directly followed with the PMDS to saved in increasing index order:
 * PMD4, PMD5, .... How many PMDs are present is determined by the tool which must
 * keep track of it when generating the final trace file.
 */
typedef struct {
	int 		pid;		/* identification of process */
	int 		cpu;		/* which cpu was used */
	unsigned long	rate;		/* initial value of this counter */
	unsigned long	stamp;		/* timestamp */
	unsigned long	ip;		/* where did the overflow interrupt happened */
	unsigned long	regs;		/* which registers overflowed (up to 64)*/
} perfmon_smpl_entry_t;



/* let's define some handy shortcuts ! */
#define pmc_plm		pmc_count_reg.pmc_plm
#define pmc_ev		pmc_count_reg.pmc_ev
#define pmc_oi		pmc_count_reg.pmc_oi
#define pmc_pm		pmc_count_reg.pmc_pm
#define pmc_es		pmc_count_reg.pmc_es
#define pmc_umask	pmc_count_reg.pmc_umask
#define pmc_thres	pmc_count_reg.pmc_thres
#define pmc_ism		pmc_count_reg.pmc_ism

/* privilege level mask */
#define PFM_PLM0	1
#define PFM_PLM1	2
#define PFM_PLM2	4
#define PFM_PLM3	8


#define PFM_WRITE_PMCS		0xa0	/* write specified PMCs */
#define PFM_WRITE_PMDS		0xa1	/* read specified PMDs */
#define PFM_READ_PMDS		0xa2	/* read specified PMDs */
#define PFM_STOP		0xa3	/* freeze + up/pp=0 */
#define PFM_START		0xa4	/* PSR.up or PSR.pp=1 */
#define PFM_ENABLE		0xa5	/* unfreeze only */
#define PFM_DISABLE		0xa6	/* freeze only (context lost) */
#define PFM_CREATE_CONTEXT	0xa7

#define PFM_RESTART		0xcf	/* restart after EAR/BTB notification */

/* for debug purposes only, will go away */
#define PFM_DEBUG_ON		0xe0
#define PFM_DEBUG_OFF		0xe1

/*
 * perfmon API flags
 */
#define PFM_FL_INHERIT_NONE	 0x00	/* never inherit a context across fork (default) */
#define PFM_FL_INHERIT_ONCE	 0x01	/* clone pfm_context only once across fork() */
#define PFM_FL_INHERIT_ALL	 0x02	/* always clone pfm_context across fork() */
#define PFM_FL_SMPL_OVFL_NOBLOCK 0x04	/* do not block on sampling buffer overflow */
#define PFM_FL_SYSTEMWIDE	 0x08	/* create a systemwide context */
/*
 * PMC API flags
 */
#define PFM_REGFL_OVFL_NOTIFY	1		/* send notification on overflow */



typedef struct {
	unsigned long ar_adr:64;	/* instruction/data address */
	unsigned long ar_mask:56;	/* address mask */
	unsigned long ar_plm:4;		/* privilege level */
    	unsigned long ar_res:4;		/* not used : align nicely */
} pm_address_range_t;

/*
 * Structure used to define a context
 */
typedef struct {
	unsigned long smpl_entries;	/* how many entries in sampling buffer */
	unsigned long smpl_regs;	/* which pmds to record on overflow */
	void	      *smpl_vaddr;	/* returns address of BTB buffer */

	pid_t	      notify_pid;	/* which process to notify on overflow */
	int	      notify_sig; 	/* XXX: not used anymore */

	int	      flags;		/* context flags (will replaced API flags) */
} pfreq_context_t;

/*
 * structure used to configure a PMC or PMD
 */
typedef struct {
	unsigned long	reg_num;	/* which register */
	unsigned long	reg_value;	/* configuration (PMC) or initial value (PMD) */
	unsigned long	reg_smpl_reset;	/* reset of sampling buffer overflow (large) */
	unsigned long	reg_ovfl_reset;	/* reset on counter overflow (small) */
	int		reg_flags;	/* (PMD): notify/don't notify */
} pfreq_reg_t;

/*
 * generic request container
 */
typedef union {
	pfreq_context_t	pfr_ctx;	
	pfreq_reg_t	pfr_reg;	
} perfmon_req_t;

/*
 * pid   : process id to refer to (must be a child)
 * ops   : which operation to perform (READ_PMD, WRITE_PMD, START, STOP...)
 * flags : further qualifies calls (valid only on WRITE_* and START)
 *         PFM_SYSTEM_WIDE, PFM_MONITOR_ALL_CPUS, PFM_FORK_INHERIT
 * cmd   : a pointer to a list of requests to process
 * count : the number of requests in the list
 */
extern int perfmonctl(int pid, int cmd, int flags, perfmon_req_t *ops, int count);

#define PMU_MAX_COUNTERS	4	/* XXX: (needs to be dynamic) implementation specific number of counters */
#define PMU_FIRST_COUNTER	4	/* index of first counter */
#define PMU_MAX_PMCS		14	/* XXX: needs to be dynamic via /proc */
#define PMU_MAX_PMDS		18	/* XXX: needs to be dynamic via /proc */
#define PMU_MAX_BTB		8	/* XXX: needs to be dynamic */
#endif /* __PERFMON_H__ */
