/*
 * This file contains the user level interface description for
 * the perfmon-2.x interface on Linux.
 *
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 */
#ifndef __PERFMON_H__
#define __PERFMON_H__

#include <sys/types.h>
#include <stdint.h>
#include <syscall.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __ia64__
#include <perfmon/perfmon_ia64.h>
#endif

#ifdef __x86_64__
#include <perfmon/perfmon_x86_64.h>
#endif

#ifdef __i386__
#include <perfmon/perfmon_i386.h>
#endif

#ifdef __powerpc__
#include <perfmon/perfmon_powerpc.h>
#endif

#ifdef __mips__
#include <perfmon/perfmon_mips64.h>
#endif

#ifdef __crayx2
#include <perfmon/perfmon_crayx2.h>
#endif

#define PFM_MAX_PMCS	PFM_ARCH_MAX_PMCS
#define PFM_MAX_PMDS	PFM_ARCH_MAX_PMDS

/*
 * number of element for each type of bitvector
 */
#define PFM_BPL		(sizeof(uint64_t)<<3)
#define PFM_BVSIZE(x)   (((x)+PFM_BPL-1) / PFM_BPL)
#define PFM_PMD_BV      PFM_BVSIZE(PFM_MAX_PMDS)
#define PFM_PMC_BV      PFM_BVSIZE(PFM_MAX_PMCS)

/*
 * PMC/PMD flags to use with pfm_write_pmds() or pfm_write_pmcs()
 *
 * reg_flags layout:
 * bit 00-15 : generic flags
 * bit 16-23 : arch-specific flags
 * bit 24-31 : error codes
 */
#define PFM_REGFL_OVFL_NOTIFY	0x1	/* PMD: send notification on overflow */
#define PFM_REGFL_RANDOM	0x2	/* PMD: randomize sampling interval   */
#define PFM_REGFL_NO_EMUL64	0x4	/* PMC: no 64-bit emulation for counter */

/*
 * generic event set flags
 */
#define PFM_SETFL_OVFL_SWITCH	0x01 /* enable switch on overflow (subject to individual switch_cnt */
#define PFM_SETFL_TIME_SWITCH	0x02 /* switch set on timeout */

/*
 * PMD/PMC return flags in case of error (ignored on input)
 *
 * Those flags are used on output and must be checked in case EINVAL is returned
 * by a command accepting a vector of values and each has a flag field, such as
 * pfarg_pmc_t or pfarg_pmd_t.
 */
#define PFM_REG_RETFL_NOTAVAIL	(1<<31) /* set if register is implemented but not available */
#define PFM_REG_RETFL_EINVAL	(1<<30) /* set if register entry is invalid */
#define PFM_REG_RETFL_NOSET	(1<<29) /* event set does not exist */
#define PFM_REG_RETFL_MASK	(PFM_REG_RETFL_NOTAVAIL|PFM_REG_RETFL_EINVAL|PFM_REG_RETFL_NOSET)

#define PFM_REG_HAS_ERROR(flag)	(((flag) & PFM_REG_RETFL_MASK) != 0)

/*
 * argument to pfm_create_context()
 */
#ifndef PFMLIB_VERSION_22
typedef struct {
	uint32_t	ctx_flags;	   /* noblock/block/syswide */
	uint32_t	ctx_reserved1;	   /* for future use */
	uint64_t	ctx_reserved3[7];  /* for future use */
} pfarg_ctx_t;
#endif

/*
 * context flags (ctx_flags)
 *
 */
#define PFM_FL_NOTIFY_BLOCK    	 0x01	/* block task on user notifications */
#define PFM_FL_SYSTEM_WIDE	 0x02	/* create a system wide context */
#define PFM_FL_OVFL_NO_MSG	 0x80   /* no overflow msgs */
#define PFM_FL_MAP_SETS		 0x10	/* event sets are remapped */

/*
 * argument for pfm_write_pmcs()
 */
typedef struct {
	uint16_t reg_num;	   			/* which register */
	uint16_t reg_set;	   			/* event set for this register */
	uint32_t reg_flags;	   			/* input: flags, return: reg error */
	uint64_t reg_value;	   			/* pmc value */
	uint64_t reg_reserved2[4];			/* for future use */
} pfarg_pmc_t;

/*
 * argument pfm_write_pmds() and pfm_read_pmds()
 */
typedef struct {
	uint16_t reg_num;	   	/* which register */
	uint16_t reg_set;	   	/* event set for this register */
	uint32_t reg_flags;	   	/* input: flags, return: reg error */
	uint64_t reg_value;	   	/* initial pmc/pmd value */
	uint64_t reg_long_reset;	/* reset after buffer overflow notification */
	uint64_t reg_short_reset;   	/* reset after counter overflow */
	uint64_t reg_last_reset_val;	/* return: PMD last reset value */
	uint64_t reg_ovfl_switch_cnt;	/* how many overflow before switch for next set */
	uint64_t reg_reset_pmds[PFM_PMD_BV]; /* which other PMDS to reset on overflow */
	uint64_t reg_smpl_pmds[PFM_PMD_BV];  /* which other PMDS to record when the associated PMD overflows */
	uint64_t reg_smpl_eventid;  	/* opaque sampling event identifier */
	uint64_t reg_random_mask; 	/* bitmask used to limit random value */
	uint32_t reg_random_seed;   	/* seed for randomization (DEPRECATED) */
	uint32_t reg_reserved2[7];	/* for future use */
} pfarg_pmd_t;

/*
 * optional argument to pfm_start(), pass NULL if no arg needed
 */
typedef struct {
	uint16_t start_set;		/* event set to start with */
	uint16_t start_reserved1;	/* for future use */
	uint32_t start_reserved2;	/* for future use */
	uint64_t reserved3[3];		/* for future use */
} pfarg_start_t;

/*
 * argument to pfm_load_context()
 */
typedef struct {
	uint32_t	load_pid;          /* thread or CPU to attach to */
	uint16_t        load_set;          /* set to load first */
	uint16_t        load_reserved1;    /* for future use */
	uint64_t        load_reserved2[3]; /* for future use */
} pfarg_load_t;

/*
 * argument to pfm_create_evtsets()/pfm_delete_evtsets()
 */
#ifndef PFMLIB_VERSION_22
typedef struct {
	uint16_t	set_id;		  /* which set */
	uint16_t	set_reserved1;	  /* for future use */
	uint32_t    	set_flags; 	  /* input: flags for set, output: err flag */
	uint64_t	set_timeout;	  /* requested/effective switch timeout in nsecs */
	uint64_t	reserved[6];	  /* for future use */
} pfarg_setdesc_t;
#endif

/*
 * argument to pfm_getinfo_evtsets()
 */
#ifndef PFMLIB_VERSION_22
typedef struct {
        uint16_t	set_id;             /* which set */
        uint16_t	set_reserved1;      /* for future use */
        uint32_t	set_flags;          /* output:flags or error */
        uint64_t 	set_ovfl_pmds[PFM_PMD_BV]; /* output: last ovfl PMDs which triggered a switch from set */
        uint64_t	set_runs;           /* output: number of times the set was active */
        uint64_t	set_timeout;        /* output:effective/leftover switch timeout in nsecs */
	uint64_t	set_act_duration;   /* output: time set was active in nsecs */
	uint64_t	set_avail_pmcs[PFM_PMC_BV];
	uint64_t	set_avail_pmds[PFM_PMD_BV];
        uint64_t	reserved[6];        /* for future use */
} pfarg_setinfo_t;

typedef struct {
	uint32_t 	msg_type;		/* PFM_MSG_OVFL */
	uint32_t 	msg_ovfl_pid;		/* process id */
	uint16_t 	msg_active_set;		/* active set at the time of overflow */
	uint16_t 	msg_ovfl_cpu;		/* cpu on which the overflow occurred */
	uint32_t	msg_ovfl_tid;		/* thread id */
	uint64_t	msg_ovfl_ip;		/* instruction pointer where overflow interrupt happened */
	uint64_t	msg_ovfl_pmds[PFM_PMD_BV];/* which PMDs overflowed */
} pfarg_ovfl_msg_t;

#endif

#define PFM_MSG_OVFL	1	/* an overflow happened */
#define PFM_MSG_END	2	/* task to which context was attached ended */

/*
 * perfmon version number
 */
#define PFM_VERSION_MAJ		 2U

#ifndef PFMLIB_VERSION_22
#define PFM_VERSION_MIN		 6U
#endif

#define PFM_VERSION		 (((PFM_VERSION_MAJ&0xffff)<<16)|(PFM_VERSION_MIN & 0xffff))
#define PFM_VERSION_MAJOR(x)	 (((x)>>16) & 0xffff)
#define PFM_VERSION_MINOR(x)	 ((x) & 0xffff)

/*
 * for backward compatibility with old code (to go away)
 */
#ifdef PFMLIB_VERSION_22
typedef struct {
 	uint16_t	set_id;		  /* which set */
	uint16_t	set_id_next;	  /* next set to go to (must use PFM_SETFL_EXPL_NEXT) */
 	uint32_t    	set_flags; 	  /* input: flags for set, output: err flag */
 	uint64_t	set_timeout;	  /* requested/effective switch timeout in nsecs */
	uint64_t	set_mmap_offset;  /* cookie to pass as mmap offset to access 64-bit virtual PMD */
	uint64_t	reserved[5];	  /* for future use */
 } pfarg_setdesc_t;

typedef struct {
	unsigned char	ctx_smpl_buf_id[16];	/* which buffer format to use */
	uint32_t	ctx_flags;		/* noblock/block/syswide */
	int32_t		ctx_fd;			/* ret arg: fd for context */
	uint64_t	ctx_smpl_buf_size;	/* ret arg: actual buffer sz */
	uint64_t	ctx_reserved3[12];	/* for future use */
} pfarg_ctx_t;

typedef struct {
	uint16_t	set_id;             /* which set */
	uint16_t	set_id_next;        /* output: next set to go to (must use PFM_SETFL_EXPL_NEXT) */
	uint32_t	set_flags;          /* output:flags or error */
	uint64_t 	set_ovfl_pmds[PFM_PMD_BV]; /* output: last ovfl PMDs which triggered a switch from set */
	uint64_t	set_runs;           /* output: number of times the set was active */
	uint64_t	set_timeout;        /* output:effective/leftover switch timeout in nsecs */
	uint64_t	set_act_duration;   /* number of cycles set was active (syswide only) */
	uint64_t	set_mmap_offset;    /* cookie to pass as mmap offset to access 64-bit virtual PMD */
	uint64_t	set_avail_pmcs[PFM_PMC_BV];
	uint64_t	set_avail_pmds[PFM_PMD_BV];
	uint64_t	reserved[4];        /* for future use */
} pfarg_setinfo_t;

#ifdef __crayx2
#define PFM_MAX_HW_PMDS 512
#else
#define PFM_MAX_HW_PMDS 256
#endif
#define PFM_HW_PMD_BV   PFM_BVSIZE(PFM_MAX_HW_PMDS)

typedef struct {
	uint32_t 	msg_type;		/* PFM_MSG_OVFL */
	uint32_t 	msg_ovfl_pid;		/* process id */
	uint64_t	msg_ovfl_pmds[PFM_HW_PMD_BV];/* which PMDs overflowed */
	uint16_t 	msg_active_set;		/* active set at the time of overflow */
	uint16_t 	msg_ovfl_cpu;		/* cpu on which the overflow occurred */
	uint32_t	msg_ovfl_tid;		/* thread id */
	uint64_t	msg_ovfl_ip;		/* instruction pointer where overflow interrupt happened */
} pfarg_ovfl_msg_t;

#define PFM_VERSION_MIN		 2U	/* minior version number */
#endif

typedef union {
	uint32_t		type;
	pfarg_ovfl_msg_t	pfm_ovfl_msg;
} pfarg_msg_t;

extern int pfm_create_context(pfarg_ctx_t *ctx, char *smpl_name, void *smpl_arg, size_t smpl_size);
extern int pfm_write_pmcs(int fd, pfarg_pmc_t *pmcs, int count);
extern int pfm_write_pmds(int fd, pfarg_pmd_t *pmds, int count);
extern int pfm_read_pmds(int fd, pfarg_pmd_t *pmds, int count);
extern int pfm_load_context(int fd, pfarg_load_t *load);
extern int pfm_start(int fd, pfarg_start_t *start);
extern int pfm_stop(int fd);
extern int pfm_restart(int fd);
extern int pfm_create_evtsets(int fd, pfarg_setdesc_t *setd, int count);
extern int pfm_getinfo_evtsets(int fd, pfarg_setinfo_t *info, int count);
extern int pfm_delete_evtsets(int fd, pfarg_setdesc_t *setd, int count);
extern int pfm_unload_context(int fd);

/*
 * until the syscall stubs are implemented by glibc
 * we define them here
 */
#ifndef __NR_pfm_create_context
#ifdef __x86_64__
#ifdef CONFIG_PFMLIB_ARCH_CRAYXT
#define __NR_pfm_create_context		273
#else
#define __NR_pfm_create_context		284
#endif
#endif /* __x86_64__ */

#ifdef __i386__
#define __NR_pfm_create_context		324
#endif

#ifdef __ia64__
#define __NR_pfm_create_context		1310
#endif

#if defined(__mips__)
#if (_MIPS_SIM == _ABIN32) || (_MIPS_SIM == _MIPS_SIM_NABI32)
#define __NR_Linux 6000
#define __NR_pfm_create_context         __NR_Linux+279
#elif (_MIPS_SIM == _ABI32) || (_MIPS_SIM == _MIPS_SIM_ABI32)
#define __NR_Linux 4000
#define __NR_pfm_create_context         __NR_Linux+316
#elif (_MIPS_SIM == _ABI64) || (_MIPS_SIM == _MIPS_SIM_ABI64)
#define __NR_Linux 5000
#define __NR_pfm_create_context         __NR_Linux+275
#endif
#endif

#ifdef __powerpc__
#define __NR_pfm_create_context		309
#endif

#ifdef __crayx2
#define __NR_pfm_create_context		294
#endif

#define __NR_pfm_write_pmcs		(__NR_pfm_create_context+1)
#define __NR_pfm_write_pmds		(__NR_pfm_create_context+2)
#define __NR_pfm_read_pmds		(__NR_pfm_create_context+3)
#define __NR_pfm_load_context		(__NR_pfm_create_context+4)
#define __NR_pfm_start			(__NR_pfm_create_context+5)
#define __NR_pfm_stop			(__NR_pfm_create_context+6)
#define __NR_pfm_restart		(__NR_pfm_create_context+7)
#define __NR_pfm_create_evtsets		(__NR_pfm_create_context+8)
#define __NR_pfm_getinfo_evtsets	(__NR_pfm_create_context+9)
#define __NR_pfm_delete_evtsets		(__NR_pfm_create_context+10)
#define __NR_pfm_unload_context		(__NR_pfm_create_context+11)
#endif /* __NR_pfm_create_context */

#ifdef __cplusplus
};
#endif

#endif /* _PERFMON_H */
