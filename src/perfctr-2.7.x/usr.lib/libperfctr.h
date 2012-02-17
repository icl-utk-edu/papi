/* $Id: libperfctr.h,v 1.50 2007/10/06 13:02:07 mikpe Exp $
 * Library interface to Linux Performance-Monitoring Counters.
 *
 * Copyright (C) 1999-2007  Mikael Pettersson
 */

#ifndef __LIB_PERFCTR_H
#define __LIB_PERFCTR_H

/*
 * The kernel/user-space API structures are not suitable
 * for applications, so we need to provide some wrappers.
 * This being the case, the wrappers may as well be
 * compatible with perfctr-2.6.x.
 *
 * Import the kernel/user-space API definitions, but
 * rename some of them to avoid conflicts with the
 * (simulated) perfctr-2.6.x API.
 */
#define CONFIG_PERFCTR 1
#define vperfctr_control	vperfctr_control_kernel
#define perfctr_cpu_control	perfctr_cpu_control_kernel
#include <linux/perfctr.h>
#undef CONFIG_PERFCTR
#undef vperfctr_control
#undef perfctr_cpu_control

#include <stdint.h>

#if defined(__i386__) || defined(__x86_64__)

/* cpu_type values */
#define PERFCTR_X86_GENERIC	0	/* any x86 with rdtsc */
#define PERFCTR_X86_INTEL_P5	1	/* no rdpmc */
#define PERFCTR_X86_INTEL_P5MMX	2
#define PERFCTR_X86_INTEL_P6	3
#define PERFCTR_X86_INTEL_PII	4
#define PERFCTR_X86_INTEL_PIII	5
#define PERFCTR_X86_CYRIX_MII	6
#define PERFCTR_X86_WINCHIP_C6	7	/* no rdtsc */
#define PERFCTR_X86_WINCHIP_2	8	/* no rdtsc */
#define PERFCTR_X86_AMD_K7	9
#define PERFCTR_X86_VIA_C3	10	/* no pmc0 */
#define PERFCTR_X86_INTEL_P4	11	/* model 0 and 1 */
#define PERFCTR_X86_INTEL_P4M2	12	/* model 2 */
#define PERFCTR_X86_AMD_K8	13
#define PERFCTR_X86_INTEL_PENTM	14	/* Pentium M */
#define PERFCTR_X86_AMD_K8C	15	/* Revision C */
#define PERFCTR_X86_INTEL_P4M3	16	/* model 3 and above */

struct perfctr_cpu_control {
	unsigned int tsc_on;
	unsigned int nractrs;		/* # of a-mode counters */
	unsigned int nrictrs;		/* # of i-mode counters */
	unsigned int pmc_map[18];
	unsigned int evntsel[18];	/* one per counter, even on P5 */
	struct {
		unsigned int escr[18];
		unsigned int pebs_enable;	/* for replay tagging */
		unsigned int pebs_matrix_vert;	/* for replay tagging */
	} p4;
	int ireset[18];			/* < 0, for i-mode counters */
	unsigned int _reserved1;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

/* version number for user-visible CPU-specific data */
#define PERFCTR_CPU_VERSION	0x0500	/* 5.0 */

#endif	/* __i386__ || __x86_64__ */

#if defined(__powerpc64__) || defined(PPC64)

/* perfctr_info.cpu_type values */
#define PERFCTR_PPC64_GENERIC	0
#define PERFCTR_PPC64_POWER4	1
#define PERFCTR_PPC64_POWER4p	2
#define PERFCTR_PPC64_970 	3
#define PERFCTR_PPC64_POWER5	4
#define PERFCTR_PPC64_970MP	5

struct perfctr_cpu_control {
	unsigned int tsc_on;
	unsigned int nractrs;		/* # of a-mode counters */
	unsigned int nrictrs;		/* # of i-mode counters */
	unsigned int pmc_map[8];
	int ireset[8];			/* [0,0x7fffffff], for i-mode counters */
	struct {
		uint32_t mmcr0, mmcra;
		uint64_t mmcr1;
	} ppc64;
};

/* version number for user-visible CPU-specific data */
#define PERFCTR_CPU_VERSION	0	/* XXX: not yet cast in stone */

#elif defined(__powerpc__)

/* perfctr_info.cpu_type values */
#define PERFCTR_PPC_GENERIC	0
#define PERFCTR_PPC_604		1
#define PERFCTR_PPC_604e	2
#define PERFCTR_PPC_750		3
#define PERFCTR_PPC_7400	4
#define PERFCTR_PPC_7450	5

struct perfctr_cpu_control {
	unsigned int tsc_on;
	unsigned int nractrs;		/* # of a-mode counters */
	unsigned int nrictrs;		/* # of i-mode counters */
	unsigned int pmc_map[6];
	unsigned int evntsel[6];	/* one per counter, even on P5 */
	int ireset[6];			/* [0,0x7fffffff], for i-mode counters */
	struct {
		unsigned int mmcr0;	/* sans PMC{1,2}SEL */
		unsigned int mmcr2;	/* only THRESHMULT */
		/* IABR/DABR/BAMR not supported */
	} ppc;
	unsigned int _reserved1;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

/* version number for user-visible CPU-specific data */
#define PERFCTR_CPU_VERSION	0	/* XXX: not yet cast in stone */

#endif	/* __powerpc__ && !(__powerpc64__ || PPC64)*/

struct perfctr_info {
	unsigned int abi_version;
	char driver_version[32];
	unsigned int cpu_type;
	unsigned int cpu_features;
	unsigned int cpu_khz;
	unsigned int tsc_to_cpu_mult;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

struct perfctr_cpu_mask {
	unsigned int nrwords;
	unsigned int mask[1];	/* actually 'nrwords' */
};

/* abi_version values: Lower 16 bits contain the CPU data version, upper
   16 bits contain the API version. Each half has a major version in its
   upper 8 bits, and a minor version in its lower 8 bits. */
#define PERFCTR_API_VERSION	0x0600	/* 6.0 */
#define PERFCTR_ABI_VERSION	((PERFCTR_API_VERSION<<16)|PERFCTR_CPU_VERSION)

struct vperfctr_control {
	int si_signo;
	struct perfctr_cpu_control cpu_control;
	unsigned int preserve;
	unsigned int _reserved1;
	unsigned int _reserved2;
	unsigned int _reserved3;
	unsigned int _reserved4;
};

struct perfctr_cpus_info {	/* malloc():d, use free() */
    struct perfctr_cpu_mask *cpus;
    struct perfctr_cpu_mask *cpus_forbidden;
};

/*
 * Library-internal procedures.
 */
int _sys_perfctr_info(int fd, struct perfctr_info*, struct perfctr_cpu_mask*, struct perfctr_cpu_mask*);
int _sys_vperfctr_open(int fd, int tid, int creat);
int _sys_vperfctr_write_control(int fd, unsigned int cpu_type, const struct vperfctr_control*);
int _sys_vperfctr_read_control(int fd, unsigned int cpu_type, struct vperfctr_control*);
int _sys_vperfctr_read_sum(int fd, struct perfctr_sum_ctrs*);
int _sys_vperfctr_read_children(int fd, struct perfctr_sum_ctrs*);
int _sys_vperfctr_unlink(int fd);
int _sys_vperfctr_iresume(int fd);
int _perfctr_get_state_user_offset(void);

/*
 * Operations on the process' own virtual-mode perfctrs.
 */

int _vperfctr_open(int creat);
int _vperfctr_control(int fd, const struct vperfctr_control*); /* XXX: should take cpu_type too */
int _vperfctr_read_control(int fd, struct vperfctr_control*); /* XXX: should take cpu_type too */
int _vperfctr_read_sum(int fd, struct perfctr_sum_ctrs*);
int _vperfctr_read_children(int fd, struct perfctr_sum_ctrs*);

struct vperfctr;	/* opaque */

struct vperfctr *vperfctr_open(void);
int vperfctr_info(const struct vperfctr*, struct perfctr_info*);
struct perfctr_cpus_info *vperfctr_cpus_info(const struct vperfctr*);
unsigned long long vperfctr_read_tsc(const struct vperfctr*);
unsigned long long vperfctr_read_pmc(const struct vperfctr*, unsigned);
int vperfctr_read_ctrs(const struct vperfctr*, struct perfctr_sum_ctrs*);
int vperfctr_read_state(const struct vperfctr*, struct perfctr_sum_ctrs*,
			struct vperfctr_control*);
int vperfctr_control(struct vperfctr*, struct vperfctr_control*);
int vperfctr_stop(struct vperfctr*);
int vperfctr_is_running(const struct vperfctr*);
int vperfctr_iresume(const struct vperfctr*);
int vperfctr_unlink(const struct vperfctr*);
void vperfctr_close(struct vperfctr*);

/*
 * Operations on other processes' virtual-mode perfctrs.
 * (Preliminary, subject to change.)
 */

struct rvperfctr;	/* opaque */

struct rvperfctr *rvperfctr_open(int pid);
int rvperfctr_pid(const struct rvperfctr*);
int rvperfctr_info(const struct rvperfctr*, struct perfctr_info*);
int rvperfctr_read_ctrs(const struct rvperfctr*, struct perfctr_sum_ctrs*);
int rvperfctr_read_state(const struct rvperfctr*, struct perfctr_sum_ctrs*,
			 struct vperfctr_control*);
int rvperfctr_control(struct rvperfctr*, struct vperfctr_control*);
int rvperfctr_stop(struct rvperfctr*);
int rvperfctr_unlink(const struct rvperfctr*);
void rvperfctr_close(struct rvperfctr*);

/*
 * Operations on global-mode perfctrs.
 */
#if 0 /* disabled pending reimplementation of kernel support */

struct gperfctr;	/* opaque */

struct gperfctr *gperfctr_open(void);
void gperfctr_close(struct gperfctr*);
int gperfctr_control(const struct gperfctr*, struct gperfctr_cpu_control*);
int gperfctr_read(const struct gperfctr*, struct gperfctr_cpu_state*);
int gperfctr_stop(const struct gperfctr*);
int gperfctr_start(const struct gperfctr*, unsigned int interval_usec);
int gperfctr_info(const struct gperfctr*, struct perfctr_info*);
struct perfctr_cpus_info *gperfctr_cpus_info(const struct gperfctr*);
#endif

/*
 * Descriptions of the events available for different processor types.
 */

enum perfctr_unit_mask_type {
    perfctr_um_type_fixed,	/* one fixed (required) value */
    perfctr_um_type_exclusive,	/* exactly one of N values */
    perfctr_um_type_bitmask,	/* bitwise 'or' of N power-of-2 values */
};

struct perfctr_unit_mask_value {
    unsigned int value;
    const char *description;	/* [NAME:]text */
};

struct perfctr_unit_mask {
    unsigned short default_value;
    enum perfctr_unit_mask_type type:8;
    unsigned char nvalues;
    struct perfctr_unit_mask_value values[1/*nvalues*/];
};

struct perfctr_event {
    unsigned short evntsel;
    unsigned short counters_set; /* P4 force this to be CPU-specific */
    const struct perfctr_unit_mask *unit_mask;
    const char *name;
    const char *description;
};

struct perfctr_event_set {
    unsigned int cpu_type;
    const char *event_prefix;
    const struct perfctr_event_set *include;
    unsigned int nevents;
    const struct perfctr_event *events;
};

const struct perfctr_event_set *perfctr_cpu_event_set(unsigned int cpu_type);

/*
 * Miscellaneous operations.
 */

/* this checks the ABI between library and kernel -- it can also
   be used by applications operating on raw file descriptors */
int _perfctr_abi_check_fd(int fd, unsigned int user_abi_version);
static __inline__ int perfctr_abi_check_fd(int fd)
{
    return _perfctr_abi_check_fd(fd, PERFCTR_ABI_VERSION);
}

int perfctr_info(int fd, struct perfctr_info *info);
struct perfctr_cpus_info *perfctr_cpus_info(int fd);
unsigned int perfctr_info_nrctrs(const struct perfctr_info*);
const char *perfctr_info_cpu_name(const struct perfctr_info*);
void perfctr_info_print(const struct perfctr_info*);
void perfctr_cpus_info_print(const struct perfctr_cpus_info*);
void perfctr_cpu_control_print(const struct perfctr_cpu_control*);
unsigned int perfctr_linux_version_code(void);
#define PERFCTR_KERNEL_VERSION(v,p,s)	(((v) << 16) + ((p) << 8) + (s))

#endif /* __LIB_PERFCTR_H */
