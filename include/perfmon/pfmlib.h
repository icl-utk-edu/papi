/*
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on:
 * Copyright (c) 2001-2007 Hewlett-Packard Development Company, L.P.
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
#ifndef __PFMLIB_H__
#define __PFMLIB_H__

#ifdef __cplusplus
extern "C" {
#endif
#include <sys/types.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdio.h>

#define LIBPFM_VERSION		(4 << 16 | 0)
#define PFM_MAJ_VERSION(v)	((v)>>16)
#define PFM_MIN_VERSION(v)	((v) & 0xffff)

/*
 * ABI revision level
 */
#define LIBPFM_ABI_VERSION	0

/*
 * priv level mask (for dfl_plm)
 */
#define PFM_PLM0	0x01 /* kernel */
#define PFM_PLM1	0x02 /* not yet used */
#define PFM_PLM2	0x04 /* not yet used */
#define PFM_PLM3	0x08 /* priv level 3, 2, 1 (x86) */
#define PFM_PLMH	0x10 /* hypervisor */

/*
 * Performance Event Source
 *
 * The source is what is providing events.
 * It can be:
 * 	- Hardware Performance Monitoring Unit (PMU)
 * 	- a particular kernel subsystem
 *
 * Identifiers are guaranteed constant across libpfm revisions
 *
 * New sources must be added at the end before PFM_PMU_MAX
 */
typedef enum {
	PFM_PMU_NONE= 0,		/* no PMU */
	PFM_PMU_GEN_IA64,	 	/* Intel IA-64 architected PMU */
	PFM_PMU_ITANIUM,	 	/* Intel Itanium   */
	PFM_PMU_ITANIUM2,		/* Intel Itanium 2 */
	PFM_PMU_MONTECITO,		/* Intel Dual-Core Itanium 2 9000 */
	PFM_PMU_AMD64,			/* AMD AMD64 (obsolete) */
	PFM_PMU_I386_P6,		/* Intel PIII (P6 core) */
	PFM_PMU_INTEL_NETBURST,		/* Intel Netburst (Pentium 4) */
	PFM_PMU_INTEL_NETBURST_P,	/* Intel Netburst Prescott (Pentium 4) */
	PFM_PMU_COREDUO,		/* Intel Core Duo/Core Solo */
	PFM_PMU_I386_PM,		/* Intel Pentium M */
	PFM_PMU_INTEL_CORE,		/* Intel Core */
	PFM_PMU_INTEL_PPRO,		/* Intel Pentium Pro */
	PFM_PMU_INTEL_PII,		/* Intel Pentium II */
	PFM_PMU_INTEL_ATOM,		/* Intel Atom */
	PFM_PMU_INTEL_NHM,		/* Intel Nehalem core PMU */
	PFM_PMU_INTEL_NHM_EX,		/* Intel Nehalem-EX core PMU */
	PFM_PMU_INTEL_NHM_UNC,		/* Intel Nehalem uncore PMU */
	PFM_PMU_INTEL_X86_ARCH,		/* Intel X86 architectural PMU */

	PFM_PMU_MIPS_20KC,		/* MIPS 20KC */
	PFM_PMU_MIPS_24K,		/* MIPS 24K */
	PFM_PMU_MIPS_25KF,		/* MIPS 25KF */
	PFM_PMU_MIPS_34K,		/* MIPS 34K */
	PFM_PMU_MIPS_5KC,		/* MIPS 5KC */
	PFM_PMU_MIPS_74K,		/* MIPS 74K */
	PFM_PMU_MIPS_R10000,		/* MIPS R10000 */
	PFM_PMU_MIPS_R12000,		/* MIPS R12000 */
	PFM_PMU_MIPS_RM7000,		/* MIPS RM7000 */
	PFM_PMU_MIPS_RM9000,		/* MIPS RM9000 */
	PFM_PMU_MIPS_SB1,		/* MIPS SB1/SB1A */
	PFM_PMU_MIPS_VR5432,		/* MIPS VR5432 */
	PFM_PMU_MIPS_VR5500,		/* MIPS VR5500 */
	PFM_PMU_MIPS_ICE9A,		/* SiCortex ICE9A */
	PFM_PMU_MIPS_ICE9B,		/* SiCortex ICE9B */
	PFM_PMU_POWERPC,		/* POWERPC */
	PFM_PMU_CELL,			/* IBM CELL */

	PFM_PMU_SPARC_ULTRA12,		/* UltraSPARC I, II, IIi, and IIe */
	PFM_PMU_SPARC_ULTRA3,		/* UltraSPARC III */
	PFM_PMU_SPARC_ULTRA3I,		/* UltraSPARC IIIi and IIIi+ */
	PFM_PMU_SPARC_ULTRA3PLUS,	/* UltraSPARC III+ and IV */
	PFM_PMU_SPARC_ULTRA4PLUS,	/* UltraSPARC IV+ */
	PFM_PMU_SPARC_NIAGARA1,		/* Niagara-1 */
	PFM_PMU_SPARC_NIAGARA2,		/* Niagara-2 */

	PFM_PMU_PPC970,			/* IBM PowerPC 970(FX,GX) */
	PFM_PMU_PPC970MP,		/* IBM PowerPC 970MP */
	PFM_PMU_POWER3,			/* IBM POWER3 */
	PFM_PMU_POWER4,			/* IBM POWER4 */
	PFM_PMU_POWER5,			/* IBM POWER5 */
	PFM_PMU_POWER5p,		/* IBM POWER5+ */
	PFM_PMU_POWER6,			/* IBM POWER6 */
	PFM_PMU_POWER7,			/* IBM POWER7 */

	PFM_PMU_PERF_EVENT,		/* perf_event PMU */
	PFM_PMU_INTEL_WSM,		/* Intel Westmere single-socket (Clarkdale) */
	PFM_PMU_INTEL_WSM_DP,		/* Intel Westmere dual-socket (Westmere-EP, Gulftwon) */
	PFM_PMU_INTEL_WSM_UNC,		/* Intel Westmere uncore PMU */

	PFM_PMU_AMD64_K7,		/* AMD AMD64 K7 */
	PFM_PMU_AMD64_K8_REVB,		/* AMD AMD64 K8 RevB */
	PFM_PMU_AMD64_K8_REVC,		/* AMD AMD64 K8 RevC */
	PFM_PMU_AMD64_K8_REVD,		/* AMD AMD64 K8 RevD */
	PFM_PMU_AMD64_K8_REVE,		/* AMD AMD64 K8 RevE */
	PFM_PMU_AMD64_K8_REVF,		/* AMD AMD64 K8 RevF */
	PFM_PMU_AMD64_K8_REVG,		/* AMD AMD64 K8 RevG */
	PFM_PMU_AMD64_FAM10H_BARCELONA,	/* AMD AMD64 Fam10h Barcelona RevB */
	PFM_PMU_AMD64_FAM10H_SHANGHAI,	/* AMD AMD64 Fam10h Shanghai RevC  */
	PFM_PMU_AMD64_FAM10H_ISTANBUL,	/* AMD AMD64 Fam10h Istanbul RevD  */

	PFM_PMU_ARM_CORTEX_A8,		/* ARM Cortex A8 */
	PFM_PMU_ARM_CORTEX_A9,		/* ARM Cortex A9 */

	PFM_PMU_TORRENT,		/* IBM Torrent hub chip */

	PFM_PMU_INTEL_SNB,		/* Intel Sandy Bridge (single socket) */
	PFM_PMU_AMD64_FAM14H_BOBCAT,	/* AMD AMD64 Fam14h Bobcat */
	PFM_PMU_AMD64_FAM15H_INTERLAGOS,/* AMD AMD64 Fam15h Interlagos */

	PFM_PMU_INTEL_SNB_EP,		/* Intel SandyBridge EP */
	/* MUST ADD NEW PMU MODELS HERE */

	PFM_PMU_MAX			/* end marker */
} pfm_pmu_t;

typedef enum {
	PFM_PMU_TYPE_UNKNOWN=0,	/* unknown PMU type */
	PFM_PMU_TYPE_CORE,	/* processor core PMU */
	PFM_PMU_TYPE_UNCORE,	/* processor socket-level PMU */
	PFM_PMU_TYPE_OS_GENERIC,/* generic OS-provided PMU */
	PFM_PMU_TYPE_MAX
} pfm_pmu_type_t;

typedef enum {
	PFM_ATTR_NONE=0,	/* no attribute */
	PFM_ATTR_UMASK,		/* unit mask */
	PFM_ATTR_MOD_BOOL,	/* register modifier */
	PFM_ATTR_MOD_INTEGER,	/* register modifier */
	PFM_ATTR_RAW_UMASK,	/* raw umask (not user visible) */

	PFM_ATTR_MAX		/* end-marker */
} pfm_attr_t;

/*
 * define additional event data types beyond historic uint64
 * what else can fit in 64 bits?
 */
typedef enum {
	PFM_DTYPE_UNKNOWN=0,	/* unkown */
	PFM_DTYPE_UINT64,	/* uint64 */
	PFM_DTYPE_INT64,	/* int64 */
	PFM_DTYPE_DOUBLE,	/* IEEE double precision float */
	PFM_DTYPE_FIXED,	/* 32.32 fixed point */
	PFM_DTYPE_RATIO,	/* 32/32 integer ratio */
	PFM_DTYPE_CHAR8,	/* 8 char unterminated string */

	PFM_DTYPE_MAX		/* end-marker */
} pfm_dtype_t;

/*
 * event attribute control: which layer is controlling
 * the attribute could be PMU, OS APIs
 */
typedef enum {
	PFM_ATTR_CTRL_UNKNOWN = 0,	/* unknown */
	PFM_ATTR_CTRL_PMU,		/* PMU hardware */
	PFM_ATTR_CTRL_PERF_EVENT,	/* perf_events kernel interface */

	PFM_ATTR_CTRL_MAX
} pfm_attr_ctrl_t;

/*
 * OS layer
 * Used when querying event or attribute information
 */
typedef enum {
	PFM_OS_NONE = 0,	/* only PMU */
	PFM_OS_PERF_EVENT,	/* perf_events PMU attribute subset + PMU */
	PFM_OS_PERF_EVENT_EXT,	/* perf_events all attributes + PMU */

	PFM_OS_MAX,
} pfm_os_t;

/* SWIG doesn't deal well with anonymous nested structures */
#ifdef SWIG
#define SWIG_NAME(x) x
#else
#define SWIG_NAME(x)
#endif /* SWIG */

/*
 * special data type for libpfm error value used to help
 * with Python support and in particular for SWIG. By using
 * a specific type we can detect library calls and trap errors
 * in one SWIG statement as opposed to having to keep track of
 * each call individually. Programs can use 'int' safely for
 * the return value.
 */
typedef int pfm_err_t;		/* error if !PFM_SUCCESS */
typedef int os_err_t;		/* error if a syscall fails */

typedef struct {
	const char		*name;		/* event name */
	const char		*desc;		/* event description */
	size_t			size;		/* struct sizeof */
	pfm_pmu_t		pmu;		/* PMU identification */
	pfm_pmu_type_t		type;		/* PMU type */
	int			nevents;	/* how many events for this PMU */
	int			first_event;	/* opaque index of first event */
	int			max_encoding;	/* max number of uint64_t to encode an event */
	int			num_cntrs;	/* number of generic counters */
	int			num_fixed_cntrs;/* number of fixed counters */
	struct {
		unsigned int	is_present:1;	/* present on host system */
		unsigned int	is_dfl:1;	/* is architecture default PMU */
		unsigned int	reserved_bits:30;
	} SWIG_NAME(flags);
} pfm_pmu_info_t;

typedef struct {
	const char		*name;	/* event name */
	const char		*desc;	/* event description */
	const char		*equiv;	/* event is equivalent to */
	size_t			size;	/* struct sizeof */
	uint64_t		code;	/* event raw code (not encoding) */
	pfm_pmu_t		pmu;	/* which PMU */
	pfm_dtype_t		dtype;	/* data type of event value */
	int			idx;	/* unique event identifier */
	int			nattrs;	/* number of attributes */
	int			reserved; /* for future use */
	struct {
		unsigned int	is_precise:1;	/* precise sampling (Intel X86=PEBS) */
		unsigned int	reserved_bits:31;
	} SWIG_NAME(flags);
} pfm_event_info_t;

typedef struct {
	const char		*name;	/* attribute symbolic name */
	const char		*desc;	/* attribute description */
	const char		*equiv;	/* attribute is equivalent to */
	size_t			size;	/* struct sizeof */
	uint64_t		code;	/* attribute code */
	pfm_attr_t		type;	/* attribute type */
	int			idx;	/* attribute opaque index */
	pfm_attr_ctrl_t		ctrl;		/* what is providing attr */
	struct {
		unsigned int    is_dfl:1;	/* is default umask */
		unsigned int    is_precise:1;	/* Intel X86: supports PEBS */
		unsigned int	reserved_bits:30;
	} SWIG_NAME(flags);
	union {
		uint64_t	dfl_val64;	/* default 64-bit value */
		const char	*dfl_str;	/* default string value */
		int		dfl_bool;	/* default boolean value */
		int		dfl_int;	/* default integer value */
	} SWIG_NAME(defaults);
} pfm_event_attr_info_t;

/*
 * use with PFM_OS_NONE for pfm_get_os_event_encoding()
 */
typedef struct {
	uint64_t	*codes;		/* out/in: event codes array */
	char		**fstr;		/* out/in: fully qualified event string */
	size_t		size;		/* sizeof struct */
	int		count;		/* out/in: # of elements in array */
	int		idx;		/* out: unique event identifier */
} pfm_pmu_encode_arg_t;

#if __WORDSIZE == 64
#define PFM_PMU_INFO_ABI0	56
#define PFM_EVENT_INFO_ABI0	64
#define PFM_ATTR_INFO_ABI0	64

#define PFM_RAW_ENCODE_ABI0	32
#else
#define PFM_PMU_INFO_ABI0	44
#define PFM_EVENT_INFO_ABI0	48
#define PFM_ATTR_INFO_ABI0	48

#define PFM_RAW_ENCODE_ABI0	20
#endif


/*
 * initialization, configuration, errors
 */
extern pfm_err_t pfm_initialize(void);
extern void pfm_terminate(void);
extern const char *pfm_strerror(int code);
extern int pfm_get_version(void);

/*
 * PMU API
 */
extern pfm_err_t pfm_get_pmu_info(pfm_pmu_t pmu, pfm_pmu_info_t *output);

/*
 * event API
 */
extern int pfm_get_event_next(int idx);
extern int pfm_find_event(const char *str);
extern pfm_err_t pfm_get_event_info(int idx, pfm_os_t os, pfm_event_info_t *output);

/*
 * event encoding API
 *
 * content of args depends on value of os (refer to man page)
 */
extern pfm_err_t pfm_get_os_event_encoding(const char *str, int dfl_plm, pfm_os_t os, void *args);

/*
 * attribute API
 */
extern pfm_err_t pfm_get_event_attr_info(int eidx, int aidx, pfm_os_t os, pfm_event_attr_info_t *output);

/*
 * library validation API
 */
extern pfm_err_t pfm_pmu_validate(pfm_pmu_t pmu_id, FILE *fp);

/*
 * older encoding API
 */
extern pfm_err_t pfm_get_event_encoding(const char *str, int dfl_plm, char **fstr, int *idx, uint64_t **codes, int *count);

/*
 * error codes
 */
#define PFM_SUCCESS		0	/* success */
#define PFM_ERR_NOTSUPP		-1	/* function not supported */
#define PFM_ERR_INVAL		-2	/* invalid parameters */
#define PFM_ERR_NOINIT		-3	/* library was not initialized */
#define PFM_ERR_NOTFOUND	-4	/* event not found */
#define PFM_ERR_FEATCOMB	-5	/* invalid combination of features */
#define PFM_ERR_UMASK	 	-6	/* invalid or missing unit mask */
#define PFM_ERR_NOMEM	 	-7	/* out of memory */
#define PFM_ERR_ATTR		-8	/* invalid event attribute */
#define PFM_ERR_ATTR_VAL	-9	/* invalid event attribute value */
#define PFM_ERR_ATTR_SET	-10	/* attribute value already set */
#define PFM_ERR_TOOMANY		-11	/* too many parameters */
#define PFM_ERR_TOOSMALL	-12	/* parameter is too small */

/*
 * event, attribute iterators
 * must be used because no guarante indexes are contiguous
 *
 * for pmu, simply iterate over pfm_pmu_t enum and use
 * pfm_get_pmu_info() and the is_present field
 */
#define pfm_for_each_event_attr(x, z) \
	for((x)=0; (x) < (z)->nattrs; (x) = (x)+1)

#define pfm_for_all_pmus(x) \
	for((x)= 0 ; (x) < PFM_PMU_MAX; (x)++)

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_H__ */
