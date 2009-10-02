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
#include <inttypes.h>

#define LIBPFM_VERSION		(4 << 16 | 0)
#define PFM_MAJ_VERSION(v)	((v)>>16)
#define PFM_MIN_VERSION(v)	((v) & 0xffff)

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
	PFM_PMU_AMD64,			/* AMD AMD64 (K7, K8, Fam 10h) */
	PFM_PMU_I386_P6,		/* Intel PIII (P6 core) */
	PFM_PMU_PENTIUM4,		/* Intel Pentium4/Xeon/EM64T */
	PFM_PMU_COREDUO,		/* Intel Core Duo/Core Solo */
	PFM_PMU_I386_PM,		/* Intel Pentium M */
	PFM_PMU_INTEL_CORE,		/* Intel Core */
	PFM_PMU_INTEL_PPRO,		/* Intel Pentium Pro */
	PFM_PMU_INTEL_PII,		/* Intel Pentium II */
	PFM_PMU_INTEL_ATOM,		/* Intel Atom */
	PFM_PMU_INTEL_NHM,		/* Intel Nehalem core PMU */
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
	PFM_PMU_CRAYX2,			/* Cray X2 */
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

	/* MUST ADD NEW PMU MODELS HERE */

	PFM_PMU_MAX			/* end marker */
} pfm_pmu_t;

typedef enum {
	PFM_ATTR_NONE=0,	/* no attribute */
	PFM_ATTR_UMASK,		/* unit mask */
	PFM_ATTR_MOD_BOOL,	/* register modifier */
	PFM_ATTR_MOD_INTEGER,	/* register modifier */

	PFM_ATTR_MAX		/* end-marker */
} pfm_attr_t;

/*
 * special data type for libpfm error value used to help
 * with Python support and in particular for SWIG. By using
 * a specific type we can detect library calls and trap errors
 * in one SWIG statement as opposed to having to keep track of
 * each call individually. Programs can use 'int' safely for
 * the return value.
 */
typedef int pfm_err_t;		/* error if !PFM_SUCCESS */

/*
 * initialization, configuration
 */
extern pfm_err_t pfm_initialize(void);
extern const char *pfm_strerror(int code);
extern int pfm_get_version(void);

/*
 * PMU API
 */
extern int pfm_pmu_present(pfm_pmu_t p);
extern const char *pfm_get_pmu_desc(pfm_pmu_t pmu);
extern const char *pfm_get_pmu_name(pfm_pmu_t pmu);

/*
 * event API
 */
extern int pfm_get_nevents(void);
extern int pfm_get_event_first(void);
extern int pfm_get_event_next(int idx);
extern const char *pfm_get_event_name(int idx);
extern const char *pfm_get_event_desc(int idx);
extern int pfm_find_event(const char *str);
extern pfm_pmu_t pfm_get_event_pmu(int idx);
extern pfm_err_t pfm_get_event_code(int idx, uint64_t *code);
extern pfm_err_t pfm_get_event_encoding(const char *str, int *idx, uint64_t **codes, int *count);

/*
 * event attribute API
 */
extern int pfm_get_event_nattrs(int idx);
extern const char *pfm_get_event_attr_name(int idx, int attr_idx);
extern const char *pfm_get_event_attr_desc(int idx, int attr_idx);
extern pfm_err_t pfm_get_event_attr_code(int idx, int attr_idx, uint64_t *code);
extern pfm_attr_t pfm_get_event_attr_type(int idx, int attr_idx);


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
#define PFM_ERR_ATTR_SET	-10	/* attribute value hardcoded */
#define PFM_ERR_TOOMANY		-11	/* too many parameters */
#define PFM_ERR_TOOSMALL	-12	/* parameter is too small */

/*
 * event, attribute iterators
 * must be used because no guarante indexes are contiguous
 *
 * for pmu, simply iterate over pfm_pmu_t enum and use
 * pfm_pmu_present().
 */
#define pfm_for_each_event(x) \
	for((x)=pfm_get_event_first(); (x) != -1; (x) = pfm_get_event_next((x)))

#define pfm_for_each_event_attr(x, z) \
	for((x)=0; (x) < pfm_get_event_nattrs((z)); (x) = (x)+1)

#ifdef __cplusplus /* extern C */
}
#endif

#endif /* __PFMLIB_H__ */
