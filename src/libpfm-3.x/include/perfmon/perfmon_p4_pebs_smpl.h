/*
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 *               Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file implements the PEBS sampling format for 32-bit
 * Intel Pentium 4/Xeon processors. Not to be used with Intel EM64T
 * processors.
 */
#ifndef __PERFMON_P4_PEBS_SMPL_H__
#define __PERFMON_P4_PEBS_SMPL_H__ 1

#ifdef __cplusplus
extern "C" {
#endif

#include <perfmon/perfmon.h>

#define PFM_P4_PEBS_SMPL_UUID { \
	0x0d, 0x85, 0x91, 0xe7, 0x49, 0x3f, 0x49, 0xae,\
	0x8c, 0xfc, 0xe8, 0xb9, 0x33, 0xe4, 0xeb, 0x8b}

/*
 * format specific parameters (passed at context creation)
 */
typedef struct {
	size_t		buf_size;	/* size of the buffer in bytes */
	size_t		intr_thres;	/* index of interrupt threshold entry */
	uint32_t	flags;		/* buffer specific flags */
	uint64_t	cnt_reset;	/* counter reset value */
	uint32_t	res1;		/* for future use */
	uint64_t	reserved[2];	/* for future use */
} pfm_p4_pebs_smpl_arg_t;

/*
 * DS Save Area as described in section 15.10.5
 */
typedef struct {
	uint32_t bts_buf_base;
	uint32_t bts_index;
	uint32_t bts_abs_max;
	uint32_t bts_intr_thres;
	uint32_t pebs_buf_base;
	uint32_t pebs_index;
	uint32_t pebs_abs_max;
	uint32_t pebs_intr_thres;
	uint64_t pebs_cnt_reset;
} pfm_p4_ds_area_t;


/*
 * This header is at the beginning of the sampling buffer returned to the user.
 *
 * Because of PEBS alignement constraints, the actual PEBS buffer area does
 * not necessarily begin right after the header. The hdr_start_offs must be
 * used to compute the first byte of the buffer. The offset is defined as
 * the number of bytes between the end of the header and the beginning of
 * the buffer. As such the formula is:
 * 	actual_buffer = (unsigned long)(hdr+1)+hdr->hdr_start_offs
 */
typedef struct {
	uint64_t		hdr_overflows;	/* #overflows for buffer */
	size_t			hdr_buf_size;	/* bytes in the buffer */
	size_t			hdr_start_offs; /* actual buffer start offset */
	uint32_t		hdr_version;	/* smpl format version */
	uint64_t		hdr_res[3];	/* for future use */
	pfm_p4_ds_area_t	hdr_ds;		/* DS management Area */
} pfm_p4_pebs_smpl_hdr_t;

/*
 * PEBS record format as described in section 15.10.6
 */
typedef struct {
	uint32_t eflags;
	uint32_t ip;
	uint32_t eax;
	uint32_t ebx;
	uint32_t ecx;
	uint32_t edx;
	uint32_t esi;
	uint32_t edi;
	uint32_t ebp;
	uint32_t esp;
} pfm_p4_pebs_smpl_entry_t;

#define PFM_P4_PEBS_SMPL_VERSION_MAJ 1U
#define PFM_P4_PEBS_SMPL_VERSION_MIN 0U
#define PFM_P4_PEBS_SMPL_VERSION (((PFM_P4_PEBS_SMPL_VERSION_MAJ&0xffff)<<16)|\
				   (PFM_P4_PEBS_SMPL_VERSION_MIN & 0xffff))

#ifdef __cplusplus
};
#endif

#endif /* __PERFMON_P4_PEBS_SMPL_H__ */
