/*
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 *               Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file implements the PEBS sampling format for 64-bit
 * Intel EM64T Pentium 4/Xeon processors. Not to be used with
 * 32-bit Intel processors.
 */
#ifndef __PERFMON_EM64T_PEBS_SMPL_H__
#define __PERFMON_EM64T_PEBS_SMPL_H__ 1

#ifdef __cplusplus
extern "C" {
#endif

#include <perfmon/perfmon.h>

#define PFM_EM64T_PEBS_SMPL_UUID { \
	0x36, 0xbe, 0x97, 0x94, 0x1f, 0xbf, 0x41, 0xdf,\
	0xb4, 0x63, 0x10, 0x62, 0xeb, 0x72, 0x9b, 0xad}

/*
 * format specific parameters (passed at context creation)
 *
 * intr_thres: index from start of buffer of entry where the
 * PMU interrupt must be triggered. It must be several samples
 * short of the end of the buffer.
 */
typedef struct {
	uint64_t	buf_size;	/* size of the buffer in bytes */
	uint64_t	intr_thres;	/* index of interrupt threshold entry */
	uint32_t	flags;		/* buffer specific flags */
	uint64_t	cnt_reset;	/* counter reset value */
	uint32_t	res1;		/* for future use */
	uint64_t	reserved[2];	/* for future use */
} pfm_em64t_pebs_smpl_arg_t;

/*
 * DS Save Area as described in section 15.10.5
 * for 32-bit but extended to 64-bit
 */
typedef struct {
	uint64_t bts_buf_base;
	uint64_t bts_index;
	uint64_t bts_abs_max;
	uint64_t bts_intr_thres;
	uint64_t pebs_buf_base;
	uint64_t pebs_index;
	uint64_t pebs_abs_max;
	uint64_t pebs_intr_thres;
	uint64_t pebs_cnt_reset;
} pfm_em64t_ds_area_t;

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
	uint64_t		hdr_buf_size;	/* bytes in the buffer */
	uint64_t		hdr_start_offs; /* actual buffer start offset */
	uint32_t		hdr_version;	/* smpl format version */
	uint64_t		hdr_res[3];	/* for future use */
	pfm_em64t_ds_area_t	hdr_ds;		/* DS management Area */
} pfm_em64t_pebs_smpl_hdr_t;

/*
 * EM64T PEBS record format as described in
 * http://www.intel.com/technology/64bitextensions/30083502.pdf
 */
typedef struct {
	uint64_t eflags;
	uint64_t ip;
	uint64_t eax;
	uint64_t ebx;
	uint64_t ecx;
	uint64_t edx;
	uint64_t esi;
	uint64_t edi;
	uint64_t ebp;
	uint64_t esp;
	uint64_t r8;
	uint64_t r9;
	uint64_t r10;
	uint64_t r11;
	uint64_t r12;
	uint64_t r13;
	uint64_t r14;
	uint64_t r15;
} pfm_em64t_pebs_smpl_entry_t;

#define PFM_EM64T_PEBS_SMPL_VERSION_MAJ 1U
#define PFM_EM64T_PEBS_SMPL_VERSION_MIN 0U
#define PFM_EM64T_PEBS_SMPL_VERSION (((PFM_EM64T_PEBS_SMPL_VERSION_MAJ&0xffff)<<16)|\
				   (PFM_EM64T_PEBS_SMPL_VERSION_MIN & 0xffff))

#ifdef __cplusplus
};
#endif

#endif /* __PERFMON_EM64T_PEBS_SMPL_H__ */
