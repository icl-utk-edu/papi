/* $Id: marshal.c,v 1.6.2.7 2008/06/22 12:48:56 mikpe Exp $
 * Performance-monitoring counters driver.
 * Structure marshalling support.
 *
 * Copyright (C) 2003-2008  Mikael Pettersson
 */
#ifdef __KERNEL__
#include <linux/version.h>
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#include <linux/config.h>
#endif
struct inode;
#include <linux/sched.h>
#include <linux/perfctr.h>
#include <linux/errno.h>
#include <linux/stddef.h>
#include <linux/string.h>
#include <asm/uaccess.h>
#else	/* !__KERNEL__ */
#define CONFIG_KPERFCTR
#include <linux/perfctr.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <stddef.h>
#include <string.h>
#define put_user(w, p)	(*(p) = (w), 0)
#define get_user(w, p)	((w) = *(p), 0)
#endif	/* !__KERNEL__ */

#include "marshal.h"

/****************************************************************
 *								*
 * Struct encoding support.					*
 *								*
 ****************************************************************/

static void stream_write(struct perfctr_marshal_stream *stream, unsigned int word)
{
	if( !stream->error ) {
		if( stream->pos >= stream->size )
			stream->error = -EOVERFLOW;
		else if( put_user(word, &stream->buffer[stream->pos]) )
			stream->error = -EFAULT;
	}
	++stream->pos;
}

static void encode_field(const void *address,
			 const struct perfctr_field_desc *field,
			 struct perfctr_marshal_stream *stream)
{
	unsigned int base_type = PERFCTR_TYPE_BASE(field->type);
	unsigned int nr_items = PERFCTR_TYPE_NRITEMS(field->type);
	unsigned int tag = field->tag;
	const char *pointer = (const char*)address + field->offset;
	unsigned int uint32_val;
	union {
		unsigned long long ull;
		unsigned int ui[2];
	} uint64_val;
	unsigned int i = 0;

	do {
		if( base_type == PERFCTR_TYPE_UINT64 ) {
			uint64_val.ull = *(unsigned long long*)pointer;
			pointer += sizeof(long long);
			if( !uint64_val.ull )
				continue;
			stream_write(stream, PERFCTR_HEADER(PERFCTR_HEADER_UINT64, tag, i));
			stream_write(stream, uint64_val.ui[0]);
			stream_write(stream, uint64_val.ui[1]);
		} else {		/* PERFCTR_TYPE_BYTES4 */
			memcpy(&uint32_val, pointer, sizeof(int));
			pointer += sizeof(int);	
			if( !uint32_val )
				continue;
			stream_write(stream, PERFCTR_HEADER(PERFCTR_HEADER_UINT32, tag, i));
			stream_write(stream, uint32_val);
		}
	} while( ++i < nr_items );
}

void perfctr_encode_struct(const void *address,
			   const struct perfctr_struct_desc *sdesc,
			   struct perfctr_marshal_stream *stream)
{
	unsigned int i;

	for(i = 0; i < sdesc->nrfields; ++i)
		encode_field(address, &sdesc->fields[i], stream);
	for(i = 0; i < sdesc->nrsubs; ++i) {
		const struct perfctr_sub_struct_desc *sub = &sdesc->subs[i];
		perfctr_encode_struct((char*)address + sub->offset, sub->sdesc, stream);
	}
}

/****************************************************************
 *								*
 * Struct decoding support.					*
 *								*
 ****************************************************************/

static int stream_read(struct perfctr_marshal_stream *stream, unsigned int *word)
{
	if( stream->pos >= stream->size )
		return 0;
	if( get_user(*word, &stream->buffer[stream->pos]) )
		return -EFAULT;
	++stream->pos;
	return 1;
}

static const struct perfctr_field_desc*
find_field(unsigned int *struct_offset,
	   const struct perfctr_struct_desc *sdesc,
	   unsigned int tag)
{
	unsigned int low, high, mid, i;
	const struct perfctr_field_desc *field;
	const struct perfctr_sub_struct_desc *sub;

	low = 0;
	high = sdesc->nrfields;	/* [low,high[ */
	while( low < high ) {
		mid = (low + high) / 2;
		field = &sdesc->fields[mid];
		if( field->tag == tag )
			return field;
		if( field->tag < tag )
			low = mid + 1;
		else
			high = mid;
	}
	for(i = 0; i < sdesc->nrsubs; ++i) {
		sub = &sdesc->subs[i];
		field = find_field(struct_offset, sub->sdesc, tag);
		if( field ) {
			*struct_offset += sub->offset;
			return field;
		}
	}
	return 0;
}

int perfctr_decode_struct(void *address,
			  const struct perfctr_struct_desc *sdesc,
			  struct perfctr_marshal_stream *stream)
{
	unsigned int header;
	int err;
	const struct perfctr_field_desc *field;
	unsigned int struct_offset;
	union {
		unsigned long long ull;
		unsigned int ui[2];
	} val;
	char *target;
	unsigned int itemnr;

	for(;;) {
		err = stream_read(stream, &header);
		if( err <= 0 )
			return err;
		struct_offset = 0;
		field = find_field(&struct_offset, sdesc, PERFCTR_HEADER_TAG(header));
		if( !field )
			goto err_eproto;
		/* a 64-bit datum must have a 64-bit target field */
		if( PERFCTR_HEADER_TYPE(header) != PERFCTR_HEADER_UINT32 &&
		    PERFCTR_TYPE_BASE(field->type) != PERFCTR_TYPE_UINT64 )
			goto err_eproto;
		err = stream_read(stream, &val.ui[0]);
		if( err <= 0 )
			goto err_err;
		target = (char*)address + struct_offset + field->offset;
		itemnr = PERFCTR_HEADER_ITEMNR(header);
		if( itemnr >= PERFCTR_TYPE_NRITEMS(field->type) )
			goto err_eproto;
		if( PERFCTR_TYPE_BASE(field->type) == PERFCTR_TYPE_UINT64 ) {
			/* a 64-bit field must have a 64-bit datum */
			if( PERFCTR_HEADER_TYPE(header) == PERFCTR_HEADER_UINT32 )
				goto err_eproto;
			err = stream_read(stream, &val.ui[1]);
			if( err <= 0 )
				goto err_err;
			((unsigned long long*)target)[itemnr] = val.ull;
		} else
			memcpy(&((unsigned int*)target)[itemnr], &val.ui[0], sizeof(int));
	}
 err_err:	/* err ? err : -EPROTO */
	if( err )
		return err;
 err_eproto:	/* saves object code over inlining it */
	return -EPROTO;
}

/****************************************************************
 *								*
 * Structure descriptors.					*
 *								*
 ****************************************************************/

#undef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define STRUCT_ARRAY_SIZE(TYPE, MEMBER) ARRAY_SIZE(((TYPE*)0)->MEMBER)

#if defined(__i386__) || defined(__x86_64__) || defined(__powerpc__) || defined(__arm__)

#define PERFCTR_TAG_CPU_CONTROL_TSC_ON	32
#define PERFCTR_TAG_CPU_CONTROL_NRACTRS	33
#define PERFCTR_TAG_CPU_CONTROL_NRICTRS	34
#define PERFCTR_TAG_CPU_CONTROL_PMC_MAP	35
#define PERFCTR_TAG_CPU_CONTROL_EVNTSEL	36
#define PERFCTR_TAG_CPU_CONTROL_IRESET	37
/* 38-40 are arch-specific, see below */
#define PERFCTR_TAG_CPU_CONTROL_RSVD1	41
#define PERFCTR_TAG_CPU_CONTROL_RSVD2	42
#define PERFCTR_TAG_CPU_CONTROL_RSVD3	43
#define PERFCTR_TAG_CPU_CONTROL_RSVD4	44
#define PERFCTR_CPU_CONTROL_NRFIELDS_0	(7 + STRUCT_ARRAY_SIZE(struct perfctr_cpu_control, pmc_map) + STRUCT_ARRAY_SIZE(struct perfctr_cpu_control, evntsel) + STRUCT_ARRAY_SIZE(struct perfctr_cpu_control, ireset))

#if defined(__i386__) || defined(__x86_64__)
#define PERFCTR_TAG_CPU_CONTROL_P4_ESCR	38
#define PERFCTR_TAG_CPU_CONTROL_P4_PE	39
#define PERFCTR_TAG_CPU_CONTROL_P4_PMV	40
#define PERFCTR_CPU_CONTROL_NRFIELDS_1	(2 + STRUCT_ARRAY_SIZE(struct perfctr_cpu_control, p4.escr))
#endif	/* __i386__ || __x86_64__ */

#if defined(__powerpc__)
#define PERFCTR_TAG_CPU_CONTROL_PPC_MMCR0	38
#define PERFCTR_TAG_CPU_CONTROL_PPC_MMCR2	39
/* 40: unused */
#define PERFCTR_CPU_CONTROL_NRFIELDS_1	2
#endif	/* __powerpc__ */

#if defined(__arm__)
#define PERFCTR_CPU_CONTROL_NRFIELDS_1	0
#endif

#define PERFCTR_CPU_CONTROL_NRFIELDS	(PERFCTR_CPU_CONTROL_NRFIELDS_0 + PERFCTR_CPU_CONTROL_NRFIELDS_1)

#define PERFCTR_TAG_SUM_CTRS_TSC	48
#define PERFCTR_TAG_SUM_CTRS_PMC	49
#define PERFCTR_SUM_CTRS_NRFIELDS	(1 + STRUCT_ARRAY_SIZE(struct perfctr_sum_ctrs, pmc))

static const struct perfctr_field_desc perfctr_sum_ctrs_fields[] = {
	{ .offset = offsetof(struct perfctr_sum_ctrs, tsc),
	  .tag = PERFCTR_TAG_SUM_CTRS_TSC,
	  .type = PERFCTR_TYPE_UINT64 },
	{ .offset = offsetof(struct perfctr_sum_ctrs, pmc),
	  .tag = PERFCTR_TAG_SUM_CTRS_PMC,
	  .type = PERFCTR_TYPE_ARRAY(STRUCT_ARRAY_SIZE(struct perfctr_sum_ctrs,pmc),
				     PERFCTR_TYPE_UINT64) },
};

const struct perfctr_struct_desc perfctr_sum_ctrs_sdesc = {
	.total_sizeof = sizeof(struct perfctr_sum_ctrs),
	.total_nrfields = PERFCTR_SUM_CTRS_NRFIELDS,
	.nrfields = ARRAY_SIZE(perfctr_sum_ctrs_fields),
	.fields = perfctr_sum_ctrs_fields,
};

static const struct perfctr_field_desc perfctr_cpu_control_fields[] = {
	{ .offset = offsetof(struct perfctr_cpu_control, tsc_on),
	  .tag = PERFCTR_TAG_CPU_CONTROL_TSC_ON,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_cpu_control, nractrs),
	  .tag = PERFCTR_TAG_CPU_CONTROL_NRACTRS,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_cpu_control, nrictrs),
	  .tag = PERFCTR_TAG_CPU_CONTROL_NRICTRS,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_cpu_control, pmc_map),
	  .tag = PERFCTR_TAG_CPU_CONTROL_PMC_MAP,
	  .type = PERFCTR_TYPE_ARRAY(STRUCT_ARRAY_SIZE(struct perfctr_cpu_control,pmc_map),
				     PERFCTR_TYPE_BYTES4) },
	{ .offset = offsetof(struct perfctr_cpu_control, evntsel),
	  .tag = PERFCTR_TAG_CPU_CONTROL_EVNTSEL,
	  .type = PERFCTR_TYPE_ARRAY(STRUCT_ARRAY_SIZE(struct perfctr_cpu_control,evntsel),
				     PERFCTR_TYPE_BYTES4) },
	{ .offset = offsetof(struct perfctr_cpu_control, ireset),
	  .tag = PERFCTR_TAG_CPU_CONTROL_IRESET,
	  .type = PERFCTR_TYPE_ARRAY(STRUCT_ARRAY_SIZE(struct perfctr_cpu_control,ireset),
				     PERFCTR_TYPE_BYTES4) },
#if defined(__i386__) || defined(__x86_64__)
	{ .offset = offsetof(struct perfctr_cpu_control, p4.escr),
	  .tag = PERFCTR_TAG_CPU_CONTROL_P4_ESCR,
	  .type = PERFCTR_TYPE_ARRAY(STRUCT_ARRAY_SIZE(struct perfctr_cpu_control,p4.escr),
				     PERFCTR_TYPE_BYTES4) },
	{ .offset = offsetof(struct perfctr_cpu_control, p4.pebs_enable),
	  .tag = PERFCTR_TAG_CPU_CONTROL_P4_PE,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_cpu_control, p4.pebs_matrix_vert),
	  .tag = PERFCTR_TAG_CPU_CONTROL_P4_PMV,
	  .type = PERFCTR_TYPE_BYTES4 },
#endif	/* __i386__ || __x86_64__ */
#if defined(__powerpc__)
	{ .offset = offsetof(struct perfctr_cpu_control, ppc.mmcr0),
	  .tag = PERFCTR_TAG_CPU_CONTROL_PPC_MMCR0,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_cpu_control, ppc.mmcr2),
	  .tag = PERFCTR_TAG_CPU_CONTROL_PPC_MMCR2,
	  .type = PERFCTR_TYPE_BYTES4 },
#endif	/* __powerpc__ */
	{ .offset = offsetof(struct perfctr_cpu_control, _reserved1),
	  .tag = PERFCTR_TAG_CPU_CONTROL_RSVD1,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_cpu_control, _reserved2),
	  .tag = PERFCTR_TAG_CPU_CONTROL_RSVD2,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_cpu_control, _reserved3),
	  .tag = PERFCTR_TAG_CPU_CONTROL_RSVD3,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_cpu_control, _reserved4),
	  .tag = PERFCTR_TAG_CPU_CONTROL_RSVD4,
	  .type = PERFCTR_TYPE_BYTES4 },
};

const struct perfctr_struct_desc perfctr_cpu_control_sdesc = {
	.total_sizeof = sizeof(struct perfctr_cpu_control),
	.total_nrfields = PERFCTR_CPU_CONTROL_NRFIELDS,
	.nrfields = ARRAY_SIZE(perfctr_cpu_control_fields),
	.fields = perfctr_cpu_control_fields,
};

#endif	/* __i386__ || __x86_64__ || __powerpc__ */

#define PERFCTR_TAG_INFO_ABI_VERSION		0
#define PERFCTR_TAG_INFO_DRIVER_VERSION		1
#define PERFCTR_TAG_INFO_CPU_TYPE		2
#define PERFCTR_TAG_INFO_CPU_FEATURES		3
#define PERFCTR_TAG_INFO_CPU_KHZ		4
#define PERFCTR_TAG_INFO_TSC_TO_CPU_MULT	5
#define PERFCTR_TAG_INFO_RSVD2			6
#define PERFCTR_TAG_INFO_RSVD3			7
#define PERFCTR_TAG_INFO_RSVD4			8
#define PERFCTR_INFO_NRFIELDS	(8 + sizeof(((struct perfctr_info*)0)->driver_version)/sizeof(int))

#define VPERFCTR_TAG_CONTROL_SIGNO		9
#define VPERFCTR_TAG_CONTROL_PRESERVE		10
#define VPERFCTR_TAG_CONTROL_FLAGS		11
#define VPERFCTR_TAG_CONTROL_RSVD2		12
#define VPERFCTR_TAG_CONTROL_RSVD3		13
#define VPERFCTR_TAG_CONTROL_RSVD4		14
#define VPERFCTR_CONTROL_NRFIELDS		(6 + PERFCTR_CPU_CONTROL_NRFIELDS)

#define GPERFCTR_TAG_CPU_CONTROL_CPU		15
#define GPERFCTR_TAG_CPU_CONTROL_RSVD1		16
#define GPERFCTR_TAG_CPU_CONTROL_RSVD2		17
#define GPERFCTR_TAG_CPU_CONTROL_RSVD3		18
#define GPERFCTR_TAG_CPU_CONTROL_RSVD4		19
#define GPERFCTR_CPU_CONTROL_NRFIELDS		(5 + PERFCTR_CPU_CONTROL_NRFIELDS)

#define GPERFCTR_TAG_CPU_STATE_CPU		20
#define GPERFCTR_TAG_CPU_STATE_RSVD1		21
#define GPERFCTR_TAG_CPU_STATE_RSVD2		22
#define GPERFCTR_TAG_CPU_STATE_RSVD3		23
#define GPERFCTR_TAG_CPU_STATE_RSVD4		24
#define GPERFCTR_CPU_STATE_ONLY_CPU_NRFIELDS	5
#define GPERFCTR_CPU_STATE_NRFIELDS	(GPERFCTR_CPU_STATE_ONLY_CPU_NRFIELDS + PERFCTR_CPU_CONTROL_NRFIELDS + PERFCTR_SUM_CTRS_NRFIELDS)

static const struct perfctr_field_desc perfctr_info_fields[] = {
	{ .offset = offsetof(struct perfctr_info, abi_version),
	  .tag = PERFCTR_TAG_INFO_ABI_VERSION,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_info, driver_version),
	  .tag = PERFCTR_TAG_INFO_DRIVER_VERSION,
	  .type = PERFCTR_TYPE_ARRAY(sizeof(((struct perfctr_info*)0)->driver_version)/sizeof(int), PERFCTR_TYPE_BYTES4) },
	{ .offset = offsetof(struct perfctr_info, cpu_type),
	  .tag = PERFCTR_TAG_INFO_CPU_TYPE,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_info, cpu_features),
	  .tag = PERFCTR_TAG_INFO_CPU_FEATURES,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_info, cpu_khz),
	  .tag = PERFCTR_TAG_INFO_CPU_KHZ,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_info, tsc_to_cpu_mult),
	  .tag = PERFCTR_TAG_INFO_TSC_TO_CPU_MULT,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_info, _reserved2),
	  .tag = PERFCTR_TAG_INFO_RSVD2,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_info, _reserved3),
	  .tag = PERFCTR_TAG_INFO_RSVD3,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct perfctr_info, _reserved4),
	  .tag = PERFCTR_TAG_INFO_RSVD4,
	  .type = PERFCTR_TYPE_BYTES4 },
};

const struct perfctr_struct_desc perfctr_info_sdesc = {
	.total_sizeof = sizeof(struct perfctr_info),
	.total_nrfields = PERFCTR_INFO_NRFIELDS,
	.nrfields = ARRAY_SIZE(perfctr_info_fields),
	.fields = perfctr_info_fields,
};

#if defined(CONFIG_PERFCTR_VIRTUAL) || !defined(__KERNEL__)
static const struct perfctr_field_desc vperfctr_control_fields[] = {
	{ .offset = offsetof(struct vperfctr_control, si_signo),
	  .tag = VPERFCTR_TAG_CONTROL_SIGNO,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct vperfctr_control, preserve),
	  .tag = VPERFCTR_TAG_CONTROL_PRESERVE,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct vperfctr_control, flags),
	  .tag = VPERFCTR_TAG_CONTROL_FLAGS,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct vperfctr_control, _reserved2),
	  .tag = VPERFCTR_TAG_CONTROL_RSVD2,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct vperfctr_control, _reserved3),
	  .tag = VPERFCTR_TAG_CONTROL_RSVD3,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct vperfctr_control, _reserved4),
	  .tag = VPERFCTR_TAG_CONTROL_RSVD4,
	  .type = PERFCTR_TYPE_BYTES4 },
};

static const struct perfctr_sub_struct_desc vperfctr_control_subs[] = {
	{ .offset = offsetof(struct vperfctr_control, cpu_control),
	  .sdesc = &perfctr_cpu_control_sdesc },
};

const struct perfctr_struct_desc vperfctr_control_sdesc = {
	.total_sizeof = sizeof(struct vperfctr_control),
	.total_nrfields = VPERFCTR_CONTROL_NRFIELDS,
	.nrfields = ARRAY_SIZE(vperfctr_control_fields),
	.fields = vperfctr_control_fields,
	.nrsubs = ARRAY_SIZE(vperfctr_control_subs),
	.subs = vperfctr_control_subs,
};
#endif	/* CONFIG_PERFCTR_VIRTUAL || !__KERNEL__ */

#if defined(CONFIG_PERFCTR_GLOBAL) || !defined(__KERNEL__)
static const struct perfctr_field_desc gperfctr_cpu_control_fields[] = {
	{ .offset = offsetof(struct gperfctr_cpu_control, cpu),
	  .tag = GPERFCTR_TAG_CPU_CONTROL_CPU,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct gperfctr_cpu_control, _reserved1),
	  .tag = GPERFCTR_TAG_CPU_CONTROL_RSVD1,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct gperfctr_cpu_control, _reserved2),
	  .tag = GPERFCTR_TAG_CPU_CONTROL_RSVD2,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct gperfctr_cpu_control, _reserved3),
	  .tag = GPERFCTR_TAG_CPU_CONTROL_RSVD3,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct gperfctr_cpu_control, _reserved4),
	  .tag = GPERFCTR_TAG_CPU_CONTROL_RSVD4,
	  .type = PERFCTR_TYPE_BYTES4 },
};

static const struct perfctr_sub_struct_desc gperfctr_cpu_control_subs[] = {
	{ .offset = offsetof(struct gperfctr_cpu_control, cpu_control),
	  .sdesc = &perfctr_cpu_control_sdesc },
};

const struct perfctr_struct_desc gperfctr_cpu_control_sdesc = {
	.total_sizeof = sizeof(struct gperfctr_cpu_control),
	.total_nrfields = GPERFCTR_CPU_CONTROL_NRFIELDS,
	.nrfields = ARRAY_SIZE(gperfctr_cpu_control_fields),
	.fields = gperfctr_cpu_control_fields,
	.nrsubs = ARRAY_SIZE(gperfctr_cpu_control_subs),
	.subs = gperfctr_cpu_control_subs,
};

static const struct perfctr_field_desc gperfctr_cpu_state_fields[] = {
	{ .offset = offsetof(struct gperfctr_cpu_state, cpu),
	  .tag = GPERFCTR_TAG_CPU_STATE_CPU,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct gperfctr_cpu_state, _reserved1),
	  .tag = GPERFCTR_TAG_CPU_STATE_RSVD1,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct gperfctr_cpu_state, _reserved2),
	  .tag = GPERFCTR_TAG_CPU_STATE_RSVD2,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct gperfctr_cpu_state, _reserved3),
	  .tag = GPERFCTR_TAG_CPU_STATE_RSVD3,
	  .type = PERFCTR_TYPE_BYTES4 },
	{ .offset = offsetof(struct gperfctr_cpu_state, _reserved4),
	  .tag = GPERFCTR_TAG_CPU_STATE_RSVD4,
	  .type = PERFCTR_TYPE_BYTES4 },
};

static const struct perfctr_sub_struct_desc gperfctr_cpu_state_subs[] = {
	{ .offset = offsetof(struct gperfctr_cpu_state, cpu_control),
	  .sdesc = &perfctr_cpu_control_sdesc },
	{ .offset = offsetof(struct gperfctr_cpu_state, sum),
	  .sdesc = &perfctr_sum_ctrs_sdesc },
};

const struct perfctr_struct_desc gperfctr_cpu_state_only_cpu_sdesc = {
	.total_sizeof = sizeof(struct gperfctr_cpu_state),
	.total_nrfields = GPERFCTR_CPU_STATE_ONLY_CPU_NRFIELDS,
	.nrfields = ARRAY_SIZE(gperfctr_cpu_state_fields),
	.fields = gperfctr_cpu_state_fields,
};

const struct perfctr_struct_desc gperfctr_cpu_state_sdesc = {
	.total_sizeof = sizeof(struct gperfctr_cpu_state),
	.total_nrfields = GPERFCTR_CPU_STATE_NRFIELDS,
	.nrfields = ARRAY_SIZE(gperfctr_cpu_state_fields),
	.fields = gperfctr_cpu_state_fields,
	.nrsubs = ARRAY_SIZE(gperfctr_cpu_state_subs),
	.subs = gperfctr_cpu_state_subs,
};
#endif	/* CONFIG_PERFCTR_GLOBAL || !__KERNEL__ */

#ifdef __KERNEL__

int perfctr_copy_from_user(void *struct_address,
			   struct perfctr_struct_buf *argp,
			   const struct perfctr_struct_desc *sdesc)
{
	struct perfctr_marshal_stream stream;

	if( get_user(stream.size, &argp->rdsize) )
		return -EFAULT;
	stream.buffer = argp->buffer;
	stream.pos = 0;
	stream.error = 0;
	memset(struct_address, 0, sdesc->total_sizeof);
	return perfctr_decode_struct(struct_address, sdesc, &stream);
}

int perfctr_copy_to_user(struct perfctr_struct_buf *argp,
			 void *struct_address,
			 const struct perfctr_struct_desc *sdesc)
{
	struct perfctr_marshal_stream stream;

	if( get_user(stream.size, &argp->wrsize) )
		return -EFAULT;
	stream.buffer = argp->buffer;
	stream.pos = 0;
	stream.error = 0;
	perfctr_encode_struct(struct_address, sdesc, &stream);
	if( stream.error )
		return stream.error;
	if( put_user(stream.pos, &argp->rdsize) )
		return -EFAULT;
	return 0;
}

#else	/* !__KERNEL__ */

#define sdesc_bufsize(sdesc)	((sdesc)->total_nrfields + (sdesc)->total_sizeof/sizeof(int))

static int common_ioctl_w(const void *arg,
			  const struct perfctr_struct_desc *sdesc,
			  struct perfctr_struct_buf *buf,
			  unsigned int bufsize)
{
	struct perfctr_marshal_stream stream;

	stream.size = bufsize;
	stream.buffer = buf->buffer;
	stream.pos = 0;
	stream.error = 0;
	perfctr_encode_struct(arg, sdesc, &stream);
	if( stream.error ) {
		errno = -stream.error;
		return -1;
	}
	buf->rdsize = stream.pos;
	return 0;
}

int perfctr_ioctl_w(int fd, unsigned int cmd, const void *arg,
		    const struct perfctr_struct_desc *sdesc)
{
	unsigned int bufsize = sdesc_bufsize(sdesc);
	union {
		struct perfctr_struct_buf buf;
		struct {
			unsigned int rdsize;
			unsigned int wrsize;
			unsigned int buffer[bufsize];
		} buf_bufsize;
	} u;
	int err;

	err = common_ioctl_w(arg, sdesc, &u.buf, bufsize);
	if( err < 0 )
		return err;
	u.buf.wrsize = 0;
	return ioctl(fd, cmd, &u.buf);
}

static int common_ioctl_r(int fd, unsigned int cmd, void *res,
			   const struct perfctr_struct_desc *sdesc,
			   struct perfctr_struct_buf *buf)
{
	struct perfctr_marshal_stream stream;
	int err;

	if( ioctl(fd, cmd, buf) < 0 )
		return -1;
	stream.size = buf->rdsize;
	stream.buffer = buf->buffer;
	stream.pos = 0;
	stream.error = 0;
	memset(res, 0, sdesc->total_sizeof);
	err = perfctr_decode_struct(res, sdesc, &stream);
	if( err < 0 ) {
		errno = -err;
		return -1;
	}
	return 0;
}

int perfctr_ioctl_r(int fd, unsigned int cmd, void *res,
		    const struct perfctr_struct_desc *sdesc)
{
	unsigned int bufsize = sdesc_bufsize(sdesc);
	union {
		struct perfctr_struct_buf buf;
		struct {
			unsigned int rdsize;
			unsigned int wrsize;
			unsigned int buffer[bufsize];
		} buf_bufsize;
	} u;

	u.buf.rdsize = 0;
	u.buf.wrsize = bufsize;
	return common_ioctl_r(fd, cmd, res, sdesc, &u.buf);
}

int perfctr_ioctl_wr(int fd, unsigned int cmd, void *argres,
		     const struct perfctr_struct_desc *arg_sdesc,
		     const struct perfctr_struct_desc *res_sdesc)
{
	unsigned int arg_bufsize = sdesc_bufsize(arg_sdesc);
	unsigned int res_bufsize = sdesc_bufsize(res_sdesc);
	unsigned int bufsize = arg_bufsize > res_bufsize ? arg_bufsize : res_bufsize;
	union {
		struct perfctr_struct_buf buf;
		struct {
			unsigned int rdsize;
			unsigned int wrsize;
			unsigned int buffer[bufsize];
		} buf_bufsize;
	} u;
	int err;

	err = common_ioctl_w(argres, arg_sdesc, &u.buf, arg_bufsize);
	if( err < 0 )
		return err;
	u.buf.wrsize = res_bufsize;
	return common_ioctl_r(fd, cmd, argres, res_sdesc, &u.buf);
}

#endif /* !__KERNEL__ */
