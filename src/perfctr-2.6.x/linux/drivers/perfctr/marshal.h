/* $Id: marshal.h,v 1.1 2003/08/19 13:37:07 mikpe Exp $
 * Performance-monitoring counters driver.
 * Structure marshalling support.
 *
 * Copyright (C) 2003  Mikael Pettersson
 */

/*
 * Each encoded datum starts with a 32-bit header word, containing
 * the datum's type (1 bit: UINT32 or UINT64), the target's field
 * tag (16 bits), and the target field's array index (15 bits).
 *
 * After the header follows the datum's value, in one (for UINT32)
 * or two (for UINT64) words. Multi-word values are emitted in
 * native word order.
 *
 * To encode a struct, encode each field with a non-zero value,
 * and place the encodings in sequence. The field order is arbitrary.
 *
 * To decode an encoded struct, first memset() the target struct
 * to zero. Then decode each encoded field in the sequence and
 * update the corresponding field in the target struct.
 */
#define PERFCTR_HEADER(TYPE,TAG,ITEMNR) (((TAG)<<16)|((ITEMNR)<<1)|(TYPE))
#define PERFCTR_HEADER_TYPE(H)		((H) & 0x1)
#define PERFCTR_HEADER_ITEMNR(H)	(((H) >> 1) & 0x7FFF)
#define PERFCTR_HEADER_TAG(H)		((H) >> 16)

#define PERFCTR_HEADER_UINT32		0
#define PERFCTR_HEADER_UINT64		1

/*
 * A field descriptor describes a struct field to the
 * encoding and decoding procedures.
 *
 * To keep the descriptors small, field tags and array sizes
 * are currently restricted to 8 and 7 bits, respectively.
 * This does not change the encoded format.
 */
struct perfctr_field_desc {
	unsigned short offset;	/* offsetof() for this field */
	unsigned char tag;	/* identifying tag in encoded format */
	unsigned char type;	/* base type (1 bit), array size - 1 (7 bits) */
};

#define PERFCTR_TYPE_ARRAY(N,T)	((((N) - 1) << 1) | (T))
#define PERFCTR_TYPE_BASE(T)	((T) & 0x1)
#define PERFCTR_TYPE_NRITEMS(T)	(((T) >> 1) + 1)

#define PERFCTR_TYPE_BYTES4	0	/* uint32 or char[4] */
#define PERFCTR_TYPE_UINT64	1	/* long long */

struct perfctr_struct_desc {
	unsigned short total_sizeof;	/* for buffer allocation and decode memset() */
	unsigned short total_nrfields;	/* for buffer allocation */
	unsigned short nrfields;
	unsigned short nrsubs;
	/* Note: the fields must be in ascending tag order */
	const struct perfctr_field_desc *fields;
	const struct perfctr_sub_struct_desc {
		unsigned short offset;
		const struct perfctr_struct_desc *sdesc;
	} *subs;
};

struct perfctr_marshal_stream {
	unsigned int size;
	unsigned int *buffer;
	unsigned int pos;
	unsigned int error;
};

extern void perfctr_encode_struct(const void *address,
				  const struct perfctr_struct_desc *sdesc,
				  struct perfctr_marshal_stream *stream);

extern int perfctr_decode_struct(void *address,
				 const struct perfctr_struct_desc *sdesc,
				 struct perfctr_marshal_stream *stream);

extern const struct perfctr_struct_desc perfctr_sum_ctrs_sdesc;
extern const struct perfctr_struct_desc perfctr_cpu_control_sdesc;
extern const struct perfctr_struct_desc perfctr_info_sdesc;
extern const struct perfctr_struct_desc vperfctr_control_sdesc;
extern const struct perfctr_struct_desc gperfctr_cpu_control_sdesc;
extern const struct perfctr_struct_desc gperfctr_cpu_state_only_cpu_sdesc;
extern const struct perfctr_struct_desc gperfctr_cpu_state_sdesc;

#ifdef __KERNEL__
extern int perfctr_copy_to_user(struct perfctr_struct_buf *argp,
				void *struct_address,
				const struct perfctr_struct_desc *sdesc);
extern int perfctr_copy_from_user(void *struct_address,
				  struct perfctr_struct_buf *argp,
				  const struct perfctr_struct_desc *sdesc);
#else
extern int perfctr_ioctl_w(int fd, unsigned int cmd, const void *arg,
			   const struct perfctr_struct_desc *sdesc);
extern int perfctr_ioctl_r(int fd, unsigned int cmd, void *res,
			   const struct perfctr_struct_desc *sdesc);
extern int perfctr_ioctl_wr(int fd, unsigned int cmd, void *argres,
			    const struct perfctr_struct_desc *arg_sdesc,
			    const struct perfctr_struct_desc *res_sdesc);
#endif /* __KERNEL__ */
