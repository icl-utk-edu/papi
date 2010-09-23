#ifndef __PFMLIB_SPARC_PRIV_H__
#define __PFMLIB_SPARC_PRIV_H__

typedef struct {
	char			*uname;	/* mask name */
	char			*udesc;	/* mask description */
	int			ubit;	/* umask bit position */
} sparc_mask_t;

#define EVENT_MASK_BITS		8
typedef struct {
	char			*name;	/* event name */
	char			*desc;	/* event description */
	char			ctrl;	/* S0 or S1 */
	char			__pad;
	int			code;	/* S0/S1 encoding */
	int			numasks;	/* number of entries in masks */
	sparc_mask_t		umasks[EVENT_MASK_BITS];
} sparc_entry_t;

typedef union {
	unsigned int val;
	struct {
		unsigned int	ctrl_s0   : 1;
		unsigned int	ctrl_s1   : 1;
		unsigned int	reserved1 : 14;
		unsigned int	code	  : 8;
		unsigned int	umask	  : 8;
	} config;
} pfm_sparc_reg_t;

#define PME_CTRL_S0		1
#define PME_CTRL_S1		2

extern int pfm_sparc_detect(void);
extern int pfm_sparc_get_encoding(void *this, pfmlib_event_desc_t *e, uint64_t *codes, int *count, pfmlib_perf_attr_t *attrs);
extern int pfm_sparc_get_event_first(void *this);
extern int pfm_sparc_get_event_next(void *this, int idx);
extern int pfm_sparc_event_is_valid(void *this, int pidx);
extern int pfm_sparc_get_event_perf_type(void *this, int pidx);
extern int pfm_sparc_validate_table(void *this, FILE *fp);
extern int pfm_sparc_get_event_attr_info(void *this, int pidx, int attr_idx, pfm_event_attr_info_t *info);
extern int pfm_sparc_get_event_info(void *this, int idx, pfm_event_info_t *info);

#endif /* __PFMLIB_SPARC_PRIV_H__ */
