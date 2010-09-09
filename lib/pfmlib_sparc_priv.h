#ifndef __PFMLIB_SPARC_PRIV_H__
#define __PFMLIB_SPARC_PRIV_H__

typedef struct {
	char			*mask_name;	/* mask name */
	char			*mask_desc;	/* mask description */
} sparc_mask_t;

#define EVENT_MASK_BITS		8
typedef struct {
	char			*name;	/* event name */
	char			*desc;	/* event description */
	char			ctrl;	/* S0 or S1 */
	char			__pad;
	int			code;	/* S0/S1 encoding */
	int			nmasks;	/* number of entries in masks */
	sparc_mask_t		masks[EVENT_MASK_BITS];
} sparc_entry_t;

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
