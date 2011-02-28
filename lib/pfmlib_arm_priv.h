/*
 * Copyright (c) 2010 University of Tennessee
 * Contributed by Vince Weaver <vweaver1@utk.edu>
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */
#ifndef __PFMLIB_ARM_PRIV_H__
#define __PFMLIB_ARM_PRIV_H__

/*
 * This file contains the definitions used for ARM processors
 */


/*
 * event description
 */
typedef struct {
	const char			*name;	/* event name */
	const char			*desc;	/* event description */
	unsigned int			code; 	/* event code */
} arm_entry_t;

typedef union pfm_arm_reg {
	unsigned int val;			/* complete register value */
   
	struct {
		unsigned int sel:8;	        /* counter */
		unsigned int reserved:24;
	} evtsel;
} pfm_arm_reg_t;

typedef struct {
	int implementer;
	int architecture;
	int part;
} pfm_arm_config_t;

extern pfm_arm_config_t pfm_arm_cfg;

extern int pfm_arm_detect(void *this);
extern int pfm_arm_get_encoding(void *this, pfmlib_event_desc_t *e);
extern int pfm_arm_get_event_first(void *this);
extern int pfm_arm_get_event_next(void *this, int idx);
extern int pfm_arm_event_is_valid(void *this, int pidx);
extern int pfm_arm_validate_table(void *this, FILE *fp);
extern int pfm_arm_get_event_attr_info(void *this, int pidx, int attr_idx, pfm_event_attr_info_t *info);
extern int pfm_arm_get_event_info(void *this, int idx, pfm_event_info_t *info);
extern int pfm_arm_get_event_nattrs(void *this, int pidx);

extern void pfm_arm_perf_validate_pattrs(void *this, pfmlib_event_desc_t *e);
extern int pfm_arm_get_perf_encoding(void *this, pfmlib_event_desc_t *e);

#endif /* __PFMLIB_ARM_PRIV_H__ */
