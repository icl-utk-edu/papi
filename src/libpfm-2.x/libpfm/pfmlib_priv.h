/*
 * Copyright (C) 2002 Hewlett-Packard Co
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
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux/ia64.
 */
#ifndef __PFMLIB_PRIV_H__
#define __PFMLIB_PRIV_H__

#include <perfmon/pfmlib.h>
#include "pfmlib_compiler_priv.h"

typedef struct {
	char 		*name;
	int		pmu_type;
	int		pme_count; /* number of events */
	int		(*get_event_code)(int i);
	unsigned long	(*get_event_vcode)(int i);
	char		*(*get_event_name)(int i);
	void		(*get_event_counters)(int i, unsigned long counters[4]);
	int		(*print_info)(int v, int (*pf)(const char *fmt,...));
	int 		(*dispatch_events)(pfmlib_param_t *p);
	int		(*get_num_counters)(void);
	int 		(*cpu_detect)(void);
	int		(*get_impl_pmcs)(unsigned long impl_pmcs[4]);
	int		(*get_impl_pmds)(unsigned long impl_pmds[4]);
	int		(*get_impl_counters)(unsigned long impl_counters[4]);
	int		(*get_event_desc)(unsigned int ev, char **str);
} pfm_pmu_support_t;

typedef struct {
	pfmlib_options_t	options;
	pfm_pmu_support_t	*current;
} pfm_config_t;	

#define PFMLIB_INITIALIZED()	(pfm_config.current != NULL)

extern pfm_config_t pfm_config;

#define PFMLIB_DEBUG()		pfm_config.options.pfm_debug
#define PFMLIB_VERBOSE()	pfm_config.options.pfm_verbose
#define pfm_current		pfm_config.current

extern void pfm_vbprintf(char *fmt,...);

#ifdef PFMLIB_DEBUG
#define DPRINT(a) \
	do { \
		if (pfm_config.options.pfm_debug) { \
			printf("%s (%s.%d): ", __FILE__, __FUNCTION__, __LINE__); printf a; } \
	} while (0)
#else
#define DPRINT(a)
#endif

#define ALIGN_DOWN(a,p)	((a) & ~((1UL<<(p))-1))
#define ALIGN_UP(a,p)	((((a) + ((1UL<<(p))-1))) & ~((1UL<<(p))-1))

typedef struct {
	unsigned long db_mask:56;
	unsigned long db_plm:4;
	unsigned long db_ig:2;
	unsigned long db_w:1;
	unsigned long db_rx:1;
} br_mask_reg_t;

typedef union {
	unsigned long  val;
	br_mask_reg_t  db;
} dbreg_t;


static inline int
pfm_get_cpu_family(void)
{
	return (int)((ia64_get_cpuid(3) >> 24) & 0xff);
}

static inline int
pfm_get_cpu_model(void)
{
	return (int)((ia64_get_cpuid(3) >> 16) & 0xff);
}

#endif /* __PFMLIB_PRIV_H__ */
