/*
 * Copyright (C) 2001-2002 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file is part of pfmon, a sample tool to measure performance 
 * of applications on Linux/ia64.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */
#ifndef __PFMLIB_PRIV_H__
#define __PFMLIB_PRIV_H__

typedef struct {
	char 		*name;
	int		pmu_type;
	unsigned long	pme_count; /* number of events */
	unsigned long	(*get_event_code)(int i);
	unsigned long	(*get_event_vcode)(int i);
	char		*(*get_event_name)(int i);
	unsigned long	(*get_event_counters)(int i);
	int		(*print_info)(int v, int (*pf)(const char *fmt,...));
	int 		(*dispatch_events)(pfmlib_param_t *p, pfarg_reg_t *pc, int *count);
	int		(*get_num_counters)(void);
	int 		(*cpu_detect)(void);
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

#ifdef __GNUC__
static inline unsigned long
ia64_get_cpuid (unsigned long regnum)
{
	unsigned long r;

	asm ("mov %0=cpuid[%r1]" : "=r"(r) : "rO"(regnum));
	return r;
}
#else
#error "need to define macro to get cpuid[] registers"
#endif


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
