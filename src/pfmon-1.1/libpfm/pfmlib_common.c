/*
 * pfmlib_common.c: set of functions common to all PMU models
 *
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

#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syscall.h>
#include <stdarg.h>

#include <perfmon/pfmlib.h>

#include "pfmlib_priv.h"

extern pfm_pmu_support_t itanium2_support;
extern pfm_pmu_support_t itanium_support;
extern pfm_pmu_support_t generic_support;

static pfm_pmu_support_t *pmus[]=
{
#ifdef CONFIG_PFMLIB_ITANIUM2
	&itanium2_support,
#endif
#ifdef CONFIG_PFMLIB_ITANIUM
	&itanium_support,
#endif

#ifdef CONFIG_PFMLIB_GENERIC
	&generic_support,	/* must always be last */
#endif
	NULL
};

/*
 * contains runtime configuration options for the library.
 * mostly for debug purposes.
 */
pfm_config_t pfm_config;

int
pfm_initialize(void)
{
	pfm_pmu_support_t **p = &pmus[0];
	while (*p) {
		if ((*p)->cpu_detect() == PFMLIB_SUCCESS) goto found;
		p++;
	}
	return PFMLIB_ERR_NOTSUPP;
found:
	pfm_config.current = *p;
	return PFMLIB_SUCCESS;
}

int
pfm_set_options(pfmlib_options_t *opt)
{
	if (opt == NULL) return PFMLIB_ERR_INVAL;

	pfm_config.options = *opt;

	return PFMLIB_SUCCESS;
}

static char *pmu_names[]={
	"generic",
	"itanium",
	"itanium2"
};

#define NB_PMU_TYPES	((sizeof(pmu_names)/sizeof(char *)))

int
pfm_get_pmu_name_bytype(int type, char **name)
{
	if (name == NULL) return PFMLIB_ERR_INVAL;

	if (type < 0 || type >= NB_PMU_TYPES) return PFMLIB_ERR_INVAL;

	*name = pmu_names[type]; 

	return PFMLIB_SUCCESS;
}

int
pfm_list_supported_pmus(int (*pf)(const char *fmt,...))
{
	pfm_pmu_support_t **p = pmus;
	char *name;

	if (pf == NULL) return PFMLIB_ERR_INVAL;

	(*pf)("supported PMU models: ");
	for (p = pmus; *p; p++) {
		(*pf)("[%s] ", pmu_names[(*p)->pmu_type]);
	}
	pfm_get_pmu_name(&name);
	(*pf)("\ndetected host PMU: %s\n", name);

	return PFMLIB_SUCCESS;
}

int
pfm_get_pmu_name(char **name)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (name == NULL) return PFMLIB_ERR_INVAL;

	*name = pmu_names[pfm_current->pmu_type];

	return PFMLIB_SUCCESS;
}

int
pfm_get_pmu_type(int *type)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (type == NULL) return PFMLIB_ERR_INVAL;

	*type = pfm_current->pmu_type;

	return PFMLIB_SUCCESS;
}

/*
 * boolean return value
 */
int
pfm_is_pmu_supported(int type)
{
	pfm_pmu_support_t **p = pmus;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	while (*p) {
		if ((*p)->pmu_type == type) return PFMLIB_SUCCESS;
		p++;
	}
	return PFMLIB_ERR_NOTSUPP;
}

int
pfm_force_pmu(int type)
{
	pfm_pmu_support_t **p = pmus;
	char *name;

	while (*p) {
		if ((*p)->pmu_type == type) goto found;
		p++;
	}
	return PFMLIB_ERR_NOTSUPP;
found:
	/* verify that this is valid */
	if ((*p)->cpu_detect() != PFMLIB_SUCCESS) return PFMLIB_ERR_NOTSUPP;

	pfm_config.current = *p;

	pfm_get_pmu_name(&name);
	printf("forced libpfm to %s support\n", name);

	pfm_current = *p;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_event_code(char *str, unsigned int *code)
{
	long ret;
	char *endptr = NULL;

	if (str == NULL) return PFMLIB_ERR_INVAL;

	ret = strtol(str,&endptr, 0);

	/* check for errors */
	if (*endptr!='\0') return PFMLIB_ERR_INVAL;

	*code = (unsigned int)ret;

	return PFMLIB_SUCCESS;
}

int
pfm_find_event_byname(char *n, int *idx)
{
	 int i;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (n == NULL || idx == NULL) return PFMLIB_ERR_INVAL;

	/* we do case insensitive comparisons */
	for(i=0; i < pfm_current->pme_count; i++) {
		if (!strcasecmp(pfm_current->get_event_name(i), n)) goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*idx = i;
	return PFMLIB_SUCCESS;
}

int
pfm_find_event_byvcode(int code, int *idx)
{
	int i;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (idx == NULL) return PFMLIB_ERR_INVAL;

	for(i=0; i < pfm_current->pme_count; i++) {
		if (pfm_current->get_event_vcode(i) == code) goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*idx = i;
	return PFMLIB_SUCCESS;
}

int
pfm_find_event_bycode(unsigned long code, int *idx)
{
	int i;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (idx == NULL) return PFMLIB_ERR_INVAL;

	for(i=0; i < pfm_current->pme_count; i++) {
		if (pfm_current->get_event_code(i) == code) goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*idx = i;
	return PFMLIB_SUCCESS;
}

int
pfm_find_event(char *v, int retry, int *ev)
{
	unsigned int num;
	int ret;


	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (v == NULL || ev == NULL) return PFMLIB_ERR_INVAL;

	if (isdigit(*v)) {
		if ((pfm_gen_event_code(v, &num)) != PFMLIB_SUCCESS) return PFMLIB_ERR_INVAL;
		ret = pfm_find_event_byvcode(num, ev);
		if (ret != PFMLIB_SUCCESS && retry) ret = pfm_find_event_bycode(num, ev);
		
	} else {
		ret = pfm_find_event_byname(v, ev);
	}
	return ret;
}

int
pfm_find_event_bycode_next(int code, int i, int *next)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (next == NULL) return PFMLIB_ERR_INVAL;

	for(++i; i < pfm_current->pme_count; i++) {
		if (pfm_current->get_event_code(i) == code) goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*next = i;
	return PFMLIB_SUCCESS;
}

int
pfm_find_event_byvcode_next(int vcode, int i, int *next)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (next == NULL) return PFMLIB_ERR_INVAL;

	for(++i; i < pfm_current->pme_count; i++) {
		if (pfm_current->get_event_vcode(i) == vcode) goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*next = i;
	return PFMLIB_SUCCESS;
}

int
pfm_get_event_name(int i, char **name)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i<0 || i >= pfm_current->pme_count || name == NULL) return PFMLIB_ERR_INVAL;

	*name = pfm_current->get_event_name(i);

	return PFMLIB_SUCCESS;
}

int
pfm_get_event_code(int i, int *ev)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i<0 || i >= pfm_current->pme_count || ev == NULL) return PFMLIB_ERR_INVAL;

	*ev = pfm_current->get_event_code(i);

	return PFMLIB_SUCCESS;
}

int
pfm_get_event_counters(int i, unsigned long *counters)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i<0 || i >= pfm_current->pme_count || counters == NULL) return PFMLIB_ERR_INVAL;

	*counters = pfm_current->get_event_counters(i);

	return PFMLIB_SUCCESS;
}

int
pfm_get_first_event(void)
{
	return 0; /* could be different */
}

int
pfm_get_next_event(int i)
{
	if (PFMLIB_INITIALIZED() == 0) return -1;

	if (i < 0 || i >= pfm_current->pme_count-1) return -1;

	return i+1;
}

/*
 * Function used to print information about a specific events. More than
 * one event can be printed in case an event code is given rather than
 * a specific name. A callback function is used for printing.
 */
int
pfm_print_event_info(char *name, int (*pf)(const char *fmt,...))
{
	int (*find_next)(int code, int i, int *next);
	long c;
        int i, code, ret;
	int code_is_used = 1, event_is_digit = 0;
	int v;
	unsigned int num;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (name == NULL || pf == NULL) return PFMLIB_ERR_INVAL;

	/* we can't quite use pfm_findevent() because we need to try
	 * both ways systematically.
	 */
	find_next = pfm_find_event_bycode_next;
	if (isdigit(*name)) {
		ret = pfm_gen_event_code(name, &num);
		if (ret != PFMLIB_SUCCESS) return ret;

		ret = pfm_find_event_bycode(num, &v);
		if (ret != PFMLIB_SUCCESS) {
			find_next = pfm_find_event_byvcode_next;
			ret  = pfm_find_event_byvcode(num, &v);
			code_is_used = 0;
		}
		event_is_digit = 1;
	} else {
		ret  = pfm_find_event_byname(name, &v);
	}

	if (ret != PFMLIB_SUCCESS) return PFMLIB_ERR_NOTFOUND;

	code = code_is_used ? pfm_current->get_event_code(v) : pfm_current->get_event_vcode(v);

	do {	
		(*pf)(	"Name   : %s\n" 
			"VCode  : 0x%lx\n"
			"Code   : 0x%lx\n",
			pfm_current->get_event_name(v), 
			pfm_current->get_event_vcode(v),
			pfm_current->get_event_code(v));
		
		(*pf)(	"PMD/PMC: [ ");

		c = pfm_current->get_event_counters(v);
		for (i=0; c; i++, c>>=1 ) {
			if (c & 0x1) (*pf)("%d ", i);
		}
		(*pf)(	"]\n");

		/* print PMU specific information */
		if (pfm_config.current->print_info) {
			pfm_config.current->print_info(v, pf);
		}
	} while (event_is_digit && find_next(code, v, &v) == PFMLIB_SUCCESS);

	return PFMLIB_SUCCESS;
}

int
pfm_print_event_info_byindex(int v, int (*pf)(const char *fmt,...))
{
	long c;
        int i;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (v <0 || v >= pfm_current->pme_count || pf == NULL) return PFMLIB_ERR_INVAL;

	(*pf)(	"Name   : %s\n" 
		"VCode  : 0x%lx\n"
		"Code   : 0x%lx\n",
		pfm_current->get_event_name(v), 
		pfm_current->get_event_vcode(v),
		pfm_current->get_event_code(v));
	
	(*pf)(	"PMD/PMC: [ ");

	c = pfm_current->get_event_counters(v);
	for (i=0; c; i++, c>>=1 ) {
		if (c & 0x1) (*pf)("%d ", i);
	}
	(*pf)(	"]\n");

	/* print PMU specific information */
	if (pfm_config.current->print_info) {
		pfm_config.current->print_info(v, pf);
	}
	return PFMLIB_SUCCESS;
}


int
pfm_dispatch_events(pfmlib_param_t *evt, pfarg_reg_t *pc, int *count)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (evt == NULL || pc == NULL || count == NULL) return PFMLIB_ERR_INVAL;

	return pfm_config.current->dispatch_events(evt, pc, count);
}

int
pfm_get_num_counters(void)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;
	
	return pfm_config.current->get_num_counters();
}

int
pfm_get_impl_pmcs(unsigned long impl_pmcs[4])
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;
	if (impl_pmcs == NULL) return PFMLIB_ERR_INVAL;
	return pfm_config.current->get_impl_pmcs(impl_pmcs);
}

int
pfm_get_impl_pmds(unsigned long impl_pmds[4])
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;
	if (impl_pmds == NULL) return PFMLIB_ERR_INVAL;
	return pfm_config.current->get_impl_pmds(impl_pmds);
}

static const char *pfmlib_err_list[]=
{
	"success",
	"not supported",
	"invalid parameters",
	"pfmlib not initialized",
	"object not found",
	"cannot assign events to counters",
	"buffer is full",
	"event used more than once",
	"invalid model specific magic number",
	"invalid combination of model specific features",
	"incompatible event sets",
	"incompatible events combination",
	"too many events",
	"code range too big",
	"empty code range",
	"invalid code range",
	"too many code ranges",
	"invalid data range",
	"too many data ranges",
	"not supported by host cpu",
	"code range is not bundle-aligned"
};
static unsigned int pfmlib_err_count = sizeof(pfmlib_err_list)/sizeof(char *);

const char *
pfm_strerror(int code)
{
	code = -code;
	if (code <0 || code >= pfmlib_err_count) return "??";
	return pfmlib_err_list[code];
}

void
pfm_vbprintf(char *fmt, ...)
{
	va_list ap;

	if (pfm_config.options.pfm_verbose == 0) return;

	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
}


/*
 * Starts monitoring for user-level monitors
 */
#undef pfm_start
void
pfm_start(void)
{
#ifdef __GNUC__
	__asm__ __volatile__("sum psr.up;;" ::: "memory" );
#elif defined(INTEL_ECC_COMPILER)
	__sum(1<<2);
#else
#error "you need to provide inline assembly from your compiler"
#endif
}

/*
 * Stops monitoring for user-level monitors
 */
#undef pfm_stop
void
pfm_stop(void)
{
#ifdef __GNUC__
	__asm__ __volatile__("rum psr.up;;" ::: "memory" );
#elif defined(INTEL_ECC_COMPILER)
	__rum(1<<2);
#else
#error "you need to provide inline assembly from your compiler"
#endif
}

/*
 * Read raw PMD: only architected width relevant. No access to
 * virtualized 64bits version used by kernel. Good for small measurements
 */
#undef pfm_get_pmd
unsigned long
pfm_get_pmd(int regnum)
{
	unsigned long retval;
#ifdef __GNUC__
	__asm__ __volatile__ ("mov %0=pmd[%1]" : "=r"(retval) : "r"(regnum));
#elif defined(INTEL_ECC_COMPILER)
	retval = 0UL;
#else
#error "you need to provide inline assembly from your compiler"
#endif
	return retval;
}


/*
 * once this API is finalized, we should implement this in GNU libc
 */
int
perfmonctl(pid_t pid, int cmd, void *arg, int narg)
{
	return syscall(__NR_perfmonctl, pid, cmd, arg, narg);
}


