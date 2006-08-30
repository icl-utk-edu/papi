/*
(unsigned long(unsigned long)) * pfmlib_common.c: set of functions common to all PMU models
 *
 * Copyright (C) 2001-2002 Hewlett-Packard Co
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

#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syscall.h>
#include <stdarg.h>
#include <limits.h>

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
	"itanium 2"
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

	pfm_current = *p;

	return PFMLIB_SUCCESS;
}

static int
pfm_gen_event_code(char *str, int *code)
{
	long ret;
	char *endptr = NULL;

	if (str == NULL || code == NULL) return PFMLIB_ERR_INVAL;

	ret = strtol(str,&endptr, 0);

	if (ret > INT_MAX) return PFMLIB_ERR_INVAL;

	*code = (int)ret;

	/* check for errors */
	if (*endptr!='\0') return PFMLIB_ERR_INVAL;

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

static int
pfm_find_event_byvcode(unsigned long vcode, int *idx)
{
	int i;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (idx == NULL) return PFMLIB_ERR_INVAL;

	for(i=0; i < pfm_current->pme_count; i++) {
		if (pfm_current->get_event_vcode(i) == vcode) goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*idx = i;
	return PFMLIB_SUCCESS;
}

int
pfm_find_event_bycode(int code, int *idx)
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
pfm_find_event(char *v, int *ev)
{
	int num;
	int ret;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (v == NULL || ev == NULL) return PFMLIB_ERR_INVAL;

	if (isdigit(*v)) {
		if ((pfm_gen_event_code(v, &num)) != PFMLIB_SUCCESS) return PFMLIB_ERR_INVAL;
		ret = pfm_find_event_bycode(num, ev);
		
	} else {
		ret = pfm_find_event_byname(v, ev);
	}
	return ret;
}

#if 0
/* obsolete */
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
#endif

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

static int
pfm_find_event_byvcode_next(unsigned long vcode, int i, int *next)
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
pfm_get_event_code(int i, int *code)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i<0 || i >= pfm_current->pme_count || code == NULL) return PFMLIB_ERR_INVAL;

	*code = pfm_current->get_event_code(i);

	return PFMLIB_SUCCESS;
}

int
pfm_get_event_counters(int i, unsigned long counters[4])
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i<0 || i >= pfm_current->pme_count) return PFMLIB_ERR_INVAL;

	pfm_current->get_event_counters(i, counters);

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

	if (i < 0 || i >= pfm_current->pme_count) return -1;

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
	unsigned long cmask[4];
	long c;
        int i, code, ret;
	int code_is_used = 1, event_is_digit = 0;
	int idx;
	int number;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (name == NULL || pf == NULL) return PFMLIB_ERR_INVAL;

	/* we can't quite use pfm_findevent() because we need to try
	 * both ways systematically.
	 */
	if (isdigit(*name)) {
		ret = pfm_gen_event_code(name, &number);
		if (ret != PFMLIB_SUCCESS) return ret;

		if (number > INT_MAX) return PFMLIB_ERR_INVAL;

		ret = pfm_find_event_bycode(number, &idx);
		if (ret != PFMLIB_SUCCESS) {
			/* XXX: ugly upper 32 bits necessarily cleared */
			ret  = pfm_find_event_byvcode((unsigned long)number, &idx);
			code_is_used = 0;
		}
		event_is_digit = 1;
	} else {
		ret  = pfm_find_event_byname(name, &idx);
	}

	if (ret != PFMLIB_SUCCESS) return PFMLIB_ERR_NOTFOUND;

	code = code_is_used ? pfm_current->get_event_code(idx) : pfm_current->get_event_vcode(idx);

	do {	
		(*pf)(	"Name   : %s\n" 
			"VCode  : 0x%lx\n"
			"Code   : 0x%lx\n",
			pfm_current->get_event_name(idx), 
			pfm_current->get_event_vcode(idx),
			pfm_current->get_event_code(idx));
		
		(*pf)(	"PMD/PMC: [ ");

		pfm_current->get_event_counters(idx, cmask);
		c = cmask[0];
		for (i=0; c; i++, c>>=1 ) {
			if (c & 0x1) (*pf)("%d ", i);
		}
		(*pf)(	"]\n");

		/* print PMU specific information */
		if (pfm_config.current->print_info) {
			pfm_config.current->print_info(idx, pf);
		}
		ret = code_is_used ? 
			pfm_find_event_bycode_next(code, idx, &idx) :
			pfm_find_event_byvcode_next((unsigned long)number, idx, &idx);

	} while (event_is_digit && ret == PFMLIB_SUCCESS);

	return PFMLIB_SUCCESS;
}

int
pfm_print_event_info_byindex(int v, int (*pf)(const char *fmt,...))
{
	unsigned long cmask[4];
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

	pfm_current->get_event_counters(v, cmask);
	c = cmask[0];
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
pfm_dispatch_events(pfmlib_param_t *evt)
{
	int max_count;
	int i;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (evt == NULL) return PFMLIB_ERR_INVAL;

	/*
	 * the default priv level must be set to something
	 */
	if (evt->pfp_dfl_plm == 0) return PFMLIB_ERR_INVAL;

	if (evt->pfp_event_count >= PMU_MAX_PMCS) return PFMLIB_ERR_INVAL;

	/*
	 * check that event descriptors are correct
	 *
	 * invalid plm bits are simply ignored
	 */
	max_count = pfm_current->pme_count; 

	for (i=0; i < evt->pfp_event_count; i++) {
		if (evt->pfp_events[i].event < 0 || evt->pfp_events[i].event >= max_count) 
			return PFMLIB_ERR_INVAL;
	}

	/* reset pc counter */
	evt->pfp_pc_count = 0;

	return pfm_config.current->dispatch_events(evt);
}

/*
 * more or less obosleted by pfm_get_impl_counters()
 */
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

int
pfm_get_impl_counters(unsigned long impl_counters[4])
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;
	if (impl_counters == NULL) return PFMLIB_ERR_INVAL;
	return pfm_config.current->get_impl_counters(impl_counters);
}

/* sorry, only English supported at this point! */
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
	if (code <0 || code >= pfmlib_err_count) return (const char *)"unknown error code";
	return pfmlib_err_list[code];
}

int
pfm_get_event_description(unsigned int i, char **str)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i >= pfm_current->pme_count || str == NULL) return PFMLIB_ERR_INVAL;

	if (pfm_current->get_event_desc == NULL) {
		*str = NULL;	
		return PFMLIB_SUCCESS;
	}
	return pfm_current->get_event_desc(i, str);
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

int
pfm_get_version(unsigned int *version)
{
	if (version == NULL) return PFMLIB_ERR_INVAL;
	*version = PFMLIB_VERSION;
	return 0;
}

/*
 * once this API is finalized, we should implement this in GNU libc
 */
int
perfmonctl(pid_t pid, int cmd, void *arg, int narg)
{
	return syscall(__NR_perfmonctl, pid, cmd, arg, narg);
}
