/*
 * pfmlib_common.c: set of functions common to all PMU models
 *
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
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
 */
#define _GNU_SOURCE /* for getline */
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>

#include <perfmon/pfmlib.h>

#include "pfmlib_priv.h"

static pfm_pmu_support_t *supported_pmus[]=
{

#ifdef CONFIG_PFMLIB_ARCH_IA64
	&montecito_support,
	&itanium2_support,
	&itanium_support,
	&generic_ia64_support,	/* must always be last for IA-64 */
#endif

#ifdef CONFIG_PFMLIB_ARCH_X86_64
	&amd64_support,
	&pentium4_support,
	&core_support,
	&gen_ia32_support, /* must always be last for x86-64 */
#endif

#ifdef CONFIG_PFMLIB_ARCH_I386
	&i386_p6_support,
	&i386_pm_support,
	&coreduo_support,
	&amd64_support,
	&pentium4_support,
	&core_support,
	&gen_ia32_support, /* must always be last for i386 */
#endif

#ifdef CONFIG_PFMLIB_ARCH_MIPS64
	&generic_mips64_support,
#endif

#ifdef CONFIG_PFMLIB_ARCH_POWERPC
	&generic_powerpc_support,
#endif

#ifdef CONFIG_PFMLIB_ARCH_CRAYX2
	&crayx2_support,
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
	pfm_pmu_support_t **p = supported_pmus;

	while(*p) {
		if ((*p)->pmu_detect() == PFMLIB_SUCCESS)
			goto found;
		p++;
	}
	return PFMLIB_ERR_NOTSUPP;
found:
	/*
	 * run a few sanity checks
	 */
	if ((*p)->pmc_count >= PFMLIB_MAX_PMCS)
		return PFMLIB_ERR_NOTSUPP;
	if ((*p)->pmd_count >= PFMLIB_MAX_PMDS)
		return PFMLIB_ERR_NOTSUPP;

	pfm_current = *p;
	return PFMLIB_SUCCESS;
}

int
pfm_set_options(pfmlib_options_t *opt)
{
	if (opt == NULL) return PFMLIB_ERR_INVAL;

	pfm_config.options = *opt;

	return PFMLIB_SUCCESS;
}

/*
 * return the name corresponding to the pmu type. Only names
 * of PMU actually compiled in the library will be returned.
 */
int
pfm_get_pmu_name_bytype(int type, char *name, size_t maxlen)
{
	pfm_pmu_support_t **p = supported_pmus;

	if (name == NULL || maxlen < 1) return PFMLIB_ERR_INVAL;

	while (*p) {
		if ((*p)->pmu_type == type) goto found;
		p++;
	}
	return PFMLIB_ERR_INVAL;
found:
	strncpy(name, (*p)->pmu_name, maxlen-1);

	/* make sure the string is null terminated */
	name[maxlen-1] = '\0';

	return PFMLIB_SUCCESS;
}

int
pfm_list_supported_pmus(int (*pf)(const char *fmt,...))
{
	pfm_pmu_support_t **p;

	if (pf == NULL) return PFMLIB_ERR_INVAL;

	(*pf)("supported PMU models: ");

	for (p = supported_pmus; *p; p++) {
		(*pf)("[%s] ", (*p)->pmu_name);;
	}

	(*pf)("\ndetected host PMU: %s\n", pfm_current ? pfm_current->pmu_name : "not detected yet");

	return PFMLIB_SUCCESS;
}

int
pfm_get_pmu_name(char *name, int maxlen)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (name == NULL || maxlen < 1) return PFMLIB_ERR_INVAL;

	strncpy(name, pfm_current->pmu_name, maxlen-1);

	name[maxlen-1] = '\0';

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
	pfm_pmu_support_t **p = supported_pmus;

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
	pfm_pmu_support_t **p = supported_pmus;

	while (*p) {
		if ((*p)->pmu_type == type) goto found;
		p++;
	}
	return PFMLIB_ERR_NOTSUPP;
found:
	pfm_current = *p;

	return PFMLIB_SUCCESS;
}

int
pfm_find_event_byname(const char *n, unsigned int *idx)
{
	char *p, *e;
	unsigned int i;
	size_t len;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (n == NULL || idx == NULL)
		return PFMLIB_ERR_INVAL;

	/*
	 * this function ignores any ':' separator
	 */
	p = strchr(n, ':');
	if (!p)
		len = strlen(n);
	else
		len = p - n;

	/*
	 * we do case insensitive comparisons
	 *
	 * event names must match completely
	 */
	for(i=0; i < pfm_current->pme_count; i++) {
		e = pfm_current->get_event_name(i);
		if (!strncasecmp(e, n, len)
		    && len == strlen(e))
			goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*idx = i;
	return PFMLIB_SUCCESS;
}

int
pfm_find_event_bycode(int code, unsigned int *idx)
{
	pfmlib_regmask_t impl_cnt;
	unsigned int i, j, num_cnt;
	int code2;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (idx == NULL) return PFMLIB_ERR_INVAL;

	if (pfm_current->flags & PFMLIB_MULT_CODE_EVENT) {
		pfm_current->get_impl_counters(&impl_cnt);
		num_cnt = pfm_current->num_cnt;

		for(i=0; i < pfm_current->pme_count; i++) {
			for(j=0; num_cnt; j++) {
				if (pfm_regmask_isset(&impl_cnt, j)) {
					pfm_current->get_event_code(i, j, &code2);
					if (code2 == code)
						goto found;
					num_cnt--;
				}
			}
		}
	} else {
		for(i=0; i < pfm_current->pme_count; i++) {
			pfm_current->get_event_code(i, PFMLIB_CNT_FIRST, &code2);
			if (code2 == code) goto found;
		}
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*idx = i;
	return PFMLIB_SUCCESS;
}

int
pfm_find_event(const char *v, unsigned int *ev)
{
	unsigned long number;
	char *endptr = NULL;
	int ret = PFMLIB_ERR_INVAL;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (v == NULL || ev == NULL)
		return PFMLIB_ERR_INVAL;

	if (isdigit((int)*v)) {
		number = strtoul(v,&endptr, 0);
		/* check for errors */
		if (*endptr!='\0')
			return PFMLIB_ERR_INVAL;

		if (number <= INT_MAX) {
			int the_int_number = (int)number;
			ret = pfm_find_event_bycode(the_int_number, ev);
		}
	} else 
		ret = pfm_find_event_byname(v, ev);
	return ret;
}

int
pfm_find_event_bycode_next(int code, unsigned int i, unsigned int *next)
{
	int code2;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (!next)
		return PFMLIB_ERR_INVAL;

	for(++i; i < pfm_current->pme_count; i++) {
		pfm_current->get_event_code(i, PFMLIB_CNT_FIRST, &code2);
		if (code2 == code) goto found;
	}
	return PFMLIB_ERR_NOTFOUND;
found:
	*next = i;
	return PFMLIB_SUCCESS;
}

int
pfm_find_event_mask(unsigned int ev, const char *str, unsigned int *mask_idx)
{
	unsigned int i, num_masks = 0;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (str == NULL || mask_idx == NULL || ev >= pfm_current->pme_count)
		return PFMLIB_ERR_INVAL;

	num_masks = pfm_num_masks(ev);
	for (i = 0; i < num_masks; i++) {
		if (!strcasecmp(pfm_current->get_event_mask_name(ev, i), str)) {
			*mask_idx = i;
			return PFMLIB_SUCCESS;
		}
	}
	return PFMLIB_ERR_UMASK;
}

int
pfm_get_event_name(unsigned int i, char *name, size_t maxlen)
{
	size_t l, j;

	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i >= pfm_current->pme_count || name == NULL || maxlen < 1) return PFMLIB_ERR_INVAL;

	strncpy(name, pfm_current->get_event_name(i), maxlen-1);

	name[maxlen-1] = '\0';
	l = strlen(name);
	for(j=0; j < l; j++)
		name[j] = (char)toupper(name[j]);

	return PFMLIB_SUCCESS;
}

int
pfm_get_event_code(unsigned int i, int *code)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i >= pfm_current->pme_count || code == NULL) return PFMLIB_ERR_INVAL;

	return pfm_current->get_event_code(i, PFMLIB_CNT_FIRST, code);

}

int
pfm_get_event_code_counter(unsigned int i, unsigned int cnt, int *code)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i >= pfm_current->pme_count || code == NULL) return PFMLIB_ERR_INVAL;

	return pfm_current->get_event_code(i, cnt, code);
}

int
pfm_get_event_counters(unsigned int i, pfmlib_regmask_t *counters)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i >= pfm_current->pme_count) return PFMLIB_ERR_INVAL;

	pfm_current->get_event_counters(i, counters);

	return PFMLIB_SUCCESS;
}

int
pfm_get_event_mask_name(unsigned int ev, unsigned int mask, char *name, size_t maxlen)
{
	char *n;
	unsigned int num;
	size_t l, j;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (ev >= pfm_current->pme_count || name == NULL || maxlen < 1)
		return PFMLIB_ERR_INVAL;

	num = pfm_num_masks(ev);
	if (num == 0)
		return PFMLIB_ERR_NOTSUPP;

	if (mask >= num)
		return PFMLIB_ERR_INVAL;

	n = pfm_current->get_event_mask_name(ev, mask);
	if (n == NULL)
		return PFMLIB_ERR_INVAL;

	l = strlen(n);
	if (l >= (maxlen-1))
		return PFMLIB_ERR_FULL;

	strcpy(name, n);

	/*
	 * present nice uniform names
	 */
	l = strlen(name);
	for(j=0; j < l; j++)
		if (islower(name[j]))
				name[j] = (char)toupper(name[j]);
	return PFMLIB_SUCCESS;
}

int
pfm_get_num_events(unsigned int *count)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (count == NULL) return PFMLIB_ERR_INVAL;

	*count = pfm_current->pme_count;

	return PFMLIB_SUCCESS;
}

int
pfm_get_num_event_masks(unsigned int ev, unsigned int *count)
{
	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (ev >= pfm_current->pme_count || count == NULL)
		return PFMLIB_ERR_INVAL;

	*count = pfm_num_masks(ev);

	return PFMLIB_SUCCESS;
}

#if 0
/*
 * check that the unavailable PMCs registers correspond
 * to implemented PMC registers
 */
static int
pfm_check_unavail_pmcs(pfmlib_regmask_t *pmcs)
{
	pfmlib_regmask_t impl_pmcs;
	pfm_current->get_impl_pmcs(&impl_pmcs);
	unsigned int i;

	for (i=0; i < PFMLIB_REG_BV; i++) {
		if ((pmcs->bits[i] & impl_pmcs.bits[i]) != pmcs->bits[i])
			return PFMLIB_ERR_INVAL;
	}
	return PFMLIB_SUCCESS;
}
#endif


/*
 * we do not check if pfp_unavail_pmcs contains only implemented PMC
 * registers. In other words, invalid registers are ignored
 */
int
pfm_dispatch_events(pfmlib_input_param_t *inp, void *model_in, pfmlib_output_param_t *outp, void *model_out)
{
	unsigned count;
	unsigned int i;
	int ret;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (inp == NULL || outp == NULL)
		return PFMLIB_ERR_INVAL;

	/*
	 * the default priv level must be set to something
	 */
	if (inp->pfp_dfl_plm == 0)
		return PFMLIB_ERR_INVAL;

	if (inp->pfp_event_count >= PFMLIB_MAX_PMCS)
		return PFMLIB_ERR_INVAL;

	count = inp->pfp_event_count;
	if (count > pfm_current->num_cnt)
		return PFMLIB_ERR_NOASSIGN;

	/*
	 * check that event and unit masks descriptors are correct
	 */
	for (i=0; i < count; i++) {
		ret = __pfm_check_event(inp->pfp_events+i);
		if (ret != PFMLIB_SUCCESS)
			return ret;
	}
	/* reset output data structure */
	memset(outp, 0, sizeof(*outp));

	return pfm_current->dispatch_events(inp, model_in, outp, model_out);
}

/*
 * more or less obosleted by pfm_get_impl_counters()
 */
int
pfm_get_num_counters(unsigned int *num)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (num == NULL) return PFMLIB_ERR_INVAL;
	
	*num = pfm_current->num_cnt;

	return PFMLIB_SUCCESS;
}

int
pfm_get_num_pmcs(unsigned int *num)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (num == NULL) return PFMLIB_ERR_INVAL;
	
	*num = pfm_current->pmc_count;

	return PFMLIB_SUCCESS;
}

int
pfm_get_num_pmds(unsigned int *num)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (num == NULL) return PFMLIB_ERR_INVAL;
	
	*num = pfm_current->pmd_count;

	return PFMLIB_SUCCESS;
}

int
pfm_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;
	if (impl_pmcs == NULL) return PFMLIB_ERR_INVAL;

	memset(impl_pmcs , 0, sizeof(*impl_pmcs));

	pfm_current->get_impl_pmcs(impl_pmcs);

	return PFMLIB_SUCCESS;
}

int
pfm_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;
	if (impl_pmds == NULL) return PFMLIB_ERR_INVAL;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	pfm_current->get_impl_pmds(impl_pmds);

	return PFMLIB_SUCCESS;
}

int
pfm_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;
	if (impl_counters == NULL) return PFMLIB_ERR_INVAL;

	memset(impl_counters, 0, sizeof(*impl_counters));

	pfm_current->get_impl_counters(impl_counters);

	return PFMLIB_SUCCESS;
}

int
pfm_get_hw_counter_width(unsigned int *width)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;
	if (width == NULL) return PFMLIB_ERR_INVAL;

	pfm_current->get_hw_counter_width(width);

	return PFMLIB_SUCCESS;
}


/* sorry, only English supported at this point! */
static char *pfmlib_err_list[]=
{
	"success",
	"not supported",
	"invalid parameters",
	"pfmlib not initialized",
	"event not found",
	"cannot assign events to counters",
	"buffer is full or too small",
	"event used more than once",
	"invalid model specific magic number",
	"invalid combination of model specific features",
	"incompatible event sets",
	"incompatible events combination",
	"too many events or unit masks",
	"code range too big",
	"empty code range",
	"invalid code range",
	"too many code ranges",
	"invalid data range",
	"too many data ranges",
	"not supported by host cpu",
	"code range is not bundle-aligned",
	"code range requires some flags in rr_flags",
	"invalid or missing unit mask",
	"out of memory"
};
static size_t pfmlib_err_count = sizeof(pfmlib_err_list)/sizeof(char *);

char *
pfm_strerror(int code)
{
	code = -code;
	if (code <0 || code >= pfmlib_err_count) return "unknown error code";
	return pfmlib_err_list[code];
}

int
pfm_get_version(unsigned int *version)
{
	if (version == NULL) return PFMLIB_ERR_INVAL;
	*version = PFMLIB_VERSION;
	return 0;
}

int
pfm_get_max_event_name_len(size_t *len)
{
	unsigned int i, j, num_masks;
	size_t max = 0, l;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;
	if (len == NULL)
		return PFMLIB_ERR_INVAL;

	for(i=0; i < pfm_current->pme_count; i++) {
		l = strlen(pfm_current->get_event_name(i));
		if (l > max) max = l;

		num_masks = pfm_num_masks(i);
		/*
		 * we need to add up all length because unit masks can
		 * be combined typically. We add 1 to account for ':'
		 * which is inserted as the unit mask separator
		 */
		for (j = 0; j < num_masks; j++) {
			l += 1 + strlen(pfm_current->get_event_mask_name(i, j));
		}
		if (l > max) max = l;
	}
	*len = max;
	return PFMLIB_SUCCESS;
}

/*
 * return the index of the event that counts elapsed cycles
 */
int
pfm_get_cycle_event(pfmlib_event_t *e)
{
	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;
	if (e == NULL)
		return PFMLIB_ERR_INVAL;

	if (!pfm_current->get_cycle_event)
		return PFMLIB_ERR_NOTSUPP;

	memset(e, 0, sizeof(*e));

	return pfm_current->get_cycle_event(e);
}

/*
 * return the index of the event that retired instructions
 */
int
pfm_get_inst_retired_event(pfmlib_event_t *e)
{
	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;
	if (e == NULL)
		return PFMLIB_ERR_INVAL;

	if (!pfm_current->get_inst_retired_event)
		return PFMLIB_ERR_NOTSUPP;

	memset(e, 0, sizeof(*e));

	return pfm_current->get_inst_retired_event(e);
}

int
pfm_get_event_description(unsigned int i, char **str)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (i >= pfm_current->pme_count || str == NULL) return PFMLIB_ERR_INVAL;

	if (pfm_current->get_event_desc == NULL) {
		*str = strdup("no description available");
		return PFMLIB_SUCCESS;
	}
	return pfm_current->get_event_desc(i, str);
}

int
pfm_get_event_mask_description(unsigned int event_idx, unsigned int mask_idx, char **desc)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (event_idx >= pfm_current->pme_count || desc == NULL) return PFMLIB_ERR_INVAL;

	if (pfm_current->get_event_mask_desc == NULL) {
		*desc = strdup("no description available");
		return PFMLIB_SUCCESS;
	}
	if (mask_idx >= pfm_current->get_num_event_masks(event_idx))
		return PFMLIB_ERR_INVAL;

	return pfm_current->get_event_mask_desc(event_idx, mask_idx, desc);
}

int
pfm_get_event_mask_code(unsigned int event_idx, unsigned int mask_idx, unsigned int *code)
{
	if (PFMLIB_INITIALIZED() == 0) return PFMLIB_ERR_NOINIT;

	if (event_idx >= pfm_current->pme_count || code == NULL) return PFMLIB_ERR_INVAL;

	if (pfm_current->get_event_mask_code == NULL) {
		*code = 0;
		return PFMLIB_SUCCESS;
	}
	if (mask_idx >= pfm_current->get_num_event_masks(event_idx))
		return PFMLIB_ERR_INVAL;

	return pfm_current->get_event_mask_code(event_idx, mask_idx, code);
}
	
int
pfm_get_full_event_name(pfmlib_event_t *e, char *name, size_t maxlen)
{
	char *n;
	size_t l, j;
	int ret;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (e == NULL || name == NULL || maxlen < 1)
		return PFMLIB_ERR_INVAL;

	ret = __pfm_check_event(e);
	if (ret != PFMLIB_SUCCESS)
		return ret;

	/*
	 * make sure the string is at least empty
	 * important for programs that do not check return value
	 * from this function!
	 */
	*name = '\0';

	n = pfm_current->get_event_name(e->event);
	l = strlen(n);
	if (l > (maxlen-1))
		return PFMLIB_ERR_FULL;

	strcpy(name, n);
	maxlen -= l + 1;
	for(j=0; j < e->num_masks; j++) {
		n = pfm_current->get_event_mask_name(e->event, e->unit_masks[j]);
		l = strlen(n);
		if (l > (maxlen-1))
			return PFMLIB_ERR_FULL;

		strcat(name, ":");
		strcat(name, n);
		maxlen -= l + 1;
	}
	/*
	 * present nice uniform names
	 */
	l = strlen(name);
	for(j=0; j < l; j++)
		if (islower(name[j]))
				name[j] = (char)toupper(name[j]);
	return PFMLIB_SUCCESS;
}
	
int
pfm_find_full_event(const char *v, pfmlib_event_t *e)
{
	char *str, *p, *q;
	unsigned int j, mask;
	int ret = PFMLIB_SUCCESS;

	if (PFMLIB_INITIALIZED() == 0)
		return PFMLIB_ERR_NOINIT;

	if (v == NULL || e == NULL)
		return PFMLIB_ERR_INVAL;

	memset(e, 0, sizeof(*e));

	/*
	 * must copy string because we modify it when parsing
	 */
	str = strdup(v);
	if (!str)
		return PFMLIB_ERR_NOMEM;

	/*
	 * find event. this function ignores ':' separator
	 */
	ret = pfm_find_event_byname(str, &e->event);
	if (ret)
		goto error;

	/*
	 * get number of unit masks for event
	 */
	j = pfm_num_masks(e->event);

	/*
	 * look for colon (unit mask separator)
	 */
	p = strchr(str, ':');

	/*
	 * error if:
	 * 	- event has unit masks and non is passed
	 * 	- event has no unit mask and at least one is passed
	 */
	if ((!p && j) || (p && !j)) {
		ret = PFMLIB_ERR_UMASK;
		goto error;
	}

	/*
	 * if no unit mask, then we are done
	 */
	if (!j) {
		free(str);
		return PFMLIB_SUCCESS;
	}

	p++;

	for( q = p; q ; p = q) {
		q = strchr(p,':');
		if (q)
			*q++ = '\0';

		ret = pfm_find_event_mask(e->event, p, &mask);
		if (ret != PFMLIB_SUCCESS)
			break;

		/* avoid duplicates */
		for(j=0; j < e->num_masks; j++)
			if (e->unit_masks[j] == mask)
				break;

		if (j == e->num_masks) {
			if (e->num_masks == PFMLIB_MAX_MASKS_PER_EVENT) {
				ret = PFMLIB_ERR_TOOMANY;
				break;
			}
			e->unit_masks[e->num_masks] = mask;
			e->num_masks++;
		}
	}
error:
	free(str);
	return ret;
}
