/*
 * pfmlib_common.c: set of functions common to all PMU models
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
 *
 * Based on:
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
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* for getline */
#endif
#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>

#include <perfmon/pfmlib.h>

#include "pfmlib_priv.h"

static pfmlib_pmu_t *pfmlib_pmus[]=
{

#ifdef CONFIG_PFMLIB_ARCH_IA64
#if 0
	&montecito_support,
	&itanium2_support,
	&itanium_support,
	&generic_ia64_support,	/* must always be last for IA-64 */
#endif
#endif

#ifdef CONFIG_PFMLIB_ARCH_X86
	//&pentium4_support,
	&intel_pii_support,
	&intel_ppro_support,
	&intel_p6_support,
	&intel_pm_support,
	&intel_coreduo_support,
	&amd64_k7_support,
	&amd64_k8_revb_support,
	&amd64_k8_revc_support,
	&amd64_k8_revd_support,
	&amd64_k8_reve_support,
	&amd64_k8_revf_support,
	&amd64_k8_revg_support,
	&amd64_fam10h_barcelona_support,
	&amd64_fam10h_shanghai_support,
	&amd64_fam10h_istanbul_support,
	&intel_core_support,
	&intel_atom_support,
	&intel_nhm_support,
	&intel_nhm_ex_support,
	&intel_nhm_unc_support,
	&intel_wsm_support,
	&intel_wsm_unc_support,
	&intel_x86_arch_support, /* must always be last for x86 */
#endif

#ifdef CONFIG_PFMLIB_ARCH_MIPS64
	&generic_mips64_support,
#endif

#ifdef CONFIG_PFMLIB_ARCH_SICORTEX
	&sicortex_support,
#endif

#ifdef CONFIG_PFMLIB_ARCH_POWERPC
	&power4_support,
	&ppc970_support,
	&ppc970mp_support,
	&power5_support,
	&power5p_support,
	&power6_support,
	&power7_support,
#endif

#ifdef CONFIG_PFMLIB_ARCH_SPARC
	&sparc_support,
#endif

#ifdef CONFIG_PFMLIB_CELL
	&cell_support,
#endif

#ifdef CONFIG_PFMLIB_ARCH_ARM
	&arm_cortex_a8_support,
	&arm_cortex_a9_support,
#endif

#ifdef __linux__
	&perf_event_support,
#endif
};
#define PFMLIB_NUM_PMUS	(sizeof(pfmlib_pmus)/sizeof(pfmlib_pmu_t *))

/*
 * pfm_pmu_t mapping to pfmlib_pmus[] index
 */
static pfmlib_pmu_t *pfmlib_pmus_map[PFM_PMU_MAX];

#define pfmlib_for_each_pmu_event(p, e) \
	for(e=(p)->get_event_first((p)); e != -1; e = (p)->get_event_next((p), e))

#define for_each_pmu_event_attr(u, i) \
	for((u)=0; (u) < (i)->nattrs; (u) = (u)+1)

#define pfmlib_for_each_pmu(x) \
	for((x)= 0 ; (x) < PFMLIB_NUM_PMUS; (x)++)

pfmlib_config_t pfm_cfg;

void
__pfm_dbprintf(const char *fmt, ...)
{
	va_list ap;

	if (pfm_cfg.debug == 0)
		return;

	va_start(ap, fmt);
	vfprintf(pfm_cfg.fp, fmt, ap);
	va_end(ap);
}

void
__pfm_vbprintf(const char *fmt, ...)
{
	va_list ap;

	if (pfm_cfg.verbose == 0)
		return;

	va_start(ap, fmt);
	vfprintf(pfm_cfg.fp, fmt, ap);
	va_end(ap);
}

/*
 * append fmt+args to str such that the string is no
 * more than max characters incl. null termination
 */
void
pfmlib_strconcat(char *str, size_t max, const char *fmt, ...)
{
	va_list ap;
	size_t len, todo;

	len = strlen(str);
	todo = max - strlen(str);
	va_start(ap, fmt);
	vsnprintf(str+len, todo, fmt, ap);
	va_end(ap);
}

static inline int
pfmlib_pmu_active(pfmlib_pmu_t *pmu)
{
        return pmu && (pmu->flags & PFMLIB_PMU_FL_ACTIVE);
}

static inline pfm_pmu_t
idx2pmu(int idx)
{
	return (idx >> PFMLIB_PMU_SHIFT) & PFMLIB_PMU_MASK;
}

static inline pfmlib_pmu_t *
pmu2pmuidx(pfm_pmu_t pmu)
{
	if (pmu < PFM_PMU_NONE || pmu >= PFM_PMU_MAX)
		return NULL;

	return pfmlib_pmus_map[pmu];
}

/*
 * external opaque idx -> PMU + internal idx
 */
static pfmlib_pmu_t *
pfmlib_idx2pidx(int idx, int *pidx)
{
	pfmlib_pmu_t *pmu;
	int pmu_id;

	if (PFMLIB_INITIALIZED() == 0)
		return NULL;

	if (idx < 0)
		return NULL;

	pmu_id = (idx >> PFMLIB_PMU_SHIFT) & PFMLIB_PMU_MASK;

	pmu = pmu2pmuidx(pmu_id);
	if (!pmu)
		return NULL;

	*pidx = idx & PFMLIB_PMU_PIDX_MASK;

	if (!pmu->event_is_valid(pmu, *pidx))
		return NULL;

	return pmu;
}

static int
pfmlib_valid_attr(pfmlib_pmu_t *pmu, int pidx, int attr)
{
	pfm_event_info_t info;
	int ret;

	memset(&info, 0, sizeof(info));
	ret = pmu->get_event_info(pmu, pidx, &info);
	if (ret != PFM_SUCCESS)
		return 0;

	return attr >= 0 && attr < info.nattrs;
}

/*
 * check environment variables for:
 *  LIBPFM_VERBOSE : enable verbose output (must be 1)
 *  LIBPFM_DEBUG   : enable debug output (must be 1)
 */
static void
pfmlib_init_debug_env(void)
{
	char *str;

	pfm_cfg.fp = stderr;

	str = getenv("LIBPFM_VERBOSE");
	if (str && isdigit((int)*str))
		pfm_cfg.verbose = *str - '0';

	str = getenv("LIBPFM_DEBUG");
	if (str && isdigit((int)*str))
		pfm_cfg.debug = *str - '0';

	str = getenv("LIBPFM_DEBUG_STDOUT");
	if (str)
		pfm_cfg.fp = stdout;

	pfm_cfg.forced_pmu = getenv("LIBPFM_FORCE_PMU");
}

static int
pfmlib_pmu_sanity_checks(pfmlib_pmu_t *p)
{
	/*
	 * check event can be encoded
	 */
	if (p->pme_count >= (1<< PFMLIB_PMU_SHIFT)) {
		DPRINT("too many events for %s\n", p->desc);
		return PFM_ERR_NOTSUPP;
	}

	return PFM_SUCCESS;
}

int
pfmlib_build_fstr(pfmlib_event_desc_t *e, char **fstr)
{
	/* nothing to do */
	if (!fstr)
		return PFM_SUCCESS;

	*fstr = malloc(strlen(e->fstr) + 2 + strlen(e->pmu->name) + 1);
	if (*fstr)
		sprintf(*fstr, "%s::%s", e->pmu->name, e->fstr);

	return fstr ? PFM_SUCCESS : PFM_ERR_NOMEM;
}

static int
pfmlib_pmu_activate(pfmlib_pmu_t *p)
{
	int ret;

	ret = pfmlib_pmu_sanity_checks(p);
	if (ret != PFM_SUCCESS)
		return ret;

	if (p->pmu_init) {
		ret = p->pmu_init(p);
		if (ret != PFM_SUCCESS)
			return ret;
	}

	p->flags |= PFMLIB_PMU_FL_ACTIVE;

	DPRINT("activated %s\n", p->desc);

	return PFM_SUCCESS;	
}

static inline int
pfmlib_match_forced_pmu(const char *name)
{
	const char *p;
	size_t l;

	/* skip any lower level specifier */
	p = strchr(pfm_cfg.forced_pmu, ',');
	if (p)
		l = p - pfm_cfg.forced_pmu;
	else
		l = strlen(pfm_cfg.forced_pmu);

	return !strncasecmp(name, pfm_cfg.forced_pmu, l);
}

static int
pfmlib_init_pmus(void)
{
	pfmlib_pmu_t *p;
	int i, ret, n = 0;
	int nsuccess = 0;
	
	if (pfm_cfg.forced_pmu) {
		char *p;
		p = strchr(pfm_cfg.forced_pmu, ',');
		n = p ? p - pfm_cfg.forced_pmu : strlen(pfm_cfg.forced_pmu);
	}

	/*
	 * activate all detected PMUs
	 * when forced, only the designated PMU
	 * is setup and activated
	 */
	for(i=0; i < PFMLIB_NUM_PMUS; i++) {

		p = pfmlib_pmus[i];

		DPRINT("trying %s\n", p->desc);

		ret = PFM_SUCCESS;

		if (!pfm_cfg.forced_pmu)
			ret = p->pmu_detect(p);
		else if (!pfmlib_match_forced_pmu(p->name))
			ret = PFM_ERR_NOTSUPP;

		pfmlib_pmus_map[p->pmu] = p;

		if (ret != PFM_SUCCESS)
			continue;

		ret = pfmlib_pmu_activate(p);
		if (ret == PFM_SUCCESS)
			nsuccess++;

		if (pfm_cfg.forced_pmu) {
			__pfm_vbprintf("PMU forced to %s (%s) : %s\n",
					p->name,
					p->desc,
					ret == PFM_SUCCESS ? "success" : "failure");
			return ret;
		}
	}
	DPRINT("%d PMU detected out of %d supported\n", nsuccess, PFMLIB_NUM_PMUS);
	return PFM_SUCCESS;
}

int
pfm_initialize(void)
{
	int ret;
	/*
	 * not atomic
	 */
	if (pfm_cfg.initdone)
		return PFM_SUCCESS;

	/*
	 * generic sanity checks
	 */
	if (PFM_PMU_MAX & (~PFMLIB_PMU_MASK)) {
		DPRINT("PFM_PMU_MAX exceeds PFMLIB_PMU_MASK\n");	
		return PFM_ERR_NOTSUPP;
	}

	pfmlib_init_debug_env();

	ret = pfmlib_init_pmus();
	if (ret == PFM_SUCCESS)
		pfm_cfg.initdone = 1;

	return ret;
}

void
pfm_terminate(void)
{
	pfmlib_pmu_t *pmu;
	int id;

	if (PFMLIB_INITIALIZED() == 0)
		return;

	pfmlib_for_each_pmu(id) {
		pmu = pfmlib_pmus[id];
		if (!pfmlib_pmu_active(pmu))
			continue;
		if (pmu->pmu_terminate)
			pmu->pmu_terminate(pmu);
	}
	pfm_cfg.initdone = 0;
}

int
pfm_find_event(const char *str)
{
	pfmlib_event_desc_t e;
	int ret;

	if (PFMLIB_INITIALIZED() == 0)
		return PFM_ERR_NOINIT;

	if (!str)
		return PFM_ERR_INVAL;

	ret = pfmlib_parse_event(str, &e);
	if (ret == PFM_SUCCESS)
		return pfmlib_pidx2idx(e.pmu, e.event);

	return ret;
}

static int
pfmlib_sanitize_event(pfmlib_event_desc_t *d)
{
	int i, j;

	/*
	 * fail if duplicate attributes are found
	 */
	for(i=0; i < d->nattrs; i++) {
		for(j=i+1; j < d->nattrs; j++) {
			if (d->attrs[i].id == d->attrs[j].id
			    && d->attrs[i].type == d->attrs[j].type)
				return PFM_ERR_ATTR_SET;
		}
	}
	return PFM_SUCCESS;
}

static int
pfmlib_parse_event_attr(char *str, pfmlib_pmu_t *pmu, int idx, int nattrs, pfmlib_event_desc_t *d)
{
	pfm_event_attr_info_t ainfo;
	char *s, *p, *q, *endptr;
	char yes[2] = "y";
	pfm_attr_t type;
	int a, has_val;
	int na, ret;

	s = str;
	na = d->nattrs;

	while(s) {
		p = strchr(s, ':');
		if (p)
			*p++ = '\0';

		q = strchr(s, '=');
		if (q)
			*q++ = '\0';

		has_val = !!q;

		for(a = 0; a < nattrs; a++) {
			ret = pmu->get_event_attr_info(pmu, idx, a, &ainfo);
			if (ret != PFM_SUCCESS)
				return ret;

			if (!strcasecmp(ainfo.name, s))
				goto found_attr;
		}
		{ pfm_event_info_t einfo;
		  pmu->get_event_info(pmu, idx, &einfo);
		  DPRINT("attr=%s not found for event %s\n", s, einfo.name);
		}
		return PFM_ERR_ATTR;
found_attr:
		if (ainfo.equiv) {
			char *z;

			/* cannot have equiv for attributes with value */
			if (has_val)
				return PFM_ERR_ATTR_VAL;
			/* copy because it is const */
			z = strdup(ainfo.equiv);
			if (!z)
				return PFM_ERR_NOMEM;

			ret = pfmlib_parse_event_attr(z, pmu, idx, nattrs, d);

			free(z);

			if (ret != PFM_SUCCESS)
				return ret;
			s = p;
			na = d->nattrs;
			continue;
		}
		type = ainfo.type;
		/*
		 * we tolerate missing value for boolean attributes.
		 * Presence of the attribute is equivalent to
		 * attr=1, i.e., attribute is set
		 */
		if (type != PFM_ATTR_UMASK && !has_val) {
			if (type != PFM_ATTR_MOD_BOOL)
				return PFM_ERR_ATTR_VAL;
			has_val = 1; s = yes; /* no const */
			goto handle_bool;
		}

		d->attrs[na].ival = 0;
		if (type == PFM_ATTR_UMASK && has_val)
			return PFM_ERR_ATTR_VAL;

		if (has_val) {
			s = q;
			ret = PFM_ERR_ATTR_VAL;
			if (!strlen(s))
				goto error;
handle_bool:
			if (na == PFMLIB_MAX_EVENT_ATTRS) {
				DPRINT("too many attributes\n");
				ret = PFM_ERR_TOOMANY;
				goto error;
			}

			endptr = NULL;
			switch(type) {
			case PFM_ATTR_UMASK:
				/* unit mask has no value */
				goto error;
			case PFM_ATTR_MOD_BOOL:
				if (strlen(s) > 1)
					goto error;

				if (tolower((int)*s) == 'y'
				    || tolower((int)*s) == 't' || *s == '1')
					d->attrs[na].ival = 1;
				else if (tolower((int)*s) == 'n'
					 || tolower((int)*s) == 'f' || *s == '0')
					d->attrs[na].ival = 0;
				else
					goto error;
				break;
			case PFM_ATTR_MOD_INTEGER:
				d->attrs[na].ival = strtoull(s, &endptr, 0);
				if (*endptr != '\0')
					goto error;
				break;
			default:
				goto error;
			}
		}
		DPRINT("na=%d id=%d type=%d idx=%d nattrs=%d name=%s\n", na, ainfo.idx, type, ainfo.idx, nattrs, ainfo.name);
		d->attrs[na].id = ainfo.idx;
		d->attrs[na].type = type;
		d->nattrs++;
		na++;
		s = p;
	}
	ret = PFM_SUCCESS;
error:
	return ret;
}

static int
pfmlib_parse_equiv_event(pfmlib_pmu_t *pmu, const char *event, pfmlib_event_desc_t *d)
{
	pfm_event_info_t einfo;
	char *str, *s, *p;
	int i;
	int ret;

	/*
	 * create copy because string is const
	 */
	s = str = strdup(event);
	if (!str)
		return PFM_ERR_NOMEM;

	p = strchr(s, ':');
	if (p)
		*p++ = '\0';

	pfmlib_for_each_pmu_event(pmu, i) {
		ret = pmu->get_event_info(pmu, i, &einfo);
		if (ret != PFM_SUCCESS)
			goto error;
		if (!strcasecmp(einfo.name, s))
			goto found;
	}
	free(str);
	return PFM_ERR_NOTFOUND;
found:
	d->pmu = pmu;
	d->event = i; /* private index */

	ret = pfmlib_parse_event_attr(p, pmu, i, einfo.nattrs, d);
error:
	free(str);
	return ret;
}

int
pfmlib_parse_event(const char *event, pfmlib_event_desc_t *d)
{
	pfm_event_info_t einfo;
	char *str, *s, *p;
	pfmlib_pmu_t *pmu;
	int i, id;
	const char *pname = NULL;
	int ret;

	/*
	 * create copy because string is const
	 */
	s = str = strdup(event);
	if (!str)
		return PFM_ERR_NOMEM;

	/*
	 * ignore everything passed after a comma
	 * (simplify dealing with const event list)
	 *
	 * safe to do before pname, because now
	 * PMU name cannot have commas in them.
	 */
	p = strchr(s, ',');
	if (p)
		*p = '\0';

	/* check for optional PMU name */
	p = strchr(s, ':');
	if (p && *(p+1) == ':') {
		*p = '\0';
		pname = s;
		s = p + 2;
		p = strchr(s, ':');
	}
	if (p)
		*p++ = '\0';
	/*
	 * for each pmu
	 */
	pfmlib_for_each_pmu(id) {
		pmu = pfmlib_pmus[id];
		/*
		 * if no explicit PMU name is given, then
		 * only look for active (detected) PMU models
		 */
		if (!pname && !pfmlib_pmu_active(pmu))
			continue;

		/*
		 * check for requested PMU name,
		 */
		if (pname && strcasecmp(pname, pmu->name))
			continue;
		/*
		 * for each event
		 */
		pfmlib_for_each_pmu_event(pmu, i) {
			ret = pmu->get_event_info(pmu, i, &einfo);
			if (ret != PFM_SUCCESS)
				goto error;
			if (!strcasecmp(einfo.name, s))
				goto found;
		}
	}
	free(str);
	return PFM_ERR_NOTFOUND;
found:
	/*
	 * handle equivalence
	 */
	if (einfo.equiv) {
		ret = pfmlib_parse_equiv_event(pmu, einfo.equiv, d);
		if (ret == PFM_SUCCESS) {
			i = d->event;
			pmu->get_event_info(pmu, i, &einfo);
		}
	} else {
		d->pmu = pmu;
		d->event = i; /* private index */
	}
	ret = pfmlib_parse_event_attr(p, pmu, i, einfo.nattrs, d);
	if (ret == PFM_SUCCESS)
		ret = pfmlib_sanitize_event(d);
error:
	free(str);
	return ret;
}

/*
 * total number of events
 */
int
pfm_get_nevents(void)
{
	pfmlib_pmu_t *pmu;
	int id, total = 0;

	pfmlib_for_each_pmu(id) {
		pmu = pfmlib_pmus[id];
		total += pmu->pme_count;
	}
	return total;
}

/* sorry, only English supported at this point! */
static const char *pfmlib_err_list[]=
{
	"success",
	"not supported",
	"invalid parameters",
	"pfmlib not initialized",
	"event not found",
	"invalid combination of model specific features",
	"invalid or missing unit mask",
	"out of memory",
	"invalid event attribute",
	"invalid event attribute value",
	"attribute value already set",
	"too many parameters",
	"parameter is too small",
};
static size_t pfmlib_err_count = sizeof(pfmlib_err_list)/sizeof(char *);

const char *
pfm_strerror(int code)
{
	code = -code;
	if (code <0 || code >= pfmlib_err_count)
		return "unknown error code";

	return pfmlib_err_list[code];
}

int
pfm_get_version(void)
{
	return LIBPFM_VERSION;
}

int
pfm_get_event_next(int idx)
{
	pfmlib_pmu_t *pmu;
	int i = 0, pidx, px;

	pmu = pfmlib_idx2pidx(idx, &pidx);
	if (!pmu)
		return -1;

	pidx = pmu->get_event_next(pmu, pidx);
	if (pidx != -1)
		return pfmlib_pidx2idx(pmu, pidx);

	px = idx2pmu(idx);

	/*
	 * ran out of event, move to next PMU
	 */
retry:
	for(; i < PFMLIB_NUM_PMUS; i++) {
		pmu = pfmlib_pmus[i];
		if (pmu->pmu == px)
			break;
	}
	if (i >= (PFMLIB_NUM_PMUS-1))
		return -1;

	pmu = pfmlib_pmus[++i];
	pidx = pmu->get_event_first(pmu);
	if (pidx == -1) {
		px = pmu->pmu;
		goto retry;
	}
	return pfmlib_pidx2idx(pmu, pidx);
}

int
pfm_get_event_first(void)
{
	pfmlib_pmu_t *pmu;
	int id, pidx;

	/* scan all compiled in PMU models */
	pfmlib_for_each_pmu(id) {
		pmu = pfmlib_pmus[id];
		pidx = pmu->get_event_first(pmu);
		if (pidx != -1)
			return pfmlib_pidx2idx(pmu, pidx);
	}
	return -1;
}

int
pfmlib_get_event_encoding(pfmlib_event_desc_t *e, uint64_t **codes, int *count, pfmlib_perf_attr_t *attrs)
{
	pfmlib_pmu_t *pmu = e->pmu;
	uint64_t *local_codes;
	int ret, local_count;
	/*
	 * too small, must reallocate
	 */
	if (*codes && *count < pmu->max_encoding)
		return PFM_ERR_TOOSMALL;

	/*
	 * count but no codes
	 */
	if (*count && !*codes)
		return PFM_ERR_INVAL;

	local_codes = *codes;
	local_count = *count;

	if (!*codes) {
		local_codes = calloc(pmu->max_encoding, sizeof(uint64_t));
		if (!local_codes)
			return PFM_ERR_NOMEM;
		local_count = pmu->max_encoding;
	}

	/* count may be adjusted to match need for event */
	ret = pmu->get_event_encoding(pmu, e, local_codes, &local_count, attrs);
	if (ret != PFM_SUCCESS) {
		if (!*codes)
			free(local_codes);
	} else {
		*codes = local_codes;
		*count = local_count;
	}
	return ret;
}

int
pfm_get_event_encoding(const char *str, int dfl_plm, char **fstr, int *idx, uint64_t **codes, int *count)
{
	pfmlib_event_desc_t e;
	uint64_t *orig_codes = NULL;
	int orig_count;
	int ret;

	if (PFMLIB_INITIALIZED() == 0)
		return PFM_ERR_NOINIT;

	if (!(str && count && codes))
		return PFM_ERR_INVAL;

	/* must provide default priv level */
	if (dfl_plm < 1)
		return PFM_ERR_INVAL;
	
	memset(&e, 0, sizeof(e));

	e.dfl_plm = dfl_plm;

	ret = pfmlib_parse_event(str, &e);
	if (ret != PFM_SUCCESS)
		return ret;
	
	orig_count = *count;
	if (codes)
		orig_codes = *codes;

	ret = pfmlib_get_event_encoding(&e, codes, count, NULL);
	if (ret != PFM_SUCCESS)
		return ret;
	/*
	 * return opaque event identifier
	 */
	if (idx)
		*idx = pfmlib_pidx2idx(e.pmu, e.event);

	ret = pfmlib_build_fstr(&e, fstr);
	if (ret != PFM_SUCCESS) {
		if (orig_codes != *codes) {
			free(codes);
			*count = orig_count;
			*codes = orig_codes;
		}
	}
	return ret;
}

static int
pfmlib_validate_encoding(char *buf, int plm)
{
	uint64_t *codes = NULL;
	int count = 0, ret;

	ret = pfm_get_event_encoding(buf, plm, NULL, NULL, &codes, &count);
	if (ret == PFM_SUCCESS) {
		int i;
		DPRINT("%s ", buf);
		for(i=0; i < count; i++)
			__pfm_dbprintf(" %#"PRIx64, codes[i]);
		__pfm_dbprintf("\n");
	}
	if (codes)
		free(codes);

	return ret;
}

static int
pfmlib_pmu_validate_encoding(pfmlib_pmu_t *pmu, FILE *fp)
{
	pfm_event_info_t einfo;
	pfm_event_attr_info_t ainfo;
	char *buf;
	size_t maxlen = 0, len;
	int i, u, n = 0, um;
	int ret, retval =PFM_SUCCESS;

	pfmlib_for_each_pmu_event(pmu, i) {
		ret = pmu->get_event_info(pmu, i, &einfo);
		if (ret != PFM_SUCCESS)
			return ret;

		len = strlen(einfo.name);
		if (len > maxlen)
			maxlen = len;

		for_each_pmu_event_attr(u, &einfo) {
			ret = pmu->get_event_attr_info(pmu, i, u, &ainfo);
			if (ret != PFM_SUCCESS)
				return ret;

			if (ainfo.type != PFM_ATTR_UMASK)
				continue;

			len = strlen(einfo.name) + strlen(ainfo.name);
			if (len > maxlen)
				maxlen = len;
		}
	}
	/* 2 = ::, 1=:, 1=eol */
	maxlen += strlen(pmu->name) + 2 + 1 + 1;
	buf = malloc(maxlen);
	if (!buf)
		return PFM_ERR_NOMEM;

	fprintf(fp, "\tcheck encoding: "); fflush(fp);
	pfmlib_for_each_pmu_event(pmu, i) {
		ret = pmu->get_event_info(pmu, i, &einfo);
		if (ret != PFM_SUCCESS)
			return ret;

		um = 0;
		for_each_pmu_event_attr(u, &einfo) {
			ret = pmu->get_event_attr_info(pmu, i, u, &ainfo);
			if (ret != PFM_SUCCESS)
				return ret;

			if (ainfo.type != PFM_ATTR_UMASK)
				continue;

			/*
			 * XXX: some events may require more than one umasks to encode
			 */
			sprintf(buf, "%s::%s:%s", pmu->name, einfo.name, ainfo.name);
			ret = pfmlib_validate_encoding(buf, PFM_PLM3|PFM_PLM0);
			if (ret != PFM_SUCCESS) {
				fprintf(fp, "cannot encode event %s : %s\n", buf, pfm_strerror(ret));
				retval = ret;
				continue;
			}
			um++;
		}
		if (um == 0) {
			sprintf(buf, "%s::%s", pmu->name, einfo.name);
			ret = pfmlib_validate_encoding(buf, PFM_PLM3|PFM_PLM0);
			if (ret != PFM_SUCCESS) {
				fprintf(fp, "cannot encode event %s : %s\n", buf, pfm_strerror(ret));
				retval = ret;
			}
		}
		n++;
	}
	free(buf);

	if (retval == PFM_SUCCESS)
		fprintf(fp, "OK (%d events)\n", n);

	return retval;
}

int
pfm_pmu_validate(pfm_pmu_t pmu_id, FILE *fp)
{
	pfmlib_pmu_t *pmu;
	int ret;

	if (fp == NULL)
		return PFM_ERR_INVAL;

	pmu = pmu2pmuidx(pmu_id);
	if (!pmu)
		return PFM_ERR_INVAL;


	fprintf(fp, "\tcheck struct: "); fflush(fp);
	if (!pmu->name) {
		fprintf(fp, "pmu id: %d :: no name\n", pmu->pmu);
		return PFM_ERR_INVAL;
	}

	if (!pmu->desc) {
		fprintf(fp, "pmu: %s :: no description\n", pmu->name);
		return PFM_ERR_INVAL;
	}

	if (pmu->pmu < PFM_PMU_NONE || pmu->pmu >= PFM_PMU_MAX) {
		fprintf(fp, "pmu: %s :: invalid PMU id\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (pfmlib_pmu_active(pmu) && !pmu->pme_count) {
		fprintf(fp, "pmu: %s :: no events\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (!pmu->pmu_detect) {
		fprintf(fp, "pmu: %s :: missing pmu_detect callback\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (!pmu->get_event_first) {
		fprintf(fp, "pmu: %s :: missing get_event_first callback\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (!pmu->get_event_next) {
		fprintf(fp, "pmu: %s :: missing get_event_next callback\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (!pmu->get_event_perf_type) {
		fprintf(fp, "pmu: %s :: missing get_event_perf_type callback\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (!pmu->get_event_info) {
		fprintf(fp, "pmu: %s :: missing get_event_info callback\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (!pmu->get_event_attr_info) {
		fprintf(fp, "pmu: %s :: missing get_event_attr_info callback\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (!pmu->get_event_encoding) {
		fprintf(fp, "pmu: %s :: missing get_event_encoding callback\n", pmu->name);
		return PFM_ERR_INVAL;
	}
	if (!pmu->max_encoding) {
		fprintf(fp, "pmu: %s :: max_encoding is zero\n", pmu->name);
		return PFM_ERR_INVAL;
	}

	fputs("OK\n", fp);

	if (pmu->validate_table) {
		fprintf(fp, "\tcheck table: "); fflush(stdout);
		ret = pmu->validate_table(pmu, fp);
		if (ret != PFM_SUCCESS)
			return ret;
		fputs("OK\n", fp);
	}
	return pfmlib_pmu_validate_encoding(pmu, fp);
}

int
pfm_get_event_info(int idx, pfm_event_info_t *info)
{
	pfmlib_pmu_t *pmu;
	int pidx, ret;

	if (!PFMLIB_INITIALIZED())
		return PFM_ERR_NOINIT;

	pmu = pfmlib_idx2pidx(idx, &pidx);
	if (!pmu)
		return PFM_ERR_INVAL;

	if (!info)
		return PFM_ERR_INVAL;

	if (info->size && info->size != sizeof(*info))
		return PFM_ERR_INVAL;

	/* default data type is uint64 */
	info->dtype = PFM_DTYPE_UINT64;

	ret = pmu->get_event_info(pmu, pidx, info);
	if (ret == PFM_SUCCESS) {
		info->pmu = pmu->pmu;
		info->idx = idx;
	}
	return ret;
}

int
pfm_get_event_attr_info(int idx, int attr_idx, pfm_event_attr_info_t *info)
{
	pfmlib_pmu_t *pmu;
	int pidx;

	if (!PFMLIB_INITIALIZED())
		return PFM_ERR_NOINIT;

	pmu = pfmlib_idx2pidx(idx, &pidx);
	if (!pmu)
		return PFM_ERR_INVAL;

	if (!info)
		return PFM_ERR_INVAL;

	if (info->size && info->size != sizeof(*info))
		return PFM_ERR_INVAL;

	if (!pfmlib_valid_attr(pmu, pidx, attr_idx))
		return PFM_ERR_ATTR;

	return pmu->get_event_attr_info(pmu, pidx, attr_idx, info);
}

int
pfm_get_pmu_info(pfm_pmu_t pmuid, pfm_pmu_info_t *info)
{
	pfmlib_pmu_t *pmu;

	if (!PFMLIB_INITIALIZED())
		return PFM_ERR_NOINIT;

	if (pmuid < PFM_PMU_NONE || pmuid >= PFM_PMU_MAX)
		return PFM_ERR_INVAL;

	if (!info)
		return PFM_ERR_INVAL;

	if (info->size && info->size != sizeof(*info))
		return PFM_ERR_INVAL;

	pmu = pfmlib_pmus_map[pmuid];
	if (!pmu)
		return PFM_ERR_NOTSUPP;

	info->name = pmu->name;
	info->desc = pmu->desc;
	info->pmu = pmuid;
	/*
	 * XXX: pme_count only valid when PMU is detected
	 */
	info->is_present = !!pfmlib_pmu_active(pmu);
	info->nevents = pmu->pme_count;
	return PFM_SUCCESS;
}

static int
pfmlib_compare_attr_id(const void *a, const void *b)
{
	const pfmlib_attr_t *t1 = a;
	const pfmlib_attr_t *t2 = b;

	if (t1->id < t2->id)
		return -1;
	return t1->id == t2->id ? 0 : 1;
}

void
pfmlib_sort_attr(pfmlib_event_desc_t *e)
{
	qsort(e->attrs, e->nattrs, sizeof(pfmlib_attr_t), pfmlib_compare_attr_id);
}
