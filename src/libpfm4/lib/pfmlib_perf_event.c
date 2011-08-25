/*
 * pfmlib_perf_events.c: encode events for perf_event API
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@google.com>
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
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <perfmon/pfmlib_perf_event.h>

#include "pfmlib_priv.h"
#include "pfmlib_perf_event_priv.h"

#define PERF_PROC_FILE "/proc/sys/kernel/perf_event_paranoid"

/*
 * contains ONLY attributes related to PMU features
 */
static const pfmlib_attr_desc_t perf_event_mods[]={
	PFM_ATTR_B("u", "monitor at user level"),	/* monitor user level */
	PFM_ATTR_B("k", "monitor at kernel level"),	/* monitor kernel level */
	PFM_ATTR_B("h", "monitor at hypervisor level"),	/* monitor hypervisor level */
	PFM_ATTR_NULL /* end-marker to avoid exporting number of entries */
};

/*
 * contains all attributes controlled by perf_events. That includes PMU attributes
 * and pure software attributes such as sampling periods
 */
static const pfmlib_attr_desc_t perf_event_ext_mods[]={
	PFM_ATTR_B("u", "monitor at user level"),	/* monitor user level */
	PFM_ATTR_B("k", "monitor at kernel level"),	/* monitor kernel level */
	PFM_ATTR_B("h", "monitor at hypervisor level"),	/* monitor hypervisor level */
	PFM_ATTR_I("period", "sampling period"),     	/* sampling period */
	PFM_ATTR_I("freq", "sampling frequency (Hz)"),     /* sampling frequency */
	PFM_ATTR_I("precise", "precise ip"),     		/* anti-skid mechanism */
	PFM_ATTR_NULL /* end-marker to avoid exporting number of entries */
};

static int
pfmlib_perf_event_encode(void *this, const char *str, int dfl_plm, void *data)
{
	pfm_perf_encode_arg_t arg;
	pfm_perf_encode_arg_t *uarg = data;
	pfmlib_os_t *os = this;
	struct perf_event_attr *attr;
	pfmlib_pmu_t *pmu;
	pfmlib_event_desc_t e;
	pfm_event_attr_info_t *a;
	size_t sz = sizeof(arg);
	uint64_t ival;
	int has_plm = 0;
	int i, plm = 0, ret;

	sz = pfmlib_check_struct(uarg, uarg->size, PFM_PERF_ENCODE_ABI0, sz);
	if (!sz)
		return PFM_ERR_INVAL;

	/* copy input */
	memcpy(&arg, uarg, sz);

	attr = arg.attr;

	memset(&e, 0, sizeof(e));

	e.osid = os->id;
	e.os_data = attr;
	e.dfl_plm = dfl_plm;

	/* after this call, need to call pfmlib_release_event() */
	ret = pfmlib_parse_event(str, &e);
	if (ret != PFM_SUCCESS)
		return ret;

	pmu = e.pmu;

	ret = PFM_ERR_NOTSUPP;
	if (!pmu->get_event_encoding[e.osid]) {
		DPRINT("PMU %s does not support PFM_OS_NONE\n", pmu->name);
		goto done;
	}

	ret = pmu->get_event_encoding[e.osid](pmu, &e);
        if (ret != PFM_SUCCESS)
		goto done;

	/*
	 * process perf_event attributes
	 */
	for (i = 0; i < e.nattrs; i++) {

		a = attr(&e, i);

		if (a->ctrl != PFM_ATTR_CTRL_PERF_EVENT)
			continue;

		ival = e.attrs[i].ival;

		switch(a->idx) {
		case PERF_ATTR_U:
			if (ival)
				plm |= PFM_PLM3;
			has_plm = 1;
			break;
		case PERF_ATTR_K:
			if (ival)
				plm |= PFM_PLM0;
			has_plm = 1;
			break;
		case PERF_ATTR_H:
			if (ival)
				plm |= PFM_PLMH;
			has_plm = 1;
			break;
		case PERF_ATTR_PE:
			if (!ival || attr->freq)
				return PFM_ERR_ATTR_VAL;
			attr->sample_period = ival;
			break;
		case PERF_ATTR_FR:
			if (!ival)
				return PFM_ERR_ATTR_VAL;
			attr->sample_freq = ival;
			attr->freq = 1;
			break;
		case PERF_ATTR_PR:
			if (ival < 0 || ival > 3)
				return PFM_ERR_ATTR_VAL;
			attr->precise_ip = ival;
			break;
		}
	}

	/*
	 * if no priv level mask was provided
	 * with the event, then use dfl_plm
	 */
	if (!has_plm)
		plm = dfl_plm;

	/*
	 * perf_event plm work by exclusion, so use logical or
	 * goal here is to set to zero any exclude_* not supported
	 * by underlying PMU
	 */
	plm |= (~pmu->supported_plm) & PFM_PLM_ALL;

	attr->exclude_user   = !(plm & PFM_PLM3);
	attr->exclude_kernel = !(plm & PFM_PLM0);
	attr->exclude_hv     = !(plm & PFM_PLMH);

	__pfm_vbprintf("PERF[type=%x config=0x%"PRIx64" config1=0x%"PRIx64
                       " e_u=%d e_k=%d e_hv=%d period=%"PRIu64" freq=%d"
                       " precise=%d] %s\n",
			attr->type,
			attr->config,
			attr->config1,
			attr->exclude_user,
			attr->exclude_kernel,
			attr->exclude_hv,
			attr->sample_period,
			attr->freq,
			attr->precise_ip,
			str);

	/*
	 * propagate event index if necessary
	 */
	arg.idx = pfmlib_pidx2idx(e.pmu, e.event);

	/*
	 * fstr not requested, stop here
	 */
	ret = PFM_SUCCESS;
	if (!arg.fstr) {
		memcpy(uarg, &arg, sz);
		goto done;
	}

	for (i=0; i < e.npattrs; i++) {
		int idx;

		if (e.pattrs[i].ctrl != PFM_ATTR_CTRL_PERF_EVENT)
			continue;

		idx = e.pattrs[i].idx;
		switch (idx) {
		case PERF_ATTR_K:
			evt_strcat(e.fstr, ":%s=%lu", perf_event_ext_mods[idx].name, !!(plm & PFM_PLM0));
			break;
		case PERF_ATTR_U:
			evt_strcat(e.fstr, ":%s=%lu", perf_event_ext_mods[idx].name, !!(plm & PFM_PLM3));
			break;
		case PERF_ATTR_H:
			evt_strcat(e.fstr, ":%s=%lu", perf_event_ext_mods[idx].name, !!(plm & PFM_PLMH));
			break;
		case PERF_ATTR_PR:
			evt_strcat(e.fstr, ":%s=%d", perf_event_ext_mods[idx].name, attr->precise_ip);
			break;
		case PERF_ATTR_PE:
		case PERF_ATTR_FR:
			if (attr->freq && attr->sample_period)
				evt_strcat(e.fstr, ":%s=%"PRIu64, perf_event_ext_mods[idx].name, attr->sample_period);
			else if (attr->sample_period)
				evt_strcat(e.fstr, ":%s=%"PRIu64, perf_event_ext_mods[idx].name, attr->sample_period);
		}
	}

	ret = pfmlib_build_fstr(&e, arg.fstr);
	if (ret == PFM_SUCCESS)
		memcpy(uarg, &arg, sz);

done:
	pfmlib_release_event(&e);
	return ret;
}
/*
 * get OS-specific event attributes
 */
static int
perf_get_os_nattrs(void *this, pfmlib_event_desc_t *e)
{
	pfmlib_os_t *os = this;
	int i = 0;

	for (; os->atdesc[i].name; i++);

	return i;
}

static int
perf_get_os_attr_info(void *this, pfmlib_event_desc_t *e)
{
	pfmlib_os_t *os = this;
	pfm_event_attr_info_t *info;
	int i, j = e->npattrs;

	for (i = 0; os->atdesc[i].name; i++, j++) {
		info = e->pattrs+j;

		info->name = os->atdesc[i].name;
		info->desc = os->atdesc[i].desc;
		info->equiv= NULL;
		info->code = i;
		info->idx  = i; /* namespace-specific index */
		info->type = os->atdesc[i].type;
		info->is_dfl = 0;
		info->ctrl = PFM_ATTR_CTRL_PERF_EVENT;
	}
	e->npattrs += i;

	return PFM_SUCCESS;
}

/*
 * old interface, maintained for backward compatibility with earlier versions of the library
 */
int
pfm_get_perf_event_encoding(const char *str, int dfl_plm, struct perf_event_attr *attr, char **fstr, int *idx)
{
	pfm_perf_encode_arg_t arg;
	size_t sz;
	int ret;

	if (PFMLIB_INITIALIZED() == 0)
		return PFM_ERR_NOINIT;

	/* idx and fstr can be NULL */
	if (!(attr && str))
		return PFM_ERR_INVAL;

	if (dfl_plm & ~(PFM_PLM_ALL))
		return PFM_ERR_INVAL;

	memset(&arg, 0, sizeof(arg));

	/*
	 * zero out everything that is passed
	 */
	if (attr->size == 0)
		sz = sizeof(*attr);
	else
		sz = attr->size;

	memset(attr, 0, sz);

	arg.attr = attr;
	arg.fstr = fstr;

	ret = pfm_get_os_event_encoding(str, dfl_plm, PFM_OS_PERF_EVENT_EXT, &arg);
	if (ret != PFM_SUCCESS)
		return ret;

	if (idx)
		*idx = arg.idx;

	return PFM_SUCCESS;
}

static int
pfm_perf_event_os_detect(void *this)
{
	int ret = access(PERF_PROC_FILE, F_OK);
	return ret ? PFM_ERR_NOTSUPP : PFM_SUCCESS;
}

pfmlib_os_t pfmlib_os_perf={
	.name = "perf_event",
	.id = PFM_OS_PERF_EVENT,
	.atdesc = perf_event_mods,
	.detect = pfm_perf_event_os_detect,
	.get_os_attr_info = perf_get_os_attr_info,
	.get_os_nattrs = perf_get_os_nattrs,
	.encode = pfmlib_perf_event_encode,
};

pfmlib_os_t pfmlib_os_perf_ext={
	.name = "perf_event extended",
	.id = PFM_OS_PERF_EVENT_EXT,
	.atdesc = perf_event_ext_mods,
	.detect = pfm_perf_event_os_detect,
	.get_os_attr_info = perf_get_os_attr_info,
	.get_os_nattrs = perf_get_os_nattrs,
	.encode = pfmlib_perf_event_encode,
};


