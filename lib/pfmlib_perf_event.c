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
#include <string.h>
#include <stdlib.h>
#include <perfmon/pfmlib_perf_event.h>
#include "pfmlib_priv.h"

static int
get_perf_event_attr(const char *str, struct perf_event_attr *hw, int *idx)
{
	pfmlib_pmu_t *pmu;
	pfmlib_event_desc_t e;
	pfmlib_perf_attr_t perf_attrs;
	uint64_t *values = NULL;
	int count = 0;
	int ret;

	ret = pfmlib_parse_event(str, &e);
	if (ret != PFM_SUCCESS)
		return ret;

	pmu = e.pmu;

	memset(&perf_attrs, 0, sizeof(perf_attrs));

	/*
 	 * values[] dynamically allocated by call because we
 	 * pass NULL
 	 *
 	 * plm contains the priv level info from HW layer
 	 */
	ret = pfmlib_get_event_encoding(&e, &values, &count, &perf_attrs);
	if (ret != PFM_SUCCESS)
		return ret;

	/* no values */
	if (!count)
		return PFM_ERR_INVAL;

	/* don't know how to deal with this in PERF */
	if (count > 1)
		return PFM_ERR_NOTSUPP;

	hw->type = pmu->get_event_perf_type(pmu, e.event);
	if (hw->type == -1)
		return PFM_ERR_NOTSUPP;

	hw->config = values[0];

	/*
 	 * do not exclude anything by default
 	 * kernel may override depending on
 	 * paranoia level
 	 */
	hw->exclude_user = 0;
	hw->exclude_kernel = 0;
	hw->exclude_hv = 0;

	/*
	 * propagate to counter_attr struct
	 */
	if (perf_attrs.plm) {
		hw->exclude_user = !(perf_attrs.plm & PFM_PLM3);
		hw->exclude_kernel = !(perf_attrs.plm & PFM_PLM0);
		hw->exclude_hv = !(perf_attrs.plm & PFM_PLMH);
	}

	__pfm_vbprintf("PERF[type=%x val=0x%"PRIx64" e_u=%d e_k=%d e_hv=%d] %s\n",
			hw->type,
			hw->config,
			hw->exclude_user,
			hw->exclude_kernel,
			hw->exclude_hv,
			str);

	free(values);
	if (idx)
		*idx = pfmlib_pidx2idx(e.pmu, e.event);

	return PFM_SUCCESS;
}

int
pfm_get_perf_event_attr(const char *str, struct perf_event_attr *hw, int *idx)
{
	if (PFMLIB_INITIALIZED() == 0)
		return PFM_ERR_NOINIT;

	if (!(hw && str))
		return PFM_ERR_INVAL;

	memset(hw, 0, sizeof(hw));

	return get_perf_event_attr(str, hw, idx);
}
