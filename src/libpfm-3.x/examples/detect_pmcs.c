/*
 * detect_pmcs.c - detect unavailable PMC registers based on perfmon2 information
 *
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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
 * applications on Linux.
 */
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

/*
 * The goal of this function is to help pfm_dispatch_events()
 * in situations where not all PMC registers are available.
 *
 * It builds a bitmask of unavailable PMC registers using either
 * an existing perfmon or none. In the latter case, it will create
 * a temporary context to retrieve the information. When a context
 * is passed, it can either be attached or detached.
 *
 * Note that there is no guarantee that the registers marked
 * as available will actually be available by the time the perfmon
 * context is loaded. 
 */
int
detect_unavail_pmcs(int fd, pfmlib_regmask_t *r_pmcs)
{
	pfarg_ctx_t ctx;
	pfarg_setinfo_t	setf;
	int ret, i, j, myfd;

	memset(r_pmcs, 0, sizeof(*r_pmcs));

	memset(&ctx, 0, sizeof(ctx));
	memset(&setf, 0, sizeof(setf));
#if  PFMLIB_REG_MAX < PFM_MAX_PMCS
#error "PFMLIB_REG_MAX too small for PFM_MAX_PMCS"
#endif

	/*
	 * if no context descriptor is passed, then create
	 * a temporary context
	 */
	if (fd == -1) {
		ret = pfm_create_context(&ctx, NULL, 0);
		if (ret)
			return 0;
		myfd = ctx.ctx_fd;
	} else {
		myfd = fd;
	}
	/*
	 * retrieve available register bitmasks from set0
	 * which is guaranteed to exist for every context
	 */
	ret = pfm_getinfo_evtsets(myfd, &setf, 1);
	if (ret == 0) {
		for(i=0; i < PFM_PMC_BV; i++) {
			for(j=0; j < 64; j++) {
				if ((setf.set_avail_pmcs[i] & (1ULL << j)) == 0)
					pfm_regmask_set(r_pmcs, (i<<6)+j);
			}
		}
	}
	if (fd == -1) close(myfd);
	return ret;
}
