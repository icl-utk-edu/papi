/*
 * detect_pmu_regs.c - detect unavailable PMD/PMC registers based on perfmon2 information
 *
 * Copyright (c) 2006-2007 Hewlett-Packard Development Company, L.P.
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
 * in situations where not all PMC/PMD registers are available.
 *
 * It builds bitmasks of *unavailable* PMC/PMD registers.
 * It can use an existing perfmon context file descriptor or if
 * non is passed, it will create a temporary context to retrieve
 * the information.
 *
 * Note that there is no guarantee that the registers marked
 * as available will actually be available by the time the perfmon
 * context is loaded. 
 *
 * arguments:
 * 	fd : a perfmon context file descriptor, or -1
 * 	r_pmcs: a bitmask for PMC availability, NULL if not needed
 * 	r_pmcs: a bitmask for PMD availability, NULL if not needed
 *
 * return:
 * 	-1: invalid file descriptor passed or cannot retrieve information
 * 	 0: success
 */
int
detect_unavail_pmu_regs(int fd, pfmlib_regmask_t *r_pmcs, pfmlib_regmask_t *r_pmds)
{
	pfarg_ctx_t ctx;
	pfarg_setinfo_t	setf;
	int ret, i, j, myfd, max;

	memset(&ctx, 0, sizeof(ctx));
	memset(&setf, 0, sizeof(setf));
	if (r_pmcs)
		memset(r_pmcs, 0, sizeof(*r_pmcs));
	if (r_pmds)
		memset(r_pmds, 0, sizeof(*r_pmds));
	/*
	 * if no context descriptor is passed, then create
	 * a temporary context
	 */
	if (fd == -1) {
		myfd = pfm_create_context(&ctx, NULL, NULL, 0);
		if (myfd == -1)
			return -1;
	} else {
		myfd = fd;
	}
	/*
	 * retrieve available register bitmasks from set0
	 * which is guaranteed to exist for every context
	 *
	 * if myfd is bogus (passed by user) then we return
	 * an error.
	 */
	ret = pfm_getinfo_evtsets(myfd, &setf, 1);
	if (ret == 0) {
		if (r_pmcs) {
			max = PFMLIB_REG_BV < PFM_PMC_BV ? PFMLIB_REG_BV : PFM_PMC_BV;
			for(i=0; i < max; i++) {
				for(j=0; j < 64; j++) {
					if ((setf.set_avail_pmcs[i] & (1ULL << j)) == 0)
						pfm_regmask_set(r_pmcs, (i<<6)+j);
				}
			}
		}
		if (r_pmds) {
			max = PFMLIB_REG_BV < PFM_PMD_BV ? PFMLIB_REG_BV : PFM_PMD_BV;
			for(i=0; i < max; i++) {
				for(j=0; j < 64; j++) {
					if ((setf.set_avail_pmds[i] & (1ULL << j)) == 0)
						pfm_regmask_set(r_pmds, (i<<6)+j);
				}
			}
		}
	}
	if (fd == -1)
		close(myfd);
	return ret;
}
