/*
 * detect_pmu_regs.c - detect unavailable PMD/PMC registers based on perfmon3 information
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

int
get_sif(int flags, pfarg_sinfo_t *sif)
{
	int fd;

	/* initialize as all available */
	if (sif) {
		memset(sif->sif_avail_pmcs, 0xff, sizeof(sif->sif_avail_pmcs));
		memset(sif->sif_avail_pmds, 0xff, sizeof(sif->sif_avail_pmds));
	}
	fd = pfm_create(flags, sif);
	if (fd > -1)
		close(fd);
	return fd > -1 ? 0 : -1;
}
/*
 * The goal of this function is to help pfm_dispatch_events()
 * in situations where not all PMC/PMD registers are available.
 *
 * It builds bitmasks of *unavailable* PMC/PMD registers from the
 * information returned by pfm_create_session().
 *
 * arguments:
 * 	sif: pfarg_sinfo_t pointer
 * 	r_pmcs: a bitmask for PMC availability, NULL if not needed
 * 	r_pmcs: a bitmask for PMD availability, NULL if not needed
 */
void
detect_unavail_pmu_regs(pfarg_sinfo_t *sif, pfmlib_regmask_t *r_pmcs, pfmlib_regmask_t *r_pmds)
{
	int i, j, max;

	if (r_pmcs) {
		memset(r_pmcs, 0, sizeof(*r_pmcs));
		max = PFMLIB_REG_BV < PFM_PMC_BV ? PFMLIB_REG_BV : PFM_PMC_BV;
		for(i=0; i < max; i++) {
			for(j=0; j < 64; j++) {
				if ((sif->sif_avail_pmcs[i] & (1ULL << j)) == 0)
					pfm_regmask_set(r_pmcs, (i<<6)+j);
			}
		}
	}
	if (r_pmds) {
		memset(r_pmds, 0, sizeof(*r_pmds));
		max = PFMLIB_REG_BV < PFM_PMD_BV ? PFMLIB_REG_BV : PFM_PMD_BV;
		for(i=0; i < max; i++) {
			for(j=0; j < 64; j++) {
				if ((sif->sif_avail_pmds[i] & (1ULL << j)) == 0)
					pfm_regmask_set(r_pmds, (i<<6)+j);
			}
		}
	}
}
