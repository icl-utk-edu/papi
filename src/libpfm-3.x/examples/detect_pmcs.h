/*
 * detect_pmcs.h - detect unavailable PMD/PMC registers based on perfmon2 information
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
#ifndef __DETECT_PMCS_H__
#define __DETECT_PMCS_H__

#include <perfmon/pfmlib.h>

/*
 * if no context exists, pass -1 for fd
 * if do not care about PMCS, pass r_pmcs as NULL
 * if do not care about PMDs, pass r_pmds as NULL
 */
extern int detect_unavail_pmu_regs(int fd, pfmlib_regmask_t *r_pmcs, pfmlib_regmask_t *r_pmds);

static inline int detect_unavail_pmcs(int fd, pfmlib_regmask_t *r_pmcs)
{
	return detect_unavail_pmu_regs(fd, r_pmcs, NULL);
}

#endif /* __DETECT_PMCS_H__ */
