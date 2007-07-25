/*
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * pfmlib_powerpc.c
 *
 * Support for libpfm for the POWERPC processor family.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <perfmon/pfmlib_powerpc.h>

/* private headers */
#include "pfmlib_priv.h"
#include "pfmlib_powerpc_priv.h"
#include "powerpc_events.h"

/* Add structures here to define the PMD and PMC mappings. */

/**
 * powerpc_get_event_code
 *
 * Return the event-select value for the specified event as
 * needed for the specified PMD counter.
 **/
static int powerpc_get_event_code(unsigned int event,
				   unsigned int pmd,
				   int *code)
{
	return 0;
}

/**
 * powerpc_get_event_name
 *
 * Return the name of the specified event.
 **/
static char *powerpc_get_event_name(unsigned int event)
{
	return "";
}

/**
 * powerpc_get_event_mask_name
 *
 * Return the name of the specified event-mask.
 **/
static char *powerpc_get_event_mask_name(unsigned int event, unsigned int mask)
{
	return "";
}

/**
 * powerpc_get_event_counters
 *
 * Fill in the 'counters' bitmask with all possible PMDs that could be
 * used to count the specified event.
 **/
static void powerpc_get_event_counters(unsigned int event,
					pfmlib_regmask_t *counters)
{
	memset(counters, 0, sizeof(*counters));
}

/**
 * powerpc_get_num_event_masks
 *
 * Count the number of available event-masks for the specified event.
 **/
static unsigned int powerpc_get_num_event_masks(unsigned int event)
{
	return 0;
}

/**
 * powerpc_dispatch_events
 *
 * Examine each desired event specified in "input" and find an appropriate
 * set of PMCs and PMDs to count them.
 **/
static int powerpc_dispatch_events(pfmlib_input_param_t *input,
				   void *model_input,
				   pfmlib_output_param_t *output,
				   void *model_output)
{
	return 0;
}

/**
 * powerpc_pmu_detect
 *
 * Determine whether the system we're running on is a PowerPC.
 * (or other CPU that uses the same PMU).
 **/
static int powerpc_pmu_detect(void)
{
	return 0;
}

/**
 * powerpc_get_impl_pmcs
 *
 * Set the appropriate bit in the impl_pmcs bitmask for each PMC that's
 * available on PowerPC.
 **/
static void powerpc_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	return;
}

/**
 * powerpc_get_impl_pmds
 *
 * Set the appropriate bit in the impl_pmcs bitmask for each PMD that's
 * available on PowerPC.
 **/
static void powerpc_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	return;
}

/**
 * powerpc_get_impl_counters
 *
 * Set the appropriate bit in the impl_counters bitmask for each counter
 * that's available on PowerPC.
 *
 * For now, all PMDs are counters, so just call get_impl_pmds().
 **/
static void powerpc_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	powerpc_get_impl_pmds(impl_counters);
}

/**
 * powerpc_get_hw_counter_width
 *
 * Return the number of usable bits in the PMD counters.
 **/
static void powerpc_get_hw_counter_width(unsigned int *width)
{
	*width = 0;
}

/**
 * powerpc_get_event_desc
 *
 * Return the description for the specified event (if it has one).
 **/
static int powerpc_get_event_desc(unsigned int event, char **desc)
{
	*desc = NULL;
	return 0;
}

/**
 * powerpc_get_event_mask_desc
 *
 * Return the description for the specified event-mask (if it has one).
 **/
static int powerpc_get_event_mask_desc(unsigned int event,
					unsigned int mask, char **desc)
{
	*desc = strdup("");
	return 0;
}

static int powerpc_get_event_mask_code(unsigned int event,
				 	unsigned int mask, unsigned int *code)
{
	*code = 0;
	return 0;
}

static int
powerpc_get_cycle_event(pfmlib_event_t *e)
{
	e->event = 0;
	e->num_masks = 0;
	e->unit_masks[0] = 0;
	return 0;

}

static int
powerpc_get_inst_retired(pfmlib_event_t *e)
{
	e->event = 0;
	e->num_masks = 0;
	e->unit_masks[0] = 0;
	return 0;
}

/**
 * powerpc_support
 **/
pfm_pmu_support_t generic_powerpc_support = {
	.pmu_name		= "PowerPC",
	.pmu_type		= PFMLIB_POWERPC_PMU,
	.pme_count		= 1,
	.pmd_count		= 1,
	.pmc_count		= 1,
	.num_cnt		= 1,
	.get_event_code		= powerpc_get_event_code,
	.get_event_name		= powerpc_get_event_name,
	.get_event_mask_name	= powerpc_get_event_mask_name,
	.get_event_counters	= powerpc_get_event_counters,
	.get_num_event_masks	= powerpc_get_num_event_masks,
	.dispatch_events	= powerpc_dispatch_events,
	.pmu_detect		= powerpc_pmu_detect,
	.get_impl_pmcs		= powerpc_get_impl_pmcs,
	.get_impl_pmds		= powerpc_get_impl_pmds,
	.get_impl_counters	= powerpc_get_impl_counters,
	.get_hw_counter_width	= powerpc_get_hw_counter_width,
	.get_event_desc         = powerpc_get_event_desc,
	.get_event_mask_desc	= powerpc_get_event_mask_desc,
	.get_event_mask_code	= powerpc_get_event_mask_code,
	.get_cycle_event	= powerpc_get_cycle_event,
	.get_inst_retired_event = powerpc_get_inst_retired,
};

