/*
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 * Copyright (c) 2006 IBM Corp.
 * Contributed by Kevin Corry <kevcorry@us.ibm.com>
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
 * pfmlib_pentium4.c
 *
 * Support for libpfm for the Pentium4/Xeon/EM64T processor family (family=15).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "pfmlib_priv.h"
#include "pfmlib_pentium4_priv.h"
#include "pentium4_events.h"

#ifdef __x86_64__
static inline void cpuid(int op, unsigned int *eax,
				 unsigned int *ebx,
				 unsigned int *ecx,
				 unsigned int *edx)
{
	__asm__("cpuid"
			: "=a" (*eax),
			"=b" (*ebx),
			"=c" (*ecx),
			"=d" (*edx)
			: "0" (op));
}

static inline unsigned int cpuid_family(void)
{
	unsigned int eax, ebx, ecx, edx;
	cpuid(1, &eax, &ebx, &ecx, &edx);
	return eax;
}

static inline void cpuid_vendor(int op, unsigned int *ebx,
				 unsigned int *ecx,
				 unsigned int *edx)
{
	unsigned int eax;

	cpuid(op, &eax, ebx, ecx, edx);
}
#endif

#ifdef __i386__
static inline unsigned int cpuid_family(void)
{
	unsigned int eax, op = 1;

	__asm__("pushl %%ebx; cpuid; popl %%ebx"
			: "=a" (eax)
			: "0" (op)
			: "ecx", "edx");
	return eax;
}

static inline void cpuid_vendor(int op,
				 unsigned int *ebx,
				 unsigned int *ecx,
				 unsigned int *edx)
{	
	/*
	 * because ebx is used in Pic mode, we need to save/restore because
	 * cpuid clobbers it. I could not figure out a way to get ebx out in
	 * one cpuid instruction. To extract ebx, we need to  move it to another
	 * register (here eax)
	 */
	__asm__("pushl %%ebx;cpuid; movl %%ebx, %%eax;popl %%ebx"
			:"=a" (*ebx)
			: "a" (op)
			: "ecx", "edx");

	__asm__("pushl %%ebx;cpuid; popl %%ebx"
			:"=c" (*ecx), "=d" (*edx)
			: "a" (op));
}
#endif

/**
 * pentium4_get_event_code
 *
 * Return the event-select value for the specified event as
 * needed for the specified PMD counter.
 **/
static int pentium4_get_event_code(unsigned int event,
				   unsigned int pmd,
				   int *code)
{
	int i, j, escr, cccr;
	int rc = PFMLIB_ERR_INVAL;

	if (pmd >= PENTIUM4_NUM_PMDS && pmd != PFMLIB_CNT_FIRST) {
		goto out;
	}

	/* Check that the specified event is allowed for the specified PMD.
	 * Each event has a specific set of ESCRs it can use, which implies
	 * a specific set of CCCRs (and thus PMDs). A specified PMD of -1
	 * means assume any allowable PMD.
	 */
	if (pmd == PFMLIB_CNT_FIRST) {
		*code = pentium4_events[event].event_select;
		rc = PFMLIB_SUCCESS;
		goto out;
	}

	for (i = 0; i < MAX_ESCRS_PER_EVENT; i++) {
		escr = pentium4_events[event].allowed_escrs[i];
		if (escr < 0) {
			continue;
		}

		for (j = 0; j < MAX_CCCRS_PER_ESCR; j++) {
			cccr = pentium4_escrs[escr].allowed_cccrs[j];
			if (cccr < 0) {
				continue;
			}

			if (pmd == pentium4_cccrs[cccr].pmd) {
				*code = pentium4_events[event].event_select;
				rc = PFMLIB_SUCCESS;
				goto out;
			}
		}
	}

out:
	return rc;
}

/**
 * pentium4_get_event_name
 *
 * Return the name of the specified event.
 **/
static char *pentium4_get_event_name(unsigned int event)
{
	return pentium4_events[event].name;
}

/**
 * pentium4_get_event_mask_name
 *
 * Return the name of the specified event-mask.
 **/
static char *pentium4_get_event_mask_name(unsigned int event, unsigned int mask)
{
	if (mask >= EVENT_MASK_BITS || pentium4_events[event].event_masks[mask].name == NULL)
		return NULL;

	return pentium4_events[event].event_masks[mask].name;
}

/**
 * pentium4_get_event_counters
 *
 * Fill in the 'counters' bitmask with all possible PMDs that could be
 * used to count the specified event.
 **/
static void pentium4_get_event_counters(unsigned int event,
					pfmlib_regmask_t *counters)
{
	int i, j, escr, cccr;

	memset(counters, 0, sizeof(*counters));

	for (i = 0; i < MAX_ESCRS_PER_EVENT; i++) {
		escr = pentium4_events[event].allowed_escrs[i];
		if (escr < 0) {
			continue;
		}

		for (j = 0; j < MAX_CCCRS_PER_ESCR; j++) {
			cccr = pentium4_escrs[escr].allowed_cccrs[j];
			if (cccr < 0) {
				continue;
			}
			pfm_regmask_set(counters, pentium4_cccrs[cccr].pmd);
		}
	}
}

/**
 * pentium4_get_num_event_masks
 *
 * Count the number of available event-masks for the specified event. All
 * valid masks in pentium4_events[].event_masks are contiguous in the array
 * and have a non-NULL name.
 **/
static int pentium4_get_num_event_masks(unsigned int event,
					unsigned int *count)
{
	int i = 0;

	while (pentium4_events[event].event_masks[i].name) {
		i++;
	}

	*count = i;
	return PFMLIB_SUCCESS;
}

/**
 * pentium4_dispatch_events
 *
 * Examine each desired event specified in "input" and find an appropriate
 * ESCR/CCCR pair that can be used to count them.
 **/
static int pentium4_dispatch_events(pfmlib_input_param_t *input,
				    void *model_input,
				    pfmlib_output_param_t *output,
				    void *model_output)
{
	unsigned int assigned_pmcs[PENTIUM4_NUM_PMCS] = {0};
	unsigned int event, event_mask, mask;
	unsigned int plm;
	unsigned int i, j, k, m, n;
	int escr, escr_pmc;
	int cccr, cccr_pmc, cccr_pmd;
	int assigned;
	pentium4_escr_value_t escr_value;
	pentium4_cccr_value_t cccr_value;

	if (input->pfp_event_count > PENTIUM4_NUM_PMDS) {
		/* Can't specify more events than we have counters. */
		return PFMLIB_ERR_TOOMANY;
	}

	if (input->pfp_dfl_plm & (PFM_PLM1|PFM_PLM2)) {
		/* Can't specify privilege levels 1 or 2. */
		return PFMLIB_ERR_INVAL;
	}

	/* Examine each event specified in input->pfp_events. i counts
	 * through the input->pfp_events array, and j counts through the
	 * PMCs in output->pfp_pmcs as they are set up.
	 */
	for (i = 0, j = 0; i < input->pfp_event_count; i++) {

		if (input->pfp_events[i].plm & (PFM_PLM1|PFM_PLM2)) {
			/* Can't specify privilege levels 1 or 2. */
			return PFMLIB_ERR_INVAL;
		}

		event = input->pfp_events[i].event;
		assigned = 0;

		/* Use the event-specific privilege mask if set.
		 * Otherwise use the default privilege mask.
		 */
		plm = input->pfp_events[i].plm ?
		      input->pfp_events[i].plm : input->pfp_dfl_plm;

		/* Examine each ESCR that this event could be assigned to. */
		for (k = 0; k < MAX_ESCRS_PER_EVENT && !assigned; k++) {
			escr = pentium4_events[event].allowed_escrs[k];
			if (escr < 0) {
				continue;
			}

			/* Make sure this ESCR isn't already assigned
			 * and isn't on the "unavailable" list.
			 */
			escr_pmc = pentium4_escrs[escr].pmc;
			if (assigned_pmcs[escr_pmc] ||
			    pfm_regmask_isset(&input->pfp_unavail_pmcs, escr_pmc)) {
				continue;
			}

			/* Examine each CCCR that can be used with this ESCR. */
			for (m = 0; m < MAX_CCCRS_PER_ESCR && !assigned; m++) {
				cccr = pentium4_escrs[escr].allowed_cccrs[m];
				if (cccr < 0) {
					continue;
				}

				/* Make sure this CCCR isn't already assigned
				 * and isn't on the "unavailable" list.
				 */
				cccr_pmc = pentium4_cccrs[cccr].pmc;
				cccr_pmd = pentium4_cccrs[cccr].pmd;
				if (assigned_pmcs[cccr_pmc] ||
				    pfm_regmask_isset(&input->pfp_unavail_pmcs, cccr_pmc)) {
					continue;
				}

				/* Found an available ESCR/CCCR pair. */
				assigned = 1;
				assigned_pmcs[escr_pmc] = 1;
				assigned_pmcs[cccr_pmc] = 1;

				/* Calculate the event-mask value. Invalid masks
				 * specified by the caller are ignored.
				 */
				event_mask = 0;
				for (n = 0; n < input->pfp_events[i].num_masks; n++) {
					mask = input->pfp_events[i].unit_masks[n];
					if (mask < EVENT_MASK_BITS &&
					    pentium4_events[event].event_masks[mask].name) {
						event_mask |= (1 << pentium4_events[event].event_masks[mask].bit);
					}
				}

				/* Set up the ESCR and CCCR register values. */
				escr_value.val = 0;

				escr_value.bits.t1_usr       = 0; /* FIXME: Assumes non-HT */
				escr_value.bits.t1_os        = 0; /* FIXME: Assumes non-HT */
				escr_value.bits.t0_usr       = (plm & PFM_PLM3) ? 1 : 0;
				escr_value.bits.t0_os        = (plm & PFM_PLM0) ? 1 : 0;
				escr_value.bits.tag_enable   = 0; /* FIXME: What do we do with the "tag" entries? */
				escr_value.bits.tag_value    = 0; /* FIXME: What do we do with the "tag" entries? */
				escr_value.bits.event_mask   = event_mask;
				escr_value.bits.event_select = pentium4_events[event].event_select;
				escr_value.bits.reserved     = 0;

				cccr_value.val = 0;

				cccr_value.bits.reserved1     = 0;
				cccr_value.bits.enable        = 1;
				cccr_value.bits.escr_select   = pentium4_events[event].escr_select;
				cccr_value.bits.active_thread = 3; /* FIXME: This is set to count when either logical
								    *        CPU is active. Need a way to distinguish
								    *        between logical CPUs when HT is enabled. */
				cccr_value.bits.compare       = 0; /* FIXME: What do we do with "threshold" settings? */
				cccr_value.bits.complement    = 0; /* FIXME: What do we do with "threshold" settings? */
				cccr_value.bits.threshold     = 0; /* FIXME: What do we do with "threshold" settings? */
				cccr_value.bits.force_ovf     = 0; /* FIXME: Do we want to allow "forcing" overflow
								    *        interrupts on all counter increments? */
				cccr_value.bits.ovf_pmi_t0    = 1;
				cccr_value.bits.ovf_pmi_t1    = 0; /* FIXME: Assumes non-HT. */
				cccr_value.bits.reserved2     = 0;
				cccr_value.bits.cascade       = 0; /* FIXME: How do we handle "cascading" counters? */
				cccr_value.bits.overflow      = 0;

				/* Set up the PMCs in the
				 * output->pfp_pmcs array.
				 */
				output->pfp_pmcs[j].reg_num = escr_pmc;
				output->pfp_pmcs[j].reg_evt_idx = i;
				output->pfp_pmcs[j].reg_value = escr_value.val;
				output->pfp_pmcs[j].reg_pmd_num = cccr_pmd;
				j++;

				output->pfp_pmcs[j].reg_num = cccr_pmc;
				output->pfp_pmcs[j].reg_evt_idx = i;
				output->pfp_pmcs[j].reg_value = cccr_value.val;
				output->pfp_pmcs[j].reg_pmd_num = cccr_pmd;
				j++;

				output->pfp_pmc_count += 2;
			}
		}

		if (k == MAX_ESCRS_PER_EVENT) {
			/* Couldn't find an available ESCR and/or CCCR. */
			return PFMLIB_ERR_NOASSIGN;
		}
	}

	return PFMLIB_SUCCESS;
}

/**
 * pentium4_pmu_detect
 *
 * Determine whether the system we're running on is a Pentium4
 * (or other CPU that uses the same PMU).
 **/
static int pentium4_pmu_detect(void)
{
	unsigned int eax;
	int ret;
	char vendor_id[16];

	/* Check that the core library supports enough registers. */
	if (PFMLIB_MAX_PMCS < PENTIUM4_NUM_PMCS ||
	    PFMLIB_MAX_PMDS < PENTIUM4_NUM_PMDS) {
		return PFMLIB_ERR_NOTSUPP;
	}

	eax = cpuid_family();
	cpuid_vendor(0, (unsigned int *)&vendor_id[0],
			(unsigned int *)&vendor_id[8],
			(unsigned int *)&vendor_id[4]);
	/*
	 * this file only supports Intel P4 (32-bit, EM64T).
	 *
	 * accept family 15 Intel processors, all models
	 */
	ret = PFMLIB_ERR_NOTSUPP;
	if (((eax>>8) & 0xf) == 15 && !strcmp(vendor_id, "GenuineIntel")) {
		ret = PFMLIB_SUCCESS;
	}
	return ret;
}

/**
 * pentium4_get_impl_pmcs
 *
 * Set the appropriate bit in the impl_pmcs bitmask for each PMC that's
 * available on Pentium4.
 *
 * FIXME: How can we detect when HyperThreading is enabled?
 **/
static void pentium4_get_impl_pmcs(pfmlib_regmask_t *impl_pmcs)
{
	unsigned int i;

	memset(impl_pmcs, 0, sizeof(*impl_pmcs));

	for(i = 0; i < PENTIUM4_NUM_PMCS; i++) {
		pfm_regmask_set(impl_pmcs, i);
	}
}

/**
 * pentium4_get_impl_pmds
 *
 * Set the appropriate bit in the impl_pmcs bitmask for each PMD that's
 * available on Pentium4.
 *
 * FIXME: How can we detect when HyperThreading is enabled?
 **/
static void pentium4_get_impl_pmds(pfmlib_regmask_t *impl_pmds)
{
	unsigned int i;

	memset(impl_pmds, 0, sizeof(*impl_pmds));

	for(i = 0; i < PENTIUM4_NUM_PMDS; i++) {
		pfm_regmask_set(impl_pmds, i);
	}
}

/**
 * pentium4_get_impl_counters
 *
 * Set the appropriate bit in the impl_counters bitmask for each counter
 * that's available on Pentium4.
 *
 * For now, all PMDs are counters, so just call get_impl_pmds().
 **/
static void pentium4_get_impl_counters(pfmlib_regmask_t *impl_counters)
{
	pentium4_get_impl_pmds(impl_counters);
}

/**
 * pentium4_get_hw_counter_width
 *
 * Return the number of usable bits in the PMD counters.
 **/
static void pentium4_get_hw_counter_width(unsigned int *width)
{
	*width = PENTIUM4_COUNTER_WIDTH;
}

/**
 * pentium4_get_event_desc
 *
 * Return the description for the specified event (if it has one).
 *
 * FIXME: In this routine, we make a copy of the description string to
 *        return. But in get_event_name(), we just return the string
 *        directly. Why the difference?
 **/
static int pentium4_get_event_desc(unsigned int event, char **desc)
{
	if (pentium4_events[event].desc) {
		*desc = strdup(pentium4_events[event].desc);
	} else {
		*desc = NULL;
	}

	return PFMLIB_SUCCESS;
}

/**
 * pentium4_get_event_mask_desc
 *
 * Return the description for the specified event-mask (if it has one).
 **/
static int pentium4_get_event_mask_desc(unsigned int event,
					unsigned int mask, char **desc)
{
	if (mask >= EVENT_MASK_BITS || pentium4_events[event].event_masks[mask].desc == NULL)
		return PFMLIB_ERR_INVAL;

	*desc = strdup(pentium4_events[event].event_masks[mask].desc);

	return PFMLIB_SUCCESS;
}

/**
 * pentium4_support
 **/
pfm_pmu_support_t pentium4_support = {
	.pmu_name		= "Pentium4/Xeon/EM64T",
	.pmu_type		= PFMLIB_PENTIUM4_PMU,
	.pme_count		= PENTIUM4_EVENT_COUNT,
	.pmd_count		= PENTIUM4_NUM_PMDS,
	.pmc_count		= PENTIUM4_NUM_PMCS,
	.num_cnt		= PENTIUM4_NUM_PMDS,
	.cycle_event		= PENTIUM4_CPU_CLK_UNHALTED,
	.inst_retired_event	= PENTIUM4_INST_RETIRED,
	.get_event_code		= pentium4_get_event_code,
	.get_event_name		= pentium4_get_event_name,
	.get_event_mask_name	= pentium4_get_event_mask_name,
	.get_event_counters	= pentium4_get_event_counters,
	.get_num_event_masks	= pentium4_get_num_event_masks,
	.dispatch_events	= pentium4_dispatch_events,
	.pmu_detect		= pentium4_pmu_detect,
	.get_impl_pmcs		= pentium4_get_impl_pmcs,
	.get_impl_pmds		= pentium4_get_impl_pmds,
	.get_impl_counters	= pentium4_get_impl_counters,
	.get_hw_counter_width	= pentium4_get_hw_counter_width,
	.get_event_desc         = pentium4_get_event_desc,
	.get_event_mask_desc	= pentium4_get_event_mask_desc,
};

