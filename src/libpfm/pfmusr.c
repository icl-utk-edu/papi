/*
 * pfmusr.c
 *
 * Copyright (C) 2001 Hewlett-Packard Co
 * Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>
 */

#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>


#include "pfmlib.h"
#include "pfm_private.h"


#define fatal_error printf

typedef perfmon_req_t __pfm_event_desc_t;


pfm_event_desc_t 
pfm_desc_alloc(void)
{
	return (pfm_event_desc_t)malloc(sizeof(__pfm_event_desc_t)*PMU_MAX_PMCS);
}

void
pfm_desc_free(pfm_event_desc_t desc)
{
	free((__pfm_event_desc_t *)desc);
}

static int
gen_events(char **arg, pfm_event_config_t *evt)
{
	int ev;
	int cnt=0;

	if (arg == NULL) return -1;

	while (*arg) {

		if (cnt == PMU_MAX_COUNTERS) goto too_many;
		/* must match vcode only */
		if ((ev = pfm_findevent(*arg,0)) == -1) goto error;

		evt->pec_evt[cnt++] = ev;

		arg++;
	}
	evt->pec_count = cnt;
	return 0;
error:
	return -1;
too_many:
	return -1;
}

static int
gen_thresholds(int *tlist, pfm_event_config_t *evt)
{
	int cnt=0;

	/*
	 * the default value for the threshold is 0: this means at least once 
	 * per cycle.
	 */
	if (tlist == NULL) {
		int i;
		for (i=0; i < evt->pec_count; i++) evt->pec_thres[i] = 0;
		return 0;
	}

	while (*tlist >= 0) {

		if (cnt == PMU_MAX_COUNTERS || cnt == evt->pec_count) goto too_many;
		/*
		 * threshold = multi-occurence -1
		 * this is because by setting threshold to n, one counts only
		 * when n+1 or more events occurs per cycle.
	 	 */
		if (*tlist > pfm_event_threshold(evt->pec_evt[cnt])-1) goto too_big;

		evt->pec_thres[cnt++] = *tlist;

		tlist++;
	}
	return 0;
too_big:
	return -1;
too_many:
	return cnt; /* ignore extra values */
}


int
pfm_install_counters(pfm_event_desc_t desc, char **ev, int *thres, int plm)
{
	pfm_event_config_t evt;
	perfmon_req_t pd[PMU_MAX_COUNTERS];
	perfmon_req_t ctx[1];
	__pfm_event_desc_t *pc = (__pfm_event_desc_t *)desc;
	int cnt, i;
	int pid = getpid();

	if (desc == PFM_DESC_NULL) return -1;

	memset(&evt, 0 , sizeof(evt));
	memset(ctx, 0, sizeof(ctx));
	memset(pd, 0, sizeof(pd));

	if (gen_events(ev, &evt) == -1) return -1;

	/* treat user args as read-only */
	if (gen_thresholds(thres, &evt) == -1) return -1;

	ctx[0].pfr_ctx.notify_pid = pid;
	ctx[0].pfr_ctx.notify_sig = SIGPROF;
	ctx[0].pfr_ctx.flags      = PFM_FL_INHERIT_NONE;

	if (perfmonctl(pid, PFM_CREATE_CONTEXT, 0, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support !\n");
			return -1;
		}
		fatal_error("Can't create PFM context %d\n", errno);
	}
	/* will reset all PMU registers and unfreeze and set up=pp=0 */
	if (perfmonctl(pid, PFM_ENABLE, 0, NULL, 0) == -1) {
		fatal_error( "child: perfmonctl error PFM_ENABLE errno %d\n",errno);
		return -1;
	}

	evt.pec_plm = plm; /* XXX: must be per counter */
	cnt = PMU_MAX_PMCS;
	if (pfm_dispatch_events(&evt, pc, &cnt) == -1) return -1;

	if (perfmonctl(pid, PFM_WRITE_PMCS, 0, pc, cnt) == -1) {
		fatal_error("child: perfmonctl error WRITE_PMCS errno %d\n",errno);
		return -1;
	}
	for(i=0; i < evt.pec_count; i++) {
		pd[i].pfr_reg.reg_num = pc[i].pfr_reg.reg_num;
	}

	if (perfmonctl(pid, PFM_WRITE_PMDS, 0, pd, evt.pec_count) == -1) {
		fatal_error( "child: perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
		return -1;
	}
	/* at this point monitoring will just need a pfm_start() to commence */

	return 0;
}

int
pfm_read_counters(pfm_event_desc_t desc, int count, unsigned long *vals)
{
	perfmon_req_t pd[PMU_MAX_COUNTERS];
	__pfm_event_desc_t *pc = (__pfm_event_desc_t *)desc;
	int i;

	/* simplistic sanity checking */
	if (pc == NULL || vals == NULL || count > PMU_MAX_COUNTERS) return -1;

	/* prepare request */

	memset(pd, 0, sizeof(pd));

	for(i=0; i < count; i++) {
		pd[i].pfr_reg.reg_num = pc[i].pfr_reg.reg_num;
	}

	if (perfmonctl(getpid(), PFM_READ_PMDS, 0, pd, count) == -1) {
		fatal_error( "child: perfmonctl error READ_PMDS errno %d\n",errno);
		return -1;
	}

	for(i=0; i < count; i++)
		vals[i] = pd[pc[i].pfr_reg.reg_num-PMU_FIRST_COUNTER].pfr_reg.reg_value;

	return 0;
}

