/*
 *
 * Copyright (C) 2001-2002 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file is part of pfmon, a sample tool to measure performance 
 * of applications on Linux/ia64.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

#include <perfmon/pfmlib.h>

#define ITA_EVENT_NAME "ALAT_INST_FAILED_CHKA_LDC_ALL"
#define MCK_EVENT_NAME "INST_FAILED_CHKA_LDC_ALAT_ALL"

#define RESET_VAL  (~0UL)

#ifdef __GNUC__
#define test_chka(res, adr) \
{ \
	__asm__ __volatile__(  \
		"{.mii ld8.a r30=[%1]}\n" \
		";;\n" \
		"{.mii st8 [%1]=r0}\n" \
		";;\n" \
		"chk.a.clr r30, 1f\n" \
		";;\n" \
		"mov %0=1\n" \
		";;\n" \
		"1:\n" \
		"mov %0=2;;\n" \
		: "=r"(res): "r"(adr): "r30", "memory"); \
}
#else
/*
 * don't quite know how to do this without the GNU inline assembly support!
 * So we force a test failure
 */
#define test_chka(res)	res = 0
#endif

int
specloop(int loop)
{
	unsigned long res;
	unsigned long tab[1];

	while ( loop-- ) {
		res = 7;
		test_chka(res,tab);

		if (res != 2) return -1;
	}
	return 0;
}

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static unsigned long handler_called, good_value;
static int allocated_pmd;

static void
fatal_error(char *fmt, ...) 
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}


void
cnt_handler(int n, struct pfm_siginfo *info, struct sigcontext *sc)
{
	pfarg_reg_t pd[1];

#if 0
	printf("Overflow notification: pid=%d @0x%lx bv=0x%lx\n", 
		info->sy_pid, 
		sc->sc_ip, 
		info->sy_pfm_ovfl[0]);
#endif

	handler_called++;

	if (info->sy_pfm_ovfl[0] == (1UL << allocated_pmd)) {

		pd[0].reg_num = allocated_pmd;

		if (perfmonctl(info->sy_pid, PFM_READ_PMDS, pd, 1) == -1) {
			fatal_error("overflow cannot read pmd: %s\n", strerror(errno));
		}
		if (pd[0].reg_value  != 0) {
			printf("invalid PMD[%d]=0x%lx value\n", allocated_pmd, pd[0].reg_value);
		} else {
			good_value++;
		}
		if (perfmonctl(info->sy_pid, PFM_RESTART, 0, 0) == -1) {
			fatal_error("overflow cannot restart: %s\n", strerror(errno));
		}

	}
}

int
main(int argc, char **argv)
{
	int ret;
	int type = 0;
	unsigned long loop, iloop;
	pid_t pid;
	pfmlib_param_t evt;
	pfarg_reg_t pd[1];
	pfarg_context_t ctx[1];
	struct sigaction act;
	pfmlib_options_t pfmlib_options;
	char *event = NULL;

#ifndef __GNUC__
	printf("This test program does not work if not compiled with GNU C.\n");
	exit(1);
#endif

	/*
	 * check for passive test
	 */
	if (argc > 2 && *argv[2] == 'p') {
		loop = argc > 1 ? atoi(argv[1]) : 1;
		return specloop(loop);
	}

	/*
	 * Initialize pfm library (required before we can use it)
	 */
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		printf("Can't initialize library\n");
		exit(1);
	}

	/*
	 * Let's make sure we run this on the right CPU
	 */
	pfm_get_pmu_type(&type);
	switch (type) {
		case PFMLIB_ITANIUM_PMU:
			event = ITA_EVENT_NAME;
			break;
		case PFMLIB_ITANIUM2_PMU:
			event = MCK_EVENT_NAME;
			break;
		default: {
			    char *model; 
			    pfm_get_pmu_name(&model);
			    fatal_error("unsupported PMU: %s\n", model);
			 }
	}

	memset(&pfmlib_options, 0, sizeof(pfmlib_options));

	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));
	memset(&evt,0, sizeof(evt));
	memset(&act,0,sizeof(act));

	pfmlib_options.pfm_debug = 1; /* set to 1 for debug */

	pfm_set_options(&pfmlib_options);


	/*
	 * install signal handler
	 */
	memset(&act,0,sizeof(act));
	act.sa_handler = (sig_t)cnt_handler;
	sigaction (SIGPROF, &act, 0);

	pid = getpid();

	/*
	 * prepare parameters to library. we don't use any Itanium
	 * specific features here. so the pfp_model is NULL.
	 */
	memset(&evt,0, sizeof(evt));

	if (pfm_find_event(event, &evt.pfp_events[0].event) != PFMLIB_SUCCESS) {
		fatal_error("Cannot find %s event\n", event);
	}
	evt.pfp_dfl_plm   = PFM_PLM0|PFM_PLM3; /* PLM3 is needed if less than C0 Itanium */
	evt.pfp_event_count = 1;

	/*
	 * let the library figure out the values for the PMCS
	 */
	if ((ret=pfm_dispatch_events(&evt)) != PFMLIB_SUCCESS) {
		fatal_error("cannot configure events: %s\n", pfm_strerror(ret));
	}

	allocated_pmd = evt.pfp_pc[0].reg_num;

	ctx[0].ctx_notify_pid = pid;
	ctx[0].ctx_flags      = PFM_FL_INHERIT_NONE;

	/*
	 * now create the context for self monitoring/per-task
	 */
	if (perfmonctl(pid, PFM_CREATE_CONTEXT, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support !\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/* 
	 * Must be done before any PMD/PMD calls (unfreeze PMU). Initialize
	 * PMC/PMD to safe values. psr.up is cleared.
	 */
	if (perfmonctl(pid, PFM_ENABLE, NULL, 0) == -1) {
		fatal_error( "child: perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	/*
	 * now initialize the PMD, with the overflow value, so that it
	 * will overflow at the first occurence of the event. Set
	 * the notify event.
	 */
	pd[0].reg_num         = evt.pfp_pc[0].reg_num;
	pd[0].reg_value       = RESET_VAL;
	pd[0].reg_long_reset  = RESET_VAL;
	pd[0].reg_short_reset = RESET_VAL;
	evt.pfp_pc[0].reg_flags = PFM_REGFL_OVFL_NOTIFY;

	/*
	 * Now program the registers
	 */
	if (perfmonctl(pid, PFM_WRITE_PMCS, evt.pfp_pc, evt.pfp_pc_count) == -1) {
		fatal_error("child: perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	if (perfmonctl(pid, PFM_WRITE_PMDS, pd, evt.pfp_event_count) == -1) {
		fatal_error( "child: perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}

	iloop = loop = argc > 1 ? atoi(argv[1]) : 1;

	/*
	 * Let's roll now
	 */
	pfm_start();

	ret=specloop(loop);

	pfm_stop();

	if (ret == -1) {
		printf("problem with forward emulation ret=%d\n", ret);
		exit(1);
	}

	pd[0].reg_num = evt.pfp_pc[0].reg_num;

	if (perfmonctl(pid, PFM_READ_PMDS, pd, evt.pfp_event_count) == -1) {
		fatal_error( "child: perfmonctl error READ_PMDS errno %d\n",errno);
		return -1;
	}

	printf("%-20lu %s\n", pd[0].reg_value, event);
	printf("Caught event %ld times, loop=%ld\n", handler_called, iloop);
	printf("test : %s\n", handler_called == iloop && good_value == iloop ? "Passed" : "Failed");

	if (handler_called == iloop) ret = 0;

	/* 
	 * let's stop this now
	 */
	if (perfmonctl(pid, PFM_DESTROY_CONTEXT, NULL, 0) == -1) {
		fatal_error( "child: perfmonctl error PFM_DESTROY errno %d\n",errno);
	}

	return ret;
}
