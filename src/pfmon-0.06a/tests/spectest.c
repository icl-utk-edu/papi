/*
 * 
 *
 * Copyright (C) 2001 Hewlett-Packard Co
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

#include "pfmlib.h"
#include "mysiginfo.h"

#define EVENT_NAME "INST_FAILED_CHKS_RETIRED.ALL"
#define RESET_VAL  0xffffffffL

#define test_chks(res) \
{ \
	__asm__ __volatile__(  \
		"ld8.s r30=[r0]\n" \
		";;\n" \
		"chk.s r30, test_fw\n" \
		";;\n" \
		"mov %0=1\n" \
		";;\n" \
		"test_fw:\n" \
		"mov %0=2;;\n" \
		: "=r"(res):: "r30", "memory"); \
}

int
specloop(int loop)
{
	int res;

	while ( loop-- ) {
		res=-7;
		test_chks(res);

		if (res != 2) return -1;
	}
	return 0;
}

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
fatal_error(char *fmt, ...) 
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}

static	perfmon_req_t pd[1];
void
cnt_handler(int n, struct mysiginfo *info, struct sigcontext *sc)
{
	printf("Overflow notification: pid=%d @0x%lx bv=0x%lx\n", info->sy_pid, sc->sc_ip, info->sy_pfm_ovfl);
	if (perfmonctl(info->sy_pid, PFM_READ_PMDS, 0, pd, 1) == -1) {
		fatal_error("overflow cannot read pmd in %d\n", info->sy_pid);
	}
	printf("PMD[4]=0x%lx\n", pd[0].pfr_reg.reg_value);

	if (perfmonctl(info->sy_pid, PFM_RESTART, 0, 0, 0) == -1) {
		fatal_error("overflow cannot restart process %d\n", info->sy_pid);
	}
}



int
main(int argc, char **argv)
{
	int ret, cnt;
	int loop, pid;
	pfm_event_config_t evt;
	perfmon_req_t pc[1];
	perfmon_req_t ctx[1];
	struct sigaction act;
	pfmlib_options_t pfmlib_options;

	memset(&pfmlib_options, 0, sizeof(pfmlib_options));
	memset(pc, 0, sizeof(pc));
	memset(pd, 0, sizeof(pd));
	memset(ctx, 0, sizeof(ctx));
	memset(&evt,0, sizeof(evt));
	memset(&act,0,sizeof(act));

	pfmlib_options.pfm_debug = 0; /* set to 1 for debug */

	pfmlib_config(&pfmlib_options);


	memset(&act,0,sizeof(act));

	act.sa_handler = (sig_t)cnt_handler;
	sigaction (SIGPROF, &act, 0);

	pid = getpid();

	memset(&evt,0, sizeof(evt));

	evt.pec_evt[0] = pfm_findevent(EVENT_NAME,0);

	if (evt.pec_evt[0] == -1) {
		fatal_error("Cannot find %s event", EVENT_NAME);
	}
	evt.pec_plm   = PFM_PLM0; /* wait for C0 for fix */
	evt.pec_count = cnt = 1;

	if (pfm_dispatch_events(&evt, pc, &cnt) == -1) {
		fatal_error("Can't dispatch event");
	}

	ctx[0].pfr_ctx.notify_pid = pid;
	ctx[0].pfr_ctx.notify_sig = SIGPROF;
	ctx[0].pfr_ctx.flags      = PFM_FL_INHERIT_NONE;

	if (perfmonctl(pid, PFM_CREATE_CONTEXT, 0, ctx, 1) == -1 ) {
		if (errno == ENOSYS) {
			fatal_error("Your kernel does not have performance monitoring support !\n");
		}
		fatal_error("Can't create PFM context %s\n", strerror(errno));
	}
	/* Must be done before any PMD/PMD calls */
	if (perfmonctl(pid, PFM_ENABLE, 0, NULL, 0) == -1) {
		fatal_error( "child: perfmonctl error PFM_ENABLE errno %d\n",errno);
	}

	pd[0].pfr_reg.reg_num        = pc[0].pfr_reg.reg_num;
	pd[0].pfr_reg.reg_value      = RESET_VAL;
	pd[0].pfr_reg.reg_smpl_reset = RESET_VAL;
	pd[0].pfr_reg.reg_ovfl_reset = RESET_VAL;

	pc[0].pfr_reg.reg_flags      = PFM_REGFL_OVFL_NOTIFY;

	if (perfmonctl(pid, PFM_WRITE_PMCS, 0, pc, cnt) == -1) {
		fatal_error("child: perfmonctl error PFM_WRITE_PMCS errno %d\n",errno);
	}
	if (perfmonctl(pid, PFM_WRITE_PMDS, 0, pd, evt.pec_count) == -1) {
		fatal_error( "child: perfmonctl error PFM_WRITE_PMDS errno %d\n",errno);
	}
	loop = argc > 1 ? atoi(argv[1]) : 1;

	pfm_start();

	ret=specloop(loop);

	pfm_stop();

	if (ret == -1) printf("problem with forward emulation ret=%d\n", ret);

	pd[0].pfr_reg.reg_num = pc[0].pfr_reg.reg_num;

	if (perfmonctl(pid, PFM_READ_PMDS, 0, pd, evt.pec_count) == -1) {
		fatal_error( "child: perfmonctl error READ_PMDS errno %d\n",errno);
		return -1;
	}

	printf("%-16ld %s\n", pd[0].pfr_reg.reg_value, EVENT_NAME);

	if (perfmonctl(pid, PFM_DISABLE, 0, NULL, 0) == -1) {
		fatal_error( "child: perfmonctl error PFM_DISABLE errno %d\n",errno);
	}

	return ret;
}
