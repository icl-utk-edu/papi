/*
 * pfmon_itanium2.h 
 *
 * Copyright (C) 2002 Hewlett-Packard Co
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

#ifndef __PFMON_ITANIUM2_H__
#define __PFMON_ITANIUM2_H__

#include <perfmon/pfmlib_itanium2.h>
typedef struct {
	struct {
		int opt_btb_ds;		/* capture branch predictions instead of targets */
		int opt_btb_tm;		/* taken/not-taken branches only */
		int opt_btb_ptm;	/* predicted target address mask: correct/incorrect */
		int opt_btb_ppm;	/* predicted path: correct/incorrect */
		int opt_btb_brt;	/* branch type mask */
		int opt_ia64;
		int opt_ia32;
		int opt_inv_rr;		/* inverse range restriction on IBRP0 */
	} pfmon_ita2_opt_flags;

	pfmlib_ita2_ear_mode_t dear_mode;
	pfmlib_ita2_ear_mode_t iear_mode;

	char *thres_arg;		/* thresholds options */
	char *irange_str;		/* instruction address range option */
	char *drange_str;		/* data address range option */
	char *chkp_func_str;		/* instruction checkpoint function option */
	char *opcm8_str;		/* opcode matcher pmc8 option */
	char *opcm9_str;		/* opcode matcher pmc9 option */

	pfmlib_ita2_param_t *params;	/* libpfm McKinley specific parameter structure */
} pfmon_ita2_options_t;

#define opt_btb_ds		pfmon_ita2_opt_flags.opt_btb_ds
#define opt_btb_brt		pfmon_ita2_opt_flags.opt_btb_brt
#define opt_btb_tm		pfmon_ita2_opt_flags.opt_btb_tm
#define opt_btb_ptm		pfmon_ita2_opt_flags.opt_btb_ptm
#define opt_btb_ppm		pfmon_ita2_opt_flags.opt_btb_ppm
#define opt_ia64		pfmon_ita2_opt_flags.opt_ia64
#define opt_ia32		pfmon_ita2_opt_flags.opt_ia32
#define opt_use_iear_tlb	pfmon_ita2_opt_flags.opt_use_iear_tlb
#define opt_use_dear_tlb	pfmon_ita2_opt_flags.opt_use_dear_tlb
#define opt_inv_rr		pfmon_ita2_opt_flags.opt_inv_rr


#endif /* __PFMON_ITANIUM2_H__ */
