/*
 * pfmon_itanium.h 
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

#ifndef __PFMON_ITANIUM_H__
#define __PFMON_ITANIUM_H__

#include <perfmon/pfmlib_itanium.h>

typedef struct {
	struct {
		int opt_btb_notar;	
		int opt_btb_notac;	
		int opt_btb_tm;	
		int opt_btb_ptm;	
		int opt_btb_ppm;	
		int opt_btb_bpt;	
		int opt_btb_nobac;	
		int opt_ia64;
		int opt_ia32;
		int opt_use_iear_tlb;
		int opt_use_dear_tlb;
	} pfmon_ita_opt_flags;

	char *thres_arg;		/* thresholds options */
	char *irange_str;		/* instruction address range option */
	char *drange_str;		/* data address range option */
	char *chkp_func_str;		/* instruction checkpoint function option */
	char *opcm8_str;		/* opcode matcher pmc8 option */
	char *opcm9_str;		/* opcode matcher pmc9 option */

	pfmlib_ita_param_t *params;	/* libpfm Itanium specific parameter structure */
} pfmon_ita_options_t;

#define opt_btb_notar		pfmon_ita_opt_flags.opt_btb_notar
#define opt_btb_notac		pfmon_ita_opt_flags.opt_btb_notac
#define opt_btb_nobac		pfmon_ita_opt_flags.opt_btb_nobac
#define opt_btb_tm		pfmon_ita_opt_flags.opt_btb_tm
#define opt_btb_ptm		pfmon_ita_opt_flags.opt_btb_ptm
#define opt_btb_ppm		pfmon_ita_opt_flags.opt_btb_ppm
#define opt_ia64		pfmon_ita_opt_flags.opt_ia64
#define opt_ia32		pfmon_ita_opt_flags.opt_ia32
#define opt_use_iear_tlb	pfmon_ita_opt_flags.opt_use_iear_tlb
#define opt_use_dear_tlb	pfmon_ita_opt_flags.opt_use_dear_tlb

#endif /* __PFMON_ITANIUM_H__ */

