/*
 * pfmon.h 
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

#ifndef __PFMON_H__
#define __PFMON_H__
#include <getopt.h>
#include <signal.h>
#include <sys/resource.h>

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib.h>

#define PFMON_VERSION		"1.1"

/* 
 * max number of cpus (threads) supported
 * Note: cannot be greater than 8*sizeof(unsigned long) because of cpu_mask.
 */
#define PFMON_MAX_CPUS		64
/*
 * maximum number of concurrent tasks that pfmon can handle.
 * At this point, this is mostly used in system wide mode, one task
 * per cpu, so this value matches the maximum number of cpus supported
 */
#define PFMON_MAX_CTX		64

#define PFMON_MAX_FILENAME_LEN	256

#ifdef PFMON_DEBUG
#define debug_print warning
#define DPRINT(a)	do { if (options.opt_debug) debug_print a; } while(0);
#else
#define DPRINT(a)
#endif

/*
 * pfmon sampling context information
 */
typedef struct {
	void *smpl_hdr;			/* virtual address of sampling buffer headers */
	FILE *smpl_fp;			/* sampling file descriptor */
	unsigned long *smpl_entry;	/* next sampling entry to output (allocated for each buffer) */
	unsigned long cpu_mask;		/* on which CPU does this apply to (system wide) */
} pfmon_smpl_ctx_t;

/*
 * sampling output format description
 */
typedef struct {
	char		*name;		/* name of the format */
	unsigned long	pmu_mask;	/* mask of which PMUs are supported */
	char		*description;	/* one line of text describing the format */
	int		(*validate)(pfmlib_param_t *evt);
	int		(*open_smpl)(char *filename, pfmon_smpl_ctx_t *smpl);
	int 		(*close_smpl)(pfmon_smpl_ctx_t *csmpl);
	int		(*process_smpl)(pfmon_smpl_ctx_t *smpl);
	int		(*print_header)(pfmon_smpl_ctx_t *smpl);
} pfmon_smpl_output_t;

#define PFMON_PMU_MASK(t)	(1UL<<(t))


typedef struct {
	struct {
		int opt_plm;	/* which privilege level to monitor (more than one possible) */
		int opt_debug;	/* print debug information */
		int opt_verbose;	/* verbose output */
		int opt_append;	/* append to output file */
		int opt_block;		/* block child task on counter overflow */
		int opt_fclone;	/* follow across fork */
		int opt_syst_wide;

		int opt_with_header;   /* generate header on output results (smpl or not) */
		int opt_use_smpl;	/* true if sampling is requested */
		int opt_aggregate_res;	/* aggregate results */
		int opt_print_cnt_mode;	/* mode for printing counters */
		int opt_show_rusage;	/* show process time */
		int opt_sysmap_syms;	/* use System.map format for symbol file */
		int opt_check_evt_only; /* stop after checking the event combination is valid */
	} program_opt_flags;

	char  **argv;			/* full command line */
	char  **command;		/* keep track of the command to execute (per-task) */

	char *opt_outfile;		/* basename for output filename for counters */

	char *smpl_file;		/* basename for sampling output file */
	unsigned long smpl_entries;	/* number of entries in sampling buffer */
	unsigned long smpl_regs;	/* which PMDs are record in sampling buffer */
	unsigned long cpu_mask;		/* which cpu to use in system wide mode */
	unsigned long session_timeout;	/* number of seconds to wait to stop system wide session */

	char		*trigger_addr_str; /* trigger address option */
	unsigned long	trigger_addr;	   /* trigger monitoring when  code address is reached */

	pfmon_smpl_ctx_t smpl_ctx[PFMON_MAX_CTX];

	unsigned long va_impl_mask;	/* bitmask of implemented virtual address bits (from palinfo) */
	unsigned long nibrs;		/* number of available instruction debug register pairs */
	unsigned long ndbrs;		/* number of available data debug register pairs */
	unsigned long max_counters;	/* maximum number of counter for the platform */
	unsigned int  pfm_version;	/* kernel perfmon version */
	unsigned int  pfm_smpl_version;	/* kernel perfmon sampling format version */

	int monitor_count;				/* how many counters specified by user */
	int monitor_events[PMU_MAX_PMDS];		/* event code, user specified order */
	unsigned int monitor_plm[PMU_MAX_PMDS];		/* per event priv level, user order */

	int rev_pc[PMU_MAX_PMDS];			/* pmd[x] -> monitor_events[y] */
	unsigned long 	long_rates[PMU_MAX_PMDS];	/* XXX: constant maybe overkill, should use max_counters */
	unsigned long 	short_rates[PMU_MAX_PMDS];	/* XXX: constant maybe overkill, should use max_counters */

	char *symbol_file;				/* name of file which holds symbol table */

	pfmon_smpl_output_t	*smpl_output;		/* which sampling output format to use */

	void *model_options;				/* point to model specific pfmon options */
} program_options_t;

#define opt_plm			program_opt_flags.opt_plm
#define opt_debug		program_opt_flags.opt_debug
#define opt_verbose		program_opt_flags.opt_verbose
#define opt_append		program_opt_flags.opt_append
#define opt_block		program_opt_flags.opt_block
#define opt_fclone		program_opt_flags.opt_fclone
#define opt_syst_wide		program_opt_flags.opt_syst_wide
#define opt_with_header		program_opt_flags.opt_with_header
#define opt_use_smpl		program_opt_flags.opt_use_smpl
#define opt_aggregate_res	program_opt_flags.opt_aggregate_res
#define opt_print_cnt_mode	program_opt_flags.opt_print_cnt_mode
#define opt_show_rusage		program_opt_flags.opt_show_rusage
#define opt_sysmap_syms		program_opt_flags.opt_sysmap_syms
#define opt_check_evt_only	program_opt_flags.opt_check_evt_only

typedef struct {
	char	*name;		/* support module name */
	int	pmu_type;	/* indicate the PMU type, must be one from pfmlib.h */
	int	(*pfmon_initialize)(pfmlib_param_t *);
	void	(*pfmon_usage)(void);
	int	(*pfmon_parse_options)(int code, char *optarg, pfmlib_param_t *evt);
	int	(*pfmon_post_options)(pfmlib_param_t *evt);
	int	(*pfmon_overflow_handler)(int n, struct pfm_siginfo *info, struct sigcontext *sc);
	int	(*pfmon_install_counters)(pid_t pid, pfmlib_param_t *evt, pfarg_reg_t *pc, int count, pfarg_reg_t *pd);
	int	(*pfmon_print_header)(FILE *fp);
	void	(*pfmon_show_detailed_event_name)(int evt);
} pfmon_support_t;

extern pfmon_support_t *pfmon_current;

extern program_options_t options;

/* from util.c */
extern void extract_pal_info(program_options_t *options);
extern void warning(char *fmt, ...);
extern void fatal_error(char *fmt, ...);
extern char * priv_level_str(unsigned long plm);
extern void print_palinfo(FILE *fp, int cpuid);
extern void print_cpuinfo(FILE *fp);
extern void gen_reverse_table(pfmlib_param_t *evt, pfarg_reg_t *pc, int *rev_pc);
extern int protect_context(pid_t pid);
extern int unprotect_context(pid_t pid);
extern int enable_pmu(pid_t pid);
extern int session_start(pid_t pid);
extern int session_stop(pid_t pid);
extern int gen_event_list(char *arg, int *evt_lst);
extern int gen_priv_levels(char *arg, unsigned int *lvl_lst);
extern int gen_event_smpl_rates(char *arg, int count, unsigned long *opt_lst);
extern int find_cpu(pid_t pid);
extern int register_exit_function(void (*func)(int));
extern void print_standard_header(FILE *fp, unsigned long cpu_mask);
extern int set_code_breakpoint(pid_t pid, int dbreg, unsigned long address);
#define PSR_MODE_CLEAR	0
#define PSR_MODE_SET	1
extern int set_psr_bit(pid_t pid, int bit, int mode);
extern void vbprintf(char *fmt, ...);
extern int convert_code_rr_param(char *param, unsigned long *start, unsigned long *end);
extern int convert_data_rr_param(char *param, unsigned long *start, unsigned long *end);
extern void gen_code_range(char *arg, unsigned long *start, unsigned long *end);
extern void gen_data_range(char *arg, unsigned long *start, unsigned long *end);
extern void counter2str(unsigned long count, char *str);
extern void show_task_rusage(struct timeval *start, struct timeval *end, struct rusage *ru);
extern int is_regular_file(char *name);
extern void check_counter_conflict(pfmlib_param_t *evt, unsigned long max_counter_mask);


/* from pfmon.c */
extern int pfmon_register_options(struct option *cmd, size_t sz);
extern int install_counters(pid_t pid, pfmlib_param_t *evt, pfarg_reg_t *pc, int count);
extern int print_results(pfarg_reg_t *pd, pfmon_smpl_ctx_t *csmpl);

/* pfmon_smpl.c */
extern void setup_sampling_rates(pfmlib_param_t *evt, char *smpl_args, char *ovfl_args);
extern int setup_sampling_output(pfmon_smpl_ctx_t *csmpl);
extern void close_sampling_output(pfmon_smpl_ctx_t *csmpl);
extern int process_smpl_buffer(pfmon_smpl_ctx_t *csmpl);
extern int pfmon_find_smpl_output(char *name, pfmon_smpl_output_t **fmt, int ignore_cpu);
extern void pfmon_list_smpl_outputs(void);
extern void pfmon_smpl_output_info(pfmon_smpl_output_t *fmt);
extern void pfmon_process_smpl_buf(pfmon_smpl_ctx_t *csmpl, pid_t pid);

/* from pfmon_system.c */
extern int measure_system_wide(pfmlib_param_t *evt, pfarg_context_t *ctx, pfarg_reg_t *pc, int count, char **argv);

/* from pfmon_task.c */
extern int measure_per_task(pfmlib_param_t *evt, pfarg_context_t *ctx, pfarg_reg_t *pc, int count, char **argv);

/* from pfmon_symbols.c */
extern void load_symbols(void);
extern int find_code_symbol_addr(char *sym, unsigned long *start, unsigned long *end);
extern int find_data_symbol_addr(char *sym, unsigned long *start, unsigned long *end);
extern void print_symbols(void);

/* from pfmon_config.c */
extern void load_config_file(void);
extern int find_opcode_matcher(char *name, unsigned long *val);
extern void print_opcode_matchers(void);

/*
 * Some useful inline functions
 */
static __inline__ int
hweight64 (unsigned long x)
{
	unsigned long result;
#ifdef __GNUC__
	__asm__ ("popcnt %0=%1" : "=r" (result) : "r" (x));
#elif defined(INTEL_ECC_COMPILER)
	result = _m64_popcnt(x);
#else
#error "you need to provide inline assembly from your compiler"
#endif
	return (int)result;
}


#endif /*__PFMON_H__ */
