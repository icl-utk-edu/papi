/*
 * pfmon_util.c  - set of helper functions part of the pfmon tool
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
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <ctype.h>
#include <sys/utsname.h>
#include <sys/ptrace.h>
#include <asm/ptrace_offsets.h>

#include "pfmon.h"

/* 
 * architecture specified minimals
 */
#define PFMON_DFL_MAX_COUNTERS  4
#define PFMON_DFL_VA_IMPL_BITS	51

/*
 * This points to an alternative exit function.
 * This function can be registered using register_exit_func()
 * 
 * This mechanism is useful when pthread are used. You don't want
 * a pthread to call exit but pthread_exit.
 */
static void (*pfmon_exit_func)(int);

static void
extract_va_mask(unsigned long *mask)
{
	FILE *fp;	
	char *p;
	unsigned long val = PFMON_DFL_VA_IMPL_BITS;
	char buffer[64];

#ifdef NUE_HACK
	fp = fopen("/proc/pal/cpu0/vm_info", "r");
	if (fp == NULL)
		fp = fopen("/tmp/pal/cpu0/vm_info", "r");
	else
#endif
	fp = fopen("/proc/pal/cpu0/vm_info", "r");
	if (fp == NULL) goto error;

	for (;;) {
		p  = fgets(buffer, sizeof(buffer)-1, fp);
		if (p == NULL) break;

		p = strchr(buffer, ':');
		if (p == NULL) goto error;

		*p = '\0'; 

		if (!strncmp("Virtual Address Space", buffer, 21))
			val = atoi(p+2);
	}
error:
	if (fp) fclose(fp);
	*mask = ((1UL << val) - 1) | (7UL << 61);
}

static void
extract_dbr_info(unsigned long *ndbrs, unsigned long *nibrs)
{
	FILE *fp;	
	char *p;
	unsigned long ibr = 0;
	unsigned long dbr = 0;
	char buffer[256];

#ifdef NUE_HACK
	fp = fopen("/proc/pal/cpu0/register_info", "r");
	if (fp == NULL)
		fp = fopen("/tmp/pal/cpu0/register_info", "r");
	else
#endif
	fp = fopen("/proc/pal/cpu0/register_info", "r");
	if (fp == NULL) goto error;

	for (;;) {
		p  = fgets(buffer, sizeof(buffer)-1, fp);
		if (p == NULL) break;

		p = strchr(buffer, ':');
		if (p == NULL) goto error;

		*p = '\0'; 
		if (!strncmp("Instruction debug register", buffer, 26)) {
			ibr = atoi(p+2);
			continue;
		}

		if (!strncmp("Data debug register", buffer, 19)) {
			dbr = atoi(p+2);
			break;
		}
	}
error:
	if (fp) fclose(fp);

	*nibrs = ibr << 1;
	*ndbrs = dbr << 1;
}

#if 0
static void
extract_simple_numeric(char *file, char *field, unsigned long *num)
{
	FILE *fp;	
	char *p;
	unsigned long val = 0;
	char buffer[64];

	fp = fopen(file, "r");
	if (fp == NULL) goto error;

	for (;;) {
		p  = fgets(buffer, sizeof(buffer)-1, fp);
		if (p == NULL) break;

		p = strchr(buffer, ':');
		if (p == NULL) goto error;

		*p = '\0'; 

		if (!strncmp(field, buffer, strlen(field))) val = atoi(p+2);
	}
error:
	if (fp) fclose(fp);

	*num = val;
}
#endif

static void
extract_max_counters(unsigned long *count)
{
	FILE *fp;	
	char *p;
	unsigned long val = PFMON_DFL_MAX_COUNTERS;
	char buffer[64];

#ifdef NUE_HACK
	fp = fopen("/proc/pal/cpu0/perfmon_info", "r");
	if (fp == NULL)
		fp = fopen("/tmp/pal/cpu0/perfmon_info", "r");
	else
#endif
	fp = fopen("/proc/pal/cpu0/perfmon_info", "r");
	if (fp == NULL) goto error;

	for (;;) {
		p  = fgets(buffer, sizeof(buffer)-1, fp);
		if (p == NULL) break;

		p = strchr(buffer, ':');
		if (p == NULL) goto error;

		*p = '\0'; 

		if (!strncmp("PMC/PMD pairs", buffer, 13))
			val = atoi(p+2);
	}
error:
	if (fp) fclose(fp);

	*count = val;
}

void
extract_pal_info(program_options_t *options)
{
	extract_va_mask(&options->va_impl_mask);
	extract_dbr_info(&options->ndbrs, &options->nibrs);
	extract_max_counters(&options->max_counters);
}

void
warning(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

int
register_exit_function(void (*func)(int))
{
	pfmon_exit_func = func;

	return 0;
}

void
fatal_error(char *fmt, ...) 
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	if (pfmon_exit_func == NULL) exit(1);

	(*pfmon_exit_func)(1);
	/* NOT REACHED */
}

void
vbprintf(char *fmt, ...)
{
	va_list ap;

	if (options.opt_verbose == 0) return;

	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
}


char *
priv_level_str(unsigned long plm)
{
	static char *priv_levels[]={
		"nothing",
		"kernel",
		"(1)",
		"kernel+(1)",
		"(2)",
		"kernel+(2)",
		"(1)+(2)",
		"kernel+(1)+(2)",
		"user",
		"kernel+user",
		"(1)+user",
		"kernel+(1)+user",
		"(2)+user",
		"kernel+(2)+user",
		"kernel+(1)+(2)+user"
	};

	if (plm > 0xf) return "invalid";

	return priv_levels[plm];
}

/*
 * XXX: not safe to call from signal handler (stdio lock)
 */
void
print_palinfo(int fd, int cpuid)
{
	FILE *fp;	
	char fn[64], *p;
	char buffer[64], *value = NULL;
	int cache_lvl = 0;
	char cache_name[64];
	int lsz=0, st_lat=0, sz=0;

	sprintf(fn, "/proc/pal/cpu%d/version_info", cpuid);
	fp = fopen(fn, "r");
	if (fp == NULL) return;
	for (;;) {
		p  = fgets(buffer, 64, fp);
		if (p == NULL) break;

		p = strchr(buffer, ':');
		if (p == NULL) goto end_it;	
		*p = '\0'; value = p+2;

		if (!strncmp("PAL_", buffer, 4)&& (buffer[4] == 'A' || buffer[4] == 'B')) {
			buffer[5] = '\0';
			p = strchr(value, ' ');
			if (p) *p = '\0';
			safe_fprintf(fd, "#\t%s: %s\n", buffer, value);
		}
	}
	fclose(fp);

	sprintf(fn, "/proc/pal/cpu%d/cache_info", cpuid);
	fp = fopen(fn, "r");
	if (fp == NULL) return;
	for (;;) {	
		p  = fgets(buffer, 64, fp);
		if (p == NULL) break;

		/* skip  blank lines */
		if (*p == '\n') continue;

		p = strchr(buffer, ':');
		if (p == NULL) goto end_it;	

		*p = '\0'; value = p+2;

		if (buffer[0] != '\t') {
			if (strchr(buffer, '/'))
				sprintf(cache_name, "L%c ", buffer[strlen(buffer)-1]);
			else {
				sprintf(cache_name, "L%c%c",
						buffer[strlen(buffer)-1],
						buffer[0]);
			}
		}

		if (!strncmp("Cache levels", buffer, 12)) {
			cache_lvl = atoi(value);
			continue;
		}
		if (!strncmp("Unique caches",buffer, 13)) {
			int s = atoi(value);
			safe_fprintf(fd, "#\tCache levels: %d Unique caches: %d\n", cache_lvl, s);
			continue;
		}
		/* skip tab */
		p = buffer+1;
		if (!strncmp("Size", p, 4)) {
			sz = atoi(value);
			continue;
		}	
		if (!strncmp("Store latency", p, 13)) {
			st_lat = atoi(value);
			continue;
		}
		if (!strncmp("Line size", p, 9)) {
			lsz = atoi(value);
			continue;
		}
		if (!strncmp("Load latency", p, 12)) {
			int s = atoi(value);
			safe_fprintf(fd, "#\t%s: %8d bytes, line %3d bytes, load_lat %3d, store_lat %3d\n", 
					cache_name, sz, lsz, s, st_lat);
		}
	}
end_it:
	fclose(fp);
}

#define CPU_MAX_STEPPINGS	12
typedef struct {
	char *name;
	char *steppings[CPU_MAX_STEPPINGS];
} cpu_info_t;

static const cpu_info_t cpu_info[]={
	{ "Itanium",  {0,0,0,0,"B3",0, "C0","C1", 0}},
	{ NULL, {0,}}
};

static int
find_cpu_type(char *name)
{
	int i;
	for(i=0; cpu_info[i].name; i++) 
		if (!strcmp(cpu_info[i].name, name)) return i;
	return -1;
}

/*
 * XXX: not safe to call from signal handler (stdio lock)
 */
void
print_cpuinfo(int fd)
{
	FILE *fp;	
	char buffer[64], *p, *value;
	int cpuid = 0, cpu_type=0, rev = 0;

	fp = fopen("/proc/cpuinfo", "r");
	if (fp == NULL) return;

	//p = fscanf(outfp, "%s %*s %s", buffer, value);
	for (;;) {
		buffer[0] = '\0';

		p  = fgets(buffer, 64, fp);
		if (p == NULL) goto end_it;

		/* skip  blank lines */
		if (*p == '\n') continue;

		p = strchr(buffer, ':');
		if (p == NULL) goto end_it;	

		/* 
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0'; value = p+2; value[strlen(value)-1] = '\0';

		if (!strncmp("processor", buffer, 9)) {
			cpuid = atoi(value);
			continue;
		}

		if (!strncmp("family", buffer, 6)) {
			cpu_type = find_cpu_type(value);
			continue;
		}

		if (!strncmp("revision", buffer, 8)) {
			rev = atoi(value);
			continue;
		}
		if (!strncmp("cpu MHz", buffer, 7)) {
			char *name, *rstr;
			float f;

			sscanf(value, "%f", &f);

			if (cpu_type == -1) {
				name = "Unknown";
				rstr = NULL;
			} else {
				name = cpu_info[cpu_type].name;
				if (rev >= 0 && rev < CPU_MAX_STEPPINGS)
					rstr = cpu_info[cpu_type].steppings[rev]; 
				else 
					rstr = NULL;
			}	
			if (rstr == NULL) rstr = "??";

			safe_fprintf(fd, "# CPU%d %s %s %4.0f Mhz\n", cpuid, 
					name,
					rstr, f);

			print_palinfo(fd, cpuid);
		}
	}

end_it:
	fclose(fp);
}
	
void
gen_reverse_table(pfmlib_param_t *evt, pfarg_reg_t *pc, int *rev_pc)
{
	int i;

	/* first initialize the array. We cannot use 0 as this 
	 * is the index of the first event
	 */
	for (i=0; i < PMU_MAX_PMDS; i++) {
		rev_pc[i] = -1;
	}
	for (i=0; i < evt->pfp_count; i++) {
		rev_pc[pc[i].reg_num] = i; /* point to corresponding monitor_event */
	}
}

int
protect_context(pid_t pid)
{
	if (perfmonctl(pid, PFM_PROTECT_CONTEXT, NULL, 0) == -1) {
		fatal_error( "child: perfmonctl error PFM_PROTECT_CONTEXT %s\n",strerror(errno));
	}
	return 0;
}

int
enable_pmu(pid_t pid)
{
	int ret;

	if ((ret = perfmonctl(pid, PFM_ENABLE, NULL, 0)) == -1) {
		warning( "child: perfmonctl error PFM_ENABLE %s\n", strerror(errno));
	}
	return ret;
}

int
session_start(pid_t pid)
{
	int ret;

	if ((ret = perfmonctl(pid, PFM_START, NULL, 0)) == -1) {
		warning( "error PFM_START: %s\n", strerror(errno));
	}
	return ret;
}

int
session_stop(pid_t pid)
{
	int ret;

	if ((ret = perfmonctl(pid, PFM_STOP, NULL, 0)) == -1) {
		warning( "error PFM_STOP: %s\n", strerror(errno));
	}
	return ret;
}
	
int
gen_event_list(char *arg, int *opt_lst)
{
	char *p;
	int ev;
	int cnt=0;

	while (arg) {
		if (cnt == options.max_counters) goto too_many;

		p = strchr(arg,',');

		if (p) *p = '\0';

		/* must match vcode only */
		if (pfm_find_event(arg,0, &ev) != PFMLIB_SUCCESS) goto error;
		
		/* place the comma back so that we preserve the argument list */
		if (p) *p++ = ',';

		opt_lst[cnt++] = ev;

		arg = p;
	}
	return cnt;
error:
	fatal_error("unknown event %s\n", arg);
too_many:
	fatal_error("too many events specified, max=%d\n", options.max_counters);
	/* NO RETURN */
	return -1;
}

int
gen_event_smpl_rates(char *arg, int count, unsigned long *opt_lst)
{
	char *p, *endptr = NULL;
	unsigned long val;
	int cnt=0;

	while (arg) {

		if (cnt == count) goto too_many;

		p = strchr(arg,',');

		if ( p ) *p++ = '\0';

		/* must match vcode only */
		val =strtoul(arg, &endptr, 10);
		if (*endptr !=',' && *endptr != '\0') goto error;

		opt_lst[cnt++] = val * -1;

		arg = p;
	}
	return cnt;
error:
	if (*arg == '\0')
		warning("empty rate specified\n");
	else
		warning("invalid rate %s\n", arg);
	return -1;
too_many:
	warning("too many rates specified, max=%d\n", count);
	return -1;
}

/*
 * XXX: cannot be called from a signal handler (stdio locking)
 */
int
find_cpu(pid_t pid)
{
	FILE *fp;	
	char *p;
	char fn[32];
	char buffer[1024];

	sprintf(fn, "/proc/%d/stat", pid);

	fp = fopen(fn, "r");
	if (fp == NULL) return -1;

	p  = fgets(buffer, sizeof(buffer)-1, fp);
	if (p == NULL) goto error;

	/* remove \n */
	p[strlen(p)] = '\0';

	p = strrchr(buffer, ' ');
	if (p == NULL) goto error;

	fclose(fp);

	return atoi(p);
error:
	if (fp) fclose(fp);
	return -1;
}

void
print_standard_header(int fd, unsigned long cpu_mask)
{
	char **argv = options.argv;
	char *name;
	struct utsname uts;
	time_t t;
	int i;

	uname(&uts);
	time(&t);

	safe_fprintf(fd, "#\n# date: %s", asctime(localtime(&t)));
	safe_fprintf(fd, "#\n# hostname: %s\n", uts.nodename);
	safe_fprintf(fd, "#\n# kernel version: %s %s %s\n", 
			uts.sysname, 
			uts.release, 
			uts.version);

	safe_fprintf(fd, "#\n# pfmon version: "PFMON_VERSION"\n");
	safe_fprintf(fd, "# kernel perfmon version: %u.%u\n#\n#\n", 
			PFM_VERSION_MAJOR(options.pfm_version),
			PFM_VERSION_MINOR(options.pfm_version));

	safe_fprintf(fd, "#\n# page size: %u bytes\n", getpagesize());
	safe_fprintf(fd, "# CLK_TCK: %lu ticks/second\n", sysconf(_SC_CLK_TCK));
	safe_fprintf(fd, "# CPU configured: %lu\n# CPU online: %lu\n", sysconf(_SC_NPROCESSORS_CONF), sysconf(_SC_NPROCESSORS_ONLN));
	safe_fprintf(fd, "# physical memory: %lu\n# physical memory available: %lu\n#\n", sysconf(_SC_PHYS_PAGES)*getpagesize(), sysconf(_SC_AVPHYS_PAGES)*getpagesize());

	print_cpuinfo(fd);

	safe_fprintf(fd, "#\n#\n# captured events:\n");

	for(i=0; i < PMU_MAX_PMDS; i++) {
		if (options.rev_pc[i] == -1) continue;

		pfm_get_event_name(options.monitor_events[options.rev_pc[i]], &name);

		safe_fprintf(fd, "#\tPMD%d: %s, %s level(s)\n", 
				i,
				name,
				options.monitor_plm[options.rev_pc[i]] ? 
				  priv_level_str(options.monitor_plm[options.rev_pc[i]]) : 
				  priv_level_str(options.opt_plm));
	} 
	safe_fprintf(fd, "#\n");


	//safe_fprintf(fd, "#\n# default privilege level: %s\n", priv_level_str(options.opt_plm));

	if (options.opt_syst_wide) {
		safe_fprintf(fd, "# monitoring mode: system wide\n");
	} else {
		safe_fprintf(fd, "# monitoring mode: per-process\n");
	}
	/*
	 * invoke CPU model specific routine to print any additional information
	 */
	if (pfmon_current->pfmon_print_header) {
		pfmon_current->pfmon_print_header(fd);
	}

	safe_fprintf(fd, "#\n# command:");
	while (*argv) safe_fprintf(fd, " %s", *argv++);

	safe_fprintf(fd, "\n#\n");
	
	if (options.opt_syst_wide) {
		int i;

		safe_fprintf(fd, "# results captured on: ");

		for(i=0; cpu_mask; i++, cpu_mask>>=1) {
			if (cpu_mask & 0x1) safe_fprintf(fd, "CPU%d ", i);
		}
		safe_fprintf(fd, "\n");
	}

	safe_fprintf(fd, "#\n#\n");
}

/*
 * thread-safe, lock-free fprintf. This is needed to fprintf() from
 * a signal handler.
 */
int
safe_fprintf(int fd, char *fmt,...)
{
#	define SAFE_FPRINTF_BUF_LEN	512
	va_list ap;
	int r;
	char buf[SAFE_FPRINTF_BUF_LEN];


	va_start(ap, fmt);
	r = vsnprintf(buf, SAFE_FPRINTF_BUF_LEN, fmt, ap);
	va_end(ap);

	if (r < -1) r = SAFE_FPRINTF_BUF_LEN;

	/*
	 * XXX: need to be optimized
	 */
	return write(fd, buf, r);
}

typedef union {
	struct {
		unsigned long mask:56;
		unsigned long plm:4;
		unsigned long ig:3;
		unsigned long x:1;
	} m;
	unsigned long value;
} dbreg_mask_reg_t;



#define IBR_ADDR_OFFSET(x)	((x <<4) + PT_IBR)
#define IBR_CTRL_OFFSET(x)	(IBR_ADDR_OFFSET(x)+8)

#define IBR_MAX 4

#define PRIV_LEVEL_USER_MASK	(1<<3)


/*
 * this function sets a code breakpoint at bundle address
 * In our context, we only support this features from user level code (of course). It is 
 * not possible to set kernel level breakpoints.
 *
 * The dbreg argument varies from 0 to 4, the configuration registers are not directly
 * visible.
 */
int
set_code_breakpoint(pid_t pid, int dbreg, unsigned long address)
{
	dbreg_mask_reg_t mask;
	int r, real_dbreg;

	if (dbreg < 0 || dbreg > IBR_MAX) return -1;

	real_dbreg = dbreg << 1;

	r = ptrace(PTRACE_POKEUSER, pid, IBR_ADDR_OFFSET(real_dbreg), address);
	if (r == -1) return -1;

	/*
	 * initialize mask
	 */
	mask.value = 0UL;

	mask.m.x   = 1;
	mask.m.plm  = PRIV_LEVEL_USER_MASK;
	/* 
	 * we want exact match here 
	 */
	mask.m.mask = ~0UL; 

	return ptrace(PTRACE_POKEUSER, pid, IBR_CTRL_OFFSET(real_dbreg), mask.value);
}
	
/*
 * Set the desginated bit in the psr. 
 *
 * If mode is:
 *	> 0 : the bit is set
 *	0   : the bit is cleared
 */ 
int
set_psr_bit(pid_t pid, int bit, int mode)
{
	unsigned long psr;

	psr = ptrace(PTRACE_PEEKUSER, pid, PT_CR_IPSR, 0);
	if (psr == -1) return -1;

	/*
	 * set the psr.up bit
	 */
	if (mode) 
		psr |= 1UL << bit;
	else
		psr &= ~(1UL << bit);

	return ptrace(PTRACE_POKEUSER, pid, PT_CR_IPSR, psr);
}

int
gen_priv_levels(char *arg, unsigned int *lvl_lst)
{
	static const struct {
		char *name;
		unsigned int val;
	} priv_lvls[]={
		{ "", 0  }, /* empty element: indicate use default value set by pfmon */
		{ "k", 1 },
		{ "u", 8 },
		{ "uk", 9 },
		{ "ku", 9 },
		{ NULL, 0}
	};

	char *p;
	int i, cnt=0;

	while (arg) {
		if (cnt == options.max_counters) goto too_many;

		p = strchr(arg,',');
			
		if (p) *p = '\0';

		for (i=0 ; priv_lvls[i].name; i++) {
			if (!strcmp(priv_lvls[i].name, arg)) goto found;
		}
		goto error;
found:
		
		/* place the comma back so that we preserve the argument list */
		if (p) *p++ = ',';

		lvl_lst[cnt++] = priv_lvls[i].val;

		arg = p;
	}
	return cnt;
error:
	if (p) *p = ',';
	warning("unknown per event privilege level %s (choices are k,u, uk)\n", arg);
	return -1;
too_many:
	warning("too many per event privilege levels specified, max=%d\n", options.max_counters);
	return -1;
}

/*
 * we abuse libpfm's return values here
 */
int
convert_data_rr_param(char *param, unsigned long *start, unsigned long *end)
{
	char *endptr;

	if (isdigit(param[0])) {
		endptr = NULL;
		*start  = strtoul(param, &endptr, 0);

		if (*endptr != '\0') return PFMLIB_ERR_INVAL;

		return 0;

	}

	load_symbols(options.symbol_file);

	return find_data_symbol_addr(param, start, end);
}

int
convert_code_rr_param(char *param, unsigned long *start, unsigned long *end)
{
	char *endptr;
	int ret;

	if (isdigit(param[0])) {
		endptr = NULL;
		*start  = strtoul(param, &endptr, 0);

		if (*endptr != '\0') return PFMLIB_ERR_INVAL;

		return 0;

	}

	if ((ret = load_symbols(options.symbol_file))) return ret;

	return find_code_symbol_addr(param, start, end);
}


static void
gen_range(char *arg, unsigned long *start, unsigned long *end, int (*convert)(char *, unsigned long *, unsigned long *))
{
	char *p;
	unsigned long *p_end = NULL;
	int ret;

	p = strchr(arg,'-');
	if (p == arg) goto error;

	if (p == NULL)
		p_end = end;
	else
		*p='\0';

	ret = (*convert)(arg, start, p_end);
	if (ret != PFMLIB_SUCCESS) goto error_convert;

	if (p == NULL)  return;

	arg = p+1;
	if (*arg == '\0') goto error;

	ret = (*convert)(arg, end, NULL);
	if (ret != PFMLIB_SUCCESS) goto error_convert;

	if (*end <= *start) {
		fatal_error("invalid address range [0x%lx-0x%lx]\n", *start, *end);
	}
	return;

error_convert:
	if (ret == PFMLIB_ERR_INVAL) fatal_error("invalid argument: %s\n", arg);
	if (ret == PFMLIB_ERR_NOTFOUND) fatal_error("symbol not found: %s\n", arg);
	if (ret == PFMLIB_ERR_NOTSUPP) fatal_error("symbol %s is from shared object: cannot resolve\n", arg);
error:
	fatal_error("invalid address range specification. Must be start-end\n");
}


void
gen_data_range(char *arg, unsigned long *start, unsigned long *end)
{
	return gen_range(arg, start, end, convert_data_rr_param);
}
	
void
gen_code_range(char *arg, unsigned long *start, unsigned long *end)
{
	return gen_range(arg, start, end, convert_code_rr_param);
}

static void
dec2sep(char *str2, char *str, char sep)
{
	int i, l, b, j, c=0;

	l = strlen(str2);
	if (l <= 3) {
		strcpy(str, str2);
		return;
	}
	b = l +  l /3 - (l%3 ==0); /* l%3=correction to avoid extraneous comma at the end */
	for(i=l, j=0; i >= 0; i--, j++) {
		if (j) c++;
		str[b-j] = str2[i];
		//printf("str2[%d]=%c str[%d]=%c b=%d i=%d j=%d l=%d c=%d\n", i, str2[i], b-j, str[b-j], b, i ,j, l, c);
		if (c == 3) {
			str[b-++j] = sep;
			c = 0;
			//printf("str2[%d]=%c str[%d]=%c b=%d i=%d j=%d l=%d c=%d\n", i, str2[i], b-j, str[b-j], b, i ,j, l, c);
		}
	}
}

void
counter2str(unsigned long count, char *str)
{
	char str2[32];

	switch(options.opt_print_cnt_mode) {
		case 1:
			sprintf(str2, "%lu", count);
			dec2sep(str2,str, ',');
			break;
		case 2:
			sprintf(str2, "%lu", count);
			dec2sep(str2,str, '.');
			break;
		case 3:
			sprintf(str, "0x%016lx", count);
			break;
		default:
			sprintf(str, "%lu", count);
			break;
	}
}


