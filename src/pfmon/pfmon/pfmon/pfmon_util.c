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
#include <sys/resource.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
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
	//extract_va_mask(&options->va_impl_mask);
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
	
void
dprintf(char *fmt, ...)
{
	va_list ap;

	if (options.opt_debug == 0) return;

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
print_palinfo(FILE *fp, int cpuid)
{
	FILE *fp1;	
	char fn[64], *p;
	char buffer[64], *value = NULL;
	int cache_lvl = 0;
	char cache_name[64];
	int lsz=0, st_lat=0, sz=0;

	sprintf(fn, "/proc/pal/cpu%d/version_info", cpuid);
	fp1 = fopen(fn, "r");
	if (fp1 == NULL) return;
	for (;;) {
		p  = fgets(buffer, 64, fp1);
		if (p == NULL) break;

		p = strchr(buffer, ':');
		if (p == NULL) goto end_it;	
		*p = '\0'; value = p+2;

		if (!strncmp("PAL_", buffer, 4)&& (buffer[4] == 'A' || buffer[4] == 'B')) {
			buffer[5] = '\0';
			p = strchr(value, ' ');
			if (p) *p = '\0';
			fprintf(fp, "#\t%s: %s\n", buffer, value);
		}
	}
	fclose(fp1);

	sprintf(fn, "/proc/pal/cpu%d/cache_info", cpuid);
	fp1 = fopen(fn, "r");
	if (fp1 == NULL) return;
	for (;;) {	
		p  = fgets(buffer, 64, fp1);
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
			fprintf(fp, "#\tCache levels: %d Unique caches: %d\n", cache_lvl, s);
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
			fprintf(fp, "#\t%s: %8d bytes, line %3d bytes, load_lat %3d, store_lat %3d\n", 
				    cache_name, sz, lsz, s, st_lat);
		}
	}
end_it:
	fclose(fp1);
}

static unsigned int
get_cpu_speed(void)
{
	FILE *fp1;
	char buffer[128], *p, *value;
	unsigned int f = 0;

	memset(buffer, 0, sizeof(buffer));

	fp1 = fopen("/proc/cpuinfo", "r");
	if (fp1 == NULL) return f;

	for (;;) {
		buffer[0] = '\0';

		p  = fgets(buffer, 127, fp1);
		if (p == NULL) goto end_it;

		/* skip  blank lines */
		if (*p == '\n') continue;

		p = strchr(buffer, ':');
		if (p == NULL) goto end_it;	

		/* 
		 * p+2: +1 = space, +2= firt character
		 * strlen()-1 gets rid of \n
		 */
		*p = '\0'; 
		value = p+2; 

		value[strlen(value)-1] = '\0';

		if (!strncmp("cpu MHz", buffer, 7)) {
			sscanf(value, "%u", &f);
			break;
		}
	}
end_it:
	fclose(fp1);
	return f;
}

/*
 * layout of CPUID[3] register
 */
typedef union {
	unsigned long value;
	struct {
	unsigned number		:  8;
	unsigned revision	:  8;
	unsigned model		:  8;
	unsigned family		:  8;
	unsigned archrev	:  8;
	unsigned reserved	: 24;
	} cpuid3;
} cpuid3_t;

typedef struct {
	unsigned long cpuid;
	char *family;
	char *code_name;
	char *stepping;
} cpu_info_t;


static cpu_info_t cpu_info[]={
  { 0x7000704 , "Itanium"   , "Merced", "C1" },
  { 0x7000604 , "Itanium"   , "Merced", "C0" },
  { 0x7000404 , "Itanium"   , "Merced", "B3" },
  { 0x7000304 , "Itanium"   , "Merced", "B3" },
  { 0x1f000704, "Itanium 2", "McKinley", "B3" },
  { 0x1f000604, "Itanium 2", "McKinley", "B2" },
  { 0x1f000504, "Itanium 2", "McKinley", "B1" },
  { 0x1f000404, "Itanium 2", "McKinley", "B0" }
};
#define NUM_CPUIDS	(sizeof(cpu_info)/sizeof(cpu_info_t))

#define CPUID_MATCH_ALL		(~0UL)
#define CPUID_MATCH_FAMILY	(0x00000000ff000000)
#define CPUID_MATCH_REVISION	(0x000000000000ff00)
#define CPUID_MATCH_MODEL	(0x0000000000ff0000)

#ifdef __GNUC__
static inline unsigned long
ia64_get_cpuid (unsigned long regnum)
{
	unsigned long r;

	asm ("mov %0=cpuid[%r1]" : "=r"(r) : "rO"(regnum));
	return r;
}

#elif defined(__ECC) && defined(__INTEL_COMPILER)

/* if you do not have this file, your compiler is too old */
#include <ia64intrin.h>

#define ia64_get_cpuid(regnum)	__getIndReg(_IA64_REG_INDR_CPUID, (regnum))

#else /* !GNUC nor INTEL_ECC */
#error "need to define a set of compiler-specific macros"
#endif


static cpu_info_t *
find_cpuid(unsigned long cpuid, unsigned long mask)
{
	unsigned int i;
	cpu_info_t *info;

	for(i=0, info = cpu_info; i < NUM_CPUIDS; i++, info++) {
		if ((info->cpuid & mask) == (cpuid & mask)) return info;
	}
	return NULL;
}


void
print_simple_cpuinfo(FILE *fp, char *msg)
{
	cpu_info_t *info;
	unsigned int freq;
	cpuid3_t cpuid;

	cpuid.value = ia64_get_cpuid(3);
	freq        = get_cpu_speed();

	fprintf(fp, "%s %u-way %uMHz ", msg ? msg : "", options.online_cpus , freq);

	info = find_cpuid(cpuid.value, CPUID_MATCH_ALL);
	if (info) {
		fprintf(fp, "%s (%s, %s)\n", info->family, info->code_name, info->stepping);
		return;
	} 
	info = find_cpuid(cpuid.value, CPUID_MATCH_FAMILY|CPUID_MATCH_MODEL);
	if (info) {
		fprintf(fp, "%s (%s)\n", info->family, info->code_name);
		return;
	}
	info = find_cpuid(cpuid.value, CPUID_MATCH_FAMILY);
	if (info) {
		fprintf(fp, "%s\n", info->family);
		return;
	}
	fprintf(fp, "CPU family %u model %u revision %u\n", cpuid.cpuid3.family, cpuid.cpuid3.model, cpuid.cpuid3.revision);
}

/*
 * XXX: not safe to call from signal handler (stdio lock)
 */
void
print_cpuinfo(FILE *fp)
{
	print_simple_cpuinfo(fp, "# host CPUs: ");
	print_palinfo(fp, 0);
}

void
gen_reverse_table(pfmlib_param_t *evt, int *rev_pc)
{
	int i;

	/* first initialize the array. We cannot use 0 as this 
	 * is the index of the first event
	 */
	for (i=0; i < PMU_MAX_PMDS; i++) {
		rev_pc[i] = -1;
	}
	for (i=0; i < evt->pfp_event_count; i++) {
		rev_pc[evt->pfp_pc[i].reg_num] = i; /* point to corresponding monitor_event */
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
gen_event_list(char *arg, pfmon_monitor_t *events)
{
	char *p;
	int ev;
	int cnt=0;

	while (arg) {
		if (cnt == options.max_counters) goto too_many;

		p = strchr(arg,',');

		if (p) *p = '\0';

		/* must match vcode only */
		if (pfm_find_event(arg, &ev) != PFMLIB_SUCCESS) goto error;
		
		/* place the comma back so that we preserve the argument list */
		if (p) *p++ = ',';

		events[cnt++].event = ev;

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

/*
 * input string written
 */
int
gen_smpl_rates(char *arg, int count, pfmon_smpl_rate_t *rates)
{
	char *p, *endptr = NULL;
	unsigned long val;
	int cnt;

	for(cnt = 0; arg; cnt++) {

		if (cnt == count) goto too_many;

		p = strchr(arg,',');

		if ( p ) *p = '\0';

		val =strtoul(arg, &endptr, 0);

		if ( p ) *p++ = ',';

		if (*endptr != ',' && *endptr != '\0') goto error;

		rates[cnt].value = -val; /* a period is a neagtive number */
		rates[cnt].flags |= PFMON_RATE_VAL_SET;

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
 * input string written
 */
int
gen_smpl_randomization(char *arg, int count, pfmon_smpl_rate_t *rates)
{
	char *p, *endptr = NULL;
	unsigned long val = 0UL;
	int cnt, element = 0;
	char c;

	for(cnt = 0; arg; cnt++) {

		if (cnt == count) goto too_many;

		element = 0; c = 0;

		p = strpbrk(arg,":,");

		if ( p ) { c = *p; *p = '\0'; }

		val =strtoul(arg, &endptr, 0);

		if ( p ) *p++ = c;

		if (*endptr != c && *endptr != '\0') goto error_seed_mask;

		if (val == 0UL) goto invalid_mask;

		rates[cnt].mask  = val;
		rates[cnt].flags |= PFMON_RATE_MASK_SET;

		arg = p;

		if (c == ',' || arg == NULL) continue;

		/* extract optional seed */

		p = strchr(arg,',');

		if (p) *p = '\0';

		val = strtoul(arg, &endptr, 0);

		if (*endptr != '\0') goto error_seed_mask;

		if (p) *p++ = ',';

		rates[cnt].seed = val;
		rates[cnt].flags |= PFMON_RATE_SEED_SET;

		arg = p;
	}
	return cnt;
too_many:
	warning("too many rates specified, max=%d\n", count);
	return -1;
error_seed_mask:
	warning("invalid %s at position %u\n", element ? "seed" : "mask", cnt+1);
	return -1;
invalid_mask:
	warning("invalid mask %lu at position %u, to use all bits use -1\n", val, cnt+1);
	return -1;
}

/*
 * XXX: cannot be called from a signal handler (stdio locking)
 */
int
find_cpu(pid_t pid)
{
#define TASK_CPU_POSITION	39 /* position of the task cpu in /proc/pid/stat */
	FILE *fp;
	int count = TASK_CPU_POSITION;
	char *p, *pp = NULL;
	char fn[32];
	char buffer[1024];

	sprintf(fn, "/proc/%d/stat", pid);

	fp = fopen(fn, "r");
	if (fp == NULL) return -1;

	p  = fgets(buffer, sizeof(buffer)-1, fp);
	if (p == NULL) goto error;

	fclose(fp);

	p = buffer;

	/* remove \n */
	p[strlen(p)-1] = '\0';
	p--;

	while (count-- && p) {
		pp = ++p;
		p = strchr(p, ' ');
	}
	if (count>-1) goto error;

	if (p) *p = '\0';

	DPRINT((">count=%d p=%lx pp=%p pp[0]=%d pp[1]=%d cpu=%d\n", count, (unsigned long)p, pp, pp[0], pp[1], 0));
	return atoi(pp);
error:
	if (fp) fclose(fp);
	return -1;
}

void
print_standard_header(FILE *fp, unsigned long cpu_mask)
{
	char **argv = options.argv;
	char *name;
	struct utsname uts;
	time_t t;
	int i;

	uname(&uts);
	time(&t);

	fprintf(fp, "#\n# date: %s", asctime(localtime(&t)));
	fprintf(fp, "#\n# hostname: %s\n", uts.nodename);
	fprintf(fp, "#\n# kernel version: %s %s %s\n", 
			uts.sysname, 
			uts.release, 
			uts.version);

	fprintf(fp, "#\n# pfmon version: "PFMON_VERSION"\n");
	fprintf(fp, "# kernel perfmon version: %u.%u\n#\n#\n", 
			PFM_VERSION_MAJOR(options.pfm_version),
			PFM_VERSION_MINOR(options.pfm_version));

	fprintf(fp, "#\n# page size: %u bytes\n", getpagesize());
	fprintf(fp, "# CLK_TCK: %lu ticks/second\n", sysconf(_SC_CLK_TCK));
	fprintf(fp, "# CPU configured: %lu\n# CPU online: %lu\n", sysconf(_SC_NPROCESSORS_CONF), sysconf(_SC_NPROCESSORS_ONLN));
	fprintf(fp, "# physical memory: %lu\n# physical memory available: %lu\n#\n", sysconf(_SC_PHYS_PAGES)*getpagesize(), sysconf(_SC_AVPHYS_PAGES)*getpagesize());

	print_cpuinfo(fp);

	fprintf(fp, "#\n#\n# captured events:\n");

	for(i=0; i < PMU_MAX_PMDS; i++) {
		if (options.rev_pc[i] == -1) continue;

		pfm_get_event_name(options.events[options.rev_pc[i]].event, &name);

		fprintf(fp, "#\tPMD%d: %s, %s level(s)\n", 
			i,
			name,
			priv_level_str(options.events[options.rev_pc[i]].plm));
	} 
	fprintf(fp, "#\n");


	//safe_fprintf(fd, "#\n# default privilege level: %s\n", priv_level_str(options.opt_plm));

	if (options.opt_syst_wide) {
		fprintf(fp, "# monitoring mode: system wide\n");
	} else {
		fprintf(fp, "# monitoring mode: per-process\n");
	}
	/*
	 * invoke CPU model specific routine to print any additional information
	 */
	if (pfmon_current->pfmon_print_header) {
		pfmon_current->pfmon_print_header(fp);
	}

	fprintf(fp, "#\n# command:");
	while (*argv) fprintf(fp, " %s", *argv++);

	fprintf(fp, "\n#\n");
	
	if (options.opt_syst_wide) {
		int i;

		fprintf(fp, "# results captured on: ");

		for(i=0; cpu_mask; i++, cpu_mask>>=1) {
			if (cpu_mask & 0x1) fprintf(fp, "CPU%d ", i);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "#\n#\n");
}

#if 0
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
#endif

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
set_code_breakpoint(pid_t pid, int dbreg, unsigned long address, int enable)
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

	mask.m.x   = enable;
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

	load_symbols();

	return find_data_symbol_addr(param, start, end);
}

int
convert_code_rr_param(char *param, unsigned long *start, unsigned long *end)
{
	char *endptr;

	if (isdigit(param[0])) {
		endptr = NULL;
		*start  = strtoul(param, &endptr, 0);

		if (*endptr != '\0') return PFMLIB_ERR_INVAL;

		return 0;

	}

	load_symbols();

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
	gen_range(arg, start, end, convert_data_rr_param);
}
	
void
gen_code_range(char *arg, unsigned long *start, unsigned long *end)
{
	gen_range(arg, start, end, convert_code_rr_param);
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

void
show_task_rusage(struct timeval *start, struct timeval *end, struct rusage *ru)
{
	long secs, suseconds;

	 secs =  end->tv_sec - start->tv_sec;

	if (end->tv_usec < start->tv_usec) {
      		end->tv_usec += 1000000;
      		secs--;
    	}
  	suseconds = end->tv_usec - start->tv_usec;

	printf ("real %ldh%02ldm%02ld.%03lds user %ldh%02ldm%02ld.%03lds sys %ldh%02ldm%02ld.%03lds\n", 
		secs / 3600, 
		(secs % 3600) / 60, 
		secs % 60,
		suseconds / 1000,

		ru->ru_utime.tv_sec / 3600, 
		(ru->ru_utime.tv_sec % 3600) / 60, 
		ru->ru_utime.tv_sec% 60,
		ru->ru_utime.tv_usec / 1000,

		ru->ru_stime.tv_sec / 3600, 
		(ru->ru_stime.tv_sec % 3600) / 60, 
		ru->ru_stime.tv_sec% 60,
		ru->ru_stime.tv_usec / 1000
		);
}

int
is_regular_file(char *name)
{
	struct stat st;

	return stat(name, &st) == -1 || S_ISREG(st.st_mode) ? 1 : 0;
}

void 
check_counter_conflict(pfmlib_param_t *evt, unsigned long max_counter_mask)
{
	pfmlib_event_t *e = evt->pfp_events;
	unsigned long cnt1, cnt2;
	unsigned int cnt = evt->pfp_event_count;
	char *name1, *name2;
	int i, j;

	for (i=0; i < cnt; i++) {
		pfm_get_event_counters(e[i].event, &cnt1);
		if (cnt1 == max_counter_mask) continue;
		for(j=i+1; j < cnt; j++) {
			pfm_get_event_counters(e[j].event, &cnt2);
			if (cnt2 == cnt1 && hweight64(cnt1) == 1) goto error;
		}
	}
	return;
error:
	pfm_get_event_name(e[i].event, &name1);
	pfm_get_event_name(e[j].event, &name2);
	fatal_error("event %s and %s cannot be measured at the same time\n", name1, name2);
}

void
pfmon_check_cpus(void)
{
	/*
	 * XXX: assume that cpu are numbered sequentially starting at 0
	 */
	options.online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
}

