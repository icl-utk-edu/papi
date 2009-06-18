/* 
* File:    any-null.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@eecs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@eecs.utk.edu
*/

/* PAPI stuff */
#include "papi.h"
#include "papi_internal.h"
#include "any-null.h"
#include "papi_memory.h"

extern papi_vector_t MY_VECTOR;

volatile unsigned int lock[PAPI_MAX_LOCK];

/*
 * Substrate setup and shutdown
 */

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * for the presets are setup here.
 */
hwi_search_t preset_map[] = {
	{PAPI_TOT_CYC, {0, {0x1, PAPI_NULL}, {0,}}},
	{PAPI_L1_DCM, {0, {0x2, PAPI_NULL}, {0,}}},
	{PAPI_TOT_INS, {0, {0x3, PAPI_NULL}, {0,}}},
	{PAPI_FP_OPS, {0, {0x4, PAPI_NULL}, {0,}}},
	{0, {0, {PAPI_NULL, PAPI_NULL}, {0,}}}
};

inline_static pid_t mygettid(void)
{
#ifdef SYS_gettid
	return(syscall(SYS_gettid));
#elif defined(__NR_gettid)
	return(syscall(__NR_gettid));
#else
	return(syscall(1105));
#endif
}

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int _any_init_substrate(int cidx)
{
	int retval = PAPI_OK;

	/* Internal function, doesn't necessarily need to be a function */
	MY_VECTOR.cmp_info.CmpIdx = cidx;

	strcpy(_papi_hwi_system_info.hw_info.vendor_string,"Vendor");
	strcpy(_papi_hwi_system_info.hw_info.model_string,"Model");
	_papi_hwi_system_info.hw_info.mhz = 1;
	_papi_hwi_system_info.hw_info.clock_mhz = 1;
	_papi_hwi_system_info.hw_info.ncpu = 1;
	_papi_hwi_system_info.hw_info.nnodes = 1;
	_papi_hwi_system_info.hw_info.totalcpus = 1;
	strcpy(MY_VECTOR.cmp_info.name, "any-null");	/* Name of the substrate we're using, usually CVS RCS Id */
	strcpy(MY_VECTOR.cmp_info.version, "$Revision$");	/* Version of this substrate, usually CVS Revision */
//   strcpy(_papi_hwi_system_info.sub_info.name,"any-null");              /* Name of the substrate we're using, usually CVS RCS Id */
//   strcpy(_papi_hwi_system_info.sub_info.version,"$Revision$");           /* Version of this substrate, usually CVS Revision */
//   _papi_hwi_system_info.sub_info.num_cntrs = 4;               /* Number of counters the substrate supports */
//   _papi_hwi_system_info.sub_info.num_mpx_cntrs = PAPI_MPX_DEF_DEG; /* Number of counters the substrate (or PAPI) can multiplex */
//   _papi_hwi_system_info.sub_info.num_native_events = 0;       

	retval = _papi_hwi_setup_all_presets(preset_map, NULL);

	return(retval);
}

/*
 * This is called whenever a thread is initialized
 */
static int _any_init(hwd_context_t *ctx)
{
#if defined(USE_PROC_PTTIMER)
	{
		char buf[LINE_MAX];
		int fd;
		sprintf(buf,"/proc/%d/task/%d/stat",getpid(),mygettid());
		fd = open(buf,O_RDONLY);
		if (fd == -1)
		{
			PAPIERROR("open(%s)",buf);
			return(PAPI_ESYS);
		}
		ctx->stat_fd = fd;
	}
#endif
	return(PAPI_OK);
}

static int _any_shutdown(hwd_context_t *ctx)
{
	return(PAPI_OK);
}

static int _any_shutdown_global(void)
{
	return(PAPI_OK);
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
static int _any_init_control_state(hwd_control_state_t *ptr){
	return PAPI_OK;
}

static int _any_update_control_state(hwd_control_state_t *ptr, NativeInfo_t *native, int count, hwd_context_t *ctx){
	return(PAPI_OK);
}

static int _any_start(hwd_context_t *ctx, hwd_control_state_t *ctrl){
	return(PAPI_OK);
}

static int _any_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long **events, int flags)
{
	return(PAPI_OK);
}

static int _any_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
	return(PAPI_OK);
}

static int _any_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
	return(PAPI_OK);
}

static int _any_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *from)
{
	return(PAPI_OK);
}

/*
 * Overflow and profile functions 
 */
static void _any_dispatch_timer(int signal, hwd_siginfo_t *si, void *context)
{
	/* Real function would call the function below with the proper args
	 * _papi_hwi_dispatch_overflow_signal(...);
	 */
	return;
}

static int _any_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
	return(PAPI_OK);
}

static int _any_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
	return(PAPI_OK);
}

static int _any_set_profile(EventSetInfo_t *ESI, int EventIndex, int threashold)
{
	return(PAPI_OK);
}

/*
 * Functions for setting up various options
 */

/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
static int _any_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
	return(PAPI_OK);
}

/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
static int _any_set_domain(hwd_control_state_t *cntrl, int domain)
{
	int found = 0;
	if ( PAPI_DOM_USER & domain ){
		found = 1;
	}
	if ( PAPI_DOM_KERNEL & domain ){
		found = 1;
	}
	if ( PAPI_DOM_OTHER & domain ){
		found = 1;
	}
	if ( !found )
		return(PAPI_EINVAL);
	return(PAPI_OK);
}

/* 
 * Timing Routines
 * These functions should return the highest resolution timers available.
 */
static long long _any_get_real_usec(void)
{
	long long retval;
	struct timeval buffer;
	gettimeofday(&buffer,NULL);
	retval = (long long)(buffer.tv_sec*1000000);
	retval += (long long)(buffer.tv_usec);
	return(retval);
}

static long long _any_get_real_cycles(void)
{
	return(_any_get_real_usec());
}

static long long _any_get_virt_usec(const hwd_context_t * ctx)
{
	long long retval;
#if defined(USE_PROC_PTTIMER)
	{
		char buf[LINE_MAX];
		long long utime, stime;
		int rv, cnt = 0, i = 0;

		rv = read(ctx->stat_fd,buf,LINE_MAX*sizeof(char));
		if (rv == -1)
		{
			PAPIERROR("read()");
			return(PAPI_ESYS);
		}
		lseek(ctx->stat_fd,0,SEEK_SET);

		buf[rv] = '\0';
		SUBDBG("Thread stat file is:%s\n",buf);
		while ((cnt != 13) && (i < rv))
		{
			if (buf[i] == ' ')
			{ cnt++; }
			i++;
		}
		if (cnt != 13)
		{
			PAPIERROR("utime and stime not in thread stat file?");
			return(PAPI_ESBSTR);
		}

		if (sscanf(buf+i,"%llu %llu",&utime,&stime) != 2)
		{
			PAPIERROR("Unable to scan two items from thread stat file at 13th space?");
			return(PAPI_ESBSTR);
		}
		retval = (utime+stime)*(long long)(1000000/sysconf(_SC_CLK_TCK));
	}
#elif defined(HAVE_CLOCK_GETTIME_THREAD)
	{
		struct timespec foo;
		syscall(__NR_clock_gettime,HAVE_CLOCK_GETTIME_THREAD,&foo);
		retval = foo.tv_sec*1000000;
		retval += foo.tv_nsec/1000;
	}
#elif defined(HAVE_PER_THREAD_TIMES)
	{
		struct tms buffer;
		times(&buffer);
		SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime);
		retval = (long long)((buffer.tms_utime+buffer.tms_stime)*(1000000/sysconf(_SC_CLK_TCK)));
		/* NOT CLOCKS_PER_SEC as in the headers! */
	}
#elif defined(HAVE_PER_THREAD_GETRUSAGE)
	{
		struct rusage buffer;
		getrusage(RUSAGE_SELF,&buffer);
		SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime);
		retval = (long long)((buffer.ru_utime.tv_sec + buffer.ru_stime.tv_sec)*1000000);
		retval += (long long)(buffer.ru_utime.tv_usec + buffer.ru_stime.tv_usec);
	}
#else
#error "No working per thread virtual timer"
#endif
	return retval;
}

static long long _any_get_virt_cycles(const hwd_context_t * ctx)
{
	return(_any_get_virt_usec(ctx)*_papi_hwi_system_info.hw_info.mhz);
}

/*
 * Native Event functions
 */
static int _any_add_prog_event(hwd_control_state_t * ctrl, unsigned int EventCode, void *inout, EventInfo_t * EventInfoArray){
	return(PAPI_OK);
}

static int _any_ntv_enum_events(unsigned int *EventCode, int modifier)
{
	if (modifier == PAPI_ENUM_FIRST) {
		*EventCode = PAPI_NATIVE_MASK; /* assumes first native event is always 0x4000000 */
		return (PAPI_OK);
	}
	if (modifier == PAPI_ENUM_EVENTS) {
		if (*EventCode == PAPI_NATIVE_MASK) { /* this is the first and only event */
			return (PAPI_ENOEVNT); /* this is the terminator */
		}
	}
	return (PAPI_EINVAL);
}

static int _any_ntv_name_to_code(char *name, unsigned int *event_code)
{
	*event_code = PAPI_NATIVE_MASK; /* this is always the first native event */
	return(PAPI_OK);
}

static int _any_ntv_code_to_name(unsigned int EventCode, char *name, int len)
{
	strncpy (name, "PAPI_ANY_NULL", len);
	return(PAPI_OK);
}

static int _any_ntv_code_to_descr(unsigned int EventCode, char *descr, int len)
{
	strncpy (descr, "Event doesn't exist, is an example for a skeleton substrate", len);
	return(PAPI_OK);
}

static int _any_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
	return(PAPI_OK);
}

static int _any_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count)
{
	const char str[]="Counter: 0  Event: 0";
	if ( count == 0 ) return(0);

	if ( strlen(str) > name_len ) return(0);

	strcpy(names, str);
	return(1);
}

/* 
 * Counter Allocation Functions, only need to implement if
 *    the substrate needs smart counter allocation.
 */

static int _any_allocate_registers(EventSetInfo_t *ESI) 
{
	return(1);
}

/* Forces the event to be mapped to only counter ctr. */
static void _any_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
}

/* This function examines the event to determine if it can be mapped 
 * to counter ctr.  Returns true if it can, false if it can't. 
 */
static int _any_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
	return(1);
} 

/* This function examines the event to determine if it has a single 
 * exclusive mapping.  Returns true if exlusive, false if 
 * non-exclusive.  
 */
static int _any_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
	return(1);
}

/* This function compares the dst and src events to determine if any 
 * resources are shared. Typically the src event is exclusive, so 
 * this detects a conflict if true. Returns true if conflict, false 
 * if no conflict.  
 */
static int _any_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
	return(0);
}

/* This function removes shared resources available to the src event
 *  from the resources available to the dst event,
 *  and reduces the rank of the dst event accordingly. Typically,
 *  the src event will be exclusive, but the code shouldn't assume it.
 *  Returns nothing.  
 */
static void _any_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
	return;
}

static void _any_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
	return;
}

/*
 * Shared Library Information and other Information Functions
 */
static int _linux_update_shlib_info(void)
{
	char fname[PAPI_HUGE_STR_LEN];
	PAPI_address_map_t *tmp, *tmp2;
	FILE *f;
	char find_data_mapname[PAPI_HUGE_STR_LEN] = "";
	int upper_bound = 0, i, index = 0, find_data_index = 0, count = 0;
	char buf[PAPI_HUGE_STR_LEN + PAPI_HUGE_STR_LEN], perm[5], dev[6], mapname[PAPI_HUGE_STR_LEN];
	unsigned long begin, end, size, inode, foo;

	sprintf(fname, "/proc/%ld/maps", (long)_papi_hwi_system_info.pid);
	f = fopen(fname, "r");

	if (!f)
	{
		PAPIERROR("fopen(%s) returned < 0", fname);
		return(PAPI_OK);
	}

	/* First count up things that look kinda like text segments, this is an upper bound */

	while (1)
	{
		if (fgets(buf, sizeof(buf), f) == NULL)
		{
			if (ferror(f))
			{
				PAPIERROR("fgets(%s, %d) returned < 0", fname, sizeof(buf));
				fclose(f);
				return(PAPI_OK);
			}
			else
				break;
		}

		sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);

		if (strlen(mapname) && (perm[0] == 'r') && (perm[1] != 'w') && (perm[2] == 'x') && (inode != 0))
		{
			upper_bound++;
		}
	}
	if (upper_bound == 0)
	{
		PAPIERROR("No segments found with r-x, inode != 0 and non-NULL mapname");
		fclose(f);
		return(PAPI_OK);
	}

	/* Alloc our temporary space */

	tmp = (PAPI_address_map_t *) papi_calloc(upper_bound, sizeof(PAPI_address_map_t));
	if (tmp == NULL)
	{
		PAPIERROR("calloc(%d) failed", upper_bound*sizeof(PAPI_address_map_t));
		fclose(f);
		return(PAPI_OK);
	}

	rewind(f);
	while (1)
	{
		if (fgets(buf, sizeof(buf), f) == NULL)
		{
			if (ferror(f))
			{
				PAPIERROR("fgets(%s, %d) returned < 0", fname, sizeof(buf));
				fclose(f);
				papi_free(tmp);
				return(PAPI_OK);
			}
			else
				break;
		}

		sscanf(buf, "%lx-%lx %4s %lx %5s %ld %s", &begin, &end, perm, &foo, dev, &inode, mapname);
		size = end - begin;

		if (strlen(mapname) == 0)
			continue;

		if ((strcmp(find_data_mapname,mapname) == 0) && (perm[0] == 'r') && (perm[1] == 'w') && (inode != 0))
		{
			tmp[find_data_index].data_start = (caddr_t) begin;
			tmp[find_data_index].data_end = (caddr_t) (begin + size);
			find_data_mapname[0] = '\0';
		}
		else if ((perm[0] == 'r') && (perm[1] != 'w') && (perm[2] == 'x') && (inode != 0))
		{
			/* Text segment, check if we've seen it before, if so, ignore it. Some entries
			have multiple r-xp entires. */

			for (i=0;i<upper_bound;i++)
			{
				if (strlen(tmp[i].name))
				{
					if (strcmp(mapname,tmp[i].name) == 0)
						break;
				}
				else
				{
					/* Record the text, and indicate that we are to find the data segment, following this map */
					strcpy(tmp[i].name,mapname);
					tmp[i].text_start = (caddr_t) begin;
					tmp[i].text_end = (caddr_t) (begin + size);
					count++;
					strcpy(find_data_mapname,mapname);
					find_data_index = i;
					break;
				}
			}
		}
	}
	if (count == 0)
	{
		PAPIERROR("No segments found with r-x, inode != 0 and non-NULL mapname");
		fclose(f);
		papi_free(tmp);
		return(PAPI_OK);
	}
	fclose(f);

	/* Now condense the list and update exe_info */
	tmp2 = (PAPI_address_map_t *) papi_calloc(count, sizeof(PAPI_address_map_t));
	if (tmp2 == NULL)
	{
		PAPIERROR("calloc(%d) failed", count*sizeof(PAPI_address_map_t));
		papi_free(tmp);
		fclose(f);
		return(PAPI_OK);
	}

	for (i=0;i<count;i++)
	{
		if (strcmp(tmp[i].name,_papi_hwi_system_info.exe_info.fullname) == 0)
		{
			_papi_hwi_system_info.exe_info.address_info.text_start = tmp[i].text_start;
			_papi_hwi_system_info.exe_info.address_info.text_end = tmp[i].text_end;
			_papi_hwi_system_info.exe_info.address_info.data_start = tmp[i].data_start;
			_papi_hwi_system_info.exe_info.address_info.data_end = tmp[i].data_end;
		}
		else
		{
			strcpy(tmp2[index].name,tmp[i].name);
			tmp2[index].text_start = tmp[i].text_start;
			tmp2[index].text_end = tmp[i].text_end;
			tmp2[index].data_start = tmp[i].data_start;
			tmp2[index].data_end = tmp[i].data_end;
			index++;
		}
	}
	papi_free(tmp);

	if (_papi_hwi_system_info.shlib_info.map){
		papi_free(_papi_hwi_system_info.shlib_info.map);
		_papi_hwi_system_info.shlib_info.map = NULL;
	}
	_papi_hwi_system_info.shlib_info.map = tmp2;
	_papi_hwi_system_info.shlib_info.count = index;

	return (PAPI_OK);
}

static char *search_cpu_info(FILE * f, char *search_str, char *line)
{
	/* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
	/* See the PCL home page for the German version of PAPI. */

	char *s;

	while (fgets(line, 256, f) != NULL) {
		if (strncmp(line, search_str, strlen(search_str)) == 0) {
			/* ignore all characters in line up to : */
			for (s = line; *s && (*s != ':'); ++s);
			if (*s)
				return (s);
		}
	}
	return (NULL);

	/* End stolen code */
}

static int _linux_get_system_info(void)
{
	int tmp, retval;
	char maxargs[PAPI_HUGE_STR_LEN], *t, *s;
	pid_t pid;
	float mhz = 0.0;
	FILE *f;

	/* Software info */

	/* Path and args */

	pid = getpid();
	if (pid < 0)
	{ PAPIERROR("getpid() returned < 0"); return(PAPI_ESYS); }
	_papi_hwi_system_info.pid = pid;

	sprintf(maxargs, "/proc/%d/exe", (int) pid);
	if (readlink(maxargs, _papi_hwi_system_info.exe_info.fullname, PAPI_HUGE_STR_LEN) < 0)
	{
		PAPIERROR("readlink(%s) returned < 0", maxargs);
		strcpy(_papi_hwi_system_info.exe_info.fullname,"");
		strcpy(_papi_hwi_system_info.exe_info.address_info.name,"");
	}
	else
	{
		/* basename can modify it's argument */
		strcpy(maxargs,_papi_hwi_system_info.exe_info.fullname);
		strcpy(_papi_hwi_system_info.exe_info.address_info.name, basename(maxargs));
	}

	/* Executable regions, may require reading /proc/pid/maps file */

	retval = _linux_update_shlib_info();

	/* PAPI_preload_option information */

	strcpy(_papi_hwi_system_info.preload_info.lib_preload_env, "LD_PRELOAD");
	_papi_hwi_system_info.preload_info.lib_preload_sep = ' ';
	strcpy(_papi_hwi_system_info.preload_info.lib_dir_env, "LD_LIBRARY_PATH");
	_papi_hwi_system_info.preload_info.lib_dir_sep = ':';

	SUBDBG("Executable is %s\n", _papi_hwi_system_info.exe_info.address_info.name);
	SUBDBG("Full Executable is %s\n", _papi_hwi_system_info.exe_info.fullname);
	SUBDBG("Text: Start %p, End %p, length %d\n",
		_papi_hwi_system_info.exe_info.address_info.text_start,
		_papi_hwi_system_info.exe_info.address_info.text_end,
		(int)(_papi_hwi_system_info.exe_info.address_info.text_end -
		_papi_hwi_system_info.exe_info.address_info.text_start));
	SUBDBG("Data: Start %p, End %p, length %d\n",
		_papi_hwi_system_info.exe_info.address_info.data_start,
		_papi_hwi_system_info.exe_info.address_info.data_end,
		(int)(_papi_hwi_system_info.exe_info.address_info.data_end -
		_papi_hwi_system_info.exe_info.address_info.data_start));
	SUBDBG("Bss: Start %p, End %p, length %d\n",
		_papi_hwi_system_info.exe_info.address_info.bss_start,
		_papi_hwi_system_info.exe_info.address_info.bss_end,
		(int)(_papi_hwi_system_info.exe_info.address_info.bss_end -
		_papi_hwi_system_info.exe_info.address_info.bss_start));

	/* Hardware info */

	_papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	_papi_hwi_system_info.hw_info.nnodes = 1;
	_papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
	_papi_hwi_system_info.hw_info.vendor = -1;

   /* Multiplex info */
/* This structure disappeared from the papi_mdi_t definition
	Don't know why or where it went...
   _papi_hwi_system_info.mpx_info.timer_sig = PAPI_SIGNAL;
   _papi_hwi_system_info.mpx_info.timer_num = PAPI_ITIMER;
   _papi_hwi_system_info.mpx_info.timer_us = PAPI_MPX_DEF_US;
*/

	if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
	{ PAPIERROR("fopen(/proc/cpuinfo) errno %d",errno); return(PAPI_ESYS); }

	/* All of this information may be overwritten by the substrate */

	/* MHZ */
	rewind(f);
	s = search_cpu_info(f, "clock", maxargs);
	if (!s) {
		rewind(f);
		s = search_cpu_info(f, "cpu MHz", maxargs);
	}
	if (s)
		sscanf(s + 1, "%f", &mhz);
	_papi_hwi_system_info.hw_info.mhz = mhz;
	_papi_hwi_system_info.hw_info.clock_mhz = mhz;

	/* Vendor Name */

	rewind(f);
	s = search_cpu_info(f, "vendor_id", maxargs);
	if (s && (t = strchr(s + 2, '\n')))
	{
		*t = '\0';
		strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
	}
	else
	{
		rewind(f);
		s = search_cpu_info(f, "vendor", maxargs);
		if (s && (t = strchr(s + 2, '\n'))) {
			*t = '\0';
			strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
		}
	}

	/* Revision */

	rewind(f);
	s = search_cpu_info(f, "stepping", maxargs);
	if (s)
	{
		sscanf(s + 1, "%d", &tmp);
		_papi_hwi_system_info.hw_info.revision = (float) tmp;
	}
	else
	{
		rewind(f);
		s = search_cpu_info(f, "revision", maxargs);
		if (s)
		{
			sscanf(s + 1, "%d", &tmp);
			_papi_hwi_system_info.hw_info.revision = (float) tmp;
		}
	}

	/* Model Name */

	rewind(f);
	s = search_cpu_info(f, "family", maxargs);
	if (s && (t = strchr(s + 2, '\n')))
	{
		*t = '\0';
		strcpy(_papi_hwi_system_info.hw_info.model_string, s + 2);
	}
	else
	{
		rewind(f);
		s = search_cpu_info(f, "vendor", maxargs);
		if (s && (t = strchr(s + 2, '\n')))
		{
			*t = '\0';
			strcpy(_papi_hwi_system_info.hw_info.vendor_string, s + 2);
		}
	}

	rewind(f);
	s = search_cpu_info(f, "model", maxargs);
	if (s)
	{
		sscanf(s + 1, "%d", &tmp);
		_papi_hwi_system_info.hw_info.model = tmp;
	}

	fclose(f);

	SUBDBG("Found %d %s(%d) %s(%d) CPU's at %f Mhz.\n",
		_papi_hwi_system_info.hw_info.totalcpus,
		_papi_hwi_system_info.hw_info.vendor_string,
		_papi_hwi_system_info.hw_info.vendor,
		_papi_hwi_system_info.hw_info.model_string,
		_papi_hwi_system_info.hw_info.model, _papi_hwi_system_info.hw_info.mhz);

	return (PAPI_OK);
}

static int _any_get_memory_info(PAPI_hw_info_t *hw, int id)
{
	return PAPI_OK;
}

static int _any_get_dmem_info(PAPI_dmem_info_t *d)
{
	d->size = 1;
	d->resident = 2;
	d->high_water_mark = 3;
	d->shared = 4;
	d->text = 5;
	d->library = 6;
	d->heap = 7;
	d->locked = 8;
	d->stack = 9;
	d->pagesize = getpagesize();

	return (PAPI_OK);
}

papi_vector_t _any_vector = {
	.cmp_info = {
		/* default component information (unspecified values are initialized to 0) */
		.num_cntrs =			MAX_COUNTERS,			/* Number of counters the substrate supports */
		.num_mpx_cntrs =		PAPI_MPX_DEF_DEG,
		.default_domain =		PAPI_DOM_USER,
		.available_domains =	PAPI_DOM_USER|PAPI_DOM_KERNEL,
		.default_granularity =	PAPI_GRN_THR,
		.available_granularities = PAPI_GRN_THR,
		.hardware_intr_sig =	PAPI_INT_SIGNAL,

		/* component specific cmp_info initializations */
		.fast_real_timer =		1,
		.fast_virtual_timer =	1,
		.attach =				1,
		.attach_must_ptrace =	1,
	},

	/* sizes of framework-opaque component-private structures */
	.size = {
		.context =				sizeof(_any_context_t),
		.control_state =		sizeof(_any_control_state_t),
		.reg_value =			sizeof(_any_register_t),
		.reg_alloc =			sizeof(_any_reg_alloc_t),
		},

	/* function pointers in this component */
	.init =					_any_init,
	.init_control_state =	_any_init_control_state,
	.start =				_any_start,
	.stop =					_any_stop,
	.read =					_any_read,
	.write =				_any_write,
	.shutdown =				_any_shutdown,
	.shutdown_global =		_any_shutdown_global,
	.ctl =					_any_ctl,
	.bpt_map_set =			_any_bpt_map_set,
	.bpt_map_avail =		_any_bpt_map_avail,
	.bpt_map_exclusive =	_any_bpt_map_exclusive,
	.bpt_map_shared =		_any_bpt_map_shared,
	.bpt_map_preempt =		_any_bpt_map_preempt,
	.bpt_map_update =		_any_bpt_map_update,
	.allocate_registers =	_any_allocate_registers,
	.update_control_state =	_any_update_control_state,
	.set_domain =			_any_set_domain,
	.reset =				_any_reset,
	.set_overflow =			_any_set_overflow,
	.set_profile =			_any_set_profile,
	.stop_profiling =		_any_stop_profiling,
	.add_prog_event =		_any_add_prog_event,
	.ntv_enum_events =		_any_ntv_enum_events,
	.ntv_name_to_code =		_any_ntv_name_to_code,
	.ntv_code_to_name =		_any_ntv_code_to_name,
	.ntv_code_to_descr =	_any_ntv_code_to_descr,
	.ntv_code_to_bits =		_any_ntv_code_to_bits,
	.ntv_bits_to_info =		_any_ntv_bits_to_info,
	.init_substrate =		_any_init_substrate,
	.dispatch_timer =		_any_dispatch_timer,
	.get_real_usec =		_any_get_real_usec,
	.get_real_cycles =		_any_get_real_cycles,
	.get_virt_cycles =		_any_get_virt_cycles,
	.get_virt_usec =		_any_get_virt_usec,
	.get_memory_info =		_any_get_memory_info,
	.get_dmem_info =		_any_get_dmem_info,

	/* OS dependent local routines */
	.update_shlib_info =	_linux_update_shlib_info,
	.get_system_info =		_linux_get_system_info
};
