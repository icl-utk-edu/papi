/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    freebsd.c
* CVS:     $Id$
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include <sys/types.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include SUBSTRATE
#include "map.h"

#if defined(DEBUG)
# define SHOW_WHERE_I_AM  { fprintf (stderr, "DEBUG: I am at function %s (file: %s, line: %d)\n", __FUNCTION__, __FILE__, __LINE__); }
#else
# define SHOW_WHERE_I_AM
#endif

#if defined(DEBUG)
# define SHOW_COUNTER(id,name) \
	{ \
		unsigned long long tmp_value; \
		int ret = pmc_read (id, &tmp_value); \
		if (ret < 0) \
			fprintf (stderr, "DEBUG: Unable to read counter %s (ID: %08x) on routine %s (file: %s, line: %d)\n", name, id, __FUNCTION__, __FILE__, __LINE__); \
		else \
			fprintf (stderr, "DEBUG: Read counter %s (ID: %08x) - value %llu on routine %s (file: %s, line: %d)\n", name, id, tmp_value, __FUNCTION__, __FILE__, __LINE__); \
	}
#else
# define SHOW_COUNTER(id,name)
#endif

static hwd_libpmc_context_t Context;

int _papi_hwd_set_domain(hwd_control_state_t *cntrl, int domain);

int init_mdi(void);
int init_presets(void);

papi_svector_t _freebsd_svector_table[] = {
 {(void (*)())_papi_hwd_update_shlib_info, VEC_PAPI_HWD_UPDATE_SHLIB_INFO},
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_init_control_state, VEC_PAPI_HWD_INIT_CONTROL_STATE },
 {(void (*)())_papi_hwd_update_control_state,VEC_PAPI_HWD_UPDATE_CONTROL_STATE},
 {(void (*)())_papi_hwd_start, VEC_PAPI_HWD_START },
 {(void (*)())_papi_hwd_stop, VEC_PAPI_HWD_STOP },
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
 {(void (*)())_papi_hwd_shutdown_global, VEC_PAPI_HWD_SHUTDOWN_GLOBAL},
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_write, VEC_PAPI_HWD_WRITE},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 {(void (*)())_papi_hwd_stop_profiling, VEC_PAPI_HWD_STOP_PROFILING},
 {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 {(void (*)())_papi_hwd_set_profile, VEC_PAPI_HWD_SET_PROFILE},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_add_prog_event, VEC_PAPI_HWD_ADD_PROG_EVENT},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {(void (*)())_papi_hwd_bpt_map_set, VEC_PAPI_HWD_BPT_MAP_SET },
 {(void (*)())_papi_hwd_bpt_map_avail, VEC_PAPI_HWD_BPT_MAP_AVAIL },
 {(void (*)())_papi_hwd_bpt_map_exclusive, VEC_PAPI_HWD_BPT_MAP_EXCLUSIVE },
 {(void (*)())_papi_hwd_bpt_map_shared, VEC_PAPI_HWD_BPT_MAP_SHARED },
 {(void (*)())_papi_hwd_bpt_map_preempt, VEC_PAPI_HWD_BPT_MAP_PREEMPT },
 {(void (*)())_papi_hwd_bpt_map_update, VEC_PAPI_HWD_BPT_MAP_UPDATE },
 {(void (*)())_papi_hwd_allocate_registers, VEC_PAPI_HWD_ALLOCATE_REGISTERS },
 {(void (*)())_papi_hwd_set_domain, VEC_PAPI_HWD_SET_DOMAIN},
 {NULL, VEC_PAPI_END}
};

/*
 * Substrate setup and shutdown
 */

/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
   int retval;

   SHOW_WHERE_I_AM;

   retval = _papi_hwi_setup_vector_table( vtable, _freebsd_svector_table);
   
#ifdef DEBUG
   /* This prints out which functions are mapped to dummy routines
    * and this should be taken out once the substrate is completed.
    * The 0 argument will print out only dummy routines, change
    * it to a 1 to print out all routines.
    */
   vector_print_table(vtable, 0);
#endif

   /* Internal function, doesn't necessarily need to be a function */
   init_mdi();

   /* Internal function, doesn't necessarily need to be a function */
   init_presets();

   return retval;
}

int init_presets(void)
{
	const struct pmc_cpuinfo *info;

	SHOW_WHERE_I_AM;

	if (pmc_cpuinfo (&info) != 0)
		return PAPI_ESYS;

	init_freebsd_libpmc_mappings();

	if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_P6") == 0)
		Context.CPUsubstrate = CPU_P6;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_PII") == 0)
		Context.CPUsubstrate = CPU_P6_2;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_PIII") == 0)
		Context.CPUsubstrate = CPU_P6_3;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_CL") == 0)
		Context.CPUsubstrate = CPU_P6_C;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_PM") == 0)
		Context.CPUsubstrate = CPU_P6_M;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "AMD_K7") == 0)
		Context.CPUsubstrate = CPU_K7;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "AMD_K8") == 0)
		Context.CPUsubstrate = CPU_K8;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_PIV") == 0)
		Context.CPUsubstrate = CPU_P4;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_ATOM") == 0)
		Context.CPUsubstrate = CPU_ATOM;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_CORE") == 0)
		Context.CPUsubstrate = CPU_CORE;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_CORE2") == 0)
		Context.CPUsubstrate = CPU_CORE2;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_CORE2EXTREME") == 0)
		Context.CPUsubstrate = CPU_CORE2EXTREME;
	else if (strcmp(pmc_name_of_cputype(info->pm_cputype), "INTEL_COREI7") == 0)
		Context.CPUsubstrate = CPU_COREI7;
	else
		/* Unknown processor! */
		Context.CPUsubstrate = CPU_UNKNOWN;

	_papi_hwi_system_info.sub_info.num_native_events = freebsd_substrate_number_of_events (Context.CPUsubstrate);

	_papi_hwi_setup_all_presets(_papi_hwd_native_info[Context.CPUsubstrate].map, NULL);

	return 0;
}

/*
 * This function is an internal function and not exposed and thus
 * it can be called anything you want as long as the information
 * is setup in _papi_hwd_init_substrate.  Below is some, but not
 * all of the values that will need to be setup.  For a complete
 * list check out papi_mdi_t, though some of the values are setup
 * and used above the substrate level.
 */
int init_mdi(void)
{
	const struct pmc_cpuinfo *info;
   
	SHOW_WHERE_I_AM;

	/* Initialize PMC library */
	if (pmc_init() < 0)
		return PAPI_ESYS;
      
	if (pmc_cpuinfo (&info) != 0)
		return PAPI_ESYS;
   
	if (info != NULL)
	{
		/* Get CPU clock rate from HW.CLOCKRATE sysctl value, and
		   MODEL from HW.MODEL */
		int mib[5];
		size_t len;
		int hw_clockrate;
		char hw_model[PAPI_MAX_STR_LEN];
     
#if !defined(__i386__) && !defined(__amd64__)
		Context.use_rdtsc = FALSE;
#else
		/* Ok, I386s/AMD64s can use RDTSC. But be careful, if the cpufreq
		   module is loaded, then CPU frequency can vary and this method
		   does not work properly! We'll use use_rdtsc to know if this
		   method is available */
		len = 5; 
		Context.use_rdtsc = sysctlnametomib ("dev.cpufreq.0.%driver", mib, &len) == -1;
#endif

		len = 3;
		if (sysctlnametomib ("hw.clockrate", mib, &len) == -1)
			return PAPI_ESYS;
		len = sizeof(hw_clockrate);
		if (sysctl (mib, 2, &hw_clockrate, &len, NULL, 0) == -1)
			return PAPI_ESYS;

		len = 3;
		if (sysctlnametomib ("hw.model", mib, &len) == -1)
			return PAPI_ESYS;
		len = PAPI_MAX_STR_LEN;
		if (sysctl (mib, 2, &hw_model, &len, NULL, 0) == -1)
			return PAPI_ESYS;
		
		/*strcpy (_papi_hwi_system_info.hw_info.vendor_string, pmc_name_of_cputype(info->pm_cputype));*/
		sprintf (_papi_hwi_system_info.hw_info.vendor_string, "%s (TSC:%c)", pmc_name_of_cputype(info->pm_cputype), Context.use_rdtsc?'Y':'N');
		strcpy (_papi_hwi_system_info.hw_info.model_string, hw_model);
		_papi_hwi_system_info.hw_info.mhz = (float) hw_clockrate;
		_papi_hwi_system_info.hw_info.ncpu = info->pm_ncpu;
		_papi_hwi_system_info.hw_info.nnodes = 1;
		_papi_hwi_system_info.hw_info.totalcpus = info->pm_ncpu;
		/* Right now, PMC states that TSC is an additional counter. However
		   it's only available as a system-wide counter and this requires
		   root access */
		_papi_hwi_system_info.sub_info.num_cntrs = info->pm_npmc - 1;
/*
		_papi_hwi_system_info.num_cntrs = MAX_COUNTERS;
		_papi_hwi_system_info.supports_program = 0;
		_papi_hwi_system_info.supports_write = 0;
		_papi_hwi_system_info.supports_hw_overflow = 0;
		_papi_hwi_system_info.supports_hw_profile = 0;
		_papi_hwi_system_info.supports_multiple_threads = 0;
		_papi_hwi_system_info.supports_64bit_counters = 0;
		_papi_hwi_system_info.supports_attach = 0;
		_papi_hwi_system_info.supports_real_usec = 0;
		_papi_hwi_system_info.supports_real_cyc = 0;
		_papi_hwi_system_info.supports_virt_usec = 0;
		_papi_hwi_system_info.supports_virt_cyc = 0;
		_papi_hwi_system_info.size_machdep = sizeof(hwd_control_state_t);
*/
		}
		else
			return PAPI_ESYS;

	return 1;
}


/*
 * This is called whenever a thread is initialized
 */
int _papi_hwd_init(hwd_context_t *ctx)
{
	SHOW_WHERE_I_AM;
	return PAPI_OK;
}

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
	SHOW_WHERE_I_AM;
	return PAPI_OK;
}

int _papi_hwd_shutdown_global(void)
{
	SHOW_WHERE_I_AM;
	return PAPI_OK;
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int _papi_hwd_init_control_state(hwd_control_state_t *ptr)
{
	/* We will default to gather counters in USER|KERNEL mode */
	SHOW_WHERE_I_AM;
	ptr->hwc_domain = PAPI_DOM_USER|PAPI_DOM_KERNEL;
	ptr->pmcs = NULL;
	ptr->counters = NULL;
	ptr->n_counters = 0;
	return PAPI_OK;
}

int _papi_hwd_update_control_state(hwd_control_state_t *ptr, NativeInfo_t *native, int count, hwd_context_t *ctx)
{
	char name[1024];
	int i;
	int res;

	SHOW_WHERE_I_AM;

	/* We're going to store which counters are being used in this EventSet.
	   As this ptr structure can be reused within many PAPI_add_event calls,
	   and domain can change we will reconstruct the table of counters
	   (ptr->counters) everytime where here.
	*/
	if (ptr->counters != NULL && ptr->n_counters > 0)
	{
		for (i = 0; i < ptr->n_counters; i++)
			if (ptr->counters[i] != NULL)
				free (ptr->counters[i]);
		free (ptr->counters);
	}
	if (ptr->pmcs != NULL)
		free (ptr->pmcs);
	if (ptr->values != NULL)
		free (ptr->values);
	if (ptr->caps != NULL)
		free (ptr->caps);

	ptr->n_counters = count;
	ptr->pmcs = (pmc_id_t*) malloc (sizeof(pmc_id_t)*count);
	ptr->caps = (uint32_t*) malloc (sizeof(uint32_t)*count);
	ptr->values = (long_long*) malloc (sizeof(long_long)*count);
	ptr->counters = (char **) malloc (sizeof(char*)*count);
	for (i = 0; i < count; i++)
		ptr->counters[i] = NULL;

	for (i = 0; i < count; i++)
	{
		res = _papi_hwd_ntv_code_to_name (native[i].ni_event, name, sizeof(name));
		if (res != PAPI_OK)
			return res;

		native[i].ni_position = i;

		/* Domains can be applied to canonical events in libpmc (not "generic") */
		if (Context.CPUsubstrate != CPU_UNKNOWN)
		{
			if (ptr->hwc_domain == (PAPI_DOM_USER|PAPI_DOM_KERNEL))
			{
				/* PMC defaults domain to OS & User. So simply copy the name of the counter */
				ptr->counters[i] = strdup (name);
				if (ptr->counters[i] == NULL)
					return PAPI_ESYS;
			}
			else if (ptr->hwc_domain == PAPI_DOM_USER)
			{
				/* This is user-domain case. Just add unitmask=usr */
				ptr->counters[i] = malloc ((strlen(name)+strlen(",usr")+1)*sizeof(char));
				if (ptr->counters[i] == NULL)
					return PAPI_ESYS;
				sprintf (ptr->counters[i], "%s,usr", name);
			}
			else /* if (ptr->hwc_domain == PAPI_DOM_KERNEL) */
			{
				/* This is the last case. Just add unitmask=os */
				ptr->counters[i] = malloc ((strlen(name)+strlen(",os")+1)*sizeof(char));
				if (ptr->counters[i] == NULL)
					return PAPI_ESYS;
				sprintf (ptr->counters[i], "%s,os", name);
			}
		}
		else
		{
			/* PMC defaults domain to OS & User. So simply copy the name of the counter */
			ptr->counters[i] = strdup (name);
			if (ptr->counters[i] == NULL)
				return PAPI_ESYS;
		}
	}

	return PAPI_OK;
}

int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
	int i, ret;

	SHOW_WHERE_I_AM;

	for (i = 0; i < ctrl->n_counters; i++)
	{
		if ((ret = pmc_allocate (ctrl->counters[i], PMC_MODE_TC, 0, PMC_CPU_ANY, &(ctrl->pmcs[i]))) < 0)
		{
#if defined(DEBUG)
			/* This shouldn't happen, it's tested previously on _papi_hwd_allocate_registers */
			fprintf (stderr, "DEBUG: %s FAILED to allocate '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
		if ((ret = pmc_capabilities (ctrl->pmcs[i],&(ctrl->caps[i]))) < 0)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: %s FAILED to get capabilites for '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			ctrl->caps[i] = 0;
		}
#if defined(DEBUG)
		fprintf (stderr, "DEBUG: %s got counter '%s' is %swrittable! [%d of %d]\n", FUNC, ctrl->counters[i], (ctrl->caps[i]&PMC_CAP_WRITE)?"":"NOT", i+1, ctrl->n_counters);
#endif
		if ((ret = pmc_start (ctrl->pmcs[i])) < 0)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: %s FAILED to start '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
	}
	return PAPI_OK;
}

int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long **events, int flags)
{
	int i, ret;

	SHOW_WHERE_I_AM;

	for (i = 0; i < ctrl->n_counters; i++)
		if ((ret = pmc_read (ctrl->pmcs[i], &(ctrl->values[i]))) < 0)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: %s FAILED to read '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
	*events = ctrl->values;

#if defined(DEBUG)
	for (i = 0; i < ctrl->n_counters; i++)
		fprintf (stderr, "DEBUG: %s counter '%s' has value %lld\n", FUNC, ctrl->counters[i], ctrl->values[i]);
#endif
	return PAPI_OK;
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
	int i, ret;

	SHOW_WHERE_I_AM;

	for (i = 0; i < ctrl->n_counters; i++)
	{
		if ((ret = pmc_stop (ctrl->pmcs[i])) < 0)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: %s FAILED to stop '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
		if ((ret = pmc_release (ctrl->pmcs[i])) < 0)
		{
#if defined(DEBUG)
			/* This shouldn't happen, it's tested previously on _papi_hwd_allocate_registers */
			fprintf (stderr, "DEBUG: %s FAILED to release '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
			return PAPI_ESYS;
		}
	}
	return PAPI_OK;
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
	int i, ret;

	SHOW_WHERE_I_AM;

	for (i = 0; i < ctrl->n_counters; i++)
	{
		/* Can we write on the counters? */
		if (ctrl->caps[i] & PMC_CAP_WRITE)
		{
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: _papi_hwd_reset is about to stop the counter %d\n", i+1);
  		SHOW_COUNTER(ctrl->pmcs[i],ctrl->counters[i]);
#endif
			if ((ret = pmc_stop (ctrl->pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to stop '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: _papi_hwd_reset is about to write the counter %d\n", i+1);
  		SHOW_COUNTER(ctrl->pmcs[i],ctrl->counters[i]);
#endif
			if ((ret = pmc_write (ctrl->pmcs[i], 0)) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to write '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: _papi_hwd_reset is about to start the counter %d\n", i+1);
  		SHOW_COUNTER(ctrl->pmcs[i],ctrl->counters[i]);
#endif
			if ((ret = pmc_start (ctrl->pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to start '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
#if defined(DEBUG)
			fprintf (stderr, "DEBUG: _papi_hwd_reset after starting the counter %d\n", i+1);
  		SHOW_COUNTER(ctrl->pmcs[i],ctrl->counters[i]);
#endif
		}
		else
			return PAPI_ESBSTR;
	}
	return PAPI_OK;
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long *from)
{
	int i, ret;

	SHOW_WHERE_I_AM;

	for (i = 0; i < ctrl->n_counters; i++)
	{
		/* Can we write on the counters? */
		if (ctrl->caps[i] & PMC_CAP_WRITE)
		{
			if ((ret = pmc_stop (ctrl->pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to stop '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
			if ((ret = pmc_write (ctrl->pmcs[i], from[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to write '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
			if ((ret = pmc_start (ctrl->pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to stop '%s' [%d of %d] ERROR = %d\n", FUNC, ctrl->counters[i], i+1, ctrl->n_counters, ret);
#endif
				return PAPI_ESYS;
			}
		}
		else
			return PAPI_ESBSTR;
	}
	return PAPI_OK;
}

/*
 * Overflow and profile functions 
 */
void _papi_hwd_dispatch_timer(int signal, hwd_siginfo_t * info, void *context)
{
  /* Real function would call the function below with the proper args
   * _papi_hwi_dispatch_overflow_signal(...);
   */
	SHOW_WHERE_I_AM;
  return;
}

int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
	SHOW_WHERE_I_AM;
  return PAPI_OK;
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
	SHOW_WHERE_I_AM;
  return PAPI_OK;
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, int EventIndex, int threashold)
{
	SHOW_WHERE_I_AM;
  return PAPI_OK;
}

/*
 * Functions for setting up various options
 */

/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int _papi_hwd_ctl (hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
	SHOW_WHERE_I_AM;
	switch (code)
	{
		case PAPI_DOMAIN:
		case PAPI_DEFDOM:
			return _papi_hwd_set_domain(&option->domain.ESI->machdep, option->domain.domain);
		case PAPI_GRANUL:
		case PAPI_DEFGRN:
			return PAPI_ESBSTR;
		default:
			return PAPI_EINVAL;
   }
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
int _papi_hwd_set_domain(hwd_control_state_t *cntrl, int domain) 
{
  int found = 0;

	SHOW_WHERE_I_AM;
	/* libpmc supports USER/KERNEL mode only when counters are native */
	if (Context.CPUsubstrate != CPU_UNKNOWN)
	{
		if (domain & (PAPI_DOM_USER|PAPI_DOM_KERNEL))
		{
			cntrl->hwc_domain = domain & (PAPI_DOM_USER|PAPI_DOM_KERNEL);
			found = 1;
		}
		return found?PAPI_OK:PAPI_EINVAL;
	}
	else
		return PAPI_ESBSTR;
}

/* 
 * Timing Routines
 * These functions should return the highest resolution timers available.
 */
long_long _papi_hwd_get_real_usec(void)
{
	/* Hey, I've seen somewhere a define called __x86_64__! Should I support it? */
#if !defined(__i386__) && !defined(__amd64__)
	/* This will surely work, but with low precision and high overhead */
	struct rusage res;

	SHOW_WHERE_I_AM;
	if ((getrusage(RUSAGE_SELF, &res) == -1))
		return PAPI_ESYS;
	return (res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec;
#else
	SHOW_WHERE_I_AM;
	if (Context.use_rdtsc)
	{
		return _papi_hwd_get_real_cycles() / _papi_hwi_system_info.hw_info.mhz;
	}
	else
	{
		struct rusage res;
		if ((getrusage(RUSAGE_SELF, &res) == -1))
			return PAPI_ESYS;
		return (res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec;
	}
#endif
}

long_long _papi_hwd_get_real_cycles(void)
{
	/* Hey, I've seen somewhere a define called __x86_64__! Should I support it? */
#if !defined(__i386__) && !defined(__amd64__)
	SHOW_WHERE_I_AM;
	/* This will surely work, but with low precision and high overhead */
   return ((long_long) _papi_hwd_get_real_usec() * _papi_hwi_system_info.hw_info.mhz);
#else
	SHOW_WHERE_I_AM;
	if (Context.use_rdtsc)
	{
		long_long cycles;
		__asm __volatile(".byte 0x0f, 0x31" : "=A" (cycles));
	  return cycles;
	}
	else
	{
		return ((long_long) _papi_hwd_get_real_usec() * _papi_hwi_system_info.hw_info.mhz);
	}
#endif
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
	struct rusage res;

	SHOW_WHERE_I_AM;

	if ((getrusage(RUSAGE_SELF, &res) == -1))
		return PAPI_ESYS;
	return (res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec;
}

long_long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
	SHOW_WHERE_I_AM;
	return ((long_long) _papi_hwd_get_virt_usec(ctx) * _papi_hwi_system_info.hw_info.mhz);
}

/*
 * Native Event functions
 */
int _papi_hwd_add_prog_event(hwd_control_state_t * ctrl, unsigned int EventCode, void *inout, EventInfo_t * EventInfoArray){
	SHOW_WHERE_I_AM;
  return PAPI_OK;
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
	int res;
	char name[1024];
	unsigned int nextCode = 1 + *EventCode;

	SHOW_WHERE_I_AM;

	res = _papi_hwd_ntv_code_to_name(nextCode, name, sizeof(name));

	if (res != PAPI_OK)
		return res;
	else
		*EventCode = nextCode;

	return PAPI_OK;
}

int _papi_hwd_ntv_code_to_name(unsigned int EventCode, char *ntv_name, int len)
{
	SHOW_WHERE_I_AM;
	int nidx;

	nidx = EventCode ^ PAPI_NATIVE_MASK;
	if (nidx >= _papi_hwi_system_info.sub_info.num_native_events)
		return PAPI_ENOEVNT;
	strncpy (ntv_name, _papi_hwd_native_info[Context.CPUsubstrate].info[EventCode & PAPI_NATIVE_AND_MASK].name, len);
	if (strlen(_papi_hwd_native_info[Context.CPUsubstrate].info[EventCode & PAPI_NATIVE_AND_MASK].name) > len-1)
		return PAPI_EBUF;
	return PAPI_OK;
}

int _papi_hwd_ntv_code_to_descr(unsigned int EventCode, char *descr, int len)
{
	SHOW_WHERE_I_AM;
	int nidx;

	nidx = EventCode ^ PAPI_NATIVE_MASK;
	if (nidx >= _papi_hwi_system_info.sub_info.num_native_events)
		return PAPI_ENOEVNT;
	strncpy (descr, _papi_hwd_native_info[Context.CPUsubstrate].info[EventCode & PAPI_NATIVE_AND_MASK].description, len);
	if (strlen(_papi_hwd_native_info[Context.CPUsubstrate].info[EventCode & PAPI_NATIVE_AND_MASK].description) > len-1)
		return PAPI_EBUF;
	return PAPI_OK;
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
	SHOW_WHERE_I_AM;
	return PAPI_OK;
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count)
{
  const char str[]="Counter: 0  Event: 0";
	SHOW_WHERE_I_AM;
  if ( count == 0 ) return(0);

  if ( strlen(str) > name_len ) return(0);

  strcpy(names, "Counter: 0  Event: 0");
  return 1;
}

/* 
 * Counter Allocation Functions, only need to implement if
 *    the substrate needs smart counter allocation.
 */

/* Here we'll check if PMC can provide all the counters the user want */
int _papi_hwd_allocate_registers (EventSetInfo_t *ESI) 
{
	char name[1024];
	int failed, allocated_counters, i, j, ret;
	pmc_id_t *pmcs;

	SHOW_WHERE_I_AM;

	failed = 0;
	pmcs = (pmc_id_t*) malloc(sizeof(pmc_id_t)*ESI->NativeCount);
	if (pmcs != NULL)
	{
		allocated_counters = 0;
		/* Check if we can allocate all the counters needed */
		for (i = 0; i < ESI->NativeCount; i++)
		{
			ret = _papi_hwd_ntv_code_to_name (ESI->NativeInfoArray[i].ni_event, name, sizeof(name));
			if (ret != PAPI_OK)
				return ret;
			if ( (ret = pmc_allocate (name, PMC_MODE_TC, 0, PMC_CPU_ANY, &pmcs[i])) < 0)
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s FAILED to allocate '%s' (0x%08x) [%d of %d] ERROR = %d\n", FUNC, name, ESI->NativeInfoArray[i].ni_event, i+1, ESI->NativeCount, ret);
#endif
				failed = 1;
				break;
			}
			else
			{
#if defined(DEBUG)
				fprintf (stderr, "DEBUG: %s SUCCEEDED allocating '%s' (0x%08x) [%d of %d]\n", FUNC, name, ESI->NativeInfoArray[i].ni_event, i+1, ESI->NativeCount);
#endif
				allocated_counters++;
			}
		}
		/* Free the counters */
		for (j = 0; j < allocated_counters; j++)
			pmc_release (pmcs[j]);
		free (pmcs);
	}
	else
		failed = 1;

	return failed?0:1;
}

/* Forces the event to be mapped to only counter ctr. */
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
	SHOW_WHERE_I_AM;
}

/* This function examines the event to determine if it can be mapped 
 * to counter ctr.  Returns true if it can, false if it can't. 
 */
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
	SHOW_WHERE_I_AM;
	return 1;
} 

/* This function examines the event to determine if it has a single 
 * exclusive mapping.  Returns true if exlusive, false if 
 * non-exclusive.  
 */
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
	SHOW_WHERE_I_AM;
	return 1;
}

/* This function compares the dst and src events to determine if any 
 * resources are shared. Typically the src event is exclusive, so 
 * this detects a conflict if true. Returns true if conflict, false 
 * if no conflict.  
 */
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
	SHOW_WHERE_I_AM;
  return 0;
}

/* This function removes shared resources available to the src event
 *  from the resources available to the dst event,
 *  and reduces the rank of the dst event accordingly. Typically,
 *  the src event will be exclusive, but the code shouldn't assume it.
 *  Returns nothing.  
 */
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
	SHOW_WHERE_I_AM;
  return;
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
	SHOW_WHERE_I_AM;
  return;
}

/*
 * Shared Library Information and other Information Functions
 */
int _papi_hwd_update_shlib_info(void){
	SHOW_WHERE_I_AM;
  return PAPI_OK;
}
