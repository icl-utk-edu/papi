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
static int _internal_set_domain(hwd_control_state_t * this_state, int domain)
{
   int found = 0;
   if (PAPI_DOM_USER & domain) {
      found = 1;
   }
   if (PAPI_DOM_KERNEL & domain) {
      found = 1;
   }
   if (PAPI_DOM_OTHER & domain) {
      found = 1;
   }
   if (!found)
      return (PAPI_EINVAL);
   return (PAPI_OK);
}

/*
 * This function has to set the bits needed to count different granularities
 * In particular PAPI_GRN_THR, PAPI_GRN_PROC, PAPI_GRN_PROCG, PAPI_GRN_SYS
 * and PAPI_GRN_SYS_CPU
 *
 * PAPI_GRN_THR are PAPI counters for each individual thread
 * PAPI_GRN_PROC are PAPI counters for each individual process
 * PAPI_GRN_PROCG are PAPI counters for each individual process group
 * PAPI_GRN_SYS are PAPI counters for the current CPU
 * PAPI_GRN_SYS_CPU are PAPI counters for all CPU's individually
 *
 * If the function works return PAPI_OK, if one of the granularities is
 * not chosen then return PAPI_EINVAL
 */
static int _internal_set_granularity(hwd_control_state_t * this_state, int granularity)
{
   switch (granularity) {
   case PAPI_GRN_THR:
      break;
   case PAPI_GRN_PROC:
      break;
   case PAPI_GRN_PROCG:
      break;
   default:
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
}

/* 
 * This calls set_domain to set the default domain being monitored
 */
static int set_default_domain(EventSetInfo_t * zero, int domain)
{
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;
   return (set_domain(current_state, domain));
}

/*
 * This calls set_granularity to set the default granularity being monitored 
 */
static int set_default_granularity(hwd_control_state_t * current_state, int granularity)
{
   return (set_granularity(current_state, granularity));
}

/* This function should tell your kernel extension that your children
 * inherit performance register information and propagate the values up
 * upon child exit and parent wait. 
 */

static int set_inherit(EventSetInfo_t * global, int arg)
{
   return (PAPI_ESBSTR);
}

/* This function sets various options in the substrate
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL
 * and PAPI_SET_INHERIT
 */
int _papi_hwd_ctl(EventSetInfo_t * zero, int code, _papi_int_option_t * option)
{
   switch (code) {
   case PAPI_SET_DEFDOM:
      return (set_default_domain(zero, option->domain.domain));
   case PAPI_SET_DOMAIN:
      return (set_domain(option->domain.ESI->machdep, option->domain.domain));
   case PAPI_SET_DEFGRN:
      return (set_default_granularity(zero, option->granularity.granularity));
   case PAPI_SET_GRANUL:
      return (set_granularity(option->granularity.ESI->machdep, option->granularity.g
                              ranularity));
   case PAPI_SET_INHERIT:
      return (set_inherit(option->inherit.inherit));
   default:
      return (PAPI_EINVAL);
   }
}

/*
 * This function should return the highest resolution wallclock timer available
 * in usecs.
 */
long long _papi_hwd_get_real_usec(void)
{
   struct timeval tv;

   gettimeofday(&tv, NULL);
   return ((tv.tv_sec * 1000000) + tv.tv_usec);
}

/*
 * This function should return the highest resolution wallclock timer available
 * in cycles
 */
long long _papi_hwd_get_real_cycles(void)
{
   return (_papi_hwd_get_real_usec() * (long long) _papi_system_info.hw_info.mhz);
}

/*
 * This function should return the highest resolution processor timer available
 * in usecs.
 */
long long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   long long retval;
   struct tms buffer;

   times(&buffer);
   SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime);
   retval = (long long)((buffer.tms_utime+buffer.tms_stime)*
     (1000000/sysconf(_SC_CLK_TCK)));
   return (retval);
}

/*
 * This function should return the highest resolution processor timer available
 * in cycles.
 */
long long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   return (_papi_hwd_get_virt_usec(zero) * (long long)_papi_system_info.hw_info.mhz);
}


/* Initialize hardware counters and get hardware information, this
 * routine is called when the PAPI process is initialized
 * (IE PAPI_library_init)
 */
int _papi_hwd_init_global(void)
{
   int retval;

   /* Fill in what we can of the papi_system_info.
    * This doesn't need to be called but sometimes
    * it is nice to seperate it out.
    */
   retval = _internal_get_system_info();
   if (retval)
      return (retval);

   /* This is usually implemented in an OS specific file
    * but can be implemented here if need be
    */
   retval = get_memory_info(&_papi_system_info.mem_info);
   if (retval)
      return (retval);

   DBG((stderr, "Found %d %s %s CPU's at %f Mhz.\n",
        _papi_system_info.hw_info.totalcpus,
        _papi_system_info.hw_info.vendor_string,
        _papi_system_info.hw_info.model_string, _papi_system_info.hw_info.mhz));

   return (PAPI_OK);
}


/*
 * This is called whenever a thread is initialized 
 */
int _papi_hwd_init()
{
}


/*
 * Utility Functions 
 */


/*
 * This initializes any variables that or anything else
 * that needs to be done to call hwd_lock and hwd_unlock
 */
static void _papi_hwd_lock_init(void)
{
}

/*
 * This locks the mutex variable
 */
void _papi_hwd_unlock(void)
{
}

/*
 * This unlocks the mutex variable
 */
void _papi_hwd_lock(void)
{
}


