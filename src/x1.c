#include <mutex.h>

/*
 * This function has to set the bits needed to count different domains
 * In particular we deal with the following:
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 * This function is called from _papi_hwd_ctl, _papi_hwd_init_control_state,
 * and set_default_domain which are all functions within x1.c
 */

/* Valid values for hwp_enable are as follows: 
 * HWPERF_ENABLE_USER,HWPERF_ENABLE_KERNEL,HWPERF_ENABLE_EXCEPTION
 * For HWPERF_ENABLE_KERNEL and HWPERF_ENABLE_EXCEPTION the user
 * must have PROC_CAP_MGT capability.
 * This only works on P-Chip and not on M-Chip or E-Chip counters
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
 * This calls set_domain to set the default domain being monitored
 */
static int set_default_domain(EventSetInfo_t * zero, int domain)
{
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;
   return (set_domain(current_state, domain));
}

/*
 * This function sets the granularity of 
 * This is called from set_default_granularity, init_config,
 * and _papi_hwd_ctl
 * Currently the X1 does not support granularity so OK is returned
 * If the granularity state is valid.
 */
static int _internal_set_granularity(hwd_control_state_t * this_state, int granularity)
{
   switch (granularity) {
   case PAPI_GRN_THR:
   case PAPI_GRN_PROC:
   case PAPI_GRN_PROCG:
      break;
   default:
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
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

/*
 * This function takes care of setting various features
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
      return (set_granularity
              (option->granularity.ESI->machdep, option->granularity.granularity));
   case PAPI_SET_INHERIT:
      return (set_inherit(option->inherit.inherit));
   default:
      return (PAPI_EINVAL);
   }
}

/*
 * This function should return the highest resolution wallclock timer available
 * in usecs.  The Cray X1 does not have a high resolution timer so we have to
 * use gettimeofday.
 */
u_long_long _papi_hwd_get_real_usec(void)
{
   struct timeval tv;

   gettimeofday(&tv, NULL);
   return ((tv.tv_sec * 1000000) + tv.tv_usec);
}

/*
 * This function should return the highest resolution wallclock timer available
 * in cycles. Since the Cray X1 does not have a high resolution we have to
 * use gettimeofday.
 */
u_long_long _papi_hwd_get_real_cycles(void)
{
   float usec, cyc;

   usec = (float) _papi_hwd_get_real_usec();
   cyc = usec * _papi_system_info.hw_info.mhz;
   return ((long long) cyc);
}

/*
 * This function should return the highest resolution processor timer available
 * in usecs.
 */
u_long_long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   long long retval;
   struct tms buffer;

   times(&buffer);
   retval = (long long) buffer.tms_utime * (long long) (1000000 / sysconf(_SC_CLK_TCK));
   return (retval);
}

/*
 * This function should return the highest resolution processor timer available
 * in cycles.
 */
u_long_long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   float usec, cyc;

   usec = (float) _papi_hwd_get_virt_usec(zero);
   cyc = usec * _papi_system_info.hw_info.mhz;
   return ((long long) cyc);
}

void _papi_hwd_error(int error, char *where)
{
   sprintf(where, "Substrate error: %s", strerror(error));
}


/* Initialize hardware counters and get information, this is called
 * when the PAPI/process is initialized
 */
int _papi_hwd_init_global(void)
{
   int retval;

   /* Fill in what we can of the papi_system_info. */

   retval = _internal_get_system_info();
   if (retval)
      return (retval);

   retval = _papi_get_memory_info(&_papi_system_info.mem_info);
   if (retval)
      return (retval);


   DBG((stderr, "Found %d %s %s CPU's at %f Mhz.\n",
        _papi_system_info.hw_info.totalcpus,
        _papi_system_info.hw_info.vendor_string,
        _papi_system_info.hw_info.model_string, _papi_system_info.hw_info.mhz));

   return (PAPI_OK);
}

static int _internal_get_system_info(void)
{
   int fd, retval;
   pid_t pid;
   char pidstr[PAPI_MAX_STR_LEN];
   prpsinfo_t psi;


   if (scaninvent(scan_cpu_info, NULL) == -1)
      return (PAPI_ESBSTR);

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);

   sprintf(pidstr, "/proc/%05d", (int) pid);
   if ((fd = open(pidstr, O_RDONLY)) == -1)
      return (PAPI_ESYS);

   if (ioctl(fd, PIOCPSINFO, (void *) &psi) == -1)
      return (PAPI_ESYS);

   close(fd);

   /* EXEinfo */

   if (getcwd(_papi_system_info.exe_info.fullname, PAPI_MAX_STR_LEN) == NULL)
      return (PAPI_ESYS);

   _papi_system_info.cpunum = psi.pr_sonproc;
   strcat(_papi_system_info.exe_info.fullname, "/");
   strcat(_papi_system_info.exe_info.fullname, psi.pr_fname);
   strncpy(_papi_system_info.exe_info.name, psi.pr_fname, PAPI_MAX_STR_LEN);

   /* HWinfo */

   _papi_system_info.hw_info.totalcpus = sysmp(MP_NPROCS);
   if (_papi_system_info.hw_info.totalcpus > 3) {
      _papi_system_info.hw_info.ncpu = 4;
      _papi_system_info.hw_info.nnodes = _papi_system_info.hw_info.totalcpus / 16;
   } else {
      _papi_system_info.hw_info.ncpu = 0;
      _papi_system_info.hw_info.nnodes = 0;
   }

  /*_papi_system_info.hw_info.mhz = getmhz();*/
   strcpy(_papi_system_info.hw_info.vendor_string, "Cray");
   _papi_system_info.hw_info.vendor = -1;

   /* Generic info */
   /* Number of counters is 64, 32 P chip, 16 M chip and 16 E chip */
   _papi_system_info.num_cntrs = 64;
   _papi_system_info.cpunum = get_cpu();

   return (PAPI_OK);
}


/*
 * This is called whenever a thread is initialized
 */
int _papi_hwd_init(EventSetInfo_t * global)
{
   return (PAPI_OK);
}


/*
 * Utility Functions
 */

static mutexlock_t lck;

void _papi_hwd_lock_init(void)
{
   init_lock(&lck);
}

void _papi_hwd_lock(void)
{
   int i;
   /* This will always aquire a lock, while acquire_lock is not
    * guaranteed, while spin_lock states:
    * If the lock isnot immediately available, the calling process will either 
    * spin (busywait) or be suspended until the lock becomes available.
    * I will try that first and check the performance and load -KSL
    */
   spin_lock(&lck);
/*
  while (acquire_lock(&lck) != 0)
  {
    DBG((stderr,"Waiting..."));
    nap(1000); 
  }
*/
}

void _papi_hwd_unlock(void)
{
   /* This call uncoditionally unlocks the mutex
    * caller beware
    */
   release_lock(&lck);
}
