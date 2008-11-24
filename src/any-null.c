#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

papi_svector_t _any_null_table[] = {
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
 {NULL, VEC_PAPI_END}
};

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
int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
   int retval = PAPI_OK;

#ifndef PAPI_NO_VECTOR
   retval = _papi_hwi_setup_vector_table( vtable, _any_null_table);
   if ( retval != PAPI_OK ) return(retval);   
#endif

   /* Internal function, doesn't necessarily need to be a function */
   strcpy(_papi_hwi_system_info.hw_info.vendor_string,"Vendor");
   strcpy(_papi_hwi_system_info.hw_info.model_string,"Model");
   _papi_hwi_system_info.hw_info.mhz = 1;
   _papi_hwi_system_info.hw_info.clock_mhz = 1;
   _papi_hwi_system_info.hw_info.ncpu = 1;
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = 1;
   strcpy(_papi_hwi_system_info.sub_info.name,"any-null");              /* Name of the substrate we're using, usually CVS RCS Id */
   strcpy(_papi_hwi_system_info.sub_info.version,"$Revision$");           /* Version of this substrate, usually CVS Revision */
   _papi_hwi_system_info.sub_info.num_cntrs = 4;               /* Number of counters the substrate supports */
   _papi_hwi_system_info.sub_info.num_mpx_cntrs = PAPI_MPX_DEF_DEG; /* Number of counters the substrate (or PAPI) can multiplex */
   _papi_hwi_system_info.sub_info.num_native_events = 0;       

   retval = _papi_hwi_setup_all_presets(preset_map, NULL);

   return(retval);
}

/*
 * This is called whenever a thread is initialized
 */
int _papi_hwd_init(hwd_context_t *ctx)
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

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
   return(PAPI_OK);
}

/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int _papi_hwd_init_control_state(hwd_control_state_t *ptr){
   return PAPI_OK;
}

int _papi_hwd_update_control_state(hwd_control_state_t *ptr, NativeInfo_t *native, int count, hwd_context_t *ctx){
   return(PAPI_OK);
}

int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *ctrl){
   return(PAPI_OK);
}

int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long **events, int flags)
{
   return(PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   return(PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *ctrl)
{
   return(PAPI_OK);
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long *from)
{
   return(PAPI_OK);
}

/*
 * Overflow and profile functions 
 */
void _papi_hwd_dispatch_timer(int signal, hwd_siginfo_t *si, void *context)
{
  /* Real function would call the function below with the proper args
   * _papi_hwi_dispatch_overflow_signal(...);
   */
  return;
}

int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  return(PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, int EventIndex, int threashold)
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
int _papi_hwd_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
  return(PAPI_ENOSUPP);
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
long long _papi_hwd_get_real_usec(void)
{
  long long retval;
  struct timeval buffer;
  gettimeofday(&buffer,NULL);
  retval = (long long)(buffer.tv_sec*1000000);
  retval += (long long)(buffer.tv_usec);
  return(retval);
}

long long _papi_hwd_get_real_cycles(void)
{
  return(_papi_hwd_get_real_usec());
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
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
     retval = (long long)(utime+stime)*1000000/_papi_hwi_system_info.sub_info.clock_ticks;
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
     retval = (long long)(buffer.tms_utime+buffer.tms_stime)*1000000/_papi_hwi_system_info.sub_info.clock_ticks;
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

long long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
  return(_papi_hwd_get_virt_usec(ctx)*_papi_hwi_system_info.hw_info.mhz);
}

/*
 * Native Event functions
 */
int _papi_hwd_add_prog_event(hwd_control_state_t * ctrl, unsigned int EventCode, void *inout, EventInfo_t * EventInfoArray){
  return(PAPI_OK);
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
  return(PAPI_OK);
}

int _papi_hwd_ntv_code_to_name(unsigned int EventCode, char *ntv_name, int len)
{
   strncpy(ntv_name, "PAPI_ANY_NULL", len);
   return (PAPI_OK);
}

int _papi_hwd_ntv_code_to_descr(unsigned int EventCode, char *ntv_descr, int len)
{
   strncpy(ntv_name, "Event doesn't exist, is an example for a skeleton substrate", len);
   return (PAPI_OK);
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
   return(PAPI_OK);
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names, unsigned int *values, int name_len, int count)
{
  const char str[]="Counter: 0  Event: 0";
  if ( count == 0 ) return(0);

  if ( strlen(str) > name_len ) return(0);

  strcpy(names, "Counter: 0  Event: 0");
  return(1);
}

/* 
 * Counter Allocation Functions, only need to implement if
 *    the substrate needs smart counter allocation.
 */

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) 
{
  return(1);
}

/* Forces the event to be mapped to only counter ctr. */
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr) {
}

/* This function examines the event to determine if it can be mapped 
 * to counter ctr.  Returns true if it can, false if it can't. 
 */
int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr) {
   return(1);
} 

/* This function examines the event to determine if it has a single 
 * exclusive mapping.  Returns true if exlusive, false if 
 * non-exclusive.  
 */
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst) {
   return(1);
}

/* This function compares the dst and src events to determine if any 
 * resources are shared. Typically the src event is exclusive, so 
 * this detects a conflict if true. Returns true if conflict, false 
 * if no conflict.  
 */
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
  return(0);
}

/* This function removes shared resources available to the src event
 *  from the resources available to the dst event,
 *  and reduces the rank of the dst event accordingly. Typically,
 *  the src event will be exclusive, but the code shouldn't assume it.
 *  Returns nothing.  
 */
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
  return;
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src) 
{
  return;
}

/*
 * Shared Library Information and other Information Functions
 */
int _papi_hwd_update_shlib_info(void){
  return(PAPI_OK);
}
