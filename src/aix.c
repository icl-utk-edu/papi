/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* This file handles the OS dependent part of the POWER3 and POWER4 architectures.
  It supports both AIX 4 and AIX 5. The switch between AIX 4 and 5 is driven by the 
  system defined value _AIX_VERSION_510.
  Other routines also include minor conditionally compiled differences.
*/

#include "papi.h"
#include SUBSTRATE
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"

/* Machine dependent info structure */
extern papi_mdi_t _papi_hwi_system_info;


/* 
 some heap information, start_of_text, start_of_data .....
 ref: http://publibn.boulder.ibm.com/doc_link/en_US/a_doc_lib/aixprggd/genprogc/sys_mem_alloc.htm#HDRA9E4A4C9921SYLV 
*/
#ifndef _P64
  #define START_OF_TEXT 0x10000000
  #define END_OF_TEXT   &_etext
  #define START_OF_DATA 0x20000000
  #define END_OF_DATA   &_end
#else
  #define START_OF_TEXT 0x100000000
  #define END_OF_TEXT   &_etext
  #define START_OF_DATA 0x110000000
  #define END_OF_DATA   &_end
#endif

static int maxgroups = 0;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/* The following is for any POWER hardware */

/* Routines to support an opaque native event table */
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
  return(native_name_map[EventCode & NATIVE_AND_MASK].name);
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
  return(native_table[native_name_map[EventCode & NATIVE_AND_MASK].index].description);
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t *bits)
{
  bits = &native_table[EventCode & NATIVE_AND_MASK].resources; /* it is not right, different type */
  return(PAPI_OK);
}

static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  ptr->counter_cmd.events[arg1] = arg2;
}

static void unset_config(hwd_control_state_t *ptr, int arg1)
{
  ptr->counter_cmd.events[arg1] = 0;
}

int update_global_hwcounters(EventSetInfo_t *global)
{
  int i, retval;
  pm_data_t data;

  retval = pm_get_data_mythread(&data);
  if (retval > 0)
    return(retval);

  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    {
#if 0
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+data.accu[i],global->hw_start[i],data.accu[i]));
#endif
      global->hw_start[i] = global->hw_start[i] + data.accu[i];
    }

  retval = pm_reset_data_mythread();
  if (retval > 0)
    return(retval);
   
  return(0);
}

static int correct_local_hwcounters(EventSetInfo_t *global, EventSetInfo_t *local, long long *correct)
{
  int i;

  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    {
#if 0
      DBG((stderr,"correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n",i,
	   global->hw_start[i]-local->hw_start[i],global->hw_start[i],local->hw_start[i]));
#endif
      correct[i] = global->hw_start[i] - local->hw_start[i];
    }

  return(0);
}

int set_domain(hwd_control_state_t *this_state, int domain)
{
  pm_mode_t *mode = &(this_state->counter_cmd.mode);

  switch (domain)
    {
    case PAPI_DOM_USER:
      mode->b.user = 1;
      mode->b.kernel = 0;
      break;
    case PAPI_DOM_KERNEL:
      mode->b.user = 0;
      mode->b.kernel = 1;
      break;
    case PAPI_DOM_ALL:
      mode->b.user = 1;
      mode->b.kernel = 1;
      break;
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

int set_granularity(hwd_control_state_t *this_state, int domain)
{
  pm_mode_t *mode = &(this_state->counter_cmd.mode);

  switch (domain)
    {
    case PAPI_GRN_THR:
      mode->b.process = 0;
      mode->b.proctree = 0;
      break;
    /* case PAPI_GRN_PROC:
      mode->b.process = 1;
      mode->b.proctree = 0;
      break;
    case PAPI_GRN_PROCG:
      mode->b.process = 0;
      mode->b.proctree = 1;
      break; */
    default:
      return(PAPI_EINVAL);
    }
  return(PAPI_OK);
}

static int set_default_domain(EventSetInfo_t *zero, int domain)
{
  hwd_control_state_t *current_state = &zero->machdep;
  return(set_domain(current_state,domain));
}

static int set_default_granularity(EventSetInfo_t *zero, int granularity)
{
  hwd_control_state_t *current_state = &zero->machdep;
  return(set_granularity(current_state,granularity));
}

static int set_inherit(int arg)
{
  return(PAPI_ESBSTR);
}

/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
int _papi_hwd_mdi_init() {
   strcpy(_papi_hwi_system_info.substrate, "$Id$");     /* Name of the substrate we're using */

   _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t)START_OF_TEXT;
   _papi_hwi_system_info.exe_info.address_info.text_end   = (caddr_t)END_OF_TEXT;
   _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t)START_OF_DATA;
   _papi_hwi_system_info.exe_info.address_info.data_end   = (caddr_t)END_OF_DATA;
   _papi_hwi_system_info.exe_info.address_info.bss_start  = (caddr_t)NULL;
   _papi_hwi_system_info.exe_info.address_info.bss_end    = (caddr_t)NULL;

   _papi_hwi_system_info.supports_64bit_counters        = 1;
   _papi_hwi_system_info.supports_real_usec             = 1;
   _papi_hwi_system_info.supports_real_cyc              = 1;

   _papi_hwi_system_info.shlib_info.map->text_start      = (caddr_t)START_OF_TEXT;
   _papi_hwi_system_info.shlib_info.map->text_end        = (caddr_t)END_OF_TEXT;
   _papi_hwi_system_info.shlib_info.map->data_start      = (caddr_t)START_OF_DATA;
   _papi_hwi_system_info.shlib_info.map->data_end        = (caddr_t)END_OF_DATA;
   _papi_hwi_system_info.shlib_info.map->bss_start       = (caddr_t)NULL;
   _papi_hwi_system_info.shlib_info.map->bss_end         = (caddr_t)NULL;

   return(PAPI_OK);
}


static int get_system_info(void)
{
  int retval;
 /* pm_info_t pminfo;*/
  struct procsinfo psi = { 0 };
  pid_t pid;
  char maxargs[PAPI_MAX_STR_LEN];

#ifndef _POWER4
#ifdef _AIXVERSION_510
  pm_groups_info_t pmgroups;
#endif
#endif

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);
  psi.pi_pid = pid;
  retval = getargs(&psi,sizeof(psi),maxargs,PAPI_MAX_STR_LEN);
  if (retval == -1)
    return(PAPI_ESYS);
  if (getcwd(_papi_hwi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == NULL)
    return(PAPI_ESYS);
  strcat(_papi_hwi_system_info.exe_info.fullname,"/");
  strcat(_papi_hwi_system_info.exe_info.fullname,maxargs);
  strncpy(_papi_hwi_system_info.exe_info.name,basename(maxargs),PAPI_MAX_STR_LEN);

#ifdef _AIXVERSION_510
  DBG((stderr,"Calling AIX 5 version of pm_init...\n"));
  retval = pm_init(PM_INIT_FLAGS, &pminfo, &pmgroups);
#else
  DBG((stderr,"Calling AIX 4 version of pm_init...\n"));
  retval = pm_init(PM_INIT_FLAGS,&pminfo);
#endif
  DBG((stderr,"...Back from pm_init\n"));

  if (retval > 0)
    return(retval);

  strcpy(_papi_hwi_system_info.substrate, "$Id$");     /* Name of the substrate we're using */

  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t)START_OF_TEXT;
  _papi_hwi_system_info.exe_info.address_info.text_end   = (caddr_t)END_OF_TEXT;
  _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t)START_OF_DATA;
  _papi_hwi_system_info.exe_info.address_info.data_end   = (caddr_t)END_OF_DATA;
  _papi_hwi_system_info.exe_info.address_info.bss_start  = (caddr_t)NULL;
  _papi_hwi_system_info.exe_info.address_info.bss_end    = (caddr_t)NULL;

  _papi_hwi_system_info.supports_64bit_counters        = 1;
  _papi_hwi_system_info.supports_real_usec             = 1;
  _papi_hwi_system_info.supports_real_cyc              = 1;

  _papi_hwi_system_info.shlib_info.map->text_start      = (caddr_t)START_OF_TEXT;
  _papi_hwi_system_info.shlib_info.map->text_end        = (caddr_t)END_OF_TEXT;
  _papi_hwi_system_info.shlib_info.map->data_start      = (caddr_t)START_OF_DATA;
  _papi_hwi_system_info.shlib_info.map->data_end        = (caddr_t)END_OF_DATA;
  _papi_hwi_system_info.shlib_info.map->bss_start       = (caddr_t)NULL;
  _papi_hwi_system_info.shlib_info.map->bss_end         = (caddr_t)NULL;

  _papi_hwi_system_info.hw_info.ncpu = _system_configuration.ncpus;
  _papi_hwi_system_info.hw_info.totalcpus = 
  _papi_hwi_system_info.hw_info.ncpu * _papi_hwi_system_info.hw_info.nnodes;
  _papi_hwi_system_info.hw_info.vendor = -1;
  strcpy(_papi_hwi_system_info.hw_info.vendor_string,"IBM");
  _papi_hwi_system_info.hw_info.model = _system_configuration.implementation;
  strcpy(_papi_hwi_system_info.hw_info.model_string,pminfo.proc_name);
  _papi_hwi_system_info.hw_info.revision = (float)_system_configuration.version;
  _papi_hwi_system_info.hw_info.mhz = (float)(pm_cycles() / 1000000.0);
  _papi_hwi_system_info.num_gp_cntrs = pminfo.maxpmcs;
  _papi_hwi_system_info.num_cntrs = pminfo.maxpmcs;

/* This field doesn't appear to exist in the PAPI 3.0 structure 
  _papi_hwi_system_info.cpunum = mycpu(); 
*/

  return(PAPI_OK);
} 

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

u_long_long _papi_hwd_get_real_usec (void)
{
  timebasestruct_t t;
  u_long_long retval;

  read_real_time(&t,TIMEBASE_SZ);
  time_base_to_time(&t,TIMEBASE_SZ);
  retval = (t.tb_high * 1000000) + t.tb_low / 1000;
  return(retval);
}

u_long_long _papi_hwd_get_real_cycles (void)
{
  u_long_long usec, cyc;

  usec = _papi_hwd_get_real_usec();
  cyc = usec * _papi_hwi_system_info.hw_info.mhz;
  return((u_long_long)cyc);
}

u_long_long _papi_hwd_get_virt_usec (const hwd_context_t *context)
{
  u_long_long retval;
  struct tms buffer;

  times(&buffer);
  retval = (u_long_long)buffer.tms_utime*(u_long_long)(1000000/CLK_TCK);
  return(retval);
}

u_long_long _papi_hwd_get_virt_cycles (const hwd_context_t *context)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_virt_usec(context);
  cyc = usec * _papi_hwi_system_info.hw_info.mhz;
  return((u_long_long)cyc);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error");
  pm_error(where,error);
}

int _papi_hwd_init_global(void)
{
  int retval;

  /* Fill in what we can of the papi_system_info. */
  
  retval = get_system_info();
  if (retval)
    return(retval);
  
  retval = get_memory_info(&_papi_hwi_system_info.mem_info);
  if (retval)
    return(retval);

  DBG((stderr,"Found %d %s %s CPU's at %f Mhz.\n",
       _papi_hwi_system_info.hw_info.totalcpus,
       _papi_hwi_system_info.hw_info.vendor_string,
       _papi_hwi_system_info.hw_info.model_string,
       _papi_hwi_system_info.hw_info.mhz));

  return(PAPI_OK);
}

int _papi_hwd_init(hwd_context_t *context)
{
  int retval;
	/* Initialize our global machdep. */

  _papi_hwd_init_control_state(&context->cntrl);
  retval=setup_native_table();
  if(!_papi_hwd_init_preset_search_map(&pminfo))
	  retval=PAPI_ESBSTR;
  retval = _papi_hwi_setup_all_presets(preset_search_map);

  return(retval);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (POWER_MAX_COUNTERS-1);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, int *to)
{
  int useme, i;
  
  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  to[i] = from[i];
	}
    }
}
  

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)

{
  return(PAPI_ESBSTR);
}

#if 1
void dump_cmd(pm_prog_t *t)
{
  fprintf(stderr,"mode.b.threshold %d\n",t->mode.b.threshold);
  fprintf(stderr,"mode.b.spare %d\n",t->mode.b.spare);
  fprintf(stderr,"mode.b.process %d\n",t->mode.b.process);
  fprintf(stderr,"mode.b.kernel %d\n",t->mode.b.kernel);
  fprintf(stderr,"mode.b.user %d\n",t->mode.b.user);
  fprintf(stderr,"mode.b.count %d\n",t->mode.b.count);
  fprintf(stderr,"mode.b.proctree %d\n",t->mode.b.proctree);
  fprintf(stderr,"events[0] %d\n",t->events[0]);
  fprintf(stderr,"events[1] %d\n",t->events[1]);
  fprintf(stderr,"events[2] %d\n",t->events[2]);
  fprintf(stderr,"events[3] %d\n",t->events[3]);
  fprintf(stderr,"events[4] %d\n",t->events[4]);
  fprintf(stderr,"events[5] %d\n",t->events[5]);
  fprintf(stderr,"events[6] %d\n",t->events[6]);
  fprintf(stderr,"events[7] %d\n",t->events[7]);
  fprintf(stderr,"reserved %d\n",t->reserved);
}

void dump_data(pm_data_t *d)
{
  int i;

  for (i=0;i<MAX_COUNTERS;i++) {
    fprintf(stderr,"accu[%d] = %lld\n", i, d->accu[i]);
  }
}
#endif

/*int _papi_hwd_reset(EventSetInfo_t *ESI, EventSetInfo_t *zero)*/
int _papi_hwd_reset(hwd_context_t *ESI, hwd_control_state_t *zero)
{
  int i, retval;

/* I think this doesn't need to be done anymore...
  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);
*/
/* I think this is now done at the hwi level...
  for (i=0;i<_papi_hwi_system_info.num_cntrs;i++)
    ESI->hw_start[i] = zero->hw_start[i];
*/
  return(PAPI_OK);
}


int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *cntrl, long_long **val)
{
  int retval;
  int i;
  static pm_data_t data;

  retval = pm_get_data_mythread(&data);
  if (retval > 0)
    return(retval);

#if 0
  dump_data(&data);
#endif

  *val = data.accu;

  return(PAPI_OK);
}

int _papi_hwd_setmaxmem(){
  return(PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
  switch (code)
    {
/* I don't understand what it means to set the default domain 
    case PAPI_SET_DEFDOM:
      return(set_default_domain(zero, option->domain.domain));
*/
    case PAPI_SET_DOMAIN:
      return(set_domain(&(option->domain.ESI->machdep), option->domain.domain));
/* I don't understand what it means to set the default granularity 
    case PAPI_SET_DEFGRN:
      return(set_default_granularity(zero, option->granularity.granularity));
*/
    case PAPI_SET_GRANUL:
      return(set_granularity(&(option->granularity.ESI->machdep), option->granularity.granularity));
#if 0
    case PAPI_SET_INHERIT:
      return(set_inherit(option->inherit.inherit));
#endif
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *cntrl, long_long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
  pm_delete_program_mythread();
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
  return(PAPI_OK);
}


void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, void *i)
{
#ifdef DEBUG
  ucontext_t *info;
  info = (ucontext_t *)i;
  DBG((stderr,"_papi_hwd_dispatch_timer() at 0x%lx\n",info->uc_mcontext.jmp_context.iar));
#endif

  _papi_hwi_dispatch_overflow_signal(i); 
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, EventSetOverflowInfo_t *overflow_option)
{
  hwd_control_state_t *this_state = &ESI->machdep;

  if (overflow_option->threshold == 0)
    {
      this_state->timer_ms = 0;
      overflow_option->timer_ms = 0;
    }
  else
    {
      this_state->timer_ms = 1; /* Millisecond intervals are the only way to go */
      overflow_option->timer_ms = 1;
    }

  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_jmpbuf.jmp_context.iar;

  return(location);
}

static atomic_p lock[PAPI_MAX_LOCK];

void _papi_hwd_lock_init(void)
{
}

#define _papi_hwd_lock(lck)			\
while(_check_lock(&lock[lck],0,1 == TRUE)	\
{						\
      usleep(1000);				\
}

#define _papi_hwd_unlock(lck)			\
do						\
{						\
  _clear_lock(&lock[lck], 0);			\
}


/* Copy the current control_state into the new thread context */
/*int _papi_hwd_start(EventSetInfo_t *ESI, EventSetInfo_t *zero)*/
int _papi_hwd_start(hwd_context_t *ctx, hwd_control_state_t *cntrl)
{ 
  int i, retval;
  hwd_control_state_t *current_state = &ctx->cntrl;
  
  /* If we are nested, merge the global counter structure
     with the current eventset */

#if 1
DBG((stderr, "Start\n"));
/*dump_state(cntrl);
dump_state(current_state);*/
#endif
  
      /* Copy the global counter structure to the current eventset */
      DBG((stderr,"Copying states\n"));
      memcpy(current_state,cntrl,sizeof(hwd_control_state_t));

      retval = pm_set_program_mythread(&current_state->counter_cmd);
      if (retval > 0) 
        return(retval);

  /* Set up the new merged control structure */
  
#if 0
/*  dump_state(cntrl);
  dump_state(current_state);*/
  dump_cmd(&current_state->counter_cmd);
#endif
      
  /* Start the counters */
  
  retval = pm_start_mythread();
  if (retval > 0) 
    return(retval);

  return(PAPI_OK);
} 

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *cntrl)
{ 
  int retval;

  retval = pm_stop_mythread();
  if (retval > 0) 
    return(retval);

  retval = pm_delete_program_mythread();
  if (retval > 0) 
    return(retval);

  return(PAPI_OK);
}

/*#if 0
void dump_state(hwd_control_state_t *s)
{
  fprintf(stderr,"master_selector %x\n",s->master_selector);
  fprintf(stderr,"event_codes %x %x %x %x %x %x %x %x\n",s->preset[0],s->preset[1],
    s->preset[2],s->preset[3],s->preset[4],s->preset[5],s->preset[6],s->preset[7]);
  fprintf(stderr,"event_selectors %x %x %x %x %x %x %x %x\n",s->selector[0],s->selector[1],
    s->selector[2],s->selector[3],s->selector[4],s->selector[5],s->selector[6],s->selector[7]);
  fprintf(stderr,"counters %x %x %x %x %x %x %x %x\n",s->counter_cmd.events[0],
    s->counter_cmd.events[1],s->counter_cmd.events[2],s->counter_cmd.events[3],
    s->counter_cmd.events[4],s->counter_cmd.events[5],s->counter_cmd.events[6],
    s->counter_cmd.events[7]);
}
#endif
*/

