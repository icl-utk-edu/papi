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

/* Locking variables */
volatile int lock_var[PAPI_MAX_LOCK] = { 0 };
atomic_p lock[PAPI_MAX_LOCK];

/* 
 some heap information, start_of_text, start_of_data .....
 ref: http://publibn.boulder.ibm.com/doc_link/en_US/a_doc_lib/aixprggd/genprogc/sys_mem_alloc.htm#HDRA9E4A4C9921SYLV 
*/
#define START_OF_TEXT &_text
#define END_OF_TEXT   &_etext
#define START_OF_DATA &_data
#define END_OF_DATA   &_edata
#define START_OF_BSS  &_edata
#define END_OF_BSS    &_end

static int maxgroups = 0;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        /* The following is for any POWER hardware */

/* Routines to support an opaque native event table */
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   return (native_name_map[EventCode & NATIVE_AND_MASK].name);
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (native_table[native_name_map[EventCode & NATIVE_AND_MASK].index].description);
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   bits = &native_table[EventCode & NATIVE_AND_MASK].resources; /* it is not right, different type */
   return (PAPI_OK);
}

/* this function would return the next native event code.
    modifer=0    it simply returns next native event code
    modifer=1    it would return information of groups this native event lives
                 0x400000ed is the native code of PM_FXLS_FULL_CYC,
		 before it returns 0x400000ee which is the next native event's
		 code, it would return *EventCode=0x400400ed, the digits 16-23
		 indicate group number
   function return value:
     PAPI_OK successful, next event is valid
     PAPI_ENOEVNT  fail, next event is invalid
*/
int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer)
{
   if (modifer == 0) {
      int index = *EventCode & NATIVE_AND_MASK;

      if (native_table[index + 1].resources.selector) {
         *EventCode = *EventCode + 1;
         return (PAPI_OK);
      } else
         return (PAPI_ENOEVNT);
   } else {
#ifdef _POWER4
      unsigned int group = (*EventCode & 0x00FF0000) >> 16;
      int index = *EventCode & 0x000000FF;
      int i;
      unsigned int tmpg;

      *EventCode = *EventCode & 0xFF00FFFF;
      for (i = 0; i < GROUP_INTS; i++) {
         tmpg = native_table[index].resources.group[i];
         if (group != 0) {
            while ((ffs(tmpg) + i * 32) <= group && tmpg != 0)
               tmpg = tmpg ^ (1 << (ffs(tmpg) - 1));
         }
         if (tmpg != 0) {
            group = ffs(tmpg) + i * 32;
            *EventCode = *EventCode | (group << 16);
            return (PAPI_OK);
         }
      }
      if (native_table[index + 1].resources.selector == 0)
         return (PAPI_ENOEVNT);
      *EventCode = *EventCode + 1;
      return (PAPI_OK);
#else
      return (PAPI_EINVAL);
#endif
   }
}

static void set_config(hwd_control_state_t * ptr, int arg1, int arg2)
{
   ptr->counter_cmd.events[arg1] = arg2;
}

static void unset_config(hwd_control_state_t * ptr, int arg1)
{
   ptr->counter_cmd.events[arg1] = 0;
}

int update_global_hwcounters(EventSetInfo_t * global)
{
   int i, retval;
   pm_data_t data;

   retval = pm_get_data_mythread(&data);
   if (retval > 0)
      return (retval);

   for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
#if 0
      DBG((stderr, "update_global_hwcounters() %d: G%lld = G%lld + C%lld\n", i,
           global->hw_start[i] + data.accu[i], global->hw_start[i], data.accu[i]));
#endif
      global->hw_start[i] = global->hw_start[i] + data.accu[i];
   }

   retval = pm_reset_data_mythread();
   if (retval > 0)
      return (retval);

   return (0);
}

static int correct_local_hwcounters(EventSetInfo_t * global, EventSetInfo_t * local,
                                    long long *correct)
{
   int i;

   for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
#if 0
      DBG((stderr, "correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n", i,
           global->hw_start[i] - local->hw_start[i], global->hw_start[i],
           local->hw_start[i]));
#endif
      correct[i] = global->hw_start[i] - local->hw_start[i];
   }

   return (0);
}

int set_domain(hwd_control_state_t * this_state, int domain)
{
   pm_mode_t *mode = &(this_state->counter_cmd.mode);
   int did = 0;

   mode->b.user = 0;
   mode->b.kernel = 0;
   if (domain & PAPI_DOM_USER) {
      did = 1;
      mode->b.user = 1;
   }
   if (domain & PAPI_DOM_KERNEL) {
      did = 1;
      mode->b.kernel = 1;
   }
   if (did)
      return (PAPI_OK);
   else
      return (PAPI_EINVAL);
/*
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
*/
}

int set_granularity(hwd_control_state_t * this_state, int domain)
{
   pm_mode_t *mode = &(this_state->counter_cmd.mode);

   switch (domain) {
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
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
}

static int set_default_domain(EventSetInfo_t * zero, int domain)
{
   hwd_control_state_t *current_state = &zero->machdep;
   return (set_domain(current_state, domain));
}

static int set_default_granularity(EventSetInfo_t * zero, int granularity)
{
   hwd_control_state_t *current_state = &zero->machdep;
   return (set_granularity(current_state, granularity));
}

static int set_inherit(int arg)
{
   return (PAPI_ESBSTR);
}

/* Initialize the system-specific settings */
/* Machine info structure. -1 is unused. */
int _papi_hwd_mdi_init()
{
   strcpy(_papi_hwi_system_info.substrate, "$Id$");  /* Name of the substrate we're using */

   _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) START_OF_TEXT;
   _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t) END_OF_TEXT;
   _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) START_OF_DATA;
   _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) END_OF_DATA;
   _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) START_OF_BSS;
   _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) END_OF_BSS;

   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;

   _papi_hwi_system_info.shlib_info.map = &(_papi_hwi_system_info.exe_info.address_info);

   return (PAPI_OK);
}


static int get_system_info(void)
{
   int retval;
   /* pm_info_t pminfo; */
   struct procsinfo psi = { 0 };
   pid_t pid;
   char maxargs[PATH_MAX];
   char pname[PATH_MAX];

#ifndef _POWER4
#ifdef _AIXVERSION_510
   pm_groups_info_t pmgroups;
#endif
#endif

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);
   psi.pi_pid = pid;
   retval = getargs(&psi, sizeof(psi), maxargs, PATH_MAX);
   if (retval == -1)
      return (PAPI_ESYS);

   if (realpath(maxargs,pname))
     strncpy(_papi_hwi_system_info.exe_info.fullname, pname, PAPI_HUGE_STR_LEN);
   else
     strncpy(_papi_hwi_system_info.exe_info.fullname, maxargs, PAPI_HUGE_STR_LEN);

   strcpy(_papi_hwi_system_info.exe_info.address_info.name,basename(maxargs));

#ifdef _AIXVERSION_510
   DBG((stderr, "Calling AIX 5 version of pm_init...\n"));
   retval = pm_init(PM_INIT_FLAGS, &pminfo, &pmgroups);
#else
   DBG((stderr, "Calling AIX 4 version of pm_init...\n"));
   retval = pm_init(PM_INIT_FLAGS, &pminfo);
#endif
   DBG((stderr, "...Back from pm_init\n"));

   if (retval > 0)
      return (retval);

   strcpy(_papi_hwi_system_info.substrate, "$Id$");  /* Name of the substrate we're using */

   _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) START_OF_TEXT;
   _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t) END_OF_TEXT;
   _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) START_OF_DATA;
   _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) END_OF_DATA;
   _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) START_OF_BSS;
   _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) END_OF_BSS;

   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;

   _papi_hwi_system_info.shlib_info.map = &(_papi_hwi_system_info.exe_info.address_info);

   _papi_hwi_system_info.hw_info.ncpu = _system_configuration.ncpus;
   _papi_hwi_system_info.hw_info.totalcpus =
       _papi_hwi_system_info.hw_info.ncpu * _papi_hwi_system_info.hw_info.nnodes;
   _papi_hwi_system_info.hw_info.vendor = -1;
   strcpy(_papi_hwi_system_info.hw_info.vendor_string, "IBM");
   _papi_hwi_system_info.hw_info.model = _system_configuration.implementation;
   strcpy(_papi_hwi_system_info.hw_info.model_string, pminfo.proc_name);
   _papi_hwi_system_info.hw_info.revision = (float) _system_configuration.version;
   _papi_hwi_system_info.hw_info.mhz = (float) (pm_cycles() / 1000000.0);
   _papi_hwi_system_info.num_gp_cntrs = pminfo.maxpmcs;
   _papi_hwi_system_info.num_cntrs = pminfo.maxpmcs;

/* This field doesn't appear to exist in the PAPI 3.0 structure 
  _papi_hwi_system_info.cpunum = mycpu(); 
*/

   return (PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

u_long_long _papi_hwd_get_real_usec(void)
{
   timebasestruct_t t;
   u_long_long retval;

   read_real_time(&t, TIMEBASE_SZ);
   time_base_to_time(&t, TIMEBASE_SZ);
   retval = (t.tb_high * 1000000) + t.tb_low / 1000;
   return (retval);
}

u_long_long _papi_hwd_get_real_cycles(void)
{
   u_long_long usec, cyc;

   usec = _papi_hwd_get_real_usec();
   cyc = usec * _papi_hwi_system_info.hw_info.mhz;
   return ((u_long_long) cyc);
}

u_long_long _papi_hwd_get_virt_usec(const hwd_context_t * context)
{
   u_long_long retval;
   struct tms buffer;

   times(&buffer);
   retval = (u_long_long) buffer.tms_utime * (u_long_long) (1000000 / CLK_TCK);
   return (retval);
}

u_long_long _papi_hwd_get_virt_cycles(const hwd_context_t * context)
{
   float usec, cyc;

   usec = (float) _papi_hwd_get_virt_usec(context);
   cyc = usec * _papi_hwi_system_info.hw_info.mhz;
   return ((u_long_long) cyc);
}

void _papi_hwd_error(int error, char *where)
{
   sprintf(where, "Substrate error");
   pm_error(where, error);
}

int _papi_hwd_init_global(void)
{
   int retval=PAPI_OK;

   /* Fill in what we can of the papi_system_info. */

   retval = get_system_info();
   if (retval)
      return (retval);

   retval = get_memory_info(&_papi_hwi_system_info.hw_info);
   if (retval)
      return (retval);

   DBG((stderr, "Found %d %s %s CPU's at %f Mhz.\n",
        _papi_hwi_system_info.hw_info.totalcpus,
        _papi_hwi_system_info.hw_info.vendor_string,
        _papi_hwi_system_info.hw_info.model_string, _papi_hwi_system_info.hw_info.mhz));

   setup_native_table();
   if (!_papi_hwd_init_preset_search_map(&pminfo)){ 
      return (PAPI_ESBSTR);}
   retval = _papi_hwi_setup_all_presets(preset_search_map);

   return (retval);
}

int _papi_hwd_init(hwd_context_t * context)
{
   int retval;
   /* Initialize our global machdep. */

   _papi_hwd_init_control_state(&context->cntrl);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
   int tmp = 0, i = 1 << (POWER_MAX_COUNTERS - 1);

   while (i) {
      tmp = i & cntr_avail_bits;
      if (tmp)
         return (tmp);
      i = i >> 1;
   }
   return (0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, int *to)
{
   int useme, i;

   for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
      useme = (1 << i) & selector;
      if (useme) {
         to[i] = from[i];
      }
   }
}


int _papi_hwd_add_prog_event(hwd_control_state_t * this_state,
                             unsigned int event, void *extra, EventInfo_t * out)
{
   return (PAPI_ESBSTR);
}

#if 1
void dump_cmd(pm_prog_t * t)
{
   fprintf(stderr, "mode.b.threshold %d\n", t->mode.b.threshold);
   fprintf(stderr, "mode.b.spare %d\n", t->mode.b.spare);
   fprintf(stderr, "mode.b.process %d\n", t->mode.b.process);
   fprintf(stderr, "mode.b.kernel %d\n", t->mode.b.kernel);
   fprintf(stderr, "mode.b.user %d\n", t->mode.b.user);
   fprintf(stderr, "mode.b.count %d\n", t->mode.b.count);
   fprintf(stderr, "mode.b.proctree %d\n", t->mode.b.proctree);
   fprintf(stderr, "events[0] %d\n", t->events[0]);
   fprintf(stderr, "events[1] %d\n", t->events[1]);
   fprintf(stderr, "events[2] %d\n", t->events[2]);
   fprintf(stderr, "events[3] %d\n", t->events[3]);
   fprintf(stderr, "events[4] %d\n", t->events[4]);
   fprintf(stderr, "events[5] %d\n", t->events[5]);
   fprintf(stderr, "events[6] %d\n", t->events[6]);
   fprintf(stderr, "events[7] %d\n", t->events[7]);
   fprintf(stderr, "reserved %d\n", t->reserved);
}

void dump_data(pm_data_t * d)
{
   int i;

   for (i = 0; i < MAX_COUNTERS; i++) {
      fprintf(stderr, "accu[%d] = %lld\n", i, d->accu[i]);
   }
}
#endif

/*int _papi_hwd_reset(EventSetInfo_t *ESI, EventSetInfo_t *zero)*/
int _papi_hwd_reset(hwd_context_t * ESI, hwd_control_state_t * zero)
{
   pm_reset_data_mythread();
   return (PAPI_OK);
}


int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long ** val)
{
   int retval;
   int i;
   static pm_data_t data;

   retval = pm_get_data_mythread(&data);
   if (retval > 0)
      return (retval);

#if 0
   dump_data(&data);
#endif

   *val = data.accu;

   return (PAPI_OK);
}

int _papi_hwd_setmaxmem()
{
   return (PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
   switch (code) {
/* I don't understand what it means to set the default domain 
    case PAPI_DEFDOM:
      return(set_default_domain(zero, option->domain.domain));
*/
   case PAPI_DOMAIN:
      return (set_domain(&(option->domain.ESI->machdep), option->domain.domain));
/* I don't understand what it means to set the default granularity 
    case PAPI_DEFGRN:
      return(set_default_granularity(zero, option->granularity.granularity));
*/
   case PAPI_GRANUL:
      return (set_granularity
              (&(option->granularity.ESI->machdep), option->granularity.granularity));
#if 0
   case PAPI_INHERIT:
      return (set_inherit(option->inherit.inherit));
#endif
   default:
      return (PAPI_EINVAL);
   }
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long events[])
{
   return (PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
   pm_delete_program_mythread();
   return (PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
   return (PAPI_OK);
}


void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *i)
{
   _papi_hwi_context_t ctx;

   ctx.si = si;
   ctx.ucontext = (hwd_ucontext_t *) i;
   _papi_hwi_dispatch_overflow_signal(&ctx, _papi_hwi_system_info.supports_hw_overflow, 0,
                                      0);
}

/*int _papi_hwd_set_overflow(EventSetInfo_t *ESI, EventSetOverflowInfo_t *overflow_option)*/
int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = &ESI->machdep;

   if (threshold == 0) {
      this_state->timer_ms = 0;
      ESI->overflow.timer_ms = 0;
   } else {
      this_state->timer_ms = 1; /* Millisecond intervals are the only way to go */
      ESI->overflow.timer_ms = 1;
   }

   return (PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_ESBSTR);
}

/*void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_jmpbuf.jmp_context.iar;

  return(location);
}*/


/* Copy the current control_state into the new thread context */
/*int _papi_hwd_start(EventSetInfo_t *ESI, EventSetInfo_t *zero)*/
int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * cntrl)
{
   int i, retval;
   hwd_control_state_t *current_state = &ctx->cntrl;

   /* If we are nested, merge the global counter structure
      with the current eventset */

#if 1
   DBG((stderr, "Start\n"));
#endif

   /* Copy the global counter structure to the current eventset */

   DBG((stderr, "Copying states\n"));
   memcpy(current_state, cntrl, sizeof(hwd_control_state_t));

   retval = pm_set_program_mythread(&current_state->counter_cmd);
 if (retval != 0)
   {
     extern unsigned long int (*_papi_hwi_thread_id_fn)(void);
     if ((retval == 13) && (_papi_hwi_thread_id_fn))
       {
	 pm_delete_program_mythread();
	 retval = pm_set_program_mythread(&current_state->counter_cmd);
	 if (retval != 0)
	   return(retval);
       }
     else
       return(retval);
   }

   /* Set up the new merged control structure */

#if 0
   dump_cmd(&current_state->counter_cmd);
#endif

   /* Start the counters */

   retval = pm_start_mythread();
   if (retval > 0)
      return (retval);

   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * cntrl)
{
   int retval;

   retval = pm_stop_mythread();
   if (retval > 0)
      return (retval);

   retval = pm_delete_program_mythread();
   if (retval > 0)
      return (retval);

   return (PAPI_OK);
}

void _papi_hwd_lock_init(void)
{
   int i;
   for (i = 0; i < PAPI_MAX_LOCK; i++)
      lock[i] = (int *) (lock_var + i);
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
