/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* PAPI stuff */

#include SUBSTRATE

extern hwi_search_t *preset_search_map;
extern native_event_entry_t *native_table;
extern hwi_search_t _papi_hwd_preset_map[];
extern papi_mdi_t _papi_hwi_system_info;

#ifdef DEBUG
void print_control(const struct pmctr_t *control) {
   SUBDBG("Command block at %p:\n", control);
   SUBDBG("   CTR0 value:%d\n", control->CTR0);
   SUBDBG("   CTR1 value:%d\n", control->CTR1);
   SUBDBG("   CTR2 value:%d\n", control->CTR2);
   SUBDBG("   SEL0 value:%d\n", control->CTR0);
   SUBDBG("   SEL1 value:%d\n", control->CTR1);
   SUBDBG("   SEL2 value:%d\n", control->CTR2);
   SUBDBG("   Ku   value:%d\n", control->Ku);
   SUBDBG("   Kp   value:%d\n", control->Kp);
   SUBDBG("   Kk   value:%d\n", control->Kk);
   SUBDBG("   CTL0 value:%d\n", control->CTL0);
   SUBDBG("   CTL1 value:%d\n", control->CTL1);
   SUBDBG("   CTL2 value:%d\n", control->CTL2);
}
#endif

/* Utility functions */

void _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   int kill_pal = 0;
   int kill_user = 0;
   int kill_kernel = 0;

   memset(&ptr->counter_cmd, 0x0, sizeof(pmctr_t));

   switch (_papi_system_info.default_domain) {
   case PAPI_DOM_USER:
      {
         kill_user = 1;
      }
      break;
   case PAPI_DOM_KERNEL:
      {
         kill_kernel = 1;
      }
      break;
   case PAPI_DOM_OTHER:
      {
         kill_pal = 1;
      }
      break;
   case PAPI_DOM_ALL:
      {
         kill_kernel = 1;
         kill_pal = 1;
         kill_user = 1;
      }
      break;
   default:
      abort();
   }

   ptr->counter_cmd.Kp = kill_pal;
   ptr->counter_cmd.Ku = kill_user;
   ptr->counter_cmd.Kk = kill_kernel;
   ptr->counter_cmd.CTL0 = CTL_OFF;
   ptr->counter_cmd.CTL1 = CTL_OFF;
   ptr->counter_cmd.CTL2 = CTL_OFF;
}

/* This function clears the current contents of the control structure and
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t *this_state,
                                   NativeInfo_t *native, int count) {
   int i;

   /* fill the counters we're using */
   _papi_hwd_init_control_state(this_state);
   for (i = 0; i < count; i++) {
      /* Add counter control command values to eventset */
      if(native[i].ni_bits.selector == CNTR1) {
         if(this_state->counter_cmd.CTL0) {
            return (PAPI_ECNFLCT);
         }
         this_state->counter_cmd.SEL0 = native[i].ni_bits.code;
         this_state->counter_cmd.CTL0 = CTL_ON;
      }
      if(native[i].ni_bits.selector == CNTR2) {
         if(this_state->counter_cmd.CTL1) {
            return (PAPI_ECNFLCT);
         }
         this_state->counter_cmd.SEL1 = native[i].ni_bits.code;
         this_state->counter_cmd.CTL1 = CTL_ON;
      }
      if(native[i].ni_bits.selector == CNTR4) {
         if(this_state->counter_cmd.CTL2) {
            return (PAPI_ECNFLCT);
         }
         this_state->counter_cmd.SEL2 = native[i].ni_bits.code;
         this_state->counter_cmd.CTL2 = CTL_ON;
      }
   }
   return(PAPI_OK);
}

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
   return 1;
}

int _papi_hwd_set_domain(hwd_control_state_t * this_state, int domain)
{
   int did = 0;

   this_state->counter_cmd.Ku = 0;
   this_state->counter_cmd.Kk = 0;
   this_state->counter_cmd.Kp = 0;

   if (domain & PAPI_DOM_USER) {
      this_state->counter_cmd.Ku = 1;
      did = 1;
   }
   if (domain & PAPI_DOM_KERNEL) {
      this_state->counter_cmd.Kk = 1;
      did = 1;
   }
   if (domain & PAPI_DOM_OTHER) {
      this_state->counter_cmd.Kp = 1;
      did = 1;
   }
   if (!did)
      return (PAPI_EINVAL);

   return (PAPI_OK);
}

int _papi_hwd_update_shlib_info(void)
{
   return PAPI_ESBSTR;
}

static float getmhz(void)
{
   long sysconf(int request);
   float p;

   p = (float) sysconf(_SC_CRAY_CPCYCLE);       /* Picoseconds */
   p = p * 1.0e-12;             /* Convert to seconds */
   p = (int) (1.0 / (p * 1000000.0));   /* Convert to MHz */
   return (p);
}

void dump_infoblk(void)
{
   int i;

   printf("infoblk:\n");
   printf("date of program creation: %s\n", _infoblk.i_date);
   printf("time of program creation: %s\n", _infoblk.i_time);
   printf("name of generating program: %s\n", _infoblk.i_pid);
   printf("version of generating program: %s\n", _infoblk.i_pvr);
   printf("O.S. version at generation time: %s\n", _infoblk.i_osvr);
   printf("creation timestamp: %d\n", _infoblk.i_udt);
   for (i = 0; i < T3E_NUMBER_OF_SEGMENTS; i++) {
      printf("i_segs[%u]\n", i);
      printf("virtual address of segment: %x\n", _infoblk.i_segs[i].vaddr);
      printf("initial physical memory size: %u\n", _infoblk.i_segs[i].size);
      printf("place.offset: %x\n", _infoblk.i_segs[i].place.seg_offset);
      printf("place.length: %u\n", _infoblk.i_segs[i].place.length);
      printf("zeroed: %u\n", _infoblk.i_segs[i].zeroed);
      printf("length of uninitialized data: %u\n", _infoblk.i_segs[i].bss);
   }
}

int _papi_hwd_get_system_info(void)
{
   pid_t pid;
   int tmp;

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);

/*  dump_infoblk(); */
   /* _papi_system_info.exe_info.fullname; */
   /* _papi_system_info.exe_info.name; */
   _papi_system_info.exe_info.text_start = (caddr_t) 0x800000000;
   _papi_system_info.exe_info.data_start = (caddr_t) 0x100000000;
   _papi_system_info.exe_info.bss_start = (caddr_t) 0x200000000;
   _papi_system_info.exe_info.text_end =
       (caddr_t) (_infoblk.i_segs[0].size + 0x800000000);
   _papi_system_info.exe_info.data_end =
       (caddr_t) (_infoblk.i_segs[1].size + 0x100000000);
   _papi_system_info.exe_info.bss_end = (caddr_t) (_infoblk.i_segs[2].size + 0x200000000);

   _papi_system_info.hw_info.ncpu = sysconf(_SC_CRAY_NCPU);
   _papi_system_info.hw_info.totalcpus = sysconf(_SC_CRAY_NCPU);
   _papi_system_info.hw_info.nnodes = 1;
   _papi_system_info.hw_info.mhz = getmhz();
   strcpy(_papi_system_info.hw_info.vendor_string, "Cray");
   _papi_system_info.hw_info.vendor = -1;
   _papi_system_info.hw_info.revision = 0.0;
   _papi_system_info.hw_info.model = -1;
   strcpy(_papi_system_info.hw_info.model_string, "Alpha 21164");

   _papi_system_info.cpunum = sysconf(_SC_CRAY_PPE);

   return (PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and
   reserve EventSet zero. */
int _papi_hwd_init_global(void) {
   int retval;

   /* Initialize outstanding values in machine info structure */
   if (_papi_hwd_mdi_init() != PAPI_OK) {
      return (PAPI_ESBSTR);
   }

   /* Fill in what we can of the papi_system_info. */
   retval = _papi_hwd_get_system_info();
   if (retval != PAPI_OK)
      return (retval);

   /* Setup presets */
   retval = _papi_hwi_setup_all_presets(preset_search_map);
   if (retval)
      return (retval);

   /* Setup memory info */
   retval =
       _papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, (int) _papi_hwi_system_info.hw_info.vendor);
   if(retval)
      return (retval);

   return(PAPI_OK);
}

int _papi_hwd_init(hwd_context_t *ctx)
{
   return(PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t * this_state,
                             unsigned int event, void *extra, EventInfo_t * out)
{
   return (PAPI_ESBSTR);
}

int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * state) {
   int error;
   if((error = _wrperf(this_state->counter_cmd, 0, 0, 0)) < 0) {
      SUBDBG("_wrperf returns: %d\n", error);
      error_return(PAPI_ESYS, VCNTRL_ERROR);
   }
#ifdef DEBUG
   print_control(&state->counter_cmd);
#endif
   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
   return(PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_papi_hwd_start(ctx, cntrl));
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * spc, long_long **dp)
{
   int shift_cnt = 0;
   int retval, selector, j = 0, i;
   long long correct[3];

   retval = update_global_hwcounters(zero);
   if (retval)
      return (retval);

   retval = correct_local_hwcounters(zero, ESI, correct);
   if (retval)
      return (retval);

   /* This routine distributes hardware counters to software counters in the
      order that they were added. Note that the higher level 
      EventSelectArray[i] entries may not be contiguous because the user
      has the right to remove an event. */

   for (i = 0; i < _papi_system_info.num_cntrs; i++) {
      selector = ESI->EventInfoArray[i].selector;
      if (selector == PAPI_NULL)
         continue;

      DBG((stderr, "Event %d, mask is 0x%x\n", j, selector));

      if (ESI->EventInfoArray[i].command == NOT_DERIVED) {
         shift_cnt = ffs(selector) - 1;
         assert(shift_cnt >= 0);
         events[j] = correct[shift_cnt];
      }

      /* If this is a derived event */

      else
         events[j] = handle_derived(&ESI->EventInfoArray[i], correct);

      /* Early exit! */

      if (++j == ESI->NumberOfEvents)
         return (PAPI_OK);
   }

   /* Should never get here */

   return (PAPI_EBUG);
}

int _papi_hwd_setmaxmem()
{
   return (PAPI_OK);
}

int _papi_hwd_ctl(EventSetInfo_t * zero, int code, _papi_int_option_t * option)
{
   switch (code) {
   case PAPI_DEFDOM:
   case PAPI_DOMAIN:
      return (_papi_hwd_set_domain(option->domain.ESI->machdep, option->domain.domain));
   case PAPI_GRANUL:
      return (set_granularity(&option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
   case PAPI_INHERIT:
      return (set_inherit(zero, option->inherit.inherit));
#endif
   default:
      return (PAPI_EINVAL);
   }
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * cntrl, long_long *from) {
   return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown_global(void)
{
   return (PAPI_OK);
}

int _papi_hwd_shutdown(hwd_context_t * ctx) {
   return (PAPI_OK);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *context)
{
   _papi_hwi_context_t ctx;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

   _papi_hwi_dispatch_overflow_signal((void *) &ctx,
                                     _papi_hwi_system_info.supports_hw_overflow,
                                      si->si_pmc_ovf_mask, 0);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = (hwd_control_state_t *) ESI->machdep;

   if (overflow_option->threshold == 0) {
      this_state->timer_ms = 0;
      overflow_option->timer_ms = 0;
   } else {
      this_state->timer_ms = 1; /* Millisecond intervals are the only way to go */
      overflow_option->timer_ms = 1;
   }

   return (PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex, int threshold) {

   /* This function is not used and shouldn't be called. */

   return (PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

/* 75 Mhz sys. clock */

u_long_long _papi_hwd_get_real_cycles(void)
{
   return (_rtc() * (u_long_long) (_papi_system_info.hw_info.mhz / 75.0));
}

u_long_long _papi_hwd_get_real_usec(void)
{
   return (_rtc() / 75);
}

u_long_long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   struct tms buffer;

   times(&buffer);
   return ((u_long_long) buffer.tms_utime * (u_long_long) (1000000 / CLK_TCK));
}

u_long_long _papi_hwd_get_virt_cycles(const hwd_context_t * ctx)
{
   float usec, cyc;

   usec = (float) _papi_hwd_get_virt_usec(ctx);
   cyc = usec * _papi_system_info.hw_info.mhz;
   return ((u_long_long) cyc);
}

void _papi_hwd_lock_init(void)
{
}

#pragma _CRI soft $MULTION
#define _papi_hwd_lock(lck)		\
do {					\
 if ($MULTION == 0) _semts(lck)		\
while(0)

#define _papi_hwd_unlock(lck)		\
do {					\
    if ($MULTION == 0) _semclr(lck);	\
while(0)

/*  DID I do the above correctly?  The manual seems to say
 * any value between 0 and 31 is valid as long as they are constants
 * -KSL
 */

/* Initialize the system-specific settings */
extern int _papi_hwd_mdi_init()
{
     /* Name of the substrate we're using */
    strcpy(_papi_hwi_system_info.substrate, "$Id$");
   _papi_hwi_system_info.supports_hw_overflow = HW_OVERFLOW;
   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;
   _papi_hwi_system_info.num_cntrs = 3;
   _papi_hwi_system_info.num_gp_cntrs = 3;

   return (PAPI_OK);
}

/* Thread hooks 

void __pdf_th_create(void)
{
  extern PAPI_notify_handler_t thread_notifier;
}

void __pdf_th_destroy(void)
{
  extern PAPI_notify_handler_t thread_notifier;
}

*/

void _papi_hwd_bpt_map_set(hwd_reg_alloc_t * dst, int ctr)
{
}

int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst)
{
   return (PAPI_OK);
}

int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   return (PAPI_OK);
}

void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
}

void _papi_hwd_bpt_map_update(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
}

/*************************************/
/* CODE TO SUPPORT OPAQUE NATIVE MAP */
/*************************************/

/* Given a native event code, returns the short text label. */
char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   return (native_table[EventCode & NATIVE_AND_MASK].name);
}

/* Given a native event code, returns the longer native event
   description. */
char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (native_table[EventCode & NATIVE_AND_MASK].description);
}

/* Given a native event code, assigns the native event's
   information to a given pointer. */
int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   if(native_table[(EventCode & NATIVE_AND_MASK)].resources.selector == 0) {
      return (PAPI_ENOEVNT);
   }
   bits = &native_table[EventCode & NATIVE_AND_MASK].resources;
   return (PAPI_OK);
}

/* Given a native event code, looks for the next event in the table
   if the next one exists.  If not, returns the proper error code. */
int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifier)
{
   if (native_table[(*EventCode & NATIVE_AND_MASK) + 1].resources.selector) {
      *EventCode = *EventCode + 1;
      return (PAPI_OK);
   } else {
      return (PAPI_ENOEVNT);
   }
}
