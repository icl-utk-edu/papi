/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

extern hwi_search_t *preset_search_map;
extern native_event_entry_t *native_table;
extern hwi_search_t _papi_hwd_t3e_preset_map;
extern native_event_entry_t _papi_hwd_t3e_native_table;
extern papi_mdi_t _papi_hwi_system_info;

#ifdef DEBUG
void print_control(pmctr_t *control) {
   SUBDBG("Command block at %p:\n", control);
   SUBDBG("   CTR0 value:%d\n", control->CTR0);
   SUBDBG("   CTR1 value:%d\n", control->CTR1);
   SUBDBG("   CTR2 value:%d\n", control->CTR2);
   SUBDBG("   SEL0 value:%d\n", control->SEL0);
   SUBDBG("   SEL1 value:%d\n", control->SEL1);
   SUBDBG("   SEL2 value:%d\n", control->SEL2);
   SUBDBG("   Ku   value:%d\n", control->Ku);
   SUBDBG("   Kp   value:%d\n", control->Kp);
   SUBDBG("   Kk   value:%d\n", control->Kk);
   SUBDBG("   CTL0 value:%d\n", control->CTL0);
   SUBDBG("   CTL1 value:%d\n", control->CTL1);
   SUBDBG("   CTL2 value:%d\n", control->CTL2);
}
#endif

int _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   int kill_pal = 0;
   int kill_user = 0;
   int kill_kernel = 0;

   memset(&ptr->counter_cmd, 0x0, sizeof(pmctr_t));

   if (_papi_hwi_system_info.default_domain & PAPI_DOM_USER)
     kill_user = 1;
   if (_papi_hwi_system_info.default_domain & PAPI_DOM_KERNEL)
     kill_kernel = 1;
   if (_papi_hwi_system_info.default_domain & PAPI_DOM_OTHER)
     kill_pal = 1;

   ptr->counter_cmd.Kp = kill_pal;
   ptr->counter_cmd.Ku = kill_user;
   ptr->counter_cmd.Kk = kill_kernel;
   ptr->counter_cmd.CTL0 = CTL_OFF;
   ptr->counter_cmd.CTL1 = CTL_OFF;
   ptr->counter_cmd.CTL2 = CTL_OFF;
   ptr->counter_cmd.SEL0 = 0x0;
   ptr->counter_cmd.SEL1 = 0x0;
   ptr->counter_cmd.SEL2 = 0x0;
   return(PAPI_OK);
}

/* This function clears the current contents of the control structure and
   updates it with whatever resources are allocated for all the native events
   in the native info structure array. */
int _papi_hwd_update_control_state(hwd_control_state_t *this_state,
                                   NativeInfo_t *native, int count, hwd_context_t *ctx) {
   int i, index;

   _papi_hwd_init_control_state(this_state);
   for(i = 0; i < count; i++) {
      index = native[i].ni_event & PAPI_NATIVE_AND_MASK;
      native[i].ni_bits.selector[0] = native_table[index].resources.selector[0];
      native[i].ni_bits.selector[1] = native_table[index].resources.selector[1];
      native[i].ni_bits.selector[2] = native_table[index].resources.selector[2];
      /* Add counter control command values to eventset */
      if(native[i].ni_bits.selector[0] != -1) {
         native[i].ni_position = 0;
         this_state->counter_cmd.SEL0 = native[i].ni_bits.selector[0];
         this_state->counter_cmd.CTL0 = CTL_ON;
      }
      if(native[i].ni_bits.selector[1] != -1) {
         native[i].ni_position = 1;
         this_state->counter_cmd.SEL1 = native[i].ni_bits.selector[1];
         this_state->counter_cmd.CTL1 = CTL_ON;
      }
      if(native[i].ni_bits.selector[2] != -1) {
         native[i].ni_position = 2;
         this_state->counter_cmd.SEL2 = native[i].ni_bits.selector[2];
         this_state->counter_cmd.CTL2 = CTL_ON;
      }
   }
#ifdef DEBUG
   print_control(&this_state->counter_cmd);
#endif
   return(PAPI_OK);
}

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI) {
   int i, index, sel0=0, sel1=0, sel2=0, flag=0;

   for(i = 0; i < ESI->NativeCount; i++) {
      index = ESI->NativeInfoArray[i].ni_event & PAPI_NATIVE_AND_MASK;
   /* Check for conflicts: These 3 events can form
      another event when 21 && 12 or 22 && 12 are
      in the same eventset, t3e doesn't realize this
      is what the user wants.  Instead, it considers
      this the event 10 or 11. */
      if(index == 21 || index == 22 || index == 12) {
         if(flag) return(0);
         flag = 1;
      }
      if(native_table[index].resources.selector[0] != -1) {
         if(sel0) return(0);
         sel0 = CTL_ON;
      }
      if(native_table[index].resources.selector[1] != -1) {
         if(sel1) return(0);
         sel1 = CTL_ON;
      }
      if(native_table[index].resources.selector[2] != -1) {
         if(sel2) return(0);
         sel2 = CTL_ON;
      }
   }
   return(1);
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
#if 0
   dump_infoblk();
#endif
   _papi_hwi_system_info.pid = pid;
   if(getcwd(_papi_hwi_system_info.exe_info.fullname, PAPI_MAX_STR_LEN) == NULL)
      return(PAPI_ESYS);
   strcat(_papi_hwi_system_info.exe_info.fullname, "/");
   strcat(_papi_hwi_system_info.exe_info.fullname, _infoblk.i_pid);
   strcpy(_papi_hwi_system_info.exe_info.address_info.name, _infoblk.i_pid);
   _papi_hwi_system_info.exe_info.address_info.text_start =
       (caddr_t)_infoblk.i_segs[0].vaddr;
   _papi_hwi_system_info.exe_info.address_info.data_start =
       (caddr_t)_infoblk.i_segs[1].vaddr;
   _papi_hwi_system_info.exe_info.address_info.bss_start =
       (caddr_t)_infoblk.i_segs[2].vaddr;
   _papi_hwi_system_info.exe_info.address_info.text_end =
       (caddr_t)(_infoblk.i_segs[0].size + _infoblk.i_segs[0].vaddr);
   _papi_hwi_system_info.exe_info.address_info.data_end =
       (caddr_t) (_infoblk.i_segs[1].size + _infoblk.i_segs[1].vaddr);
   _papi_hwi_system_info.exe_info.address_info.bss_end =
       (caddr_t) (_infoblk.i_segs[2].size + _infoblk.i_segs[2].vaddr);

   _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_CRAY_NCPU);
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_CRAY_NCPU);
   _papi_hwi_system_info.hw_info.mhz = getmhz();
   strcpy(_papi_hwi_system_info.hw_info.vendor_string, "Cray");
   _papi_hwi_system_info.hw_info.vendor = -1;
   _papi_hwi_system_info.hw_info.revision = 0.0;
   _papi_hwi_system_info.hw_info.model = -1;
   strcpy(_papi_hwi_system_info.hw_info.model_string, "Alpha 21164");

   return (PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and
   reserve EventSet zero. */

papi_svector_t _unicos_ev5_table[] = {
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
 {(void (*)())_papi_hwd_allocate_registers, VEC_PAPI_HWD_ALLOCATE_REGISTERS },
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {(void (*)())_papi_hwd_get_dmem_info, VEC_PAPI_HWD_GET_DMEM_INFO},
 {(void (*)())_papi_stop_profiling, VEC_PAPI_STOP_PROFILING},
 {(void (*)())_papi_hwd_set_overflow, VEC_PAPI_HWD_SET_OVERFLOW},
 {(void (*)())_papi_hwd_ntv_enum_events, VEC_PAPI_HWD_NTV_ENUM_EVENTS},
 {(void (*)())_papi_hwd_ntv_code_to_name, VEC_PAPI_HWD_NTV_CODE_TO_NAME},
 {(void (*)())_papi_hwd_ntv_code_to_descr, VEC_PAPI_HWD_NTV_CODE_TO_DESCR},
 {(void (*)())_papi_hwd_ntv_code_to_bits, VEC_PAPI_HWD_NTV_CODE_TO_BITS},
 {(void (*)())_papi_hwd_ntv_bits_to_info, VEC_PAPI_HWD_NTV_BITS_TO_INFO},
 {NULL, VEC_PAPI_END}
};

int _papi_hwd_init_substrate(void) {
   int retval;


   /* Initialize outstanding values in machine info structure */
   if (_papi_hwd_mdi_init() != PAPI_OK) {
      return (PAPI_ESBSTR);
   }

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _unicos_ev5_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

   /* Fill in what we can of the papi_system_info. */
   retval = _papi_hwd_get_system_info();
   if (retval != PAPI_OK)
      return (retval);

   /* Setup presets */
   native_table = &_papi_hwd_t3e_native_table;
   preset_search_map = &_papi_hwd_t3e_preset_map;
   retval = _papi_hwi_setup_all_presets(&_papi_hwd_t3e_preset_map, NULL);
   if (retval)
      return (retval);

   /* Setup memory info */
   if(_papi_hwd_get_memory_info(&_papi_hwi_system_info.hw_info, 0))
      return (retval);

   return(PAPI_OK);
}

int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * state) {
   int error;

   /* clear the accumulating counter values */
   state->counter_cmd.CTR0 = 0;
   state->counter_cmd.CTR1 = 0;
   state->counter_cmd.CTR2 = 0;
#ifdef DEBUG
   print_control(&state->counter_cmd);
#endif
   if((error = _wrperf(state->counter_cmd, 0, 0, 0)) < 0) {
      SUBDBG("_wrperf returns: %d\n", error);
      return(PAPI_ESYS);
   }
   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t *ctx, hwd_control_state_t *state) {
   return(PAPI_OK);
}

int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t *cntrl) {
   return(_papi_hwd_start(ctx, cntrl));
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long **events, int flags)
{
   pmctr_t *pmctr;
   long long pc_data[4];

   if(_rdperf(pc_data))
      return(PAPI_ESBSTR);

   pmctr = (pmctr_t *) & pc_data[0];
   ctrl->values[0] = (pc_data[1] << 16) + (long long)pmctr->CTR0;
   ctrl->values[1] = (pc_data[2] << 16) + (long long)pmctr->CTR1;
   ctrl->values[2] = (pc_data[3] << 14) + (long long)pmctr->CTR2;
   *events = ctrl->values;
#ifdef DEBUG
   if (ISLEVEL(DEBUG_SUBSTRATE)) {
      SUBDBG("raw val hardware index 0 is %lld\n",
            (long long) ctrl->values[0]);
      SUBDBG("raw val hardware index 1 is %lld\n",
            (long long) ctrl->values[1]);
      SUBDBG("raw val hardware index 2 is %lld\n",
            (long long) ctrl->values[2]);
   }
#endif
   return(PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{
   switch (code) {
   case PAPI_DEFDOM:
   case PAPI_DOMAIN:
      return (_papi_hwd_set_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_GRANUL:
   case PAPI_DEFGRN:
      return(PAPI_ESBSTR);
#if 0
   case PAPI_INHERIT:
      return (set_inherit(ctx, option->inherit.inherit));
#endif
   default:
      return (PAPI_EINVAL);
   }
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *context)
{
   _papi_hwi_context_t ctx;
   ThreadInfo_t *t = NULL;

   ctx.si = si;
   ctx.ucontext = (ucontext_t *)context;

   _papi_hwi_dispatch_overflow_signal((void *) &ctx, NULL, 0, 0, &t);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = (hwd_control_state_t *) &ESI->machdep;

   if (!threshold) {
      ESI->overflow.timer_ms = 0;
   } else {
      ESI->overflow.timer_ms = 1;
   }

   return (PAPI_OK);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI) {
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

/* Initialize the system-specific settings */
extern int _papi_hwd_mdi_init()
{
    /* Name of the substrate we're using */
    strcpy(_papi_hwi_system_info.substrate, "$Id$");
   _papi_hwi_system_info.supports_64bit_counters = 1;
   _papi_hwi_system_info.supports_real_usec = 1;
   _papi_hwi_system_info.supports_real_cyc = 1;
   _papi_hwi_system_info.num_cntrs = 3;
   _papi_hwi_system_info.num_gp_cntrs = 3;

   return (PAPI_OK);
}

/* 75 Mhz sys. clock */

long long _papi_hwd_get_real_cycles(void)
{
   return (((long long)_rtc() * (long long)_papi_hwi_system_info.hw_info.mhz) / (long long)75);
}

long long _papi_hwd_get_real_usec(void)
{
   return ((long long)_rtc() / (long long)75);
}

long long _papi_hwd_get_virt_usec(const hwd_context_t * ctx)
{
   struct tms buffer;
   long long retval;

   times(&buffer);
   SUBDBG("user %d system %d\n",(int)buffer.tms_utime,(int)buffer.tms_stime);
   retval = (long long)((buffer.tms_utime+buffer.tms_stime)*
     (1000000/CLK_TCK));
   return retval;
}

long long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   return (_papi_hwd_get_virt_usec(zero) * (long long)_papi_hwi_system_info.hw_info.mhz);
}
