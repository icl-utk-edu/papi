/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "tru64-alpha.h"
#include "papi_preset.h"

extern EventSetInfo_t *default_master_eventset;
int papi_debug;

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };

/* Globals */

static hwd_search_t findem_ev67[] = {
   {PAPI_TOT_CYC, {-1, PF67_RET_INST_AND_CYCLES}},
   {PAPI_TOT_INS, {PF67_RET_INST_AND_CYCLES, -1}},
   {PAPI_RES_STL, {-1, PF67_CYCLES_AND_REPLAY_TRAPS}},
   {-1, {-1, -1}}
};

static hwi_search_t findem_ev6[] = {
   {PAPI_TOT_CYC, {0, {NATIVE_MASK | 0, PAPI_NULL}}},
   {PAPI_TOT_INS, {0, {NATIVE_MASK | 1, PAPI_NULL}}},
   {PAPI_BR_CN, {0, {NATIVE_MASK | 2, PAPI_NULL}}},
   {PAPI_RES_STL, {0, {NATIVE_MASK | 7, PAPI_NULL}}},
   {0, {0, {0, 0}}}
};

static native_info_t ev6_native_table[] = {
/* 0  */ {"cycles", {PF6_MUX0_CYCLES, PF6_MUX1_CYCLES}},
/* 1  */ {"retinst", {PF6_MUX0_RET_INSTRUCTIONS, -1}},
/* 2  */ {"retcondbranch", {-1, PF6_MUX1_RET_COND_BRANCHES}},
/* 3  */ {"retdtb1miss", {-1, PF6_MUX1_RET_DTB_SINGLE_MISSES}},
/* 4  */ {"retdtb2miss", {-1, PF6_MUX1_RET_DTB_DOUBLE_MISSES}},
/* 5  */ {"retitbmiss", {-1, PF6_MUX1_RET_ITB_MISSES}},
/* 6  */ {"retunaltrap", {-1, PF6_MUX1_RET_UNALIGNED_TRAPS}},
/* 7  */ {"replay", {-1, PF6_MUX1_REPLAY_TRAP}}
};

extern papi_mdi_t _papi_hwi_system_info;
/* Utility functions */

/* Input as code from HWRPB, Thanks Bill Gray. */


static int set_domain(hwd_control_state_t * this_state, int domain)
{
   return (PAPI_ESBSTR);
}

static int set_granularity(hwd_control_state_t * this_state, int domain)
{
   return (PAPI_ESBSTR);
}

static int set_default_domain(hwd_control_state_t * ctrl, int domain)
{
   return (set_domain(ctrl, domain));
}

static int set_default_granularity(hwd_control_state_t * ctrl,
                                   int granularity)
{
   return (set_granularity(ctrl, granularity));
}

static int set_inherit(EventSetInfo_t * zero, int arg)
{
   return (PAPI_ESBSTR);
}

static void init_config(hwd_control_state_t * ptr)
{
   memset(&ptr->counter_cmd, 0x0, sizeof(ptr->counter_cmd));
}

static int get_system_info(void)
{
   int fd, retval, family;
   prpsinfo_t info;
   struct cpu_info cpuinfo;
   long proc_type;
   pid_t pid;
   char pname[PATH_MAX], *ptr;

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);
   sprintf(pname, "/proc/%05d", (int) pid);

   fd = open(pname, O_RDONLY);
   if (fd == -1)
      return (PAPI_ESYS);
   if (ioctl(fd, PIOCPSINFO, &info) == -1)
      return (PAPI_ESYS);
   close(fd);

   /* Cut off any arguments to exe */
   {
     char *tmp;
     tmp = strchr(info.pr_psargs, ' ');
     if (tmp != NULL)
       *tmp = '\0';
   }

   if (realpath(info.pr_psargs,pname))
     strncpy(_papi_hwi_system_info.exe_info.fullname, pname, PAPI_HUGE_STR_LEN);
   else
     strncpy(_papi_hwi_system_info.exe_info.fullname, info.pr_psargs, PAPI_HUGE_STR_LEN);

   strcpy(_papi_hwi_system_info.exe_info.address_info.name, info.pr_fname);

   /* retval = pm_init(0,&tmp);
      if (retval > 0)
      return(retval); */

   if (getsysinfo
       (GSI_CPU_INFO, (char *) &cpuinfo, sizeof(cpuinfo), NULL, NULL,
        NULL) == -1)
      return PAPI_ESYS;

   if (getsysinfo
       (GSI_PROC_TYPE, (char *) &proc_type, sizeof(proc_type), 0, 0,
        0) == -1)
      return PAPI_ESYS;
   proc_type &= 0xffffffff;

/*
  _papi_hwi_system_info.hw_info.ncpu = cpuinfo.current_cpu;
*/
   _papi_hwi_system_info.hw_info.mhz = (float) cpuinfo.mhz;
   _papi_hwi_system_info.hw_info.ncpu = cpuinfo.cpus_in_box;
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus =
       _papi_hwi_system_info.hw_info.ncpu *
       _papi_hwi_system_info.hw_info.nnodes;
   _papi_hwi_system_info.hw_info.vendor = -1;
   strcpy(_papi_hwi_system_info.hw_info.vendor_string, "Compaq");
   _papi_hwi_system_info.hw_info.model = proc_type;

   _papi_hwi_system_info.num_sp_cntrs = 1;
   strcpy(_papi_hwi_system_info.hw_info.model_string, "Alpha ");
   family = cpu_implementation_version();

   /* program text segment, data segment  address info */
   _papi_hwi_system_info.exe_info.address_info.text_start =
       (caddr_t) & _ftext;
   _papi_hwi_system_info.exe_info.address_info.text_end =
       (caddr_t) & _etext;
   _papi_hwi_system_info.exe_info.address_info.data_start = 
                                            (caddr_t) & _fdata;
   _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) & _edata;
   _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) & _fbss;
   _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) & _ebss;

   if (family == 0) {
      strcat(_papi_hwi_system_info.hw_info.model_string, "21064");
      _papi_hwi_system_info.num_cntrs = 2;
      _papi_hwi_system_info.num_gp_cntrs = 2;
   }
   if (family == 2) {
      strcat(_papi_hwi_system_info.hw_info.model_string, "21264");
      _papi_hwi_system_info.num_cntrs = 2;
      _papi_hwi_system_info.num_gp_cntrs = 2;
   } else if (family == 1) {
      strcat(_papi_hwi_system_info.hw_info.model_string, "21164");
      _papi_hwi_system_info.num_cntrs = 3;
      _papi_hwi_system_info.num_gp_cntrs = 3;
   } else
      return (PAPI_ESBSTR);

   if (family != 2) {
      fprintf(stderr, "PAPI: Don't know processor family %d\n", family);
      return (PAPI_ESBSTR);
   }


   retval = _papi_hwi_setup_all_presets(findem_ev6);
   if (retval)
      return (retval);

   return (PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

extern u_int read_cycle_counter(void);
extern u_int read_virt_cycle_counter(void);

long long _papi_hwd_get_real_usec(void)
{
   struct timespec res;

   if ((clock_gettime(CLOCK_REALTIME, &res) == -1))
      return (PAPI_ESYS);
   return (res.tv_sec * 1000000) + (res.tv_nsec / 1000);
}

long long _papi_hwd_get_real_cycles(void)
{
   return ((long long) _papi_hwd_get_real_usec() *
           _papi_hwi_system_info.hw_info.mhz);
}

long long _papi_hwd_get_virt_usec(EventSetInfo_t * zero)
{
   struct rusage res;

   if ((getrusage(RUSAGE_SELF, &res) == -1))
      return (PAPI_ESYS);
   return ((res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec);
}

long long _papi_hwd_get_virt_cycles(EventSetInfo_t * zero)
{
   return ((long long) _papi_hwd_get_virt_usec(zero) *
           _papi_hwi_system_info.hw_info.mhz);
}

void _papi_hwd_error(int error, char *where)
{
   sprintf(where, "Substrate error");
}

int _papi_hwd_init_global(void)
{
   int retval;

   /* Fill in what we can of the papi_hwi_system_info. */

   retval = get_system_info();
   if (retval)
      return (retval);

   DBG((stderr, "Found %d %s %s CPU's at %f Mhz.\n",
        _papi_hwi_system_info.hw_info.totalcpus,
        _papi_hwi_system_info.hw_info.vendor_string,
        _papi_hwi_system_info.hw_info.model_string,
        _papi_hwi_system_info.hw_info.mhz));

   return (PAPI_OK);
}

int _papi_hwd_init(hwd_context_t * ctx)
{
   long arg;
   int fd;


   fd = open("/dev/pfcntr", O_RDONLY | PCNTOPENALL);
/*
  fd = open("/dev/pfcntr",O_RDONLY | PCNTOPENONE);
*/
   if (fd == -1)
      return (PAPI_ESYS);

   arg = PFM_COUNTERS;
   if (ioctl(fd, PCNTSETITEMS, &arg) == -1) {
    bail:
      close(fd);
      return (PAPI_ESYS);
   }

   if (ioctl(fd, PCNTLOGSELECT) == -1)
      goto bail;

   ctx->fd = fd;

/*
  init_config(machdep); 
*/

   return (PAPI_OK);
}

/* Go from highest counter to lowest counter. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
   int tmp = 0, i = 1 << (_papi_hwi_system_info.num_cntrs - 1);

   while (i) {
      tmp = i & cntr_avail_bits;
      if (tmp)
         return (tmp);
      i = i >> 1;
   }
   return (0);
}

static int get_avail_hwcntr_num(int cntr_avail_bits)
{
   int tmp = 0, i = _papi_hwi_system_info.num_cntrs - 1;

   while (i) {
      tmp = (1 << i) & cntr_avail_bits;
      if (tmp)
         return (i);
      i--;
   }
   return (0);
}

static void set_hwcntr_codes(int selector, long *from, ev_control_t * to)
{
   int useme, i;

   for (i = 0; i < _papi_hwi_system_info.num_cntrs; i++) {
      useme = (1 << i) & selector;
      if (useme) {
         to->ev6 |= from[i];
      }
   }
}

int _papi_hwd_add_event(hwd_control_state_t * this_state,
                        unsigned int EventCode, EventInfo_t * out)
{
   int selector = 0;
   int avail = 0;
   long tmp_cmd[MAX_COUNTERS], *codes;

   if (EventCode & PRESET_MASK) {
      int preset_index;
      int derived;

      preset_index = EventCode & PRESET_AND_MASK;

      selector = preset_map[preset_index].selector;
      if (selector == 0)
         return (PAPI_ENOEVNT);
      derived = preset_map[preset_index].derived;

      /* Find out which counters are available. */

      avail = selector & ~this_state->selector;

      /* If not derived */

      if (preset_map[preset_index].derived == 0) {
         /* Pick any counter available */

         selector = get_avail_hwcntr_bits(avail);
         if (selector == 0)
            return (PAPI_ECNFLCT);
      } else {
         /* Check the case that if not all the counters 
            required for the derived event are available */

         if ((avail & selector) != selector)
            return (PAPI_ECNFLCT);
      }

      /* Get the codes used for this event */

      codes = preset_map[preset_index].counter_cmd;
/*
      out->command = derived;
      out->operand_index = preset_map[preset_index].operand_index;
*/
   } else {
      int hwcntr_num;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;    /* 0 through 7 */
      if ((hwcntr_num > _papi_hwi_system_info.num_gp_cntrs) ||
          (hwcntr_num < 0))
         return (PAPI_EINVAL);

      tmp_cmd[hwcntr_num] = EventCode >> 8;     /* 0 through 50 */
      if (tmp_cmd[hwcntr_num] > 50)
         return (PAPI_EINVAL);

      selector = 1 << hwcntr_num;

      /* Check if the counter is available */

      if (this_state->selector & selector)
         return (PAPI_ECNFLCT);

      codes = tmp_cmd;
   }

   /* Lower eight bits tell us what counters we need */

   assert((this_state->selector | 0xff) == 0xff);

   /* Perform any initialization of the control bits */

   if (this_state->selector == 0)
      init_config(this_state);

   /* Turn on the bits for this counter */

   set_hwcntr_codes(selector, codes, &this_state->counter_cmd);

   /* Update the new counter select field. */

   this_state->selector |= selector;

   /* Inform the upper level that the software event 'index' 
      consists of the following information. */

/*
  out->code = EventCode;
  out->selector = selector;
*/

   return (PAPI_OK);
}

int _papi_hwd_rem_event(hwd_control_state_t * this_state, EventInfo_t * in)
{
   int selector, used, preset_index;

#if 0
   /* Find out which counters used. */

   used = in->selector;

   /* Clear out counters that are part of this event. */

   this_state->selector = this_state->selector ^ used;
#endif

   return (PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t * this_state,
                             unsigned int event, void *extra,
                             EventInfo_t * out)
{
   return (PAPI_ESBSTR);
}

void dump_cmd(ev_control_t * t)
{
   DBG((stderr, "Command block at %p: 0x%x\n", t, t->ev6));
}

/* EventSet zero contains the 'current' state of the counting hardware */

#if 0
int _papi_hwd_merge(EventSetInfo_t * ESI, EventSetInfo_t * zero)
{
   int i, retval;
   hwd_control_state_t *this_state = &ESI->machdep;
   hwd_control_state_t *current_state = &zero->machdep;
   union pmctrs_ev6 start_em;
   long tmp;

   /* If we ARE NOT nested, 
      just copy the global counter structure to the current eventset */

   if (current_state->selector == 0x0) {
      current_state->selector = this_state->selector;
      memcpy(&current_state->counter_cmd, &this_state->counter_cmd,
             sizeof(current_state->counter_cmd));

      /* clear driver */

      retval = ioctl(current_state->fd, PCNTCLEARCNT);
      if (retval == -1)
         return (PAPI_ESYS);

      /* select events */

      DBG((stderr, "PCNT6MUX command %lx\n",
           current_state->counter_cmd.ev6));
      retval =
          ioctl(current_state->fd, PCNT6MUX,
                &current_state->counter_cmd.ev6);
      if (retval == -1)
         return (PAPI_ESYS);

      /* zero and restart selected counters */

      start_em.pmctrs_ev6_long = 0;
      start_em.pmctrs_ev6_cpu = PMCTRS_ALL_CPUS;
      start_em.pmctrs_ev6_select = PF6_SEL_COUNTER_0 | PF6_SEL_COUNTER_1;
      retval =
          ioctl(current_state->fd, PCNT6RESTART,
                &start_em.pmctrs_ev6_long);
      if (retval == -1)
         return PAPI_ESYS;

      return (PAPI_OK);
   }
   /* If we ARE nested, 
      carefully merge the global counter structure with the current eventset */
   else {
      int tmp, hwcntrs_in_both, hwcntrs_in_all, hwcntr;

      /* Stop the current context */

      /* Update the global values */

      retval = update_global_hwcounters(zero);
      if (retval)
         return (retval);

      /* Delete the current context */

      hwcntrs_in_both = this_state->selector & current_state->selector;
      hwcntrs_in_all = this_state->selector | current_state->selector;

      /* Check for events that are shared between eventsets and 
         therefore require no modification to the control state. */

      /* First time through, error check */

      tmp = hwcntrs_in_all;
      while ((i = ffs(tmp))) {
         hwcntr = 1 << (i - 1);
         tmp = tmp ^ hwcntr;
         if (hwcntr & hwcntrs_in_both) {
            if (!
                (counter_event_shared
                 (&this_state->counter_cmd, &current_state->counter_cmd,
                  i - 1)))
               return (PAPI_ECNFLCT);
         } else
             if (!
                 (counter_event_compat
                  (&this_state->counter_cmd, &current_state->counter_cmd,
                   i - 1)))
            return (PAPI_ECNFLCT);
      }

      /* Now everything is good, so actually do the merge */

      tmp = hwcntrs_in_all;
      while ((i = ffs(tmp))) {
         hwcntr = 1 << (i - 1);
         tmp = tmp ^ hwcntr;
         if (hwcntr & hwcntrs_in_both) {
            ESI->hw_start[i - 1] = zero->hw_start[i - 1];
            zero->multistart.SharedDepth[i - 1]++;
         } else if (hwcntr & this_state->selector) {
            current_state->selector |= hwcntr;
            counter_event_copy(&this_state->counter_cmd,
                               &current_state->counter_cmd, i - 1);
            ESI->hw_start[i - 1] = 0;
            zero->hw_start[i - 1] = 0;
         }
      }
   }

   /* Set up the new merged control structure */

#if 0
   dump_cmd(&current_state->counter_cmd);
#endif

   /* Stop the current context */

   retval = ioctl(current_state->fd, PCNTCLEARCNT);
   if (retval == -1)
      return (PAPI_ESYS);

   /* (Re)start the counters */

   retval =
       ioctl(current_state->fd, PCNT6MUX, &current_state->counter_cmd);
   if (retval == -1)
      return (PAPI_ESYS);

   return (PAPI_OK);
}

int _papi_hwd_unmerge(EventSetInfo_t * ESI, EventSetInfo_t * zero)
{
   int i, tmp, hwcntr, retval;
   hwd_control_state_t *this_state = &ESI->machdep;
   hwd_control_state_t *current_state = &zero->machdep;

   if ((zero->multistart.num_runners - 1) == 0) {
      current_state->selector = 0;
      return (PAPI_OK);
   } else {
      tmp = this_state->selector;
      while ((i = ffs(tmp))) {
         hwcntr = 1 << (i - 1);
         if (zero->multistart.SharedDepth[i - 1] - 1 < 0)
            current_state->selector ^= hwcntr;
         else
            zero->multistart.SharedDepth[i - 1]--;
         tmp ^= hwcntr;
      }
      return (PAPI_OK);
   }
}
#endif

int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int retval;
   union pmctrs_ev6 values_ev6;

   ctrl->cntrs[0]=0;
   ctrl->cntrs[1]=0;
   retval = ioctl(ctx->fd, PCNTRDISABLE);
   if (retval == -1)
      return (PAPI_ESYS);

   /* clear drivers counts */
   retval = ioctl(ctx->fd, PCNTCLEARCNT);
   if (retval == -1)
      return (PAPI_ESYS);


   /* Zero and enable hardware counters */

   values_ev6.pmctrs_ev6_long = 0;
   values_ev6.pmctrs_ev6_cpu = PMCTRS_ALL_CPUS;
   values_ev6.pmctrs_ev6_select = PF6_SEL_COUNTER_0 | PF6_SEL_COUNTER_1;
   retval = ioctl(ctx->fd, PCNT6RESTART, &values_ev6.pmctrs_ev6_long);
   if (retval == -1)
      return PAPI_ESYS;

   return (PAPI_OK);
}

int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * ctrl,
                   long_long ** events)
{
   int retval;
   struct pfcntrs_ev6 cntrs[EV_MAX_CPUS];
   struct pfcntrs_ev6 *ev6 = cntrs;
   union pmctrs_ev6 values_ev6;

   /* Whoa boy... */

   retval = ioctl(ctx->fd, PCNTRDISABLE);
   if (retval == -1)
      return (PAPI_ESYS);

   /* Get vals for the driver, thanks to Bill Gray! */

   retval = ioctl(ctx->fd, PCNT6READCNTRS, &ev6);
   if (retval == -1)
      return (PAPI_ESYS);

   ctrl->cntrs[0] += cntrs[0].pf_cntr0;
   ctrl->cntrs[1] += cntrs[0].pf_cntr1;
   *events = ctrl->cntrs;

   /* clear drivers counts */
   retval = ioctl(ctx->fd, PCNTCLEARCNT);
   if (retval == -1)
      return (PAPI_ESYS);


   /* Zero and enable hardware counters */

   values_ev6.pmctrs_ev6_long = 0;
   values_ev6.pmctrs_ev6_cpu = PMCTRS_ALL_CPUS;
   values_ev6.pmctrs_ev6_select = PF6_SEL_COUNTER_0 | PF6_SEL_COUNTER_1;
   retval = ioctl(ctx->fd, PCNT6RESTART, &values_ev6.pmctrs_ev6_long);
   if (retval == -1)
      return PAPI_ESYS;
   return (PAPI_OK);
}

int _papi_hwd_setmaxmem() {
   return (PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code,
                  _papi_int_option_t * option)
{
   switch (code) {
   case PAPI_DEFDOM:
      return (set_default_domain
              (&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DOMAIN:
      return (set_domain
              (&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DEFGRN:
      return (set_default_granularity
              (&option->domain.ESI->machdep,
               option->granularity.granularity));
   case PAPI_GRANUL:
      return (set_granularity
              (&option->granularity.ESI->machdep,
               option->granularity.granularity));
   default:
      return (PAPI_EINVAL);
   }
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * ctrl,
                    long long events[])
{
   /* should not return this error, since alpha support write function */
   return (PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
   int retval;

   retval = close(ctx->fd);
   if (retval == -1)
      return (PAPI_ESYS);
   return (PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
#if 0
   hwd_control_state_t *current_state = NULL;
   int retval;

   if (default_master_eventset)
      current_state = &default_master_eventset->machdep;
   if (current_state && current_state->fd) {
      retval = close(current_state->fd);
      if (retval == -1)
         return (PAPI_ESYS);
   }
#endif
   return (PAPI_OK);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex,
                           int threshold)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_EMISC);
}

int _papi_hwd_set_profile(EventSetInfo_t * ESI, int EventIndex,
                          int threshold)
{

   return (PAPI_OK);
}

int _papi_hwd_stop_profiling(ThreadInfo_t * master, EventSetInfo_t * ESI)
{

   return (PAPI_OK);
}

/*
void *_papi_hwd_get_overflow_address(void *context)
{
   void *location;
   struct sigcontext *info = (struct sigcontext *) context;
   location = (void *) info->sc_pc;

   return (location);
}
*/

/* start the hardware counting */
int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int retval;

   /* clear driver */
   retval = ioctl(ctx->fd, PCNTCLEARCNT);
   if (retval == -1)
      return (PAPI_ESYS);

   /* select events */
   DBG((stderr, "PCNT6MUX command %lx\n", ctrl->counter_cmd.ev6));
   retval = ioctl(ctx->fd, PCNT6MUX, &ctrl->counter_cmd.ev6);
   if (retval == -1)
      return (PAPI_ESYS);
   return (_papi_hwd_reset(ctx, ctrl));
}

/* stop the counting */
int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int retval;

   retval = ioctl(ctx->fd, PCNTRDISABLE);
   if (retval == -1)
      return (PAPI_ESYS);

   return (PAPI_OK);
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer)
{
   int index = *EventCode & NATIVE_AND_MASK;

   if (index < MAX_NATIVE_EVENT - 1) {
      *EventCode = *EventCode + 1;
      return (PAPI_OK);
   } else
      return (PAPI_ENOEVNT);
}

int _papi_hwd_update_shlib_info(void)
{
   return (PAPI_ESBSTR);
}

void _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   return;
}

/* this function will be called when adding events to the eventset and
   deleting events from the eventset
*/
int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                                   NativeInfo_t * native, int count)
{
   ev_control_t *ev_cmd = &this_state->counter_cmd;
   int i, nidx1, nidx2;
   long cmd0, cmd1;

   /* clear the control register */
   ev_cmd->ev6 = 0;

   /* eventset is empty */
   if (count == 0)
      return (PAPI_OK);

   cmd0 = -1;
   cmd1 = -1;
   /* one native event */
   if (count == 1) {
      nidx1 = native[0].ni_event & NATIVE_AND_MASK;
      cmd0 = ev6_native_table[nidx1].encode[0];
      native[0].ni_position = 0;
      if (cmd0 == -1) {
         cmd1 = ev6_native_table[nidx1].encode[1];
         native[0].ni_position = 1;
      }
   }

   /* two native events */
   if (count == 2) {
      int avail1, avail2;

      avail1 = 0;
      avail2 = 0;
      nidx1 = native[0].ni_event & NATIVE_AND_MASK;
      nidx2 = native[1].ni_event & NATIVE_AND_MASK;
      if (ev6_native_table[nidx1].encode[0] != -1)
         avail1 = 0x1;
      if (ev6_native_table[nidx1].encode[1] != -1)
         avail1 += 0x2;
      if (ev6_native_table[nidx2].encode[0] != -1)
         avail2 = 0x1;
      if (ev6_native_table[nidx2].encode[1] != -1)
         avail2 += 0x2;
      if ((avail1 | avail2) != 0x3)
         return (PAPI_ECNFLCT);
      if (avail1 == 0x3) {
         if (avail2 == 0x1) {
            cmd0 = ev6_native_table[nidx2].encode[0];
            cmd1 = ev6_native_table[nidx1].encode[1];
            native[0].ni_position = 1;
            native[1].ni_position = 0;
         } else {
            cmd1 = ev6_native_table[nidx2].encode[1];
            cmd0 = ev6_native_table[nidx1].encode[0];
            native[0].ni_position = 0;
            native[1].ni_position = 1;
         }
      } else {
         if (avail1 == 0x1) {
            cmd0 = ev6_native_table[nidx1].encode[0];
            cmd1 = ev6_native_table[nidx2].encode[1];
            native[0].ni_position = 0;
            native[1].ni_position = 1;
         } else {
            cmd0 = ev6_native_table[nidx2].encode[0];
            cmd1 = ev6_native_table[nidx1].encode[1];
            native[0].ni_position = 1;
            native[1].ni_position = 0;
         }
      }
   }

   if (cmd0 == -1)
      cmd0 = 0;
   if (cmd1 == -1)
      cmd1 = 0;
   ev_cmd->ev6 = cmd0 | cmd1;

   return (PAPI_OK);
}

int _papi_hwd_allocate_registers(EventSetInfo_t * ESI)
{
   return 1;
}

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   int nidx;

   nidx = EventCode ^ NATIVE_MASK;
   if (nidx >= 0 && nidx < PAPI_MAX_NATIVE_EVENTS)
      return (ev6_native_table[nidx].name);
   else
      return NULL;
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (_papi_hwd_ntv_code_to_name(EventCode));
}


void _papi_hwd_lock_init(void)
{
}

void _papi_hwd_lock(int lck)
{
   return;
}

void _papi_hwd_unlock(int lck)
{
   return;
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si,
                              ucontext_t * info)
{
   _papi_hwi_context_t ctx;
   ctx.si = si;
   ctx.ucontext = info;

   _papi_hwi_dispatch_overflow_signal((void *) &ctx,
                                      _papi_hwi_system_info.
                                      supports_hw_overflow, 0, 0);
}

int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t * dst, int ctr)
{
   return (PAPI_OK);
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t * dst, int ctr)
{
}

/* This function examines the event to determine
    if it has a single exclusive mapping.
    Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t * dst)
{
   return (PAPI_OK);
}

/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
   return (PAPI_OK);
}

/* This function removes the counters available to the src event
    from the counters available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t * dst,
                               hwd_reg_alloc_t * src)
{
}

/* This function updates the selection status of
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
}
