/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "tru64-alpha.h"

extern EventSetInfo_t *default_master_eventset;

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };

#if 0
static hwd_search_t findem_ev4[] = {
   {PAPI_TOT_CYC, {PF_CYCLES, -1}},
   {PAPI_TOT_INS, {PF_ISSUES, -1}},
   {-1, {-1,}}
};
static hwd_search_t findem_ev5[] = {
   {PAPI_TOT_CYC, {PF5_MUX0_CYCLES, -1, PF5_MUX2_C_CYCLES}},
   {PAPI_TOT_INS, {PF5_MUX0_ISSUES, -1, -1}},
   {-1, {-1,}}
};
static hwd_search_t findem_pca[] = {
   {PAPI_TOT_CYC, {PF5_MUX0_CYCLES, -1, PF5_MUX2_C_CYCLES}},
   {PAPI_TOT_INS, {PF5_MUX0_ISSUES, -1, -1}},
   {-1, {-1,}}
};
static int setmuxcode = -1;
static int getcntcode = -1;
static int cntselcode = -1;
#endif

/* Globals */

static hwd_search_t findem_ev67[] = {
   {PAPI_TOT_CYC, {-1, PF67_RET_INST_AND_CYCLES, -1}},
   {PAPI_TOT_INS, {PF67_RET_INST_AND_CYCLES, -1, -1}},
   {PAPI_RES_STL, {-1, PF67_CYCLES_AND_REPLAY_TRAPS, -1}},
   {-1, {-1, -1, -1}}
};

static hwd_search_t findem_ev6[] = {
   {PAPI_TOT_CYC, {PF6_MUX0_CYCLES, PF6_MUX1_CYCLES, -1}},
   {PAPI_TOT_INS, {PF6_MUX0_RET_INSTRUCTIONS, -1, -1}},
   {PAPI_BR_CN, {-1, PF6_MUX1_RET_COND_BRANCHES, -1}},
   {PAPI_RES_STL, {-1, PF6_MUX1_REPLAY_TRAP, -1}},
   {-1, {-1, -1, -1}}
};

/* Utility functions */

/* Input as code from HWRPB, Thanks Bill Gray. */

static int setup_all_presets(int family, int model)
{
   int first, event, derived, hwnum;
   hwd_search_t *findem;
   char str[PAPI_MAX_STR_LEN];
   int num = _papi_system_info.num_gp_cntrs;
   int code;

   DBG((stderr, "Family %d, model %d\n", family, model));

   if (family == 2)
      findem = findem_ev6;
   else {
      fprintf(stderr, "PAPI: Don't know processor family %d, model %d\n", family, model);
      return (PAPI_ESBSTR);
   }

#if 0
   if (family == 0) {
      findem = findem_ev4;
      setmuxcode = PCNTSETMUX;
      getcntcode = PCNTGETCNT;
   } else if (family == 1) {
      if ((model == PCA56_CPU) || (model == PCA57_CPU))
         findem = findem_pca;
      else
         findem = findem_ev5;
      setmuxcode = PCNT5MUX;
      getcntcode = PCNT5GETCNT;
      cntselcode = PF5_SEL_COUNTER_0 | PF5_SEL_COUNTER_1 | PF5_SEL_COUNTER_2;
   } else if (family == 2) {
      if (model >= EV67_CPU)
         findem = findem_ev67;
      else
         findem = findem_ev6;
      setmuxcode = PCNT6MUX;
      getcntcode = PCNT6GETCNT;
      cntselcode = PF6_SEL_COUNTER_0 | PF6_SEL_COUNTER_1;
   } else {
      fprintf(stderr, "Unknown processor model %d family %d\n", model, family);
      return (PAPI_ESBSTR);
   }
#endif

   while ((code = findem->papi_code) != -1) {
      int i, index;

      index = code & PRESET_AND_MASK;
      preset_map[index].derived = NOT_DERIVED;
      preset_map[index].operand_index = 0;
      for (i = 0; i < num; i++) {
         if (findem->findme[i] != -1) {
            preset_map[index].selector |= 1 << i;
            preset_map[index].counter_cmd[i] = findem->findme[i];
            sprintf(str, "0x%x", findem->findme[i]);
            if (strlen(preset_map[index].note))
               strcat(preset_map[index].note, ",");
            strcat(preset_map[index].note, str);
         }
      }
      if (preset_map[index].selector != 0) {
         DBG((stderr, "Preset %d found, selector 0x%x\n",
              index, preset_map[index].selector));
      }
      findem++;
   }
   return (PAPI_OK);
}

static int counter_event_compat(const ev_control_t * a, const ev_control_t * b, int cntr)
{
   return (1);
}

static void counter_event_copy(ev_control_t * a, const ev_control_t * b, int cntr)
{
   long al = a->ev6;
   long bl = b->ev6;
   long mask = 0xf0 >> cntr;
   DBG((stderr, "copy: A %x B %x C %d M %x\n", al, bl, cntr, mask));

   bl = bl & mask;
   al = al | mask;
   al = al ^ mask;
   al = al & bl;
   a->ev6 = al;

   DBG((stderr, "A is now %x\n", al));
}

static int counter_event_shared(const ev_control_t * a, const ev_control_t * b, int cntr)
{
   long al = a->ev6;
   long bl = b->ev6;
   long mask = 0xf0 >> cntr;
   DBG((stderr, "shared?: A %x B %x C %d M %x\n", al, bl, cntr, mask));

   bl = bl & mask;
   al = al & mask;

   if (al == bl) {
      DBG((stderr, "shared!\n", al, bl));
      return (1);
   } else {
      DBG((stderr, "not shared!\n", al, bl));
      return (0);
   }
}

static int update_global_hwcounters(EventSetInfo_t * global)
{
   int i, retval;
   hwd_control_state_t *current_state = (hwd_control_state_t *) global->machdep;
   struct pfcntrs_ev6 tev6;
   struct pfcntrs_ev6 cntrs[EV_MAX_CPUS];
   struct pfcntrs_ev6 *ev6 = cntrs;
   union pmctrs_ev6 values_ev6;
   long counter_values[EV_MAX_COUNTERS] = { 0, 0, 0 };

   /* Whoa boy... */

   retval = ioctl(current_state->fd, PCNTRDISABLE);
   if (retval == -1)
      return (PAPI_ESYS);

   /* Get vals for the driver, thanks to Bill Gray! */

   retval = ioctl(current_state->fd, PCNT6READCNTRS, &ev6);
   if (retval == -1)
      return (PAPI_ESYS);

   /* Get CPU vals. */

/*  for (i=0;i<_papi_system_info.hw_info.ncpu;i++)
    { */
   /* Do the math */

   counter_values[0] += cntrs[0].pf_cntr0;
   counter_values[1] += cntrs[0].pf_cntr1;
   DBG((stderr, "Actual values %ld %ld \n", counter_values[0], counter_values[1]));

/*
      DBG((stderr,"Actual values %d %ld %ld \n",i,counter_values[0],counter_values[1]));
    }*/

   DBG((stderr, "update_global_hwcounters() %d: G%lld = G%lld + C%lld\n", 0,
        global->hw_start[0] + counter_values[0], global->hw_start[0], counter_values[0]));

   if (current_state->selector & 0x1)
      global->hw_start[0] = global->hw_start[0] + counter_values[0];

   if (current_state->selector & 0x2) {
      DBG((stderr, "update_global_hwcounters() %d: G%lld = G%lld + C%lld\n", 1,
           global->hw_start[1] + counter_values[1], global->hw_start[1],
           counter_values[1]));
      global->hw_start[1] = global->hw_start[1] + counter_values[1];
   }

   /* Clear driver counts */

   retval = ioctl(current_state->fd, PCNTCLEARCNT);
   if (retval == -1)
      return (PAPI_ESYS);

   /* Zero and enable hardware counters */

   values_ev6.pmctrs_ev6_long = 0;
   values_ev6.pmctrs_ev6_cpu = PMCTRS_ALL_CPUS;
   values_ev6.pmctrs_ev6_select = PF6_SEL_COUNTER_0 | PF6_SEL_COUNTER_1;
   retval = ioctl(current_state->fd, PCNT6RESTART, &values_ev6.pmctrs_ev6_long);
   if (retval == -1)
      return PAPI_ESYS;

   return (0);
}

static int correct_local_hwcounters(EventSetInfo_t * global, EventSetInfo_t * local,
                                    long long *correct)
{
   int i;

   for (i = 0; i < _papi_system_info.num_cntrs; i++) {
      DBG((stderr, "correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n", i,
           global->hw_start[i] - local->hw_start[i], global->hw_start[i],
           local->hw_start[i]));
      correct[i] = global->hw_start[i] - local->hw_start[i];
   }

   return (0);
}

static int set_domain(hwd_control_state_t * this_state, int domain)
{
   return (PAPI_ESBSTR);
}

static int set_granularity(hwd_control_state_t * this_state, int domain)
{
   return (PAPI_ESBSTR);
}

static int set_default_domain(EventSetInfo_t * zero, int domain)
{
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;
   return (set_domain(current_state, domain));
}

static int set_default_granularity(EventSetInfo_t * zero, int granularity)
{
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;
   return (set_granularity(current_state, granularity));
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
   char pname[PAPI_MAX_STR_LEN], *ptr;

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

   if (getcwd(_papi_system_info.exe_info.fullname, PAPI_MAX_STR_LEN) == NULL)
      return (PAPI_ESYS);
   strcat(_papi_system_info.exe_info.fullname, "/");
   strcat(_papi_system_info.exe_info.fullname, info.pr_fname);
   strncpy(_papi_system_info.exe_info.name, info.pr_fname, PAPI_MAX_STR_LEN);

   /* retval = pm_init(0,&tmp);
      if (retval > 0)
      return(retval); */

   if (getsysinfo(GSI_CPU_INFO, (char *) &cpuinfo, sizeof(cpuinfo), NULL, NULL, NULL) ==
       -1)
      return PAPI_ESYS;

   if (getsysinfo(GSI_PROC_TYPE, (char *) &proc_type, sizeof(proc_type), 0, 0, 0) == -1)
      return PAPI_ESYS;
   proc_type &= 0xffffffff;

   _papi_system_info.cpunum = cpuinfo.current_cpu;
   _papi_system_info.hw_info.mhz = (float) cpuinfo.mhz;
   _papi_system_info.hw_info.ncpu = cpuinfo.cpus_in_box;
   _papi_system_info.hw_info.nnodes = 1;
   _papi_system_info.hw_info.totalcpus =
       _papi_system_info.hw_info.ncpu * _papi_system_info.hw_info.nnodes;
   _papi_system_info.hw_info.vendor = -1;
   strcpy(_papi_system_info.hw_info.vendor_string, "Compaq");
   _papi_system_info.hw_info.model = proc_type;

   _papi_system_info.num_sp_cntrs = 1;
   strcpy(_papi_system_info.hw_info.model_string, "Alpha ");
   family = cpu_implementation_version();

   if (family == 0) {
      strcat(_papi_system_info.hw_info.model_string, "21064");
      _papi_system_info.num_cntrs = 2;
      _papi_system_info.num_gp_cntrs = 2;
   }
   if (family == 2) {
      strcat(_papi_system_info.hw_info.model_string, "21264");
      _papi_system_info.num_cntrs = 2;
      _papi_system_info.num_gp_cntrs = 2;
   } else if (family == 1) {
      strcat(_papi_system_info.hw_info.model_string, "21164");
      _papi_system_info.num_cntrs = 3;
      _papi_system_info.num_gp_cntrs = 3;
   } else
      return (PAPI_ESBSTR);

   retval = setup_all_presets(family, proc_type);
   if (retval)
      return (retval);

   return (PAPI_OK);
}

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

extern u_int read_cycle_counter(void);
extern u_int read_virt_cycle_counter(void);

#if 0
static long long real_time = 0;
static long long virt_time = 0;
static long long real_then = 0;
static long long virt_then = 0;
static const long long time_max = 0x7fffffff;

static void update_global_time(void)
{
   long long real_now = (long long) read_cycle_counter();
   long long virt_now = (long long) read_virt_cycle_counter();

   DBG((stderr, "REAL: %lld %lld %lld %lld\n", real_time, real_now, real_then,
        real_now - real_then));
   if (real_now < real_then)
      real_time += real_now + (time_max - real_then);
   else
      real_time += real_now - real_then;
   real_then = real_now;

   DBG((stderr, "VIRT: %lld %lld %lld %lld\n", virt_time, virt_now, virt_then,
        virt_now - virt_then));
   if (virt_now < virt_then)
      virt_time += virt_now + (time_max - virt_then);
   else
      virt_time += virt_now - virt_then;
   virt_then = virt_now;
}

int start_overflow_timer(void)
{
   int retval;
   struct sigaction action, oaction;
   struct sigevent event;
   struct itimerspec value;
   timer_t timer;

   memset(&action, 0x00, sizeof(struct sigaction));
   action.sa_flags = SA_RESTART;
   action.sa_handler = (void (*)(int)) update_global_time;
   if (sigaction(SIGRTMIN, &action, &oaction) < 0)
      return (PAPI_ESYS);

   memset(&event, 0x00, sizeof(struct sigevent));
   event.sigev_notify = SIGEV_SIGNAL;
   event.sigev_signo = SIGRTMIN;
   retval = timer_create(CLOCK_REALTIME, &event, &timer);
   if (retval == -1)
      return (PAPI_ESYS);

   memset(&value, 0x00, sizeof(struct itimerspec));
   value.it_interval.tv_sec = 1;
   value.it_value.tv_sec = 1;

   retval = timer_settime(timer, 0, &value, NULL);
   if (retval == -1)
      return (PAPI_ESYS);

   update_global_time();
}
#endif

long long _papi_hwd_get_real_usec(void)
{
#if 0
   if (real_then) {
      update_global_time();
      return (real_time / _papi_system_info.hw_info.mhz);
   } else
      start_overflow_timer();
#endif
#ifdef O
   return ((long long) read_cycle_counter() / _papi_system_info.hw_info.mhz);
#endif
   struct timespec res;

   if ((clock_gettime(CLOCK_REALTIME, &res) == -1))
      return (PAPI_ESYS);
   return (res.tv_sec * 1000000) + (res.tv_nsec / 1000);
}

long long _papi_hwd_get_real_cycles(void)
{
#if 0
   if (real_then) {
      update_global_time();
      return (real_time);
   } else
      start_overflow_timer();
#endif
/*  return((long long)read_cycle_counter());
*/
   return ((long long) _papi_hwd_get_real_usec() * _papi_system_info.hw_info.mhz);
}

long long _papi_hwd_get_virt_usec(EventSetInfo_t * zero)
{
#if 0
   if (virt_then) {
      update_global_time();
      return (virt_time / _papi_system_info.hw_info.mhz);
   } else
      start_overflow_timer();
#endif
#ifdef O
   return ((long long) read_virt_cycle_counter() / _papi_system_info.hw_info.mhz);
#endif
   struct rusage res;

   if ((getrusage(RUSAGE_SELF, &res) == -1))
      return (PAPI_ESYS);
   return ((res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec);
}

long long _papi_hwd_get_virt_cycles(EventSetInfo_t * zero)
{
#if 0
   if (virt_then) {
      update_global_time();
      return (virt_time);
   } else
      start_overflow_timer();
#endif
/*  return((long long)read_virt_cycle_counter());
*/
   return ((long long) _papi_hwd_get_virt_usec(zero) * _papi_system_info.hw_info.mhz);
}

void _papi_hwd_error(int error, char *where)
{
   sprintf(where, "Substrate error");
}

int _papi_hwd_init_global(void)
{
   int retval;

   /* Fill in what we can of the papi_system_info. */

   retval = get_system_info();
   if (retval)
      return (retval);

   DBG((stderr, "Found %d %s %s CPU's at %f Mhz.\n",
        _papi_system_info.hw_info.totalcpus,
        _papi_system_info.hw_info.vendor_string,
        _papi_system_info.hw_info.model_string, _papi_system_info.hw_info.mhz));

   return (PAPI_OK);
}

int _papi_hwd_init(EventSetInfo_t * zero)
{
   long arg;
   int fd;
   hwd_control_state_t *machdep = (hwd_control_state_t *) zero->machdep;

   /* Initialize our global machdep. */

#if 0
   int cpu_num, cpu_mask = 0;
   if (getsysinfo(GSI_CURRENT_CPU, (caddr_t) & cpu_num, sizeof(cpu_num), NULL, NULL, NULL)
       < 1)
      abort();
   fprintf(stderr, "CPU %d\n", cpu_num);
   cpu_mask = 1 << cpu_num;
   if (bind_to_cpu(getpid(), cpu_mask, BIND_NO_INHERIT) == -1)
      abort();
#endif

   fd = open("/dev/pfcntr", O_RDONLY | PCNTOPENALL);
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

   machdep->fd = fd;

   init_config(machdep);

   return (PAPI_OK);
}

/* Go from highest counter to lowest counter. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
   int tmp = 0, i = 1 << (_papi_system_info.num_cntrs - 1);

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
   int tmp = 0, i = _papi_system_info.num_cntrs - 1;

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

   for (i = 0; i < _papi_system_info.num_cntrs; i++) {
      useme = (1 << i) & selector;
      if (useme) {
         to->ev6 |= from[i];
      }
   }
}

int _papi_hwd_add_event(hwd_control_state_t * this_state, unsigned int EventCode,
                        EventInfo_t * out)
{
   int selector = 0;
   int avail = 0;
   long tmp_cmd[EV_MAX_COUNTERS], *codes;

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
      out->command = derived;
      out->operand_index = preset_map[preset_index].operand_index;
   } else {
      int hwcntr_num;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;    /* 0 through 7 */
      if ((hwcntr_num > _papi_system_info.num_gp_cntrs) || (hwcntr_num < 0))
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

   out->code = EventCode;
   out->selector = selector;

   return (PAPI_OK);
}

int _papi_hwd_rem_event(hwd_control_state_t * this_state, EventInfo_t * in)
{
   int selector, used, preset_index;

   /* Find out which counters used. */

   used = in->selector;

   /* Clear out counters that are part of this event. */

   this_state->selector = this_state->selector ^ used;

   return (PAPI_OK);
}

int _papi_hwd_add_prog_event(hwd_control_state_t * this_state,
                             unsigned int event, void *extra, EventInfo_t * out)
{
   return (PAPI_ESBSTR);
}

/*   if (t->model == 0)
    {
      fprintf(stderr,"Command block at %p: items 0x%x\n",t,t->items);
      fprintf(stderr,"iccsr_pc1 %d\n",t->mux.iccsr_pc1);
      fprintf(stderr,"iccsr_pc0 %d\n",t->mux.iccsr_pc0);
      fprintf(stderr,"iccsr_mux0 0x%x\n",t->mux.iccsr_mux0);
      fprintf(stderr,"iccsr_mux1 0x%x\n",t->mux.iccsr_mux1);
      fprintf(stderr,"iccsr_disable 0x%x\n",t->mux.iccsr_disable);
    }
  else if (t->model == 1)
    {
      fprintf(stderr,"
    }
  else if (t->model == 2)
    {
    }
  else
    abort();
*/

void dump_cmd(ev_control_t * t)
{
   DBG((stderr, "Command block at %p: 0x%x\n", t, t->ev6));
}

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo_t * ESI, EventSetInfo_t * zero)
{
   int i, retval;
   hwd_control_state_t *this_state = (hwd_control_state_t *) ESI->machdep;
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;
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

      DBG((stderr, "PCNT6MUX command %lx\n", current_state->counter_cmd.ev6));
      retval = ioctl(current_state->fd, PCNT6MUX, &current_state->counter_cmd.ev6);
      if (retval == -1)
         return (PAPI_ESYS);

      /* zero and restart selected counters */

      start_em.pmctrs_ev6_long = 0;
      start_em.pmctrs_ev6_cpu = PMCTRS_ALL_CPUS;
      start_em.pmctrs_ev6_select = PF6_SEL_COUNTER_0 | PF6_SEL_COUNTER_1;
      retval = ioctl(current_state->fd, PCNT6RESTART, &start_em.pmctrs_ev6_long);
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
                 (&this_state->counter_cmd, &current_state->counter_cmd, i - 1)))
               return (PAPI_ECNFLCT);
         } else
             if (!
                 (counter_event_compat
                  (&this_state->counter_cmd, &current_state->counter_cmd, i - 1)))
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
            counter_event_copy(&this_state->counter_cmd, &current_state->counter_cmd,
                               i - 1);
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

   retval = ioctl(current_state->fd, PCNT6MUX, &current_state->counter_cmd);
   if (retval == -1)
      return (PAPI_ESYS);

   return (PAPI_OK);
}

int _papi_hwd_unmerge(EventSetInfo_t * ESI, EventSetInfo_t * zero)
{
   int i, tmp, hwcntr, retval;
   hwd_control_state_t *this_state = (hwd_control_state_t *) ESI->machdep;
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;

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

int _papi_hwd_reset(EventSetInfo_t * ESI, EventSetInfo_t * zero)
{
   int i, retval;

   retval = update_global_hwcounters(zero);
   if (retval)
      return (retval);

   for (i = 0; i < _papi_system_info.num_cntrs; i++)
      ESI->hw_start[i] = zero->hw_start[i];

   return (PAPI_OK);
}

static long long handle_derived_add(int selector, long long *from)
{
   int pos;
   long long retval = 0;

   while ((pos = ffs(selector))) {
      DBG((stderr, "Compound event, adding %lld to %lld\n", from[pos - 1], retval));
      retval += from[pos - 1];
      selector ^= 1 << pos - 1;
   }
   return (retval);
}

static long long handle_derived_subtract(int operand_index, int selector, long long *from)
{
   int pos;
   long long retval = from[operand_index];

   selector = selector ^ (1 << operand_index);
   while (pos = ffs(selector)) {
      DBG((stderr, "Compound event, subtracting %lld to %lld\n", from[pos - 1], retval));
      retval -= from[pos - 1];
      selector ^= 1 << pos - 1;
   }
   return (retval);
}

static long long units_per_second(long long units, long long cycles)
{
   return ((long long) ((float) units * _papi_system_info.hw_info.mhz * 1000000.0 /
                        (float) cycles));
}

static long long handle_derived_ps(int operand_index, int selector, long long *from)
{
   int pos;

   pos = ffs(selector ^ (1 << operand_index)) - 1;
   assert(pos != 0);

   return (units_per_second(from[pos], from[operand_index]));
}

static long long handle_derived_add_ps(int operand_index, int selector, long long *from)
{
   int add_selector = selector ^ (1 << operand_index);
   long long tmp = handle_derived_add(add_selector, from);
   return (units_per_second(tmp, from[operand_index]));
}

static long long handle_derived(EventInfo_t * cmd, long long *from)
{
   switch (cmd->command) {
   case DERIVED_ADD:
      return (handle_derived_add(cmd->selector, from));
   case DERIVED_ADD_PS:
      return (handle_derived_add_ps(cmd->operand_index, cmd->selector, from));
   case DERIVED_SUB:
      return (handle_derived_subtract(cmd->operand_index, cmd->selector, from));
   case DERIVED_PS:
      return (handle_derived_ps(cmd->operand_index, cmd->selector, from));
   default:
      abort();
   }
}

int _papi_hwd_read(EventSetInfo_t * ESI, EventSetInfo_t * zero, long long *events)
{
   int shift_cnt = 0;
   int retval, selector, j = 0, i;
   long long correct[EV_MAX_COUNTERS];

   retval = update_global_hwcounters(zero);
   if (retval)
      return (retval);

   retval = correct_local_hwcounters(zero, ESI, correct);
   if (retval)
      return (retval);

   /* This routine distributes hardware counters to software counters in the
      order that they were added. Note that the higher level 
      EventInfoArray[i] entries may not be contiguous because the user
      has the right to remove an event. */

   for (i = 0; i < _papi_system_info.num_cntrs; i++) {
      selector = ESI->EventInfoArray[i].selector;
      if (selector == PAPI_NULL)
         continue;

      assert(selector != 0);
      DBG((stderr, "Event index %d, selector is 0x%x\n", j, selector));

      /* If this is not a derived event */

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
      return (set_default_domain(zero, option->domain.domain));
   case PAPI_DOMAIN:
      return (set_domain(option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DEFGRN:
      return (set_default_granularity(zero, option->granularity.granularity));
   case PAPI_GRANUL:
      return (set_granularity
              (option->granularity.ESI->machdep, option->granularity.granularity));
#if 0
   case PAPI_INHERIT:
      return (set_inherit(zero, option->inherit.inherit));
#endif
   default:
      return (PAPI_EINVAL);
   }
}

int _papi_hwd_write(EventSetInfo_t * master, EventSetInfo_t * ESI, long long events[])
{
   return (PAPI_ESBSTR);
}

int _papi_hwd_shutdown(EventSetInfo_t * zero)
{
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;
   int retval;

   if (current_state && current_state->fd) {
      retval = close(current_state->fd);
      if (retval == -1)
         return (PAPI_ESYS);
   }
   return (PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
   hwd_control_state_t *current_state = NULL;
   int retval;

   if (default_master_eventset)
      current_state = (hwd_control_state_t *) default_master_eventset->machdep;
   if (current_state && current_state->fd) {
      retval = close(current_state->fd);
      if (retval == -1)
         return (PAPI_ESYS);
   }
   return (PAPI_OK);
}

int _papi_hwd_query(int preset_index, int *flags, char **note)
{
   if (preset_map[preset_index].selector == 0)
      return (0);
   if (preset_map[preset_index].derived)
      *flags = PAPI_DERIVED;
   if (preset_map[preset_index].note)
      *note = preset_map[preset_index].note;
   return (1);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, EventSetOverflowInfo_t * overflow_option)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_EMISC);
}

int _papi_hwd_set_profile(EventSetInfo_t * ESI, EventSetProfileInfo_t * profile_option)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_EMISC);
}

int _papi_hwd_stop_profiling(EventSetInfo_t * ESI, EventSetInfo_t * master)
{
   /* This function is not used and shouldn't be called. */

   return (PAPI_EMISC);
}


void *_papi_hwd_get_overflow_address(void *context)
{
   void *location;
   struct sigcontext *info = (struct sigcontext *) context;
   location = (void *) info->sc_pc;

   return (location);
}

void _papi_hwd_lock_init(void)
{
}

#define _papi_hwd_lock(lck)	\
do				\
{				\
}while(0)

#define _papi_hwd_unlock(lck)	\
do				\
{				\
}while(0)

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, ucontext_t * info)
{
   _papi_hwi_dispatch_overflow_signal((void *) info);
}

/* Machine info structure. -1 is initialized by _papi_hwd_init. */

papi_mdi _papi_system_info =
    { "$Id$",
   1.0,                         /*  version */
   -1,                          /*  cpunum */
   {
    -1,                         /*  ncpu */
    1,                          /*  nnodes */
    -1,                         /*  totalcpus */
    -1,                         /*  vendor */
    "",                         /*  vendor string */
    -1,                         /*  model */
    "",                         /*  model string */
    0.0,                        /*  revision */
    0.0                         /*  mhz */
    },
   {
    "",
    "",
    (caddr_t) & _ftext,
    (caddr_t) & _etext,
    (caddr_t) NULL,
    (caddr_t) NULL,
    (caddr_t) NULL,
    (caddr_t) NULL,
    "_RLD_LIST",                /* How to preload libs */
    },
   {0,                          /*total_tlb_size */
    0,                          /*itlb_size */
    0,                          /*itlb_assoc */
    0,                          /*dtlb_size */
    0,                          /*dtlb_assoc */
    0,                          /*total_L1_size */
    0,                          /*L1_icache_size */
    0,                          /*L1_icache_assoc */
    0,                          /*L1_icache_lines */
    0,                          /*L1_icache_linesize */
    0,                          /*L1_dcache_size */
    0,                          /*L1_dcache_assoc */
    0,                          /*L1_dcache_lines */
    0,                          /*L1_dcache_linesize */
    0,                          /*L2_cache_size */
    0,                          /*L2_cache_assoc */
    0,                          /*L2_cache_lines */
    0,                          /*L2_cache_linesize */
    0,                          /*L3_cache_size */
    0,                          /*L3_cache_assoc */
    0,                          /*L3_cache_lines */
    0                           /*L3_cache_linesize */
    },
   -1,                          /*  num_cntrs */
   -1,                          /*  num_gp_cntrs */
   -1,                          /*  grouped_counters */
   -1,                          /*  num_sp_cntrs */
   -1,                          /*  total_presets */
   -1,                          /*  total_events */
   PAPI_DOM_USER,               /* default domain */
   PAPI_GRN_THR,                /* default granularity */
   0,                           /* We can use add_prog_event */
   0,                           /* We can write the counters */
   0,                           /* supports HW overflow */
   0,                           /* supports HW profile */
   1,                           /* supports 64 bit virtual counters */
   0,                           /* supports child inheritance */
   0,                           /* supports attaching to another process */
   1,                           /* We can use the real_usec call */
   1,                           /* We can use the real_cyc call */
   1,                           /* We can use the virt_usec call */
   1,                           /* We can use the virt_cyc call */
   0,                           /* HW read resets the counters */
   sizeof(hwd_control_state_t),
   {0}
};
