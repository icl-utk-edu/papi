/* Based on tru64-alpha.c.
 Mods for alpha-linux by Glenn Laguna, Sandia National Lab, galagun@sandia.gov
*/

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#define PF5_SEL_COUNTER_0	1       /* Op applies to counter 0 */
#define PF5_SEL_COUNTER_1	2       /* Op applies to counter 1 */
#define PF5_SEL_COUNTER_2	4       /* Op applies to counter 2 */

#define CLOCK_REALTIME 1

#define PIOC            ('q'<<8)
#define    PIOCPSINFO      (PIOC|30)    /* get ps(1) information */


extern EventSetInfo_t *default_master_eventset;

extern int HW_driver_start(int *, int);

long int model_to_proctype(char *);

static inline char *search_cpu_info(FILE *, char *, char *);


static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };


/* Globals */

static hwd_search_t findem_ev5[] = {
   {PAPI_TOT_CYC, {IPR_EVT_CYCLES, -1, -1}},
   {PAPI_TOT_IIS, {IPR_EVT_ISSUES, -1, -1}},
   {PAPI_L1_ICM, {-1, IPR_EVT_ICACHE_MISS, -1}},
   {PAPI_L1_DCM, {-1, IPR_EVT_DCACHE_MISS, -1}},
   {PAPI_L1_TCM, {-1, -1, IPR_EVT_SCACHE_MISS}},
   {PAPI_TLB_DM, {-1, -1, IPR_EVT_DTB_MISS}},
   {PAPI_TLB_IM, {-1, -1, IPR_EVT_ITB_MISS}},
   {PAPI_MEM_SCY, {-1, -1, IPR_EVT_MEM_BARRIER_CYCLES}},
   {PAPI_BR_CN, {-1, IPR_EVT_COND_BRANCHES, -1}},
   {PAPI_BR_MSP, {-1, -1, IPR_EVT_BRANCH_MISPR}},
   {PAPI_INT_INS, {-1, IPR_EVT_INTEGER_OPS, -1}},
   {PAPI_FP_INS, {-1, IPR_EVT_FLOAT_OPS, -1}},
   {PAPI_LD_INS, {-1, IPR_EVT_LOADS, -1}},
   {PAPI_SR_INS, {-1, IPR_EVT_STORES, -1}},
   {PAPI_BR_INS, {-1, IPR_EVT_BRANCHES, -1}},
   {PAPI_L1_DCA, {-1, IPR_EVT_DCACHE_ACCESS, -1}},
   {PAPI_L1_ICA, {-1, IPR_EVT_ICACHE_ACCESS, -1}},
   {PAPI_L2_TCR, {-1, IPR_EVT_SCACHE_READ, -1}},
   {PAPI_L2_TCW, {-1, IPR_EVT_SCACHE_WRITE, -1}},
   {-1, {-1, -1, -1}}
};

static hwd_search_t findem_ev67[] = {
   {PAPI_TOT_CYC, {-1, 0x0, -1}},
   {PAPI_TOT_INS, {0x0, -1, -1}},
   {PAPI_RES_STL, {-1, 0xC, -1}},
   {-1, {-1, -1, -1}}
};

static hwd_search_t findem_ev6[] = {
   {PAPI_TOT_CYC, {IPR_EVT_CYCLESA, -1, -1}},
   {PAPI_TOT_INS, {IPR_EVT_RET_INSTRUCTIONS, -1, -1}},
   {PAPI_BR_CN, {-1, IPR_EVT_RET_COND_BRANCHES, -1}},
   {PAPI_RES_STL, {-1, IPR_EVT_REPLAY_TRAP, -1}},
   {-1, {-1, -1, -1}}
};


static int setup_all_presets(int family, int model)
{
   int first, event, derived, hwnum;
   hwd_search_t *findem;
   char str[PAPI_MAX_STR_LEN];
   int num = _papi_system_info.num_gp_cntrs;
   int code;

   SUBDBG("Family %d, model %d\n", family, model);

   if (family == 2)
      findem = findem_ev6;
   else if (family == 1)
      findem = findem_ev5;
   else {
      PAPIERROR("Unknown processor family %d, model %d", family, model);
      return (PAPI_ESBSTR);
   }

   while ((code = findem->papi_code) != -1) {
      int i, index;

      index = code & PAPI_PRESET_AND_MASK;
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
         SUBDBG("Preset %d found, selector 0x%x\n",
              index, preset_map[index].selector);
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
   SUBDBG("copy: A %x B %x C %d M %x\n", al, bl, cntr, mask);

   bl = bl & mask;
   al = al | mask;
   al = al ^ mask;
   al = al & bl;
   a->ev6 = al;

   SUBDBG("A is now %x\n", al);
}

static int counter_event_shared(const ev_control_t * a, const ev_control_t * b, int cntr)
{
   long al = a->ev6;
   long bl = b->ev6;
   long mask = 0xf0 >> cntr;
   SUBDBG("shared?: A %x B %x C %d M %x\n", al, bl, cntr, mask);

   bl = bl & mask;
   al = al & mask;

   if (al == bl) {
      SUBDBG("shared!\n", al, bl);
      return (1);
   } else {
      SUBDBG("not shared!\n", al, bl);
      return (0);
   }
}

static int update_global_hwcounters(EventSetInfo_t * global)
{
   int i, retval;
   hwd_control_state_t *current_state = (hwd_control_state_t *) global->machdep;
   long *counter_values;

   counter_values = (long int *) calloc(EV_MAX_COUNTERS, sizeof(long int));

   retval = HW_driver_read(counter_values, _papi_system_info.hw_info.model);
   if (retval == -1)
      return PAPI_ESYS;

   SUBDBG("Actual values %ld %ld \n", counter_values[0], counter_values[1]);

   SUBDBG("update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",
        0,
        global->hw_start[0] + counter_values[0], global->hw_start[0], counter_values[0]);

   if (current_state->selector & 0x1) {
      SUBDBG("update_global_hwcounters() %d: G%lld = G%lld + C%lld\n", 0,
           global->hw_start[0] + counter_values[0], global->hw_start[0],
           counter_values[0]);
      global->hw_start[0] = counter_values[0];
   }

   if (current_state->selector & 0x2) {
      SUBDBG("update_global_hwcounters() %d: G%lld = G%lld + C%lld\n", 1,
           global->hw_start[1] + counter_values[1], global->hw_start[1],
           counter_values[1]);
      global->hw_start[1] = counter_values[1];
   }

   /* Clear driver counts */

   HW_driver_clear();

   free(counter_values);

   return (0);
}

static int correct_local_hwcounters(EventSetInfo_t * global, EventSetInfo_t * local,
                                    long_long *correct)
{
   int i;

   for (i = 0; i < _papi_system_info.num_cntrs; i++) {
      SUBDBG("correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n", i,
           global->hw_start[i] - local->hw_start[i], global->hw_start[i],
           local->hw_start[i]);
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
   int retval, family;
   prpsinfo_t info;
   long proc_type;
   pid_t pid;
   char *ptr;
   FILE *f;
   float mhz;
   char maxargs[PAPI_MAX_STR_LEN], *t, *s, *sproc;

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);
   _papi_hwi_system_info.pid = pid;

   if ((f = fopen("/proc/cpuinfo", "r")) == NULL)
      return (PAPI_ESYS);

   sprintf(maxargs, "/proc/%d/exe", (int) pid);
   if (readlink(maxargs, _papi_hwi_system_info.exe_info.fullname, PAPI_HUGE_STR_LEN) < 0)
     return(PAPI_ESYS);
   
   /* basename can modify it's argument */
   strcpy(maxargs,_papi_hwi_system_info.exe_info.fullname);
   strcpy(_papi_hwi_system_info.exe_info.address_info.name, basename(maxargs));

   _papi_system_info.cpunum = 0;
   _papi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
   _papi_system_info.hw_info.nnodes = 1;
   _papi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);
   _papi_system_info.hw_info.vendor = -1;

   /*proc type */
   rewind(f);
   s = search_cpu_info(f, "cpu model", maxargs);
   proc_type = model_to_proctype(s);

   /* Mhz */
   rewind(f);
   s = search_cpu_info(f, "cycle frequency", maxargs);
   if (s)
      sscanf(s + 1, "%f", &mhz);
   mhz = mhz / 1000000;
   _papi_system_info.hw_info.mhz = mhz;

/****************************/
   _papi_system_info.hw_info.vendor = -1;
   strcpy(_papi_system_info.hw_info.vendor_string, "Compaq");

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

   _papi_system_info.hw_info.model = proc_type;

   //  printf("Setting presets for family = %d, proc_type = %d\n",
   //     family, proc_type);
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


long_long _papi_hwd_get_real_usec(void)
{
#ifdef O
   return ((long_long) read_cycle_counter() / _papi_system_info.hw_info.mhz);
#endif
   struct timeval res;

   if ((gettimeofday(&res, NULL) == -1))
      return (PAPI_ESYS);
   return (res.tv_sec * 1000000) + (res.tv_usec);
}

long_long _papi_hwd_get_real_cycles(void)
{
   return ((long_long) _papi_hwd_get_real_usec() * _papi_system_info.hw_info.mhz);
}

long_long _papi_hwd_get_virt_usec(EventSetInfo_t * zero)
{
#ifdef O
   return ((long_long) read_virt_cycle_counter() / _papi_system_info.hw_info.mhz);
#endif
   struct rusage res;

   if ((getrusage(RUSAGE_SELF, &res) == -1))
      return (PAPI_ESYS);
   return ((res.ru_utime.tv_sec * 1000000) + res.ru_utime.tv_usec);
}

long_long _papi_hwd_get_virt_cycles(EventSetInfo_t * zero)
{
   return ((long_long) _papi_hwd_get_virt_usec(zero) * _papi_system_info.hw_info.mhz);
}

static void lock_init(void)
{
}

papi_svector_t _linux_alpha_table[] = {
 {(void (*)())_papi_hwd_get_overflow_address,VEC_PAPI_HWD_GET_OVERFLOW_ADDRESS},
 {(void (*)())_papi_hwd_init, VEC_PAPI_HWD_INIT},
 {(void (*)())_papi_hwd_dispatch_timer, VEC_PAPI_HWD_DISPATCH_TIMER},
 {(void (*)())_papi_hwd_ctl, VEC_PAPI_HWD_CTL},
 {(void (*)())_papi_hwd_get_real_usec, VEC_PAPI_HWD_GET_REAL_USEC},
 {(void (*)())_papi_hwd_get_real_cycles, VEC_PAPI_HWD_GET_REAL_CYCLES},
 {(void (*)())_papi_hwd_get_virt_cycles, VEC_PAPI_HWD_GET_VIRT_CYCLES},
 {(void (*)())_papi_hwd_get_virt_usec, VEC_PAPI_HWD_GET_VIRT_USEC},
 {(void (*)())_papi_hwd_read, VEC_PAPI_HWD_READ },
 {(void (*)())_papi_hwd_shutdown, VEC_PAPI_HWD_SHUTDOWN },
 {(void (*)())_papi_hwd_shutdown_global, VEC_PAPI_HWD_SHUTDOWN_GLOBAL},
 {(void (*)())_papi_hwd_reset, VEC_PAPI_HWD_RESET},
 {NULL, VEC_PAPI_END}
};

int _papi_hwd_init_substrate(papi_vectors_t *vtable)
{
   int retval;

  /* Setup the vector entries that the OS knows about */
#ifndef PAPI_NO_VECTOR
  retval = _papi_hwi_setup_vector_table( vtable, _linux_alpha_table);
  if ( retval != PAPI_OK ) return(retval);
#endif

   /* Fill in what we can of the papi_system_info. */
   retval = get_system_info();
   if (retval)
      return (retval);

   lock_init();

   SUBDBG("Found %d %s %s CPU's at %f Mhz.\n",
        _papi_system_info.hw_info.totalcpus,
        _papi_system_info.hw_info.vendor_string,
        _papi_system_info.hw_info.model_string, _papi_system_info.hw_info.mhz);

   return (PAPI_OK);
}

int _papi_hwd_init(EventSetInfo_t * zero)
{
   init_ipr_lib();
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

   if (EventCode & PAPI_PRESET_MASK) {
      int preset_index;
      int derived;

      preset_index = EventCode & PAPI_RESET_AND_MASK;

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

void dump_cmd(ev_control_t * t)
{
   SUBDBG("Command block at %p: 0x%x\n", t, t->ev6);
}

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo_t * ESI, EventSetInfo_t * zero)
{
   int retval;
   hwd_control_state_t *this_state = (hwd_control_state_t *) ESI->machdep;
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;
   int commands[3] = { 0, 0, 0 };

   current_state->selector = this_state->selector;
   memcpy(&current_state->counter_cmd, &this_state->counter_cmd,
          sizeof(current_state->counter_cmd));


   /* select events */
   SUBDBG("PCNT6MUX command %lx\n", current_state->counter_cmd.ev6);
   commands[0] = current_state->counter_cmd.ev6;
   commands[1] = commands[2] = 0;

   /* zero and restart selected counters */
   retval = HW_driver_start(commands, _papi_system_info.hw_info.model);

   return (PAPI_OK);
}


int _papi_hwd_unmerge(EventSetInfo_t * ESI, EventSetInfo_t * zero)
{
   int i, tmp, hwcntr, retval;
   hwd_control_state_t *this_state = (hwd_control_state_t *) ESI->machdep;
   hwd_control_state_t *current_state = (hwd_control_state_t *) zero->machdep;

   current_state->selector = 0;
   return (PAPI_OK);
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

static long_long handle_derived_add(int selector, long_long *from)
{
   int pos;
   long_long retval = 0;

   while ((pos = ffs(selector))) {
      SUBDBG("Compound event, adding %lld to %lld\n", from[pos - 1], retval);
      retval += from[pos - 1];
      selector ^= 1 << pos - 1;
   }
   return (retval);
}

static long_long handle_derived_subtract(int operand_index, int selector, long_long *from)
{
   int pos;
   long_long retval = from[operand_index];

   selector = selector ^ (1 << operand_index);
   while (pos = ffs(selector)) {
      SUBDBG("Compound event, subtracting %lld to %lld\n", from[pos - 1], retval);
      retval -= from[pos - 1];
      selector ^= 1 << pos - 1;
   }
   return (retval);
}

static long_long units_per_second(long_long units, long_long cycles)
{
   return ((long_long) ((float) units * _papi_system_info.hw_info.mhz * 1000000.0 /
                        (float) cycles));
}

static long_long handle_derived_ps(int operand_index, int selector, long_long *from)
{
   int pos;

   pos = ffs(selector ^ (1 << operand_index)) - 1;
   assert(pos != 0);

   return (units_per_second(from[pos], from[operand_index]));
}

static long_long handle_derived_add_ps(int operand_index, int selector, long_long *from)
{
   int add_selector = selector ^ (1 << operand_index);
   long_long tmp = handle_derived_add(add_selector, from);
   return (units_per_second(tmp, from[operand_index]));
}

static long_long handle_derived(EventInfo_t * cmd, long_long *from)
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
      PAPIERROR("BUG! Unknown derived command %d, returning 0",evi->derived);
      return((long_long)0);
   }
}

int _papi_hwd_read(EventSetInfo_t * ESI, EventSetInfo_t * zero, long_long *events, int flags)
{
   int shift_cnt = 0;
   int retval, selector, j = 0, i;
   long_long correct[EV_MAX_COUNTERS];

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
      SUBDBG("Event index %d, selector is 0x%x\n", j, selector);

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
   default:
      return (PAPI_EINVAL);
   }
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

void *_papi_hwd_get_overflow_address(void *context)
{
   void *location;
   struct sigcontext *info = (struct sigcontext *) context;
   location = (void *) info->sc_pc;

   return (location);
}

void _papi_hwd_lock(void)
{
}

void _papi_hwd_unlock(void)
{
}

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
    /* I think this is tru64 nonsense and not used in linux/*
       /*                           (caddr_t)&_ftext, */
    /*                              (caddr_t)&_etext, */
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



static inline char *search_cpu_info(FILE * f, char *search_str, char *line)
{
   /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
   /* See the PCL home page for the German version of PAPI. */

   char *s;

   while (fgets(line, 256, f) != NULL) {
      if (strstr(line, search_str) != NULL) {
         /* ignore all characters in line up to : */
         for (s = line; *s && (*s != ':'); ++s);
         if (*s)
            return (s);
      }
   }
   return (NULL);

   /* End stolen code */
}


long int model_to_proctype(char *model)
{
   if (strstr(model, "EV67"))
      return EV67_CPU;
   else if (strstr(model, "EV56"))
      return EV56_CPU;
   else if (strstr(model, "EV6"))
      return EV6_CPU;
   else {
      printf("Unknown cpu\n");
      return -1;
   }
}
