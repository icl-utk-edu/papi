/* 
* File:    solaris-ultra.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
* Mods:    Larry Meadows(helped us to build the native table dynamically)  
*              
*/

/* to understand this program, first you should read the user's manual
   about UltraSparc II and UltraSparc III, then the man pages
   about cpc_take_sample(cpc_event_t *event)
*/

#include "papi.h"
#include "papi_internal.h"

#ifdef CPC_ULTRA3_I
#define LASTULTRA3 CPC_ULTRA3_I
#else
#define LASTULTRA3 CPC_ULTRA3_PLUS
#endif

#define MAX_ENAME 40

static void action(void *arg, int regno, const char *name, uint8_t bits);

/* Probably could dispense with this and just use native_table */
typedef struct ctr_info {
    char *name;	/* Counter name */
    int bits[2];	/* bits for register */
    int bitmask; /* 1 = pic0; 2 = pic1; 3 = both */
} ctr_info_t;

typedef struct einfo {
    unsigned int papi_event;
    char *event_str;
} einfo_t;
static einfo_t us3info[] = {
   {PAPI_FP_INS, "FA_pipe_completion+FM_pipe_completion"},
   {PAPI_FAD_INS, "FA_pipe_completion"},
   {PAPI_FML_INS, "FM_pipe_completion"},
   {PAPI_TLB_IM, "ITLB_miss"},
   {PAPI_TLB_DM, "DTLB_miss"},
   {PAPI_TOT_CYC, "Cycle_cnt"},
   {PAPI_TOT_IIS, "Instr_cnt"},
   {PAPI_TOT_INS, "Instr_cnt"},
   {PAPI_L2_TCM, "EC_misses"},
   {PAPI_L2_ICM, "EC_ic_miss"},
   {PAPI_L1_ICM, "IC_miss"},
   {PAPI_L1_LDM, "DC_rd_miss"},
   {PAPI_L1_STM, "DC_wr_miss"},
   {PAPI_BR_MSP, "IU_Stat_Br_miss_taken+IU_Stat_Br_miss_untaken"},
   {PAPI_L1_DCR, "DC_rd"},
   {PAPI_L1_DCW, "DC_wr"},
   {PAPI_L1_ICH, "IC_ref"},	/* Is this really hits only? */
   {PAPI_L1_ICA, "IC_ref+IC_miss"},	/* Ditto? */
   {PAPI_L2_TCH, "EC_ref-EC_misses"},
   {PAPI_L2_TCA, "EC_ref"},
};

static einfo_t us2info[] = {
   {PAPI_L1_ICM, "IC_ref-IC_hit"},
   {PAPI_L2_TCM, "EC_ref-EC_hit"},
   {PAPI_CA_SNP, "EC_snoop_cb"},
   {PAPI_CA_INV, "EC_snoop_inv"},
   {PAPI_L1_LDM, "DC_rd-DC_rd_hit"},
   {PAPI_L1_STM, "DC_wr-DC_wr_hit"},
   {PAPI_BR_MSP, "Dispatch0_mispred"},
   {PAPI_TOT_IIS, "Instr_cnt"},
   {PAPI_TOT_INS, "Instr_cnt"},
   {PAPI_LD_INS, "DC_rd"},
   {PAPI_SR_INS, "DC_wr"},
   {PAPI_TOT_CYC, "Cycle_cnt"},
   {PAPI_L1_DCR, "DC_rd"},
   {PAPI_L1_DCW, "DC_wr"},
   {PAPI_L1_ICH, "IC_hit"},
   {PAPI_L2_ICH, "EC_ic_hit"},
   {PAPI_L1_ICA, "IC_ref"},
   {PAPI_L2_TCH, "EC_hit"},
   {PAPI_L2_TCA, "EC_ref"},
};

static native_info_t *native_table;
static hwi_search_t *preset_table;

static struct ctr_info *ctrs;
static int ctr_size;
static int nctrs;

static int build_tables(void);
static void add_preset(hwi_search_t *tab, int *np, einfo_t e);

/* Globals used to access the counter registers. */

static int cpuver;
static int pcr_shift[2];

#if 0
/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */
   /* Phil says this is false */

/* the number in this preset_search map table is the native event index 
   in the native event table, when it ORs the PAPI_NATIVE_MASK, it becomes the
   native event code. 
*/
/* UltraSparc II preset search table */
hwi_search_t usii_preset_search_map[] = {
   /* L1 Cache Imisses */
   {PAPI_L1_ICM, {DERIVED_SUB, {PAPI_NATIVE_MASK | 4, PAPI_NATIVE_MASK | 14}}},
   /* L2 Total Cache misses */
   {PAPI_L2_TCM, {DERIVED_SUB, {PAPI_NATIVE_MASK | 8, PAPI_NATIVE_MASK | 18}}},
   /* Req. for snoop */
   {PAPI_CA_SNP, {0, {PAPI_NATIVE_MASK | 20, PAPI_NULL}}},
   /* Req. invalidate cache line */
   {PAPI_CA_INV, {0, {PAPI_NATIVE_MASK | 10, PAPI_NULL}}},
   /* L1LM */
   {PAPI_L1_LDM, {DERIVED_SUB, {PAPI_NATIVE_MASK | 5, PAPI_NATIVE_MASK | 15}}},
   /* L1SM */
   {PAPI_L1_STM, {DERIVED_SUB, {PAPI_NATIVE_MASK | 6, PAPI_NATIVE_MASK | 16}}},
   /* Cond. branch inst. mispred. */
   {PAPI_BR_MSP, {0, {PAPI_NATIVE_MASK | 12, PAPI_NULL}}},
   /* Total inst. issued */
   {PAPI_TOT_IIS, {0, {PAPI_NATIVE_MASK | 1, PAPI_NULL}}},
   /* Total inst. executed */
   {PAPI_TOT_INS, {0, {PAPI_NATIVE_MASK | 1, PAPI_NULL}}},
   /* Loads executed */
   {PAPI_LD_INS, {0, {PAPI_NATIVE_MASK | 5, PAPI_NULL}}},
   /* Stores executed */
   {PAPI_SR_INS, {0, {PAPI_NATIVE_MASK | 6, PAPI_NULL}}},
   /* Total cycles */
   {PAPI_TOT_CYC, {0, {PAPI_NATIVE_MASK | 0, PAPI_NULL}}},
   /* L1 data cache reads */
   {PAPI_L1_DCR, {0, {PAPI_NATIVE_MASK | 5, PAPI_NULL}}},
   /* L1 data cache writes */
   {PAPI_L1_DCW, {0, {PAPI_NATIVE_MASK | 6, PAPI_NULL}}},
   /* L1 instruction cache hits */
   {PAPI_L1_ICH, {0, {PAPI_NATIVE_MASK | 14, PAPI_NULL}}},
   /* L2 instruction cache hits */
   {PAPI_L2_ICH, {0, {PAPI_NATIVE_MASK | 21, PAPI_NULL}}},
   /* L1 instruction cache accesses */
   {PAPI_L1_ICA, {0, {PAPI_NATIVE_MASK | 4, PAPI_NULL}}},
   /* L2 total cache hits */
   {PAPI_L2_TCH, {0, {PAPI_NATIVE_MASK | 18, PAPI_NULL}}},
   /* L2 total cache accesses */
   {PAPI_L2_TCA, {0, {PAPI_NATIVE_MASK | 8, PAPI_NULL}}},
   /* Terminator */
   {0, {0, {0, 0}}}
};

/* UltraSparc III preset search table */
hwi_search_t usiii_preset_search_map[] = {
   /* Floating point instructions */
   {PAPI_FP_INS, {DERIVED_ADD, {PAPI_NATIVE_MASK | 22, PAPI_NATIVE_MASK | 68}}},
   /* pic0 FA_pipe_completion and pic1 FM_pipe_completion */
   /* Floating point add instructions */
   {PAPI_FAD_INS, {0, {PAPI_NATIVE_MASK | 22, PAPI_NULL}}},  /* pic0 FA_pipe_completion */
   /* Floating point multiply instructions */
   {PAPI_FML_INS, {0, {PAPI_NATIVE_MASK | 68, PAPI_NULL}}},/* pic1 FM_pipe_completion */
   /* ITLB */
   {PAPI_TLB_IM, {0, {PAPI_NATIVE_MASK | 47, PAPI_NULL}}},/* pic1 ITLB_miss */
   /* DITLB */
   {PAPI_TLB_DM, {0, {PAPI_NATIVE_MASK | 48, PAPI_NULL}}},/* pic1 DTLB_miss */
   /* Total cycles */
   {PAPI_TOT_CYC, {0, {PAPI_NATIVE_MASK | 0, PAPI_NULL}}},/* pic0 and pic1 Cycle_cnt */
   /* Total inst. issued */
   {PAPI_TOT_IIS, {0, {PAPI_NATIVE_MASK | 1, PAPI_NULL}}},   /* pic0 and pic1 Instr_cnt */
   /* Total inst. executed */
   {PAPI_TOT_INS, {0, {PAPI_NATIVE_MASK | 1, PAPI_NULL}}},   /* pic0 and pic1 Instr_cnt */
   /* L2 Total Cache misses */
   {PAPI_L2_TCM, {0, {PAPI_NATIVE_MASK | 42, PAPI_NULL}}},   /* pic1 EC_misses */
   /* L2 Total ICache misses */
   {PAPI_L2_ICM, {0, {PAPI_NATIVE_MASK | 45, PAPI_NULL}}},   /* pic1 EC_ic_miss */
   /* L1 Total ICache misses */
   {PAPI_L1_ICM, {0, {PAPI_NATIVE_MASK | 38, PAPI_NULL}}},   /* pic1 IC_miss (actually hits) */
   /* L1 Load Misses */
   {PAPI_L1_LDM, {0, {PAPI_NATIVE_MASK | 39, PAPI_NULL}}},   /* pic1 DC_rd_miss */
   /* L1 Store Misses */
   {PAPI_L1_STM, {0, {PAPI_NATIVE_MASK | 40, PAPI_NULL}}},   /* pic1 DC_wr_miss */
   /* Cond. branch inst. mispred. */
   {PAPI_BR_MSP, {0, {PAPI_NATIVE_MASK | 32, PAPI_NULL}}},   /* pic1 Dispatch0_mispred */
   /* pic0 Cycle_cnt, pic1 Instr_cnt */
   /* L1 data cache reads */
   {PAPI_L1_DCR, {0, {PAPI_NATIVE_MASK | 8, PAPI_NULL}}},    /* pic0 DC_rd */
   /* L1 data cache writes */
   {PAPI_L1_DCW, {0, {PAPI_NATIVE_MASK | 9, PAPI_NULL}}},    /* pic0 DC_wr */
   /* L1 instruction cache hits */
   {PAPI_L1_ICH, {0, {PAPI_NATIVE_MASK | 7, PAPI_NULL}}},    /* pic0 IC_ref (actually hits only) */
   /* L1 instruction cache accesses */
   {PAPI_L1_ICA, {DERIVED_ADD, {PAPI_NATIVE_MASK | 7, PAPI_NATIVE_MASK | 38}}},
   /* pic0 IC_ref (actually hits only) + pic1 IC_miss */
   /* L2 total cache hits */
   {PAPI_L2_TCH, {DERIVED_SUB, {PAPI_NATIVE_MASK | 10, PAPI_NATIVE_MASK | 42}}},
   /* pic0 EC_ref - pic1 EC_misses */
   /* L2 total cache accesses */
   {PAPI_L2_TCA, {0, {PAPI_NATIVE_MASK | 10, PAPI_NULL}}},   /* pic0 EC_ref */
   /* Terminator */
   {0, {0, {0, 0}}}
};

/* the encoding array in native_info_t is the encodings for PCR.SL
   and PCR.SU, encoding[0] is for PCR.SL and encoding[1] is for PCR.SU,
   the value -1 means it is not supported by the corresponding Performance
   Instrumentation Counter register. For example, Cycle_cnt can be counted
   by PICL and PICU, but Dispatch0_IC_miss can be only counted by PICL.
   These encoding information will be used to allocate register to events
   and update the control structure.
*/
/* UltraSparc II native event table */
native_info_t usii_native_table[] = {
/* 0  */ {"Cycle_cnt", {0x0, 0x0}},
/* 1  */ {"Instr_cnt", {0x1, 0x1}},
/* 2  */ {"Dispatch0_IC_miss", {0x2, -1}},
/* 3  */ {"Dispatch0_storeBuf", {0x3, -1}},
/* 4  */ {"IC_ref", {0x8, -1}},
/* 5  */ {"DC_rd", {0x9, -1}},
/* 6  */ {"DC_wr", {0xa, -1}},
/* 7  */ {"Load_use", {0xb, -1}},
/* 8  */ {"EC_ref", {0xc, -1}},
/* 9  */ {"EC_write_hit_RDO", {0xd, -1}},
/* 10 */ {"EC_snoop_inv", {0xe, -1}},
/* 11 */ {"EC_rd_hit", {0xf, -1}},
/* 12 */ {"Dispatch0_mispred", {-1, 0x2}},
/* 13 */ {"Dispatch0_FP_use", {-1, 0x3}},
/* 14 */ {"IC_hit", {-1, 0x8}},
/* 15 */ {"DC_rd_hit", {-1, 0x9}},
/* 16 */ {"DC_wr_hit", {-1, 0xa}},
/* 17 */ {"Load_use_RAW", {-1, 0xb}},
/* 18 */ {"EC_hit", {-1, 0xc}},
/* 19 */ {"EC_wb", {-1, 0xd}},
/* 20 */ {"EC_snoop_cb", {-1, 0xe}},
/* 21 */ {"EC_ic_hit", {-1, 0xf}}
};

/* UltraSparc III native event table */
native_info_t usiii_native_table[] = {
/* 0  */ {"Cycle_cnt", {0x0, 0x0}},
/* 1  */ {"Instr_cnt", {0x1, 0x1}},
/* 2  */ {"Dispatch0_IC_miss", {0x2, -1}},
/* 3  */ {"Dispatch0_br_target", {0x3, -1}},
/* 4  */ {"Dispatch0_2nd_br", {0x4, -1}},
/* 5  */ {"Rstall_storeQ", {0x5, -1}},
/* 6  */ {"Rstall_IU_use", {0x6, -1}},
/* 7  */ {"IC_ref", {0x8, -1}},
/* 8  */ {"DC_rd", {0x9, -1}},
/* 9  */ {"DC_wr", {0xa, -1}},
/* 10 */ {"EC_ref", {0xc, -1}},
/* 11 */ {"EC_write_hit_RTO", {0xd, -1}},
/* 12 */ {"EC_snoop_inv", {0xe, -1}},
/* 13 */ {"EC_rd_miss", {0xf, -1}},
/* 14 */ {"PC_port0_rd", {0x10, -1}},
/* 15 */ {"SI_snoop", {0x11, -1}},
/* 16 */ {"SI_ciq_flow", {0x12, -1}},
/* 17 */ {"SI_owned", {0x13, -1}},
/* 18 */ {"SW_count0", {0x14, -1}},
/* 19 */ {"IU_Stat_Br_miss_taken", {0x15, -1}},
/* 20 */ {"IU_Stat_Br_count_taken", {0x16, -1}},
/* 21 */ {"Dispatch_rs_mispred", {0x17, -1}},
/* 22 */ {"FA_pipe_completion", {0x18, -1}},
/* 23 */ {"EC_wb_remote", {0x19, -1}},
/* 24 */ {"EC_miss_local", {0x1a, -1}},
/* 25 */ {"EC_miss_mtag_remote", {0x1b, -1}},
/* 26 */ {"MC_reads_0", {0x20, -1}},
/* 27 */ {"MC_reads_1", {0x21, -1}},
/* 28 */ {"MC_reads_2", {0x22, -1}},
/* 29 */ {"MC_reads_3", {0x23, -1}},
/* 30 */ {"MC_stalls_0", {0x24, -1}},
/* 31 */ {"MC_stalls_2", {0x25, -1}},
/* 32 */ {"Dispatch0_mispred", {-1, 0x2}},
/* 33 */ {"IC_miss_cancelled", {-1, 0x3}},
/* 34 */ {"Re_DC_missovhd", {-1, 0x4}},
/* 35 */ {"Re_FPU_bypass", {-1, 0x5}},
/* 36 */ {"Re_DC_miss", {-1, 0x6}},
/* 37 */ {"Re_EC_miss", {-1, 0x7}},
/* 38 */ {"IC_miss", {-1, 0x8}},
/* 39 */ {"DC_rd_miss", {-1, 0x9}},
/* 40 */ {"DC_wr_miss", {-1, 0xa}},
/* 41 */ {"Rstall_FP_use", {-1, 0xb}},
/* 42 */ {"EC_misses", {-1, 0xc}},
/* 43 */ {"EC_wb", {-1, 0xd}},
/* 44 */ {"EC_snoop_cb", {-1, 0xe}},
/* 45 */ {"EC_ic_miss", {-1, 0xf}},
/* 46 */ {"Re_PC_miss", {-1, 0x10}},
/* 47 */ {"ITLB_miss", {-1, 0x11}},
/* 48 */ {"DTLB_miss", {-1, 0x12}},
/* 49 */ {"WC_miss", {-1, 0x13}},
/* 50 */ {"WC_snoop_cb", {-1, 0x14}},
/* 51 */ {"WC_scrubbed", {-1, 0x15}},
/* 52 */ {"WC_wb_wo_read", {-1, 0x16}},
/* 53 */ {"PC_soft_hit", {-1, 0x18}},
/* 54 */ {"PC_snoop_inv", {-1, 0x19}},
/* 55 */ {"PC_hard_hit", {-1, 0x1a}},
/* 56 */ {"PC_port1_rd", {-1, 0x1b}},
/* 57 */ {"SW_count1", {-1, 0x1c}},
/* 58 */ {"IU_Stat_Br_miss_untaken", {-1, 0x1d}},
/* 59 */ {"IU_Stat_Br_count_untaken", {-1, 0x1e}},
/* 60 */ {"PC_MS_miss", {-1, 0x1f}},
/* 61 */ {"MC_writes_0", {-1, 0x20}},
/* 62 */ {"MC_writes_1", {-1, 0x21}},
/* 63 */ {"MC_writes_2", {-1, 0x22}},
/* 64 */ {"MC_writes_3", {-1, 0x23}},
/* 65 */ {"MC_stalls_1", {-1, 0x24}},
/* 66 */ {"MC_stalls_3", {-1, 0x25}},
/* 67 */ {"Re_RAW_miss", {-1, 0x26}},
/* 68 */ {"FM_pipe_completion", {-1, 0x27}},
/* 69 */ {"EC_miss_mtag_remote", {-1, 0x28}},
/* 70 */ {"EC_miss_remote", {-1, 0x29}}
};
#endif

extern papi_mdi_t _papi_hwi_system_info;

hwi_search_t *preset_search_map;
/*static native_info_t *native_table;*/

#ifdef DEBUG
static void dump_cmd(papi_cpc_event_t * t)
{
   SUBDBG("cpc_event_t.ce_cpuver %d\n", t->cmd.ce_cpuver);
   SUBDBG("ce_tick %llu\n", t->cmd.ce_tick);
   SUBDBG("ce_pic[0] %llu ce_pic[1] %llu\n", t->cmd.ce_pic[0], t->cmd.ce_pic[1]);
   SUBDBG("ce_pcr 0x%llx\n", t->cmd.ce_pcr);
   SUBDBG("flags %x\n", t->flags);
}
#endif

static void dispatch_emt(int signal, siginfo_t * sip, void *arg)
{
   int event_counter;
   _papi_hwi_context_t ctx;

   ctx.si = sip;
   ctx.ucontext = arg;

   SUBDBG("%d, %p, %p\n",signal,sip,arg);

   if (sip->si_code == EMT_CPCOVF) {
      papi_cpc_event_t *sample;
      EventSetInfo_t *ESI;
      ThreadInfo_t *thread = NULL;
      int t, overflow_vector, readvalue;

      thread = _papi_hwi_lookup_thread();
      ESI = (EventSetInfo_t *) thread->running_eventset;

      if ((ESI == NULL) || ((ESI->state & PAPI_OVERFLOWING) == 0))
	{
	  OVFDBG("Either no eventset or eventset not set to overflow.\n");
	  return;
	}

      if (ESI->master != thread)
	{
	  PAPIERROR("eventset->thread 0x%lx vs. current thread 0x%lx mismatch",ESI->master,thread);
	  return;
	}

      event_counter = ESI->overflow.event_counter;
      sample = &(ESI->machdep.counter_cmd);

      /* GROSS! This is a hack to 'push' the correct values 
         back into the hardware, such that when PAPI handles
         the overflow and reads the values, it gets the correct ones.
       */

      /* Find which HW counter overflowed */

      if (ESI->EventInfoArray[ESI->overflow.EventIndex[0]].pos[0] == 0)
         t = 0;
      else
         t = 1;

      if (cpc_take_sample(&sample->cmd) == -1)
         return;
      if (event_counter == 1) {
         /* only one event is set to be the overflow monitor */ 

         /* generate the overflow vector */
         overflow_vector = 1 << t;
         /* reset the threshold */
         sample->cmd.ce_pic[t] = UINT64_MAX - ESI->overflow.threshold[0];
      } else {
         /* two events are set to be the overflow monitors */ 
         overflow_vector = 0;
         readvalue = sample->cmd.ce_pic[0];
         if (readvalue >= 0) {
            /* the first counter overflowed */

            /* generate the overflow vector */
            overflow_vector = 1;
            /* reset the threshold */
            if (t == 0)
               sample->cmd.ce_pic[0] = UINT64_MAX - ESI->overflow.threshold[0];
            else
               sample->cmd.ce_pic[0] = UINT64_MAX - ESI->overflow.threshold[1];
         }
         readvalue = sample->cmd.ce_pic[1];
         if (readvalue >= 0) {
            /* the second counter overflowed */

            /* generate the overflow vector */
            overflow_vector ^= 1 << 1;
            /* reset the threshold */
            if (t == 0)
               sample->cmd.ce_pic[1] = UINT64_MAX - ESI->overflow.threshold[1];
            else
               sample->cmd.ce_pic[1] = UINT64_MAX - ESI->overflow.threshold[0];
         }
         SUBDBG("overflow_vector, = %d\n", overflow_vector);
         /* something is wrong here */
         if (overflow_vector == 0)
	   {
	     PAPIERROR("BUG! overflow_vector is 0, dropping interrupt");
	     return;
	   }
      }

      /* Call the regular overflow function in extras.c */
      _papi_hwi_dispatch_overflow_signal(&ctx, 
           _papi_hwi_system_info.supports_hw_overflow, overflow_vector, 0, &thread);

#if DEBUG
      dump_cmd(sample);
#endif
      /* push back the correct values and start counting again*/
      if (cpc_bind_event(&sample->cmd, sample->flags) == -1)
         return;
   } else {
      SUBDBG("dispatch_emt() dropped, si_code = %d\n", sip->si_code);
      return;
   }
}

static int scan_prtconf(char *cpuname, int len_cpuname, int *hz, int *ver)
{
   /* This code courtesy of our friends in Germany. Thanks Rudolph Berrendorf! */
   /* See the PCL home page for the German version of PAPI. */
   /* Modified by Nils Smeds, all new bugs are my fault */
   /*    The routine now looks for the first "Node" with the following: */
   /*           "device_type"     = 'cpu'                    */
   /*           "name"            = (Any value)              */
   /*           "sparc-version"   = (Any value)              */
   /*           "clock-frequency" = (Any value)              */
   int ihz, version;
   char line[256], cmd[80], name[256];
   FILE *f = NULL;
   char cmd_line[PATH_MAX+PATH_MAX], fname[L_tmpnam];
   unsigned int matched;

   /*??? system call takes very long */
   /* get system configuration and put output into file */

   tmpnam(fname);
   SUBDBG("Temporary name %s\n",fname);

   sprintf(cmd_line, "/usr/sbin/prtconf -vp > %s",fname);
   SUBDBG("Executing %s\n",cmd_line);
   if (system(cmd_line) == -1) {
      remove(fname);
      return -1;
   }

   f = fopen(fname, "r");
   /* open output file */
   if (f == NULL) {
      remove(fname);
      return -1;
   }

   /* ignore all lines until we reach something with a sparc line */
   matched = 0x0;
   ihz = -1;
   while (fgets(line, 256, f) != NULL) {
      /*SUBDBG(">>> %s",line);*/ 
      if ((sscanf(line, "%s", cmd) == 1)
          && strstr(line, "Node 0x")) {
         matched = 0x0;
         /*SUBDBG("Found 'Node' -- search reset. (0x%2.2x)\n",matched);*/ 
      } else {
         if (strstr(cmd, "device_type:") && strstr(line, "'cpu'")) {
            matched |= 0x1;
            SUBDBG("Found 'cpu'. (0x%2.2x)\n",matched);
         } else if (!strcmp(cmd, "sparc-version:") &&
                    (sscanf(line, "%s %x", cmd, &version) == 2)) {
            matched |= 0x2;
            SUBDBG("Found version=%d. (0x%2.2x)\n", version, matched); 
         } else if (!strcmp(cmd, "clock-frequency:") &&
                    (sscanf(line, "%s %x", cmd, &ihz) == 2)) {
            matched |= 0x4;
            SUBDBG("Found ihz=%d. (0x%2.2x)\n", ihz,matched);
         } else if (!strcmp(cmd, "name:") && (sscanf(line, "%s %s", cmd, name) == 2)) {
            matched |= 0x8;
            SUBDBG("Found name: %s. (0x%2.2x)\n", name,matched); 
         }
      }
      if ((matched & 0xF) == 0xF)
         break;
   }
   SUBDBG("Parsing found name=%s, speed=%dHz, version=%d\n", name, ihz, version);

   if (matched ^ 0x0F)
      ihz = -1;
   else {
      *hz = (float) ihz;
      *ver = version;
      strncpy(cpuname, name, len_cpuname);
   }

   return ihz;

   /* End stolen code */
}

static int set_domain(hwd_control_state_t * this_state, int domain)
{
   papi_cpc_event_t *command = &this_state->counter_cmd;
   cpc_event_t *event = &command->cmd;
   uint64_t pcr = event->ce_pcr;
   int did = 0;

   pcr = pcr | 0x7;
   pcr = pcr ^ 0x7;
   if (domain & PAPI_DOM_USER) {
      pcr = pcr | 1 << CPC_ULTRA_PCR_USR;
      did = 1;
   }
   if (domain & PAPI_DOM_KERNEL) {
      pcr = pcr | 1 << CPC_ULTRA_PCR_SYS;
      did = 1;
   }
   /* DOMAIN ERROR */
   if (!did)
      return (PAPI_EINVAL);

   event->ce_pcr = pcr;

   return (PAPI_OK);
}

static int set_granularity(hwd_control_state_t * this_state, int domain)
{
   switch (domain) {
   case PAPI_GRN_PROCG:
   case PAPI_GRN_SYS:
   case PAPI_GRN_SYS_CPU:
   case PAPI_GRN_PROC:
      return(PAPI_ESBSTR);
   case PAPI_GRN_THR:
      break;
   default:
      return (PAPI_EINVAL);
   }
   return (PAPI_OK);
}

/* Utility functions */

/* This is a wrapper arount fprintf(stderr,...) for cpc_walk_events() */
void print_walk_names(void *arg, int regno, const char *name, uint8_t bits)
{
   SUBDBG(arg, regno, name, bits);
}

static char * getbasename(char *fname)
{
    char *temp;
 
    temp = strrchr(fname, '/');
    if( temp == NULL) return fname;
       else return temp+1;
}

static int get_system_info(void)
{
   int retval;
   pid_t pid;
   char maxargs[PAPI_MAX_STR_LEN] = "<none>";
   psinfo_t psi;
   int fd;
   int i, hz, version;
   char cpuname[PAPI_MAX_STR_LEN], pname[PATH_MAX];
   const char *name;

   /* Check counter access */

   if (cpc_version(CPC_VER_CURRENT) != CPC_VER_CURRENT)
      return (PAPI_ESBSTR);
   SUBDBG("CPC version %d successfully opened\n", CPC_VER_CURRENT);

   if (cpc_access() == -1)
      return (PAPI_ESBSTR);

   /* Global variable cpuver */

   cpuver = cpc_getcpuver();
   SUBDBG("Got %d from cpc_getcpuver()\n", cpuver);
   if (cpuver == -1)
      return (PAPI_ESBSTR);

#ifdef DEBUG
   {
      if (ISLEVEL(DEBUG_SUBSTRATE)) {
         name = cpc_getcpuref(cpuver);
         if (name)
            SUBDBG("CPC CPU reference: %s\n", name);
         else
            SUBDBG("Could not get a CPC CPU reference\n");

         for (i = 0; i < cpc_getnpic(cpuver); i++) {
	   SUBDBG("\n%6s %-40s %8s\n", "Reg", "Symbolic name", "Code");
            cpc_walk_names(cpuver, i, "%6d %-40s %02x\n", print_walk_names);
         }
         SUBDBG("\n");
      }
   }
#endif


   /* Initialize other globals */

   if ((retval = build_tables()) != PAPI_OK)
      return retval;

   preset_search_map = preset_table;
   if (cpuver <= CPC_ULTRA2) {
      SUBDBG("cpuver (==%d) <= CPC_ULTRA2 (==%d)\n", cpuver, CPC_ULTRA2);
      pcr_shift[0] = CPC_ULTRA_PCR_PIC0_SHIFT;
      pcr_shift[1] = CPC_ULTRA_PCR_PIC1_SHIFT;
      _papi_hwi_system_info.supports_hw_overflow = 0;
      _papi_hwi_system_info.using_hw_overflow = 0;
   } else if (cpuver <= LASTULTRA3) {
      SUBDBG("cpuver (==%d) <= CPC_ULTRA3x (==%d)\n", cpuver, LASTULTRA3);
      pcr_shift[0] = CPC_ULTRA_PCR_PIC0_SHIFT;
      pcr_shift[1] = CPC_ULTRA_PCR_PIC1_SHIFT;
      _papi_hwi_system_info.supports_hw_overflow = 1;
      _papi_hwi_system_info.using_hw_overflow = 1;
   } else
      return (PAPI_ESBSTR);

   /* Path and args */

   pid = getpid();
   if (pid == -1)
      return (PAPI_ESYS);

   /* Turn on microstate accounting for this process and any LWPs. */

   sprintf(maxargs, "/proc/%d/ctl", (int) pid);
   if ((fd = open(maxargs, O_WRONLY)) == -1)
      return (PAPI_ESYS);
   {
      int retval;
      struct {
         long cmd;
         long flags;
      } cmd;
      cmd.cmd = PCSET;
      cmd.flags = PR_MSACCT | PR_MSFORK;
      retval = write(fd, &cmd, sizeof(cmd));
      close(fd);
      SUBDBG("Write PCSET returned %d\n", retval);
      if (retval != sizeof(cmd))
         return (PAPI_ESYS);
   }

   /* Get executable info */

   sprintf(maxargs, "/proc/%d/psinfo", (int) pid);
   if ((fd = open(maxargs, O_RDONLY)) == -1)
      return (PAPI_ESYS);
   read(fd, &psi, sizeof(psi));
   close(fd);

   /* Cut off any arguments to exe */
   {
     char *tmp;
     tmp = strchr(psi.pr_psargs, ' ');
     if (tmp != NULL)
       *tmp = '\0';
   }

   if (realpath(psi.pr_psargs,pname))
     strncpy(_papi_hwi_system_info.exe_info.fullname, pname, PAPI_HUGE_STR_LEN);
   else
     strncpy(_papi_hwi_system_info.exe_info.fullname, psi.pr_psargs, PAPI_HUGE_STR_LEN);

   /* please don't use pr_fname here, because it can only store less that 
      16 characters */
   strcpy(_papi_hwi_system_info.exe_info.address_info.name,basename(_papi_hwi_system_info.exe_info.fullname));

   SUBDBG("Full Executable is %s\n", _papi_hwi_system_info.exe_info.fullname);

   /* Executable regions, reading /proc/pid/maps file */
   retval = _papi_hwd_update_shlib_info();

   /* Hardware info */

   _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
   _papi_hwi_system_info.hw_info.nnodes = 1;
   _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);

   retval = scan_prtconf(cpuname, PAPI_MAX_STR_LEN, &hz, &version);
   if (retval == -1)
      return (PAPI_ESBSTR);

   strcpy(_papi_hwi_system_info.hw_info.model_string, cpc_getcciname(cpuver));
   _papi_hwi_system_info.hw_info.model = cpuver;
   strcpy(_papi_hwi_system_info.hw_info.vendor_string, "SUN unknown");
   _papi_hwi_system_info.hw_info.vendor = -1;
   _papi_hwi_system_info.hw_info.revision = version;

   _papi_hwi_system_info.hw_info.mhz = ((float) hz / 1.0e6);
   SUBDBG("hw_info.mhz = %f\n", _papi_hwi_system_info.hw_info.mhz);

   /* Number of PMCs */

   retval = cpc_getnpic(cpuver);
   if (retval < 1)
      return (PAPI_ESBSTR);
   _papi_hwi_system_info.num_gp_cntrs = retval;
   _papi_hwi_system_info.num_cntrs = retval;
   SUBDBG("num_cntrs = %d\n", _papi_hwi_system_info.num_cntrs);

   /* program text segment, data segment  address info */
/*
   _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t) & _start;
   _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t) & _etext;
   _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t) & _etext + 1;
   _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t) & _edata;
   _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t) & _edata + 1;
   _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t) & _end;
*/

   /* Setup presets */

   retval = _papi_hwi_setup_all_presets(preset_search_map, NULL);
   if (retval)
      return (retval);

   return (PAPI_OK);
}


static int
build_tables(void)
{
    int i;
    int regno;
    int npic;
    einfo_t *ep;
    int n;
    int npresets;

    npic = cpc_getnpic(cpuver);
    nctrs = 0;
    for (regno = 0; regno < npic; ++regno) {
	cpc_walk_names(cpuver, regno, 0, action);
    }
    SUBDBG("%d counters\n", nctrs);
    if ((ctrs = malloc(nctrs*sizeof(struct ctr_info))) == 0) {
	return PAPI_ENOMEM;
    }
    nctrs = 0;
    for (regno = 0; regno < npic; ++regno) {
	cpc_walk_names(cpuver, regno, (void *)1, action);
    }
    SUBDBG("%d counters\n", nctrs);
#if DEBUG
    if (ISLEVEL(DEBUG_SUBSTRATE)) {
    for (i = 0; i < nctrs; ++i) {
	SUBDBG("%s: bits (%x,%x) pics %x\n", ctrs[i].name, ctrs[i].bits[0],
	    ctrs[i].bits[1],
	    ctrs[i].bitmask);
    } }
#endif
    /* Build the native event table */
    if ((native_table = malloc(nctrs*sizeof(native_info_t))) == 0) {
	free(ctrs);
	return PAPI_ENOMEM;
    }
    for (i = 0; i < nctrs; ++i) {
	native_table[i].name[39] = 0;
	strncpy(native_table[i].name, ctrs[i].name, 39);
	if (ctrs[i].bitmask&1)
	    native_table[i].encoding[0] = ctrs[i].bits[0];
	else
	    native_table[i].encoding[0] = -1;
	if (ctrs[i].bitmask&2)
	    native_table[i].encoding[1] = ctrs[i].bits[1];
	else
	    native_table[i].encoding[1] = -1;
    }
    free(ctrs);

    /* Build the preset table */
    if (cpuver <= CPC_ULTRA2) {
	n = sizeof(us2info) / sizeof(einfo_t);
	ep = us2info;
    }
    else if (cpuver <= LASTULTRA3) {
	n = sizeof(us3info) / sizeof(einfo_t);
	ep = us3info;
    }
    else
	return PAPI_ESBSTR;
    preset_table = malloc((n+1)*sizeof(hwi_search_t));
    npresets = 0;
    for (i = 0; i < n; ++i) {
	add_preset(preset_table, &npresets, ep[i]);
    }
    memset(&preset_table[npresets], 0, sizeof(hwi_search_t));

#ifdef DEBUG
    if (ISLEVEL(DEBUG_SUBSTRATE)) {
    SUBDBG("Native table: %d\n", nctrs);
    for (i = 0; i < nctrs; ++i) {
	SUBDBG("%40s: %8x %8x\n", native_table[i].name,
	    native_table[i].encoding[0], native_table[i].encoding[1]);
    }
    SUBDBG("\nPreset table: %d\n", npresets);
    for (i = 0; preset_table[i].event_code != 0; ++i) {
	SUBDBG("%8x: op %2d e0 %8x e1 %8x\n",
		preset_table[i].event_code,
		preset_table[i].data.derived,
		preset_table[i].data.native[0],
		preset_table[i].data.native[1]);
    } }
#endif
    return PAPI_OK;
}

static int
srch_event(char *e1)
{
    int i;

    for (i = 0; i < nctrs; ++i) {
	if (strcmp(e1, native_table[i].name) == 0)
	    break;
    }
    if (i >= nctrs)
	return -1;
    return i;
}

static void
add_preset(hwi_search_t *tab, int *np, einfo_t e)
{
    /* Parse the event info string and build the PAPI preset.
     * If parse fails, just return, otherwise increment the table
     * size. We assume that the table is big enough.
     */
    char *p;
    char *q;
    char op;
    char e1[MAX_ENAME], e2[MAX_ENAME];
    int i;
    int ne;
    int ne2;

    p = e.event_str;
    /* Assume p is the name of a native event, the sum of two
     * native events, or the difference of two native events.
     * This could be extended with a real parser (hint).
     */
    while (isspace(*p)) ++p;
    q = p;
    i = 0;
    while (isalnum(*p) || (*p == '_')) {
	if (i >= MAX_ENAME-1)
	    break;
	e1[i] = *p++;
	++i;
    }
    e1[i] = 0;
    if (*p == '+' || *p == '-')
	op = *p++;
    else
	op = 0;
    while (isspace(*p)) ++p;
    q = p;
    i = 0;
    while (isalnum(*p) || (*p == '_')) {
	if (i >= MAX_ENAME-1)
	    break;
	e2[i] = *p++;
	++i;
    }
    e2[i] = 0;

    if (e2[0] == 0 && e1[0] == 0) {
	return;
    }
    if (e2[0] == 0 || op == 0) {
	ne = srch_event(e1);
	if (ne == -1)
	    return;
	tab[*np].event_code = e.papi_event;
	tab[*np].data.derived = 0;
	tab[*np].data.native[0] = PAPI_NATIVE_MASK | ne;
	tab[*np].data.native[1] = PAPI_NULL;
	memset(tab[*np].data.operation, 0, OPS);
	++*np;
	return;
    }
    ne = srch_event(e1);
    ne2 = srch_event(e2);
    if (ne == -1 || ne2 == -1)
	return;
    tab[*np].event_code = e.papi_event;
    tab[*np].data.derived = (op == '-') ? DERIVED_SUB : DERIVED_ADD;
    tab[*np].data.native[0] = PAPI_NATIVE_MASK | ne;
    tab[*np].data.native[1] = PAPI_NATIVE_MASK | ne2;
    memset(tab[*np].data.operation, 0, OPS);
    ++*np;
}

void
action(void *arg, int regno, const char *name, uint8_t bits)
{
    int i;

    if (arg == 0) {
	++nctrs;
	return;
    }
    assert(regno == 0 || regno == 1);
    for (i = 0; i < nctrs; ++i) {
	if (strcmp(ctrs[i].name, name) == 0) {
	    ctrs[i].bits[regno] = bits;
	    ctrs[i].bitmask |=  (1 << regno);
	    return;
	}
    }
    memset(&ctrs[i], 0, sizeof(ctrs[i]));
    ctrs[i].name = strdup(name);
    if (ctrs[i].name == 0) {
	perror("strdup");
	exit(1);
    }
    ctrs[i].bits[regno] = bits;
    ctrs[i].bitmask = (1 << regno);
    ++nctrs;
}

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static int set_inherit(EventSetInfo_t * global, int arg)
{
   return (PAPI_ESBSTR);

/*
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  papi_cpc_event_t *command= &machdep->counter_cmd;

  return(PAPI_EINVAL);
*/

#if 0
   if (arg == 0) {
      if (command->flags & CPC_BIND_LWP_INHERIT)
         command->flags = command->flags ^ CPC_BIND_LWP_INHERIT;
   } else if (arg == 1) {
      command->flags = command->flags | CPC_BIND_LWP_INHERIT;
   } else
      return (PAPI_EINVAL);

   return (PAPI_OK);
#endif
}

static int set_default_domain(hwd_control_state_t * ctrl_state, int domain)
{
   /* This doesn't exist on this platform */

   if (domain == PAPI_DOM_OTHER)
      return (PAPI_EINVAL);

   return (set_domain(ctrl_state, domain));
}

static int set_default_granularity(hwd_control_state_t * current_state, int granularity)
{
   return (set_granularity(current_state, granularity));
}

rwlock_t lock[PAPI_MAX_LOCK];

static void lock_init(void)
{
  memset(lock,0x0,sizeof(rwlock_t)*PAPI_MAX_LOCK);
}

int _papi_hwd_init_global(void)
{
   int retval;

   /* Fill in what we can of the papi_system_info. */

   retval = get_system_info();
   if (retval)
      return (retval);

   lock_init();

   SUBDBG("Found %d %s %s CPU's at %f Mhz.\n",
          _papi_hwi_system_info.hw_info.totalcpus,
          _papi_hwi_system_info.hw_info.vendor_string,
          _papi_hwi_system_info.hw_info.model_string, _papi_hwi_system_info.hw_info.mhz);

   return (PAPI_OK);
}

int _papi_hwd_init(hwd_context_t * zero)
{
   return (PAPI_OK);
}

void _papi_hwd_error(int error, char *where)
{
   sprintf(where, "Substrate error: %s", strerror(error));
}

int _papi_hwd_add_prog_event(hwd_control_state_t * this_state,
                             unsigned int event, void *extra, EventInfo_t * out)
{
   return (PAPI_ESBSTR);
}

/* reset the hardware counter */
int _papi_hwd_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int retval;

   /* reset the hardware counter */
   ctrl->counter_cmd.cmd.ce_pic[0] = 0;
   ctrl->counter_cmd.cmd.ce_pic[1] = 0;
   /* let's rock and roll */
   retval = cpc_bind_event(&ctrl->counter_cmd.cmd, ctrl->counter_cmd.flags);
   if (retval == -1)
      return (PAPI_ESYS);

   return (PAPI_OK);
}


int _papi_hwd_read(hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long ** events)
{
   int retval;

   retval = cpc_take_sample(&ctrl->counter_cmd.cmd);
   if (retval == -1)
      return (PAPI_ESYS);

   *events = (long_long *)ctrl->counter_cmd.cmd.ce_pic;

   return PAPI_OK;
}

int _papi_hwd_ctl(hwd_context_t * ctx, int code, _papi_int_option_t * option)
{

   switch (code) {
   case PAPI_DEFDOM:
      return (set_default_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DOMAIN:
      return (set_domain(&option->domain.ESI->machdep, option->domain.domain));
   case PAPI_DEFGRN:
      return (set_default_granularity
              (&option->domain.ESI->machdep, option->granularity.granularity));
   case PAPI_GRANUL:
      return (set_granularity
              (&option->granularity.ESI->machdep, option->granularity.granularity));
   default:
      return (PAPI_EINVAL);
   }
}

int _papi_hwd_write(hwd_context_t * ctx, hwd_control_state_t * ctrl, long long events[])
{
   return (PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t * ctx)
{
   return (PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
   (void) cpc_rele();
   return (PAPI_OK);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t * si, void *info)
{
   _papi_hwi_context_t ctx;
   ThreadInfo_t *t = NULL;

   ctx.si = si;
   ctx.ucontext = info;
   _papi_hwi_dispatch_overflow_signal((void *) &ctx,
                                      _papi_hwi_system_info.supports_hw_overflow, 0, 0, &t);
}

int _papi_hwd_set_overflow(EventSetInfo_t * ESI, int EventIndex, int threshold)
{
   hwd_control_state_t *this_state = &ESI->machdep;
   papi_cpc_event_t *arg = &this_state->counter_cmd;
   int hwcntr;

   if (threshold == 0) {
      if (this_state->overflow_num == 1) {
         arg->flags ^= CPC_BIND_EMT_OVF;
         if (sigaction(SIGEMT, NULL, NULL) == -1)
            return (PAPI_ESYS);
         this_state->overflow_num = 0;
      } else this_state->overflow_num--;

   } else {
      struct sigaction act;
      /* increase the counter for overflow events */
      this_state->overflow_num++;

      act.sa_sigaction = dispatch_emt;
      memset(&act.sa_mask, 0x0, sizeof(act.sa_mask));
      act.sa_flags = SA_RESTART | SA_SIGINFO;
      if (sigaction(SIGEMT, &act, NULL) == -1)
         return (PAPI_ESYS);

      arg->flags |= CPC_BIND_EMT_OVF;
      hwcntr = ESI->EventInfoArray[EventIndex].pos[0];
      if (hwcntr == 0)
         arg->cmd.ce_pic[0] = UINT64_MAX - (uint64_t) threshold;
      else if (hwcntr == 1)
         arg->cmd.ce_pic[1] = UINT64_MAX - (uint64_t) threshold;
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
   ESI->profile.overflowcount = 0;
   return (PAPI_OK);
}

/*
void *_papi_hwd_get_overflow_address(void *context)
{
   void *location;
   ucontext_t *info = (ucontext_t *) context;
   location = (void *) info->uc_mcontext.gregs[REG_PC];

   return (location);
}
*/

int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   int retval;

   /* reset the hardware counter */
   if (ctrl->overflow_num==0)
   {
      ctrl->counter_cmd.cmd.ce_pic[0] = 0;
      ctrl->counter_cmd.cmd.ce_pic[1] = 0;
   }
   /* let's rock and roll */
   retval = cpc_bind_event(&ctrl->counter_cmd.cmd, ctrl->counter_cmd.flags);
   if (retval == -1)
      return (PAPI_ESYS);

   return (PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
   cpc_bind_event(NULL, 0);
   return PAPI_OK;
}

int _papi_hwd_remove_event(hwd_register_map_t * chosen, unsigned int hardware_index,
                           hwd_control_state_t * out)
{
   return PAPI_OK;
}

int _papi_hwd_encode_native(char *name, int *code)
{
   return (PAPI_OK);
}

int _papi_hwd_allocate_registers(EventSetInfo_t * ESI)
{
   return 1;
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer)
{
   int index = *EventCode & PAPI_NATIVE_AND_MASK;

   if (cpuver <= CPC_ULTRA2) {
      if (index < MAX_NATIVE_EVENT_USII - 1) {
         *EventCode = *EventCode + 1;
         return (PAPI_OK);
      } else
         return (PAPI_ENOEVNT);
   } else if (cpuver <= LASTULTRA3) {
      if (index < MAX_NATIVE_EVENT - 1) {
         *EventCode = *EventCode + 1;
         return (PAPI_OK);
      } else
         return (PAPI_ENOEVNT);
   };
   return (PAPI_ENOEVNT);
}

char *_papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
   int nidx;

   nidx = EventCode ^ PAPI_NATIVE_MASK;
   if (nidx >= 0 && nidx < PAPI_MAX_NATIVE_EVENTS)
      return (native_table[nidx].name);
   return NULL;
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
   return (_papi_hwd_ntv_code_to_name(EventCode));
}

static void copy_value(unsigned int val, char *nam, char *names, unsigned int *values, int len)
{
   *values = val;
   strncpy(names, nam, len);
   names[len-1] = 0;
}

int _papi_hwd_ntv_bits_to_info(hwd_register_t *bits, char *names,
                               unsigned int *values, int name_len, int count)
{
   int i = 0;
   copy_value(bits->event[0], "US Ctr 0", &names[i*name_len], &values[i], name_len);
   if (++i == count) return(i);
   copy_value(bits->event[1], "US Ctr 1", &names[i*name_len], &values[i], name_len);
   return(++i);
}

int _papi_hwd_ntv_code_to_bits(unsigned int EventCode, hwd_register_t * bits)
{
   int index = EventCode & PAPI_NATIVE_AND_MASK;

   if(cpuver <= CPC_ULTRA2) {
      if(index >= MAX_NATIVE_EVENT_USII) {
         return(PAPI_ENOEVNT);
      }
   } else if(cpuver <= LASTULTRA3) {
      if(index >= MAX_NATIVE_EVENT) {
         return(PAPI_ENOEVNT);
      }
   } else return(PAPI_ENOEVNT);

   bits->event[0] = native_table[index].encoding[0];
   bits->event[1] = native_table[index].encoding[1];
   return(PAPI_OK);
}

void _papi_hwd_init_control_state(hwd_control_state_t * ptr)
{
   ptr->counter_cmd.flags = 0x0;
   ptr->counter_cmd.cmd.ce_cpuver = cpuver;
   ptr->counter_cmd.cmd.ce_pcr = 0x0;
   ptr->counter_cmd.cmd.ce_pic[0] = 0;
   ptr->counter_cmd.cmd.ce_pic[1] = 0;
   set_domain(ptr, _papi_hwi_system_info.default_domain);
   set_granularity(ptr, _papi_hwi_system_info.default_granularity);
   return;
}

int _papi_hwd_update_control_state(hwd_control_state_t * this_state,
                    NativeInfo_t * native, int count, hwd_context_t * zero)
{
   int i, nidx1, nidx2, hwcntr;
   uint64_t tmp, cmd0, cmd1, pcr;

/* save the last three bits */
   pcr = this_state->counter_cmd.cmd.ce_pcr & 0x7;

/* clear the control register */
   this_state->counter_cmd.cmd.ce_pcr = pcr;

/* no native events left */
   if (count == 0)
      return (PAPI_OK);

   cmd0 = -1;
   cmd1 = -1;
/* one native event */
   if (count == 1) {
      nidx1 = native[0].ni_event & PAPI_NATIVE_AND_MASK;
      hwcntr = 0;
      cmd0 = native_table[nidx1].encoding[0];
      native[0].ni_position = 0;
      if (cmd0 == -1) {
         cmd1 = native_table[nidx1].encoding[1];
         native[0].ni_position = 1;
      }
      tmp = 0;
   }

/* two native events */
   if (count == 2) {
      int avail1, avail2;

      avail1 = 0;
      avail2 = 0;
      nidx1 = native[0].ni_event & PAPI_NATIVE_AND_MASK;
      nidx2 = native[1].ni_event & PAPI_NATIVE_AND_MASK;
      if (native_table[nidx1].encoding[0] != -1)
         avail1 = 0x1;
      if (native_table[nidx1].encoding[1] != -1)
         avail1 += 0x2;
      if (native_table[nidx2].encoding[0] != -1)
         avail2 = 0x1;
      if (native_table[nidx2].encoding[1] != -1)
         avail2 += 0x2;
      if ((avail1 | avail2) != 0x3)
         return (PAPI_ECNFLCT);
      if (avail1 == 0x3) {
         if (avail2 == 0x1) {
            cmd0 = native_table[nidx2].encoding[0];
            cmd1 = native_table[nidx1].encoding[1];
            native[0].ni_position = 1;
            native[1].ni_position = 0;
         } else {
            cmd1 = native_table[nidx2].encoding[1];
            cmd0 = native_table[nidx1].encoding[0];
            native[0].ni_position = 0;
            native[1].ni_position = 1;
         }
      } else {
         if (avail1 == 0x1) {
            cmd0 = native_table[nidx1].encoding[0];
            cmd1 = native_table[nidx2].encoding[1];
            native[0].ni_position = 0;
            native[1].ni_position = 1;
         } else {
            cmd0 = native_table[nidx2].encoding[0];
            cmd1 = native_table[nidx1].encoding[1];
            native[0].ni_position = 1;
            native[1].ni_position = 0;
         }
      }
   }

/* set the control register */
   if (cmd0 != -1) {
      tmp = ((uint64_t) cmd0 << pcr_shift[0]);
   }
   if (cmd1 != -1) {
      tmp = tmp | ((uint64_t) cmd1 << pcr_shift[1]);
   }
   this_state->counter_cmd.cmd.ce_pcr = tmp | pcr;
#if DEBUG
   dump_cmd(&this_state->counter_cmd);
#endif

   return (PAPI_OK);
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
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
}

/* This function updates the selection status of
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t * dst, hwd_reg_alloc_t * src)
{
}

long_long _papi_hwd_get_real_usec(void)
{
   return ((long_long) gethrtime() / (long_long) 1000);
}

long_long _papi_hwd_get_real_cycles(void)
{
   return(_papi_hwd_get_real_usec() * (long long) _papi_hwi_system_info.hw_info.mhz);
}

long_long _papi_hwd_get_virt_usec(const hwd_context_t * zero)
{
   return ((long long) gethrvtime() / (long long) 1000);
}

long_long _papi_hwd_get_virt_cycles(const hwd_context_t * zero)
{
   return (((long long) gethrvtime() / (long long) 1000) * (long long) _papi_hwi_system_info.hw_info.mhz);
}

int _papi_hwd_update_shlib_info(void)
{
   /*??? system call takes very long */

   char cmd_line[PATH_MAX+PATH_MAX], fname[L_tmpnam];
   char line[256];
   char address[16], size[10], flags[64], objname[256];
   PAPI_address_map_t *tmp = NULL;

   FILE *f=NULL;
   int t_index=0, length, i;
   long addr;
   struct map_record {
      long address;
      int size;
      int flags;
      char objname[256];
      struct map_record * next;
   }  *tmpr, *head, *curr;

   tmpnam(fname);
   SUBDBG("Temporary name %s\n",fname);

   sprintf(cmd_line, "/bin/pmap %d > %s",getpid(), fname);
   if (system(cmd_line) == -1) {
      remove(fname);
      exit(-1);
   }
   f = fopen(fname, "r");   
   if (f == NULL ) {
      remove(fname);
      exit(-1);
   }
   /* ignore the first line */
   fgets(line, 256, f);
   head = curr = NULL;
   while (fgets(line, 256, f) != NULL) {
      /* discard the last line */
      if (strncmp(line, " total", 6) != 0 )
      {
         sscanf(line, "%s %s %s %s", address, size, flags, objname);
         if (objname[0] == '/' )  
         {
            tmpr = (struct map_record *)malloc(sizeof(struct map_record));
            if (tmpr==NULL) return(-1);
            tmpr->next = NULL;
            if (curr ) {
               curr->next = tmpr;
               curr = tmpr;
            }
            if (head == NULL) {
               curr = head = tmpr;
            }

            SUBDBG("%s\n", objname);

            if ( strstr(flags, "read") && strstr(flags, "exec") )
            {
              if ( !strstr(flags, "write") )  /* text segment */
               { 
                  t_index++;
                  tmpr->flags =1;
               } else {
                  tmpr->flags =0;
               }
               sscanf(address, "%x", &tmpr->address);
               sscanf(size,"%d", &tmpr->size);
               tmpr->size *= 1024;
               strcpy(tmpr->objname, objname);
            }
            
         }
         
      }
   }
   tmp = (PAPI_address_map_t *) calloc(t_index-1, sizeof(PAPI_address_map_t));

   if (tmp == NULL)
     { PAPIERROR("Error allocating shared library address map"); return(PAPI_ENOMEM); }
   
   t_index = -1;
   tmpr = curr = head;
   i=0;
   while (curr != NULL )
   {
      if (strcmp(_papi_hwi_system_info.exe_info.address_info.name,
                          basename(curr->objname))== 0 )
      {
         if ( curr->flags ) 
         {
            _papi_hwi_system_info.exe_info.address_info.text_start =
                                      (caddr_t) curr->address;
            _papi_hwi_system_info.exe_info.address_info.text_end =
                                      (caddr_t) (curr->address + curr->size);
         } else {
            _papi_hwi_system_info.exe_info.address_info.data_start =
                                      (caddr_t) curr->address;
            _papi_hwi_system_info.exe_info.address_info.data_end =
                                      (caddr_t) (curr->address + curr->size);
         }
      } else {
         if ( curr->flags ) 
         {
            t_index++;
            tmp[t_index].text_start = (caddr_t) curr->address;
            tmp[t_index].text_end =(caddr_t) (curr->address+curr->size);
               strncpy(tmp[t_index].name, curr->objname,PAPI_HUGE_STR_LEN-1 );
               tmp[t_index].name[PAPI_HUGE_STR_LEN-1]='\0';
         } else {
               if (t_index <0 )  continue;
               tmp[t_index].data_start = (caddr_t) curr->address;
               tmp[t_index].data_end = (caddr_t) (curr->address+ curr->size);
         }
      }
      tmpr = curr->next;
      /* free the temporary allocated memory */
      free(curr);
      curr = tmpr;
   }  /* end of while */ 
   fclose(f);
   if (_papi_hwi_system_info.shlib_info.map)
      free(_papi_hwi_system_info.shlib_info.map);
   _papi_hwi_system_info.shlib_info.map = tmp;
   _papi_hwi_system_info.shlib_info.count = t_index+1;

   return(PAPI_OK);

}

#if 0
/* once the bug in dladdr is fixed by SUN, (now dladdr caused deadlock when
   used with pthreads) this function can be used again */
int _papi_hwd_update_shlib_info(void)
{
   char fname[80], name[PATH_MAX];
   prmap_t newp;
   int count, t_index;
   FILE * map_f;
   void * vaddr;
   Dl_info dlip;
   PAPI_address_map_t *tmp = NULL;

   sprintf(fname, "/proc/%d/map", getpid());
   map_f = fopen(fname, "r");

   /* count the entries we need */
   count =0;
   t_index=0;
   while ( fread(&newp, sizeof(prmap_t), 1, map_f) > 0 ) {
      vaddr = (void*)(1+(newp.pr_vaddr)); // map base address 
      if (dladdr(vaddr, &dlip) > 0) {
         count++;
         if ((newp.pr_mflags & MA_EXEC) && (newp.pr_mflags & MA_READ) ) {
            if ( !(newp.pr_mflags & MA_WRITE)) 
               t_index++;
         }
         strcpy(name,dlip.dli_fname);
         if (strcmp(_papi_hwi_system_info.exe_info.address_info.name, 
                          basename(name))== 0 ) {
            if ((newp.pr_mflags & MA_EXEC) && (newp.pr_mflags & MA_READ) ) {
               if ( !(newp.pr_mflags & MA_WRITE)) {
                  _papi_hwi_system_info.exe_info.address_info.text_start = 
                                      (caddr_t) newp.pr_vaddr;
                  _papi_hwi_system_info.exe_info.address_info.text_end =
                                      (caddr_t) (newp.pr_vaddr+newp.pr_size);
               } else {
                  _papi_hwi_system_info.exe_info.address_info.data_start = 
                                      (caddr_t) newp.pr_vaddr;
                  _papi_hwi_system_info.exe_info.address_info.data_end =
                                      (caddr_t) (newp.pr_vaddr+newp.pr_size);
               }  
            }
         }
      } 

   }
   rewind(map_f);
   tmp = (PAPI_address_map_t *) calloc(t_index-1, sizeof(PAPI_address_map_t));

   if (tmp == NULL)
     { PAPIERROR("Error allocating shared library address map"); return(PAPI_ENOMEM); }
   t_index=-1;
   while ( fread(&newp, sizeof(prmap_t), 1, map_f) > 0 ) {
      vaddr = (void*)(1+(newp.pr_vaddr)); // map base address
      if (dladdr(vaddr, &dlip) > 0) {  // valid name
         strcpy(name,dlip.dli_fname);
         if (strcmp(_papi_hwi_system_info.exe_info.address_info.name, 
                          basename(name))== 0 ) 
            continue;
         if ((newp.pr_mflags & MA_EXEC) && (newp.pr_mflags & MA_READ) ) {
            if ( !(newp.pr_mflags & MA_WRITE)) {
               t_index++;
               tmp[t_index].text_start = (caddr_t) newp.pr_vaddr;
               tmp[t_index].text_end =(caddr_t) (newp.pr_vaddr+newp.pr_size);
               strncpy(tmp[t_index].name, dlip.dli_fname,PAPI_HUGE_STR_LEN-1 );
               tmp[t_index].name[PAPI_HUGE_STR_LEN-1]='\0';
            } else {
               if (t_index <0 )  continue;
               tmp[t_index].data_start = (caddr_t) newp.pr_vaddr;
               tmp[t_index].data_end = (caddr_t) (newp.pr_vaddr+newp.pr_size);
            }
         }
      }
   }

   fclose(map_f);

   if (_papi_hwi_system_info.shlib_info.map) 
      free(_papi_hwi_system_info.shlib_info.map);
   _papi_hwi_system_info.shlib_info.map = tmp;
   _papi_hwi_system_info.shlib_info.count = t_index+1;

   return(PAPI_OK);
}
#endif
