/* 
* File:    solaris-ultra.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
*/  

/* to understand this program, first you should read the user's manual
   about UltraSparc II and UltraSparc III, then the man pages
   about cpc_take_sample(cpc_event_t *event)
*/

#include SUBSTRATE
#include "papi_protos.h"
#include "papi_preset.h"

/* Globals used to access the counter registers. */

static int cpuver;
static int pcr_shift[2]; 
static uint64_t pcr_event_mask[2];
static uint64_t pcr_inv_mask[2];

int papi_debug;

/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

/* the number in this preset_search map table is the native event index 
   in the native event table, when it ORs the NATIVE_MASK, it becomes the
   native event code. 
*/
/* UltraSparc II preset search table */
hwi_search_t usii_preset_search_map[] = {
  /* L1 Cache Imisses */
  {PAPI_L1_ICM,{DERIVED_SUB,{NATIVE_MASK|4,NATIVE_MASK|14}}},		
  /* L2 Total Cache misses*/
  {PAPI_L2_TCM,{DERIVED_SUB,{NATIVE_MASK|8,NATIVE_MASK|18}}},			
  /* Req. for snoop*/
  {PAPI_CA_SNP,{0,{NATIVE_MASK|20,0}}},	
  /* Req. invalidate cache line*/
  {PAPI_CA_INV,{0,{NATIVE_MASK|10,0}}},		 	
  /* L1LM */
  {PAPI_L1_LDM,{DERIVED_SUB,{NATIVE_MASK|5,NATIVE_MASK|15}}},			
  /* L1SM */
  {PAPI_L1_STM,{DERIVED_SUB,{NATIVE_MASK|6,NATIVE_MASK|16}}},			
  /* Cond. branch inst. mispred.*/
  {PAPI_BR_MSP,{0,{NATIVE_MASK|12, 0}}},          
  /* Total inst. issued*/
  {PAPI_TOT_IIS,{0,{NATIVE_MASK|1, 0}}},	
  /* Total inst. executed*/
  {PAPI_TOT_INS,{0,{NATIVE_MASK|1, 0}}}, 
  /* Loads executed*/
  {PAPI_LD_INS,{0,{NATIVE_MASK|5,0}}},		
  /* Stores executed*/
  {PAPI_SR_INS,{0,{NATIVE_MASK|6,0}}},		
  /* Total cycles */
  {PAPI_TOT_CYC,{0,{NATIVE_MASK|0,0}}},		
  /* IPS */
  {PAPI_IPS,{DERIVED_PS,{NATIVE_MASK|0,NATIVE_MASK|1}}},			
  /* L1 data cache reads */
  {PAPI_L1_DCR,{0,{NATIVE_MASK|5,0}}},		
  /* L1 data cache writes */
  {PAPI_L1_DCW,{0,{NATIVE_MASK|6,0}}},		
  /* L1 instruction cache hits */
  {PAPI_L1_ICH,{0,{NATIVE_MASK|14, 0}}},
  /* L2 instruction cache hits */
  {PAPI_L2_ICH,{0,{NATIVE_MASK|21,0}}},		
  /* L1 instruction cache accesses */
  {PAPI_L1_ICA,{0,{NATIVE_MASK|4,0}}},		
  /* L2 total cache hits */
  {PAPI_L2_TCH,{0,{NATIVE_MASK|18, 0}}},		
  /* L2 total cache accesses */
  {PAPI_L2_TCA,{0,{NATIVE_MASK|8,0}}},
  /* Terminator */
  {0,{0,{0,0}}}};

/* UltraSparc III preset search table */
hwi_search_t usiii_preset_search_map[] = {
  /* Floating point instructions */
  {PAPI_FP_INS,{DERIVED_ADD,{NATIVE_MASK|22,NATIVE_MASK|68}}}, 
                    /* pic0 FA_pipe_completion and pic1 FM_pipe_completion */
  /* Floating point add instructions */
  {PAPI_FAD_INS,{0,{NATIVE_MASK|22,0}}},       /* pic0 FA_pipe_completion */
  /* Floating point multiply instructions */
  {PAPI_FML_INS,{0,{NATIVE_MASK|68,0}}},       /* pic1 FM_pipe_completion */
  /* ITLB */
  {PAPI_TLB_IM,{0,{NATIVE_MASK|47,0}}},        /* pic1 ITLB_miss */
  /* DITLB */
  {PAPI_TLB_DM,{0,{NATIVE_MASK|48, 0}}},       /* pic1 DTLB_miss */
  /* Total cycles */
  {PAPI_TOT_CYC,{0,{NATIVE_MASK|0,0}}},        /* pic0 and pic1 Cycle_cnt */				
  /* Total inst. issued*/
  {PAPI_TOT_IIS,{0,{NATIVE_MASK|1,0}}},        /* pic0 and pic1 Instr_cnt */				
  /* Total inst. executed*/
  {PAPI_TOT_INS,{0,{NATIVE_MASK|1,0}}},        /* pic0 and pic1 Instr_cnt */		
  /* L2 Total Cache misses*/
  {PAPI_L2_TCM,{0,{NATIVE_MASK|42, 0}}},       /* pic1 EC_misses */			
  /* L2 Total ICache misses*/
  {PAPI_L2_ICM,{0,{NATIVE_MASK|45, 0}}},       /* pic1 EC_ic_miss */			
  /* L1 Total ICache misses */
  {PAPI_L1_ICM,{0,{NATIVE_MASK|38, 0}}},       /* pic1 IC_miss (actually hits) */      		
  /* L1 Load Misses */
  {PAPI_L1_LDM,{0,{NATIVE_MASK|39, 0}}},       /* pic1 DC_rd_miss */			
  /* L1 Store Misses */
  {PAPI_L1_STM,{0,{NATIVE_MASK|40, 0}}},       /* pic1 DC_wr_miss */			
  /* Cond. branch inst. mispred.*/
  {PAPI_BR_MSP,{0,{NATIVE_MASK|32, 0}}},       /* pic1 Dispatch0_mispred */
  /* IPS */
  {PAPI_IPS,{DERIVED_PS,{NATIVE_MASK|0,NATIVE_MASK|1}}},  
                                          /* pic0 Cycle_cnt, pic1 Instr_cnt */
  /* L1 data cache reads */
  {PAPI_L1_DCR,{0,{NATIVE_MASK|8,0}}},	      /* pic0 DC_rd */	
  /* L1 data cache writes */
  {PAPI_L1_DCW,{0,{NATIVE_MASK|9,0}}},	      /* pic0 DC_wr */	
  /* L1 instruction cache hits */
  {PAPI_L1_ICH,{0,{NATIVE_MASK|7,0}}},    /* pic0 IC_ref (actually hits only) */
  /* L1 instruction cache accesses */
  {PAPI_L1_ICA,{DERIVED_ADD,{NATIVE_MASK|7,NATIVE_MASK|38}}}, 
                          /* pic0 IC_ref (actually hits only) + pic1 IC_miss */
  /* L2 total cache hits */
  {PAPI_L2_TCH,{DERIVED_SUB,{NATIVE_MASK|10,NATIVE_MASK|42}}},
                                             /* pic0 EC_ref - pic1 EC_misses */
  /* L2 total cache accesses */
  {PAPI_L2_TCA,{0,{NATIVE_MASK|10,0}}},       /* pic0 EC_ref */
  /* Terminator */
  {0,{0,{0,0}}}};

/* the encoding array in native_info_t is the encodings for PCR.SL
   and PCR.SU, encoding[0] is for PCR.SL and encoding[1] is for PCR.SU,
   the value -1 means it is not supported by the corresponding Performance
   Instrumentation Counter register. For example, Cycle_cnt can be counted
   by PICL and PICU, but Dispatch0_IC_miss can be only counted by PICL.
   These encoding information will be used to allocate register to events
   and update the control structure.
*/
/* UltraSparc II native event table */
native_info_t  usii_native_table[]= {
/* 0  */   {"Cycle_cnt", {0x0, 0x0}},
/* 1  */   {"Instr_cnt", {0x1, 0x1}},
/* 2  */   {"Dispatch0_IC_miss", {0x2,-1}},
/* 3  */   {"Dispatch0_storeBuf", {0x3, -1}},
/* 4  */   {"IC_ref", {0x8, -1}},
/* 5  */   {"DC_rd", {0x9,-1}},
/* 6  */   {"DC_wr", {0xa,-1}},
/* 7  */   {"Load_use", {0xb, -1}},
/* 8  */   {"EC_ref", {0xc, -1}},
/* 9  */   {"EC_write_hit_RDO",{0xd,-1}},
/* 10 */   {"EC_snoop_inv", {0xe, -1}},
/* 11 */   {"EC_rd_hit", {0xf,-1}},
/* 12 */   {"Dispatch0_mispred", {-1, 0x2}},
/* 13 */   {"Dispatch0_FP_use", {-1, 0x3}},
/* 14 */   {"IC_hit", {-1, 0x8}},
/* 15 */   {"DC_rd_hit", {-1,0x9}},
/* 16 */   {"DC_wr_hit", {-1, 0xa}},
/* 17 */   {"Load_use_RAW", {-1,0xb}},
/* 18 */   {"EC_hit", {-1, 0xc}},
/* 19 */   {"EC_wb", {-1, 0xd}},
/* 20 */   {"EC_snoop_cb", {-1, 0xe}},
/* 21 */   {"EC_ic_hit", {-1, 0xf}}
};

/* UltraSparc III native event table */
native_info_t  usiii_native_table[]= {
/* 0  */   {"Cycle_cnt", {0x0, 0x0}},
/* 1  */   {"Instr_cnt", {0x1, 0x1}},
/* 2  */   {"Dispatch0_IC_miss", {0x2,-1}},
/* 3  */   {"Dispatch0_br_target", {0x3, -1}},
/* 4  */   {"Dispatch0_2nd_br", {0x4, -1}},
/* 5  */   {"Rstall_storeQ", {0x5, -1}},
/* 6  */   {"Rstall_IU_use", {0x6, -1}},
/* 7  */   {"IC_ref", {0x8, -1}},
/* 8  */   {"DC_rd", {0x9,-1}},
/* 9  */   {"DC_wr", {0xa,-1}},
/* 10 */   {"EC_ref", {0xc, -1}},
/* 11 */   {"EC_write_hit_RTO",{0xd,-1}},
/* 12 */   {"EC_snoop_inv", {0xe, -1}},
/* 13 */   {"EC_rd_miss", {0xf,-1}},
/* 14 */   {"PC_port0_rd", {0x10, -1}},
/* 15 */   {"SI_snoop", {0x11, -1}},
/* 16 */   {"SI_ciq_flow", {0x12, -1}},
/* 17 */   {"SI_owned", {0x13, -1}},
/* 18 */   {"SW_count0", {0x14, -1}},
/* 19 */   {"IU_Stat_Br_miss_taken", {0x15, -1}},
/* 20 */   {"IU_Stat_Br_count_taken", {0x16, -1}},
/* 21 */   {"Dispatch_rs_mispred", {0x17, -1}},
/* 22 */   {"FA_pipe_completion", {0x18, -1}},
/* 23 */   {"EC_wb_remote", {0x19, -1}},
/* 24 */   {"EC_miss_local", {0x1a, -1}},
/* 25 */   {"EC_miss_mtag_remote", {0x1b, -1}},
/* 26 */   {"MC_reads_0", {0x20, -1}},
/* 27 */   {"MC_reads_1", {0x21, -1}},
/* 28 */   {"MC_reads_2", {0x22, -1}},
/* 29 */   {"MC_reads_3", {0x23, -1}},
/* 30 */   {"MC_stalls_0", {0x24, -1}},
/* 31 */   {"MC_stalls_2", {0x25, -1}},
/* 32 */   {"Dispatch0_mispred", {-1, 0x2}},
/* 33 */   {"IC_miss_cancelled", {-1, 0x3}},
/* 34 */   {"Re_DC_missovhd", {-1, 0x4}},
/* 35 */   {"Re_FPU_bypass", {-1, 0x5}},
/* 36 */   {"Re_DC_miss", {-1, 0x6}},
/* 37 */   {"Re_EC_miss", {-1, 0x7}},
/* 38 */   {"IC_miss", {-1, 0x8}},
/* 39 */   {"DC_rd_miss", {-1,0x9}},
/* 40 */   {"DC_wr_miss", {-1, 0xa}},
/* 41 */   {"Rstall_FP_use", {-1, 0xb}},
/* 42 */   {"EC_misses", {-1, 0xc}},
/* 43 */   {"EC_wb", {-1, 0xd}},
/* 44 */   {"EC_snoop_cb", {-1, 0xe}},
/* 45 */   {"EC_ic_miss", {-1, 0xf}},
/* 46 */   {"Re_PC_miss", {-1, 0x10}},
/* 47 */   {"ITLB_miss", {-1, 0x11}},
/* 48 */   {"DTLB_miss", {-1, 0x12}},
/* 49 */   {"WC_miss", {-1, 0x13}},
/* 50 */   {"WC_snoop_cb", {-1, 0x14}},
/* 51 */   {"WC_scrubbed", {-1, 0x15}},
/* 52 */   {"WC_wb_wo_read", {-1, 0x16}},
/* 53 */   {"PC_soft_hit", {-1, 0x18}},
/* 54 */   {"PC_snoop_inv", {-1, 0x19}},
/* 55 */   {"PC_hard_hit", {-1, 0x1a}},
/* 56 */   {"PC_port1_rd", {-1, 0x1b}},
/* 57 */   {"SW_count1", {-1, 0x1c}},
/* 58 */   {"IU_Stat_Br_miss_untaken", {-1, 0x1d}},
/* 59 */   {"IU_Stat_Br_count_untaken", {-1, 0x1e}},
/* 60 */   {"PC_MS_miss", {-1, 0x1f}},
/* 61 */   {"MC_writes_0", {-1, 0x20}},
/* 62 */   {"MC_writes_1", {-1, 0x21}},
/* 63 */   {"MC_writes_2", {-1, 0x22}},
/* 64 */   {"MC_writes_3", {-1, 0x23}},
/* 65 */   {"MC_stalls_1", {-1, 0x24}},
/* 66 */   {"MC_stalls_3", {-1, 0x25}},
/* 67 */   {"Re_RAW_miss", {-1, 0x26}},
/* 68 */   {"FM_pipe_completion", {-1, 0x27}},
/* 69 */   {"EC_miss_mtag_remote", {-1, 0x28}},
/* 70 */   {"EC_miss_remote", {-1, 0x29}}
};

extern papi_mdi_t _papi_hwi_system_info;
int _papi_hwi_event_index_map[MAX_COUNTERS];

hwi_search_t *preset_search_map;
static native_info_t *native_table;

#ifdef DEBUG
static void dump_cmd(papi_cpc_event_t *t)
{
  DBG((stderr,"cpc_event_t.ce_cpuver %d\n",t->cmd.ce_cpuver));
  DBG((stderr,"ce_tick %llu\n",t->cmd.ce_tick));
  DBG((stderr,"ce_pic[0] %llu ce_pic[1] %llu\n",t->cmd.ce_pic[0],t->cmd.ce_pic[1]));
  DBG((stderr,"ce_pcr 0x%llx\n",t->cmd.ce_pcr));
  DBG((stderr,"flags %x\n",t->flags));
}
#endif 

static void dispatch_emt(int signal, siginfo_t *sip, void *arg)
{
  int event_counter;
  _papi_hwi_context_t ctx;

  ctx.si = sip;
  ctx.ucontext = arg;

#ifdef DEBUG
  if (papi_debug)
    psignal(signal, "dispatch_emt");
#endif

  if (sip->si_code == EMT_CPCOVF)
  {
    papi_cpc_event_t *sample;
    EventSetInfo_t *ESI;
    ThreadInfo_t *thread;
    int t, overflow_vector, readvalue;

    thread = _papi_hwi_lookup_in_thread_list();
    ESI = (EventSetInfo_t *)thread->event_set_overflowing;

    event_counter=ESI->overflow.event_counter;
    sample = &(ESI->machdep.counter_cmd);

    /* GROSS! This is a hack to 'push' the correct values 
	 back into the hardware, such that when PAPI handles
     the overflow and reads the values, it gets the correct ones.
    */
      
    /* Find which HW counter is overflowing */
      
    if (ESI->EventInfoArray[ESI->overflow.EventIndex[0]].pos[0] == 0)
      t = 0;
    else
      t = 1;
      
    if ( cpc_take_sample(&sample->cmd) == -1)
      return;
    if(event_counter==1) {
      overflow_vector=1<<t; 
      sample->cmd.ce_pic[t] = UINT64_MAX - ESI->overflow.threshold[0];
    }
    else {
      overflow_vector=0;
      readvalue=sample->cmd.ce_pic[0];
      if (readvalue >= 0 )
      {
          overflow_vector=1;
          if (t==0)
            sample->cmd.ce_pic[0] = UINT64_MAX - ESI->overflow.threshold[0];
          else
            sample->cmd.ce_pic[0] = UINT64_MAX - ESI->overflow.threshold[1];
      }
      readvalue=sample->cmd.ce_pic[1];
      if (readvalue >= 0 )
      {
          overflow_vector ^= 1<<1; 
          if (t==0)
            sample->cmd.ce_pic[1] = UINT64_MAX - ESI->overflow.threshold[1];
          else
            sample->cmd.ce_pic[1] = UINT64_MAX - ESI->overflow.threshold[0];
      }
      DBG((stderr,"overflow_vector, = %d\n",overflow_vector));
      if (overflow_vector==0) abort();
    }

    /* Call the regular overflow function in extras.c */
    _papi_hwi_dispatch_overflow_signal(&ctx,  _papi_hwi_system_info.supports_hw_overflow, overflow_vector, 0);

#if 0
    /* Reset the threshold */
      
    if ( cpc_take_sample(&sample->cmd) == -1)
      return;
    sample->cmd.ce_pic[t] = UINT64_MAX - ESI->overflow.threshold[0];
#endif
      
#if DEBUG
    dump_cmd(sample);
#endif
    if (cpc_bind_event(&sample->cmd,sample->flags) == -1)
       return;
  }
  else
  {
    DBG((stderr,"dispatch_emt() dropped, si_code = %d\n",sip->si_code));
    return;
  }
}

static int scan_prtconf(char *cpuname,int len_cpuname,int *hz, int *ver)
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
  FILE *f;
  char cmd_line[80], fname[L_tmpnam];
  unsigned int matched;
  
  /*??? system call takes very long */
  /* get system configuration and put output into file */
  sprintf(cmd_line, "/usr/sbin/prtconf -vp >%s", tmpnam(fname));
  DBG((stderr,"/usr/sbin/prtconf -vp > %s \n", fname));
  if(system(cmd_line) == -1)
    {
      remove(fname);
      return -1;
    }
  
  /* open output file */
  if((f = fopen(fname, "r")) == NULL)
    {
      remove(fname);
      return -1;
    }
  
  DBG((stderr,"Parsing %s...\n", fname));
  /* ignore all lines until we reach something with a sparc line */
  matched = 0x0; ihz = -1;
  while(fgets(line, 256, f) != NULL)
    {
      /* DBG((stderr,">>> %s",line)); */
      if((sscanf(line, "%s", cmd) == 1)
	 && strstr(line, "Node 0x")) {
	matched = 0x0;
        /* DBG((stderr,"Found 'Node' -- search reset. (0x%2.2x)\n",matched)); */
        }
      else {
	 if (strstr(cmd, "device_type:") &&
             strstr(line, "'cpu'" )) {
            matched |= 0x1;
            /* DBG((stderr,"Found 'cpu'. (0x%2.2x)\n",matched)); */
            }
	 else if (!strcmp(cmd, "sparc-version:") &&
               (sscanf(line, "%s %x", cmd, &version) == 2) ) {
            matched |= 0x2;
            /* DBG((stderr,"Found version=%d. (0x%2.2x)\n", version, matched)); */
            }
	 else if (!strcmp(cmd, "clock-frequency:") &&
               (sscanf(line, "%s %x", cmd, &ihz) == 2) ) {
            matched |= 0x4;
            /* DBG((stderr,"Found ihz=%d. (0x%2.2x)\n", ihz,matched)); */
            }
	 else if (!strcmp(cmd, "name:") &&
               (sscanf(line, "%s %s", cmd, name) == 2) ) {
            matched |= 0x8;
            /* DBG((stderr,"Found name: %s. (0x%2.2x)\n", name,matched)); */
            }
      }
     if((matched & 0xF) == 0xF ) break;
    }
  DBG((stderr,"Parsing found name=%s, speed=%dHz, version=%d\n", name, ihz,version));
  
  if(matched ^ 0x0F)
    ihz = -1;
  else {
    *hz = (float) ihz;
    *ver = version;
    strncpy(cpuname,name,len_cpuname);
    }

  return ihz;

  /* End stolen code */
}

static int set_domain(hwd_control_state_t *this_state, int domain)
{
  papi_cpc_event_t *command= &this_state->counter_cmd;
  cpc_event_t *event = &command->cmd;
  uint64_t pcr = event->ce_pcr;

  /* This doesn't exist on this platform */

  if (domain == PAPI_DOM_OTHER)
    return(PAPI_EINVAL);

  pcr = pcr | 0x7;
  pcr = pcr ^ 0x7;
  if (domain & PAPI_DOM_USER)
    pcr = pcr | 1 << CPC_ULTRA_PCR_USR;
  if (domain & PAPI_DOM_KERNEL)
    pcr = pcr | 1 << CPC_ULTRA_PCR_SYS;

  event->ce_pcr = pcr;

  return(PAPI_OK);
}

static int set_granularity(hwd_control_state_t *this_state, int domain)
{
  return(PAPI_OK);
}

/* Utility functions */

/* This is a wrapper arount fprintf(stderr,...) for cpc_walk_events() */
void print_walk_names(void  *arg,  int regno,  const char *name, uint8_t bits)
{
  fprintf(stderr,arg,regno,name,bits);
}

static int get_system_info(void)
{
  int retval;
  pid_t pid;
  char maxargs[PAPI_MAX_STR_LEN] = "<none>";
  psinfo_t psi;
  int fd;
  int i,hz,version;
  char cpuname[PAPI_MAX_STR_LEN];
  const char *name;

  /* Check counter access */

  if (cpc_version(CPC_VER_CURRENT) != CPC_VER_CURRENT)
    return(PAPI_ESBSTR);
  DBG((stderr,"CPC version %d successfully opened\n",CPC_VER_CURRENT));

  if (cpc_access() == -1)
    return(PAPI_ESBSTR);

  /* Global variable cpuver */

  cpuver = cpc_getcpuver();
  DBG((stderr,"Got %d from cpc_getcpuver()\n",cpuver));
  if (cpuver == -1)
    return(PAPI_ESBSTR);

#ifdef DEBUG
  {
    if (papi_debug) 
    {
      name = cpc_getcpuref(cpuver);
      if(name)
        fprintf(stderr,"CPC CPU reference: %s\n",name);
      else
        fprintf(stderr,"Could not get a CPC CPU reference.\n");
  
      for(i=0;i<cpc_getnpic(cpuver);i++) 
      {
        fprintf(stderr,"\n%6s %-40s %8s\n","Reg","Symbolic name","Code");
        cpc_walk_names(cpuver, i, "%6d %-40s %02x\n",print_walk_names);
      }
      fprintf(stderr,"\n");
    }
  }
#endif


  /* Initialize other globals */

  if (cpuver <= CPC_ULTRA2)
  {
    DBG((stderr,"cpuver (==%d) <= CPC_ULTRA2 (==%d)\n",cpuver,CPC_ULTRA2));
    pcr_shift[0] = CPC_ULTRA_PCR_PIC0_SHIFT; 
    pcr_shift[1] = CPC_ULTRA_PCR_PIC1_SHIFT; 
    pcr_event_mask[0] = (CPC_ULTRA2_PCR_PIC0_MASK<<CPC_ULTRA_PCR_PIC0_SHIFT);
    pcr_event_mask[1] = (CPC_ULTRA2_PCR_PIC1_MASK<<CPC_ULTRA_PCR_PIC1_SHIFT);
    pcr_inv_mask[0] = ~(pcr_event_mask[0]);
    pcr_inv_mask[1] = ~(pcr_event_mask[1]);
    preset_search_map= usii_preset_search_map;
    native_table= usii_native_table;
  }
  else if (cpuver == CPC_ULTRA3)
    {
      DBG((stderr,"cpuver (==%d) == CPC_ULTRA3 (==%d)\n",cpuver,CPC_ULTRA3));
      pcr_shift[0] = CPC_ULTRA_PCR_PIC0_SHIFT; 
      pcr_shift[1] = CPC_ULTRA_PCR_PIC1_SHIFT; 
      pcr_event_mask[0] = (CPC_ULTRA3_PCR_PIC0_MASK<<CPC_ULTRA_PCR_PIC0_SHIFT);
      pcr_event_mask[1] = (CPC_ULTRA3_PCR_PIC1_MASK<<CPC_ULTRA_PCR_PIC1_SHIFT);
      pcr_inv_mask[0] = ~(pcr_event_mask[0]);
      pcr_inv_mask[1] = ~(pcr_event_mask[1]);
      _papi_hwi_system_info.supports_hw_overflow = 1;
      preset_search_map= usiii_preset_search_map;
      native_table= usiii_native_table;
    }
    else
      return(PAPI_ESBSTR);

  /* Path and args */

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);

  /* Turn on microstate accounting for this process and any LWPs. */
  
  sprintf(maxargs,"/proc/%d/ctl",(int) pid);
  if ((fd = open(maxargs,O_WRONLY)) == -1)
    return(PAPI_ESYS);
  {
    int retval;
    struct { long cmd; long flags; } cmd;
    cmd.cmd = PCSET;
    cmd.flags = PR_MSACCT | PR_MSFORK;
    retval = write(fd,&cmd,sizeof(cmd));
    close(fd);
    DBG((stderr,"Write PCSET returned %d\n",retval));
    if (retval != sizeof(cmd))
      return(PAPI_ESYS);
  }

  /* Get executable info */

  sprintf(maxargs,"/proc/%d/psinfo",(int)pid);
  if ((fd = open(maxargs,O_RDONLY)) == -1)
    return(PAPI_ESYS);
  read(fd,&psi,sizeof(psi));
  close(fd);
  {
    char *tmp;

    tmp = strchr(psi.pr_psargs,' ');
    if (tmp != NULL)
      *tmp = '\0';
  }
  strncpy(_papi_hwi_system_info.exe_info.fullname,psi.pr_psargs,PAPI_MAX_STR_LEN);
  strncpy(_papi_hwi_system_info.exe_info.name,basename(psi.pr_psargs),PAPI_MAX_STR_LEN);
  DBG((stderr,"Executable is %s\n",_papi_hwi_system_info.exe_info.name));
  DBG((stderr,"Full Executable is %s\n",_papi_hwi_system_info.exe_info.fullname));

  /* Hardware info */

  _papi_hwi_system_info.hw_info.ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  _papi_hwi_system_info.hw_info.nnodes = 1;
  _papi_hwi_system_info.hw_info.totalcpus = sysconf(_SC_NPROCESSORS_CONF);

  retval = scan_prtconf(cpuname,PAPI_MAX_STR_LEN,&hz,&version);
  if (retval == -1)
    return(PAPI_ESBSTR);

  strcpy(_papi_hwi_system_info.hw_info.model_string,cpc_getcciname(cpuver));
  _papi_hwi_system_info.hw_info.model = cpuver;
  strcpy(_papi_hwi_system_info.hw_info.vendor_string,"SUN unknown");
  _papi_hwi_system_info.hw_info.vendor = -1;
  _papi_hwi_system_info.hw_info.revision = version;

  _papi_hwi_system_info.hw_info.mhz = ( (float) hz / 1.0e6 );
  DBG((stderr,"hw_info.mhz = %f\n",_papi_hwi_system_info.hw_info.mhz));

  /* Number of PMCs */

  retval = cpc_getnpic(cpuver);
  if (retval < 1)
    return(PAPI_ESBSTR);
  _papi_hwi_system_info.num_gp_cntrs = retval;
  _papi_hwi_system_info.num_cntrs = retval;
  DBG((stderr,"num_cntrs = %d\n",_papi_hwi_system_info.num_cntrs));

  /* program text segment, data segment  address info */
  _papi_hwi_system_info.exe_info.address_info.text_start = (caddr_t)&_start;
  _papi_hwi_system_info.exe_info.address_info.text_end = (caddr_t)&_etext;
  _papi_hwi_system_info.exe_info.address_info.data_start = (caddr_t)&_etext+1;
  _papi_hwi_system_info.exe_info.address_info.data_end = (caddr_t)&_edata;
  _papi_hwi_system_info.exe_info.address_info.bss_start = (caddr_t)&_edata+1;
  _papi_hwi_system_info.exe_info.address_info.bss_end = (caddr_t)&_end;

  /* Setup presets */

  retval = _papi_hwi_setup_all_presets(preset_search_map);
  if (retval)
    return(retval);

  return(PAPI_OK);
} 

/* This function should tell your kernel extension that your children
   inherit performance register information and propagate the values up
   upon child exit and parent wait. */

static int set_inherit(EventSetInfo_t *global, int arg)
{
  return(PAPI_ESBSTR);

/*
  hwd_control_state_t *machdep = (hwd_control_state_t *)global->machdep;
  papi_cpc_event_t *command= &machdep->counter_cmd;

  return(PAPI_EINVAL);
*/

#if 0
  if (arg == 0)
    {
      if (command->flags & CPC_BIND_LWP_INHERIT)
	command->flags = command->flags ^ CPC_BIND_LWP_INHERIT;
    }
  else if (arg == 1)
    {
      command->flags = command->flags | CPC_BIND_LWP_INHERIT;
    }
  else
    return(PAPI_EINVAL);

  return(PAPI_OK);
#endif
}

static int set_default_domain(hwd_control_state_t *ctrl_state, int domain)
{
  /* This doesn't exist on this platform */

  if (domain == PAPI_DOM_OTHER)
    return(PAPI_EINVAL);

  return(set_domain(ctrl_state,domain)); 
}

static int set_default_granularity(hwd_control_state_t *current_state, int granularity)
{
  return(set_granularity(current_state,granularity));
}

/* Low level functions, should not handle errors, just return codes. */

/* this function is called by PAPI_library_init */
int _papi_hwd_init_global(void)
{
  int retval;

  /* Fill in what we can of the papi_system_info. */
  
  retval = get_system_info();
  if (retval)
    return(retval);
  
  DBG((stderr,"Found %d %s %s CPU's at %f Mhz.\n",
       _papi_hwi_system_info.hw_info.totalcpus,
       _papi_hwi_system_info.hw_info.vendor_string,
       _papi_hwi_system_info.hw_info.model_string,
       _papi_hwi_system_info.hw_info.mhz));

  return(PAPI_OK);
}

int _papi_hwd_init(hwd_context_t *zero)
{
  return(PAPI_OK);
}

u_long_long _papi_hwd_get_real_usec (void)
{
  return((long long)gethrtime()/(long long)1000);
}

u_long_long _papi_hwd_get_real_cycles (void)
{
/*
  return(get_tick());
*/
  u_long_long usec, cyc;

  usec = _papi_hwd_get_real_usec();
  cyc = usec * (long long)_papi_hwi_system_info.hw_info.mhz;
  return((u_long_long)cyc);
}

u_long_long _papi_hwd_get_virt_usec (const hwd_context_t *zero)
{
  return((long long)gethrvtime()/(long long)1000);
}

u_long_long _papi_hwd_get_virt_cycles (const hwd_context_t *zero)
{
  return(_papi_hwd_get_virt_usec(NULL) * (long long)_papi_hwi_system_info.hw_info.mhz);
}

void _papi_hwd_error(int error, char *where)
{
  sprintf(where,"Substrate error: %s",strerror(error));
}

int _papi_hwd_add_prog_event(hwd_control_state_t *this_state, 
			     unsigned int event, void *extra, EventInfo_t *out)
{
  return(PAPI_ESBSTR);
}

/* reset the starting number */
int _papi_hwd_reset(hwd_context_t *ctx, hwd_control_state_t * ctrl)
{
  int retval;

  retval = cpc_take_sample(&ctrl->counter_cmd.cmd);
  if (retval == -1)
    return(PAPI_ESYS);
  ctrl->values[0] = ctrl->counter_cmd.cmd.ce_pic[0] ;
  ctrl->values[1] = ctrl->counter_cmd.cmd.ce_pic[1] ;

  return(PAPI_OK);
}


int _papi_hwd_read(hwd_context_t *ctx, hwd_control_state_t *ctrl, long_long **events)
{
  int retval;

  retval = cpc_take_sample(&ctrl->counter_cmd.cmd);
  if (retval == -1)
    return(PAPI_ESYS);
  ctrl->counter_cmd.cmd.ce_pic[0] -= (u_long_long)ctrl->values[0];
  ctrl->counter_cmd.cmd.ce_pic[1] -= (u_long_long)ctrl->values[1];

  *events = ctrl->counter_cmd.cmd.ce_pic;

  return PAPI_OK;
}

int _papi_hwd_setmaxmem(){
  return(PAPI_OK);
}

int _papi_hwd_ctl(hwd_context_t *ctx, int code, _papi_int_option_t *option)
{

  switch (code)
    {
    case PAPI_DEFDOM:
      return(set_default_domain(&option->domain.ESI->machdep, option->domain.domain));
    case PAPI_DOMAIN:
      return(set_domain(&option->domain.ESI->machdep, option->domain.domain));
    case PAPI_DEFGRN:
      return(set_default_granularity(&option->domain.ESI->machdep, option->granularity.granularity));
    case PAPI_GRANUL:
      return(set_granularity(&option->granularity.ESI->machdep, option->granularity.granularity));
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_write(hwd_context_t *ctx, hwd_control_state_t *ctrl, long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(hwd_context_t *ctx)
{
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
  (void)cpc_rele();
  return(PAPI_OK);
}

void _papi_hwd_dispatch_timer(int signal, siginfo_t *si, void *info)
{
/*
  DBG((stderr,"_papi_hwd_dispatch_timer() at 0x%lx\n", 
        info->uc_mcontext.gregs[REG_PC]));
*/
  _papi_hwi_context_t ctx;

  ctx.si=si;
  ctx.ucontext=info;
/*
  _papi_hwi_dispatch_overflow_signal((void *)info); 
*/
  _papi_hwi_dispatch_overflow_signal((void *)&ctx, _papi_hwi_system_info.supports_hw_overflow, 0, 0); 
}

int _papi_hwd_set_overflow(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  hwd_control_state_t *this_state = &ESI->machdep;
  papi_cpc_event_t *arg = &this_state->counter_cmd;
  int hwcntr;

  if (threshold == 0)
  {
    arg->flags ^= CPC_BIND_EMT_OVF;
    if (sigaction(SIGEMT, NULL, NULL) == -1)
	  return(PAPI_ESYS);
  }
  else
  {
    struct sigaction act;

    act.sa_sigaction = dispatch_emt;
    memset(&act.sa_mask,0x0,sizeof(act.sa_mask));
    act.sa_flags = SA_RESTART|SA_SIGINFO;
    if (sigaction(SIGEMT, &act, NULL) == -1)
	  return(PAPI_ESYS);


    arg->flags |= CPC_BIND_EMT_OVF;
    hwcntr = ESI->EventInfoArray[EventIndex].pos[0];
    if (hwcntr == 0)
      arg->cmd.ce_pic[0] = UINT64_MAX - (uint64_t)threshold;
    else if (hwcntr == 1)
      arg->cmd.ce_pic[1] = UINT64_MAX - (uint64_t)threshold;
  }

  return(PAPI_OK);
}

int _papi_hwd_set_profile(EventSetInfo_t *ESI, int EventIndex, int threshold)
{
  /* This function is not used and shouldn't be called. */

  return(PAPI_ESBSTR);
}

int _papi_hwd_stop_profiling(ThreadInfo_t *master, EventSetInfo_t *ESI)
{
  ESI->profile.overflowcount=0;
  return(PAPI_OK);
}


void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  ucontext_t *info = (ucontext_t *)context;
  location = (void *)info->uc_mcontext.gregs[REG_PC];

  return(location);
}

rwlock_t lock[PAPI_MAX_LOCK];

void _papi_hwd_lock_init(void)
{
}

int _papi_hwd_start(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
  int retval;

  retval = cpc_bind_event(&ctrl->counter_cmd.cmd,
              ctrl->counter_cmd.flags);
  if (retval == -1)
    return(PAPI_ESYS);

  retval = cpc_take_sample(&ctrl->counter_cmd.cmd);
  if (retval == -1)
    return(PAPI_ESYS);

  ctrl->values[0]= ctrl->counter_cmd.cmd.ce_pic[0];
  ctrl->values[1]= ctrl->counter_cmd.cmd.ce_pic[1];

  return(PAPI_OK);
}

int _papi_hwd_stop(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
  cpc_bind_event(NULL, 0);
  return PAPI_OK;
}

int _papi_hwd_remove_event(hwd_register_map_t *chosen, unsigned int hardware_index, hwd_control_state_t *out)
{
  return PAPI_OK;
}

int _papi_hwd_update_shlib_info(void)
{
  return PAPI_OK;
}

int _papi_hwd_encode_native(char *name, int *code)
{
  return(PAPI_OK);
}

int _papi_hwd_allocate_registers(EventSetInfo_t *ESI )
{
  return 1;
}

int _papi_hwd_ntv_enum_events(unsigned int *EventCode, int modifer)
{
 int index=*EventCode & NATIVE_AND_MASK;

  if (cpuver <= CPC_ULTRA2) {
    if(index < MAX_NATIVE_EVENT_USII-1 ) {
      *EventCode=*EventCode+1;
      return(PAPI_OK);
    } else  return(PAPI_ENOEVNT);
  } else 
    if (cpuver == CPC_ULTRA3) {
      if(index < MAX_NATIVE_EVENT-1 ) {
        *EventCode=*EventCode+1;
        return(PAPI_OK);
      } else  return(PAPI_ENOEVNT);
    };
  return(PAPI_ENOEVNT);
}

char * _papi_hwd_ntv_code_to_name(unsigned int EventCode)
{
  int nidx;

  nidx= EventCode^NATIVE_MASK;
  if (nidx>=0 && nidx<PAPI_MAX_NATIVE_EVENTS)
    return(native_table[nidx].name);
  return NULL;
}

char *_papi_hwd_ntv_code_to_descr(unsigned int EventCode)
{
  return(_papi_hwd_ntv_code_to_name(EventCode));
}

void _papi_hwd_init_control_state(hwd_control_state_t *ptr)
{
  ptr->counter_cmd.flags = 0x0;
  ptr->counter_cmd.cmd.ce_cpuver = cpuver;
  ptr->counter_cmd.cmd.ce_pcr = 0x0;
  ptr->counter_cmd.cmd.ce_pic[0]=0;
  ptr->counter_cmd.cmd.ce_pic[1]=0;
  set_domain(ptr,_papi_hwi_system_info.default_domain);
  set_granularity(ptr,_papi_hwi_system_info.default_granularity);
  return;
}

int _papi_hwd_update_control_state(hwd_control_state_t *this_state, NativeInfo_t *native, int count)
{
  int i, nidx1, nidx2, hwcntr;
  uint64_t tmp , cmd0, cmd1, pcr;

/* save the last three bits */
  pcr = this_state->counter_cmd.cmd.ce_pcr & 0x7;

/* clear the control register */
  this_state->counter_cmd.cmd.ce_pcr = pcr;

/* no native events left */
  if (count == 0) return(PAPI_OK);

  cmd0 = -1;
  cmd1 = -1;
/* one native event */
  if (count == 1) 
  {
    nidx1 = native[0].ni_event & NATIVE_AND_MASK;
    hwcntr=0;
    cmd0 = native_table[nidx1].encoding[0];
    native[0].ni_position = 0;
    if ( cmd0 == -1 )
    {
      cmd1 = native_table[nidx1].encoding[1];
      native[0].ni_position = 1;
    }
    tmp = 0;
  }

/* two native events */
  if ( count == 2) 
  {
    int avail1, avail2;

    avail1=0;
    avail2=0;
    nidx1 = native[0].ni_event & NATIVE_AND_MASK;
    nidx2 = native[1].ni_event & NATIVE_AND_MASK;
    if ( native_table[nidx1].encoding[0] != -1 )
      avail1= 0x1;
    if ( native_table[nidx1].encoding[1] != -1 )
      avail1 += 0x2;
    if ( native_table[nidx2].encoding[0] != -1 )
      avail2= 0x1;
    if ( native_table[nidx2].encoding[1] != -1 )
      avail2 += 0x2;
    if ( ( avail1 | avail2 ) != 0x3 )
      return(PAPI_ECNFLCT);   
    if (avail1 == 0x3 )
    {
      if (avail2 == 0x1)
      {
        cmd0 = native_table[nidx2].encoding[0];
        cmd1 = native_table[nidx1].encoding[1];
        native[0].ni_position = 1;
        native[1].ni_position = 0;
      }
      else 
      {
        cmd1 = native_table[nidx2].encoding[1];
        cmd0 = native_table[nidx1].encoding[0];
        native[0].ni_position = 0;
        native[1].ni_position = 1;
      }
    }
    else
    {
      if (avail1 == 0x1)
      {
        cmd0 = native_table[nidx1].encoding[0];
        cmd1 = native_table[nidx2].encoding[1];
        native[0].ni_position = 0;
        native[1].ni_position = 1;
      }
      else
      {
        cmd0 = native_table[nidx2].encoding[0];
        cmd1 = native_table[nidx1].encoding[1];
        native[0].ni_position = 1;
        native[1].ni_position = 0;
      }
    }
  }

/* set the control register */
  if (cmd0 != -1)
  {
    tmp = ((uint64_t)cmd0 << pcr_shift[0]);
  }
  if (cmd1!= -1)
  {
    tmp = tmp | ((uint64_t)cmd1 << pcr_shift[1]);
  }
  this_state->counter_cmd.cmd.ce_pcr = tmp | pcr;
#if DEBUG
    dump_cmd(&this_state->counter_cmd);
#endif

  return(PAPI_OK);
}


int _papi_hwd_bpt_map_avail(hwd_reg_alloc_t *dst, int ctr)
{
  return(PAPI_OK);
}

/* This function forces the event to
    be mapped to only counter ctr.
    Returns nothing.
*/
void _papi_hwd_bpt_map_set(hwd_reg_alloc_t *dst, int ctr)
{
}

/* This function examines the event to determine
    if it has a single exclusive mapping.
    Returns true if exlusive, false if non-exclusive.
*/
int _papi_hwd_bpt_map_exclusive(hwd_reg_alloc_t *dst)
{
  return(PAPI_OK);
}

/* This function compares the dst and src events
    to determine if any counters are shared. Typically the src event
    is exclusive, so this detects a conflict if true.
    Returns true if conflict, false if no conflict.
*/
int _papi_hwd_bpt_map_shared(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
  return(PAPI_OK);
}

/* This function removes the counters available to the src event
    from the counters available to the dst event,
    and reduces the rank of the dst event accordingly. Typically,
    the src event will be exclusive, but the code shouldn't assume it.
    Returns nothing.
*/
void _papi_hwd_bpt_map_preempt(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
}

/* This function updates the selection status of
    the dst event based on information in the src event.
    Returns nothing.
*/
void _papi_hwd_bpt_map_update(hwd_reg_alloc_t *dst, hwd_reg_alloc_t *src)
{
}

