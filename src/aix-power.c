/* This substrate should never malloc anything. All allocation should be
   done by the high level API. */

#include "aix-power.h"

static hwd_preset_t preset_map[PAPI_MAX_PRESET_EVENTS] = { 0 };

static pmapi_search_t preset_name_map_604[PAPI_MAX_PRESET_EVENTS] = {
  /* L1 Cache Dmisses */
  {0,{"PM_DC_MISS",0,0,0,0,0,0,0}},		
  /* L1 Cache Imisses */
  {0,{"PM_IC_MISS",0,0,0,0,0,0,0}},		
  /* L2 Cache Dmisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L2 Cache Imisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L3 Cache Dmisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L3 Cache Imisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L1 Total Cache misses */
  {DERIVED_ADD,{"PM_DC_MISS","PM_IC_MISS",0,0,0,0,0,0}},
  /* L2 Total Cache misses*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* L3 Total Cache misses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* Req. for snoop*/
  {0,{"PM_SNOOP_RECV",0,0,0,0,0,0,0}},	
  /* Req. shared cache line*/
  {0,{0,0,0,0,0,0,0,0}},		 	
  /* Req. clean cache line*/
  {0,{0,0,0,0,0,0,0,0}},		 	
  /* Req. invalidate cache line*/
  {0,{0,0,0,0,0,0,0,0}},		 	
  /* Req. intervention cache line*/
  {0,{0,0,0,0,0,0,0,0}},		 	
  /* L3 Load misses*/
  {0,{0,0,0,0,0,0,0,0}},		
  /* L3 Store misses*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* BRU idle cycles*/
  {0,{0,0,0,0,0,0,0,0}},		
  /* FXU idle cycles*/
  {0,{0,0,0,0,0,0,0,0}},		
  /* FPU idle cycles*/
  {0,{0,0,0,0,0,0,0,0}},          
  /* LSU idle cycles*/
  {0,{0,0,0,0,0,0,0,0}},          
  /* D-TLB misses*/
  {0,{"PM_DTLB_MISS",0,0,0,0,0,0,0}},		
  /* I-TLB misses*/
  {0,{"PM_ITLB_MISS",0,0,0,0,0,0,0}},		
  /* Total TLB misses*/
  {DERIVED_ADD,{"PM_DTLB_MISS","PM_ITLB_MISS",0,0,0,0,0,0}},		
  /* L1LM*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* L1SM*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* L2 Load misses */
  {0,{"PM_LD_MISS_EXCEED_L2",0,0,0,0,0,0,0}},			
  /* L2 Store misses */
  {0,{"PM_ST_MISS_EXCEED_L2",0,0,0,0,0,0,0}},			
  /* Btacmiss*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* prefmiss*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* L3DCH*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* TLB shootdowns*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Suc. store conditional instructions*/
  {0,{"PM_RESRV_CMPL",0,0,0,0,0,0,0}},	
  /* Failed store conditional instructions*/
  {DERIVED_SUB,{"PM_RESRV_RQ","PM_RESRV_CMPL",0,0,0,0,0,0}},	
  /* Total store conditional instructions*/
  {0,{"PM_RESRV_RQ",0,0,0,0,0,0,0}},			
  /* Cycles stalled waiting for memory */
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cycles stalled waiting for memory read */
  {0,{"PM_LD_MISS_CYC",0,0,0,0,0,0,0}},   	
  /* Cycles stalled waiting for memory write */
  {0,{0,0,0,0,0,0,0,0}},   	
  /* Cycles no instructions issued */
  {0,{0,0,0,0,0,0,0,0}},	
  /* Cycles max instructions issued */
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cycles no instructions completed */
  {0,{0,0,0,0,0,0,0,0}},	
  /* Cycles max instructions completed */
  {0,{0,0,0,0,0,0,0,0}},		
  /* Hardware interrupts */
  {0,{0,0,0,0,0,0,0,0}},		
  /* Uncond. branches executed*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. Branch inst. executed*/
  {0,{"PM_BR_FINISH",0,0,0,0,0,0,0}},		
  /* Cond. Branch inst. taken*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. Branch inst. not taken*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. branch inst. mispred.*/
  {0,{"PM_BR_MPRED",0,0,0,0,0,0,0}},          
  /* Cond. branch inst. pred. */
  {0,{0,0,0,0,0,0,0,0}},		
  /* FMA's */
  {0,{0,0,0,0,0,0,0,0}},		
  /* Total inst. issued*/
  {0,{"PM_INST_DISP",0,0,0,0,0,0,0}},		
  /* Total inst. executed*/
  {0,{"PM_INST_CMPL",0,0,0,0,0,0,0}},		
  /* Integer inst. executed*/
  {0,{"PM_FXU_CMPL",0,0,0,0,0,0,0}},		
  /* Floating Pt. inst. executed*/
  {0,{"PM_FPU_CMPL",0,0,0,0,0,0,0}},	        
  /* Loads executed*/
  {0,{"PM_LD_CMPL",0,0,0,0,0,0,0}},		
  /* Stores executed*/
  {0,{0,0,0,0,0,0,0,0}},		
  /* Branch inst. executed*/
  {0,{"PM_BR_CMPL",0,0,0,0,0,0,0}},	
  /* Vector/SIMD inst. executed */
  {0,{0,0,0,0,0,0,0,0}},			
  /* FLOPS */
  {DERIVED_PS,{"PM_CYC","PM_FPU_CMPL",0,0,0,0,0,0}},			
  /* 58*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cycles FP units are stalled*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Total cycles */
  {0,{"PM_CYC",0,0,0,0,0,0,0}},		
  /* IPS */
  {DERIVED_PS,{"PM_CYC","PM_INST_CMPL",0,0,0,0,0,0}},			
  /* load/store*/
  {0,{"PM_LS_EXEC",0,0,0,0,0,0,0}},		
  /* Synchronization inst. executed*/
  {0,{"PM_SYNC",0,0,0,0,0,0,0}},		
  /* L1 data cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP mult */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP add */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP Div */
  { 0,{0,0,0,0,0,0,0,0}},
  /* FP Sqrt */
  { 0,{0,0,0,0,0,0,0,0}},
  /* FP inv */
  {0,{0,0,0,0,0,0,0,0}}
};

static pmapi_search_t preset_name_map_604e[PAPI_MAX_PRESET_EVENTS] = {
  /* L1 Cache Dmisses */
  {0,{"PM_DC_MISS",0,0,0,0,0,0,0}},		
  /* L1 Cache Imisses */
  {0,{"PM_IC_MISS",0,0,0,0,0,0,0}},		
  /* L2 Cache Dmisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L2 Cache Imisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L3 Cache Dmisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L3 Cache Imisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L1 Total Cache misses */
  {DERIVED_ADD,{"PM_DC_MISS","PM_IC_MISS",0,0,0,0,0,0}},
  /* L2 Total Cache misses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L3 Total Cache Dmisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* Req. for snoop*/
  {0,{"PM_SNOOP_RECV",0,0,0,0,0,0,0}},	
  /* Req. shared cache line*/
  {0,{"PM_LD_MISS_DC_SHR",0,0,0,0,0,0,0}},		 	
  /* Req. clean cache line*/
  {0,{0,0,0,0,0,0,0,0}},		 	
  /* Req. invalidate cache line*/
  {0,{"PM_WR_HIT_SHR_KILL_BRC",0,0,0,0,0,0,0}},		 	
  /* Req. intervention cache line*/
  {0,{"PM_WR_HIT_SHR_KILL_BRC",0,0,0,0,0,0,0}},		 	
  /* L3 load misses */
  {0,{0,0,0,0,0,0,0,0}},			
  /* L3 store misses */
  {0,{0,0,0,0,0,0,0,0}},			
  /* BRU idle cycles*/
  {0,{"PM_BRU_IDLE",0,0,0,0,0,0,0}},		
  /* FXU idle cycles*/
  {0,{"PM_MCI_IDLE",0,0,0,0,0,0,0}},		
  /* FPU idle cycles*/
  {0,{"PM_FPU_IDLE",0,0,0,0,0,0,0}},          
  /* LSU idle cycles*/
  {0,{"PM_LSU_IDLE",0,0,0,0,0,0,0}},          
  /* D-TLB misses*/
  {0,{"PM_DTLB_MISS",0,0,0,0,0,0,0}},		
  /* I-TLB misses*/
  {0,{"PM_ITLB_MISS",0,0,0,0,0,0,0}},		
  /* Total TLB misses*/
  {DERIVED_ADD,{"PM_DTLB_MISS","PM_ITLB_MISS",0,0,0,0,0,0}},
  /* L1 Load misses */
  {0,{0,0,0,0,0,0,0,0}},			
  /* L1 Store misses */
  {0,{0,0,0,0,0,0,0,0}},			
  /* L2 Load misses */
  {0,{"PM_LD_MISS_EXCEED_L2",0,0,0,0,0,0,0}},			
  /* L2 Store misses */
  {0,{"PM_ST_MISS_EXCEED_L2",0,0,0,0,0,0,0}},			
  /* BTACmiss*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Prefmiss*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* L3DCH*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* TLB shootdowns*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Suc. store conditional instructions*/
  {0,{"PM_RESRV_CMPL",0,0,0,0,0,0,0}},	
  /* Failed store conditional instructions*/
  {DERIVED_SUB,{"PM_RESRV_RQ","PM_RESRV_CMPL",0,0,0,0,0,0}},	
  /* Total store conditional instructions*/
  {0,{"PM_RESRV_RQ",0,0,0,0,0,0,0}},			
  /* Cycles stalled waiting for memory */
  {DERIVED_ADD,{"PM_CMPLU_WT_LD","PM_CMPLU_WT_ST",0,0,0,0,0,0}},
  /* Cycles stalled waiting for memory read */
  {0,{"PM_CMPLU_WT_LD",0,0,0,0,0,0,0}},   	
  /* Cycles stalled waiting for memory write */
  {0,{"PM_CMPLU_WT_ST",0,0,0,0,0,0,0}},   	
  /* Cycles no/min instructions issued */
  {0,{"PM_1INST_DISP",0,0,0,0,0,0,0}},	
  /* Cycles max instructions issued */
  {0,{"PM_4INST_DISP",0,0,0,0,0,0,0}},	
  /* Cycles no/min instructions completed */
  {0,{"PM_1INST_DISP",0,0,0,0,0,0,0}},	
  /* Cycles max instructions completed */
  {0,{"PM_4INST_DISP",0,0,0,0,0,0,0}},	
  /* Hardware interrupts */
  {0,{0,0,0,0,0,0,0,0}},		
  /* Uncond. branches executed*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. Branch inst. executed*/
  {0,{"PM_BR_FINISH",0,0,0,0,0,0,0}},		
  /* Cond. Branch inst. taken*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. Branch inst. not taken*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. branch inst. mispred.*/
  {0,{"PM_BR_MPRED",0,0,0,0,0,0,0}},          
  /* Cond. branch inst. pred. */
  {0,{0,0,0,0,0,0,0,0}},		
  /* FMA's */
  {0,{0,0,0,0,0,0,0,0}},		
  /* Total inst. issued*/
  {0,{"PM_INST_DISP",0,0,0,0,0,0,0}},		
  /* Total inst. executed*/
  {0,{"PM_INST_CMPL",0,0,0,0,0,0,0}},		
  /* Integer inst. executed*/
  {0,{"PM_FXU_CMPL",0,0,0,0,0,0,0}},		
  /* Floating Pt. inst. executed*/
  {0,{"PM_FPU_CMPL",0,0,0,0,0,0,0}},	        
  /* Loads executed*/
  {0,{"PM_LD_CMPL",0,0,0,0,0,0,0}},		
  /* Stores executed*/
  {0,{0,0,0,0,0,0,0,0}},		
  /* Branch inst. executed*/
  {0,{"PM_BR_CMPL",0,0,0,0,0,0,0}},		
  /* Vector/SIMD inst. executed */
  {0,{0,0,0,0,0,0,0,0}},			
  /* FLOPS */
  {DERIVED_PS,{"PM_CYC","PM_FPU_CMPL",0,0,0,0,0,0}},			
  /* CPU stall cycles*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cycles FP units are stalled*/
  {0,{"PM_FPU_WT",0,0,0,0,0,0,0}},		
  /* Total cycles */
  {0,{"PM_CYC",0,0,0,0,0,0,0}},		
  /* IPS */
  {DERIVED_PS,{"PM_CYC","PM_INST_CMPL",0,0,0,0,0,0}},
  /* load/stores executed*/
  {0,{"PM_LS_EXEC",0,0,0,0,0,0,0}},		
  /* Synchronization inst. executed */
  {0,{"PM_SYNC",0,0,0,0,0,0,0}},		
  /* L1 data cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP mult */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP add */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP Div */
  { 0,{0,0,0,0,0,0,0,0}},
  /* FP Sqrt */
  { 0,{0,0,0,0,0,0,0,0}},
  /* FP inv */
  {0,{0,0,0,0,0,0,0,0}}
};

static pmapi_search_t preset_name_map_630[PAPI_MAX_PRESET_EVENTS] = { 
  /* L1 Cache Dmisses */
  {DERIVED_ADD,{"PM_LD_MISS_L1","PM_ST_MISS",0,0,0,0,0,0}},
  /* L1 Cache Imisses */
  {0,{"PM_IC_MISS",0,0,0,0,0,0,0}},		
  /* L2 Cache Dmisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L2 Cache Imisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L3 Cache Dmisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L3 Cache Imisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* L1 Total Cache misses */
  {DERIVED_ADD,{"PM_IC_MISS","PM_LD_MISS_L1","PM_ST_MISS",0,0,0,0,0}},
  /* L2 Total Cache misses*/
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 Total Cache Dmisses*/
  {0,{0,0,0,0,0,0,0,0}}, 			
  /* Req. for snoop*/
  {0,{"PM_SNOOP",0,0,0,0,0,0,0}},		
  /* Req. shared cache line*/
  {0,{"PM_SNOOP_E_TO_S",0,0,0,0,0,0,0}},		 	
  /* Req. clean cache line*/
  {0,{0,0,0,0,0,0,0,0}},		 	
  /* Req. invalidate cache line*/
  {0,{0,0,0,0,0,0,0,0}},		 	
  /* Req. intervention cache line*/
  {0,{"PM_SNOOP_PUSH_INT",0,0,0,0,0,0,0}},		 	
  /* L3 load misses */
  {0,{0,0,0,0,0,0,0,0}},	
  /* L3 store misses */
  {0,{0,0,0,0,0,0,0,0}},	
  /* BRU idle cycles*/
  {0,{"PM_BRU_IDLE",0,0,0,0,0,0,0}},		
  /* FXU idle cycles*/
  {0,{"PM_FXU_IDLE",0,0,0,0,0,0,0}},		
  /* FPU idle cycles*/
  {0,{"PM_FPU_IDLE",0,0,0,0,0,0,0}},          
  /* LSU idle cycles*/
  {0,{"PM_LSU_IDLE",0,0,0,0,0,0,0}},          
  /* D-TLB misses*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* I-TLB misses*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Total TLB misses*/
  {0,{"PM_TLB_MISS",0,0,0,0,0,0,0}},		
  /* L1 Load misses */
  {0,{"PM_LD_MISS_L1",0,0,0,0,0,0,0}},			
  /* L1 Store misses */
  {0,{"PM_ST_MISS",0,0,0,0,0,0,0}},			
  /* L2 Load misses */
  {0,{"PM_LD_MISS_EXCEED_L2",0,0,0,0,0,0,0}},			
  /* L2 Store misses */
  {0,{"PM_ST_MISS_EXCEED_L2",0,0,0,0,0,0,0}},			
  /* BTAC misses*/
  {0,{"PM_BTAC_MISS",0,0,0,0,0,0,0}},			
  /* unused */
  {0,{"PM_PREF_MATCH_DEM_MISS",0,0,0,0,0,0,0}},			
  /* unused */
  {0,{0,0,0,0,0,0,0,0}},			
  /* TLB shootdowns*/
  {0,{"PM_TLBSYNC_RERUN",0,0,0,0,0,0,0}},			
  /* Suc. store conditional instructions*/
  {0,{"PM_RESRV_CMPL",0,0,0,0,0,0,0}},	
  /* Failed store conditional instructions*/
  {0,{"PM_ST_COND_FAIL",0,0,0,0,0,0,0}},	
  /* Total store conditional instructions*/
  {0,{"PM_RESRV_RQ",0,0,0,0,0,0}},			
  /* Cycles stalled waiting for memory */
  {DERIVED_ADD,{"PM_CMPLU_WT_LD","PM_CMPLU_WT_ST",0,0,0,0,0,0}},
  /* Cycles stalled waiting for memory read */
  {0,{"PM_CMPLU_WT_LD",0,0,0,0,0,0,0}},   	
  /* Cycles stalled waiting for memory write */
  {0,{"PM_CMPLU_WT_ST",0,0,0,0,0,0,0}},   	
  /* Cycles no instructions issued */
  {0,{"PM_0INST_DISP",0,0,0,0,0,0,0}},	
  /* Cycles max instructions issued */
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cycles no instructions completed */
  {0,{"PM_0INST_CMPL",0,0,0,0,0,0,0}},	
  /* Cycles max instructions completed */
  {0,{0,0,0,0,0,0,0,0}},		
  /* Hardware interrupts */
  {0,{0,0,0,0,0,0,0,0}},		
  /* Uncond. branches executed*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. Branch inst. executed*/
  {0,{"PM_CBR_DISP",0,0,0,0,0,0}},		        
  /* Cond. Branch inst. taken*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. Branch inst. not taken*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cond. branch inst. mispred.*/
  {0,{"PM_MPRED_BR_CAUSED_GC",0,0,0,0,0,0,0}},
  /* Cond. branch inst. pred. */
  {0,{"PM_BR_PRED",0,0,0,0,0,0,0}},		
  /* FMA's */
  {0,{"PM_EXEC_FMA",0,0,0,0,0,0,0}},		
  /* Total inst. issued*/
  {0,{"PM_INST_DISP",0,0,0,0,0,0,0}},		
  /* Total inst. executed*/
  {0,{"PM_INST_CMPL",0,0,0,0,0,0,0}},		
  /* Integer inst. executed*/
  {DERIVED_ADD,{"PM_FXU0_PROD_RESULT","PM_FXU1_PROD_RESULT","PM_FXU2_PROD_RESULT",0,0,0,0,0}},
  /* Floating Pt. inst. executed*/
  {DERIVED_ADD,{"PM_FPU0_CMPL","PM_FPU1_CMPL",0,0,0,0,0,0}},	
  /* Loads executed*/
  {0,{"PM_LD_CMPL",0,0,0,0,0,0,0}},		
  /* Stores executed*/
  {0,{"PM_ST_CMPL",0,0,0,0,0,0,0}},		
  /* Branch inst. executed*/
  {0,{"PM_BR_CMPL",0,0,0,0,0,0,0}},		
  /* Vector/SIMD inst. executed */
  {0,{0,0,0,0,0,0,0,0}},			
  /* FLOPS */
  {DERIVED_ADD_PS,{"PM_CYC","PM_FPU0_CMPL","PM_FPU1_CMPL",0,0,0,0,0}},
  /* 58*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Cycles FP units are stalled*/
  {0,{0,0,0,0,0,0,0,0}},			
  /* Total cycles */
  {0,{"PM_CYC",0,0,0,0,0,0,0}},		
  /* IPS */
  {DERIVED_PS,{"PM_CYC","PM_INST_CMPL",0,0,0,0,0,0}},
  /* load/stores */
  {DERIVED_ADD,{"PM_LD_CMPL","PM_ST_CMPL",0,0,0,0,0,0}},
  /* Synchronization inst. executed */
  {0,{"PM_SYNC",0,0,0,0,0,0,0}},		
  /* L1 data cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 data cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 instruction cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache hits */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache accesses */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache reads */
  {0,{0,0,0,0,0,0,0,0}},
  /* L1 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L2 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* L3 total cache writes */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP mult */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP add */
  {0,{0,0,0,0,0,0,0,0}},
  /* FP Div */
  { 0,{"PM_FPU_FDIV",0,0,0,0,0,0,0}},
  /* FP Sqrt */
  { 0,{"PM_FPU_FSQRT",0,0,0,0,0,0,0}},
  /* FP inv */
  {0,{0,0,0,0,0,0,0,0}}
};

/* Utility functions */

/* Find all the hwcntrs that name lives on */

static int find_hwcounter(pm_info_t *info, char *name, hwd_preset_t *preset)
{
  int index, did_something = 0, pmc, ev, found = 0;
  pm_events_t *wevp;
  char *note = NULL;

  for (pmc = 0; pmc < info->maxpmcs; pmc++) 
    {
      wevp = info->list_events[pmc];
      for (ev = 0; ev < info->maxevents[pmc]; ev++, wevp++) 
	{
	  if (strcmp(name, wevp->short_name) == 0) 
	    {
	      preset->counter_cmd[pmc] = wevp->event_id;
	      preset->selector |= 1 << pmc;
	      did_something++;
	      DBG((stderr,"Found %s on hardware counter %d, event %d\n",name,pmc,wevp->event_id));
	      break;
	    }
	}
    }

  if (did_something)
    {
      strncpy(preset->note,name,PAPI_MAX_STR_LEN);
      return(1);
    }
  else
    abort();
}

static int setup_all_presets(pm_info_t *info)
{
  pmapi_search_t *findem;
  int pnum,did_something = 0,pmc,derived;
  
  if (__power_630())
    findem = preset_name_map_630;
  else if (__power_604())
    {
      if (strstr(info->proc_name,"604e"))
	findem = preset_name_map_604e;
      else
	findem = preset_name_map_604;
    }
  else
    return(PAPI_ESBSTR);

  for (pnum = 0; pnum < PAPI_MAX_PRESET_EVENTS; pnum++)
    {
      /* Events are always stored in the first slot onward*/
      if (findem[pnum].findme[0])
	{
	  /* If we have a name for it and not derived */
	  if (findem[pnum].derived == 0)
	    {
	      /* If we find it, then on to the next preset */
	      DBG((stderr,"Looking for preset %d, %s\n",pnum,findem[pnum].findme[0]));
	      find_hwcounter(info,findem[pnum].findme[0],&preset_map[pnum]);
	      did_something++;
	    }
	  else 
	    {
	      hwd_preset_t tmp;
	      int free_hwcntrs, need_one_hwcntr, hwcntr_num, err = 0, all_selector = 0, first = -1;
	      unsigned char all_command[MAX_COUNTERS];
	      char note[PAPI_MAX_STR_LEN];
	      
	      pmc = 0;
	      note[0] = '\0';
	      memset(all_command,0x00,sizeof(unsigned char)*MAX_COUNTERS);
	      DBG((stderr,"Looking for preset %d, compound event\n",pnum,findem[pnum].findme[0]));
	      while (findem[pnum].findme[pmc])
		{
		  memset(&tmp,0x00,sizeof(tmp));
		  DBG((stderr,"Looking for preset %d, %s\n",pnum,findem[pnum].findme[pmc]));
		  if (find_hwcounter(info,findem[pnum].findme[pmc],&tmp) == 0)
		    {
		      err = 1;
		      break;
		    }
		  /* tmp.selector now contains all the counters with findme */
		  /* first, find what's currently available */
		  free_hwcntrs = ~all_selector;
		  /* second, of those available, what can we choose */
		  need_one_hwcntr = free_hwcntrs & tmp.selector;
		  if (need_one_hwcntr == 0)
		    {
		      err = 1;
		      break;
		    }
		  /* third, pick one */
		  hwcntr_num = get_avail_hwcntr_num(need_one_hwcntr);
		  need_one_hwcntr = 1 << hwcntr_num;
		  /* fourth, add it to our set */
		  all_selector |= need_one_hwcntr;
		  all_command[hwcntr_num] = tmp.counter_cmd[hwcntr_num];
		  if (strlen(note)+strlen(findem[pnum].findme[pmc]+1) < PAPI_MAX_STR_LEN)
		    {
		      strcat(note,findem[pnum].findme[pmc]);
		      strcat(note,",");
		    }
		  /* Fifth, if it's the first one, then set the operand index */
		  if (first == -1)
		    first = hwcntr_num;
		  /* On to the next register */
		  pmc++;
		}

	      /* If we're successful */
	      
	      if (err == 0)
		{
		  memcpy(preset_map[pnum].counter_cmd,all_command,sizeof(unsigned char)*MAX_COUNTERS);
		  preset_map[pnum].selector = all_selector;
		  preset_map[pnum].derived = findem[pnum].derived;
		  preset_map[pnum].operand_index = first;
		  note[strlen(note)-1] = '\0';
		  strcpy(preset_map[pnum].note,note);
		  DBG((stderr,"Found compound preset %d on 0x%x, first operand is %d\n",pnum,all_selector,first));
		  did_something++;
		  continue;
		}
	      fprintf(stderr,"Did not find compound preset %d on 0x%x\n",pnum,all_selector);	  
	      abort();
	    }
	}
    }
  return(did_something ? 0 : PAPI_ESBSTR);
}

static void set_config(hwd_control_state_t *ptr, int arg1, int arg2)
{
  ptr->counter_cmd.events[arg1] = arg2;
}

static void unset_config(hwd_control_state_t *ptr, int arg1)
{
  ptr->counter_cmd.events[arg1] = 0;
}

static int counter_shared(hwd_control_state_t *a, hwd_control_state_t *b, int cntr)
{
  if (a->counter_cmd.events[cntr] == b->counter_cmd.events[cntr])
    return(1);

  return(0);
}

static int update_global_hwcounters(EventSetInfo *global)
{
  int i, retval;
  pm_data_t data;

  retval = pm_get_data_mythread(&data);
  if (retval > 0)
    return(retval);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      DBG((stderr,"update_global_hwcounters() %d: G%lld = G%lld + C%lld\n",i,
	   global->hw_start[i]+data.accu[i],global->hw_start[i],data.accu[i]));
      global->hw_start[i] = global->hw_start[i] + data.accu[i];
    }

  retval = pm_reset_data_mythread();
  if (retval > 0)
    return(retval);
   
  return(0);
}

static int correct_local_hwcounters(EventSetInfo *global, EventSetInfo *local, long long *correct)
{
  int i;

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      DBG((stderr,"correct_local_hwcounters() %d: L%lld = G%lld - L%lld\n",i,
	   global->hw_start[i]-local->hw_start[i],global->hw_start[i],local->hw_start[i]));
      correct[i] = global->hw_start[i] - local->hw_start[i];
    }

  return(0);
}

static int set_domain(hwd_control_state_t *this_state, int domain)
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

static int set_granularity(hwd_control_state_t *this_state, int domain)
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

static int set_default_domain(EventSetInfo *zero, int domain)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_domain(current_state,domain));
}

static int set_default_granularity(EventSetInfo *zero, int granularity)
{
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  return(set_granularity(current_state,granularity));
}

static int set_inherit(int arg)
{
  return(PAPI_ESBSTR);
}

static void init_config(hwd_control_state_t *ptr)
{
  ptr->counter_cmd.events[0] = -1;
  ptr->counter_cmd.events[1] = -1;
  ptr->counter_cmd.events[2] = -1;
  ptr->counter_cmd.events[3] = -1;
  ptr->counter_cmd.events[4] = -1;
  ptr->counter_cmd.events[5] = -1;
  ptr->counter_cmd.events[6] = -1;
  ptr->counter_cmd.events[7] = -1;
  set_domain(ptr,_papi_system_info.default_domain);
  set_granularity(ptr,_papi_system_info.default_granularity);
}

static int get_system_info(void)
{
  int retval;
  pm_info_t tmp;
  struct procsinfo psi = { 0 };
  pid_t pid;
  char maxargs[PAPI_MAX_STR_LEN];

  pid = getpid();
  if (pid == -1)
    return(PAPI_ESYS);
  psi.pi_pid = pid;
  retval = getargs(&psi,sizeof(psi),maxargs,PAPI_MAX_STR_LEN);
  if (retval == -1)
    return(PAPI_ESYS);
  if (getcwd(_papi_system_info.exe_info.fullname,PAPI_MAX_STR_LEN) == NULL)
    return(PAPI_ESYS);
  strcat(_papi_system_info.exe_info.fullname,"/");
  strcat(_papi_system_info.exe_info.fullname,maxargs);
  strncpy(_papi_system_info.exe_info.name,basename(maxargs),PAPI_MAX_STR_LEN);

  retval = pm_init(PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT,&tmp);
  if (retval > 0)
    return(retval);

  _papi_system_info.hw_info.ncpu = _system_configuration.ncpus;
  _papi_system_info.hw_info.totalcpus = 
    _papi_system_info.hw_info.ncpu * _papi_system_info.hw_info.nnodes;
  _papi_system_info.hw_info.vendor = -1;
  strcpy(_papi_system_info.hw_info.vendor_string,"IBM");
  _papi_system_info.hw_info.model = _system_configuration.implementation;
  strcpy(_papi_system_info.hw_info.model_string,tmp.proc_name);
  _papi_system_info.hw_info.revision = (float)_system_configuration.version;
  retval = pm_cycles() / 10000.0;
  _papi_system_info.hw_info.mhz = (float)(int)(retval / 100.0);
  _papi_system_info.num_gp_cntrs = tmp.maxpmcs;
  _papi_system_info.num_cntrs = tmp.maxpmcs;
  _papi_system_info.cpunum = mycpu();
  _papi_system_info.exe_info.text_end = (caddr_t)&_etext;
  retval = setup_all_presets(&tmp);
  if (retval)
    return(retval);

  return(PAPI_OK);
} 

/* Low level functions, should not handle errors, just return codes. */

/* At init time, the higher level library should always allocate and 
   reserve EventSet zero. */

long long _papi_hwd_get_real_usec (void)
{
  timebasestruct_t t;
  long long retval;

  read_real_time(&t,TIMEBASE_SZ);
  time_base_to_time(&t,TIMEBASE_SZ);
  retval = (t.tb_high * 1000000) + t.tb_low / 1000;
  return(retval);
}

long long _papi_hwd_get_real_cycles (void)
{
  float usec, cyc;

  usec = (float)_papi_hwd_get_real_usec();
  cyc = usec * _papi_system_info.hw_info.mhz;
  return((long long)cyc);
}

long long _papi_hwd_get_virt_usec (void)
{
  return(-1);
}

long long _papi_hwd_get_virt_cycles (void)
{
  return(-1);
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
  
  DBG((stderr,"Found %d %s %s CPU's at %f Mhz.\n",
       _papi_system_info.hw_info.totalcpus,
       _papi_system_info.hw_info.vendor_string,
       _papi_system_info.hw_info.model_string,
       _papi_system_info.hw_info.mhz));

  return(PAPI_OK);
}

int _papi_hwd_init(EventSetInfo *zero)
{
  /* Initialize our global machdep. */

  init_config(zero->machdep);

  return(PAPI_OK);
}

/* Go from highest counter to lowest counter. Why? Because there are usually
   more counters on #1, so we try the least probable first. */

static int get_avail_hwcntr_bits(int cntr_avail_bits)
{
  int tmp = 0, i = 1 << (MAX_COUNTERS-1);
  
  while (i)
    {
      tmp = i & cntr_avail_bits;
      if (tmp)
	return(tmp);
      i = i >> 1;
    }
  return(0);
}

static int get_avail_hwcntr_num(int cntr_avail_bits)
{
  int tmp = 0, i = MAX_COUNTERS - 1;
  
  while (i)
    {
      tmp = (1 << i) & cntr_avail_bits;
      if (tmp)
	return(i);
      i--;
    }
  return(0);
}

static void set_hwcntr_codes(int selector, unsigned char *from, int *to)
{
  int useme, i;
  
  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      useme = (1 << i) & selector;
      if (useme)
	{
	  to[i] = from[i];
	}
    }
}

int _papi_hwd_add_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector = 0;
  int avail = 0;
  unsigned char tmp_cmd[MAX_COUNTERS];
  unsigned char *codes;

  if (EventCode & PRESET_MASK)
    { 
      int preset_index;
      int derived;

      preset_index = EventCode ^ PRESET_MASK; 

      selector = preset_map[preset_index].selector;
      if (selector == 0)
	return(PAPI_ENOEVNT);
      derived = preset_map[preset_index].derived;

      /* Find out which counters are available. */

      avail = selector & ~this_state->selector;

      /* If not derived */

      if (preset_map[preset_index].derived == 0) 
	{
	  /* Pick any counter available */

	  selector = get_avail_hwcntr_bits(avail);
	  if (selector == 0)
	    return(PAPI_ECNFLCT);
	}    
      else
	{
	  /* Check the case that if not all the counters 
	     required for the derived event are available */

	  if ((avail & selector) != selector)
	    return(PAPI_ECNFLCT);	    
	}

      /* Get the codes used for this event */

      codes = preset_map[preset_index].counter_cmd;
      ESI->EventInfoArray[index].command = derived;
      ESI->EventInfoArray[index].operand_index = preset_map[preset_index].operand_index;
    }
  else
    {
      int hwcntr_num;

      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  /* 0 through 7 */ 
      if (hwcntr_num > _papi_system_info.num_gp_cntrs)
	return(PAPI_EINVAL);

      tmp_cmd[hwcntr_num] = EventCode >> 8; /* 0 through 50 */
      if (tmp_cmd[hwcntr_num] > 50)
	return(PAPI_EINVAL); 

      selector = 1 << hwcntr_num;

      /* Check if the counter is available */
      
      if (this_state->selector & selector)
	return(PAPI_ECNFLCT);	    

      codes = tmp_cmd;
    }

  /* Lower eight bits tell us what counters we need */

  assert((this_state->selector | 0xff) == 0xff);

  /* Perform any initialization of the control bits */

  if (this_state->selector == 0)
    init_config(this_state);
  
  /* Turn on the bits for this counter */

  set_hwcntr_codes(selector,codes,this_state->counter_cmd.events);

  /* Update the new counter select field. */

  this_state->selector |= selector;

  /* Inform the upper level that the software event 'index' 
     consists of the following information. */

  ESI->EventInfoArray[index].code = EventCode;
  ESI->EventInfoArray[index].selector = selector;

  return(PAPI_OK);
}

int _papi_hwd_rem_event(EventSetInfo *ESI, int index, unsigned int EventCode)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  int selector, used, preset_index;

  /* Find out which counters used. */
  
  used = ESI->EventInfoArray[index].selector;
 
  if (EventCode & PRESET_MASK)
    { 
      preset_index = EventCode ^ PRESET_MASK; 

      selector = preset_map[preset_index].selector;
      if (selector == 0)
	return(PAPI_ENOEVNT);
    }
  else
    {
      int hwcntr_num, code;
      
      /* Support for native events here, only 1 counter at a time. */

      hwcntr_num = EventCode & 0xff;  /* 0 through 7 */ 
      if (hwcntr_num > _papi_system_info.num_gp_cntrs)
	return(PAPI_EINVAL);

      code = EventCode >> 8; /* 0 through 50 */
      if (code > 50)
	return(PAPI_EINVAL); 

      selector = 1 << hwcntr_num;
    }

  /* Check if these counters aren't used. */

  if ((used & selector) != used)
    return(PAPI_EINVAL);

  /* Clear out counters that are part of this event. */
  /* Remember, that selector might encode duplicate events
     so we need to know only the ones that are used. */
  
  this_state->selector = this_state->selector ^ (selector & used);

  return(PAPI_OK);
}

int _papi_hwd_add_prog_event(EventSetInfo *ESI, int index, unsigned int event, void *extra)
{
  return(PAPI_ESBSTR);
}

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

/* EventSet zero contains the 'current' state of the counting hardware */

int _papi_hwd_merge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;
  
  /* If we are nested, merge the global counter structure
     with the current eventset */

  if (current_state->selector)
    {
      int hwcntrs_in_both, hwcntr;

      /* Stop the current context */

      retval = pm_stop_mythread();
      if (retval > 0) 
	return(retval); 
  
      /* Update the global values */

      retval = update_global_hwcounters(zero);
      if (retval)
	return(retval);

      /* Delete the current context */

      retval = pm_delete_program_mythread();
      if (retval > 0)
	return(retval);

      hwcntrs_in_both = this_state->selector & current_state->selector;

      for (i = 0; i < _papi_system_info.num_cntrs; i++)
	{
	  /* Check for events that are shared between eventsets and 
	     therefore require no modification to the control state. */
	  
	  hwcntr = 1 << i;
	  if (hwcntr & hwcntrs_in_both)
	    {
	      if (counter_shared(this_state, current_state, i))
		zero->multistart.SharedDepth[i]++;
	      else
		return(PAPI_ECNFLCT);
	      ESI->hw_start[i] = zero->hw_start[i];
	    }

	  /* Merge the unshared configuration registers. */
	  
	  else if (this_state->selector & hwcntr)
	    {
	      current_state->selector |= hwcntr;
	      current_state->counter_cmd.mode.w = this_state->counter_cmd.mode.w;
	      current_state->counter_cmd.events[i] = this_state->counter_cmd.events[i];
	      ESI->hw_start[i] = 0;
	      zero->hw_start[i] = 0;
	    }
	}
    }
  else
    {
      /* If we are NOT nested, just copy the global counter 
	 structure to the current eventset */

      current_state->selector = this_state->selector;
      memcpy(&current_state->counter_cmd,&this_state->counter_cmd,sizeof(pm_prog_t));

    }

  /* Set up the new merged control structure */
  
#ifdef DEBUG
  dump_cmd(&current_state->counter_cmd);
#endif
      
  retval = pm_set_program_mythread(&current_state->counter_cmd);
  if (retval > 0) 
    return(retval);

  /* (Re)start the counters */
  
  retval = pm_start_mythread();
  if (retval > 0) 
    return(retval);

  return(PAPI_OK);
} 

int _papi_hwd_unmerge(EventSetInfo *ESI, EventSetInfo *zero)
{ 
  int i, hwcntr, retval;
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;
  hwd_control_state_t *current_state = (hwd_control_state_t *)zero->machdep;

  retval = pm_stop_mythread();
  if (retval > 0) 
    return(retval); 
  
  for (i = 0; i < _papi_system_info.num_cntrs; i++)
    {
      /* Check for events that are NOT shared between eventsets and 
	 therefore require modification to the control state. */
      
      hwcntr = 1 << i;
      if (hwcntr & this_state->selector)
	{
	  if (zero->multistart.SharedDepth[i] - 1 < 0)
	    current_state->selector ^= hwcntr;
	  else
	    zero->multistart.SharedDepth[i]--;
	}
    }

  if (ESI->state & PAPI_OVERFLOWING)
    {
      retval = _papi_hwi_stop_overflow_timer(ESI, zero);
      if (retval < PAPI_OK)
	return(PAPI_EBUG);
    }
  
  /* If we're not the outermost EventSet, then we need to start again 
     because someone is still running. */

  if (zero->multistart.num_runners - 1)
    {
      retval = pm_start_mythread();
      if (retval > 0) 
	return(retval);
    }
  else
    {
      retval = pm_delete_program_mythread();
      if (retval > 0) 
	return(retval);
    }

  return(PAPI_OK);
}

int _papi_hwd_reset(EventSetInfo *ESI, EventSetInfo *zero)
{
  int i, retval;

  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    ESI->hw_start[i] = zero->hw_start[i];

  return(PAPI_OK);
}

static long long handle_derived_add(int selector, long long *from)
{
  int pos;
  long long retval = 0;

  while ((pos = ffs(selector)))
    {
      DBG((stderr,"Compound event, adding %lld to %lld\n",from[pos-1],retval));
      retval += from[pos-1];
      selector ^= 1 << pos-1;
    }
  return(retval);
}

static long long handle_derived_subtract(int operand_index, int selector, long long *from)
{
  int pos;
  long long retval = from[operand_index];

  selector = selector ^ (1 << operand_index);
  while (pos = ffs(selector))
    {
      DBG((stderr,"Compound event, subtracting %lld to %lld\n",from[pos-1],retval));
      retval -= from[pos-1];
      selector ^= 1 << pos-1;
    }
  return(retval);
}

static long long units_per_second(long long units, long long cycles)
{
  return((long long)((float)units * _papi_system_info.hw_info.mhz * 1000000.0 / (float)cycles));
}

static long long handle_derived_ps(int operand_index, int selector, long long *from)
{
  int pos;

  pos = ffs(selector ^ (1 << operand_index)) - 1;
  assert(pos >= 0);

  return(units_per_second(from[pos],from[operand_index]));
}

static long long handle_derived_add_ps(int operand_index, int selector, long long *from)
{
  int add_selector = selector ^ (1 << operand_index);
  long long tmp = handle_derived_add(add_selector, from);
  return(units_per_second(tmp, from[operand_index]));
}

static long long handle_derived(EventInfo_t *cmd, long long *from)
{
  switch (cmd->command)
    {
    case DERIVED_ADD: 
      return(handle_derived_add(cmd->selector, from));
    case DERIVED_ADD_PS:
      return(handle_derived_add_ps(cmd->operand_index, cmd->selector, from));
    case DERIVED_SUB:
      return(handle_derived_subtract(cmd->operand_index, cmd->selector, from));
    case DERIVED_PS:
      return(handle_derived_ps(cmd->operand_index, cmd->selector, from));
    default:
      abort();
    }
}

int _papi_hwd_read(EventSetInfo *ESI, EventSetInfo *zero, long long *events)
{
  int shift_cnt = 0;
  int retval, selector, j = 0, i;
  long long correct[MAX_COUNTERS];

  retval = update_global_hwcounters(zero);
  if (retval)
    return(retval);

  retval = correct_local_hwcounters(zero, ESI, correct);
  if (retval)
    return(retval);

  /* This routine distributes hardware counters to software counters in the
     order that they were added. Note that the higher level 
     EventInfoArray[i] entries may not be contiguous because the user
     has the right to remove an event. */

  for (i=0;i<_papi_system_info.num_cntrs;i++)
    {
      selector = ESI->EventInfoArray[i].selector;
      if (selector == PAPI_NULL)
	continue;

      assert(selector != 0);
      DBG((stderr,"Event index %d, selector is 0x%x\n",j,selector));

      /* If this is not a derived event */

      if (ESI->EventInfoArray[i].command == NOT_DERIVED)
	{
	  shift_cnt = ffs(selector) - 1;
	  assert(shift_cnt >= 0);
	  events[j] = correct[shift_cnt];
	}
      
      /* If this is a derived event */

      else 
	events[j] = handle_derived(&ESI->EventInfoArray[i], correct);
	
      /* Early exit! */

      if (++j == ESI->NumberOfCounters)
	return(PAPI_OK);
    }

  /* Should never get here */

  return(PAPI_EBUG);
}

int _papi_hwd_ctl(EventSetInfo *zero, int code, _papi_int_option_t *option)
{
  switch (code)
    {
    case PAPI_SET_DEFDOM:
      return(set_default_domain(zero, option->domain.domain));
    case PAPI_SET_DOMAIN:
      return(set_domain(option->domain.ESI->machdep, option->domain.domain));
    case PAPI_SET_DEFGRN:
      return(set_default_granularity(zero, option->granularity.granularity));
    case PAPI_SET_GRANUL:
      return(set_granularity(option->granularity.ESI->machdep, option->granularity.granularity));
    case PAPI_SET_INHERIT:
      return(set_inherit(option->inherit.inherit));
    default:
      return(PAPI_EINVAL);
    }
}

int _papi_hwd_write(EventSetInfo *master, EventSetInfo *ESI, long long events[])
{ 
  return(PAPI_ESBSTR);
}

int _papi_hwd_shutdown(EventSetInfo *zero)
{
  pm_delete_program_mythread();
  return(PAPI_OK);
}

int _papi_hwd_shutdown_global(void)
{
  return(PAPI_OK);
}

int _papi_hwd_query(int preset_index, int *flags, char **note)
{ 
  if (preset_map[preset_index].selector == 0)
    return(0);
  if (preset_map[preset_index].derived)
    *flags = PAPI_DERIVED;
  if (preset_map[preset_index].note)
    *note = preset_map[preset_index].note;
  return(1);
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

int _papi_hwd_set_overflow(EventSetInfo *ESI, EventSetOverflowInfo_t *overflow_option)
{
  hwd_control_state_t *this_state = (hwd_control_state_t *)ESI->machdep;

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

int _papi_hwd_set_profile(EventSetInfo *ESI, EventSetProfileInfo_t *profile_option)
{
  /* This function is not used and shouldn't be called. */

  abort();
}

void *_papi_hwd_get_overflow_address(void *context)
{
  void *location;
  struct sigcontext *info = (struct sigcontext *)context;
  location = (void *)info->sc_jmpbuf.jmp_context.iar;

  return(location);
}

static volatile int lock_var = 0;
static volatile atomic_p lock;

void _papi_hwd_lock_init(void)
{
  lock = &lock_var;
}

void _papi_hwd_lock(void)
{
  while (_check_lock(lock,0,1) == TRUE)
    {
      DBG((stderr,"Waiting..."));
      usleep(1000);
    }
}

void _papi_hwd_unlock(void)
{
  _clear_lock(lock, 0);
}

/* Machine info structure. -1 is initialized by _papi_hwd_init. */

papi_mdi _papi_system_info = { "$Id$",
			      1.0, /*  version */
			       -1,  /*  cpunum */
			       { 
				 -1,  /*  ncpu */
				  1,  /*  nnodes */
				 -1,  /*  totalcpus */
				 -1,  /*  vendor */
				 "",  /*  vendor string */
				 -1,  /*  model */
				 "",  /*  model string */
				0.0,  /*  revision */
				 -1  /*  mhz */ 
			       },
			       {
				 "",
				 "",
				 (caddr_t)0x10000200,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 (caddr_t)-1,
				 ""
			       },
			       -1,  /*  num_cntrs */
			       -1,  /*  num_gp_cntrs */
			       -1,  /*  grouped_counters */
			       -1,  /*  num_sp_cntrs */
			       -1,  /*  total_presets */
			       -1,  /*  total_events */
			        PAPI_DOM_USER, /* default domain */
			        PAPI_GRN_THR,  /* default granularity */
			        0,  /* We can use add_prog_event */
			        0,  /* We can write the counters */
			        0,  /* supports HW overflow */
			        0,  /* supports HW profile */
			        1,  /* supports 64 bit virtual counters */
			        0,  /* supports child inheritance */
			        0,  /* supports attaching to another process */
			        1,  /* We can use the real_usec call */
			        1,  /* We can use the real_cyc call */
			        0,  /* We can use the virt_usec call */
			        0,  /* We can use the virt_cyc call */
			        0,  /* HW Read also resets the counters */
			        sizeof(hwd_control_state_t), 
			        NULL };

