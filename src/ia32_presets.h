/* 
* File:    ia32_presets.h
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    nils smeds
*          smeds@pdc.kth.se
*/  

/*
  PAPI Preset maps for Intel P6 and AMD K7 processors.
  Used by both linux-perfctr.c and win32.c substrates.
*/

static hwd_preset_t p6_preset_map[PAPI_MAX_PRESET_EVENTS] = { 
  {CNTR2|CNTR1,0,0,{{0x45,0x45,0x0,0x0}},""},	// L1 Cache Dmisses 
  {CNTR2|CNTR1,0,0,{{0xf28,0xf28,0x0,0x0}},""},	// L1 Cache Imisses 
  {0,0,0,{{0,0,0x0,0x0}},""}, 			// L2 Cache Dmisses
  {CNTR2|CNTR1,0,0,{{0x81,0x81,0x0,0x0}},""},	// L2 Cache Imisses 
  {0,0,0,{{0,0,0x0,0x0}},""}, 			// L3 Cache Dmisses
  {0,0,0,{{0,0,0x0,0x0}},""}, 			// L3 Cache Imisses
  {CNTR2|CNTR1,0,0,{{0xf2e,0xf2e,0x0,0x0}},""},	// L1 Total Cache misses 
  {CNTR2|CNTR1,0,0,{{0x24,0x24,0x0,0x0}},""}, 	// L2 Total Cache misses
  {0,0,0,{{0,0,0x0,0x0}},""}, 			// L3 Total Cache misses
  {0,0,0,{{0,0,0x0,0x0}},""},			// Snoops
  {CNTR2|CNTR1,0,0,{{0x222e,0x222e,0x0,0x0}},""},	// Req. access to shared cache line
  {CNTR2|CNTR1,0,0,{{0x212e,0x212e,0x0,0x0}},""},	// Req. access to clean cache line
  {CNTR2|CNTR1,0,0,{{0x2069,0x2069,0x0,0x0}},""},	// Req. Cache Line Invalidation
  {CNTR2|CNTR1,0,0,{{0x2e2e,0x2e2e,0x0,0x0}},""},	// Req. Cache Line Intervention
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3 LDM
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3 STM
  {0,0,0,{{0,0,0x0,0x0}},""},			// cycles branch idle
  {0,0,0,{{0,0,0x0,0x0}},""},			// cycles int idle
  {0,0,0,{{0,0,0x0,0x0}},""},			// cycles fpu idle
  {0,0,0,{{0,0,0x0,0x0}},""},			// cycles load/store idle
  {0,0,0,{{0,0,0x0,0x0}},""},		 	// D-TLB misses
  {CNTR2|CNTR1,0,0,{{0x85,0x85,0x0,0x0}},""},	// I-TLB misses
  {0,0,0,{{0,0,0x0,0x0}},""},			// Total TLB misses
  {CNTR2|CNTR1,0,0,{{0xf29,0xf29,0x0,0x0}},""},	// L1 load M
  {CNTR2|CNTR1,0,0,{{0xf2A,0xf2A,0x0,0x0}},""},	// L1 store M
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2 load M
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2 store M
  {CNTR2|CNTR1,0,0,{{0xe2,0xe2,0x0,0x0}},""},	// BTAC misses
  {0,0,0,{{0,0,0x0,0x0}},""},	                // Prefmiss
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3DCH
  {0,0,0,{{0,0,0x0,0x0}},""},			// TLB shootdowns
  {0,0,0,{{0,0,0x0,0x0}},""},			// Failed Store cond.
  {0,0,0,{{0,0,0x0,0x0}},""},			// Suc. store cond.
  {0,0,0,{{0,0,0x0,0x0}},""},			// total. store cond.
  {0,0,0,{{0,0,0x0,0x0}},""},	                // Cycles stalled waiting for memory /
  {0,0,0,{{0,0,0x0,0x0}},""},		   	// Cycles stalled waiting for memory read /
  {0,0,0,{{0,0,0x0,0x0}},""},		   	// Cycles stalled waiting for memory write /
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles no instructions issued /
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles max instructions issued /
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles no instructions comleted /
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles max instructions completed /
  {CNTR2|CNTR1,0,0,{{0xC8,0xC8,0x0,0x0}},""},	// hardware interrupts
  {0,0,0,{{0,0,0x0,0x0}},""},	                // Uncond. branches executed
  {CNTR2|CNTR1,0,0,{{0xC4,0xC4,0x0,0x0}},""},	// Cond. Branch inst. executed
  {CNTR2|CNTR1,0,0,{{0xC9,0xC9,0x0,0x0}},""},	// Cond. Branch inst. taken
  {CNTR2|CNTR1,DERIVED_SUB,0,{{0xC4,0xC9,0x0,0x0}},""}, // Cond. Branch inst. not taken
  {CNTR2|CNTR1,0,0,{{0xC5,0xC5,0x0,0x0}},""},	// Cond. branch inst. mispred.
  {CNTR2|CNTR1,DERIVED_SUB,0,{{0xC4,0xC5,0x0,0x0}},""}, // Cond. branch inst. corr. pred.
  {0,0,0,{{0,0,0x0,0x0}},""},			// FMA
  {CNTR2|CNTR1,0,0,{{0xD0,0xD0,0x0,0x0}},""},	// Total inst. issued
  {CNTR2|CNTR1,0,0,{{0xC0,0xC0,0x0,0x0}},""},	// Total inst. executed
  {0,0,0,{{0,0,0x0,0x0}},""},			// Integer inst. executed
  {CNTR1,0,0,{{0xC1,0,0x0,0x0}},""},		// Floating Pt. inst. executed
  {0,0,0,{{0,0,0x0,0x0}},""},			// Loads executed
  {0,0,0,{{0,0,0x0,0x0}},""},			// Stores executed
  {CNTR2|CNTR1,0,0,{{0xC4,0xC4,0x0,0x0}},""},	// Branch inst. executed
  {CNTR2|CNTR1,0,0,{{0xB0,0xB0,0x0,0x0}},""},	// Vector/SIMD inst. executed 
  {CNTR2|CNTR1,DERIVED_PS,1,{{0xC1,0x79,0x0,0x0}},""}, // FLOPS
  {CNTR2|CNTR1,0,0,{{0xA2,0xA2,0x0,0x0}},""},	// Cycles any resource stalls
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles FPU stalled
  {CNTR2|CNTR1,0,0,{{0x79,0x79,0x0,0x0}},""},	// Total cycles
  {CNTR2|CNTR1,DERIVED_PS,1,{{0xC0,0x79,0x0,0x0}},""}, // IPS
  {0,0,0,{{0,0,0,0}},""},			// Total load/store inst. exec
  {0,0,0,{{0,0,0x0,0x0}},""},			// SYnc exec.
  {CNTR2|CNTR1,DERIVED_SUB,0,{{0x43,0x45,0x0,0x0}},""}, // L1_DCH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_DCH
  {CNTR2|CNTR1,0,0,{{0x43,0x43,0x0,0x0}},""},	// L1_DCA
  {CNTR2|CNTR1,DERIVED_ADD,0,{{0xf29,0xf2a,0x0,0x0}},""}, // L2_DCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_DCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_DCR
  {CNTR2|CNTR1,0,0,{{0xf29,0xf29,0x0,0x0}},""},	// L2_DCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_DCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_DCW
  {CNTR2|CNTR1,0,0,{{0xf2a,0xf2a,0x0,0x0}},""},	// L2_DCW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_DCW
  {CNTR2|CNTR1,DERIVED_SUB,0,{{0x80,0xf28,0x0,0x0}},""}, // L1_ICH
  {CNTR2|CNTR1,DERIVED_SUB,0,{{0xf28,0x81,0x0,0x0}},""}, // L2_ICH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_ICH
  {CNTR2|CNTR1,0,0,{{0x80,0x80,0x0,0x0}},""},	// L1_ICA
  {CNTR2|CNTR1,0,0,{{0xf28,0xf28,0x0,0x0}},""},	// L2_ICA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_ICA
  {CNTR2|CNTR1,0,0,{{0x80,0x80,0x0,0x0}},""},	// L1_ICR
  {CNTR2|CNTR1,0,0,{{0xf28,0xf28,0x0,0x0}},""},	// L2_ICR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_ICR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_ICW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_ICW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_ICW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_TCH
  {CNTR2|CNTR1,DERIVED_SUB,0,{{0xf2e,0x24,0x0,0x0}},""}, // L2_TCH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_TCH
  {CNTR2|CNTR1,DERIVED_ADD,0,{{0x43,0x80,0x0,0x0}},""},	// L1_TCA
  {CNTR2|CNTR1,0,0,{{0xf2e,0xf2e,0x0,0x0}},""},	// L2_TCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_TCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_TCR
  {CNTR2|CNTR1,DERIVED_ADD,0,{{0xf29,0xf28,0x0,0x0}},""}, // L2_TCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_TCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_TCW
  {CNTR2|CNTR1,0,0,{{0xf2a,0xf2a,0x0,0x0}},""},	// L2_TCW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_TCW
  {CNTR2,0,0,{{0,0x12,0x0,0x0}},""},		// FPM
  {0,0,0,{{0,0,0x0,0x0}},""},			// FPA
  {CNTR2,0,0,{{0,0x13,0x0,0x0}},""},		// FPD
  {0,0,0,{{0,0,0x0,0x0}},""},			// FPSQ
  {0,0,0,{{0,0,0x0,0x0}},""},			// FPI
};

static hwd_preset_t k7_preset_map[PAPI_MAX_PRESET_EVENTS] = { 
  {ALLCNTRS,0,0,{{0x41,0x41,0x41,0x41}},""},	// L1 Cache Dmisses 
  {ALLCNTRS,0,0,{{0x81,0x81,0x81,0x81}},""},	// L1 Cache Imisses 
  {ALLCNTRS,0,0,{{0x42,0x42,0x42,0x42}},""}, 	// L2 Cache Dmisses
  {ALLCNTRS,0,0,{{0x83,0x83,0x83,0x83}},""}, 	// L2 Cache Imisses
  {0,0,0,{{0,0,0x0,0x0}},""}, 			// L3 Cache Dmisses
  {0,0,0,{{0,0,0x0,0x0}},""}, 			// L3 Cache Imisses
  {ALLCNTRS,0,0,{{0x73,0x73,0x73,0x73}},""}, 	// L1 Total Cache misses 
  {0,0,0,{{0,0,0x0,0x0}},""}, 			// L2 Total Cache misses
  {0,0,0,{{0,0,0x0,0x0}},""}, 			// L3 Total Cache misses
  {0,0,0,{{0,0,0x0,0x0}},""},			// Snoops
  {0,0,0,{{0,0,0x0,0x0}},""},			// Req. access to shared cache line
  {0,0,0,{{0,0,0x0,0x0}},""},			// Req. access to clean cache line
  {0,0,0,{{0,0,0x0,0x0}},""},			// Req. Cache Line Invalidation
  {0,0,0,{{0,0,0x0,0x0}},""},			// Req. Cache Line Intervention
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3 LDM
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3 STM
  {0,0,0,{{0,0,0x0,0x0}},""},			// cycles branch idle
  {0,0,0,{{0,0,0x0,0x0}},""},			// cycles int idle
  {0,0,0,{{0,0,0x0,0x0}},""},			// cycles fpu idle
  {0,0,0,{{0,0,0x0,0x0}},""},			// cycles load/store idle
  {ALLCNTRS,0,0,{{0x46,0x46,0x46,0x46}},""},	// D-TLB misses
  {CNTR1|CNTR2,DERIVED_ADD,0,{{0x84,0x85,0x0,0x0}},""}, // I-TLB misses
  {CNTR1|CNTR2|CNTR3,DERIVED_ADD,0,{{0x84,0x85,0x46,0x0}},""}, // Total TLB misses
  {0,0,0,{{0,0,0x0,0x0}},""},		   	// L1 load M
  {0,0,0,{{0,0,0x0,0x0}},""},		   	// L1 store M
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2 load M
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2 store M
  {0,0,0,{{0,0,0x0,0x0}},""},		   	// BTAC misses
  {0,0,0,{{0,0,0x0,0x0}},""},	                // Prefmiss
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3DCH
  {0,0,0,{{0,0,0x0,0x0}},""},			// TLB shootdowns
  {0,0,0,{{0,0,0x0,0x0}},""},			// Failed Store cond.
  {0,0,0,{{0,0,0x0,0x0}},""},			// Suc. store cond.
  {0,0,0,{{0,0,0x0,0x0}},""},			// total. store cond.
  {0,0,0,{{0,0,0x0,0x0}},""},	                // Cycles stalled waiting for memory /
  {0,0,0,{{0,0,0x0,0x0}},""},		   	// Cycles stalled waiting for memory read /
  {0,0,0,{{0,0,0x0,0x0}},""},		   	// Cycles stalled waiting for memory write /
  {0,0,0,{{0,0,0x0,0x0}},""},		   	// Cycles no instructions issued /
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles max instructions issued /
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles no instructions completed /
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles max instructions completed /
  {ALLCNTRS,0,0,{{0xcf,0xcf,0xcf,0xcf}},""},	// hardware interrupts
  {ALLCNTRS,0,0,{{0xc6,0xc6,0xc6,0xc6}},""},	// Uncond. branches executed
  {ALLCNTRS,0,0,{{0xC2,0xC2,0xc2,0xc2}},""},	// Cond. Branch inst. executed
  {ALLCNTRS,0,0,{{0xC4,0xC4,0xc4,0xc4}},""},	// Cond. Branch inst. taken
  {CNTR1|CNTR2,DERIVED_SUB,0,{{0xC4,0xC2,0x0,0x0}},""}, // Cond. Branch inst. not taken
  {ALLCNTRS,0,0,{{0xC3,0xC3,0xC3,0xC3}},""},	// Cond. branch inst. mispred.
  {CNTR1|CNTR2,DERIVED_SUB,0,{{0xC2,0xC3,0x0,0x0}},""}, // Cond. branch inst. corr. pred.
  {0,0,0,{{0,0,0x0,0x0}},""},			// FMA
  {0,0,0,{{0,0,0x0,0x0}},""},                   // Total inst. issued
  {ALLCNTRS,0,0,{{0xC0,0xC0,0xC0,0xC0}},""},	// Total inst. executed
  {0,0,0,{{0,0,0x0,0x0}},""},			// Integer inst. executed
  {0,0,0,{{0,0,0x0,0x0}},""},                   // Floating Pt. inst. executed
  {0,0,0,{{0,0,0x0,0x0}},""},			// Loads executed
  {0,0,0,{{0,0,0x0,0x0}},""},			// Stores executed
  {ALLCNTRS,0,0,{{0xC4,0xC4,0x0,0x0}},""},	// Branch inst. executed
  {ALLCNTRS,0,0,{{0xB0,0xB0,0x0,0x0}},""},	// Vector/SIMD inst. executed 
  {0,0,0,{{0,0,0x0,0x0}},""},                   // FLOPS
  {ALLCNTRS,0,0,{{0xd9,0xd9,0xd9,0xd9}},""},    // Cycles any resource stalls
  {0,0,0,{{0,0,0x0,0x0}},""},			// Cycles FPU stalled
  {ALLCNTRS,0,0,{{0x76,0x76,0x76,0x76}},""},	// Total cycles
  {CNTR1|CNTR2,DERIVED_PS,1,{{0xC0,0x76,0x0,0x0}},""}, // IPS
  {0,0,0,{{0,0,0x0,0x0}},""},			// Total load/store inst. exec
  {0,0,0,{{0,0,0x0,0x0}},""},			// SYnc exec.
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_DCH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_DCH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_DCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_DCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_DCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_DCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_DCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_DCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_DCW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_DCW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_DCW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_ICH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_ICH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_ICH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_ICA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_ICA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_ICA
  {ALLCNTRS,0,0,{{0x80,0x80,0x80,0x80}},""},	// L1_ICR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_ICR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_ICR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_ICW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_ICW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_ICW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_TCH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_TCH
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_TCH
  {CNTR1|CNTR2,DERIVED_ADD,0,{{0x40,0x80,0x0,0x0}},""}, // L1_TCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_TCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_TCA
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_TCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_TCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_TCR
  {0,0,0,{{0,0,0x0,0x0}},""},			// L1_TCW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L2_TCW
  {0,0,0,{{0,0,0x0,0x0}},""},			// L3_TCW
  {0,0,0,{{0,0,0x0,0x0}},""},			// FPM
  {0,0,0,{{0,0,0x0,0x0}},""},			// FPA
  {0,0,0,{{0,0,0x0,0x0}},""},			// FPD
  {0,0,0,{{0,0,0x0,0x0}},""},			// FPSQ
  {0,0,0,{{0,0,0x0,0x0}},""},			// FPI
};

