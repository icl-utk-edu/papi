/* 
 * x86 performance-monitoring counters driver support routines.
 *
 * Modified for Windows NT/2000 by dkterpstra 05/2001
 */
#include <ntddk.h>
#include "pmc_kernel.h"
#include "pmc_x86.h"
#include "msr-index.h"


void __inline _rmsr(uint32_t *v, uint32_t reg)
{
  __asm
  {
    mov ecx, reg
    rdmsr
    mov ebx, v
    mov [ebx], eax
    mov [ebx + 4], edx
  }
}


uint (*translate_pmc)(uint);
uint (*translate_pmd)(uint);

uint p5_translate_pmc(uint reg)
{
  if (reg == 0)
    return MSR_P5_CESR;
  return 0;
}

uint p5_translate_pmd(uint reg)
{
  if (reg <= 1)
    return MSR_P5_CTR0 + reg;
  return 0;
}

uint p6_translate_pmc(uint reg)
{
  switch (reg)
  {
    case 0:
    case 1:
      return MSR_P6_EVNTSEL0 + reg;
    case 16:
    case 17:
      return MSR_CORE_PERF_FIXED_CTR_CTRL - 16 + reg;
  }
  return 0;
}

uint p6_translate_pmd(uint reg)
{
  switch (reg)
  {
    case 0:
    case 1:
      return MSR_P6_PERFCTR0 + reg;
    case 16:
    case 17:
    case 18:
      return MSR_CORE_PERF_FIXED_CTR0 - 16 + reg;
  }
  return 0;
}


uint k7_translate_pmc(uint reg)
{
  switch (reg)
  {
    case 0:
    case 1:
    case 2:
    case 3:
      return MSR_K7_EVNTSEL0 + reg;
  }
  return 0;
}

uint k7_translate_pmd(uint reg)
{
  switch (reg)
  {
  case 0:
  case 1:
  case 2:
  case 3:
    return MSR_K7_PERFCTR0 + reg;
  }
  return 0;
}

/*
 * bitmask for pfm_p4_regmap.reg_type
 */
#define PFM_REGT_NA		0x0000	/* not available */
#define PFM_REGT_EN		0x0001	/* has enable bit (cleared on ctxsw) */
#define PFM_REGT_ESCR		0x0002	/* P4: ESCR */
#define PFM_REGT_CCCR		0x0004	/* P4: CCCR */
#define PFM_REGT_PEBS		0x0010	/* PEBS related */
#define PFM_REGT_NOHT		0x0020	/* unavailable with HT */
#define PFM_REGT_CTR		0x0040	/* counter */

struct pfm_p4_regmap {
	/*
	 * one each for the logical CPUs.  Index 0 corresponds to T0 and
	 * index 1 corresponds to T1.  Index 1 can be zero if no T1
	 * complement reg exists.
	 */
	unsigned long addrs[2]; /* 2 = number of threads */
	unsigned int ctr;	/* for CCCR/PERFEVTSEL, associated counter */
	unsigned int reg_type;
};


/*
 * With HyperThreading enabled:
 *
 *  The ESCRs and CCCRs are divided in half with the top half
 *  belonging to logical processor 0 and the bottom half going to
 *  logical processor 1. Thus only half of the PMU resources are
 *  accessible to applications.
 *
 *  PEBS is not available due to the fact that:
 *  	- MSR_PEBS_MATRIX_VERT is shared between the threads
 *      - IA32_PEBS_ENABLE is shared between the threads
 *
 * With HyperThreading disabled:
 *
 * The full set of PMU resources is exposed to applications.
 *
 * The mapping is chosen such that PMCxx -> MSR is the same
 * in HT and non HT mode, if register is present in HT mode.
 *
 */
#define PFM_REGT_NHTESCR (PFM_REGT_ESCR|PFM_REGT_NOHT)
#define PFM_REGT_NHTCCCR (PFM_REGT_CCCR|PFM_REGT_NOHT|PFM_REGT_EN)
#define PFM_REGT_NHTPEBS (PFM_REGT_PEBS|PFM_REGT_NOHT|PFM_REGT_EN)
#define PFM_REGT_NHTCTR  (PFM_REGT_CTR|PFM_REGT_NOHT)
#define PFM_REGT_ENAC    (PFM_REGT_CCCR|PFM_REGT_EN)

static struct pfm_p4_regmap p4_pmc_addrs[PFM_MAX_PMCS] = 
{
  /*pmc 0 */    {{MSR_P4_BPU_ESCR0, MSR_P4_BPU_ESCR1}, 0, PFM_REGT_ESCR}, /*   BPU_ESCR0,1 */
	/*pmc 1 */    {{MSR_P4_IS_ESCR0, MSR_P4_IS_ESCR1}, 0, PFM_REGT_ESCR}, /*    IS_ESCR0,1 */
	/*pmc 2 */    {{MSR_P4_MOB_ESCR0, MSR_P4_MOB_ESCR1}, 0, PFM_REGT_ESCR}, /*   MOB_ESCR0,1 */
	/*pmc 3 */    {{MSR_P4_ITLB_ESCR0, MSR_P4_ITLB_ESCR1}, 0, PFM_REGT_ESCR}, /*  ITLB_ESCR0,1 */
	/*pmc 4 */    {{MSR_P4_PMH_ESCR0, MSR_P4_PMH_ESCR1}, 0, PFM_REGT_ESCR}, /*   PMH_ESCR0,1 */
	/*pmc 5 */    {{MSR_P4_IX_ESCR0, MSR_P4_IX_ESCR1}, 0, PFM_REGT_ESCR}, /*    IX_ESCR0,1 */
	/*pmc 6 */    {{MSR_P4_FSB_ESCR0, MSR_P4_FSB_ESCR1}, 0, PFM_REGT_ESCR}, /*   FSB_ESCR0,1 */
	/*pmc 7 */    {{MSR_P4_BSU_ESCR0, MSR_P4_BSU_ESCR1}, 0, PFM_REGT_ESCR}, /*   BSU_ESCR0,1 */
	/*pmc 8 */    {{MSR_P4_MS_ESCR0, MSR_P4_MS_ESCR1}, 0, PFM_REGT_ESCR}, /*    MS_ESCR0,1 */
	/*pmc 9 */    {{MSR_P4_TC_ESCR0, MSR_P4_TC_ESCR1}, 0, PFM_REGT_ESCR}, /*    TC_ESCR0,1 */
	/*pmc 10*/    {{MSR_P4_TBPU_ESCR0, MSR_P4_TBPU_ESCR1}, 0, PFM_REGT_ESCR}, /*  TBPU_ESCR0,1 */
	/*pmc 11*/    {{MSR_P4_FLAME_ESCR0, MSR_P4_FLAME_ESCR1}, 0, PFM_REGT_ESCR}, /* FLAME_ESCR0,1 */
	/*pmc 12*/    {{MSR_P4_FIRM_ESCR0, MSR_P4_FIRM_ESCR1}, 0, PFM_REGT_ESCR}, /*  FIRM_ESCR0,1 */
	/*pmc 13*/    {{MSR_P4_SAAT_ESCR0, MSR_P4_SAAT_ESCR1}, 0, PFM_REGT_ESCR}, /*  SAAT_ESCR0,1 */
	/*pmc 14*/    {{MSR_P4_U2L_ESCR0, MSR_P4_U2L_ESCR1}, 0, PFM_REGT_ESCR}, /*   U2L_ESCR0,1 */
	/*pmc 15*/    {{MSR_P4_DAC_ESCR0, MSR_P4_DAC_ESCR1}, 0, PFM_REGT_ESCR}, /*   DAC_ESCR0,1 */
	/*pmc 16*/    {{MSR_P4_IQ_ESCR0, MSR_P4_IQ_ESCR1}, 0, PFM_REGT_ESCR}, /*    IQ_ESCR0,1 (only model 1 and 2) */
	/*pmc 17*/    {{MSR_P4_ALF_ESCR0, MSR_P4_ALF_ESCR1}, 0, PFM_REGT_ESCR}, /*   ALF_ESCR0,1 */
	/*pmc 18*/    {{MSR_P4_RAT_ESCR0, MSR_P4_RAT_ESCR1}, 0, PFM_REGT_ESCR}, /*   RAT_ESCR0,1 */
	/*pmc 19*/    {{MSR_P4_SSU_ESCR0, 0}, 0, PFM_REGT_ESCR}, /*   SSU_ESCR0   */
	/*pmc 20*/    {{MSR_P4_CRU_ESCR0, MSR_P4_CRU_ESCR1}, 0, PFM_REGT_ESCR}, /*   CRU_ESCR0,1 */
	/*pmc 21*/    {{MSR_P4_CRU_ESCR2, MSR_P4_CRU_ESCR3}, 0, PFM_REGT_ESCR}, /*   CRU_ESCR2,3 */
	/*pmc 22*/    {{MSR_P4_CRU_ESCR4, MSR_P4_CRU_ESCR5}, 0, PFM_REGT_ESCR}, /*   CRU_ESCR4,5 */

	/*pmc 23*/    {{MSR_P4_BPU_CCCR0, MSR_P4_BPU_CCCR2}, 0, PFM_REGT_ENAC}, /*   BPU_CCCR0,2 */
	/*pmc 24*/    {{MSR_P4_BPU_CCCR1, MSR_P4_BPU_CCCR3}, 1, PFM_REGT_ENAC}, /*   BPU_CCCR1,3 */
	/*pmc 25*/    {{MSR_P4_MS_CCCR0, MSR_P4_MS_CCCR2}, 2, PFM_REGT_ENAC}, /*    MS_CCCR0,2 */
	/*pmc 26*/    {{MSR_P4_MS_CCCR1, MSR_P4_MS_CCCR3}, 3, PFM_REGT_ENAC}, /*    MS_CCCR1,3 */
	/*pmc 27*/    {{MSR_P4_FLAME_CCCR0, MSR_P4_FLAME_CCCR2}, 4, PFM_REGT_ENAC}, /* FLAME_CCCR0,2 */
	/*pmc 28*/    {{MSR_P4_FLAME_CCCR1, MSR_P4_FLAME_CCCR3}, 5, PFM_REGT_ENAC}, /* FLAME_CCCR1,3 */
	/*pmc 29*/    {{MSR_P4_IQ_CCCR0, MSR_P4_IQ_CCCR2}, 6, PFM_REGT_ENAC}, /*    IQ_CCCR0,2 */
	/*pmc 30*/    {{MSR_P4_IQ_CCCR1, MSR_P4_IQ_CCCR3}, 7, PFM_REGT_ENAC}, /*    IQ_CCCR1,3 */
	/*pmc 31*/    {{MSR_P4_IQ_CCCR4, MSR_P4_IQ_CCCR5}, 8, PFM_REGT_ENAC}, /*    IQ_CCCR4,5 */
	/* non HT extensions */
	/*pmc 32*/    {{MSR_P4_BPU_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   BPU_ESCR1   */
	/*pmc 33*/    {{MSR_P4_IS_ESCR1,     0},  0, PFM_REGT_NHTESCR}, /*    IS_ESCR1   */
	/*pmc 34*/    {{MSR_P4_MOB_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   MOB_ESCR1   */
	/*pmc 35*/    {{MSR_P4_ITLB_ESCR1,   0},  0, PFM_REGT_NHTESCR}, /*  ITLB_ESCR1   */
	/*pmc 36*/    {{MSR_P4_PMH_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   PMH_ESCR1   */
	/*pmc 37*/    {{MSR_P4_IX_ESCR1,     0},  0, PFM_REGT_NHTESCR}, /*    IX_ESCR1   */
	/*pmc 38*/    {{MSR_P4_FSB_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   FSB_ESCR1   */
	/*pmc 39*/    {{MSR_P4_BSU_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   BSU_ESCR1   */
	/*pmc 40*/    {{MSR_P4_MS_ESCR1,     0},  0, PFM_REGT_NHTESCR}, /*    MS_ESCR1   */
	/*pmc 41*/    {{MSR_P4_TC_ESCR1,     0},  0, PFM_REGT_NHTESCR}, /*    TC_ESCR1   */
	/*pmc 42*/    {{MSR_P4_TBPU_ESCR1,   0},  0, PFM_REGT_NHTESCR}, /*  TBPU_ESCR1   */
	/*pmc 43*/    {{MSR_P4_FLAME_ESCR1,  0},  0, PFM_REGT_NHTESCR}, /* FLAME_ESCR1   */
	/*pmc 44*/    {{MSR_P4_FIRM_ESCR1,   0},  0, PFM_REGT_NHTESCR}, /*  FIRM_ESCR1   */
	/*pmc 45*/    {{MSR_P4_SAAT_ESCR1,   0},  0, PFM_REGT_NHTESCR}, /*  SAAT_ESCR1   */
	/*pmc 46*/    {{MSR_P4_U2L_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   U2L_ESCR1   */
	/*pmc 47*/    {{MSR_P4_DAC_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   DAC_ESCR1   */
	/*pmc 48*/    {{MSR_P4_IQ_ESCR1,     0},  0, PFM_REGT_NHTESCR}, /*    IQ_ESCR1   (only model 1 and 2) */
	/*pmc 49*/    {{MSR_P4_ALF_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   ALF_ESCR1   */
	/*pmc 50*/    {{MSR_P4_RAT_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   RAT_ESCR1   */
	/*pmc 51*/    {{MSR_P4_CRU_ESCR1,    0},  0, PFM_REGT_NHTESCR}, /*   CRU_ESCR1   */
	/*pmc 52*/    {{MSR_P4_CRU_ESCR3,    0},  0, PFM_REGT_NHTESCR}, /*   CRU_ESCR3   */
	/*pmc 53*/    {{MSR_P4_CRU_ESCR5,    0},  0, PFM_REGT_NHTESCR}, /*   CRU_ESCR5   */
	/*pmc 54*/    {{MSR_P4_BPU_CCCR1,    0},  9, PFM_REGT_NHTCCCR}, /*   BPU_CCCR1   */
	/*pmc 55*/    {{MSR_P4_BPU_CCCR3,    0}, 10, PFM_REGT_NHTCCCR}, /*   BPU_CCCR3   */
	/*pmc 56*/    {{MSR_P4_MS_CCCR1,     0}, 11, PFM_REGT_NHTCCCR}, /*    MS_CCCR1   */
	/*pmc 57*/    {{MSR_P4_MS_CCCR3,     0}, 12, PFM_REGT_NHTCCCR}, /*    MS_CCCR3   */
	/*pmc 58*/    {{MSR_P4_FLAME_CCCR1,  0}, 13, PFM_REGT_NHTCCCR}, /* FLAME_CCCR1   */
	/*pmc 59*/    {{MSR_P4_FLAME_CCCR3,  0}, 14, PFM_REGT_NHTCCCR}, /* FLAME_CCCR3   */
	/*pmc 60*/    {{MSR_P4_IQ_CCCR2,     0}, 15, PFM_REGT_NHTCCCR}, /*    IQ_CCCR2   */
	/*pmc 61*/    {{MSR_P4_IQ_CCCR3,     0}, 16, PFM_REGT_NHTCCCR}, /*    IQ_CCCR3   */
	/*pmc 62*/    {{MSR_P4_IQ_CCCR5,     0}, 17, PFM_REGT_NHTCCCR}, /*    IQ_CCCR5   */
	/*pmc 63*/    {{0x3f2,     0}, 0, PFM_REGT_NHTPEBS},/* PEBS_MATRIX_VERT */
	/*pmc 64*/    {{0x3f1,     0}, 0, PFM_REGT_NHTPEBS} /* PEBS_ENABLE   */
  };

  static struct pfm_p4_regmap p4_pmd_addrs[PFM_MAX_PMDS] = {
	/*pmd 0 */    {{MSR_P4_BPU_PERFCTR0, MSR_P4_BPU_PERFCTR2}, 0, PFM_REGT_CTR},  /*   BPU_CTR0,2  */
	/*pmd 1 */    {{MSR_P4_BPU_PERFCTR1, MSR_P4_BPU_PERFCTR3}, 0, PFM_REGT_CTR},  /*   BPU_CTR1,3  */
	/*pmd 2 */    {{MSR_P4_MS_PERFCTR0, MSR_P4_MS_PERFCTR2}, 0, PFM_REGT_CTR},  /*    MS_CTR0,2  */
	/*pmd 3 */    {{MSR_P4_MS_PERFCTR1, MSR_P4_MS_PERFCTR3}, 0, PFM_REGT_CTR},  /*    MS_CTR1,3  */
	/*pmd 4 */    {{MSR_P4_FLAME_PERFCTR0, MSR_P4_FLAME_PERFCTR2}, 0, PFM_REGT_CTR},  /* FLAME_CTR0,2  */
	/*pmd 5 */    {{MSR_P4_FLAME_PERFCTR1, MSR_P4_FLAME_PERFCTR3}, 0, PFM_REGT_CTR},  /* FLAME_CTR1,3  */
	/*pmd 6 */    {{MSR_P4_IQ_PERFCTR0, MSR_P4_IQ_PERFCTR2}, 0, PFM_REGT_CTR},  /*    IQ_CTR0,2  */
	/*pmd 7 */    {{MSR_P4_IQ_PERFCTR1, MSR_P4_IQ_PERFCTR3}, 0, PFM_REGT_CTR},  /*    IQ_CTR1,3  */
	/*pmd 8 */    {{MSR_P4_IQ_PERFCTR4, MSR_P4_IQ_PERFCTR5}, 0, PFM_REGT_CTR},  /*    IQ_CTR4,5  */
	/*
	 * non HT extensions
	 */
	/*pmd 9 */    {{MSR_P4_BPU_PERFCTR2,     0}, 0, PFM_REGT_NHTCTR},  /*   BPU_CTR2    */
	/*pmd 10*/    {{MSR_P4_BPU_PERFCTR3,     0}, 0, PFM_REGT_NHTCTR},  /*   BPU_CTR3    */
	/*pmd 11*/    {{MSR_P4_MS_PERFCTR2,     0}, 0, PFM_REGT_NHTCTR},  /*    MS_CTR2    */
	/*pmd 12*/    {{MSR_P4_MS_PERFCTR3,     0}, 0, PFM_REGT_NHTCTR},  /*    MS_CTR3    */
	/*pmd 13*/    {{MSR_P4_FLAME_PERFCTR2,     0}, 0, PFM_REGT_NHTCTR},  /* FLAME_CTR2    */
	/*pmd 14*/    {{MSR_P4_FLAME_PERFCTR3,     0}, 0, PFM_REGT_NHTCTR},  /* FLAME_CTR3    */
	/*pmd 15*/    {{MSR_P4_IQ_PERFCTR2,     0}, 0, PFM_REGT_NHTCTR},  /*    IQ_CTR2    */
	/*pmd 16*/    {{MSR_P4_IQ_PERFCTR3,     0}, 0, PFM_REGT_NHTCTR},  /*    IQ_CTR3    */
	/*pmd 17*/    {{MSR_P4_IQ_PERFCTR5,     0}, 0, PFM_REGT_NHTCTR},  /*    IQ_CTR5    */
};

uint p4_translate_pmc(uint reg)
{
  if (reg >= sizeof(p4_pmc_addrs) / sizeof(struct pfm_p4_regmap))
    return 0;
  return p4_pmc_addrs[reg].addrs[0];
}

uint p4_translate_pmd(uint reg)
{
  if (reg >= sizeof(p4_pmd_addrs) / sizeof(struct pfm_p4_regmap))
    return 0;
  return p4_pmd_addrs[reg].addrs[0];
}


void write_control(pfarg_pmc_t *r, uint32_t count)
{
	uint i;
  for (i = 0; i < count; ++i)
  {
    uint32_t *v = (uint32_t *)&r[i].reg_value, reg;
    if (reg = translate_pmc(r[i].reg_num))
      _wrmsr(reg, v[0], v[1]);
  }
}

void read_write_data(pfarg_pmd_t *r, uint count, uint write)
{
  uint i;
  for (i = 0; i < count; ++i)
  {
    uint32_t *v = (uint32_t *)&r[i].reg_value, reg;
    if (reg = translate_pmd(r[i].reg_num))
    {
      if (write)
        _wrmsr(reg, v[0], v[1]);
      else
        _rmsr(v, reg);
    }
  }
}

/****************************************************************
 *																*
 * Processor detection and initialization procedures.			*
 *																*
 ****************************************************************/

static int intel_init(int family, int stepping, int model)
{
	switch (family) {
	case 5:
		translate_pmc = p5_translate_pmc;
    translate_pmd = p5_translate_pmd;
		return STATUS_SUCCESS;
	case 6:
	  /* P-Pro with SMM support will enter halt state if
     * the PCE bit is set and a RDPMC is issued    -KSL
		 */
	   if (model == 1 && stepping == 9) 
		    return STATUS_NO_INTEL_INIT;

      translate_pmc = p6_translate_pmc;
      translate_pmd = p6_translate_pmd;
  		return STATUS_SUCCESS;
  case 15:
      translate_pmc = p4_translate_pmc;
      translate_pmd = p4_translate_pmd;
      return STATUS_SUCCESS;
	}
	return STATUS_NO_INTEL_INIT;
}

static int amd_init(int family)
{
	switch(family) {
	case 6:	   /* K7 Athlon. Model 1 does not have a local APIC. */
	case 15:	/* K8 Opteron. Uses same control routines as Athlon. */
		translate_pmc = k7_translate_pmc;
    translate_pmd = k7_translate_pmd;
		return STATUS_SUCCESS;
	}
	return STATUS_NO_AMD_INIT;
}

/****************************************************************
 *																*
 * cpuid vendor string and features procedure.					*
 *																*
 ****************************************************************/

// Returns zero if no cpuid instruction; else returns 1
// Returns completed pmc_info struct on success 
static int cpu_id(struct CPUInfo *info) {
   uint32_t regs[4];
   char *s;

   // NOTE: Earlier versions of this routine checked for the existence of
   // the cpuid instruction before proceeding.
   // It's 2006. I think we can assume that any processor this is running on
   // is post-486 and will thus have the cpuid instruction.

	// Get the Vendor String, features, and family/model/stepping info
   __cpuid(regs, 0);
   s = info->vendor;
   ((uint32_t *)s)[0] = regs[1];
   ((uint32_t *)s)[1] = regs[3];
   ((uint32_t *)s)[2] = regs[2];
   __cpuid(regs, 1);

	info->features = regs[3];
	info->family = (regs[0] >> 8) & 0xF;		// bits 11 - 8
	info->model  = (regs[0] >> 4) & 0xF;    // Bits  7 - 4
	info->stepping=(regs[0])      & 0xF;    // bits  3 - 0
	return 1;
}


/****************************************************************
 *																*
 * system visible routines.										*
 *																*
 ****************************************************************/
#define MSR 0x00000020	// bit 5
#define TSC 0x00000010	// bit 4
#define MMX 0x00800000	// bit 23

// returns 0 for success or negative for failure
int kern_pmc_init()
{
	int status = STATUS_UNKNOWN_CPU_INIT;
	struct CPUInfo info;
   ULONG64 cr4;

	if (!cpu_id(&info)) return STATUS_NO_CPUID;

	// we need to at least support MSR registers, RDTSC, & RDPMC
	if(!(info.features & MSR)) return STATUS_NO_MSR;
	if(!(info.features & TSC)) return STATUS_NO_TSC;
	if(!(info.features & MMX)) return STATUS_NO_MMX;	// assume MMX tracks RDPMC

	if (!strncmp(info.vendor, "GenuineIntel", 12)) status = intel_init(info.family,info.stepping,info.model);
	else if (!strncmp(info.vendor, "AuthenticAMD", 12)) status = amd_init(info.family);
  // we really don't need to claim support for Cyrix, do we?
  // else if (!strncmp(info.vendor, "CyrixInstead", 12)) status = cyrix_init(info.family);
	if (status == STATUS_SUCCESS) set_cr4_pce();

	return status;
}

void kern_pmc_exit()
{
	clear_cr4_pce();
}
