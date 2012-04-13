/* $Id: x86.c,v 1.2.2.11 2010/11/07 19:46:06 mikpe Exp $
 * x86-specific perfctr library procedures.
 *
 * Copyright (C) 1999-2010  Mikael Pettersson
 */
#include <stdio.h>
#include "libperfctr.h"

struct cpuid {	/* The field order must not be changed. */
    unsigned int eax;
    unsigned int ebx;	/* When eax was 1, &ebx should be the start */
    unsigned int edx;	/* of the 12-byte vendor identification string. */
    unsigned int ecx;
};

static void get_cpuid(unsigned int op, struct cpuid *cpuid)
{
    unsigned int save_ebx;
    unsigned int tmp_ebx;

    __asm__(
	"movl %%ebx, %0\n\t"
	"cpuid\n\t"
	"movl %%ebx, %1\n\t"
	"movl %0, %%ebx"
	: "=m"(save_ebx), "=m"(tmp_ebx), "=a"(cpuid->eax), "=d"(cpuid->edx), "=c"(cpuid->ecx)
	: "a"(op));
    cpuid->ebx = tmp_ebx;
}

static unsigned int atom_nrctrs(void)
{
    struct cpuid cpuid;

    get_cpuid(0, &cpuid);
    if (cpuid.eax < 0xA) {
	printf("%s: cpuid[0].eax == %u, unable to query 0xA leaf\n",
	       __FUNCTION__, cpuid.eax);
	return 0;
    }
    get_cpuid(0xA, &cpuid);
    if ((cpuid.eax & 0xff) < 2) {
	printf("%s: cpuid[0xA].eax == 0x%08x appears bogus\n",
	       __FUNCTION__, cpuid.eax);
	return 0;
    }
    return ((cpuid.eax >> 8) & 0xff) + (cpuid.edx & 0x1f);
}

unsigned int perfctr_info_nrctrs(const struct perfctr_info *info)
{
    switch (info->cpu_type) {
#if !defined(__x86_64__)
      case PERFCTR_X86_INTEL_P5:
      case PERFCTR_X86_INTEL_P5MMX:
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
      case PERFCTR_X86_CYRIX_MII:
      case PERFCTR_X86_WINCHIP_C6:
      case PERFCTR_X86_WINCHIP_2:
      case PERFCTR_X86_INTEL_PENTM:
      case PERFCTR_X86_INTEL_CORE:
	return 2;
      case PERFCTR_X86_AMD_K7:
	return 4;
      case PERFCTR_X86_VIA_C3:
	return 1;
      case PERFCTR_X86_INTEL_P4:
      case PERFCTR_X86_INTEL_P4M2:
	return 18;
#endif
      case PERFCTR_X86_INTEL_P4M3:
	return 18;
      case PERFCTR_X86_AMD_K8:
      case PERFCTR_X86_AMD_K8C:
      case PERFCTR_X86_AMD_FAM10H:
	return 4;
      case PERFCTR_X86_INTEL_CORE2:
	return 5;
      case PERFCTR_X86_INTEL_ATOM:
	return atom_nrctrs();
      case PERFCTR_X86_INTEL_NHLM:
      case PERFCTR_X86_INTEL_WSTMR:
	return 7;
      case PERFCTR_X86_GENERIC:
      default:
	return 0;
    }
}

const char *perfctr_info_cpu_name(const struct perfctr_info *info)
{
    switch (info->cpu_type) {
      case PERFCTR_X86_GENERIC:
	return "Generic x86 with TSC";
#if !defined(__x86_64__)
      case PERFCTR_X86_INTEL_P5:
        return "Intel Pentium";
      case PERFCTR_X86_INTEL_P5MMX:
        return "Intel Pentium MMX";
      case PERFCTR_X86_INTEL_P6:
        return "Intel Pentium Pro";
      case PERFCTR_X86_INTEL_PII:
        return "Intel Pentium II";
      case PERFCTR_X86_INTEL_PIII:
        return "Intel Pentium III";
      case PERFCTR_X86_CYRIX_MII:
        return "Cyrix 6x86MX/MII/III";
      case PERFCTR_X86_WINCHIP_C6:
	return "WinChip C6";
      case PERFCTR_X86_WINCHIP_2:
	return "WinChip 2/3";
      case PERFCTR_X86_AMD_K7:
	return "AMD K7";
      case PERFCTR_X86_VIA_C3:
	return "VIA C3";
      case PERFCTR_X86_INTEL_P4:
	return "Intel Pentium 4";
      case PERFCTR_X86_INTEL_P4M2:
	return "Intel Pentium 4 Model 2";
      case PERFCTR_X86_INTEL_PENTM:
	return "Intel Pentium M";
      case PERFCTR_X86_INTEL_CORE:
	return "Intel Core";
#endif
      case PERFCTR_X86_INTEL_CORE2:
	return "Intel Core 2";
      case PERFCTR_X86_INTEL_P4M3:
	return "Intel Pentium 4 Model 3";
      case PERFCTR_X86_AMD_K8:
	return "AMD K8";
      case PERFCTR_X86_AMD_K8C:
	return "AMD K8 Revision C";
      case PERFCTR_X86_AMD_FAM10H:
	return "AMD Family 10h";
      case PERFCTR_X86_INTEL_ATOM:
	return "Intel Atom";
      case PERFCTR_X86_INTEL_NHLM:
	return "Intel Nehalem";
      case PERFCTR_X86_INTEL_WSTMR:
	return "Intel Westmere";
      default:
        return "?";
    }
}

void perfctr_cpu_control_print(const struct perfctr_cpu_control *control)
{
    unsigned int i, nractrs, nrictrs, nrctrs;

    nractrs = control->nractrs;
    nrictrs = control->nrictrs;
    nrctrs = control->nractrs + nrictrs;

    printf("tsc_on\t\t\t%u\n", control->tsc_on);
    printf("nractrs\t\t\t%u\n", nractrs);
    if (nrictrs)
	printf("nrictrs\t\t\t%u\n", nrictrs);
    for(i = 0; i < nrctrs; ++i) {
        if (control->pmc_map[i] >= 18) /* for Core2 fixed counters or P4 fast rdpmc */
            printf("pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i]);
        else
            printf("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
        printf("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
        if (control->p4.escr[i])
            printf("escr[%u]\t\t\t0x%08X\n", i, control->p4.escr[i]);
	if (i >= nractrs)
	    printf("ireset[%u]\t\t%d\n", i, control->ireset[i]);
    }
    if (control->p4.pebs_enable)
	printf("pebs_enable\t\t0x%08X\n", control->p4.pebs_enable);
    if (control->p4.pebs_matrix_vert)
	printf("pebs_matrix_vert\t0x%08X\n", control->p4.pebs_matrix_vert);
}
