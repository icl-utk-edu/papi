/* $Id$
 * x86-specific perfctr library procedures.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */
#include <stdio.h>
#include "libperfctr.h"

unsigned int perfctr_info_nrctrs(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
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
	return 4;
      case PERFCTR_X86_INTEL_CORE:
	return 2;
      case PERFCTR_X86_GENERIC:
      default:
	return 0;
    }
}

const char *perfctr_info_cpu_name(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
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
#endif
      case PERFCTR_X86_INTEL_CORE:
	return "Intel Core";
      case PERFCTR_X86_INTEL_P4M3:
	return "Intel Pentium 4 Model 3";
      case PERFCTR_X86_AMD_K8:
	return "AMD K8";
      case PERFCTR_X86_AMD_K8C:
	return "AMD K8 Revision C";
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
    if( nrictrs )
	printf("nrictrs\t\t\t%u\n", nrictrs);
    for(i = 0; i < nrctrs; ++i) {
        if( control->pmc_map[i] >= 18 ) /* for P4 'fast rdpmc' cases */
            printf("pmc_map[%u]\t\t0x%08X\n", i, control->pmc_map[i]);
        else
            printf("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
        printf("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
        if( control->p4.escr[i] )
            printf("escr[%u]\t\t\t0x%08X\n", i, control->p4.escr[i]);
	if( i >= nractrs )
	    printf("ireset[%u]\t\t%d\n", i, control->ireset[i]);
    }
    if( control->p4.pebs_enable )
	printf("pebs_enable\t\t0x%08X\n", control->p4.pebs_enable);
    if( control->p4.pebs_matrix_vert )
	printf("pebs_matrix_vert\t0x%08X\n", control->p4.pebs_matrix_vert);
}
