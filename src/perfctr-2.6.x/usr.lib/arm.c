/* $Id: arm.c,v 1.1.2.1 2007/02/11 20:15:03 mikpe Exp $
 * ARM-specific perfctr library procedures.
 *
 * Copyright (C) 2005-2007  Mikael Pettersson
 */
#include <stdio.h>
#include "libperfctr.h"

unsigned int perfctr_info_nrctrs(const struct perfctr_info *info)
{
    switch (info->cpu_type) {
      case PERFCTR_ARM_XSC1:
	return 2;
      case PERFCTR_ARM_XSC2:
	return 4;
      default:
	return 0;
    }
}

const char *perfctr_info_cpu_name(const struct perfctr_info *info)
{
    switch (info->cpu_type) {
      case PERFCTR_ARM_XSC1:
	return "XScale1";
      case PERFCTR_ARM_XSC2:
	return "XScale2";
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
	printf("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
        printf("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
	if (i >= nractrs)
	    printf("ireset[%u]\t\t%d\n", i, control->ireset[i]);
    }
}
