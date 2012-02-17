/* $Id: ppc.c,v 1.2.2.3 2004/11/13 16:31:09 mikpe Exp $
 * PPC32-specific perfctr library procedures.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */
#include <stdio.h>
#include "libperfctr.h"
#include "ppc.h"

void perfctr_info_cpu_init(struct perfctr_info *info)
{
    unsigned int pvr = mfspr(SPRN_PVR); /* trapped & emulated by the kernel */
    unsigned int cpu_type;

    switch( PVR_VER(pvr) ) {
      case 0x0004: /* 604 */
	cpu_type = PERFCTR_PPC_604;
	break;
      case 0x0009: /* 604e */
      case 0x000A: /* 604ev */
	cpu_type = PERFCTR_PPC_604e;
	break;
      case 0x0008: /* 750/740 */
      case 0x7000: case 0x7001: /* 750FX */
      case 0x7002: /* 750GX */
	cpu_type = PERFCTR_PPC_750;
	break;
      case 0x000C: /* 7400 */
      case 0x800C: /* 7410 */
	cpu_type = PERFCTR_PPC_7400;
	break;
      case 0x8000: /* 7451/7441 */
      case 0x8001: /* 7455/7445 */
      case 0x8002: /* 7457/7447 */
      case 0x8003: /* 7447A */
      case 0x8004: /* 7448 */
	cpu_type = PERFCTR_PPC_7450;
	break;
      default:
	cpu_type = PERFCTR_PPC_GENERIC;
    }
    info->cpu_type = cpu_type;
}

unsigned int perfctr_info_nrctrs(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_PPC_604:
	return 2;
      case PERFCTR_PPC_604e:
      case PERFCTR_PPC_750:
      case PERFCTR_PPC_7400:
	return 4;
      case PERFCTR_PPC_7450:
	return 6;
      default:
	return 0;
    }
}

const char *perfctr_info_cpu_name(const struct perfctr_info *info)
{
    switch( info->cpu_type ) {
      case PERFCTR_PPC_GENERIC:
	return "Generic PowerPC with TB";
      case PERFCTR_PPC_604:
	return "PowerPC 604";
      case PERFCTR_PPC_604e:
	return "PowerPC 604e";
      case PERFCTR_PPC_750:
	return "PowerPC 750";
      case PERFCTR_PPC_7400:
	return "PowerPC 7400";
      case PERFCTR_PPC_7450:
	return "PowerPC 7450";
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
	printf("pmc_map[%u]\t\t%u\n", i, control->pmc_map[i]);
        printf("evntsel[%u]\t\t0x%08X\n", i, control->evntsel[i]);
	if( i >= nractrs )
	    printf("ireset[%u]\t\t%d\n", i, control->ireset[i]);
    }
    if( control->ppc.mmcr0 )
	printf("mmcr0\t\t\t0x%08X\n", control->ppc.mmcr0);
    if( control->ppc.mmcr2 )
	printf("mmcr2\t\t\t0x%08X\n", control->ppc.mmcr2);
}
