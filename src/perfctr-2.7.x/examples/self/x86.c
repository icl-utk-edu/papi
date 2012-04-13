/* $Id: x86.c,v 1.5 2005/03/14 01:48:42 mikpe Exp $
 * x86-specific code.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libperfctr.h"
#include "arch.h"

void do_setup(const struct perfctr_info *info,
	      struct perfctr_cpu_control *cpu_control)
{
    unsigned int tsc_on = 1;
    unsigned int nractrs = 1;
    unsigned int pmc_map0 = 0;
    unsigned int evntsel0 = 0;

    memset(cpu_control, 0, sizeof *cpu_control);

    /* Attempt to set up control to count clocks via the TSC
       and retired instructions via PMC0. */
    switch( info->cpu_type ) {
      case PERFCTR_X86_GENERIC:
	nractrs = 0;		/* no PMCs available */
	break;
#if !defined(__x86_64__)
      case PERFCTR_X86_INTEL_P5:
      case PERFCTR_X86_INTEL_P5MMX:
      case PERFCTR_X86_CYRIX_MII:
	/* event 0x16 (INSTRUCTIONS_EXECUTED), count at CPL 3 */
	evntsel0 = 0x16 | (2 << 6);
	break;
      case PERFCTR_X86_INTEL_P6:
      case PERFCTR_X86_INTEL_PII:
      case PERFCTR_X86_INTEL_PIII:
      case PERFCTR_X86_INTEL_PENTM:
      case PERFCTR_X86_AMD_K7:
#endif
      case PERFCTR_X86_AMD_K8:
      case PERFCTR_X86_AMD_K8C:
	/* event 0xC0 (INST_RETIRED), count at CPL > 0, Enable */
	evntsel0 = 0xC0 | (1 << 16) | (1 << 22);
	break;
#if !defined(__x86_64__)
      case PERFCTR_X86_WINCHIP_C6:
	tsc_on = 0;		/* no working TSC available */
	evntsel0 = 0x02;	/* X86_INSTRUCTIONS */
	break;
      case PERFCTR_X86_WINCHIP_2:
	tsc_on = 0;		/* no working TSC available */
	evntsel0 = 0x16;	/* INSTRUCTIONS_EXECUTED */
	break;
      case PERFCTR_X86_VIA_C3:
	pmc_map0 = 1;		/* redirect PMC0 to PERFCTR1 */
	evntsel0 = 0xC0;	/* INSTRUCTIONS_EXECUTED */
	break;
      case PERFCTR_X86_INTEL_P4:
      case PERFCTR_X86_INTEL_P4M2:
#endif
      case PERFCTR_X86_INTEL_P4M3:
	/* PMC0: IQ_COUNTER0 with fast RDPMC */
	pmc_map0 = 0x0C | (1 << 31);
	/* IQ_CCCR0: required flags, ESCR 4 (CRU_ESCR0), Enable */
	evntsel0 = (0x3 << 16) | (4 << 13) | (1 << 12);
	/* CRU_ESCR0: event 2 (instr_retired), NBOGUSNTAG, CPL>0 */
	cpu_control->p4.escr[0] = (2 << 25) | (1 << 9) | (1 << 2);
	break;
      default:
	fprintf(stderr, "cpu type %u (%s) not supported\n",
		info->cpu_type, perfctr_info_cpu_name(info));
	exit(1);
    }
    cpu_control->tsc_on = tsc_on;
    cpu_control->nractrs = nractrs;
    cpu_control->pmc_map[0] = pmc_map0;
    cpu_control->evntsel[0] = evntsel0;
}
