/* $Id: arm.c,v 1.1.2.1 2007/02/11 20:14:31 mikpe Exp $
 * ARM-specific code.
 *
 * Copyright (C) 2005-2007  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libperfctr.h"
#include "arch.h"

void do_setup(const struct perfctr_info *info,
	      struct perfctr_cpu_control *cpu_control)
{
    memset(cpu_control, 0, sizeof *cpu_control);
    switch (info->cpu_type) {
      case PERFCTR_ARM_XSC1:
      case PERFCTR_ARM_XSC2:
	cpu_control->tsc_on = 1;
	cpu_control->nractrs = 1;
	cpu_control->pmc_map[0] = 0;
	cpu_control->evntsel[0] = 0x07; /* INSTRUCTIONS_EXECUTED */
    }
}
