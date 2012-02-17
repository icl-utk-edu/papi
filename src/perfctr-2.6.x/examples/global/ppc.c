/* $Id: ppc.c,v 1.1.2.1 2004/06/21 22:41:44 mikpe Exp $
 * PPC32-specific code.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libperfctr.h"
#include "arch.h"

void setup_control(const struct perfctr_info *info,
		   struct perfctr_cpu_control *control)
{
    memset(control, 0, sizeof *control);
    control->tsc_on = 1;
    if (info->cpu_type > PERFCTR_PPC_GENERIC) {
	control->nractrs = 1;
	control->pmc_map[0] = 0;
	control->evntsel[0] = 0x02; /* INSTRUCTIONS_COMPLETED */
	counting_mips = 1;
    }
}
