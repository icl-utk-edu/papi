/* $Id: ppc.c,v 1.1.2.2 2004/12/19 13:53:11 mikpe Exp $
 * PPC32-specific code.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */
#include <sys/ucontext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libperfctr.h"
#include "arch.h"

unsigned long ucontext_pc(const struct ucontext *uc)
{
    /* glibc-2.3.3 (YDL4) changed the type of uc->uc_mcontext,
     * breaking code which worked in glibc-2.3.1 (YDL3.0.1).
     * This formulation works with both, and is cleaner than
     * selecting glibc-2.3.3 specific code with "#ifdef NGREG".
     */
    return uc->uc_mcontext.regs->nip;
}

void do_setup(const struct perfctr_info *info,
	      struct perfctr_cpu_control *cpu_control)
{
    memset(cpu_control, 0, sizeof *cpu_control);

    cpu_control->tsc_on = 1;
    cpu_control->nractrs = 0;
    cpu_control->nrictrs = 1;
    cpu_control->pmc_map[0] = 0;

    /* INSTRUCTIONS_COMPLETED */
    cpu_control->evntsel[0] = 0x02;

    /* overflow after 100 events */
    cpu_control->ireset[0] = 0x80000000-100;

    /* not kernel mode, enable interrupts, enable PMC1 interrupts */
    cpu_control->ppc.mmcr0 = (1<<(31-1)) | (1<<(31-5)) | (1<<(31-16));
}
